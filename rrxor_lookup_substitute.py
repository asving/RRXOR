"""
Substitute each layer's actual residual delta with a conditional-mean lookup table,
indexed by (bit[t-1], bit[t], phase, t mod 12). Measure how well the model still works.

The intent: if a substitution preserves behavior, we have a Python function for that
layer (literally a precomputed 144-vector table). If it breaks behavior, the layer is
doing more than the lookup captures.
"""

import argparse, json, os
import numpy as np
import torch
import torch.nn.functional as F

from rrxor_belief_experiment import (
    rrxor_tensors, stationary, sample_sequence, beliefs_from_emissions,
    MiniTransformer, STATE_PHASE,
)
from rrxor_reconstruct_probe import hooked_forward


N_PHASE = 3
N_POSMOD = 12
N_BITS_BACK = 2   # bit[t-1] and bit[t]
N_CONDS = (2 ** N_BITS_BACK) * N_PHASE * N_POSMOD   # 4 * 3 * 12 = 144


def make_conds(ems, phases, seq_len):
    """For a single sequence: return (L,) array of condition ids; -1 where invalid."""
    L = len(ems)
    cond = np.full(L, -1, dtype=np.int64)
    for t in range(1, L):
        b_prev = int(ems[t - 1])
        b_curr = int(ems[t])
        ph = int(phases[t])
        pm = t % N_POSMOD
        cond[t] = (((b_prev * 2 + b_curr) * N_PHASE + ph) * N_POSMOD) + pm
    return cond


@torch.no_grad()
def gather_deltas(model, T, pi, n_seqs, seq_len, rng):
    """Run the model on n_seqs sequences and collect deltas + condition ids."""
    deltas = {f"L{k}_{w}_delta": [] for k in range(4) for w in ("attn", "mlp")}
    conds = []
    for _ in range(n_seqs):
        ems, _ = sample_sequence(T, pi, seq_len, rng)
        b = beliefs_from_emissions(T, pi, ems)
        phase = np.array([STATE_PHASE[np.argmax(bt)] for bt in b[1:]])
        x = torch.from_numpy(ems).long().unsqueeze(0)
        cache = hooked_forward(model, x)
        for k, lst in deltas.items():
            lst.append(cache[k].squeeze(0).numpy())
        conds.append(make_conds(ems, phase, seq_len))
    deltas = {k: np.concatenate(v, axis=0) for k, v in deltas.items()}
    conds = np.concatenate(conds, axis=0)
    return deltas, conds


def fit_lookup(deltas, conds, n_conds, D):
    """Return mean delta per condition. Empty cells get the global mean."""
    table = np.zeros((n_conds, D), dtype=np.float32)
    counts = np.zeros(n_conds, dtype=np.int64)
    valid = conds >= 0
    global_mean = deltas[valid].mean(axis=0)
    for c in range(n_conds):
        m = (conds == c)
        cnt = int(m.sum())
        counts[c] = cnt
        if cnt > 0:
            table[c] = deltas[m].mean(axis=0).astype(np.float32)
        else:
            table[c] = global_mean.astype(np.float32)
    return table, counts


@torch.no_grad()
def substituted_forward(model, x, conds, substitute_set, lookups):
    """Run a forward pass where the listed components are replaced by their lookup.

    substitute_set: set of strings like {'L0_attn_delta', 'L2_mlp_delta', ...}
    lookups: dict from component name to (n_conds, D) numpy array
    conds: (L,) numpy array of condition ids per position (-1 → use original delta)
    """
    B, L = x.shape
    assert B == 1
    pos_idx = torch.arange(L, device=x.device).unsqueeze(0)
    h = model.tok(x) + model.pos(pos_idx)
    mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
    valid = conds >= 0

    for k, blk in enumerate(model.blocks):
        x_norm = blk.norm1(h)
        attn_native, _ = blk.self_attn(x_norm, x_norm, x_norm,
                                       attn_mask=mask, need_weights=False)
        key_a = f"L{k}_attn_delta"
        if key_a in substitute_set:
            table = lookups[key_a]
            lookup_vals = np.zeros((L, table.shape[1]), dtype=np.float32)
            lookup_vals[valid] = table[conds[valid]]
            attn_out = torch.from_numpy(lookup_vals).unsqueeze(0)
            # at invalid positions, fall back to native delta
            invalid_mask = ~valid
            if invalid_mask.any():
                attn_out[0, invalid_mask] = attn_native[0, invalid_mask]
        else:
            attn_out = attn_native
        h_attn = h + attn_out

        x_norm2 = blk.norm2(h_attn)
        mlp_native = blk.linear2(blk.dropout(blk.activation(blk.linear1(x_norm2))))
        key_m = f"L{k}_mlp_delta"
        if key_m in substitute_set:
            table = lookups[key_m]
            lookup_vals = np.zeros((L, table.shape[1]), dtype=np.float32)
            lookup_vals[valid] = table[conds[valid]]
            mlp_out = torch.from_numpy(lookup_vals).unsqueeze(0)
            if (~valid).any():
                mlp_out[0, ~valid] = mlp_native[0, ~valid]
        else:
            mlp_out = mlp_native
        h = h_attn + mlp_out

    h_final = model.ln(h)
    logits = model.head(h_final)
    return logits


def eval_substitution(model, T, pi, n_seqs, seq_len, rng, substitute_set, lookups):
    """Return phase-2 accuracy and mean |Δ logit_diff| versus clean."""
    p2_correct = 0; p2_total = 0
    ld_diffs = []
    ph_metric = {ph: [] for ph in (0, 1, 2)}
    for _ in range(n_seqs):
        ems, _ = sample_sequence(T, pi, seq_len, rng)
        b = beliefs_from_emissions(T, pi, ems)
        phase = np.array([STATE_PHASE[np.argmax(bt)] for bt in b[1:]])
        conds = make_conds(ems, phase, seq_len)
        x = torch.from_numpy(ems).long().unsqueeze(0)
        # Native
        with torch.no_grad():
            l_clean = model(x).squeeze(0).numpy()
        # Substituted
        l_sub = substituted_forward(model, x, conds, substitute_set, lookups).squeeze(0).numpy()
        ld_clean = l_clean[:, 1] - l_clean[:, 0]
        ld_sub = l_sub[:, 1] - l_sub[:, 0]
        xor = np.full(seq_len, -1, dtype=np.int64)
        xor[1:] = ems[1:] ^ ems[:-1]
        for t in range(1, seq_len):
            ph_metric[phase[t]].append((ld_sub[t], xor[t], ld_clean[t]))
            if phase[t] == 2:
                p2_total += 1
                if (ld_sub[t] > 0) == (xor[t] == 1):
                    p2_correct += 1
        ld_diffs.append((ld_sub[1:] - ld_clean[1:]))
    ld_diffs = np.concatenate(ld_diffs)
    out = {
        "phase2_accuracy_sub": float(p2_correct / max(p2_total, 1)),
        "phase2_n": int(p2_total),
        "mean_abs_logit_diff_change": float(np.mean(np.abs(ld_diffs))),
        "mean_logit_diff_change": float(np.mean(ld_diffs)),
    }
    # Per-phase logit_diff magnitude
    for ph in (0, 1, 2):
        a = np.array([t[0] for t in ph_metric[ph]])
        c = np.array([t[2] for t in ph_metric[ph]])
        out[f"ph{ph}_sub_logit_diff_mean"] = float(a.mean())
        out[f"ph{ph}_sub_logit_diff_std"] = float(a.std())
        out[f"ph{ph}_clean_logit_diff_mean"] = float(c.mean())
        out[f"ph{ph}_clean_logit_diff_std"] = float(c.std())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="results/rrxor_belief/naive_ctx")
    ap.add_argument("--outdir", default="results/rrxor_belief/naive_ctx/circuit")
    ap.add_argument("--n-fit-seqs", type=int, default=240)
    ap.add_argument("--n-eval-seqs", type=int, default=80)
    ap.add_argument("--seq-len", type=int, default=128)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    prior = np.load(os.path.join(args.source, "results.npz"), allow_pickle=True)
    ta = prior["args"].item()
    model = MiniTransformer(
        vocab=2, d_model=ta["d_model"], n_heads=ta["n_heads"],
        n_layers=ta["n_layers"], max_len=ta["seq_len"] + 2,
    )
    model.load_state_dict(torch.load(os.path.join(args.source, "model.pt"), map_location="cpu"))
    model.eval()
    T = rrxor_tensors(); pi = stationary(T)

    D = ta["d_model"]

    # Fit lookups
    print(f"fitting lookups with {args.n_fit_seqs} sequences of length {args.seq_len}...")
    fit_rng = np.random.default_rng(0)
    deltas, conds = gather_deltas(model, T, pi, args.n_fit_seqs, args.seq_len, fit_rng)
    lookups = {}
    print("Lookup sample counts (min/median/max across 144 cells):")
    for key, d in deltas.items():
        table, counts = fit_lookup(d, conds, N_CONDS, D)
        lookups[key] = table
        c_nz = counts[counts > 0]
        print(f"  {key}: cells_with_data={len(c_nz)}/{N_CONDS}  min={c_nz.min()}  "
              f"median={int(np.median(c_nz))}  max={c_nz.max()}")

    eval_rng = np.random.default_rng(1)
    components = [f"L{k}_{w}_delta" for k in range(4) for w in ("attn", "mlp")]

    print("\n=== Single-component substitution ===")
    print(f"(eval on {args.n_eval_seqs} sequences)")
    single = {}
    for comp in components:
        res = eval_substitution(model, T, pi, args.n_eval_seqs, args.seq_len,
                                 np.random.default_rng(1), {comp}, lookups)
        single[comp] = res
        print(f"  sub {comp}: acc_phase2={res['phase2_accuracy_sub']:.4f}  "
              f"|Δlogit_diff|={res['mean_abs_logit_diff_change']:.3f}  "
              f"ph2 mag sub/clean = {res['ph2_sub_logit_diff_mean']:.2f}/{res['ph2_clean_logit_diff_mean']:.2f}")

    # Cumulative top-down: start by substituting L3_mlp, then add L3_attn, then L2_mlp, etc.
    print("\n=== Top-down cumulative substitution ===")
    cum_top_down = {}
    order = ["L3_mlp_delta", "L3_attn_delta",
             "L2_mlp_delta", "L2_attn_delta",
             "L1_mlp_delta", "L1_attn_delta",
             "L0_mlp_delta", "L0_attn_delta"]
    acc = set()
    for comp in order:
        acc.add(comp)
        res = eval_substitution(model, T, pi, args.n_eval_seqs, args.seq_len,
                                 np.random.default_rng(1), set(acc), lookups)
        cum_top_down["+".join(sorted(acc))] = res
        print(f"  sub {{{', '.join(sorted(acc))}}}: "
              f"acc_phase2={res['phase2_accuracy_sub']:.4f}  "
              f"|Δlogit|={res['mean_abs_logit_diff_change']:.3f}  "
              f"ph2 mag = {res['ph2_sub_logit_diff_mean']:.2f}")

    # Bottom-up: start by substituting L0, then add L1, etc.  This reveals whether
    # early-layer substitution irrecoverably corrupts the residual stream.
    print("\n=== Bottom-up cumulative substitution ===")
    cum_bottom_up = {}
    order_bu = ["L0_attn_delta", "L0_mlp_delta",
                "L1_attn_delta", "L1_mlp_delta",
                "L2_attn_delta", "L2_mlp_delta",
                "L3_attn_delta", "L3_mlp_delta"]
    acc = set()
    for comp in order_bu:
        acc.add(comp)
        res = eval_substitution(model, T, pi, args.n_eval_seqs, args.seq_len,
                                 np.random.default_rng(1), set(acc), lookups)
        cum_bottom_up["+".join(sorted(acc))] = res
        print(f"  sub {{{', '.join(sorted(acc))}}}: "
              f"acc_phase2={res['phase2_accuracy_sub']:.4f}  "
              f"|Δlogit|={res['mean_abs_logit_diff_change']:.3f}  "
              f"ph2 mag = {res['ph2_sub_logit_diff_mean']:.2f}")

    # All at once
    print("\n=== All components substituted (full lookup-table model) ===")
    res_all = eval_substitution(model, T, pi, args.n_eval_seqs, args.seq_len,
                                 np.random.default_rng(1), set(components), lookups)
    print(f"  acc_phase2={res_all['phase2_accuracy_sub']:.4f}  "
          f"|Δlogit|={res_all['mean_abs_logit_diff_change']:.3f}  "
          f"ph2 mag sub/clean = {res_all['ph2_sub_logit_diff_mean']:.2f}/{res_all['ph2_clean_logit_diff_mean']:.2f}")
    for ph in (0, 1, 2):
        print(f"   phase {ph}: sub_mean={res_all[f'ph{ph}_sub_logit_diff_mean']:.3f}  "
              f"sub_std={res_all[f'ph{ph}_sub_logit_diff_std']:.3f}  "
              f"clean_mean={res_all[f'ph{ph}_clean_logit_diff_mean']:.3f}")

    out = {
        "single_substitution": single,
        "cumulative_top_down": cum_top_down,
        "cumulative_bottom_up": cum_bottom_up,
        "all_substituted": res_all,
        "n_conds": N_CONDS,
        "condition_format": "(bit[t-1], bit[t], phase, t mod 12)",
    }
    with open(os.path.join(args.outdir, "lookup_substitute.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.outdir}/lookup_substitute.json")


if __name__ == "__main__":
    main()
