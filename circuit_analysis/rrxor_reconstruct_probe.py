"""
Diagnostics for activation-level reconstruction of the RRXOR circuit.

For each layer's residual deltas (attn-delta, mlp-delta), measure how much variance
is explained by a 16-cell condition (bit[t-2], bit[t-1], bit[t], phase mod 3 — note
phase is determined by the bit history at fixed position, but explicitly conditioning
keeps the analysis simple).

Also:
  - characterize the token and positional embedding structure
  - report layer-by-layer linear probe scores for (bit[t-2], bit[t-1], bit[t], phase, XOR)
"""

import argparse, json, os
import numpy as np
import torch
import torch.nn.functional as F

from rrxor_belief_experiment import (
    rrxor_tensors, stationary, sample_sequence, beliefs_from_emissions,
    MiniTransformer, STATE_PHASE,
)


@torch.no_grad()
def hooked_forward(model, x):
    """Same as rrxor_circuit_analysis.hooked_forward but return only the named tensors we need."""
    B, L = x.shape
    pos_idx = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
    h = model.tok(x) + model.pos(pos_idx)
    mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
    cache = {"embed": h.clone()}
    for k, blk in enumerate(model.blocks):
        x_norm = blk.norm1(h)
        attn_out, _ = blk.self_attn(x_norm, x_norm, x_norm,
                                    attn_mask=mask, need_weights=False)
        h_attn = h + attn_out
        cache[f"L{k}_pre"] = h.clone()
        cache[f"L{k}_attn_delta"] = attn_out.clone()
        cache[f"L{k}_post_attn"] = h_attn.clone()

        x_norm2 = blk.norm2(h_attn)
        mlp_out = blk.linear2(blk.dropout(blk.activation(blk.linear1(x_norm2))))
        cache[f"L{k}_mlp_delta"] = mlp_out.clone()
        h = h_attn + mlp_out
        cache[f"L{k}_post_mlp"] = h.clone()
    h_final = model.ln(h)
    cache["final"] = h_final.clone()
    cache["logits"] = model.head(h_final)
    return cache


def collect(model, T, pi, n_seqs, seq_len, rng):
    """Collect activations and labels across many sequences."""
    cache_accum = None
    ems_acc, phase_acc = [], []
    for _ in range(n_seqs):
        ems, _ = sample_sequence(T, pi, seq_len, rng)
        b = beliefs_from_emissions(T, pi, ems)
        phase = np.array([STATE_PHASE[np.argmax(bt)] for bt in b[1:]])
        x = torch.from_numpy(ems).long().unsqueeze(0)
        cache = hooked_forward(model, x)
        if cache_accum is None:
            cache_accum = {k: [] for k in cache}
        for k, v in cache.items():
            cache_accum[k].append(v.squeeze(0).numpy())
        ems_acc.append(ems)
        phase_acc.append(phase)
    out = {k: np.concatenate(v, axis=0) for k, v in cache_accum.items()}
    ems_flat = np.concatenate(ems_acc)
    ph_flat = np.concatenate(phase_acc)
    return out, ems_flat, ph_flat


def linear_probe(X, y, n_classes, train_frac=0.8, seed=0):
    N = X.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)
    ntr = int(train_frac * N)
    tr, te = idx[:ntr], idx[ntr:]
    Y_oh = np.eye(n_classes)[y]
    Xtr = np.concatenate([X[tr], np.ones((len(tr), 1))], axis=1)
    W, *_ = np.linalg.lstsq(Xtr, Y_oh[tr], rcond=None)
    Xte = np.concatenate([X[te], np.ones((len(te), 1))], axis=1)
    pred = (Xte @ W).argmax(axis=1)
    return float((pred == y[te]).mean())


def variance_explained_by_condition(deltas, condition_id, n_conditions):
    """deltas: (N, D). condition_id: (N,) ints in [0, n_conditions).
    Returns: between-class variance fraction (R^2 of class-mean predictor on each output dim,
    averaged across dims).
    """
    N, D = deltas.shape
    grand = deltas.mean(axis=0, keepdims=True)
    ss_tot = ((deltas - grand) ** 2).sum()
    ss_within = 0.0
    for c in range(n_conditions):
        m = condition_id == c
        if m.sum() < 2:
            continue
        cmean = deltas[m].mean(axis=0, keepdims=True)
        ss_within += ((deltas[m] - cmean) ** 2).sum()
    r2 = 1.0 - ss_within / max(ss_tot, 1e-12)
    return float(r2)


def build_condition_ids(ems, phases, lookback=2):
    """For each position t with t >= lookback, the condition is the tuple
    (bit[t-2], bit[t-1], bit[t], phase[t]).  16 phase-typed combinations.
    For positions t < lookback, condition_id = -1 (excluded)."""
    N = len(ems)
    cond = np.full(N, -1, dtype=np.int64)
    for t in range(lookback, N):
        bm2 = ems[t - 2]; bm1 = ems[t - 1]; b0 = ems[t]
        cond[t] = ((bm2 * 2 + bm1) * 2 + b0) * 3 + phases[t]
    return cond


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="results/rrxor_belief/naive_ctx")
    ap.add_argument("--outdir", default="results/rrxor_belief/naive_ctx/circuit")
    ap.add_argument("--n-seqs", type=int, default=120)
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

    # --- Embedding analysis ---
    tok_emb = model.tok.weight.detach().numpy()   # (2, 64)
    pos_emb = model.pos.weight.detach().numpy()   # (max_len, 64)
    print("=== TOKEN EMBEDDING ===")
    print(f"  shape={tok_emb.shape}")
    print(f"  ||tok[0]||={np.linalg.norm(tok_emb[0]):.3f}  ||tok[1]||={np.linalg.norm(tok_emb[1]):.3f}")
    diff = tok_emb[1] - tok_emb[0]
    print(f"  ||tok[1]-tok[0]||={np.linalg.norm(diff):.3f}")
    avg = (tok_emb[0] + tok_emb[1]) / 2
    print(f"  ||avg||={np.linalg.norm(avg):.3f}   (mean part — adds at every position)")

    print("\n=== POSITION EMBEDDING ===")
    print(f"  shape={pos_emb.shape}")
    # Phase mod 3 probe: does pos_emb[t] linearly encode (t mod 3)?
    used_len = ta["seq_len"]
    pe = pos_emb[:used_len]
    phase_t = np.arange(used_len) % 3
    # We don't actually know whether the model uses "absolute t mod 3" or the
    # process's actual phase (which depends on initial state).  But within a
    # sequence starting from stationary, the process phase = (initial_phase + t)
    # mod 3, so position-mod-3 and phase differ by a sequence-specific offset.
    acc_pe = linear_probe(pe, phase_t, n_classes=3)
    print(f"  linear probe pos_emb -> (t mod 3) acc = {acc_pe:.3f}")
    # PCA: how concentrated is the variance?
    pe_centered = pe - pe.mean(0, keepdims=True)
    U, S, _ = np.linalg.svd(pe_centered, full_matrices=False)
    cum = np.cumsum(S ** 2) / (S ** 2).sum()
    print(f"  pos_emb PCA cumulative var: top1={cum[0]:.3f}  top2={cum[1]:.3f}  "
          f"top3={cum[2]:.3f}  top5={cum[4]:.3f}")

    # --- Collect activations ---
    print(f"\ncollecting {args.n_seqs} sequences of length {args.seq_len}...")
    rng = np.random.default_rng(0)
    acts, ems, phases = collect(model, T, pi, args.n_seqs, args.seq_len, rng)
    N = len(ems)
    print(f"N total positions = {N}")
    bit_curr = ems
    bit_prev = np.full(N, -1, dtype=np.int64)
    bit_prev[1:] = ems[:-1]
    bit_prev2 = np.full(N, -1, dtype=np.int64)
    bit_prev2[2:] = ems[:-2]
    xor = np.full(N, -1, dtype=np.int64)
    xor[1:] = ems[1:] ^ ems[:-1]

    valid = bit_prev2 >= 0
    cond = build_condition_ids(ems, phases, lookback=2)  # 16-bit cond * 3 phases = 24 categories

    # --- Per-stage linear probe sweep for the four features ---
    print("\n=== PER-STAGE LINEAR PROBES (bit_curr, bit_prev, bit_prev2, phase, XOR) ===")
    stages = (
        ["embed"]
        + [f"L{k}_{w}" for k in range(4) for w in ("pre", "post_attn", "post_mlp")]
        + ["final"]
    )
    probe_table = []
    for s in stages:
        X = acts[s]
        accs = {
            "bit_curr": linear_probe(X[valid], bit_curr[valid], 2),
            "bit_prev": linear_probe(X[valid], bit_prev[valid], 2),
            "bit_prev2": linear_probe(X[valid], bit_prev2[valid], 2),
            "phase": linear_probe(X[valid], phases[valid], 3),
            "xor": linear_probe(X[valid], xor[valid], 2),
        }
        probe_table.append({"stage": s, **accs})
        print(f"  {s:>15}: bc={accs['bit_curr']:.3f}  bp={accs['bit_prev']:.3f}  "
              f"bp2={accs['bit_prev2']:.3f}  ph={accs['phase']:.3f}  xor={accs['xor']:.3f}")

    # --- Variance explained by condition for each layer's deltas ---
    print("\n=== VARIANCE EXPLAINED BY CONDITION (bit_prev2, bit_prev, bit_curr, phase) ===")
    print("    R^2 close to 1 → the delta can be replaced by a 24-entry lookup table")
    ve_table = []
    for k in range(4):
        for name in ("attn_delta", "mlp_delta"):
            key = f"L{k}_{name}"
            d = acts[key][valid]
            r2 = variance_explained_by_condition(d, cond[valid], n_conditions=24)
            ve_table.append({"component": key, "r2_by_condition": r2,
                             "norm_mean": float(np.linalg.norm(d, axis=1).mean())})
            print(f"  {key}: R^2 by (bit-2,bit-1,bit-0,phase) = {r2:.4f}  "
                  f"mean ||delta|| = {np.linalg.norm(d, axis=1).mean():.3f}")

    # --- Same R^2 but as a function of broader lookbacks too, for L3 specifically ---
    # phase determines the model's logit only via the gate; XOR sign is determined by
    # bit_prev and bit_curr.  Check whether L3 deltas are well-explained by smaller
    # conditioning sets:
    print("\n  L3 deltas under reduced conditioning:")
    # condition only on (phase, xor)
    cond_phx = phases * 2 + (xor.clip(min=0))
    cond_phx[xor < 0] = -1
    valid_xor = xor >= 0
    for k_idx, name in [(3, "attn_delta"), (3, "mlp_delta")]:
        key = f"L{k_idx}_{name}"
        d = acts[key][valid_xor]
        r2 = variance_explained_by_condition(d, cond_phx[valid_xor], n_conditions=6)
        print(f"    {key} under (phase, xor)-only conditioning [6 cells]: R^2 = {r2:.4f}")

    # --- Save raw activation summaries for use by the reconstruction script ---
    summary = {
        "tok_emb_shape": list(tok_emb.shape),
        "pos_emb_shape": list(pos_emb.shape),
        "pos_emb_phase_probe_acc": acc_pe,
        "pos_emb_pca_cumvar": cum.tolist()[:10],
        "tok_emb_diff_norm": float(np.linalg.norm(diff)),
        "tok_emb_mean_norm": float(np.linalg.norm(avg)),
        "probe_table": probe_table,
        "variance_explained_by_condition": ve_table,
    }
    with open(os.path.join(args.outdir, "reconstruct_probe.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {args.outdir}/reconstruct_probe.json")


if __name__ == "__main__":
    main()
