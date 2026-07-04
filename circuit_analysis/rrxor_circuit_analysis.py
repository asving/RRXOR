"""
Mechanistic analysis of the trained RRXOR (naive_ctx) transformer.

Goal: establish whether the model
  (a) tracks the last two emitted bits, and
  (b) computes XOR at phase-2 (deterministic) positions,
and identify the layer at which XOR materializes.

All tests run on CPU; the model is 4-layer, d=64, 2 heads, vocab=2.
"""

import argparse
import json
import os
import numpy as np
import torch
import torch.nn.functional as F

from rrxor_belief_experiment import (
    rrxor_tensors, stationary, sample_sequence, beliefs_from_emissions,
    MiniTransformer, STATE_PHASE,
)


# ---------------------------------------------------------------------------
# Hooked forward: reimplements MiniTransformer.forward but caches every
# pre-/post-attn/post-mlp residual and the attention weights per head.
# ---------------------------------------------------------------------------

@torch.no_grad()
def hooked_forward(model, x):
    """x: (B, L) ints.  Returns dict of named activations."""
    B, L = x.shape
    pos_idx = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
    h = model.tok(x) + model.pos(pos_idx)
    mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)

    cache = {"embed": h.clone()}
    for k, blk in enumerate(model.blocks):
        x_norm = blk.norm1(h)
        attn_out, attn_w = blk.self_attn(
            x_norm, x_norm, x_norm,
            attn_mask=mask, need_weights=True, average_attn_weights=False,
        )
        h_attn = h + attn_out
        cache[f"L{k}_pre"] = h.clone()
        cache[f"L{k}_post_attn"] = h_attn.clone()
        cache[f"L{k}_attn_w"] = attn_w.clone()  # (B, n_heads, L, L)

        x_norm2 = blk.norm2(h_attn)
        mlp_out = blk.linear2(blk.dropout(blk.activation(blk.linear1(x_norm2))))
        h = h_attn + mlp_out
        cache[f"L{k}_post_mlp"] = h.clone()

    h_final = model.ln(h)
    logits = model.head(h_final)
    cache["final"] = h_final.clone()
    cache["logits"] = logits.clone()
    return cache


def verify_hooked_matches_native(model, x):
    """Sanity-check that hooked_forward reproduces the model's native forward."""
    with torch.no_grad():
        native_logits = model(x)
    cache = hooked_forward(model, x)
    diff = (cache["logits"] - native_logits).abs().max().item()
    return diff


# ---------------------------------------------------------------------------
# Linear probes
# ---------------------------------------------------------------------------

def linear_probe_regression(X, Y, train_frac=0.8, seed=0):
    """Affine probe X -> Y minimizing MSE. Returns R^2 on held-out and the weights."""
    N = X.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)
    ntr = int(train_frac * N)
    tr, te = idx[:ntr], idx[ntr:]
    Xtr = np.concatenate([X[tr], np.ones((len(tr), 1))], axis=1)
    W, *_ = np.linalg.lstsq(Xtr, Y[tr], rcond=None)
    Xte = np.concatenate([X[te], np.ones((len(te), 1))], axis=1)
    Yhat = Xte @ W
    ss_res = ((Y[te] - Yhat) ** 2).sum(axis=0)
    ss_tot = ((Y[te] - Y[te].mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1.0 - ss_res / np.clip(ss_tot, 1e-12, None)
    return float(np.mean(r2)), W


def linear_probe_classify(X, y, n_classes, train_frac=0.8, seed=0):
    """One-vs-rest LSQ classifier. Returns held-out accuracy."""
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
    acc = float((pred == y[te]).mean())
    return acc, W


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------

def collect(model, T, pi, n_seqs, seq_len, rng):
    """Run hooked_forward on many sequences. Returns flat arrays plus per-sequence attn."""
    stages = (
        ["embed"]
        + [f"L{k}_{w}" for k in range(4) for w in ("pre", "post_attn", "post_mlp")]
        + ["final"]
    )
    acts = {s: [] for s in stages}
    logits_all = []
    phases_all = []
    ems_prev_all = []   # ems[t-1]
    ems_curr_all = []   # ems[t]
    xor_all = []        # ems[t-1] XOR ems[t]
    target_all = []     # ems[t+1] (what the model predicts)
    attn_by_layer = [[] for _ in range(4)]   # each entry: list of (n_heads, L, L)

    for _ in range(n_seqs):
        ems, _ = sample_sequence(T, pi, seq_len, rng)
        b = beliefs_from_emissions(T, pi, ems)
        beliefs_at_pos = b[1:]
        phase_at_pos = np.array([STATE_PHASE[np.argmax(bt)] for bt in beliefs_at_pos])

        x = torch.from_numpy(ems).long().unsqueeze(0)
        cache = hooked_forward(model, x)

        for s in stages:
            acts[s].append(cache[s].squeeze(0).numpy())
        logits_all.append(cache["logits"].squeeze(0).numpy())
        for k in range(4):
            attn_by_layer[k].append(cache[f"L{k}_attn_w"].squeeze(0).numpy())

        # labels at every position t (1 <= t <= seq_len-2 to have both prev and target valid)
        L = len(ems)
        ems_prev = np.full(L, -1, dtype=np.int64)
        ems_prev[1:] = ems[:-1]
        target = np.full(L, -1, dtype=np.int64)
        target[:-1] = ems[1:]
        xor = np.full(L, -1, dtype=np.int64)
        xor[1:] = ems[1:] ^ ems[:-1]

        phases_all.append(phase_at_pos)
        ems_prev_all.append(ems_prev)
        ems_curr_all.append(ems.copy())
        xor_all.append(xor)
        target_all.append(target)

    out = {s: np.concatenate(acts[s], axis=0) for s in stages}
    return {
        "acts": out,
        "logits": np.concatenate(logits_all, axis=0),
        "phases": np.concatenate(phases_all, axis=0),
        "ems_prev": np.concatenate(ems_prev_all, axis=0),
        "ems_curr": np.concatenate(ems_curr_all, axis=0),
        "xor": np.concatenate(xor_all, axis=0),
        "target": np.concatenate(target_all, axis=0),
        "attn_by_layer": attn_by_layer,
        "stages": stages,
    }


def behavioral_xor_check(D):
    """Test A/B: phase-conditional logit decomposition.
    At phase=2 (XOR position), logit[1]-logit[0] should have sign (-1)^xor and large |.|.
    At phase=0,1, the logit difference should be near zero.
    """
    logit_diff = D["logits"][:, 1] - D["logits"][:, 0]  # (N,)
    phases = D["phases"]
    xor = D["xor"]
    mask_valid = xor >= 0  # have prev+curr bits

    out = {}
    for ph in [0, 1, 2]:
        m = mask_valid & (phases == ph)
        if m.sum() == 0:
            continue
        # Split by ground-truth XOR value
        for xv in [0, 1]:
            mm = m & (xor == xv)
            if mm.sum() == 0:
                continue
            ld = logit_diff[mm]
            out[f"phase{ph}_xor{xv}"] = {
                "n": int(mm.sum()),
                "logit_diff_mean": float(ld.mean()),
                "logit_diff_std": float(ld.std()),
                "p_token1_mean": float(torch.sigmoid(torch.tensor(ld)).mean()),
            }
    # Behavioral accuracy at phase=2: does argmax match XOR?
    m2 = mask_valid & (phases == 2)
    pred_at_phase2 = (logit_diff[m2] > 0).astype(int)
    truth_at_phase2 = xor[m2]
    out["phase2_pred_accuracy"] = float((pred_at_phase2 == truth_at_phase2).mean())
    out["phase2_n"] = int(m2.sum())
    return out


def layer_probes(D):
    """Test C/D: probe each stage for last-2-bits (4-class), XOR (binary), and phase."""
    stages = D["stages"]
    phases = D["phases"]
    ems_prev = D["ems_prev"]
    ems_curr = D["ems_curr"]
    xor = D["xor"]
    valid = ems_prev >= 0
    last2 = (ems_prev * 2 + ems_curr).astype(np.int64)

    rows = []
    for s in stages:
        X = D["acts"][s]
        # last-2-bits (joint) classification
        acc_last2, _ = linear_probe_classify(X[valid], last2[valid], n_classes=4)
        # current bit alone
        acc_curr, _ = linear_probe_classify(X[valid], ems_curr[valid], n_classes=2)
        # previous bit alone
        acc_prev, _ = linear_probe_classify(X[valid], ems_prev[valid], n_classes=2)
        # XOR (binary)
        acc_xor, _ = linear_probe_classify(X[valid], xor[valid], n_classes=2)
        # XOR restricted to phase=2 (where the model is actually predicting XOR)
        m2 = valid & (phases == 2)
        acc_xor_p2, _ = linear_probe_classify(X[m2], xor[m2], n_classes=2)
        # Phase classification
        acc_phase, _ = linear_probe_classify(X, phases, n_classes=3)
        rows.append({
            "stage": s,
            "last2_acc": acc_last2,
            "curr_bit_acc": acc_curr,
            "prev_bit_acc": acc_prev,
            "xor_acc_all": acc_xor,
            "xor_acc_phase2": acc_xor_p2,
            "phase_acc": acc_phase,
        })
    return rows


def attention_by_phase(D, n_show=8):
    """Test F: average attention pattern by query phase.
    For each layer/head, average over queries at phase 2 (XOR) and phase 0,1 (random),
    looking at how much weight goes to the most recent few keys (relative offsets).
    """
    out = {}
    for k in range(4):
        # attn_by_layer[k] is a list of (n_heads, L, L)
        all_attn = np.stack(D["attn_by_layer"][k], axis=0)  # (n_seqs, n_heads, L, L)
        n_seqs, n_heads, L, _ = all_attn.shape
        # For each query position (row), compute weight at relative offsets 0..min_back
        max_back = 6
        per_head_by_phase = {}
        # we have per-position phases concatenated across sequences;
        # reshape back to (n_seqs, L)
        phases_2d = D["phases"].reshape(n_seqs, L)
        for ph in [0, 1, 2]:
            rows = []
            for h_idx in range(n_heads):
                # gather attention rows where query phase == ph
                # for each (seq, query), look at attn[seq, h, query, max(0,query-max_back):query+1]
                # We'll compute average weight at relative offset r = query - key (r=0 is self).
                bucket = np.zeros(max_back + 1)
                count = 0
                for s_idx in range(n_seqs):
                    for q in range(L):
                        if phases_2d[s_idx, q] != ph:
                            continue
                        for r in range(min(max_back, q) + 1):
                            bucket[r] += all_attn[s_idx, h_idx, q, q - r]
                        count += 1
                if count > 0:
                    bucket /= count
                rows.append({"head": h_idx, "avg_by_offset": bucket.tolist(), "n_queries": count})
            per_head_by_phase[f"phase{ph}"] = rows
        out[f"layer{k}"] = per_head_by_phase
    return out


def causal_patch_test(model, T, pi, rng, n_pairs=200, max_seq_len=128):
    """Test E: input-level causal intervention.
    Generate pairs of sequences (A, B) that differ in exactly one bit at some position p,
    chosen so that this bit is one of the two ingredients of a phase-2 XOR at position p+1
    or p+2. Run both, compare logits at the affected phase-2 query.
    """
    out = []
    for _ in range(n_pairs):
        ems_a, _ = sample_sequence(T, pi, max_seq_len, rng)
        ba = beliefs_from_emissions(T, pi, ems_a)
        phases = np.array([STATE_PHASE[np.argmax(bt)] for bt in ba[1:]])
        # find a phase-2 query position q (where target is XOR), with valid prev bit
        p2_positions = np.where(phases == 2)[0]
        p2_positions = p2_positions[p2_positions >= 2]
        if len(p2_positions) == 0:
            continue
        q = int(rng.choice(p2_positions))
        # at query q, prediction target is XOR(ems[q-1], ems[q]).
        # If we flip ems[q-1] (a previous-R1 emission), the XOR flips.
        # We do an "input intervention": ems_b = ems_a, ems_b[q-1] ^= 1.
        # NB: changing ems_b[q-1] makes ems_b inconsistent with the HMM downstream of q-1,
        # but we are only checking the model's behavior on the modified context.
        ems_b = ems_a.copy()
        ems_b[q - 1] ^= 1

        with torch.no_grad():
            xa = torch.from_numpy(ems_a).long().unsqueeze(0)
            xb = torch.from_numpy(ems_b).long().unsqueeze(0)
            la = model(xa).squeeze(0).numpy()
            lb = model(xb).squeeze(0).numpy()
        # Logits at position q (predicting token at q+1)
        diff_a = la[q, 1] - la[q, 0]
        diff_b = lb[q, 1] - lb[q, 0]
        true_xor_a = int(ems_a[q - 1] ^ ems_a[q])
        true_xor_b = int(ems_b[q - 1] ^ ems_b[q])
        out.append({
            "q": q,
            "ems_a_q_minus1": int(ems_a[q - 1]),
            "ems_a_q": int(ems_a[q]),
            "logit_diff_a": float(diff_a),
            "logit_diff_b": float(diff_b),
            "true_xor_a": true_xor_a,
            "true_xor_b": true_xor_b,
        })
    # Summarize: at how many positions does the sign of logit_diff flip when ems[q-1] is flipped?
    n = len(out)
    flipped = sum(1 for r in out if np.sign(r["logit_diff_a"]) != np.sign(r["logit_diff_b"]))
    correct_a = sum(1 for r in out if (r["logit_diff_a"] > 0) == (r["true_xor_a"] == 1))
    correct_b = sum(1 for r in out if (r["logit_diff_b"] > 0) == (r["true_xor_b"] == 1))
    return {
        "n_pairs": n,
        "frac_logit_sign_flipped": flipped / max(n, 1),
        "frac_correct_clean": correct_a / max(n, 1),
        "frac_correct_intervened": correct_b / max(n, 1),
        "details_first_10": out[:10],
    }


def activation_patch_test(model, T, pi, rng, n_pairs=100, max_seq_len=128):
    """Pure activation patching: pick query q at phase 2, pick a source position p in {q-1, q-2}.
    For each layer k, swap the position-p residual at layer k between two sequences that have the
    same context EXCEPT ems[p]. Continue the forward from layer k onward, measure logit-diff change
    at position q.
    Returns: per-layer, per-source-position effect.
    """
    results = {f"src=q-{d}": {f"L{k}": [] for k in range(4)} for d in (1, 2)}
    for _ in range(n_pairs):
        ems_a, _ = sample_sequence(T, pi, max_seq_len, rng)
        ba = beliefs_from_emissions(T, pi, ems_a)
        phases = np.array([STATE_PHASE[np.argmax(bt)] for bt in ba[1:]])
        cand = np.where(phases == 2)[0]
        cand = cand[cand >= 2]
        if len(cand) == 0:
            continue
        q = int(rng.choice(cand))
        for d in (1, 2):
            p = q - d
            ems_b = ems_a.copy()
            ems_b[p] ^= 1
            xa = torch.from_numpy(ems_a).long().unsqueeze(0)
            xb = torch.from_numpy(ems_b).long().unsqueeze(0)
            with torch.no_grad():
                la = model(xa).squeeze(0).numpy()
                lb = model(xb).squeeze(0).numpy()
                cache_a = hooked_forward(model, xa)
            diff_a = la[q, 1] - la[q, 0]
            diff_b_clean = lb[q, 1] - lb[q, 0]
            # For each layer k, patch position p's post_attn residual from B into A,
            # then continue forward from there. This isolates the "what changes when
            # the bit information at position p is removed/swapped at layer k".
            for k in range(4):
                with torch.no_grad():
                    # Get residual entering layer k for both
                    cache_b = hooked_forward(model, xb)
                    h_a_pre_k = cache_a[f"L{k}_pre"].clone()
                    # Swap only position p of the residual at layer k's input
                    h_patched = h_a_pre_k.clone()
                    h_patched[0, p, :] = cache_b[f"L{k}_pre"][0, p, :]
                    # Continue forward from layer k
                    h = h_patched
                    L_len = h.shape[1]
                    mask = torch.triu(torch.ones(L_len, L_len, dtype=torch.bool), diagonal=1)
                    for kk in range(k, 4):
                        blk = model.blocks[kk]
                        x_norm = blk.norm1(h)
                        attn_out, _ = blk.self_attn(x_norm, x_norm, x_norm, attn_mask=mask,
                                                    need_weights=False)
                        h_attn = h + attn_out
                        x_norm2 = blk.norm2(h_attn)
                        mlp_out = blk.linear2(blk.dropout(blk.activation(blk.linear1(x_norm2))))
                        h = h_attn + mlp_out
                    h_final = model.ln(h)
                    logits = model.head(h_final)
                    lp = logits.squeeze(0).numpy()
                diff_patched = lp[q, 1] - lp[q, 0]
                # Effect size: how much did patching shift toward the B answer?
                # Normalize by the clean A->B shift (diff_b_clean - diff_a) if it's nonzero.
                denom = diff_b_clean - diff_a
                if abs(denom) < 1e-6:
                    eff = 0.0
                else:
                    eff = (diff_patched - diff_a) / denom
                results[f"src=q-{d}"][f"L{k}"].append(eff)
    summary = {}
    for k_src, by_layer in results.items():
        summary[k_src] = {k_layer: {
            "mean_effect": float(np.mean(v)) if v else None,
            "median_effect": float(np.median(v)) if v else None,
            "n": len(v),
        } for k_layer, v in by_layer.items()}
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="results/rrxor_belief/naive_ctx")
    ap.add_argument("--outdir", default="results/rrxor_belief/naive_ctx/circuit")
    ap.add_argument("--n-seqs", type=int, default=80)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--patch-pairs", type=int, default=80)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Load model
    prior = np.load(os.path.join(args.source, "results.npz"), allow_pickle=True)
    train_args = prior["args"].item()
    seq_len_train = train_args["seq_len"]
    d_model = train_args["d_model"]
    n_layers = train_args["n_layers"]
    n_heads = train_args["n_heads"]

    model = MiniTransformer(
        vocab=2, d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        max_len=seq_len_train + 2,
    )
    model.load_state_dict(torch.load(os.path.join(args.source, "model.pt"), map_location="cpu"))
    model.eval()

    # Sanity check
    rng = np.random.default_rng(args.seed)
    T = rrxor_tensors()
    pi = stationary(T)
    sample_x = torch.from_numpy(sample_sequence(T, pi, 32, rng)[0]).long().unsqueeze(0)
    diff = verify_hooked_matches_native(model, sample_x)
    print(f"hooked-vs-native max abs diff: {diff:.3e}", flush=True)
    assert diff < 1e-4, "hooked_forward doesn't match native forward!"

    # Collect activations
    print(f"collecting {args.n_seqs} sequences of length {args.seq_len}...", flush=True)
    D = collect(model, T, pi, n_seqs=args.n_seqs, seq_len=args.seq_len, rng=rng)

    # Test A/B: behavioral XOR
    print("--- behavioral phase-conditional logit check ---", flush=True)
    beh = behavioral_xor_check(D)
    for k, v in beh.items():
        print(f"  {k}: {v}", flush=True)

    # Test C/D: layer probes
    print("--- layer-by-layer probes ---", flush=True)
    rows = layer_probes(D)
    for r in rows:
        print(f"  {r['stage']:>15}: last2={r['last2_acc']:.3f}  "
              f"curr={r['curr_bit_acc']:.3f}  prev={r['prev_bit_acc']:.3f}  "
              f"xor_all={r['xor_acc_all']:.3f}  xor_p2={r['xor_acc_phase2']:.3f}  "
              f"phase={r['phase_acc']:.3f}", flush=True)

    # Test F: attention pattern
    print("--- attention pattern by query phase (avg weight at relative offsets) ---", flush=True)
    attn = attention_by_phase(D)
    for layer, by_phase in attn.items():
        print(f"  {layer}:", flush=True)
        for ph, heads in by_phase.items():
            for h in heads:
                bk = [f"{w:.2f}" for w in h["avg_by_offset"]]
                print(f"    {ph} head{h['head']}: rel_offset[0..6] = [{', '.join(bk)}]", flush=True)

    # Test E (simple): input-level intervention
    print("--- input-level intervention (flip ems[q-1]) ---", flush=True)
    rng2 = np.random.default_rng(args.seed + 1)
    patch = causal_patch_test(model, T, pi, rng2, n_pairs=args.patch_pairs,
                              max_seq_len=args.seq_len)
    print(f"  n_pairs = {patch['n_pairs']}", flush=True)
    print(f"  fraction logit-sign flipped after bit flip = {patch['frac_logit_sign_flipped']:.3f}",
          flush=True)
    print(f"  clean accuracy = {patch['frac_correct_clean']:.3f}", flush=True)
    print(f"  intervened accuracy = {patch['frac_correct_intervened']:.3f}", flush=True)

    # Test E (residual patching): activation-level intervention at each layer
    print("--- activation-level patching at each layer ---", flush=True)
    rng3 = np.random.default_rng(args.seed + 2)
    act_patch = activation_patch_test(
        model, T, pi, rng3, n_pairs=min(args.patch_pairs, 60),
        max_seq_len=args.seq_len,
    )
    for src, by_layer in act_patch.items():
        print(f"  {src}:", flush=True)
        for k, v in by_layer.items():
            print(f"    patch entering {k}: mean_effect={v['mean_effect']:.3f}  "
                  f"median={v['median_effect']:.3f}  n={v['n']}", flush=True)

    # Save
    summary = {
        "behavioral": beh,
        "layer_probes": rows,
        "attention_by_phase": attn,
        "input_intervention": patch,
        "activation_patching": act_patch,
        "model_args": {
            "seq_len_train": seq_len_train, "d_model": d_model,
            "n_layers": n_layers, "n_heads": n_heads,
        },
    }
    with open(os.path.join(args.outdir, "circuit_analysis.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {args.outdir}/circuit_analysis.json", flush=True)


if __name__ == "__main__":
    main()
