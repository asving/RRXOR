"""
How does L3 attention compute the phase posterior?

Phase has a random per-sequence offset, so any phase-dependent attention pattern
must be content-driven.  The informative content for phase inference is XOR
*violations*: a position s with b_s != b_{s-1} XOR b_{s-2} cannot be a phase-2
emission, which eliminates one of the three phase hypotheses for the whole
sequence.  Consistent positions are weak (p=1/2) evidence.

Tests, for each head at L1-L3 (query positions t >= 16, synchronized):
  (a) mean attention weight by relative offset, split by the QUERY's true phase.
      Differences across query phase at fixed offset => content-driven selection.
  (b) mean attention weight by the SOURCE's violation bit at fixed offset.
  (c) eta^2: fraction of weight variance at fixed (head, offset) explained by
      (query phase, source violation).
"""

import argparse, json, os
import numpy as np
import torch

from rrxor_belief_experiment import (
    rrxor_tensors, stationary, sample_sequence, MiniTransformer, STATE_PHASE,
)
from rrxor_circuit_analysis import hooked_forward as hooked_forward_attn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="results/rrxor_belief/naive_ctx")
    ap.add_argument("--outdir", default="results/rrxor_belief/naive_ctx/circuit")
    ap.add_argument("--n-seqs", type=int, default=100)
    ap.add_argument("--seq-len", type=int, default=96)
    ap.add_argument("--t-min", type=int, default=16)
    ap.add_argument("--max-offset", type=int, default=12)
    args = ap.parse_args()

    prior = np.load(os.path.join(args.source, "results.npz"), allow_pickle=True)
    ta = prior["args"].item()
    model = MiniTransformer(
        vocab=2, d_model=ta["d_model"], n_heads=ta["n_heads"],
        n_layers=ta["n_layers"], max_len=ta["seq_len"] + 2,
    )
    model.load_state_dict(torch.load(os.path.join(args.source, "model.pt"), map_location="cpu"))
    model.eval()
    T = rrxor_tensors(); pi = stationary(T)

    rng = np.random.default_rng(0)
    L = args.seq_len
    # records[(layer, head, offset)] -> list of (weight, query_phase, source_violation)
    weights, qphase, sviol = {}, {}, {}
    for layer in (1, 2, 3):
        for head in range(ta["n_heads"]):
            for off in range(1, args.max_offset + 1):
                weights[(layer, head, off)] = []
                qphase[(layer, head, off)] = []
                sviol[(layer, head, off)] = []

    print(f"collecting attention from {args.n_seqs} sequences...")
    for _ in range(args.n_seqs):
        ems, states = sample_sequence(T, pi, L, rng)
        tp = STATE_PHASE[states]
        viol = np.full(L, -1, dtype=np.int64)
        viol[2:] = (ems[2:] != (ems[1:-1] ^ ems[:-2])).astype(np.int64)
        x = torch.from_numpy(ems).long().unsqueeze(0)
        cache = hooked_forward_attn(model, x)
        for layer in (1, 2, 3):
            aw = cache[f"L{layer}_attn_w"].squeeze(0).numpy()  # (heads, L, L)
            for head in range(aw.shape[0]):
                for t in range(args.t_min, L):
                    for off in range(1, args.max_offset + 1):
                        s = t - off
                        weights[(layer, head, off)].append(aw[head, t, s])
                        qphase[(layer, head, off)].append(tp[t])
                        sviol[(layer, head, off)].append(viol[s])

    out = {}
    for layer in (1, 2, 3):
        for head in range(ta["n_heads"]):
            print(f"\n=== L{layer} head {head} ===")
            print(f"  {'off':>4} {'mean_w':>7} | {'q_ph0':>6} {'q_ph1':>6} {'q_ph2':>6} | "
                  f"{'src_ok':>7} {'src_viol':>8} | {'eta2_qph':>8} {'eta2_viol':>9}")
            for off in range(1, args.max_offset + 1):
                w = np.array(weights[(layer, head, off)])
                qp = np.array(qphase[(layer, head, off)])
                sv = np.array(sviol[(layer, head, off)])
                m = sv >= 0
                w, qp, sv = w[m], qp[m], sv[m]
                by_qp = [w[qp == p].mean() for p in range(3)]
                by_sv = [w[sv == v].mean() if (sv == v).sum() > 10 else np.nan for v in (0, 1)]
                tot = w.var()
                eta_qp = (sum((qp == p).sum() * (w[qp == p].mean() - w.mean())**2 for p in range(3))
                          / max(len(w) * tot, 1e-12))
                eta_sv = (sum((sv == v).sum() * (w[sv == v].mean() - w.mean())**2 for v in (0, 1))
                          / max(len(w) * tot, 1e-12))
                key = f"L{layer}_h{head}_off{off}"
                out[key] = {"mean_w": float(w.mean()), "by_query_phase": [float(v) for v in by_qp],
                            "by_source_violation": [float(v) for v in by_sv],
                            "eta2_query_phase": float(eta_qp), "eta2_violation": float(eta_sv)}
                if w.mean() > 0.01 or max(eta_qp, eta_sv) > 0.05:
                    print(f"  {-off:>4} {w.mean():>7.3f} | "
                          f"{by_qp[0]:>6.3f} {by_qp[1]:>6.3f} {by_qp[2]:>6.3f} | "
                          f"{by_sv[0]:>7.3f} {by_sv[1]:>8.3f} | {eta_qp:>8.3f} {eta_sv:>9.3f}")

    with open(os.path.join(args.outdir, "l3_attn_phase.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.outdir}/l3_attn_phase.json")


if __name__ == "__main__":
    main()
