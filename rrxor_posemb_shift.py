"""
Causal test of the positional-embedding-phase hypothesis.

If the phase gate were read off absolute position (mod 3), feeding the same bit
sequence at positional offset k would shift the gate by k mod 3 and destroy
phase-2 accuracy for k % 3 != 0.  If phase is inferred from content, behavior
should be invariant (up to mild OOD degradation) for all k.

Shifts keep relative positions intact, so the offset(-1) attention at L2 is
unaffected; this isolates the absolute-position channel.
"""

import argparse, json, os
import numpy as np
import torch

from rrxor_belief_experiment import (
    rrxor_tensors, stationary, sample_sequence, MiniTransformer, STATE_PHASE,
)


@torch.no_grad()
def forward_with_pos_offset(model, x, k):
    B, L = x.shape
    pos_idx = torch.arange(k, k + L).unsqueeze(0).expand(B, L)
    h = model.tok(x) + model.pos(pos_idx)
    mask = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)
    for blk in model.blocks:
        h = blk(h, src_mask=mask)
    return model.head(model.ln(h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="results/rrxor_belief/naive_ctx")
    ap.add_argument("--outdir", default="results/rrxor_belief/naive_ctx/circuit")
    ap.add_argument("--n-seqs", type=int, default=120)
    ap.add_argument("--seq-len", type=int, default=128)
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
    seqs, phases = [], []
    for _ in range(args.n_seqs):
        ems, states = sample_sequence(T, pi, args.seq_len, rng)
        seqs.append(ems)
        phases.append(STATE_PHASE[states])
    X = torch.from_numpy(np.stack(seqs)).long()
    PH = np.stack(phases)
    ems_all = np.stack(seqs)
    xor = np.zeros_like(ems_all); xor[:, 1:] = ems_all[:, 1:] ^ ems_all[:, :-1]

    out = {}
    print(f"{'k':>4} {'k%3':>4} {'ph2 acc':>8} {'|ld| ph2':>9} {'|ld| ph01':>10}")
    for k in [0, 1, 2, 3, 4, 6, 30, 100]:
        lg = forward_with_pos_offset(model, X, k).numpy()
        ld = lg[:, :, 1] - lg[:, :, 0]
        # evaluate at t >= 16 (synchronized regime)
        m2 = (PH[:, 16:] == 2)
        ld_e = ld[:, 16:]; xor_e = xor[:, 16:]
        acc2 = float((((ld_e > 0) == (xor_e == 1))[m2]).mean())
        g2 = float(np.abs(ld_e[m2]).mean())
        g01 = float(np.abs(ld_e[~m2]).mean())
        out[f"k={k}"] = {"phase2_acc": acc2, "gate_ph2": g2, "gate_ph01": g01}
        print(f"{k:>4} {k%3:>4} {acc2:>8.4f} {g2:>9.2f} {g01:>10.2f}")

    with open(os.path.join(args.outdir, "posemb_shift.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.outdir}/posemb_shift.json")


if __name__ == "__main__":
    main()
