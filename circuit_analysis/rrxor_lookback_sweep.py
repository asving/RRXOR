"""
How does R^2-by-condition grow as we widen the lookback window?
And: does adding absolute-position information further help?

If R^2 -> ~1 under (bits, phase) alone for some k, the layer is a Boolean function.
If it saturates short of 1 but jumps when we add (position mod 12) or similar, the
layer has position-coupled computation that must be modeled explicitly.
"""

import argparse, json, os
import numpy as np
import torch

from rrxor_belief_experiment import (
    rrxor_tensors, stationary, sample_sequence, beliefs_from_emissions,
    MiniTransformer, STATE_PHASE,
)
from rrxor_reconstruct_probe import hooked_forward, variance_explained_by_condition


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="results/rrxor_belief/naive_ctx")
    ap.add_argument("--outdir", default="results/rrxor_belief/naive_ctx/circuit")
    ap.add_argument("--n-seqs", type=int, default=160)
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
    rng = np.random.default_rng(0)
    deltas = {f"L{k}_{w}": [] for k in range(4) for w in ("attn_delta", "mlp_delta")}
    ems_acc, ph_acc, t_acc = [], [], []
    for s_idx in range(args.n_seqs):
        ems, _ = sample_sequence(T, pi, args.seq_len, rng)
        b = beliefs_from_emissions(T, pi, ems)
        phase = np.array([STATE_PHASE[np.argmax(bt)] for bt in b[1:]])
        x = torch.from_numpy(ems).long().unsqueeze(0)
        cache = hooked_forward(model, x)
        for k, lst in deltas.items():
            lst.append(cache[k].squeeze(0).numpy())
        ems_acc.append(ems)
        ph_acc.append(phase)
        t_acc.append(np.arange(args.seq_len))
    deltas = {k: np.concatenate(v, axis=0) for k, v in deltas.items()}
    ems = np.concatenate(ems_acc)
    phases = np.concatenate(ph_acc)
    t_abs = np.concatenate(t_acc)

    N = len(ems)
    # Build bit history B = (bit[t-K], ..., bit[t-1], bit[t]) for K up to 5.
    K_max = 5
    valid = np.zeros(N, dtype=bool)
    valid[K_max:] = True
    bits = np.zeros((N, K_max + 1), dtype=np.int64)
    for k in range(K_max + 1):
        bits[K_max:, K_max - k] = ems[K_max - k : N - k]   # bits[:, 0] = bit[t-K_max], bits[:, K_max] = bit[t]

    print("=== R^2 by (last-k bits, phase) for each layer's residual delta ===")
    print("    k=0 ⇒ phase only (3 cells); k=K ⇒ (last K+1 bits, phase) ⇒ 2^(K+1) * 3 cells")
    rows = []
    layers = list(deltas.keys())
    header = ["component"] + [f"k={k}" for k in range(K_max + 1)] + ["+pos_mod_12"]
    print("  " + "  ".join(f"{h:>14}" for h in header))
    for comp in layers:
        d = deltas[comp][valid]
        row = [comp]
        for k_use in range(K_max + 1):
            # condition_id from last k_use+1 bits + phase
            sub = bits[valid][:, K_max - k_use:K_max + 1]  # shape (Nv, k_use+1)
            cond = np.zeros(sub.shape[0], dtype=np.int64)
            for col in range(sub.shape[1]):
                cond = cond * 2 + sub[:, col]
            cond = cond * 3 + phases[valid]
            n_conds = (2 ** (k_use + 1)) * 3
            r2 = variance_explained_by_condition(d, cond, n_conds)
            row.append(f"{r2:.3f}")
        # Add position mod 12 to the K_max condition
        sub = bits[valid][:, :]
        cond = np.zeros(sub.shape[0], dtype=np.int64)
        for col in range(sub.shape[1]):
            cond = cond * 2 + sub[:, col]
        cond = cond * 3 + phases[valid]
        # Combine with t mod 12 (must avoid blowing up n_conds beyond data per cell)
        pos_mod = t_abs[valid] % 12
        cond_pos = cond * 12 + pos_mod
        n_conds_pos = (2 ** (K_max + 1)) * 3 * 12
        r2_pos = variance_explained_by_condition(d, cond_pos, n_conds_pos)
        row.append(f"{r2_pos:.3f}")
        rows.append(row)
        print("  " + "  ".join(f"{x:>14}" for x in row))

    out = {"k_max": K_max, "rows": rows, "header": header}
    with open(os.path.join(args.outdir, "lookback_sweep.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.outdir}/lookback_sweep.json")


if __name__ == "__main__":
    main()
