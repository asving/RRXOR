"""
Close the L3_mlp loose end: propagate each contrast piece of each late-layer lookup
table through the final LayerNorm + unembed, per cell.

For each of the 12 cells c, take the mean pre-final-LN residual h_c.  For a given
component's contrast piece  piece_k(c) = c_k[c] * beta_k  (from the 12x64 lookup
decomposition), the contribution of that piece to the logit difference at cell c is

    contrib_k(c) = u . LN(h_c) - u . LN(h_c - piece_k(c))

which respects LN's nonlinearity (removing a norm-108 vector changes the LN scale).
If the L3_mlp b_prev piece's contribution is XOR-signed at phase-2 cells, the
"LN folds (b_prev, b_curr) into XOR" hypothesis is confirmed.

Also: inertness check — per-token fluctuations of L3_mlp_delta around its cell mean,
pushed through the LN Jacobian at h_c, compared to the systematic contributions.
"""

import argparse, json, os
import numpy as np
import torch

from rrxor_belief_experiment import (
    rrxor_tensors, stationary, sample_sequence, beliefs_from_emissions,
    MiniTransformer, STATE_PHASE,
)
from rrxor_reconstruct_probe import hooked_forward
from rrxor_lookup_decompose import build_contrasts, make_conds_12

CELL_NAMES = [f"bp{c//6}_bc{(c%6)//3}_ph{c%3}" for c in range(12)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="results/rrxor_belief/naive_ctx")
    ap.add_argument("--outdir", default="results/rrxor_belief/naive_ctx/circuit")
    ap.add_argument("--n-seqs", type=int, default=200)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--t-min", type=int, default=16, help="synchronized positions only")
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
    D = ta["d_model"]

    @torch.no_grad()
    def logit_diff_of(h):  # h: (M, D) numpy -> (M,) logit_diff through LN + head
        t = torch.from_numpy(h.astype(np.float32))
        lg = model.head(model.ln(t)).numpy()
        return lg[:, 1] - lg[:, 0]

    # --- collect per-cell means of pre-final-LN residual & L3_mlp deltas ---
    rng = np.random.default_rng(0)
    H, LD, conds = [], [], []
    l3mlp = []
    print(f"collecting {args.n_seqs} sequences...")
    for _ in range(args.n_seqs):
        ems, _ = sample_sequence(T, pi, args.seq_len, rng)
        b = beliefs_from_emissions(T, pi, ems)
        phase = np.array([STATE_PHASE[np.argmax(bt)] for bt in b[1:]])
        cond = make_conds_12(ems, phase, args.seq_len)
        cond[:args.t_min] = -1
        x = torch.from_numpy(ems).long().unsqueeze(0)
        cache = hooked_forward(model, x)
        H.append(cache["L3_post_mlp"].squeeze(0).numpy())
        lg = cache["logits"].squeeze(0).numpy()
        LD.append(lg[:, 1] - lg[:, 0])
        l3mlp.append(cache["L3_mlp_delta"].squeeze(0).numpy())
        conds.append(cond)
    H = np.concatenate(H); LD = np.concatenate(LD)
    l3mlp = np.concatenate(l3mlp); conds = np.concatenate(conds)
    valid = conds >= 0

    h_c = np.zeros((12, D)); ld_c = np.zeros(12)
    for c in range(12):
        m = conds == c
        h_c[c] = H[m].mean(axis=0)
        ld_c[c] = LD[m].mean()

    # sanity: logit_diff of mean residual vs mean logit_diff
    ld_of_mean = logit_diff_of(h_c)
    print("\n=== sanity: u.LN(mean resid) vs mean logit_diff per cell ===")
    for c in range(12):
        print(f"  {CELL_NAMES[c]}: LN(mean)={ld_of_mean[c]:>7.2f}   mean(LN)={ld_c[c]:>7.2f}")

    # --- decompose late-layer lookup tables into contrast pieces ---
    tables = np.load(os.path.join(args.outdir, "lookup_tables_12.npz"))
    C, names = build_contrasts()

    results = {}
    for comp in ["L3_mlp_delta", "L3_attn_delta", "L2_mlp_delta"]:
        table = tables[comp]  # (12, D)
        print(f"\n=== contribution of {comp} contrast pieces to logit_diff "
              f"(remove piece from pre-LN residual) ===")
        comp_res = {}
        # contribution of each contrast piece, per cell
        hdr = "  ".join(f"{CELL_NAMES[c]:>13}" for c in range(12))
        print(f"  {'piece':>22} | {hdr}")
        for j, name in enumerate(names):
            ck = C[:, j]
            beta = (table.T @ ck) / (ck * ck).sum()      # (D,)
            piece = np.outer(ck, beta)                    # (12, D)
            ld_without = logit_diff_of(h_c - piece)
            contrib = ld_of_mean - ld_without
            if np.abs(contrib).max() < 0.5:
                continue
            comp_res[name] = contrib.tolist()
            print(f"  {name:>22} | " + "  ".join(f"{v:>13.2f}" for v in contrib))
        results[comp] = comp_res

    # --- whole-component removal (compare with kill ablations) ---
    print("\n=== whole-component removal at the mean (cf. circuit_prune) ===")
    for comp in ["L3_mlp_delta", "L3_attn_delta"]:
        ld_without = logit_diff_of(h_c - tables[comp])
        ph2 = [c for c in range(12) if c % 3 == 2]
        print(f"  remove {comp}: phase-2 cells logit_diff "
              + ", ".join(f"{CELL_NAMES[c]}={ld_without[c]:.2f}" for c in ph2))

    # --- inertness of within-cell fluctuations (exp 6, light) ---
    print("\n=== inertness: within-cell L3_mlp fluctuations through LN Jacobian ===")
    gamma = model.ln.weight.detach().numpy()
    u = (model.head.weight[1] - model.head.weight[0]).detach().numpy()
    out_inert = {}
    for c in range(12):
        m = conds == c
        resid_fluct = l3mlp[m] - l3mlp[m].mean(axis=0, keepdims=True)  # (M, D)
        h = h_c[c]
        mu, var = h.mean(), h.var()
        hhat = (h - mu) / np.sqrt(var + 1e-5)
        # LN Jacobian at h_c: J = (I - 1/D - outer(hhat,hhat)/D) / sqrt(var)
        g = u * gamma
        Jg = (g - g.mean() - hhat * (g @ hhat) / D) / np.sqrt(var + 1e-5)
        proj = resid_fluct @ Jg
        out_inert[CELL_NAMES[c]] = {"std_logit_from_fluct": float(proj.std()),
                                    "mean_cell_logit": float(ld_of_mean[c])}
        print(f"  {CELL_NAMES[c]}: std of fluct-induced logit_diff = {proj.std():.3f} "
              f"(cell mean logit_diff = {ld_of_mean[c]:.2f})")

    with open(os.path.join(args.outdir, "ln_propagate.json"), "w") as f:
        json.dump({
            "cell_names": CELL_NAMES,
            "ld_of_mean_resid": ld_of_mean.tolist(),
            "mean_ld_per_cell": ld_c.tolist(),
            "contrast_contributions": results,
            "inertness": out_inert,
        }, f, indent=2)
    print(f"\nwrote {args.outdir}/ln_propagate.json")


if __name__ == "__main__":
    main()
