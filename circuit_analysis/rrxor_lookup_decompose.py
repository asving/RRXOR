"""
12-cell minimal lookup: (bit[t-1], bit[t], phase) -> residual delta.

For each layer, decompose the 12 x 64 lookup table into orthogonal contrasts:
  const, b_curr, b_prev, xor, phase_special (ph=2 vs ph=0,1), phase_01 (ph=0 vs 1),
  and their interactions. The contrast that carries the most variance tells us
  what the layer is "writing" into the residual stream.
"""

import argparse, json, os
import numpy as np
import torch

from rrxor_belief_experiment import (
    rrxor_tensors, stationary, sample_sequence, beliefs_from_emissions,
    MiniTransformer, STATE_PHASE,
)
from rrxor_reconstruct_probe import hooked_forward
from rrxor_lookup_substitute import gather_deltas, fit_lookup, substituted_forward, eval_substitution


N_PHASE = 3
N_CONDS_12 = 12   # 2 (b_prev) * 2 (b_curr) * 3 (phase)


def make_conds_12(ems, phases, seq_len):
    """Single sequence -> (L,) condition ids in [0, 12), -1 for t=0."""
    L = len(ems)
    cond = np.full(L, -1, dtype=np.int64)
    for t in range(1, L):
        b_prev = int(ems[t - 1])
        b_curr = int(ems[t])
        ph = int(phases[t])
        cond[t] = (b_prev * 2 + b_curr) * N_PHASE + ph
    return cond


def build_contrasts():
    """Return a (12, 12) matrix of orthogonal contrasts, plus their names.

    Cell ordering: c = (b_prev*2 + b_curr)*3 + phase
    Contrasts:
      1. const                : +1 everywhere
      2. b_curr               : 2*b_curr - 1
      3. b_prev               : 2*b_prev - 1
      4. xor                  : 2*(b_prev XOR b_curr) - 1
      5. ph_special           : +2 for phase=2, -1 for phase in {0,1}
      6. ph_01                : +1 for phase=1, -1 for phase=0, 0 for phase=2
      7. b_curr x ph_special  : product
      8. b_curr x ph_01       : product
      9. b_prev x ph_special  : product
     10. b_prev x ph_01       : product
     11. xor x ph_special     : product   (the "phase-gated XOR amplifier" contrast)
     12. xor x ph_01          : product
    """
    cols = []
    names = []
    cells = []
    for c in range(12):
        b_prev = c // 6
        b_curr = (c % 6) // 3
        ph = c % 3
        cells.append((b_prev, b_curr, ph))
    cells = np.array(cells)
    b_prev_s = 2 * cells[:, 0] - 1
    b_curr_s = 2 * cells[:, 1] - 1
    xor_s = b_prev_s * b_curr_s   # +1 for XOR=0, -1 for XOR=1 — equivalent up to sign
    ph = cells[:, 2]
    ph_special = np.where(ph == 2, 2.0, -1.0)
    ph_01 = np.where(ph == 0, -1.0, np.where(ph == 1, 1.0, 0.0))

    cols.append(np.ones(12));                names.append("const")
    cols.append(b_curr_s.astype(float));     names.append("b_curr")
    cols.append(b_prev_s.astype(float));     names.append("b_prev")
    cols.append(xor_s.astype(float));        names.append("xor")
    cols.append(ph_special);                 names.append("ph2_vs_01")
    cols.append(ph_01);                      names.append("ph0_vs_1")
    cols.append(b_curr_s * ph_special);      names.append("b_curr x ph2_vs_01")
    cols.append(b_curr_s * ph_01);           names.append("b_curr x ph0_vs_1")
    cols.append(b_prev_s * ph_special);      names.append("b_prev x ph2_vs_01")
    cols.append(b_prev_s * ph_01);           names.append("b_prev x ph0_vs_1")
    cols.append(xor_s * ph_special);         names.append("xor x ph2_vs_01")
    cols.append(xor_s * ph_01);              names.append("xor x ph0_vs_1")

    C = np.stack(cols, axis=1)   # (12, 12)
    return C, names


def decompose(table, C, names):
    """table: (12, D). C: (12, K) of orthogonal contrasts.
    Returns list of (name, beta_vec, norm_squared_explained, frac_explained)."""
    centered = table - table.mean(axis=0, keepdims=True)
    total_ss = (centered ** 2).sum()
    rows = []
    for j, name in enumerate(names):
        c = C[:, j]
        norm_sq = (c * c).sum()
        beta = (table.T @ c) / norm_sq   # (D,)
        # variance explained = ||beta||^2 * ||c||^2  (because the projection contributes
        # beta * c^T to the table; its Frobenius norm-squared is ||beta||^2 * ||c||^2)
        explained = float((beta * beta).sum() * norm_sq)
        rows.append({
            "name": name,
            "beta_norm": float(np.linalg.norm(beta)),
            "explained_ss": explained,
        })
    # Re-express as fractions of (centered) variance.  Note the "const" contrast doesn't
    # explain centered variance; we drop it from the fractional report.
    centered_total = total_ss
    for r in rows:
        r["frac_of_centered_var"] = (
            None if r["name"] == "const" else float(r["explained_ss"] / max(centered_total, 1e-12))
        )
    return rows, float(np.sqrt(total_ss + (table.mean(axis=0) ** 2).sum() * 12))   # full Frobenius


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="results/rrxor_belief/naive_ctx")
    ap.add_argument("--outdir", default="results/rrxor_belief/naive_ctx/circuit")
    ap.add_argument("--n-fit-seqs", type=int, default=320)
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

    # --- Fit 12-cell lookups using (b_prev, b_curr, phase) ---
    print(f"fitting 12-cell lookups ({args.n_fit_seqs} seqs)...")
    deltas_full, _ = gather_deltas(model, T, pi, args.n_fit_seqs, args.seq_len,
                                    np.random.default_rng(0))
    # We need conds_12, not conds_144.  Recompute conditions cell-by-cell.
    fit_rng = np.random.default_rng(0)
    conds_12 = []
    deltas_12 = {f"L{k}_{w}_delta": [] for k in range(4) for w in ("attn", "mlp")}
    for _ in range(args.n_fit_seqs):
        ems, _ = sample_sequence(T, pi, args.seq_len, fit_rng)
        b = beliefs_from_emissions(T, pi, ems)
        phase = np.array([STATE_PHASE[np.argmax(bt)] for bt in b[1:]])
        cond = make_conds_12(ems, phase, args.seq_len)
        x = torch.from_numpy(ems).long().unsqueeze(0)
        cache = hooked_forward(model, x)
        for k in deltas_12:
            deltas_12[k].append(cache[k].squeeze(0).numpy())
        conds_12.append(cond)
    deltas_12 = {k: np.concatenate(v, axis=0) for k, v in deltas_12.items()}
    conds_12 = np.concatenate(conds_12, axis=0)

    lookups = {}
    for key, d in deltas_12.items():
        table, counts = fit_lookup(d, conds_12, N_CONDS_12, D)
        lookups[key] = table
        print(f"  {key}: cells_filled={int((counts>0).sum())}/{N_CONDS_12}  "
              f"min={counts[counts>0].min()}  median={int(np.median(counts[counts>0]))}")

    # --- Verify the 12-cell lookup substitution still works ---
    print("\n=== sanity: full 12-cell-lookup substitution ===")
    # Need adapted eval; re-use substituted_forward but with 12-cell conds
    components = list(lookups.keys())
    # Run a quick eval inline:
    p2_correct, p2_total, ld_diffs = 0, 0, []
    ph_std = {0: [], 1: [], 2: []}
    eval_rng = np.random.default_rng(1)
    for _ in range(args.n_eval_seqs):
        ems, _ = sample_sequence(T, pi, args.seq_len, eval_rng)
        b = beliefs_from_emissions(T, pi, ems)
        phase = np.array([STATE_PHASE[np.argmax(bt)] for bt in b[1:]])
        cond = make_conds_12(ems, phase, args.seq_len)
        x = torch.from_numpy(ems).long().unsqueeze(0)
        with torch.no_grad():
            l_clean = model(x).squeeze(0).numpy()
        l_sub = substituted_forward(model, x, cond, set(components), lookups).squeeze(0).numpy()
        ld_clean = l_clean[:, 1] - l_clean[:, 0]
        ld_sub = l_sub[:, 1] - l_sub[:, 0]
        xor = np.zeros(args.seq_len, dtype=np.int64); xor[1:] = ems[1:] ^ ems[:-1]
        for t in range(1, args.seq_len):
            if phase[t] == 2:
                p2_total += 1
                if (ld_sub[t] > 0) == (xor[t] == 1):
                    p2_correct += 1
            ph_std[int(phase[t])].append(ld_sub[t])
        ld_diffs.append(ld_sub[1:] - ld_clean[1:])
    ld_diffs = np.concatenate(ld_diffs)
    print(f"  phase-2 accuracy = {p2_correct/max(p2_total,1):.4f}")
    print(f"  mean |Δlogit_diff| vs clean = {float(np.mean(np.abs(ld_diffs))):.3f}")
    for ph in (0, 1, 2):
        a = np.array(ph_std[ph])
        print(f"  phase {ph}: sub_mean={a.mean():.3f} sub_std={a.std():.3f}")

    # --- Decompose each lookup ---
    C, names = build_contrasts()
    print("\n=== contrast decomposition of each 12 x 64 lookup table ===")
    print(f"{'component':>18} | {'||const||':>10} | "
          f"top 3 non-const contrasts (frac of centered variance)")
    out_decomp = {}
    layer_order = [f"L{k}_{w}_delta" for k in range(4) for w in ("attn", "mlp")]
    for comp in layer_order:
        table = lookups[comp]
        rows, total_frob = decompose(table, C, names)
        # Sort non-const contrasts by explained
        nc_rows = [r for r in rows if r["name"] != "const"]
        nc_rows.sort(key=lambda r: -r["explained_ss"])
        const_norm = next(r["beta_norm"] for r in rows if r["name"] == "const")
        top3 = nc_rows[:3]
        report = "  ".join(
            f"{r['name']}={r['frac_of_centered_var']*100:.1f}% (||β||={r['beta_norm']:.2f})"
            for r in top3
        )
        print(f"  {comp:>18} | {const_norm:>10.2f} | {report}")
        out_decomp[comp] = {
            "const_norm": const_norm,
            "all_contrasts": rows,
            "total_frobenius": total_frob,
        }

    with open(os.path.join(args.outdir, "lookup_decompose.json"), "w") as f:
        json.dump({
            "decomp": out_decomp,
            "twelve_cell_eval": {
                "phase2_accuracy": p2_correct / max(p2_total, 1),
                "mean_abs_delta_logit_diff": float(np.mean(np.abs(ld_diffs))),
                "phase2_n": p2_total,
            },
            "contrast_names": names,
        }, f, indent=2)
    # Also save the raw 12-cell tables for inspection
    np.savez(os.path.join(args.outdir, "lookup_tables_12.npz"), **{
        k: v for k, v in lookups.items()
    })
    print(f"\nwrote {args.outdir}/lookup_decompose.json and lookup_tables_12.npz")


if __name__ == "__main__":
    main()
