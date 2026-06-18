"""
How does the model know the phase?  Phase is NOT a function of absolute position
(sequences start from the stationary distribution, so phase = (random offset + t) mod 3).

Tests:
  1. Synchronization curve: per-position-t phase accuracy of (a) the optimal Bayes
     observer (argmax of the phase-marginal posterior) and (b) a linear probe on the
     model's residual stream.  If the model's curve tracks Bayes, phase is inferred
     from the bit history, not read off position.
  2. Gate vs phase posterior: the model's logit_diff against the Bayes log-odds of
     the next bit, including positions where the Bayes phase posterior is still
     uncertain.  A graded match means the model carries a graded phase belief.
  3. Gate by (t mod 3) x (true phase): if the gate were driven by position mod 3 it
     could not follow the true phase across random per-sequence offsets.
  4. Adversarial all-zeros stream: consistent with every phase assignment forever,
     so the Bayes phase posterior never fully synchronizes.  Does the model's
     prediction match the Bayes prediction on this stream?
"""

import argparse, json, os
import numpy as np
import torch

from rrxor_belief_experiment import (
    rrxor_tensors, stationary, sample_sequence, beliefs_from_emissions,
    MiniTransformer, STATE_PHASE,
)
from rrxor_reconstruct_probe import hooked_forward


def bayes_quantities(T, pi, ems):
    """Per position t: phase posterior (3,), P(next bit = 1)."""
    b = beliefs_from_emissions(T, pi, ems)  # (L+1, 5); b[t+1] = belief after ems[0..t]
    bb = b[1:]                              # belief about state after emission t
    phase_post = np.stack([bb[:, STATE_PHASE == p].sum(axis=1) for p in range(3)], axis=1)
    p_next1 = bb @ T[:, 1, :].sum(axis=1)
    return phase_post, p_next1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="results/rrxor_belief/naive_ctx")
    ap.add_argument("--outdir", default="results/rrxor_belief/naive_ctx/circuit")
    ap.add_argument("--n-seqs", type=int, default=300)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--probe-stage", default="L3_post_attn")
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
    L = args.seq_len
    n = args.n_seqs

    acts = []            # (n, L, D) at probe stage
    logit_diff = []      # (n, L)
    true_phase = []      # (n, L)  phase of next emission site, from TRUE states
    bayes_phase = []     # (n, L, 3)
    bayes_p1 = []        # (n, L)
    print(f"collecting {n} sequences of length {L}...")
    for _ in range(n):
        ems, states = sample_sequence(T, pi, L, rng)
        x = torch.from_numpy(ems).long().unsqueeze(0)
        cache = hooked_forward(model, x)
        acts.append(cache[args.probe_stage].squeeze(0).numpy())
        lg = cache["logits"].squeeze(0).numpy()
        logit_diff.append(lg[:, 1] - lg[:, 0])
        true_phase.append(STATE_PHASE[states])
        pp, p1 = bayes_quantities(T, pi, ems)
        bayes_phase.append(pp)
        bayes_p1.append(p1)
    acts = np.stack(acts)                # (n, L, D)
    logit_diff = np.stack(logit_diff)
    true_phase = np.stack(true_phase)
    bayes_phase = np.stack(bayes_phase)
    bayes_p1 = np.stack(bayes_p1)

    # --- 1. Synchronization curve -----------------------------------------
    # Bayes accuracy per t
    bayes_pred = bayes_phase.argmax(axis=2)
    bayes_acc_t = (bayes_pred == true_phase).mean(axis=0)
    # Model probe: train one probe on positions t >= 16 (where phase is knowable),
    # evaluate per position on held-out sequences.
    D = acts.shape[2]
    ntr = int(0.8 * n)
    Xtr = acts[:ntr, 16:].reshape(-1, D)
    ytr = true_phase[:ntr, 16:].reshape(-1)
    W, *_ = np.linalg.lstsq(
        np.concatenate([Xtr, np.ones((len(Xtr), 1))], axis=1), np.eye(3)[ytr], rcond=None)
    Xte = np.concatenate([acts[ntr:], np.ones((n - ntr, L, 1))], axis=2)
    pred = (Xte @ W).argmax(axis=2)
    model_acc_t = (pred == true_phase[ntr:]).mean(axis=0)

    print(f"\n=== 1. synchronization curve (probe stage: {args.probe_stage}) ===")
    print(f"{'t':>4} {'bayes_acc':>10} {'probe_acc':>10} {'gate@ph2':>10} {'gate@ph01':>10}")
    gate2_t = np.full(L, np.nan); gate01_t = np.full(L, np.nan)
    for t in range(L):
        m2 = true_phase[:, t] == 2
        if m2.sum() > 5:
            gate2_t[t] = np.abs(logit_diff[m2, t]).mean()
        if (~m2).sum() > 5:
            gate01_t[t] = np.abs(logit_diff[~m2, t]).mean()
    for t in list(range(0, 16)) + [20, 24, 32, 48, 64, 96, 127]:
        print(f"{t:>4} {bayes_acc_t[t]:>10.3f} {model_acc_t[t]:>10.3f} "
              f"{gate2_t[t]:>10.2f} {gate01_t[t]:>10.2f}")

    # --- 2. graded gate vs Bayes posterior --------------------------------
    bayes_logodds = np.log(np.clip(bayes_p1, 1e-9, 1 - 1e-9) /
                           np.clip(1 - bayes_p1, 1e-9, 1 - 1e-9))
    ld = logit_diff.reshape(-1); blo = bayes_logodds.reshape(-1)
    pph2 = bayes_phase[:, :, 2].reshape(-1)
    r_all = np.corrcoef(ld, np.clip(blo, -15, 15))[0, 1]
    # Restrict to positions where the Bayes phase-2 posterior is genuinely uncertain
    mid = (pph2 > 0.25) & (pph2 < 0.75)
    r_mid = np.corrcoef(ld[mid], np.clip(blo[mid], -15, 15))[0, 1] if mid.sum() > 50 else np.nan
    # Gate magnitude binned by Bayes phase-2 posterior
    print("\n=== 2. gate vs Bayes phase-2 posterior ===")
    print(f"  corr(logit_diff, bayes_logodds) all positions: {r_all:.3f}")
    print(f"  corr on phase-uncertain positions (0.25<P(ph2)<0.75, n={int(mid.sum())}): {r_mid:.3f}")
    bins = [0, .1, .25, .5, .75, .9, .999, 1.001]
    print(f"  {'P(ph2) bin':>16} {'n':>7} {'mean |logit_diff|':>18} {'mean |bayes_lo|':>16}")
    gate_bins = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (pph2 >= lo) & (pph2 < hi)
        if m.sum() < 20: continue
        row = (f"[{lo:.2f},{hi:.2f})", int(m.sum()),
               float(np.abs(ld[m]).mean()), float(np.abs(np.clip(blo[m], -15, 15)).mean()))
        gate_bins.append(row)
        print(f"  {row[0]:>16} {row[1]:>7} {row[2]:>18.2f} {row[3]:>16.2f}")

    # --- 3. gate by (t mod 3) x true phase --------------------------------
    print("\n=== 3. mean |logit_diff| by (t mod 3, true phase) ===")
    tmod = np.tile(np.arange(L) % 3, (n, 1)).reshape(-1)
    tp = true_phase.reshape(-1)
    table = np.zeros((3, 3))
    for a in range(3):
        for p in range(3):
            m = (tmod == a) & (tp == p)
            table[a, p] = np.abs(ld[m]).mean()
    print("            phase0   phase1   phase2")
    for a in range(3):
        print(f"  t%3={a}   {table[a,0]:>7.2f}  {table[a,1]:>7.2f}  {table[a,2]:>7.2f}")

    # --- 4. all-zeros adversarial stream ----------------------------------
    print("\n=== 4. all-zeros stream (never fully synchronizes) ===")
    ems0 = np.zeros(L, dtype=np.int64)
    pp0, p10 = bayes_quantities(T, pi, ems0)
    with torch.no_grad():
        lg0 = model(torch.from_numpy(ems0).long().unsqueeze(0)).squeeze(0).numpy()
    ld0 = lg0[:, 1] - lg0[:, 0]
    model_p1 = 1 / (1 + np.exp(-ld0))
    print(f"  {'t':>4} {'bayes P(ph2)':>13} {'bayes P(1)':>11} {'model P(1)':>11} {'logit_diff':>11}")
    for t in list(range(0, 12)) + [16, 24, 48, 96, 127]:
        print(f"  {t:>4} {pp0[t,2]:>13.3f} {p10[t]:>11.3f} {model_p1[t]:>11.3f} {ld0[t]:>11.2f}")
    r0 = np.corrcoef(model_p1, p10)[0, 1]
    print(f"  corr(model P(1), bayes P(1)) on all-zeros stream: {r0:.3f}")
    print(f"  max |bayes P(1) - model P(1)|: {np.abs(p10 - model_p1).max():.3f}")

    out = {
        "probe_stage": args.probe_stage,
        "bayes_acc_t": bayes_acc_t.tolist(),
        "model_probe_acc_t": model_acc_t.tolist(),
        "gate_phase2_by_t": gate2_t.tolist(),
        "gate_phase01_by_t": gate01_t.tolist(),
        "corr_logitdiff_bayes_logodds_all": float(r_all),
        "corr_logitdiff_bayes_logodds_phase_uncertain": float(r_mid),
        "gate_by_bayes_ph2_bins": gate_bins,
        "gate_by_tmod3_phase": table.tolist(),
        "allzeros": {
            "bayes_p_ph2": pp0[:, 2].tolist(), "bayes_p1": p10.tolist(),
            "model_p1": model_p1.tolist(),
            "corr": float(r0), "max_abs_gap": float(np.abs(p10 - model_p1).max()),
        },
    }
    with open(os.path.join(args.outdir, "phase_sync.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.outdir}/phase_sync.json")


if __name__ == "__main__":
    main()
