"""PREREG5 grading + figure: correction-conditional geometry trajectories."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

base = os.path.dirname(os.path.abspath(__file__))
RUNS = ["seed0", "m27_nl6_seed1", "fx0.03_seed0", "fx0.2_seed0"]
TITLES = {"seed0": "rrxor3 (seed0)", "m27_nl6_seed1": "matryoshka Z27 (nl6 seed1)",
          "fx0.03_seed0": "leaky fx ε=0.03", "fx0.2_seed0": "leaky fx ε=0.2"}
data = {r: json.load(open(os.path.join(base, "runs", r, "corrprobe.json"))) for r in RUNS}

M27_RUNGS = [0.6931, 0.6758, 0.6386, 0.5328, 0.4621]

def m27_eras(ev):
    steps = np.array([e["step"] for e in ev]); ce = np.array([e["ce_avg"] for e in ev])
    eras = []
    for k in range(1, len(M27_RUNGS)):
        hi, lo = M27_RUNGS[k - 1] - 0.01, M27_RUNGS[k] + 0.01
        above = steps[ce > hi]; below = steps[ce < lo]
        s0 = above.max() if len(above) else steps[0]
        s1 = below.min() if len(below) else steps[-1]
        eras.append((int(s0), int(max(s1, s0 + 1))))
    return eras   # era k -> transition to rung k+1

ERAS = {"seed0": [(50, 550), (550, 1450)],           # devtomo change-point windows
        "m27_nl6_seed1": m27_eras(data["m27_nl6_seed1"]["eval_loss"])}
print("m27 eras (ladder traversals):", ERAS["m27_nl6_seed1"])

def series(r, k, predet=False):
    d = data[r]
    t = d["traj_predet"] if predet else d["traj"]
    return np.array(d["steps"]), np.array(t[k], dtype=float)

def rise_time(steps, D):
    Df = np.nanmean(D[-3:])
    if Df < 0.10:
        return None, Df
    idx = np.where(D > 0.5 * Df)[0]
    return (int(steps[idx[0]]) if len(idx) else None), Df

print("\n=== rise times (tau = first ckpt > 0.5 * final excess) ===")
summary = {}
for r in RUNS:
    d = data[r]
    summary[r] = {}
    for k in d["traj"]:
        steps, D = series(r, k)
        if k in ("u", "m"):                       # raw-R2 grading (prereg amendment)
            D = D - D[0]
        tau, Df = rise_time(steps, D)
        peak = float(np.nanmax(D))
        summary[r][k] = {"tau": tau, "final": round(float(Df), 3),
                         "peak": round(peak, 3)}
        print(f"  {r:16s} {k:4s}: tau {str(tau):>6s}  final {Df:+.3f}  peak {peak:+.3f}")
    if d.get("traj_predet"):
        for k in d["traj_predet"]:
            steps, D = series(r, k, predet=True)
            tau, Df = rise_time(steps, D)
            summary[r][k + "_predet"] = {"tau": tau, "final": round(float(Df), 3),
                                         "peak": round(float(np.nanmax(D)), 3)}
            print(f"  {r:16s} {k + '@pd':6s}: tau {str(tau):>6s}  final {Df:+.3f}  "
                  f"peak {np.nanmax(D):+.3f}")

json.dump({"summary": summary, "eras": {k: v for k, v in ERAS.items()}},
          open(os.path.join(base, "corrprobe_summary.json"), "w"), indent=1)

COLS = {"u": "#e0a63b", "m": "#b08a5a", "Q3": "#4fae62", "Q9x": "#3b7dd8",
        "EXC": "#8a5fd6", "USE": "#e04b5a", "Lam": "#3b7dd8", "Q": "#4fae62",
        "F": "#8a5fd6", "g9": "#2f8f9d", "g27": "#d67fb8"}
fig, axes = plt.subplots(2, 2, figsize=(14.5, 9))
for ax, r in zip(axes.flat, RUNS):
    d = data[r]
    for k in d["traj"]:
        steps, D = series(r, k)
        lab = k
        if k in ("u", "m"):
            D = D - D[0]; lab = k + " (raw ΔR²)"
        ax.plot(steps, D, "o-", ms=3, color=COLS.get(k, "#999"), label=lab)
    if d.get("traj_predet"):
        for k in ("EXC", "USE"):
            steps, D = series(r, k, predet=True)
            ax.plot(steps, D, "--", lw=1.2, color=COLS[k], alpha=0.7,
                    label=k + " @pre-det")
    for i, (s0, s1) in enumerate(ERAS.get(r, [])):
        ax.axvspan(s0, s1, color=f"C{i}", alpha=0.08)
        ax.text((s0 * s1) ** 0.5, ax.get_ylim()[1] * 0.02 + 0.9, f"era{i+1}",
                fontsize=7, color="#666", ha="center")
    ax.axhline(0.10, color="#888", ls=":", lw=0.8)
    ax.axhline(0.05, color="#bbb", ls=":", lw=0.8)
    ax.set_xscale("log"); ax.set_xlabel("training step")
    ax.set_ylabel("decodability excess over window16 baseline")
    ax.set_title(TITLES[r], fontsize=11)
    ax.legend(fontsize=7.5, ncol=2)
fig.suptitle("PREREG5: correction-conditional belief geometry — probe trajectories "
             "D_ℓ(t)\n(shaded = loss-ladder transition eras; prediction: each quotient "
             "rises in its own era; USE never)", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(base, "corrprobe.png"), dpi=140)
print("\nsaved corrprobe.png + corrprobe_summary.json")
