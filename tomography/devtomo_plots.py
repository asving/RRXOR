"""Figures for developmental tomography v2 (reads devtomo2.npz etc. from a run dir)."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

name = sys.argv[1] if len(sys.argv) > 1 else "m27_nl6_seed1"
base = os.path.dirname(os.path.abspath(__file__))
run = os.path.join(base, "runs", name)
z = np.load(os.path.join(run, "devtomo2.npz"), allow_pickle=True)
A, steps, bnames, wins, CM, cnames = (z["A"], z["steps"], list(z["bnames"]),
                                      z["wins"], z["CM"], list(z["cnames"]))
WIRE = np.load(os.path.join(run, "devtomo2_wire.npy"))
RB = json.load(open(os.path.join(run, "devtomo2_rollback.json")))
K, NB = len(cnames), len(bnames)
mid = (steps[:-1] + steps[1:]) / 2
rate = A / np.maximum(np.diff(steps), 1)[:, None, None] * 1000    # nats per 1k steps

CCOL = {}
for c in cnames:
    if "xor_" in c or c == "det": CCOL[c] = "#3b7dd8"
    elif "xnor" in c: CCOL[c] = "#e04b5a"
    elif c in ("match",) or c.endswith("_c0"): CCOL[c] = "#4fae62"
    else: CCOL[c] = "#e39b2d"

# ---------------- Figure 1: the construction log ----------------
vmax = np.percentile(np.abs(rate), 99)
fig, axes = plt.subplots(K, 1, figsize=(13, 2.1 * K + 1.6), sharex=True)
edges = np.concatenate([[max(steps[0], 20)], steps[1:]])
for a, ax in enumerate(axes):
    pc = ax.pcolormesh(edges, np.arange(NB + 1), rate[:, a, :].T, cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax)
    ax.set_yticks(np.arange(NB) + 0.5)
    ax.set_yticklabels(bnames, fontsize=5.5)
    ax.set_xscale("log")
    for (w0, w1) in wins:
        ax.axvline(max(w0, 21), color="k", ls=":", lw=0.7, alpha=0.6)
    ax2 = ax.twinx()
    ax2.plot(steps, CM[a], color="k", lw=1.6)
    ax2.plot(steps, CM[a], color=CCOL[cnames[a]], lw=1.2)
    ax2.set_ylabel("CE", fontsize=7)
    ax.set_ylabel(cnames[a], fontsize=9, color=CCOL[cnames[a]], fontweight="bold")
axes[-1].set_xlabel("training step")
fig.colorbar(pc, ax=axes, fraction=0.015, pad=0.06,
             label="attribution rate (nats / 1k steps)\nred = hurts cluster · blue = helps")
fig.suptitle(f"{name}: the construction log — which blocks' realized updates move each "
             f"task, when\n(line = cluster loss; dotted = discovered windows)", fontsize=11)
fig.savefig(os.path.join(run, "tomo_constructionlog.png"), dpi=150, bbox_inches="tight")
print("saved tomo_constructionlog.png")

# ---------------- Figure 2: rollback ladder + wiring ----------------
fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.6), width_ratios=[1.5, 1])
ax = axes[0]
none = RB["none"]
wlist = [f"{w0}-{w1}" for (w0, w1) in wins]
for wi, wl in enumerate(wlist):
    for a in range(K):
        if f"{wl}:top4" in RB:
            ax.plot(wi - 0.08, RB[f"{wl}:top4"]["ce"][a], "o", ms=7,
                    color=CCOL[cnames[a]])
        if f"{wl}:rand4" in RB:
            ax.plot(wi + 0.12, RB[f"{wl}:rand4"]["ce"][a], "o", ms=5, mfc="none",
                    color=CCOL[cnames[a]], alpha=0.7)
refs = ([(0.6931, "ln2"), (1.5041, "xnor @ mod-3 stage"), (1.0986, "flip @ mod-9 stage"),
         (0.4055, "xnor @ mod-9 stage"), (0.2513, "xor @ mod-3 stage")]
        if any("_c" in c for c in cnames) else
        [(0.6931, "ln2"), (1.0986, "mis @ ungated stage"), (0.4055, "det/match @ ungated")])
for y, lab in refs:
    ax.axhline(y, color="gray", ls="--", lw=0.7, alpha=0.6)
    ax.text(len(wlist) - 0.4, y + 0.02, lab, fontsize=6.5, color="gray")
for a in range(K):
    ax.plot(-0.6, none[a], "*", ms=11, color=CCOL[cnames[a]])
ax.text(-0.75, max(none) + 0.1, "final\nmodel", fontsize=7, ha="center")
ax.set_xticks(range(len(wlist)))
ax.set_xticklabels([w.replace("-", "–\n") for w in wlist], fontsize=6.5)
ax.set_xlabel("window rolled back (top-4 attributed blocks ● vs random-4 ○)")
ax.set_ylabel("per-cluster CE after rollback (nats)")
ax.set_title("rollback = weight-space time travel:\nundoing a window's blocks restores earlier ladder rungs")
handles = [plt.Line2D([], [], marker="o", ls="", color=CCOL[c], label=c) for c in cnames]
ax.legend(handles=handles, fontsize=7, loc="upper left")

ax = axes[1]
im = ax.imshow(np.where(WIRE > 0, WIRE, np.nan), cmap="viridis", vmin=0.2, vmax=0.9)
ax.set_xticks(range(len(wlist))); ax.set_xticklabels(wlist, rotation=45, fontsize=6)
ax.set_yticks(range(len(wlist))); ax.set_yticklabels(wlist, fontsize=6)
ax.set_xlabel("reader window"); ax.set_ylabel("writer window")
ax.set_title("socket–plug wiring:\ncos(write dirs, later read dirs)")
for i in range(len(wlist)):
    for j in range(len(wlist)):
        if WIRE[i, j] > 0:
            ax.text(j, i, f"{WIRE[i,j]:.2f}", ha="center", va="center", fontsize=5.5,
                    color="w" if WIRE[i, j] < 0.7 else "k")
fig.colorbar(im, ax=ax, fraction=0.045)
fig.suptitle(f"{name}: causal validation and wiring", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(run, "tomo_validation.png"), dpi=150, bbox_inches="tight")
print("saved tomo_validation.png")
