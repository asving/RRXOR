"""Robustness: kappa per quantile; amplitude-contraction check; figure."""
import json, math
import torch
from pcfg4 import dirichlet_tables
# per-level single-child channel contraction (second singular value of centered Markov op)
for aname, alpha in (("pcfg4b a=0.05", 0.05),):
    P = dirichlet_tables(alpha)
    lams = []
    for lev in range(4):
        for side, M in (("L", P[lev].sum(2)), ("R", P[lev].sum(1))):   # P(b|a), P(c|a)
            Mc = M - M.mean(0, keepdim=True)
            s = torch.linalg.svdvals(Mc)
            lams.append(float(s[0]))
        print(f"{aname} level {lev+1}: contraction lambda(L)={lams[-2]:.3f} lambda(R)={lams[-1]:.3f}")
g = {0: 0.5261, 1: 0.2127, 2: 0.1421, 3: 0.0582}
print("measured info ratios g_b/g_b+1:", "  ".join(f"{g[b]/g[b+1]:.2f}" for b in range(3)),
      "| sqrt (amplitude ratios):", "  ".join(f"{math.sqrt(g[b]/g[b+1]):.2f}" for b in range(3)))

# kappa per quantile, pcfg4b and pcfg2
data = {"pcfg4b": {"g": [0.2127, 0.1421, 0.0582],
                   0.2: [63, 58, 253], 0.5: [190, 293, 598], 0.8: [477, 755, 978]},
        "pcfg2":  {"g": [0.4022, 0.1176, 0.0223],
                   0.2: [79, 370, 531], 0.5: [229, 737, 1082], 0.8: [451, 1372, 1636]}}
def fit(xs, ys):
    n = len(xs); mx, my = sum(xs)/n, sum(ys)/n
    sl = sum((x-mx)*(y-my) for x, y in zip(xs, ys)) / sum((x-mx)**2 for x in xs)
    return sl
for name, d in data.items():
    ks = [f"q={q}: kappa={-fit([math.log(x) for x in d['g']], [math.log(t) for t in d[q]]):.2f}"
          for q in (0.2, 0.5, 0.8)]
    print(f"{name}: " + "  ".join(ks))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6, 4.5), facecolor="#111")
ax.set_facecolor("#111")
COLS = {"pcfg2 (det)": "#e6a817", "pcfg3 (e.05)": "#4fc3f7", "pcfg4 (a.3)": "#ef5350", "pcfg4b (a.05)": "#9ccc65"}
ALL = {"pcfg2 (det)":  ([0.4022, 0.1176, 0.0223], [229, 737, 1082]),
       "pcfg3 (e.05)": ([0.2312, 0.0495], [246, 1067]),
       "pcfg4 (a.3)":  ([0.0310], [155]),
       "pcfg4b (a.05)":([0.2127, 0.1421, 0.0582], [190, 293, 598])}
for name, (gs, ts) in ALL.items():
    ax.loglog(gs, ts, "o-", color=COLS[name], label=name, lw=1.5, ms=6)
import numpy as np
xs = np.logspace(-1.7, -0.3, 10)
ax.loglog(xs, 60 * xs**-1, "--", color="#888", lw=1, label="kappa=1 (info budget)")
ax.loglog(xs, 14 * xs**-2, ":", color="#888", lw=1, label="kappa=2 (square)")
ax.set_xlabel("contextual gain g_b (nats/position)", color="w")
ax.set_ylabel("tau50 (steps to 50% coverage)", color="w")
ax.set_title("time-to-learn vs information dilution, PCFG family", color="w", fontsize=11)
ax.tick_params(colors="w"); [s.set_color("#555") for s in ax.spines.values()]
ax.legend(fontsize=7, facecolor="#222", labelcolor="w", edgecolor="#444")
plt.tight_layout(); plt.savefig("ladder_model.png", dpi=140, facecolor="#111")
print("saved ladder_model.png")
