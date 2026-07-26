import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

BG, PANEL, FG, DIM = "#0e1117", "#161b22", "#e6edf3", "#8b949e"
ACC = {"pcfg2": "#e6a817", "pcfg3": "#4fc3f7", "pcfg4": "#ef5350", "pcfg4b": "#9ccc65"}
plt.rcParams.update({"text.color": FG, "axes.edgecolor": "#30363d", "axes.labelcolor": FG,
                     "xtick.color": DIM, "ytick.color": DIM, "font.size": 9})

fig = plt.figure(figsize=(16.5, 11), facecolor=BG)
gs = GridSpec(3, 3, figure=fig, width_ratios=[1.15, 1, 1], height_ratios=[1, 1, 0.14],
              left=0.035, right=0.985, top=0.87, bottom=0.03, wspace=0.25, hspace=0.3)

fig.text(0.035, 0.965, "The Dilution Ladder", fontsize=26, weight="bold", color=FG)
fig.text(0.035, 0.925, "How hierarchical latents are learned from token supervision — theory vs. practice on the PCFG family "
         "(depth-4 trees, V=4; deterministic RHM → fully entropic Dirichlet grammars)",
         fontsize=11, color=DIM)

# ---------------- THEORY column ----------------
axT = fig.add_subplot(gs[:2, 0]); axT.set_facecolor(PANEL); axT.set_xticks([]); axT.set_yticks([])
for s in axT.spines.values(): s.set_color("#30363d")
th = [
 (0.97, r"$\bf{1.\ The\ observable}$", FG, 11),
 (0.925, r"Order-$r$ latents live $r$ levels above the tokens. Their consuming", DIM, 9),
 (0.895, r"class has Bayes floor $B_r$, zero-evidence marginal $M_r$, and", DIM, 9),
 (0.865, r"contextual information $g_r = M_r - B_r$ (nats/position).", DIM, 9),
 (0.81, r"$c_r(t)  =  \frac{M_r - \mathrm{CE}_r(t)}{M_r - B_r}\ \in [0,1]$", FG, 13),
 (0.745, r"$c_r(t)$ = fraction of order-$r$ information captured at sample $t$.", DIM, 9),
 (0.69, r"$\bf{2.\ Dilution\ (why\ tokens\ are\ partial\ views)}$", FG, 11),
 (0.645, r"Each tree level is a channel $T_\ell$; its $\pi$-weighted operator has", DIM, 9),
 (0.615, r"second singular value $\lambda_2$. Crossing a level multiplies the usable", DIM, 9),
 (0.585, r"squared correlation by the SDPI constant  $\eta_\ell = \lambda_2(T_\ell)^2$:", DIM, 9),
 (0.525, r"$\max_f \mathrm{Corr}^2(f(X),\,h^{(r)}) \leq \prod_{\ell \leq r} \eta_\ell  =  D(0\to r)$", FG, 12),
 (0.46, r"Information $\approx$ amplitude$^2$: the 'square factor' is inside $\eta$.", DIM, 9),
 (0.43, r"Both dependencies (on $r$ and on gap $k$) are one object: $\prod \eta$.", DIM, 9),
 (0.375, r"$\bf{3.\ Dynamics}$", FG, 11),
 (0.315, r"$\dot c_r  =  \beta\, f_r\,(1-c_r) \sum_{j<r} c_j(t)\, D(j\to r), \quad c_0 \equiv 1$", FG, 12),
 (0.25, r"$f_r$ = loss weight of the consuming class (frequency $\times\, g_r$):", DIM, 9),
 (0.22, r"supervision stays at tokens, so cost grows with depth regardless.", DIM, 9),
 (0.165, r"$\bf{Predictions}$", FG, 11),
 (0.12, r"P1: tokens-to-learn  $\tau_r \propto g_r^{-\kappa}$, $\kappa \approx 1$ (not 2) $\Rightarrow$ geometric", DIM, 9),
 (0.09, r"ladder with ratio $\approx 1/\eta$ per level.   P2: $g$-ratios $=$ SDPI products.", DIM, 9),
 (0.06, r"P3: shadows present $\Rightarrow$ parallel race ($j\!=\!0$ term); renewal only if gated.", DIM, 9),
]
for y, s, c, fs in th:
    axT.text(0.05, y, s, transform=axT.transAxes, fontsize=fs, color=c, va="top")

# ---------------- Panel A: power law ----------------
axA = fig.add_subplot(gs[0, 1]); axA.set_facecolor(PANEL)
DATA = {"pcfg2": ([0.4022, 0.1176, 0.0223], [229, 737, 1082], [0.766], [62]),
        "pcfg3": ([0.2312, 0.0495], [246, 1067], [0.591], [53]),
        "pcfg4": ([0.0310], [155], [0.249], [50]),
        "pcfg4b": ([0.2127, 0.1421, 0.0582], [190, 293, 598], [0.526], [49])}
for k, (g, t, gc, tc) in DATA.items():
    axA.loglog(g, t, "o", color=ACC[k], label=k, ms=7)
    axA.loglog(gc, tc, "o", color=ACC[k], ms=7, mfc="none")
xs = np.logspace(-1.75, -0.1, 20)
axA.loglog(xs, 42 * xs**-1.0, "--", color="#666", lw=1.2)
axA.text(0.012, 4000, r"$\tau \propto g^{-1}$", color="#999", fontsize=10)
axA.loglog(xs, 10 * xs**-2.0, ":", color="#666", lw=1.2)
axA.text(0.10, 3200, r"$g^{-2}$ (excluded)", color="#777", fontsize=8, rotation=-18)
import matplotlib.ticker as mtick
axA.set_xlim(0.015, 1.0)
axA.xaxis.set_major_locator(mtick.LogLocator(base=10, subs=(1.0,)))
axA.xaxis.set_minor_locator(mtick.LogLocator(base=10, subs=(2, 3, 5, 7)))
axA.xaxis.set_minor_formatter(mtick.NullFormatter())
axA.set_xlabel(r"contextual information $g_b$ (nats/position)")
axA.set_ylabel(r"$\tau_{50}$ (steps to half coverage)")
axA.set_title("A · Tokens-to-learn vs. dilution — one power law", fontsize=10, color=FG, loc="left")
axA.legend(fontsize=7, facecolor=PANEL, labelcolor=FG, edgecolor="#30363d", loc="lower left")
axA.text(0.97, 0.95, "fits: κ = 0.87 (r = −.997), 0.52\nhollow = censored (≤ first eval)",
         transform=axA.transAxes, fontsize=7.5, color=DIM, ha="right", va="top")

# ---------------- Panel B: ODE vs measured ----------------
axB = fig.add_subplot(gs[0, 2]); axB.set_facecolor(PANEL)
G = {0: 0.5261, 1: 0.2127, 2: 0.1421, 3: 0.0582}
M = {0: 0.9384, 1: 0.9654, 2: 1.2610, 3: 1.2378}
rows = [json.loads(l) for l in open("runs/pcfg4a0.05_dense_seed0/eval.jsonl") if json.loads(l)["step"] <= 2500]
steps = [r["step"] for r in rows]
CB = ["#9ccc65", "#4fc3f7", "#e6a817", "#ef5350"]
for b in range(4):
    cov = [max(0, min(1, (M[b] - r[f"ce_b{b}"]) / G[b])) for r in rows]
    axB.plot(steps, cov, "-", color=CB[b], lw=2, label=f"b{b} measured")
eta_fit, dt = 0.0192, 5.0
c = {b: 0.0 for b in range(4)}
ts_m, cs_m = [], {b: [] for b in range(4)}
t = 0.0
while t <= 2500:
    ts_m.append(t)
    for b in range(4):
        cs_m[b].append(c[b])
        c[b] = min(1.0, c[b] + dt * eta_fit * G[b] * (1 - c[b]))
    t += dt
for b in range(4):
    axB.plot(ts_m, cs_m[b], "--", color=CB[b], lw=1.2, alpha=0.85)
axB.set_xlabel("training step"); axB.set_ylabel(r"coverage $c_b(t)$")
axB.set_title("B · Parallel-race ODE (dashed) vs. training (solid), pcfg4b", fontsize=10, color=FG, loc="left")
axB.legend(fontsize=7, facecolor=PANEL, labelcolor=FG, edgecolor="#30363d", loc="lower right")
axB.text(0.33, 0.06, "one shared β; rates = β·$g_b$",
         transform=axB.transAxes, fontsize=7.5, color=DIM)

# ---------------- Panel C: eta ladder ----------------
axC = fig.add_subplot(gs[1, 1]); axC.set_facecolor(PANEL)
lab = [r"$g_0/g_1$", r"$g_1/g_2$", r"$g_2/g_3$"]
pred, meas = [1.73, 1.34, 3.28], [2.47, 1.50, 2.44]
x = np.arange(3)
axC.bar(x - 0.18, pred, 0.36, color="#4fc3f7", label=r"theory: $\prod \eta$ along path")
axC.bar(x + 0.18, meas, 0.36, color="#9ccc65", label="measured (BP oracles)")
axC.set_xticks(x); axC.set_xticklabels(lab, fontsize=10)
axC.set_ylabel("information ratio per added level")
axC.set_title("C · SDPI ladder predicts the dilution ratios (pcfg4b)", fontsize=10, color=FG, loc="left")
axC.legend(fontsize=7.5, facecolor=PANEL, labelcolor=FG, edgecolor="#30363d")
axC.text(0.03, 0.95, "single-path bound; deviations run in the\nmulti-leaf-evidence direction (b0 ↑, b3 ↓)",
         transform=axC.transAxes, fontsize=7.5, color=DIM, va="top")

# ---------------- Panel D: derivative test ----------------
axD = fig.add_subplot(gs[1, 2]); axD.set_facecolor(PANEL)
dc = json.load(open("deriv_curves.json"))
ts, pro = dc["steps"], {int(k): v for k, v in dc["probe"].items()}
def sm(v): return [sum(v[max(0, i-1):i+2]) / len(v[max(0, i-1):i+2]) for i in range(len(v))]
for r, col in ((2, "#4fc3f7"), (3, "#e6a817"), (4, "#ef5350")):
    cs, cp = sm(pro[r]), sm(pro[r - 1])
    X, Y = [], []
    for i in range(1, len(cs) - 1):
        if 0.05 < cs[i] < 0.92:
            X.append(cp[i]); Y.append(((cs[i+1] - cs[i-1]) / (ts[i+1] - ts[i-1])) / max(1 - cs[i], 0.08))
    axD.plot(X, Y, "o", color=col, ms=5, alpha=0.8, label=f"order {r} rate vs $c_{{{r-1}}}$")
axD.set_xlabel(r"lower-order consolidation $c_{r-1}(t)$")
axD.set_ylabel(r"normalized rate $\dot c_r/(1-c_r)$")
axD.set_title("D · No bootstrap route: rate does not rise with $c_{r-1}$", fontsize=10, color=FG, loc="left")
axD.legend(fontsize=7.5, facecolor=PANEL, labelcolor=FG, edgecolor="#30363d")
axD.text(0.03, 0.06, "probe-based latent coverage, ckpts every 100 steps\ncorr = −0.65…−0.75 (pcfg2 control: −0.17…+0.05)",
         transform=axD.transAxes, fontsize=7.5, color=DIM)

# ---------------- verdict strip ----------------
axV = fig.add_subplot(gs[2, :]); axV.set_facecolor("#101418"); axV.set_xticks([]); axV.set_yticks([])
for s in axV.spines.values(): s.set_color("#30363d")
axV.text(0.012, 0.62, "VERDICTS", fontsize=10, weight="bold", color="#9ccc65", va="center")
axV.text(0.075, 0.78, "κ ≈ 1 vs information (the square lives inside η = λ₂²)   ·   ladder ratios predicted by SDPI products to ~40%   ·   every net ends at the exact Bayes floor",
         fontsize=9, color=FG, va="center")
axV.text(0.075, 0.3, "learning = parallel race paced by the diluted direct signal — no bootstrap route detected (panel D); renewal dynamics reserved for gated tasks (m27 manufactured gradients)",
         fontsize=9, color=FG, va="center")
plt.savefig("poster_dilution_ladder.png", dpi=150, facecolor=BG)
print("saved poster_dilution_ladder.png")
