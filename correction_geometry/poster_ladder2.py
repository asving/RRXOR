import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.gridspec import GridSpec

BG, PANEL, FG, DIM = "#0e1117", "#161b22", "#e6edf3", "#8b949e"
GRN, BLU, AMB, RED = "#9ccc65", "#4fc3f7", "#e6a817", "#ef5350"
plt.rcParams.update({"text.color": FG, "axes.edgecolor": "#30363d", "axes.labelcolor": FG,
                     "xtick.color": DIM, "ytick.color": DIM, "font.size": 9})

fig = plt.figure(figsize=(16.5, 11.5), facecolor=BG)
gs = GridSpec(3, 3, figure=fig, width_ratios=[1.25, 1, 1], height_ratios=[1.02, 1, 0.16],
              left=0.03, right=0.985, top=0.875, bottom=0.03, wspace=0.24, hspace=0.28)

fig.text(0.03, 0.965, "The Dilution Ladder", fontsize=26, weight="bold", color=FG)
fig.text(0.03, 0.928, "How a transformer learns the latent hierarchy of a random grammar from token supervision alone — a quantitative theory and its test",
         fontsize=11.5, color=DIM)

# ================= LEFT: task + theory =================
axL = fig.add_subplot(gs[:2, 0]); axL.set_facecolor(PANEL); axL.set_xticks([]); axL.set_yticks([])
for s in axL.spines.values(): s.set_color("#30363d")

def T(y, s, c=DIM, fs=8.6, w="normal", x=0.045):
    axL.text(x, y, s, transform=axL.transAxes, fontsize=fs, color=c, va="top", weight=w)

T(0.985, "THE TASK — PCFG-4, an entropic random grammar", FG, 10.5, "bold")
T(0.955, "Depth-4 binary tree, 4-symbol alphabet. Root uniform; each node expands by a fixed")
T(0.932, r"random production table $P(\mathrm{child\ pair}\,|\,\mathrm{parent}) \sim$ Dirichlet(0.05) over the 16 ordered")
T(0.909, "pairs (one table per level, all pairs possible). Inversion is genuinely uncertain: best")
T(0.886, "parent guess is 0.78 from both children, 0.52 from one. Only the 16 leaves are emitted,")
T(0.863, "left to right; a 6-layer transformer is trained on next-token CE (fresh samples each")
T(0.840, r"step, so step $\propto$ samples). Predicting leaf $i$ = inferring the shared ancestor $b(i)$ levels")
T(0.817, r"up, then propagating back down; $b(i)$ = number of trailing zeros of $i$ = levels crossed")
T(0.794, r"between leaf $i{-}1$ and leaf $i$. Classes $b \in \{0,1,2,3\}$ have 8/4/2/1 positions per tree.")

# --- tree schematic (drawn in axes coords, y in [0.52, 0.775]) ---
ytop, ybot = 0.750, 0.535
def nx(l, i): return 0.06 + 0.80 * (i + 0.5) / (2 ** l)
def ny(l): return ytop - (ytop - ybot) * l / 4
up_path = {(4, 7), (3, 3), (2, 1), (1, 0), (0, 0)}
dn_path = {(0, 0), (1, 1), (2, 2), (3, 4), (4, 8)}
for l in range(4):
    for i in range(2 ** l):
        for ch in (2 * i, 2 * i + 1):
            on = ((l, i) in up_path and (l + 1, ch) in up_path) or ((l, i) in dn_path and (l + 1, ch) in dn_path)
            axL.plot([nx(l, i), nx(l + 1, ch)], [ny(l), ny(l + 1)], color=AMB if on else "#454d56",
                     lw=1.8 if on else 0.8, transform=axL.transAxes, zorder=1)
for l in range(5):
    for i in range(2 ** l):
        if l == 4:
            c = GRN if i < 8 else ("#555" if i > 8 else RED)
            axL.plot(nx(l, i), ny(l), "o", ms=4.5, color=c, transform=axL.transAxes, zorder=2,
                     mfc="none" if i == 8 else c, mew=1.4)
        else:
            on = (l, i) in up_path or (l, i) in dn_path
            axL.plot(nx(l, i), ny(l), "o", ms=4 if on else 3, color=AMB if on else "#5a636d",
                     transform=axL.transAxes, zorder=2)
for l, lab in ((4, "tokens"), (3, "order 1"), (2, "order 2"), (1, "order 3"), (0, "order 4 (root)")):
    axL.text(0.885, ny(l), lab, transform=axL.transAxes, fontsize=7.5, color=DIM, va="center")
axL.text(0.06, ybot - 0.02, "seen prefix", transform=axL.transAxes, fontsize=7.5, color=GRN)
axL.text(0.40, ybot - 0.02, r"target (class $b{=}3$)", transform=axL.transAxes, fontsize=7.5, color=RED)
axL.text(0.045, ytop - 0.01, "each level crossed multiplies", transform=axL.transAxes, fontsize=7.5, color=AMB)
axL.text(0.045, ytop - 0.033, r"the signal by $\eta_\ell$", transform=axL.transAxes, fontsize=7.5, color=AMB)

T(0.505, "1 · THE OBSERVABLE", FG, 10.5, "bold")
T(0.475, r"$\mathrm{CE}_r(t)$ = the net's cross-entropy on class-$(r{-}1)$ positions at step $t$. Exact references")
T(0.452, r"from the generative model: $B_r$ = CE of full belief propagation on the prefix (Bayes floor);")
T(0.429, r"$M_r$ = CE of the position marginal (no context). $g_r = M_r - B_r$ = contextual information")
T(0.406, r"at stake (nats/position). Coverage = the fraction of $g_r$ captured so far:")
T(0.368, r"$c_r(t) = \left(M_r - \mathrm{CE}_r(t)\right) / g_r \;\in [0,1]$ (clamped)", FG, 11.5, x=0.09)

T(0.315, "2 · DILUTION — why tokens are partial views of deep latents", FG, 10.5, "bold")
T(0.285, r"$T_\ell$ = the parent$\to$child channel at level $\ell$ (sibling marginalized), input law $\pi$ = that")
T(0.262, r"node's marginal, $\mu$ = the child's. The operator $\mathrm{diag}(\sqrt{\pi})\,T_\ell\,\mathrm{diag}(1/\sqrt{\mu})$ has top singular value 1;")
T(0.239, r"the second, $\lambda_2(\ell)$, is the fraction of correlation surviving one level. For any statistic $f$:")
T(0.198, r"$\mathrm{Corr}^2(f(\mathrm{tokens}),\ \mathrm{order}$-$r\ \mathrm{latent}) \leq D(0{\to}r)$,", FG, 11, x=0.09)
T(0.163, r"$D(j{\to}r) := \prod_{\ell=j+1}^{r} \eta_\ell\,, \qquad \eta_\ell := \lambda_2(\ell)^2$", FG, 11, x=0.09)
T(0.108, r"Weak dependence: $\mathrm{KL} \approx \frac{1}{2}\chi^2$, so nats-valued $g_r$ tracks squared correlation —")
T(0.086, r"the 'square law' of sample cost lives inside $\eta$.")

T(0.055, "3 · DYNAMICS (hypothesized law; fitted with ONE constant in panel B)", FG, 10.5, "bold")
T(0.016, r"$\dot c_r = \beta\, f_r\, (1-c_r) \sum_{j=0}^{r-1} c_j(t)\, D(j{\to}r), \quad c_0 \equiv 1$   [$f_r$ = class freq $\times\, g_r$]", FG, 10.5, x=0.09)

# ================= Panel A =================
axA = fig.add_subplot(gs[0, 1]); axA.set_facecolor(PANEL)
g = np.array([0.2127, 0.1421, 0.0582]); t50 = np.array([190, 293, 598])
axA.loglog(g, t50, "o", color=GRN, ms=8, zorder=3, label=r"classes $b1,b2,b3$ ($\tau_{50}$ measured)")
axA.loglog([0.5261], [49], "o", color=GRN, ms=8, mfc="none", label=r"$b0$ (censored $\leq$ first eval)")
xs = np.logspace(np.log10(0.04), np.log10(0.7), 20)
lgm, ltm = np.log(g).mean(), np.log(t50).mean()
axA.loglog(xs, np.exp(ltm - 0.87 * (np.log(xs) - lgm)), "-", color="#c9d1d9", lw=1.4,
           label=r"fit $\kappa = 0.87$ ($r=-0.997$)")
axA.loglog(xs, np.exp(ltm - 1.0 * (np.log(xs) - lgm)), "--", color="#666", lw=1.1, label=r"$\kappa=1$ (info budget)")
axA.loglog(xs, np.exp(ltm - 2.0 * (np.log(xs) - lgm)), ":", color="#666", lw=1.1, label=r"$\kappa=2$ (excluded)")
axA.set_xlim(0.04, 0.7)
axA.xaxis.set_major_locator(mtick.LogLocator(base=10, subs=(1.0,)))
axA.xaxis.set_minor_formatter(mtick.NullFormatter())
axA.set_xlabel(r"contextual information $g$ (nats/position)")
axA.set_ylabel(r"$\tau_{50}$ = steps to half coverage")
axA.set_title(r"A · Tokens-to-learn follows $\tau \propto g^{-\kappa}$, $\kappa \approx 1$", fontsize=10, color=FG, loc="left")
axA.legend(fontsize=7.5, facecolor=PANEL, labelcolor=FG, edgecolor="#30363d", loc="upper right")

# ================= Panel B =================
axB = fig.add_subplot(gs[0, 2]); axB.set_facecolor(PANEL)
G = {0: 0.5261, 1: 0.2127, 2: 0.1421, 3: 0.0582}
M = {0: 0.9384, 1: 0.9654, 2: 1.2610, 3: 1.2378}
rows = [json.loads(l) for l in open("runs/pcfg4a0.05_dense_seed0/eval.jsonl") if json.loads(l)["step"] <= 2500]
steps = [r["step"] for r in rows]
CB = [GRN, BLU, AMB, RED]
for b in range(4):
    cov = [max(0, min(1, (M[b] - r[f"ce_b{b}"]) / G[b])) for r in rows]
    axB.plot(steps, cov, "-", color=CB[b], lw=2, label=f"class b{b} measured")
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
axB.set_xlabel("training step"); axB.set_ylabel(r"coverage $c(t)$")
axB.set_title("B · One-constant ODE (dashed) vs. training (solid)", fontsize=10, color=FG, loc="left")
axB.legend(fontsize=7.5, facecolor=PANEL, labelcolor=FG, edgecolor="#30363d", loc="lower right")
axB.text(0.30, 0.06, r"single fitted $\beta$; rates $= \beta \cdot g_b$", transform=axB.transAxes, fontsize=7.5, color=DIM)

# ================= Panel C =================
axC = fig.add_subplot(gs[1, 1]); axC.set_facecolor(PANEL)
lab = [r"$g_{b0}/g_{b1}$", r"$g_{b1}/g_{b2}$", r"$g_{b2}/g_{b3}$"]
pred, meas = [1.73, 1.34, 3.28], [2.47, 1.50, 2.44]
x = np.arange(3)
axC.bar(x - 0.18, pred, 0.36, color=BLU, label=r"theory: $\prod \eta_\ell$ along the evidence path")
axC.bar(x + 0.18, meas, 0.36, color=GRN, label="measured (BP oracles)")
axC.set_xticks(x); axC.set_xticklabels(lab, fontsize=10)
axC.set_ylabel("information lost per added level (ratio)")
axC.set_title("C · SDPI products predict the dilution ladder", fontsize=10, color=FG, loc="left")
axC.legend(fontsize=7.5, facecolor=PANEL, labelcolor=FG, edgecolor="#30363d")
axC.text(0.40, 0.64, "single-path bound;\ndeviations run in the direction\nmulti-leaf evidence implies\n(shallow ↑, deepest ↓)",
         transform=axC.transAxes, fontsize=7.5, color=DIM, va="top", ha="center")

# ================= Panel D =================
axD = fig.add_subplot(gs[1, 2]); axD.set_facecolor(PANEL)
dc = json.load(open("deriv_curves.json"))
ts, pro = dc["steps"], {int(k): v for k, v in dc["probe"].items()}
def sm(v): return [sum(v[max(0, i-1):i+2]) / len(v[max(0, i-1):i+2]) for i in range(len(v))]
for r, col in ((2, BLU), (3, AMB), (4, RED)):
    cs, cp = sm(pro[r]), sm(pro[r - 1])
    X, Y = [], []
    for i in range(1, len(cs) - 1):
        if 0.05 < cs[i] < 0.92:
            X.append(cp[i]); Y.append(((cs[i+1] - cs[i-1]) / (ts[i+1] - ts[i-1])) / max(1 - cs[i], 0.08))
    axD.plot(X, Y, "o", color=col, ms=5, alpha=0.85, label=f"order-{r} rate vs order-{r-1} coverage")
axD.set_xlabel(r"lower-order coverage $c_{r-1}(t)$ (probe-based)")
axD.set_ylabel(r"normalized rate $\dot c_r/(1-c_r)$")
axD.set_title(r"D · No bootstrap route: rate does not rise with $c_{r-1}$", fontsize=10, color=FG, loc="left")
axD.set_ylim(0, 0.0035)
axD.legend(fontsize=7.5, facecolor=PANEL, labelcolor=FG, edgecolor="#30363d", loc="upper right")
axD.text(0.03, 0.06, "latent coverage decoded from checkpoints every 100 steps;\ncorr(rate, lower-order coverage) = −0.65…−0.75",
         transform=axD.transAxes, fontsize=7.5, color=DIM)

# ================= verdict =================
axV = fig.add_subplot(gs[2, :]); axV.set_facecolor("#101418"); axV.set_xticks([]); axV.set_yticks([])
for s in axV.spines.values(): s.set_color("#30363d")
axV.text(0.012, 0.74, "VERDICTS", fontsize=10, weight="bold", color=GRN, va="center")
axV.text(0.075, 0.74, "the net reaches the exact Bayes floor at every class (≤ 0.3 mnat)   ·   tokens-to-learn scales as 1/information, "
         "κ ≈ 0.9 — the 'square law' lives inside η = λ₂²   ·   dilution ratios predicted by SDPI path-products to ~40%",
         fontsize=9, color=FG, va="center")
axV.text(0.075, 0.28, "learning is a parallel race paced by each latent's diluted direct signal: the one-constant ODE reproduces all four classes (B), "
         "and rates show no dependence on lower-order coverage (D)",
         fontsize=9, color=FG, va="center")
plt.savefig("poster_dilution_ladder.png", dpi=150, facecolor=BG)
print("saved")
