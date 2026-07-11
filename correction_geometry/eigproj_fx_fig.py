"""Figure: the fractal belief attractor of leaky RRXOR, extracted from behavior."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

base = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(base, "eigproj_fx.json")))
VERT = np.array([[0, 1], [-np.sqrt(3) / 2, -0.5], [np.sqrt(3) / 2, -0.5]])

fig = plt.figure(figsize=(15.5, 9.2), facecolor="white")
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.9], wspace=0.18, hspace=0.24,
                      left=0.04, right=0.98, top=0.82, bottom=0.05)

for row, run in enumerate(["fx0.03_seed0", "fx0.2_seed0"]):
    r = d[run]
    eps = r["eps"]
    W = np.array(r["W"]); Q = np.array(r["Q"])
    lam = [complex(a, b) for a, b in r["lam"]]
    clock = next(v for v in lam if v.imag > 0.02)
    Xtrue = Q @ VERT
    Xb = np.concatenate([W, np.ones((len(W), 1))], 1)
    M, *_ = np.linalg.lstsq(Xb, Xtrue, rcond=None)
    Xext = Xb @ M
    cols = np.clip(Q, 0, 1)
    for col, X, title in [(0, Xtrue, "exact leaky filter (ground truth)"),
                          (1, Xext, "extracted from behavior (clock plane, affine-aligned)")]:
        ax = fig.add_subplot(gs[row, col])
        tri = np.vstack([VERT, VERT[:1]])
        ax.plot(tri[:, 0], tri[:, 1], color="#ccc", lw=1)
        ax.scatter(X[:, 0], X[:, 1], s=3.5, c=cols, alpha=0.7, lw=0)
        ax.plot(0, 0, "+", color="#888", ms=8)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(-1.0, 1.0); ax.set_ylim(-0.75, 1.1)
        for sp in ax.spines.values():
            sp.set_visible(False)
        if col == 0:
            ax.text(-0.98, 1.02, f"ε = {eps}", fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=10)
    ax = fig.add_subplot(gs[row, 2])
    r2 = r["r2"]
    bars = [("class posterior ←\nclock plane", r2["fwd"], "#3f9e58"),
            ("class posterior ←\nnilpotent", max(r2["nil_q"], 0), "#b8bcc6"),
            ("pending values ←\nnilpotent", r2["nil_p"], "#e8933a"),
            ("pending values ←\nclock plane", max(r2["w_p"], 0), "#b8bcc6")]
    ys = np.arange(len(bars))[::-1]
    for y, (lab, v, c) in zip(ys, bars):
        ax.barh(y, max(v, 0.005), color=c, height=0.55)
        ax.text(max(v, 0.005) + 0.02, y, f"{v:.4f}", va="center", fontsize=9)
    ax.set_yticks(ys); ax.set_yticklabels([b[0] for b in bars], fontsize=8.5)
    ax.set_xlim(0, 1.3)
    ax.set_title(f"clock eigenvalue {abs(clock):.3f}∠{abs(np.angle(clock)):.3f}\n"
                 f"(1−ε = {1-eps:.3f}; net slightly lossier than Bayes)", fontsize=9.5,
                 loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    if row == 1:
        ax.set_xlabel("held-out R²", fontsize=9)

fig.suptitle("Leaky RRXOR: the FRACTAL belief attractor, extracted from behavior alone\n",
             fontsize=13.5, fontweight="bold", x=0.04, ha="left", y=0.97)
fig.text(0.04, 0.925,
         "Same construction as the clean case — w(h) = clock-eigenspace coordinate of the reconstructed machine's state, one affine map to the class-posterior simplex —\n"
         "but under leak the clock sits INSIDE the unit circle (|λ| ≈ 1−ε), so beliefs never reach the vertices: the reachable set is the leaky filter's fractal attractor,\n"
         "contracted toward uniform. Left: exact filter. Middle: the net's behavior, projected. Right: the factorization stays exact (phase in the damped rotation pair,\n"
         "pending det/parity values in the nilpotent modes). More leak ⇒ smaller clock radius ⇒ the attractor hugs the center: forgetting = spectral contraction.",
         fontsize=9.2, color="#444", va="top")
fig.savefig(os.path.join(base, "eigproj_fx.png"), dpi=150)
print("saved eigproj_fx.png")
