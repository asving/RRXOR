"""Figure: the phase-belief simplex extracted from behavior via the transfer's
rotation eigenspace — the sync tree appears in the clock plane."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

base = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(base, "eigproj.json")))
rows = d["rows"]

W = np.concatenate([np.array(r["w"]) for r in rows])
PI = np.concatenate([np.array(r["pi"]) for r in rows])
# affine map simplex -> plane, fitted on all data (for theory overlays)
Xb = np.concatenate([PI, np.ones((len(PI), 1))], 1)
M, *_ = np.linalg.lstsq(Xb, W, rcond=None)
def to_plane(pi3):
    pi3 = np.asarray(pi3, dtype=float)
    return np.concatenate([pi3, [1.0]]) @ M

# theoretical MSP points: beliefs (2^a,2^b,2^c)/Z with kills
theory = set()
for a in range(3):
    for b in range(3):
        for c in range(3):
            for mask in [(1, 1, 1), (1, 1, 0), (1, 0, 0), (0, 1, 1), (0, 1, 0),
                         (0, 0, 1), (1, 0, 1)]:
                v = np.array([2.0 ** a * mask[0], 2.0 ** b * mask[1], 2.0 ** c * mask[2]])
                if v.sum() > 0:
                    v = v / v.sum()
                    theory.add(tuple(np.round(v, 4)))
theory = np.array(sorted(theory))
# keep only theory points the data actually visits
tp_all = np.stack([to_plane(p) for p in theory])
keep = [i for i, p in enumerate(tp_all)
        if (np.linalg.norm(W - p, axis=1) < 0.08).sum() >= 5]
theory = theory[keep]

fig = plt.figure(figsize=(16.5, 6.6), facecolor="white")
gs = fig.add_gridspec(2, 5, width_ratios=[2.1, 1, 1, 1, 1.35], wspace=0.25,
                      hspace=0.3, left=0.03, right=0.985, top=0.74, bottom=0.06)

def scatter_panel(ax, w, pi, title, s=6):
    cols = np.clip(pi, 0, 1)
    ax.scatter(w[:, 0], w[:, 1], s=s, c=cols, alpha=0.8, lw=0)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, fontsize=9.5)

axA = fig.add_subplot(gs[:, 0])
scatter_panel(axA, W, PI, "", s=8)
tp = np.stack([to_plane(p) for p in theory])
axA.scatter(tp[:, 0], tp[:, 1], s=90, facecolors="none", edgecolors="#30343c",
            lw=1.1, zorder=5)
axA.set_title("A · all positions pooled — the MSP sync tree in the clock plane\n"
              "(color = exact posterior; ○ = filter-reachable beliefs)",
              fontsize=10, loc="left", pad=8)

for i, P in enumerate([4, 6, 8, 60]):
    r = next(r for r in rows if r["P"] == P)
    ax = fig.add_subplot(gs[i // 2, 1 + i % 2] if False else gs[divmod(i, 2)[1] * 0 + i // 2, 1 + i % 2])
    scatter_panel(ax, np.array(r["w"]), np.array(r["pi"]),
                  f"context length {P}" + ("  (pre-sync)" if P < 9 else
                                           ("  (synced)" if P == 60 else "")))
    ax.scatter(tp[:, 0], tp[:, 1], s=40, facecolors="none", edgecolors="#30343c",
               lw=0.7, zorder=5)

axC = fig.add_subplot(gs[:, 4])
r2 = d["r2"]
bars = [("posterior ←\nclock plane", r2["fwd"], "#3f9e58"),
        ("posterior ←\nnilpotent", r2["nil_pi"], "#b8bcc6"),
        ("candidate u ←\nnilpotent (pre-det)", r2["nil_u_predet"], "#e8933a"),
        ("candidate u ←\nclock plane", max(r2["w_u"], 0), "#b8bcc6")]
ys = np.arange(len(bars))[::-1]
for y, (lab, v, c) in zip(ys, bars):
    axC.barh(y, max(v, 0.004), color=c, height=0.55)
    axC.text(max(v, 0.004) + 0.02, y, f"{v:.4f}", va="center", fontsize=9)
axC.set_yticks(ys); axC.set_yticklabels([b[0] for b in bars], fontsize=8.5)
axC.set_xlim(0, 1.25); axC.set_xlabel("held-out R²", fontsize=9)
axC.set_title("C · the spectral factorization:\nphase lives in the rotation pair,\n"
              "pending bits in the nilpotent modes", fontsize=10, loc="left")
axC.spines[["top", "right"]].set_visible(False)

fig.suptitle("The phase-belief simplex, extracted from behavior alone: project the reconstructed "
             "machine's state onto the transfer operator's rotation eigenspace",
             fontsize=13, fontweight="bold", x=0.03, ha="left", y=0.97)
fig.text(0.03, 0.905,
         "w(h) = ⟨left-eigvec(λ=e^{2πi/3}), z(h)⟩ / ⟨left-eigvec(λ=1), z(h)⟩ for every history h — no probes, no activations, no ground truth used in the construction.\n"
         "The exact offset posterior maps onto the plane by ONE affine map at all context lengths (R² 0.999): vertices = synced, interior = the elimination filter's\n"
         "reachable mixed states (the MSP tree). Eigenvalue magnitude = memory lifetime: |λ|=1 rotation = phase (kept forever), λ≈0 = pending bits (die in a block).",
         fontsize=9.2, color="#444", va="top")
fig.savefig(os.path.join(base, "eigproj.png"), dpi=150)
print("saved eigproj.png")
