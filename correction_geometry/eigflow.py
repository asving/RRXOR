"""The continuous picture: how a rung is installed — eigenvalue flow through the drops."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import viridis
from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable

base = os.path.dirname(os.path.abspath(__file__))
flow = json.load(open(os.path.join(base, "runs", "seed0", "hankel_flow.json")))
m27 = json.load(open(os.path.join(base, "runs", "m27_nl6_seed1", "hankel27.json")))

fig = plt.figure(figsize=(17, 6.4), facecolor="white")
gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1, 1.25], wspace=0.27,
                      left=0.045, right=0.985, top=0.77, bottom=0.11)

# ---- A: complex-plane flow (rrxor3) ----
ax = fig.add_subplot(gs[0])
th = np.linspace(0, 2 * np.pi, 300)
ax.plot(np.cos(th), np.sin(th), color="#e3e3e3", lw=1)
for a in (0, 2 * np.pi / 3, -2 * np.pi / 3):
    ax.plot(np.cos(a), np.sin(a), "x", color="#d63f4f", ms=12, mew=2, zorder=5)
norm = LogNorm(vmin=25, vmax=8000)
for f in flow:
    st = max(f["step"], 25)
    c = viridis(norm(st))
    e = np.array([complex(x, y) for x, y in f["eigs"]])
    ax.plot(e.real, e.imag, "o", color=c, ms=3.2, alpha=0.75, zorder=3)
path = []
for f in flow:
    e = [complex(x, y) for x, y in f["eigs"]]
    onf = [v for v in e if v.imag > 0.02 and 0.85 < abs(np.angle(v)) / (2 * np.pi / 3) < 1.15]
    if onf:
        path.append((f["step"], max(onf, key=abs)))
if path:
    pz = np.array([v for _, v in path])
    ax.plot(pz.real, pz.imag, "-", color="#30343c", lw=1.3, alpha=0.75, zorder=6)
    ax.annotate("", xy=(pz.real[-1], pz.imag[-1]), xytext=(pz.real[-2], pz.imag[-2]),
                arrowprops=dict(arrowstyle="-|>", color="#30343c", lw=1.3))
    ax.text(-0.15, 0.62, "the on-frequency pair's route\n(angle right, then magnitude)",
            fontsize=7.8, color="#30343c", ha="center")
cb = fig.colorbar(ScalarMappable(norm=norm, cmap="viridis"), ax=ax, fraction=0.045,
                  pad=0.02)
cb.set_label("training step", fontsize=8)
ax.set_xlim(-1.2, 1.25); ax.set_ylim(-1.2, 1.2); ax.set_aspect("equal")
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title("A · RRXOR-3: all transfer eigenvalues, every checkpoint\n"
             "(red × = cube roots; color = training step)", fontsize=10.5, loc="left")

# ---- B: clock magnitude & angle vs step (dense) ----
ax = fig.add_subplot(gs[1])
steps, on_m, off_pts = [], [], []
for f in flow:
    st = max(f["step"], 25)
    e = [complex(x, y) for x, y in f["eigs"]]
    cl = [v for v in e if abs(v.imag) > 0.02]
    onf = [abs(v) for v in cl if 0.85 < abs(np.angle(v)) / (2 * np.pi / 3) < 1.15]
    for v in cl:
        if not (0.85 < abs(np.angle(v)) / (2 * np.pi / 3) < 1.15):
            off_pts.append((st, abs(v)))
    steps.append(st); on_m.append(max(onf) if onf else 0.0)
steps = np.array(steps)
if off_pts:
    op = np.array(off_pts)
    ax.plot(op[:, 0], op[:, 1], "o", color="#b8bcc6", ms=2.6, alpha=0.7,
            label="off-frequency complex pairs (transient flicker)")
ax.plot(steps, on_m, "o-", color="#2f6fce", ms=3.5, lw=1.4,
        label="on-frequency pair (angle within 15% of 2π/3)")
ax.axhline(1.0, color="#d63f4f", ls=":", lw=1)
ax.axvspan(550, 2500, color="#3f9e58", alpha=0.08, lw=0)
ax.axvline(2600, color="#3a7d4c", ls="--", lw=0.9)
ax.text(2750, 0.30, "det CE\nconverged", fontsize=7.6, color="#3a7d4c")
ax.annotate("angle locks,\nthen magnitude ramps\n(mostly post-behavior)",
            xy=(3600, 0.93), xytext=(500, 0.72), fontsize=8, color="#2f6fce",
            arrowprops=dict(arrowstyle="-", color="#2f6fce", lw=0.9, alpha=0.6))
ax.set_xscale("log"); ax.set_xlabel("training step", fontsize=9)
ax.set_ylabel("|λ| of the clock pair", fontsize=9)
ax.set_ylim(0, 1.12)
ax.set_title("B · the clock's installation, step by step\n"
             "(green span = the gating drop)", fontsize=10.5, loc="left")
ax.legend(fontsize=7.6, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)

# ---- C: matryoshka per-root-class installation cohorts ----
ax = fig.add_subplot(gs[2])
LAT = 2 * np.pi / 27
def class_mags(entry):
    out = np.zeros(27)
    for x, y in entry["eigs"]:
        v = complex(x, y)
        kk = int(round(np.angle(v) / LAT)) % 27
        if abs(np.angle(v) - round(np.angle(v) / LAT) * LAT) < 0.35 * LAT:
            out[kk] = max(out[kk], abs(v))
    return out
st27 = np.array([t["step"] for t in m27["traj"]], dtype=float); st27[st27 < 25] = 25
M = np.stack([class_mags(t) for t in m27["traj"]])          # T x 27
COH = {"cube (k=9,18)": ([9, 18], "#e8933a"),
       "ninth (k=3,6,12,15,21,24)": ([3, 6, 12, 15, 21, 24], "#3f9e58"),
       "fine 27th (remaining 18)": ([k for k in range(1, 27) if k % 3], "#7d54c9")}
for lab, (ks, c) in COH.items():
    for k in ks:
        ax.plot(st27, M[:, k], color=c, lw=0.6, alpha=0.15)
    ax.plot(st27, M[:, ks].mean(1), color=c, lw=2.4, label=lab)
ax.plot(st27, M[:, 0], color="#666", lw=1.2, ls="--", label="k=0 (stationary)")
for i, (a, b) in enumerate([(2200, 2400), (2600, 3200), (3600, 5200)]):
    ax.axvspan(a, b, color=f"C{i+1}", alpha=0.10, lw=0)
    ax.text((a * b) ** 0.5, [0.03, 0.08, 0.03][i], f"drop {i+2}", fontsize=8, ha="center", color=f"C{i+1}")
ax.axhline(1.0, color="#d63f4f", ls=":", lw=0.9)
ax.set_xscale("log"); ax.set_xlim(700, 19000); ax.set_ylim(0, 1.3)
ax.set_xlabel("training step", fontsize=9)
ax.set_ylabel("max |λ| in each 27th-root class", fontsize=9)
ax.set_title("C · matryoshka: each root cohort is installed by its own drop",
              fontsize=10.5, loc="left")
ax.legend(fontsize=7.8, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("How a rung is installed: the eigenvalue flow through the loss drops",
             fontsize=13.5, fontweight="bold", x=0.045, ha="left", y=0.97)
fig.text(0.045, 0.90,
         "Left/middle: RRXOR-3 at every checkpoint (25-step resolution through the transition) — the route the clock pair takes to e^{±2πi/3}. "
         "Right: matryoshka's 27 root classes in three cohorts;\neach cohort's magnitude rises during its own drop (cube → ninth → fine), the spectral form of the manufactured-gradient cascade.",
         fontsize=9.3, color="#444", va="top")
fig.savefig(os.path.join(base, "eigflow.png"), dpi=150)
print("saved eigflow.png")
