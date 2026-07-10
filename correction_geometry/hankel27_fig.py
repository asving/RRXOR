"""Figure: the root-of-unity tower in matryoshka's behavioral transfer operator."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

base = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(base, "runs", "m27_nl6_seed1", "hankel27.json")))

fig = plt.figure(figsize=(17, 8.2), facecolor="white")
gs = fig.add_gridspec(2, 5, height_ratios=[1.15, 1], hspace=0.38, wspace=0.3,
                      left=0.045, right=0.985, top=0.82, bottom=0.09)

PANELS = [("post-drop1 (1000)", "after drop 1 · the ⊕ bet\n(no clock)"),
          ("post-drop2 (2600)", "after drop 2 · mod-3 gate\n(3rd roots)"),
          ("post-drop3 (3400)", "after drop 3 · mod-9 rules\n(9th roots)"),
          ("final (16000)", "final · full pattern\n(27th roots)")]
th = np.linspace(0, 2 * np.pi, 300)

def circle_panel(ax, evals, title):
    ax.plot(np.cos(th), np.sin(th), color="#e3e3e3", lw=1)
    for k in range(27):
        a = 2 * np.pi * k / 27
        if k % 9 == 0:
            ax.plot(np.cos(a), np.sin(a), "x", color="#d63f4f", ms=10, mew=1.8, zorder=3)
        elif k % 3 == 0:
            ax.plot(np.cos(a), np.sin(a), "+", color="#e8933a", ms=9, mew=1.6, zorder=3)
        else:
            ax.plot(np.cos(a), np.sin(a), ".", color="#b9b9c4", ms=5, zorder=3)
    e = np.array([complex(x, y) for x, y in evals])
    ax.plot(e.real, e.imag, "o", color="#2f6fce", ms=5, alpha=0.85, zorder=4)
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=9.8)

for i, (key, title) in enumerate(PANELS):
    circle_panel(fig.add_subplot(gs[0, i]), d["stages"][key], title)
circle_panel(fig.add_subplot(gs[0, 4]), d["true"], "true generator\n(all 27th roots)")

ax = fig.add_subplot(gs[1, :])
st = np.array([t["step"] for t in d["traj"]], dtype=float); st[st < 25] = 25
def snapped(entry):
    hit = set()
    for x, y in entry["eigs"]:
        v = complex(x, y)
        if 0.8 < abs(v) < 1.15:
            kk = round(np.angle(v) / (2 * np.pi / 27))
            if abs(np.angle(v) - kk * 2 * np.pi / 27) < 0.25 * 2 * np.pi / 27:
                hit.add(int(kk) % 27)
    return len(hit)
cnt = [snapped(t) for t in d["traj"]]
fr = np.array([t.get("fitr2", 1.0) for t in d["traj"]])
ax.plot(st, cnt, "-", color="#2f6fce", lw=1.6, alpha=0.5, zorder=2)
al = np.clip((fr - 0.94) / 0.06, 0.25, 1.0)
for x, y, a in zip(st, cnt, al):
    ax.plot(x, y, "o", color="#2f6fce", ms=7, alpha=float(a), zorder=3)
ax.plot([], [], "o-", color="#2f6fce",
        label="27th-root classes hit (opacity = operator fit R²; faint = between machines)")
for lv, lab in [(1, "no clock"), (3, "3rd roots"), (9, "9th roots"), (27, "27th roots")]:
    ax.axhline(lv, color="#d63f4f", ls=":", lw=0.9)
    ax.text(20000, lv + 0.5, f"{lv} · {lab}", fontsize=8.4, color="#d63f4f", ha="right")
for i, (a, b) in enumerate([(560, 700), (2200, 2400), (2600, 3200), (3600, 5200)]):
    ax.axvspan(a, b, color=f"C{i}", alpha=0.10, lw=0)
    ax.text((a * b) ** 0.5, 29.3, f"drop {i+1}", fontsize=8.2, ha="center", color=f"C{i}")
ax.set_xscale("log"); ax.set_xlim(25, 21000); ax.set_ylim(0, 31)
ax.set_xlabel("training step", fontsize=10)
ax.set_ylabel("# eigenvalues on the unit circle", fontsize=10)
ax.legend(fontsize=9, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("The root-of-unity tower: matryoshka's behavioral transfer operator fills the unit circle 1 → 3 → 9 → 27 across the drops",
             fontsize=13.5, fontweight="bold", x=0.045, ha="left", y=0.97)
fig.text(0.045, 0.915,
         "Spectral (Hankel/OOM) reconstruction from the net's conditionals at each stage; markers on the circle: × = 3rd roots (mod-3 clock), + = 9th roots, · = 27th.\n"
         "Each stage's behavioral machine is a cycle whose length is its rule-pattern period — the dual of the Z27 ⊃ Z9 ⊃ Z3 quotient tower appears in the spectrum.",
         fontsize=9.4, color="#444", va="top")
fig.savefig(os.path.join(base, "hankel27_fig.png"), dpi=150)
print("saved hankel27_fig.png")
