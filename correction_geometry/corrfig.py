"""Explainer figure for the correction-conditional geometry result (PREREG5/RESULTS5)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

base = os.path.dirname(os.path.abspath(__file__))
def load(r):
    d = json.load(open(os.path.join(base, "runs", r, "corrprobe.json")))
    ev = [json.loads(l) for l in open(os.path.join(base, "runs", r, "eval.jsonl"))]
    return d, ev

m27, m27ev = load("m27_nl6_seed1")
fx, fxev = load("fx0.03_seed0")
r3, r3ev = load("seed0")

def xs(steps):
    s = np.array(steps, dtype=float); s[s < 25] = 25; return s

C = {"u": "#e0a63b", "Q3": "#3f9e58", "EXC": "#7d54c9", "USE": "#d63f4f",
     "g9": "#2f8f9d", "Lam": "#2f6fce", "Q": "#3f9e58", "F": "#7d54c9",
     "loss": "#30343c"}
ERAS_M27 = [(560, 700), (2200, 2400), (2600, 3200), (3600, 5200)]
ERA_COL = ["#e0a63b", "#3f9e58", "#2f8f9d", "#7d54c9"]

fig = plt.figure(figsize=(17, 9.6), facecolor="white")
gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], hspace=0.34, wspace=0.16,
                      left=0.055, right=0.985, top=0.80, bottom=0.07)

def two_strip(cell):
    sub = cell.subgridspec(2, 1, height_ratios=[1, 2.1], hspace=0.09)
    a0, a1 = fig.add_subplot(sub[0]), fig.add_subplot(sub[1])
    a0.tick_params(labelbottom=False)
    for a in (a0, a1):
        a.set_xscale("log")
        a.spines[["top", "right"]].set_visible(False)
    return a0, a1

# ============ Panel A: matryoshka staircase + kill shot ============
aL, aD = two_strip(gs[:, 0])
st = xs([e["step"] for e in m27ev]); ce = [e["ce_avg"] for e in m27ev]
aL.plot(st, ce, color=C["loss"], lw=1.8)
RUNGS = [(0.6931, "uniform"), (0.6758, "ungated ⊕ bet"), (0.6386, "gate to det slots (mod-3)"),
         (0.5328, "learn mod-9 rules"), (0.4621, "resolve exceptions (mod-27)")]
for v, lab in RUNGS:
    aL.axhline(v, color="#b8bcc6", ls="--", lw=0.8, zorder=0)
    aL.text(16500, v + 0.004, f"{lab}  {v:.3f}", fontsize=7.2, va="bottom",
            ha="right", color="#666")
aL.set_xlim(25, 19000); aL.set_ylim(0.44, 0.72)
aL.set_ylabel("eval CE (nats)", fontsize=9)
aL.set_title("A · task WITH a ladder: geometry arrives stage-by-stage, and only what a stage needs",
             fontsize=11.5, loc="left", pad=10, fontweight="bold")
sm = xs(m27["steps"])
series = {"u": np.array(m27["traj"]["u"]) - m27["traj"]["u"][0],
          "Q3": np.array(m27["traj"]["Q3"]), "EXC": np.array(m27["traj"]["EXC"]),
          "g9": np.array(m27["traj"]["g9"]), "USE": np.array(m27["traj"]["USE"])}
for k in ("u", "Q3", "EXC", "USE"):
    aD.plot(sm, series[k], color=C[k], lw=2.4 if k == "USE" else 2.0)
aD.plot(sm, series["g9"], color=C["g9"], lw=1.1, ls="--")
for ax in (aL, aD):
    for (s0, s1), c in zip(ERAS_M27, ERA_COL):
        ax.axvspan(s0, s1, color=c, alpha=0.10, lw=0)
aD.axhline(0, color="#ccc", lw=0.8)
aD.set_xlim(25, 19000); aD.set_ylim(-0.55, 1.1)
aD.set_ylabel("decodability excess over 16-token window", fontsize=9)
aD.set_xlabel("training step", fontsize=9)
ann = [("u — the ⊕ candidate\n(stage-1 correction)", C["u"], 900, 0.87, 480, 0.99),
       ("Q3 — mod-3 phase, gates the bet\n(rise starts ON the era-2 traversal)", C["Q3"],
        330, 0.52, 2700, 0.15),
       ("EXC — which exception is next,\nflips the bet (stage 4)", C["EXC"],
        8200, 0.72, 11500, 0.33),
       ("g9 — mod-9 alignment: lives in attention\nWIRING, barely in the stream (peak .12)",
        C["g9"], 700, 0.36, 7500, 0.03),
       ("USE — refinements NO correction needs:\nbelow baseline at all 76 checkpoints",
        C["USE"], 45, -0.33, 800, -0.115)]
for txt, c, tx, ty, ax_, ay in ann:
    aD.annotate(txt, xy=(ax_, ay), xytext=(tx, ty), fontsize=8.4, color=c,
                fontweight="bold", ha="left",
                arrowprops=dict(arrowstyle="-", color=c, lw=1.0, alpha=0.6))

# ============ Panel B: leaky = continuous limit ============
bL, bD = two_strip(gs[0, 1])
st = xs([e["step"] for e in fxev]); ce = [e["ce_avg"] for e in fxev]
bL.plot(st, ce, color=C["loss"], lw=1.8)
bL.axhline(0.5516, color="#b8bcc6", ls="--", lw=0.8)
bL.text(150, 0.575, "Bayes floor .552", fontsize=7.6, color="#666", ha="left")
bL.set_xlim(25, 9000); bL.set_ylabel("eval CE", fontsize=9)
bL.set_title("B · task WITHOUT a ladder (leak ε=0.03): geometry ramps",
             fontsize=11.5, loc="left", pad=10, fontweight="bold")
sf = xs(fx["steps"])
for k, lab, xy, tip in [("Lam", "Λ log-odds", (5600, 0.02), (7000, 0.21)),
                        ("Q", "decision quotient", (3400, 0.53), (6000, 0.42)),
                        ("F", "sync fiber (lags ~1k steps,\nthen forms: +0.30)", (55, 0.47),
                         (3300, -0.15))]:
    D = np.array(fx["traj"][k])
    bD.plot(sf, D, color=C[k], lw=2.0)
    bD.annotate(lab, xy=tip, xytext=xy, fontsize=8.2, color=C[k], fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=C[k], lw=1.0, alpha=0.6))
bD.axhline(0, color="#ccc", lw=0.8)
bD.set_xlim(25, 9000); bD.set_ylim(-0.8, 0.62)
bD.set_xlabel("training step", fontsize=9)
bD.text(38, 0.20, "smooth rises, no staircase:\ndiscreteness lives in the task,\nnot in the learner",
        fontsize=8.4, color="#444", style="italic")

# ============ Panel C: consolidation lag ============
cL, cD = two_strip(gs[1, 1])
st = xs([e["step"] for e in r3ev]); ce2 = [e["ce2"] for e in r3ev]
cL.plot(st, ce2, color=C["loss"], lw=1.8)
cL.axvspan(550, 1450, color="#3f9e58", alpha=0.10, lw=0)
cL.text(880, 0.60, "behavior solved\n(det CE → 0)", fontsize=7.8, color="#3a7d4c", ha="center")
cL.set_xlim(25, 9000); cL.set_ylabel("det-slot CE", fontsize=9)
cL.set_title("C · behavior first, linear code later (RRXOR-3)",
             fontsize=11.5, loc="left", pad=10, fontweight="bold")
sr = xs(r3["steps"])
for k, lab, xy, tip in [("Q3", "offset posterior", (500, 0.50), (2900, 0.42)),
                        ("Lam", "gate log-odds Λ", (600, 0.30), (2600, 0.12))]:
    D = np.array(r3["traj"][k])
    cD.plot(sr, D, color=C[k], lw=2.0)
    cD.annotate(lab, xy=tip, xytext=xy, fontsize=8.2, color=C[k], fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=C[k], lw=1.0, alpha=0.6))
cD.axvspan(550, 1450, color="#3f9e58", alpha=0.10, lw=0)
cD.axhline(0, color="#ccc", lw=0.8); cD.axhline(0.10, color="#ccc", lw=0.6, ls=":")
cD.annotate("", xy=(2000, 0.13), xytext=(1450, 0.13),
            arrowprops=dict(arrowstyle="->", color="#d63f4f", lw=1.6))
cD.text(3300, -0.52, "consolidation lag ~1k steps:\ncode linearizes AFTER the circuit works,\nkeeps sharpening after loss converges",
        fontsize=8.2, color="#d63f4f", ha="center")
cD.set_xlim(25, 9000); cD.set_ylim(-0.78, 0.75)
cD.set_xlabel("training step", fontsize=9)

fig.suptitle("Which belief geometries get built? Exactly the ones each stage's correction needs — when it needs them",
             fontsize=15, fontweight="bold", x=0.055, ha="left", y=0.965)
fig.text(0.055, 0.925,
         "Every latent is probed at every checkpoint (held-out R², minus what a 16-token window of raw inputs already gives). Development is a filtration: each circuit forms exactly\n"
         "the statistic needed to CORRECT the previous one — never the task's full belief simplex. The 27-way phase is perfectly inferable from context throughout, yet its\n"
         "never-consumed refinements (USE, red) are never formatted.",
         fontsize=9.3, color="#444", va="top")
fig.savefig(os.path.join(base, "corrfig.png"), dpi=150)
print("saved corrfig.png")
