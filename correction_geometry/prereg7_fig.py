"""PREREG7 figure: loss placement — endpoint equal, path reshaped, machine-ness local."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def curve(path, key="ce_last"):
    pts = [(json.loads(l)["step"], json.loads(l).get(key)) for l in open(path)]
    return np.array([p for p in pts if p[1] is not None], dtype=float)

fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), facecolor="white")
fig.subplots_adjust(left=0.05, right=0.985, top=0.70, bottom=0.12, wspace=0.24)

ax = axes[0]
COLS = {"rrxor3": "#3f9e58", "m27": "#7d54c9", "mess3": "#b8860b"}
for task, b, l in [("rrxor3", "runs/seed0/celast.jsonl", "runs/rrxor3_last_seed0/eval.jsonl"),
                   ("m27", "runs/m27_nl6_seed1/celast.jsonl", "runs/m27_last_seed1/eval.jsonl"),
                   ("mess3", "runs/mess3_seed0/celast.jsonl", "runs/mess3_last_seed0/eval.jsonl")]:
    cb, cl = curve(b), curve(l)
    cb[:, 0] = np.maximum(cb[:, 0], 25); cl[:, 0] = np.maximum(cl[:, 0], 25)
    ax.plot(cb[:, 0], cb[:, 1], color=COLS[task], lw=2, label=f"{task} dense")
    ax.plot(cl[:, 0], cl[:, 1], color=COLS[task], lw=1.6, ls="--", label=f"{task} last-only")
ax.set_xscale("log"); ax.set_xlabel("training step (= sequences × 256)")
ax.set_ylabel("CE at the last position")
ax.set_title("A · same endpoint, ~3–6× more sequences:\nno qualitative barrier",
             fontsize=10.5, loc="left")
ax.legend(fontsize=7.5, ncol=2)
ax.spines[["top", "right"]].set_visible(False)

ax = axes[1]
CC = {"xor": "#3f9e58", "xnor": "#e8933a", "flip": "#d63f4f"}
for k in CC:
    cb = curve("runs/m27_nl6_seed1/celast.jsonl", f"ce_last_{k}")
    cl = curve("runs/m27_last_seed1/eval.jsonl", f"ce_last_{k}")
    cb[:, 0] = np.maximum(cb[:, 0], 25); cl[:, 0] = np.maximum(cl[:, 0], 25)
    ax.plot(cb[:, 0], cb[:, 1], color=CC[k], lw=2, label=f"{k} dense")
    ax.plot(cl[:, 0], cl[:, 1], color=CC[k], lw=1.6, ls="--", label=f"{k} last-only")
ax.set_xscale("log"); ax.set_xlabel("training step")
ax.set_ylabel("CE at the last position, per class")
ax.set_title("B · m27: dense = staircase, xnor spike to 1.36;\n"
             "last-only = muted excursion, then smooth co-descent",
             fontsize=10.5, loc="left")
ax.legend(fontsize=7.5, ncol=2)
ax.spines[["top", "right"]].set_visible(False)

ax = axes[2]
prof = json.load(open("prereg7_posprofile.json"))
for name, c in [("rrxor3_last_seed0", "#3f9e58"), ("m27_last_seed1", "#7d54c9")]:
    pl = np.array(prof[name]["last"]); pd = np.array(prof[name]["dense"])
    xs = np.arange(1, len(pl) + 1)
    ax.plot(xs, pl, color=c, lw=1.4, label=f"{name.split('_last')[0]} last-only")
    ax.plot(xs, pd, color=c, lw=1.4, ls=":", alpha=0.7, label=f"{name.split('_last')[0]} dense")
ax.axhline(np.log(2), color="#999", ls="--", lw=0.8)
ax.text(5, np.log(2) + 0.03, "ln 2", fontsize=8, color="#666")
ax.set_xlabel("context position"); ax.set_ylabel("CE at position")
ax.set_title("C · the last-only net is a machine at ONE position:\n"
             "calibrated at its target, worse-than-uniform elsewhere",
             fontsize=10.5, loc="left")
ax.legend(fontsize=7.5)
ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("Loss placement (PREREG7): last-token-only training solves even the deep tower — "
             "dense supervision buys speed and position-uniformity, not possibility",
             fontsize=13, fontweight="bold", x=0.05, ha="left", y=0.97)
fig.text(0.05, 0.895,
         "Same-seed pairs (identical init and data; only the loss mask differs). Matched on total loss terms, last-only is far MORE efficient per term (dense@6.1M terms is\n"
         "still at uniform; last-only@6.1M has solved the task) — the bootstrap-via-dense-loss prediction is refuted for tape-visible tasks. What dense loss does buy:\n"
         "~3–6× fewer sequences, calibration at every position, and the STAIRCASE itself — under last-only supervision the ladder melts into smooth co-descent.",
         fontsize=9.2, color="#444", va="top")
fig.savefig("prereg7.png", dpi=150)
print("saved prereg7.png")
