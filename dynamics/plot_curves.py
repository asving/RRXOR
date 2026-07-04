"""Prediction 1 figure: per-type loss curves with preregistered reference values."""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

seed = sys.argv[1] if len(sys.argv) > 1 else "0"
run = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", f"seed{seed}")
rows = [json.loads(l) for l in open(os.path.join(run, "eval.jsonl"))]
steps = [r["step"] for r in rows]

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

ax = axes[0]
ax.plot(steps, [r["ce2"] for r in rows], label="xor position (type 2)", color="tab:blue")
ax.plot(steps, [r["ce0"] for r in rows], label="random pos (type 0)", color="tab:orange")
ax.plot(steps, [r["ce1"] for r in rows], label="random pos (type 1)", color="tab:red")
ax.plot(steps, [r["ce_avg"] for r in rows], label="average", color="black", lw=1, alpha=0.5)
for y, lab, c in [(0.6931, "ln2 baseline/final random", "gray"),
                  (0.7520, "stage-1 random (pred)", "tab:red"),
                  (0.4055, "stage-1 xor (pred)", "tab:blue"),
                  (0.4621, "final avg (pred)", "black")]:
    ax.axhline(y, ls=":", color=c, alpha=0.6)
    ax.text(steps[-1], y, f" {lab}", fontsize=7, va="center", color=c)
ax.set_xscale("log"); ax.set_xlabel("step"); ax.set_ylabel("CE (nats), target pos >= 33")
ax.set_title("Per-position-type loss"); ax.legend(fontsize=8); ax.set_xlim(left=10)

ax = axes[1]
for ty, c in [(2, "tab:blue"), (0, "tab:orange"), (1, "tab:red")]:
    ax.plot(steps, [r[f"pxor{ty}"] for r in rows], label=f"type {ty}", color=c)
for y in (1.0, 2 / 3, 0.5):
    ax.axhline(y, ls=":", color="gray", alpha=0.6)
ax.set_xscale("log"); ax.set_xlabel("step"); ax.set_ylabel("p(xor of prev two)")
ax.set_title("Probability on the xor value, by target type"); ax.legend(fontsize=8)
ax.set_xlim(left=10)

fig.suptitle(f"RRXOR seed {seed}")
fig.tight_layout()
out = os.path.join(run, "curves.png")
fig.savefig(out, dpi=140)
print(out)

# numeric summary: max excursion of random-type loss above ln2
import math
ln2 = math.log(2)
for ty in (0, 1):
    peak = max(rows, key=lambda r: r[f"ce{ty}"] if r["step"] >= 100 else -1)
    print(f"type{ty}: peak ce {peak['ce'+str(ty)]:.4f} at step {peak['step']}  "
          f"(excursion above ln2: {peak['ce'+str(ty)]-ln2:+.4f}); final {rows[-1]['ce'+str(ty)]:.4f}")
print(f"type2: final {rows[-1]['ce2']:.4f}; avg final {rows[-1]['ce_avg']:.4f} (pred 0.4621)")
