"""P1/P2 figure: per-class cascade curves for a matryoshka run, with preregistered plateaus."""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

name = sys.argv[1] if len(sys.argv) > 1 else "m27_nl6_seed0"
base = os.path.dirname(os.path.abspath(__file__))
run = os.path.join(base, "runs", name)
rows = [json.loads(l) for l in open(os.path.join(run, "eval.jsonl"))]
steps = [r["step"] for r in rows]

fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
ax = axes[0]
for k, c in [("xnor", "tab:red"), ("flip", "tab:purple"), ("xor", "tab:blue"),
             ("rnd", "tab:orange")]:
    ax.plot(steps, [r[f"ce_{k}"] for r in rows], label=k, color=c)
stage_marks = {"xnor": [0.8979, 1.5041, 0.4055], "flip": [0.5232, 0.2513, 1.0986],
               "xor": [0.5232, 0.2513], "rnd": [0.7106]}
for k, c in [("xnor", "tab:red"), ("flip", "tab:purple"), ("xor", "tab:blue"),
             ("rnd", "tab:orange")]:
    for y in stage_marks[k]:
        ax.axhline(y, ls=":", color=c, alpha=0.35)
ax.axhline(0.6931, ls=":", color="gray", alpha=0.6)
ax.set_xscale("log"); ax.set_xlim(left=20); ax.set_xlabel("step")
ax.set_ylabel("CE (nats), target pos >= 108")
ax.set_title("Per-class cascade (dotted = preregistered stage values)")
ax.legend(fontsize=8)

ax = axes[1]
ax.plot(steps, [r["ce_avg"] for r in rows], color="black")
for y, lab in [(0.6931, "stage 0"), (0.6758, "stage 1"), (0.6386, "stage 2 (mod 3)"),
               (0.5328, "stage 3 (mod 9)"), (0.4621, "stage 4 (mod 27)")]:
    ax.axhline(y, ls=":", color="gray")
    ax.text(steps[-1], y, f" {lab}: {y}", fontsize=7, va="center")
ax.set_xscale("log"); ax.set_xlim(left=20); ax.set_xlabel("step")
ax.set_title("Average loss vs the preregistered plateau ladder")

fig.suptitle(name)
fig.tight_layout()
out = os.path.join(run, "cascade.png")
fig.savefig(out, dpi=140)
print(out)

# stage-exit times
def first(cond):
    for i in range(len(rows) - 1):
        if cond(rows[i]) and cond(rows[i + 1]):
            return rows[i]["step"]
    return None

import math
print(f"tau_ungated(ce_xor<0.60) {first(lambda r: r['ce_xor'] < 0.60)}  "
      f"tau_mod3(ce_xor<0.35) {first(lambda r: r['ce_xor'] < 0.35)}  "
      f"tau_mod9(ce_xor<0.05) {first(lambda r: r['ce_xor'] < 0.05)}  "
      f"tau_mod27(ce_xnor<0.20) {first(lambda r: r['ce_xnor'] < 0.20)}")
if not math.isnan(rows[-1].get("ce_flip", float("nan"))):
    peak_flip = max(r["ce_flip"] for r in rows if r["step"] >= 100)
    print(f"flip peak {peak_flip:.3f} (pred rise to ~1.099); "
          f"xnor peak {max(r['ce_xnor'] for r in rows if r['step'] >= 100):.3f} (pred ~1.504)")
print(f"final avg {rows[-1]['ce_avg']:.4f}")
