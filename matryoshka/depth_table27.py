"""P3/P5: final loss vs depth (truncation onto the plateau ladder) + nested-vs-flat timing."""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

base = os.path.dirname(os.path.abspath(__file__))
PLATEAUS = {"stage1 (ungated)": 0.6758, "stage2 (mod3)": 0.6386,
            "stage3 (mod9)": 0.5328, "stage4 (mod27)": 0.4621}

rowsall = []
for run in sorted(glob.glob(os.path.join(base, "runs", "m27*"))):
    name = os.path.basename(run)
    if name.endswith("_16k"):
        continue
    rows = [json.loads(l) for l in open(os.path.join(run, "eval.jsonl"))]
    tail = [r["ce_avg"] for r in rows if r["step"] >= rows[-1]["step"] - 1000]
    final = float(np.mean(tail))
    near = min(PLATEAUS.items(), key=lambda kv: abs(kv[1] - final))
    t_final = next((r["step"] for r in rows if r["ce_avg"] < 0.470), None)
    cfg = json.load(open(os.path.join(run, "config.json")))
    rowsall.append({"name": name, "nl": cfg["nl"], "flat": cfg.get("flat", False),
                    "final": final, "nearest": near[0], "dist": abs(near[1] - final),
                    "t_below_0.470": t_final})
    print(f"{name:22s} nl={cfg['nl']}  final {final:.4f}  nearest {near[0]:18s} "
          f"(|d|={abs(near[1]-final):.4f})  t(avg<0.470)={t_final}")

json.dump(rowsall, open(os.path.join(base, "analysis_depth27.json"), "w"))

fig, ax = plt.subplots(figsize=(7.5, 4.5))
for flat, mk, lab in [(False, "o", "nested"), (True, "s", "flat control")]:
    xs = [r["nl"] + (0.08 if r["name"].endswith("seed1") else -0.08)
          for r in rowsall if r["flat"] == flat]
    ys = [r["final"] for r in rowsall if r["flat"] == flat]
    ax.plot(xs, ys, mk, label=lab, ms=7, alpha=0.8)
for lab, y in PLATEAUS.items():
    ax.axhline(y, ls=":", color="gray")
    ax.text(8.2, y, f"{lab} {y}", fontsize=7, va="center")
ax.axhline(0.6931, ls=":", color="gray"); ax.text(8.2, 0.6931, "baseline 0.6931", fontsize=7)
ax.set_xlabel("n_layers"); ax.set_ylabel("final CE avg (nats)")
ax.set_title("Depth truncation onto the preregistered plateau ladder")
ax.legend()
fig.tight_layout()
out = os.path.join(base, "depth27.png")
fig.savefig(out, dpi=140)
print(out)
