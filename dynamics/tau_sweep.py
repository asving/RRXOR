"""Test A analysis: tau_gate / tau_solve / excursion vs scaffold noise p (+ tau additivity)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

base = os.path.dirname(os.path.abspath(__file__))

def first_step(rows, cond):
    for i in range(len(rows) - 1):
        if cond(rows[i]) and cond(rows[i + 1]):
            return rows[i]["step"]
    return None

def metrics(name):
    rows = [json.loads(l) for l in open(os.path.join(base, "runs", name, "eval.jsonl"))]
    t0 = first_step(rows, lambda r: r["ce1"] < 0.70)
    post = [r for r in rows if t0 is not None and r["step"] >= t0]
    return {
        "name": name,
        "t_calib": t0,
        "exc_peak": max((r["ce1"] for r in post), default=None),
        "exc_onset": first_step(post, lambda r: r["ce1"] > 0.71),
        "tau_xor": first_step(rows, lambda r: r["pxor2"] >= 0.60),
        "tau_gate": first_step(rows, lambda r: r["pxor2"] - (r["pxor0"] + r["pxor1"]) / 2 >= 0.15),
        "tau_solve": first_step(rows, lambda r: r["ce2"] < 0.20),
        "final_avg": rows[-1]["ce_avg"],
    }

PS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
table = []
for name in [f"xp0{i}_seed{s}" for i in range(6) for s in (0, 1)] + \
            [f"seed{s}" for s in (0, 1, 2)]:
    if os.path.exists(os.path.join(base, "runs", name, "eval.jsonl")):
        m = metrics(name)
        table.append(m)
        print(f"{m['name']:12s} exc_onset {str(m['exc_onset']):>6s} exc_peak "
              f"{m['exc_peak']:.3f} tau_xor {str(m['tau_xor']):>6s} tau_gate "
              f"{str(m['tau_gate']):>6s} tau_solve {str(m['tau_solve']):>6s} "
              f"final {m['final_avg']:.4f}")

json.dump(table, open(os.path.join(base, "analysis_tauA.json"), "w"))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for key, ax in [("tau_gate", axes[0]), ("exc_onset", axes[1])]:
    for s, mk in [(0, "o"), (1, "s")]:
        xs, ys = [], []
        for i, p in enumerate(PS):
            m = next((t for t in table if t["name"] == f"xp0{i}_seed{s}"), None)
            if m and m[key] is not None:
                xs.append(p); ys.append(m[key])
        ax.plot(xs, ys, mk + "-", label=f"seed {s}")
    bvals = [t[key] for t in table if t["name"].startswith("seed") and t[key] is not None]
    if bvals:
        ax.axhline(np.mean(bvals), ls="--", color="gray",
                   label=f"baseline (no aux bit): {np.mean(bvals):.0f}")
    ax.set_xlabel("aux xor-bit flip probability p"); ax.set_ylabel("step")
    ax.set_title(key); ax.set_yscale("log"); ax.legend(fontsize=8)
fig.suptitle("Test A: scaffold dose-response")
fig.tight_layout(); fig.savefig(os.path.join(base, "tauA.png"), dpi=140)

# tau_bind additivity check
bg = [t["tau_gate"] for t in table if t["name"].startswith("seed") and t["tau_gate"]]
bx = [t["tau_xor"] for t in table if t["name"].startswith("seed") and t["tau_xor"]]
sg = [t["tau_gate"] for t in table if t["name"].startswith("xp00") and t["tau_gate"]]
if bg and bx and sg:
    print(f"\nadditivity: tau_gate(baseline) = {np.mean(bg):.0f}  vs  "
          f"tau_xor(baseline) + tau_gate(p=0) = {np.mean(bx):.0f} + {np.mean(sg):.0f} "
          f"= {np.mean(bx) + np.mean(sg):.0f}")
print(os.path.join(base, "tauA.png"))
