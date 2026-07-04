"""Test B analysis: which route does the final net use for phase?

OOD diagnosis: evaluate eps-trained nets on eps=0 sequences (cue removed).
Cue-route net -> phase collapses OOD (ce2 rises, pxor2 drops). Violation-route -> ce2 ~ 0.
Also: excursion peak (max ce1 post-calibration) vs eps — prediction B2 says it shrinks.
"""
import glob, json, os
import torch
import numpy as np
from train import GPT, gen_batch, evaluate

dev = "cuda"
base = os.path.dirname(os.path.abspath(__file__))
g = torch.Generator().manual_seed(999)
oseq, otypes = gen_batch(2048, 96, g, bias_eps=0.0)
oseq, otypes = oseq.to(dev), otypes.to(dev)

RUNS = [(0.0, f"seed{s}") for s in (0, 1, 2)] + \
       [(e, f"ep{t}_seed{s}") for e, t in [(0.02, "02"), (0.05, "05"), (0.1, "10"), (0.2, "20")]
        for s in (0, 1)]

model = GPT().to(dev)
res = []
for eps, name in RUNS:
    run = os.path.join(base, "runs", name)
    if not os.path.exists(os.path.join(run, "eval.jsonl")):
        continue
    cp = sorted(glob.glob(os.path.join(run, "ckpts", "*.pt")))[-1]
    model.load_state_dict(torch.load(cp)); model.eval()
    ev = torch.load(os.path.join(run, "evalset.pt"))
    ind = evaluate(model, ev["eseq"].to(dev), ev["etypes"].to(dev))
    ood = evaluate(model, oseq, otypes)
    rows = [json.loads(l) for l in open(os.path.join(run, "eval.jsonl"))]
    t0 = next((r["step"] for r in rows if r["ce1"] < 0.70), None)
    exc = max((r["ce1"] for r in rows if t0 is not None and r["step"] >= t0), default=None)
    res.append({"eps": eps, "name": name, "ce2_ind": ind["ce2"], "ce2_ood": ood["ce2"],
                "pxor2_ood": ood["pxor2"], "ce1_ood": ood["ce1"], "exc_peak": exc})
    print(f"{name:12s} eps {eps:.2f}  ce2 in-dist {ind['ce2']:.4f}  OOD(eps=0) "
          f"ce2 {ood['ce2']:.4f} pxor2 {ood['pxor2']:.3f} ce1 {ood['ce1']:.4f}  "
          f"exc_peak {exc:.3f}")

json.dump(res, open(os.path.join(base, "analysis_routeB.json"), "w"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
eps_vals = sorted(set(r["eps"] for r in res))
for ax, key, lab in [(axes[0], "ce2_ood", "ce2 on eps=0 data (cue removed)"),
                     (axes[1], "exc_peak", "type-1 excursion peak during training")]:
    for e in eps_vals:
        ys = [r[key] for r in res if r["eps"] == e and r[key] is not None]
        ax.plot([e] * len(ys), ys, "o", color="tab:blue")
        ax.plot([e], [np.mean(ys)], "_", ms=20, color="tab:red")
    ax.set_xlabel("training cue strength eps"); ax.set_title(lab)
axes[0].axhline(0.0, ls=":", color="gray"); axes[0].axhline(0.4055, ls=":", color="gray")
axes[0].text(0.15, 0.41, "ungated hedge 0.405", fontsize=7)
axes[1].axhline(np.log(2), ls=":", color="gray")
axes[1].text(0.12, np.log(2) + 0.003, "ln2 (no excursion)", fontsize=7)
fig.suptitle("Test B: route selection vs cue strength")
fig.tight_layout(); fig.savefig(os.path.join(base, "routeB.png"), dpi=140)
print(os.path.join(base, "routeB.png"))
