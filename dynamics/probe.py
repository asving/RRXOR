"""Prediction 2b: linear-probe heatmaps over (layer x training step).

Targets at position t (from residual stream at position t):
  - xor feature: seq[t] ^ seq[t-1]  (the value the next token takes if it is an xor position)
  - phase:       (off + t) % 3      (= types[:, t])
"""
import glob, json, os, sys
import torch
import torch.nn.functional as F
from train import GPT

name = sys.argv[1] if len(sys.argv) > 1 else "0"
if name.isdigit():
    name = f"seed{name}"
dev = "cuda"
base = os.path.dirname(os.path.abspath(__file__))
run = os.path.join(base, "runs", name)
cfg = json.load(open(os.path.join(run, "config.json"))) if os.path.exists(
    os.path.join(run, "config.json")) else {"aux": False}
ev = torch.load(os.path.join(run, "evalset.pt"))
seq, types = ev["eseq"][:512].to(dev), ev["etypes"][:512].to(dev)
eaux = ev.get("eaux")
eaux = eaux[:512].to(dev) if (cfg["aux"] and eaux is not None) else None
POS = list(range(40, 91, 5))

ckpts = sorted(glob.glob(os.path.join(run, "ckpts", "*.pt")))

def fit_probe(X, y, ncls, gtr):
    # standardize, split by sequence, full-batch logistic regression
    n = X.shape[0]
    Xm, Xs = X.mean(0, keepdim=True), X.std(0, keepdim=True) + 1e-6
    X = (X - Xm) / Xs
    W = torch.zeros(X.shape[1], ncls, device=dev, requires_grad=True)
    b = torch.zeros(ncls, device=dev, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=0.05)
    Xtr, ytr, Xte, yte = X[:int(0.7 * n)], y[:int(0.7 * n)], X[int(0.7 * n):], y[int(0.7 * n):]
    for _ in range(300):
        loss = F.cross_entropy(Xtr @ W + b, ytr) + 1e-4 * W.pow(2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return ((Xte @ W + b).argmax(-1) == yte).float().mean().item()

results = []
model = GPT(aux=cfg["aux"]).to(dev)
for cp in ckpts:
    step = int(os.path.basename(cp)[4:10])
    model.load_state_dict(torch.load(cp)); model.eval()
    with torch.no_grad():
        _, resid = model(seq, eaux, return_resid=True)    # 5 x (B, L, d)
    # gather features/labels; order by sequence so the 70/30 split is by sequence
    xor_y = (seq[:, POS] ^ seq[:, [p - 1 for p in POS]]).reshape(-1)
    ph_y = types[:, POS].reshape(-1)
    row = {"step": step, "xor": [], "phase": []}
    for li, r in enumerate(resid):
        X = r[:, POS, :].reshape(-1, r.shape[-1]).float()
        row["xor"].append(fit_probe(X, xor_y, 2, None))
        row["phase"].append(fit_probe(X, ph_y, 3, None))
    results.append(row)
    print(f"step {step:6d}  xor  {['%.2f' % a for a in row['xor']]}  "
          f"phase {['%.2f' % a for a in row['phase']]}", flush=True)

json.dump(results, open(os.path.join(run, "probe_results.json"), "w"))

# heatmaps
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

steps = [r["step"] for r in results]
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for ax, key, vmin, base_lab in [(axes[0], "xor", 0.5, "chance 0.5"),
                                (axes[1], "phase", 1 / 3, "chance 0.33")]:
    M = np.array([r[key] for r in results]).T  # layers x ckpts
    im = ax.imshow(M, aspect="auto", origin="lower", vmin=vmin, vmax=1.0, cmap="viridis",
                   extent=[0, len(steps), -0.5, 4.5])
    ax.set_xticks(np.arange(len(steps))[:: max(1, len(steps) // 8)] + 0.5)
    ax.set_xticklabels([steps[i] for i in range(0, len(steps), max(1, len(steps) // 8))],
                       fontsize=7)
    ax.set_yticks(range(5)); ax.set_yticklabels(["embed", "L1", "L2", "L3", "L4"])
    ax.set_xlabel("checkpoint (step)"); ax.set_title(f"{key} probe acc ({base_lab})")
    fig.colorbar(im, ax=ax)
fig.suptitle(f"Linear probe accuracy over (layer x training time), {name}")
fig.tight_layout()
out = os.path.join(run, "probe_heatmap.png")
fig.savefig(out, dpi=140)
print(out)
