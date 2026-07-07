"""Task discovery on original RRXOR-3 runs (seed0/1/2, xp00_seed0).

Same pipeline as taskdisc.py; scoring against BOTH taxonomies:
  hand types {type0, type1, type2}  and  luck taxonomy {rnd-match, rnd-mismatch, det}.
"""
import glob, json, os, sys
import torch
import torch.nn.functional as F
import numpy as np
from train import GPT

name = sys.argv[1] if len(sys.argv) > 1 else "seed0"
dev = "cuda"
base = os.path.dirname(os.path.abspath(__file__))
run = os.path.join(base, "runs", name)
cfgp = os.path.join(run, "config.json")
cfg = json.load(open(cfgp)) if os.path.exists(cfgp) else {"L": 96, "d": 128, "nl": 4,
                                                          "aux": False}
use_aux = cfg.get("aux", False)
ev = torch.load(os.path.join(run, "evalset.pt"))
seq, types = ev["eseq"][:512].to(dev), ev["etypes"][:512].to(dev)
eaux = ev.get("eaux")
eaux = eaux[:512].to(dev) if (use_aux and eaux is not None) else None
B, L = seq.shape
TPOS = torch.arange(33, L - 1, device=dev)
model = GPT(L=cfg["L"], d=cfg["d"], nl=cfg["nl"], aux=use_aux).to(dev)
ckpts = sorted(glob.glob(os.path.join(run, "ckpts", "*.pt")))
steps = np.array([int(os.path.basename(c)[4:10]) for c in ckpts])

CE = torch.zeros(len(ckpts), B, len(TPOS))
for i, cp in enumerate(ckpts):
    model.load_state_dict(torch.load(cp)); model.eval()
    with torch.no_grad():
        logits = model(seq[:, :-1], None if eaux is None else eaux[:, :-1])
        logp = F.log_softmax(logits, -1)
        ce = -logp.gather(-1, seq[:, 1:, None]).squeeze(-1)
    CE[i] = ce[:, TPOS - 1].cpu()

X = CE.permute(1, 2, 0).reshape(-1, len(ckpts))
X = F.avg_pool1d(F.pad(X[:, None], (1, 1), mode="replicate"), 3, stride=1).squeeze(1)
match = torch.zeros_like(seq)
match[:, 2:] = (seq[:, 2:] == (seq[:, 1:-1] ^ seq[:, :-2])).long()
ty = types[:, TPOS].reshape(-1).cpu()                      # hand labels 0/1/2
mt = match[:, TPOS].reshape(-1).cpu()
luck = torch.where(ty == 2, torch.tensor(2), mt)           # 2=det, 1=rnd-match, 0=rnd-mis
g = torch.Generator().manual_seed(0)
sub = torch.randperm(X.shape[0], generator=g)[:8000]
Xs, yt, yl = X[sub], ty[sub], luck[sub]

def kmeans(X, k, iters=60, restarts=10, seed=0):
    gg = torch.Generator().manual_seed(seed)
    best = (None, 1e18)
    for r in range(restarts):
        C = X[torch.randperm(X.shape[0], generator=gg)[:k]].clone()
        for _ in range(iters):
            d = torch.cdist(X, C)
            a = d.argmin(1)
            C = torch.stack([X[a == j].mean(0) if (a == j).any() else C[j]
                             for j in range(k)])
        inert = (X - C[a]).pow(2).sum().item()
        if inert < best[1]:
            best = (a.clone(), inert)
    return best[0]

def silhouette(X, a):
    D = torch.cdist(X, X)
    s = torch.zeros(len(a))
    for i in range(len(a)):
        own = D[i][a == a[i]]
        A = own.sum() / max(len(own) - 1, 1)
        Bv = min(D[i][a == j].mean() for j in a.unique() if j != a[i])
        s[i] = (Bv - A) / max(A, Bv)
    return s.mean().item()

def ari(a, b):
    a, b = np.asarray(a), np.asarray(b)
    ct = np.zeros((int(a.max()) + 1, int(b.max()) + 1))
    for i, j in zip(a, b):
        ct[i, j] += 1
    c2 = lambda x: x * (x - 1) / 2
    sij = c2(ct).sum(); si = c2(ct.sum(1)).sum(); sj = c2(ct.sum(0)).sum()
    n = c2(len(a)); exp = si * sj / n
    return (sij - exp) / ((si + sj) / 2 - exp)

sil_idx = torch.randperm(len(Xs), generator=g)[:2000]
sils = {}
for k in range(2, 7):
    ak = kmeans(Xs, k)
    sils[k] = silhouette(Xs[sil_idx], ak[sil_idx])
kstar = max(sils, key=sils.get)
assign = kmeans(Xs, kstar)
print(f"{name}: silhouettes " + " ".join(f"k{k}:{v:.3f}" for k, v in sils.items()))
print(f"k* = {kstar}  ARI vs hand-types = {ari(assign.numpy(), yt.numpy()):.3f}   "
      f"ARI vs luck-taxonomy = {ari(assign.numpy(), yl.numpy()):.3f}")
lnames = ["rnd-mis", "rnd-match", "det"]
for j in range(kstar):
    m = assign == j
    counts = [(yl[m] == v).sum().item() for v in range(3)]
    tcounts = [(yt[m] == v).sum().item() for v in range(3)]
    maj = int(np.argmax(counts))
    print(f"  cluster {j}: n={m.sum().item():5d}  luck-majority={lnames[maj]:9s} "
          f"purity={counts[maj]/max(m.sum().item(),1):.3f}  luck={counts} types={tcounts}")

# events
CM = torch.stack([Xs[assign == j].mean(0) for j in range(kstar)]).numpy()
events = []
for j in range(kstar):
    dl = np.diff(CM[j]) / np.maximum(np.diff(steps), 1) * 1000
    for sign in (-1, 1):
        reg = dl * sign > 0.03
        i = 0
        while i < len(reg):
            if reg[i]:
                j2 = i
                while j2 + 1 < len(reg) and reg[j2 + 1]:
                    j2 += 1
                ext = i + int(np.argmax(np.abs(dl[i:j2 + 1])))
                mag = CM[j][j2 + 1] - CM[j][i]
                if abs(mag) > 0.04:
                    events.append({"c": j, "sign": "drop" if sign < 0 else "RISE",
                                   "step": int(steps[ext]), "mag": round(float(mag), 2)})
                i = j2 + 1
            else:
                i += 1
events.sort(key=lambda e: e["step"])
groups = []
for e in events:
    if groups and e["step"] - groups[-1][-1]["step"] <= 300:
        groups[-1].append(e)
    else:
        groups.append([e])
print("  events: " + "  |  ".join(
    f"E@{grp[0]['step']}: " + " ".join(f"c{e['c']}:{e['sign']}{e['mag']:+.2f}"
                                       for e in grp) for grp in groups))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7.5, 4.2))
for j in range(kstar):
    ax.plot(steps, CM[j], label=f"c{j} (n={(assign == j).sum()})")
for y, lab in [(0.6931, "ln2"), (0.4055, "-ln(2/3)"), (1.0986, "-ln(1/3)")]:
    ax.axhline(y, ls=":", color="gray", alpha=0.5)
    ax.text(steps[-1], y, lab, fontsize=6, color="gray")
ax.set_xscale("log"); ax.legend(fontsize=8); ax.set_xlabel("step"); ax.set_ylabel("CE")
ax.set_title(f"{name}: discovered task trajectories (k*={kstar})")
fig.tight_layout(); fig.savefig(os.path.join(run, "taskdisc3.png"), dpi=140)
json.dump({"sils": sils, "kstar": kstar,
           "ari_types": float(ari(assign.numpy(), yt.numpy())),
           "ari_luck": float(ari(assign.numpy(), yl.numpy()))},
          open(os.path.join(run, "taskdisc3.json"), "w"))
print("  saved", os.path.join(run, "taskdisc3.png"))
