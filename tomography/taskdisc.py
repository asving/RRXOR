"""PREREG4 steps 1,2,4: unsupervised task discovery from per-instance loss trajectories.

No labels are used until scoring. Outputs taskdisc.json + taskdisc.png in the run dir.
"""
import glob, json, math, os, sys
import torch
import torch.nn.functional as F
import numpy as np
from train import GPT
from train27 import class_masks

name = sys.argv[1] if len(sys.argv) > 1 else "m27_nl6_seed1"
dev = "cuda"
base = os.path.dirname(os.path.abspath(__file__))
run = os.path.join(base, "runs", name)
cfg = json.load(open(os.path.join(run, "config.json")))
ev = torch.load(os.path.join(run, "evalset.pt"))
seq, cls = ev["eseq"][:512].to(dev), ev["ecls"][:512].to(dev)
B, L = seq.shape
TPOS = torch.arange(108, 191, device=dev)                 # target positions
model = GPT(L=cfg["L"], d=cfg["d"], nl=cfg["nl"]).to(dev)
ckpts = sorted(glob.glob(os.path.join(run, "ckpts", "*.pt")))
steps = np.array([int(os.path.basename(c)[4:10]) for c in ckpts])

# ---- pass 1: per-instance CE trajectories ----
CE = torch.zeros(len(ckpts), B, len(TPOS))
for i, cp in enumerate(ckpts):
    model.load_state_dict(torch.load(cp)); model.eval()
    with torch.no_grad():
        logits = model(seq)
        logp = F.log_softmax(logits[:, :-1], -1)
        ce = -logp.gather(-1, seq[:, 1:, None]).squeeze(-1)   # target pos j+1 at col j
    CE[i] = ce[:, TPOS - 1].cpu()
    if i % 20 == 0:
        print(f"ckpt {i}/{len(ckpts)}", flush=True)

X = CE.permute(1, 2, 0).reshape(-1, len(ckpts))            # (N, T)
X = F.avg_pool1d(F.pad(X[:, None], (1, 1), mode="replicate"), 3, stride=1).squeeze(1)
masks = class_masks(cls, cfg.get("flat", False))
lab = torch.zeros(B, L, dtype=torch.long, device=dev)
for v, k in enumerate(("rnd", "xor", "xnor", "flip")):
    lab[masks[k]] = v
truth = lab[:, TPOS].reshape(-1).cpu()
g = torch.Generator().manual_seed(0)
sub = torch.randperm(X.shape[0], generator=g)[:8000]
Xs, ys = X[sub], truth[sub]

# ---- k-means + silhouette ----
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
    ks = a.unique()
    s = torch.zeros(len(a))
    for i in range(len(a)):
        own = D[i][a == a[i]]; A = (own.sum() - 0) / max(len(own) - 1, 1)
        Bv = min(D[i][a == j].mean() for j in ks if j != a[i])
        s[i] = (Bv - A) / max(A, Bv)
    return s.mean().item()

sil_idx = torch.randperm(len(Xs), generator=g)[:2000]
sils = {}
for k in range(2, 9):
    ak = kmeans(Xs, k)
    sils[k] = silhouette(Xs[sil_idx], ak[sil_idx])
    print(f"k={k}: silhouette {sils[k]:.3f}", flush=True)
if len(sys.argv) > 2:
    kstar = int(sys.argv[2])
    print(f"[POST-HOC] forcing k = {kstar} (preregistered rule chose {max(sils, key=sils.get)})")
else:
    kstar = max(sils, key=sils.get)
assign = kmeans(Xs, kstar)
# secondary: recursive split — silhouette WITHIN each cluster
for j in assign.unique().tolist():
    Xj = Xs[assign == j]
    if len(Xj) < 400:
        continue
    subj = torch.randperm(len(Xj), generator=g)[:1500]
    ss = {kk: silhouette(Xj[subj], kmeans(Xj, kk)[subj]) for kk in (2, 3)}
    print(f"  within-cluster {j} (n={len(Xj)}): sub-silhouette k=2: {ss[2]:.3f}  k=3: {ss[3]:.3f}")

def ari(a, b):
    a, b = np.asarray(a), np.asarray(b)
    ct = np.zeros((a.max() + 1, b.max() + 1))
    for i, j in zip(a, b):
        ct[i, j] += 1
    c2 = lambda x: x * (x - 1) / 2
    sij = c2(ct).sum(); si = c2(ct.sum(1)).sum(); sj = c2(ct.sum(0)).sum()
    n = c2(len(a)); exp = si * sj / n
    return (sij - exp) / ((si + sj) / 2 - exp)

A = ari(assign.numpy(), ys.numpy())
print(f"\nk* = {kstar}  ARI vs truth = {A:.3f}")
names = ["rnd", "xor", "xnor", "flip"]
for j in range(kstar):
    m = assign == j
    counts = [(ys[m] == v).sum().item() for v in range(4)]
    maj = int(np.argmax(counts))
    print(f"cluster {j}: n={m.sum().item():5d}  majority={names[maj]:4s} "
          f"purity={counts[maj]/max(m.sum().item(),1):.3f}  counts={counts}")

# ---- events on cluster means ----
CM = torch.stack([Xs[assign == j].mean(0) for j in range(kstar)]).numpy()
events = []
for j in range(kstar):
    dl = np.diff(CM[j]) / np.maximum(np.diff(steps), 1) * 1000
    for sign, thr in ((-1, -0.03), (+1, 0.03)):
        reg = dl * sign > abs(thr) * 1  # boolean
        i = 0
        while i < len(reg):
            if reg[i]:
                j2 = i
                while j2 + 1 < len(reg) and reg[j2 + 1]:
                    j2 += 1
                ext = i + int(np.argmax(np.abs(dl[i:j2 + 1])))
                mag = (CM[j][j2 + 1] - CM[j][i])
                if abs(mag) > 0.04:
                    events.append({"cluster": j, "sign": "drop" if sign < 0 else "RISE",
                                   "step": int(steps[ext]), "mag": round(float(mag), 3)})
                i = j2 + 1
            else:
                i += 1
events.sort(key=lambda e: e["step"])
merged, groups = [], []
for e in events:
    if groups and e["step"] - groups[-1][-1]["step"] <= 500:
        groups[-1].append(e)
    else:
        groups.append([e])
print("\nevent groups (co-timed drops & rises):")
for gi, grp in enumerate(groups):
    print(f"  E{gi} @ ~{grp[0]['step']}: " + "  ".join(
        f"c{e['cluster']}:{e['sign']}{e['mag']:+.2f}" for e in grp))

# ---- matching: causal fingerprints per discovered cluster ----
model.load_state_dict(torch.load(ckpts[-1])); model.eval()
nh = 4

@torch.no_grad()
def manual_forward(head_zero=(), clean=None):
    x = model.wte(seq) + model.wpe(torch.arange(L, device=dev))[None]
    resid = [x]
    for li, blk in enumerate(model.blocks, 1):
        x_in = x
        h = blk.ln1(x)
        q, k, v = blk.qkv(h).chunk(3, -1)
        D = q.shape[-1]; hd = D // nh
        q, k, v = (z.view(B, L, nh, hd).transpose(1, 2) for z in (q, k, v))
        att = ((q @ k.transpose(-1, -2)) / math.sqrt(hd) +
               torch.triu(torch.full((L, L), float("-inf"), device=dev), 1)).softmax(-1)
        o = att @ v
        for (zl, zh) in head_zero:
            if zl == li:
                o[:, zh] = 0
        x = x + blk.proj(o.transpose(1, 2).reshape(B, L, D))
        x = x + blk.mlp(blk.ln2(x))
        if clean and li in clean:
            V = clean[li]
            d = x - x_in
            x = x_in + d - (d @ V) @ V.T
        resid.append(x)
    return model.head(model.lnf(x)), resid

logits0, resid0 = manual_forward()
m_lab = torch.zeros_like(seq); m_lab[:, 2:] = seq[:, 2:] ^ seq[:, 1:-1] ^ seq[:, :-2]
valid = torch.zeros(B, L, dtype=torch.bool, device=dev); valid[:, 2:] = True
D4 = (resid0[4] - resid0[3])[valid].float()
mv = m_lab[valid]
w = (D4[mv == 1].mean(0) - D4[mv == 0].mean(0))
w = (w / w.norm())[:, None]

def per_cluster_ce(logits):
    logp = F.log_softmax(logits[:, :-1], -1)
    ce = -logp.gather(-1, seq[:, 1:, None]).squeeze(-1)[:, TPOS - 1].reshape(-1).cpu()
    return [ce[sub][assign == j].mean().item() for j in range(kstar)]

conds = {"none": {}, "zero L3h0 (transport)": {"head_zero": [(3, 0)]},
         "clean m@L4 (evidence)": {"clean": {4: w}},
         "zero L6h3 (xnor head)": {"head_zero": [(6, 3)]},
         "zero L5h2 (pooler)": {"head_zero": [(5, 2)]}}
print("\ncausal fingerprints (per-cluster CE):")
fp = {}
for cn, kw in conds.items():
    with torch.no_grad():
        vals = per_cluster_ce(manual_forward(**kw)[0])
    fp[cn] = vals
    print(f"  {cn:24s} " + "  ".join(f"c{j}:{v:.3f}" for j, v in enumerate(vals)))

json.dump({"kstar": kstar, "sil": sils, "ari": float(A),
           "assign_counts": [int((assign == j).sum()) for j in range(kstar)],
           "events": events, "fingerprints": fp},
          open(os.path.join(run, "taskdisc.json"), "w"))
torch.save({"sub": sub, "assign": assign, "truth": ys},
           os.path.join(run, "taskdisc_assign.pt"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
cols = plt.cm.tab10.colors
for j in range(kstar):
    axes[0].plot(steps, CM[j], color=cols[j], label=f"cluster {j} (n={(assign==j).sum()})")
for grp in groups:
    axes[0].axvline(grp[0]["step"], color="gray", ls=":", alpha=0.5)
axes[0].set_xscale("log"); axes[0].legend(fontsize=8)
axes[0].set_title(f"DISCOVERED clusters (no labels), k*={kstar}, ARI={A:.3f}")
axes[0].set_xlabel("step"); axes[0].set_ylabel("CE (nats)")
for v, (nm, c) in enumerate(zip(names, ["tab:orange", "tab:blue", "tab:red", "tab:purple"])):
    axes[1].plot(steps, Xs[ys == v].mean(0).numpy(), color=c, label=nm)
axes[1].set_xscale("log"); axes[1].legend(fontsize=8)
axes[1].set_title("TRUE classes (for comparison)"); axes[1].set_xlabel("step")
fig.tight_layout(); fig.savefig(os.path.join(run, "taskdisc.png"), dpi=140)
print("saved", os.path.join(run, "taskdisc.png"))
