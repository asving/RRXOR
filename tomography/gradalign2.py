"""Decompose M(t): support overlap vs within-module alignment, per-module migration.

cos_ab(t) = sum_m w_ab^m(t) * c_ab^m(t),  w = |g_a^m||g_b^m| / (|g_a||g_b|).
Outputs: localization heatmaps for ALL clusters; gradient norms; per-module signed
contributions w*c for the det-mis and det-match pairs (the migration picture).
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
ev = torch.load(os.path.join(run, "evalset.pt"))
seq = ev["eseq"][:512].to(dev)
B, L = seq.shape
TPOS = torch.arange(33, L - 1, device=dev)
model = GPT(L=96, d=128, nl=4).to(dev)
ckpts = sorted(glob.glob(os.path.join(run, "ckpts", "*.pt")))
steps = np.array([int(os.path.basename(c)[4:10]) for c in ckpts])

# rebuild clusters (deterministic, same as gradalign.py)
CE = torch.zeros(len(ckpts), B, len(TPOS))
for i, cp in enumerate(ckpts):
    model.load_state_dict(torch.load(cp)); model.eval()
    with torch.no_grad():
        logits = model(seq[:, :-1])
        logp = F.log_softmax(logits, -1)
        CE[i] = (-logp.gather(-1, seq[:, 1:, None]).squeeze(-1))[:, TPOS - 1].cpu()
X = CE.permute(1, 2, 0).reshape(-1, len(ckpts))
X = F.avg_pool1d(F.pad(X[:, None], (1, 1), mode="replicate"), 3, stride=1).squeeze(1)

def kmeans(X, k, iters=60, restarts=10, seed=0):
    gg = torch.Generator().manual_seed(seed)
    best = (None, 1e18)
    for r in range(restarts):
        C = X[torch.randperm(X.shape[0], generator=gg)[:k]].clone()
        for _ in range(iters):
            d = torch.cdist(X, C); a = d.argmin(1)
            C = torch.stack([X[a == j].mean(0) if (a == j).any() else C[j] for j in range(k)])
        inert = (X - C[a]).pow(2).sum().item()
        if inert < best[1]:
            best = (a.clone(), inert)
    return best[0]

assign = kmeans(X, 3)
CMs = torch.stack([X[assign == j].mean(0) for j in range(3)])
det_c = int(CMs[:, -1].argmin()); mis_c = int(CMs[:, len(steps) // 3].argmax())
match_c = [j for j in range(3) if j not in (det_c, mis_c)][0]
order = [det_c, match_c, mis_c]; names = ["det", "match", "mis"]

inst_b = torch.arange(B, device=dev)[:, None].expand(B, len(TPOS)).reshape(-1)
inst_p = TPOS[None, :].expand(B, len(TPOS)).reshape(-1)
pnames = [n for n, _ in model.named_parameters()]
def mg(n):
    if "wte" in n or "wpe" in n: return "emb"
    if "head" in n or "lnf" in n: return "head"
    li = int(n.split(".")[1]) + 1
    return f"L{li}.{'attn' if ('qkv' in n or 'proj' in n or 'ln1' in n) else 'mlp'}"
groups = [mg(n) for n in pnames]
gset = ["emb", "L1.attn", "L1.mlp", "L2.attn", "L2.mlp", "L3.attn", "L3.mlp",
        "L4.attn", "L4.mlp", "head"]

def cluster_grads(j):
    m = (assign == j).to(dev)
    logits = model(seq[:, :-1])
    logp = F.log_softmax(logits, -1)
    ce = -logp.gather(-1, seq[:, 1:, None]).squeeze(-1)
    loss = ce[inst_b[m], inst_p[m] - 1].mean()
    return torch.autograd.grad(loss, list(model.parameters()))

T, G = len(ckpts), len(gset)
locs = np.zeros((T, 3, G)); norms = np.zeros((T, 3))
pairs = [(0, 1, "det·match"), (0, 2, "det·mis"), (1, 2, "match·mis")]
contrib = np.zeros((T, 3, G))                      # w*c per module per pair
for i, cp in enumerate(ckpts):
    model.load_state_dict(torch.load(cp)); model.eval()
    GS = [cluster_grads(j) for j in order]
    modvec = []
    for a in range(3):
        mv = {g: [] for g in gset}
        for g, grp in zip(GS[a], groups):
            mv[grp].append(g.flatten())
        mv = {g: torch.cat(v) for g, v in mv.items()}
        modvec.append(mv)
        tot = sum(float(v.pow(2).sum()) for v in mv.values())
        norms[i, a] = tot ** 0.5
        for gi, g in enumerate(gset):
            locs[i, a, gi] = float(mv[g].pow(2).sum()) / (tot + 1e-30)
    for pi, (a, b, _) in enumerate(pairs):
        na, nb = norms[i, a], norms[i, b]
        for gi, g in enumerate(gset):
            contrib[i, pi, gi] = float(modvec[a][g] @ modvec[b][g]) / (na * nb + 1e-30)
    if i % 20 == 0:
        print(f"ckpt {i}/{T}", flush=True)

np.savez(os.path.join(run, "gradalign2.npz"), locs=locs, norms=norms,
         contrib=contrib, steps=steps, gset=np.array(gset))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for a, nm in enumerate(names):
    ax = axes[a]
    im = ax.imshow(locs[:, a].T, aspect="auto", origin="lower", cmap="viridis",
                   vmin=0, vmax=0.7, extent=[0, T, -0.5, G - 0.5])
    ax.set_yticks(range(G)); ax.set_yticklabels(gset, fontsize=7)
    ax.set_xticks(np.arange(T)[:: max(1, T // 6)])
    ax.set_xticklabels([steps[k] for k in range(0, T, max(1, T // 6))], fontsize=7)
    ax.set_title(f"where ḡ_{nm} lives (‖g‖² fraction)")
    fig.colorbar(im, ax=ax)
fig.suptitle(f"{name}: gradient localization by cluster")
fig.tight_layout(); fig.savefig(os.path.join(run, "gradalign2_loc.png"), dpi=140)

fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.2))
ax = axes[0]
for a, nm, c in zip(range(3), names, ["tab:blue", "tab:green", "tab:red"]):
    ax.plot(steps, norms[:, a], color=c, label=nm)
ax.set_xscale("log"); ax.set_yscale("log"); ax.legend(fontsize=8)
ax.set_title("cluster-gradient norms ‖ḡ_c(t)‖"); ax.set_xlabel("step")
for pi, ax, ttl in [(1, axes[1], "det·match: module contributions w·c"),
                    (2, axes[2], "det·mis: module contributions w·c")]:
    A = contrib[:, pi - 0 if pi == 1 else 1] if False else None
for pnum, ax in [(0, axes[1]), (1, axes[2])]:
    A = contrib[:, pnum]                       # (T, G) signed contributions
    top = np.argsort(-np.abs(A).max(0))[:5]
    for gi in top:
        ax.plot(steps, A[:, gi], label=gset[gi])
    ax.plot(steps, A.sum(1), "k--", lw=1, label="total cos")
    ax.axhline(0, color="gray", ls=":")
    ax.set_xscale("log"); ax.legend(fontsize=7)
    ax.set_title(f"{pairs[pnum][2]}: per-module w·c (sums to cos)")
    ax.set_xlabel("step")
fig.suptitle(f"{name}: decomposition of M(t) into module contributions")
fig.tight_layout(); fig.savefig(os.path.join(run, "gradalign2_decomp.png"), dpi=140)
print("saved", os.path.join(run, "gradalign2_loc.png"), "and gradalign2_decomp.png")
