"""Per-parameter support analysis of cluster gradients (RRXOR-3).

p_i = g_i^2 / |g|^2 as a distribution over parameters. For each pair:
  BC = sum sqrt(p q) = cos(|g_a|,|g_b|)   (shared support)
  sigma = cos(g_a,g_b) / BC               (sign coherence within shared support)
Per cluster: participation ratio PR = 1/sum p_i^2 (effective # params).
Also: top named parameters of the overlap integrand sqrt(p q) at key checkpoints.
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

CE = torch.zeros(len(ckpts), B, len(TPOS))
for i, cp in enumerate(ckpts):
    model.load_state_dict(torch.load(cp)); model.eval()
    with torch.no_grad():
        logp = F.log_softmax(model(seq[:, :-1]), -1)
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
# flat-index -> parameter name map
pinfo = []
off = 0
for n, p in model.named_parameters():
    pinfo.append((n, off, off + p.numel(), p.shape))
    off += p.numel()
def locate(idx):
    for n, a, b, sh in pinfo:
        if a <= idx < b:
            rel = idx - a
            return f"{n}[{np.unravel_index(rel, sh)}]"
    return "?"

def grads():
    out = []
    for j in order:
        m = (assign == j).to(dev)
        logp = F.log_softmax(model(seq[:, :-1]), -1)
        ce = -logp.gather(-1, seq[:, 1:, None]).squeeze(-1)
        loss = ce[inst_b[m], inst_p[m] - 1].mean()
        gs = torch.autograd.grad(loss, list(model.parameters()))
        out.append(torch.cat([g.flatten() for g in gs]))
    return out

pairs = [(0, 1, "det·match"), (0, 2, "det·mis"), (1, 2, "match·mis")]
T = len(ckpts)
BCs, SIG, PRs = np.zeros((T, 3)), np.zeros((T, 3)), np.zeros((T, 3))
TOPSHARE = np.zeros((T, 3))                      # frac of BC from top 1000 coords
KEY = {600: None, 2000: None, 8000: None}
for i, cp in enumerate(ckpts):
    model.load_state_dict(torch.load(cp)); model.eval()
    G = grads()
    A = [g.abs() for g in G]
    nrm = [float(g.norm()) for g in G]
    for a in range(3):
        p = (G[a] / (nrm[a] + 1e-30)).pow(2)
        PRs[i, a] = 1.0 / float(p.pow(2).sum())
    for pi, (a, b, _) in enumerate(pairs):
        integrand = A[a] * A[b] / (nrm[a] * nrm[b] + 1e-30)   # = sqrt(p q)
        bc = float(integrand.sum())
        cos = float(G[a] @ G[b]) / (nrm[a] * nrm[b] + 1e-30)
        BCs[i, pi] = bc; SIG[i, pi] = cos / (bc + 1e-12)
        top = integrand.topk(1000).values
        TOPSHARE[i, pi] = float(top.sum()) / (bc + 1e-30)
    if int(steps[i]) in KEY:
        a, b = 0, 2                                          # det·mis
        integrand = A[a] * A[b]
        tv, ti = integrand.topk(8)
        KEY[int(steps[i])] = [(locate(int(k)),
                               float(torch.sign(G[a][k] * G[b][k])))
                              for k in ti]
    if i % 20 == 0:
        print(f"ckpt {i}/{T}", flush=True)

np.savez(os.path.join(run, "gradsupport.npz"), BC=BCs, SIG=SIG, PR=PRs,
         TOPSHARE=TOPSHARE, steps=steps)
print("\ntop-8 shared-support parameters for det·mis (sign of agreement):")
for s, lst in KEY.items():
    if lst:
        print(f"  step {s}:")
        for nm, sg in lst:
            print(f"    {'-' if sg < 0 else '+'} {nm}")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for pi, (_, __, lab) in enumerate(pairs):
    axes[0].plot(steps, BCs[:, pi], label=lab)
    axes[1].plot(steps, SIG[:, pi], label=lab)
axes[0].set_title("shared support BC = cos(|g_a|,|g_b|)")
axes[1].set_title("sign coherence σ = cos/BC (agree/oppose within support)")
for ax in axes[:2]:
    ax.set_xscale("log"); ax.legend(fontsize=8); ax.axhline(0, color="gray", ls=":")
    ax.set_xlabel("step")
for a, nm, c in zip(range(3), names, ["tab:blue", "tab:green", "tab:red"]):
    axes[2].plot(steps, PRs[:, a], color=c, label=nm)
axes[2].set_yscale("log"); axes[2].set_xscale("log")
axes[2].set_title("effective support size PR = 1/Σp² (of 1.05M params)")
axes[2].legend(fontsize=8); axes[2].set_xlabel("step")
fig.suptitle(f"{name}: per-parameter support decomposition of cluster gradients")
fig.tight_layout(); fig.savefig(os.path.join(run, "gradsupport.png"), dpi=140)
print("saved", os.path.join(run, "gradsupport.png"))
