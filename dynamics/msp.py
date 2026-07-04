"""Test C (PREREG2): coarsening trajectory of the belief geometry.

Ground-truth Bayes filter over generator states [A, B0, B1, C0, C1]:
  A: emit u~U -> B_u;  B_v: emit w~U -> C_{v^w};  C_x: emit x -> A.
Belief after token t = state distribution before emitting t+1 (aligned with resid at pos t).

Per checkpoint: ridge regression resid(L4, pos 2..94) -> belief; R^2 on held-out seqs.
Outputs: msp_r2.png (R^2 vs step, overlaid with ce1 relief), msp_unfold.png (simplex scatter).
"""
import glob, json, os, sys
import torch
from train import GPT

name = sys.argv[1] if len(sys.argv) > 1 else "seed0"
dev = "cuda"
base = os.path.dirname(os.path.abspath(__file__))
run = os.path.join(base, "runs", name)
ev = torch.load(os.path.join(run, "evalset.pt"))
seq = ev["eseq"][:512].to(dev)
B, L = seq.shape
POS = list(range(2, L - 1))

def beliefs(seq_cpu):
    B, L = seq_cpu.shape
    pi = torch.zeros(B, 5)
    pi[:, 0] = 1 / 3; pi[:, 1:] = 1 / 6
    out = torch.zeros(B, L, 5)
    ar = torch.arange(B)
    for t in range(L):
        o = seq_cpu[:, t]
        new = torch.zeros_like(pi)
        new[ar, 1 + o] += pi[:, 0] * 0.5              # A -emit o-> B_o
        new[ar, 3 + o] += pi[:, 1] * 0.5              # B_0 -emit o-> C_{0^o}
        new[ar, 3 + (1 - o)] += pi[:, 2] * 0.5        # B_1 -emit o-> C_{1^o}
        new[:, 0] += pi[ar, 3 + o]                    # C_o -emit o-> A
        pi = new / new.sum(1, keepdim=True).clamp_min(1e-12)
        out[:, t] = pi
    return out

bel = beliefs(seq.cpu()).to(dev)                       # (B, L, 5)
Y5 = bel[:, POS, :].reshape(-1, 5)
Y3 = torch.stack([Y5[:, 0], Y5[:, 1] + Y5[:, 2], Y5[:, 3] + Y5[:, 4]], 1)  # A/B/C marginal
ntr = int(0.7 * B) * len(POS)                          # split by sequence (block layout)

def ridge_r2(X, Y):
    Xb = torch.cat([X, torch.ones_like(X[:, :1])], 1)
    Xtr, Ytr, Xte, Yte = Xb[:ntr], Y[:ntr], Xb[ntr:], Y[ntr:]
    W = torch.linalg.solve(Xtr.T @ Xtr + 1.0 * torch.eye(Xtr.shape[1], device=dev),
                           Xtr.T @ Ytr)
    P = Xte @ W
    r2 = 1 - ((Yte - P) ** 2).sum() / ((Yte - Yte.mean(0)) ** 2).sum()
    return r2.item(), P

ckpts = sorted(glob.glob(os.path.join(run, "ckpts", "*.pt")))
model = GPT().to(dev)
rows, scat = [], {}
SCAT_AT = [50, 400, 800, 1500, 2500, 8000]
for cp in ckpts:
    step = int(os.path.basename(cp)[4:10])
    model.load_state_dict(torch.load(cp)); model.eval()
    with torch.no_grad():
        _, resid = model(seq, return_resid=True)
    X = resid[4][:, POS, :].reshape(-1, resid[4].shape[-1]).float()
    X = (X - X[:ntr].mean(0)) / (X[:ntr].std(0) + 1e-6)
    r5, _ = ridge_r2(X, Y5)
    r3, P3 = ridge_r2(X, Y3)
    rows.append({"step": step, "r2_full5": r5, "r2_phase3": r3})
    if step in SCAT_AT:
        scat[step] = P3.cpu()
    print(f"step {step:6d}  R2(5-state) {r5:.3f}  R2(phase) {r3:.3f}", flush=True)

json.dump(rows, open(os.path.join(run, "msp_r2.json"), "w"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# R^2 trajectory vs the loss curves
er = [json.loads(l) for l in open(os.path.join(run, "eval.jsonl"))]
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot([r["step"] for r in rows], [r["r2_phase3"] for r in rows], "o-", ms=3,
        label="R$^2$ resid$\\to$phase belief (A/B/C)", color="tab:green")
ax.plot([r["step"] for r in rows], [r["r2_full5"] for r in rows], "s-", ms=3,
        label="R$^2$ resid$\\to$full 5-state belief", color="tab:purple", alpha=0.6)
ax.set_xscale("log"); ax.set_xlabel("step"); ax.set_ylabel("held-out R$^2$"); ax.set_ylim(0, 1)
ax2 = ax.twinx()
ax2.plot([r["step"] for r in er], [r["ce1"] for r in er], color="tab:red", alpha=0.5,
         label="ce type-1 (random pos)")
ax2.axhline(np.log(2), ls=":", color="gray"); ax2.set_ylabel("CE (nats)", color="tab:red")
ax.legend(loc="upper left", fontsize=8); ax.set_title(f"Belief-geometry R$^2$ vs training, {name}")
fig.tight_layout(); fig.savefig(os.path.join(run, "msp_r2.png"), dpi=140)

# simplex unfolding scatter
tri = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]])   # A, B, C corners
true_xy = (Y3.cpu().numpy() @ tri)
cls = Y3.argmax(1).cpu().numpy()
fig, axes = plt.subplots(1, len(scat) + 1, figsize=(3 * (len(scat) + 1), 3.2))
axes[0].scatter(true_xy[::7, 0], true_xy[::7, 1], c=cls[::7], cmap="brg", s=2, alpha=0.4)
axes[0].set_title("ground-truth MSP\n(phase marginal)", fontsize=9)
for ax_, (step, P) in zip(axes[1:], sorted(scat.items())):
    xy = P.numpy() @ tri
    ax_.scatter(xy[::7, 0], xy[::7, 1], c=cls[-len(P):][::7], cmap="brg", s=2, alpha=0.4)
    ax_.set_title(f"step {step}", fontsize=9)
for ax_ in axes:
    ax_.plot(*np.vstack([tri, tri[:1]]).T, color="k", lw=0.5)
    ax_.set_xticks([]); ax_.set_yticks([]); ax_.set_aspect("equal")
fig.suptitle("Residual stream projected onto phase-belief simplex (held-out seqs)")
fig.tight_layout(); fig.savefig(os.path.join(run, "msp_unfold.png"), dpi=140)
print(os.path.join(run, "msp_unfold.png"))
