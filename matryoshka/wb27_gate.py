"""Whitebox step 2: is the net's gate the Bayes log-odds, and is it an additive
violation comb?

s_t = (2u_t - 1) * (logit1 - logit0)  — the net's log-odds on the xor candidate.
Lambda_t = log[(qX + qR/2) / (qN + qR/2)] — Bayes log-odds from the verified filter.

(i) calibration: corr(s, Lambda) overall / sync band / by next-slot type.
(ii) matched filter: ridge-regress s_t and Lambda_t on lagged violation bits
     {m_{t-l}}_{l=0..80}; compare R^2 and the weight combs.
"""
import glob, json, os, sys
import torch
import numpy as np
from train import GPT
from train27 import NESTED

name = sys.argv[1] if len(sys.argv) > 1 else "m27_nl6_seed1"
dev = "cuda"
base = os.path.dirname(os.path.abspath(__file__))
run = os.path.join(base, "runs", name)
cfg = json.load(open(os.path.join(run, "config.json")))
ev = torch.load(os.path.join(run, "evalset.pt"))
seq, cls = ev["eseq"].to(dev), ev["ecls"].to(dev)
B, L = seq.shape
rule = torch.tensor(NESTED, device=dev)

# Bayes filter (validated in msp27_full.py: CE 0.4621, argmax 1.000)
@torch.no_grad()
def beliefs27(seq):
    pi = torch.full((B, 27), 1 / 27, device=dev)
    out = torch.zeros(B, L, 27, device=dev)
    classes = torch.arange(27, device=dev)
    det = classes % 3 == 2
    crule = rule[(classes // 3) % 9]
    out[:, 0] = pi
    for t in range(1, L):
        pi = pi.roll(1, dims=1)
        if t >= 2:
            u = (seq[:, t - 1] ^ seq[:, t - 2]).float()
            x = seq[:, t].float()
            pred = (u[:, None] + crule[None, :].float()) % 2
            lik = torch.where(det[None, :], (pred == x[:, None]).float(),
                              torch.full_like(pred, 0.5))
            pi = pi * lik
        pi = pi / pi.sum(1, keepdim=True).clamp_min(1e-30)
        out[:, t] = pi
    return out

PI = beliefs27(seq)
classes = torch.arange(27, device=dev)
nxt_det = (classes + 1) % 3 == 2
nrule = rule[(((classes + 1) % 27) // 3) % 9]
qR = PI[:, :, ~nxt_det].sum(-1)
qX = PI[:, :, nxt_det & (nrule == 0)].sum(-1)
qN = PI[:, :, nxt_det & (nrule == 1)].sum(-1)
CLIP = 12.0
Lam = (torch.log((qX + qR / 2).clamp_min(1e-9)) -
       torch.log((qN + qR / 2).clamp_min(1e-9))).clamp(-CLIP, CLIP)

model = GPT(L=cfg["L"], d=cfg["d"], nl=cfg["nl"]).to(dev)
cp = sorted(glob.glob(os.path.join(run, "ckpts", "*.pt")))[-1]
model.load_state_dict(torch.load(cp)); model.eval()
with torch.no_grad():
    logits = model(seq)
lam_net = logits[:, :, 1] - logits[:, :, 0]
u = torch.zeros_like(seq); u[:, 1:] = seq[:, 1:] ^ seq[:, :-1]
s = (2 * u.float() - 1) * lam_net
s = s.clamp(-CLIP, CLIP)

pos = torch.arange(L, device=dev)
def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return (a * b).sum() / (a.norm() * b.norm() + 1e-9)

bands = {"all(4-190)": (4, 190), "sync(4-40)": (4, 40), "late(108-190)": (108, 190)}
print("=== gate calibration: corr(net s_t, Bayes Lambda_t) ===")
for bn, (a, b_) in bands.items():
    pm = (pos >= a) & (pos <= b_)
    S, Lm = s[:, pm].flatten(), Lam[:, pm].flatten()
    print(f"{bn:14s} r = {corr(S, Lm).item():.3f}   "
          f"(pre-det queries only: r = "
          f"{corr(s[:, pm][(cls % 3 == 1)[:, pm]], Lam[:, pm][(cls % 3 == 1)[:, pm]]).item():.3f})")

# ---- matched filter: s_t and Lambda_t from lagged violation bits ----
m_lab = torch.zeros_like(seq); m_lab[:, 2:] = seq[:, 2:] ^ seq[:, 1:-1] ^ seq[:, :-2]
NL = 81
tpos = torch.arange(NL + 2, L, device=dev)
predet = (cls % 3 == 1)[:, tpos]
X = torch.stack([m_lab[:, tpos - l].float() for l in range(NL)], -1)  # (B, T, NL)
Xf = X[predet]                                          # pre-det samples
Ys, Yl = s[:, tpos][predet], Lam[:, tpos][predet]
n = Xf.shape[0]; ntr = int(0.7 * n)
def ridge(X, Y, lam=10.0):
    Xb = torch.cat([X, torch.ones_like(X[:, :1])], 1)
    W = torch.linalg.solve(Xb[:ntr].T @ Xb[:ntr] + lam * torch.eye(Xb.shape[1], device=dev),
                           Xb[:ntr].T @ Y[:ntr])
    P = Xb[ntr:] @ W
    r2 = 1 - ((Y[ntr:] - P) ** 2).sum() / ((Y[ntr:] - Y[:ntr].mean()) ** 2).sum()
    return r2.item(), W[:-1].cpu().numpy()

r2_s, w_s = ridge(Xf, Ys)
r2_l, w_l = ridge(Xf, Yl)
print(f"\n=== matched filter (81 lagged violation bits, pre-det positions) ===")
print(f"net gate s_t:      R2 = {r2_s:.3f}")
print(f"Bayes Lambda_t:    R2 = {r2_l:.3f}")
print(f"weight-comb correlation w_s vs w_l: r = "
      f"{np.corrcoef(w_s.ravel(), w_l.ravel())[0,1]:.3f}")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
idx = torch.randperm(min(n - ntr, 20000))[:6000]
axes[0].scatter(Lam[:, tpos][predet][ntr:][idx].cpu(), s[:, tpos][predet][ntr:][idx].cpu(),
                s=2, alpha=0.15)
axes[0].plot([-CLIP, CLIP], [-CLIP, CLIP], "r--", lw=1)
axes[0].set_xlabel("Bayes log-odds $\\Lambda_t$"); axes[0].set_ylabel("net gate $s_t$")
axes[0].set_title("Gate calibration (pre-det positions, all bands)")
axes[1].plot(range(NL), w_s / (np.abs(w_s).max() + 1e-9), label="net gate comb", lw=1)
axes[1].plot(range(NL), w_l / (np.abs(w_l).max() + 1e-9), label="Bayes comb", lw=1, alpha=0.7)
for k in range(0, NL, 27):
    axes[1].axvline(k, color="gray", ls=":", alpha=0.4)
axes[1].set_xlabel("lag $\\ell$ (tokens)"); axes[1].set_ylabel("normalized weight")
axes[1].set_title("Matched-filter weights on violation bits"); axes[1].legend()
fig.tight_layout(); fig.savefig(os.path.join(run, "wb_gate.png"), dpi=140)
print("saved", os.path.join(run, "wb_gate.png"))
