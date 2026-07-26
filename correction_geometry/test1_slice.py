"""PREREG12 Test-1 pilot slice: compute A (degree-2 reachability) for cells with
known formation verdicts + one fresh probe (dead pair-function on rrxor3)."""
import glob
import torch
from train import GPT, gen_batch as gen_rrxor3
from train27 import gen_batch27, NESTED

dev = "cuda"
torch.manual_seed(0)

def ridge_r2(X, Y, lam=1e-2):
    n = X.shape[0]; ntr = int(0.7 * n)
    mu, sd = X[:ntr].mean(0), X[:ntr].std(0) + 1e-6
    Xb = torch.cat([(X - mu) / sd, torch.ones(n, 1)], 1)
    W = torch.linalg.solve(Xb[:ntr].T @ Xb[:ntr] + lam * torch.eye(Xb.shape[1]), Xb[:ntr].T @ Y[:ntr])
    P = Xb[ntr:] @ W
    return float(1 - ((Y[ntr:] - P) ** 2).sum() / ((Y[ntr:] - Y[:ntr].mean(0)) ** 2).sum())

# ---------- m27 cells: A at Phi0 (degree-2 token features) for xor-part and xnor-correction ----------
g = torch.Generator().manual_seed(5)
seq, cls = gen_batch27(4096, 192, g)
t0 = 150                                   # a late position
det = (cls[:, t0] % 3 == 2)
S = seq[det].double() * 2 - 1              # +-1 encoding
C = cls[det]
W_LAG = 12
feats = [S[:, t0 - l] for l in range(1, W_LAG + 1)]
feats += [S[:, t0 - a] * S[:, t0 - b] for a in range(1, W_LAG + 1) for b in range(a + 1, W_LAG + 1)]
X = torch.stack(feats, 1)
tgt = S[:, t0]                              # det target (+-1)
blk = (C[:, t0] // 3) % 9
xnor_corr = (torch.tensor(NESTED)[blk] == 1).double() * 2 - 1     # the flip indicator
A_tgt = ridge_r2(X, tgt[:, None])
A_xnor0 = ridge_r2(X, xnor_corr[:, None])
Xp = torch.cat([X, torch.nn.functional.one_hot(blk, 9).double()], 1)
A_xnor1 = ridge_r2(Xp, xnor_corr[:, None])
print(f"m27: A(det-target | deg2 tokens)      = {A_tgt:.3f}   (xor-part; pooled sign (5/9)^2 = {(5/9)**2:.3f})")
print(f"m27: A(xnor-corr  | deg2 tokens)      = {max(0,A_xnor0):.3f}   (gated at Phi0: expect ~0)")
print(f"m27: A(xnor-corr  | +phase-mod-9)     = {A_xnor1:.3f}   (reachable after prerequisite)")

# ---------- rrxor3 fresh probe: dead pair-function x_{t-3} xor x_{t-7} ----------
ev = torch.load("runs/seed0/evalset.pt")
eseq = ev["eseq"][:2048].to(dev)
model = GPT(L=96, nl=4).to(dev)
ck = sorted(glob.glob("runs/seed0/ckpts/*.pt"))
model.load_state_dict(torch.load(ck[-1], map_location=dev)); model.eval()
with torch.no_grad():
    _, resids = model(eseq, return_resid=True)
rnd = GPT(L=96, nl=4).to(dev)               # random-init control
torch.manual_seed(123)
for p in rnd.parameters():
    torch.nn.init.normal_(p, std=0.02)
with torch.no_grad():
    _, resR = rnd(eseq, return_resid=True)
POS = list(range(40, 90, 7))
f = (eseq[:, [p - 3 for p in POS]] ^ eseq[:, [p - 7 for p in POS]]).double().cpu()
def probe_stack(rs):
    X = torch.cat([rs[li][:, POS] for li in (2, 3, 4)], -1).reshape(-1, 3 * rs[2].shape[-1]).cpu().double()
    return ridge_r2(X, f.reshape(-1, 1), lam=10)
print(f"rrxor3 dead cell: probe R2 trained {probe_stack(resids):.3f} vs random-init {probe_stack(resR):.3f} "
      f"(formation = excess; predict ~0)")
print(f"rrxor3 xor cell: A = 1 by identity (deg-2 member), formed (known: solved to floor)")
print(f"pcfg4b pair-posterior: A = 1 (deterministic deg-2 of 2 tokens), formed (probe R2 ~ .99, known)")
