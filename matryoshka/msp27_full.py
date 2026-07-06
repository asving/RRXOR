"""Does the net carry the FULL 27-offset belief simplex, or only its decision quotient?

Targets from a ground-truth Bayes filter over offsets (belief over class of current token):
  pi   : full 27-dim posterior
  Qpi  : 3-dim decision functional — masses of next-slot {random, det-XOR, det-XNOR}
         (linear in pi: fixed sums of coordinates)
  Cpi  : 24-dim orthogonal complement ("fiber" coordinates the task never consumes)

Features: concatenation of ALL residual layers (7 x 128) at position t, from
  (a) trained net, (b) random-init net (control). Plus a token-window baseline
  (last 16 tokens + their xor bits u + violation bits m — the "modulo xor" concession).

Bands: sync (t 4-30), mid (31-60), late (108-180). Ridge (lambda on val), R2 on held-out
sequences. Prediction of the quotient account: R2(Qpi) high everywhere incl. sync,
R2(Cpi) ~ baseline. Simplex expectation: both high.
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

# ---------------- ground-truth Bayes filter over 27 offsets ----------------
@torch.no_grad()
def beliefs27(seq):
    B, L = seq.shape
    pi = torch.full((B, 27), 1 / 27, device=dev)
    out = torch.zeros(B, L, 27, device=dev)
    classes = torch.arange(27, device=dev)
    det = classes % 3 == 2
    crule = rule[(classes // 3) % 9]
    out[:, 0] = pi
    for t in range(1, L):
        pi = pi.roll(1, dims=1)                       # class advances by 1 mod 27
        if t >= 2:
            u = (seq[:, t - 1] ^ seq[:, t - 2]).float()          # (B,)
            x = seq[:, t].float()
            pred = (u[:, None] + crule[None, :].float()) % 2     # det value per class
            lik = torch.where(det[None, :],
                              (pred == x[:, None]).float(),      # 1 or 0
                              torch.full_like(pred, 0.5))
            pi = pi * lik
        pi = pi / pi.sum(1, keepdim=True).clamp_min(1e-30)
        out[:, t] = pi
    return out

PI = beliefs27(seq)                                    # (B, L, 27)

# oracle self-check: Bayes CE from the filter should match the analytic optimum 0.4621
nxt = PI.roll(1, dims=2)                               # dist over class of NEXT token: c -> c+1
classes = torch.arange(27, device=dev)
det = classes % 3 == 2
crule = rule[(classes // 3) % 9].float()
qR = nxt[:, :, ~det].sum(-1)
u = torch.zeros_like(seq, dtype=torch.float); u[:, 1:] = (seq[:, 1:] ^ seq[:, :-1]).float()
xorm = (crule[det[classes].nonzero().squeeze()] if False else crule[det]).float()  # rules of det classes
qdet = nxt[:, :, det]                                  # (B,L,9)
p_next_is_u = qdet[:, :, xorm == 0].sum(-1) + qR * 0.5
x_next = seq.roll(-1, dims=1).float()
p1 = torch.where(u.bool(), p_next_is_u, 1 - p_next_is_u)  # P(next=1)
ce = -(x_next * (p1.clamp(1e-9, 1 - 1e-9)).log() + (1 - x_next) * (1 - p1).clamp(1e-9, 1).log())
pos = torch.arange(L, device=dev)
lateband = (pos >= 108) & (pos < L - 1)
print(f"oracle check: Bayes CE (late) = {ce[:, lateband].mean().item():.4f}  "
      f"(analytic optimum 0.4621); argmax acc late = "
      f"{(PI[:, lateband].argmax(-1) == cls[:, lateband]).float().mean().item():.3f}")

# ---------------- targets: quotient and fiber ----------------
Qrows = torch.zeros(3, 27, device=dev)
nxt_det = (classes + 1) % 3 == 2
nrule = rule[(((classes + 1) % 27) // 3) % 9]
Qrows[0, ~nxt_det] = 1
Qrows[1, nxt_det & (nrule == 0)] = 1
Qrows[2, nxt_det & (nrule == 1)] = 1
Qb = torch.linalg.qr(Qrows.T)[0].T                     # orthonormalized quotient basis (3,27)
full = torch.linalg.qr(torch.cat([Qb.T, torch.eye(27, device=dev)], 1))[0]
Cb = full[:, 3:27].T                                   # fiber basis (24,27)

TGT = {"pi": PI, "Qpi": PI @ Qb.T, "Cpi": PI @ Cb.T}

# ---------------- feature sets ----------------
model = GPT(L=cfg["L"], d=cfg["d"], nl=cfg["nl"]).to(dev)
cp = sorted(glob.glob(os.path.join(run, "ckpts", "*.pt")))[-1]
model.load_state_dict(torch.load(cp)); model.eval()
with torch.no_grad():
    _, resid_t = model(seq, return_resid=True)
torch.manual_seed(999)
rnet = GPT(L=cfg["L"], d=cfg["d"], nl=cfg["nl"]).to(dev).eval()
with torch.no_grad():
    _, resid_r = rnet(seq, return_resid=True)

def concat(resids):
    return torch.cat(resids, -1)                       # (B, L, 7*128)

W = 16
def window_feats():
    m = torch.zeros_like(seq); m[:, 2:] = seq[:, 2:] ^ seq[:, 1:-1] ^ seq[:, :-2]
    uu = torch.zeros_like(seq); uu[:, 1:] = seq[:, 1:] ^ seq[:, :-1]
    feats = []
    for lag in range(W):
        for src in (seq, uu, m):
            feats.append(src.roll(lag, dims=1))
    return torch.stack(feats, -1).float()              # (B, L, 48)

FEATS = {"trained_concat": concat(resid_t), "randinit_concat": concat(resid_r),
         "window16+u+m": window_feats()}

# ---------------- ridge ----------------
def ridge_r2(X, Y, tr, va, te):
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xs = (X - mu) / sd
    Xb = torch.cat([Xs, torch.ones_like(Xs[:, :1])], 1)
    A = Xb[tr].T @ Xb[tr]
    b = Xb[tr].T @ Y[tr]
    best = (-1e9, None)
    for lam in (1e-2, 1e-1, 1, 10, 100, 1000):
        Wm = torch.linalg.solve(A + lam * torch.eye(A.shape[0], device=dev), b)
        P = Xb[va] @ Wm
        r2 = 1 - ((Y[va] - P) ** 2).sum() / ((Y[va] - Y[tr].mean(0)) ** 2).sum()
        if r2 > best[0]:
            best = (r2, Wm)
    P = Xb[te] @ best[1]
    return (1 - ((Y[te] - P) ** 2).sum() / ((Y[te] - Y[tr].mean(0)) ** 2).sum()).item()

BANDS = {"sync(4-30)": (4, 30), "mid(31-60)": (31, 60), "late(108-180)": (108, 180)}
sidx = torch.arange(B, device=dev)
tr_s, va_s, te_s = sidx < int(0.6 * B), (sidx >= int(0.6 * B)) & (sidx < int(0.7 * B)), sidx >= int(0.7 * B)

results = {}
for bname, (a, b_) in BANDS.items():
    posm = (pos >= a) & (pos <= b_)
    def flat(T):
        return T[:, posm].reshape(-1, T.shape[-1])
    n_per = int(posm.sum().item())
    def split(mask_s):
        return mask_s[:, None].expand(B, n_per).reshape(-1)
    tr, va, te = split(tr_s), split(va_s), split(te_s)
    for fname, F in FEATS.items():
        Xf = flat(F.float())
        for tname, T in TGT.items():
            r2 = ridge_r2(Xf, flat(T), tr, va, te)
            results[f"{bname}|{fname}|{tname}"] = r2
    line = " ".join(f"{t}:{results[f'{bname}|{f}|{t}']:+.3f}"
                    for f in FEATS for t in TGT)
    print(f"{bname:14s} " + line, flush=True)

print("\n--- table: R2 (held-out seqs), rows = features, cols = target ---")
for bname in BANDS:
    print(f"[{bname}]")
    for fname in FEATS:
        print(f"  {fname:16s} " + "  ".join(
            f"{t}: {results[f'{bname}|{fname}|{t}']:+.3f}" for t in TGT))
json.dump(results, open(os.path.join(run, "msp27_full.json"), "w"))
print("saved", os.path.join(run, "msp27_full.json"))
