"""Re-score everything on the CONSISTENT subset (donor symbol yields a valid parse
all the way to the root). Inconsistent counterfactuals have zero-vector BP targets:
~0 KL contribution but garbage slopes."""
import glob, os
import torch
from train_multi import GPTV
import pcfg2 as P
exec(open("pcfg2_causal.py").read().split("model = GPTV")[0])

ok = cf["pred"].sum(-1) > 0.5
print(f"consistent counterfactuals: {int(ok.sum())}/{B} ({float(ok.float().mean()):.2f})")
model = GPTV(L=96, nl=6, vocab=P.V).to(dev)
ck = sorted(glob.glob(os.path.join(base, "runs", run, "ckpts", "*.pt")))
model.load_state_dict(torch.load(ck[-1], map_location=dev)); model.eval()
with torch.no_grad():
    _, resids = model(seq, return_resid=True)

def fwd(patchfn=None, at=None, keeps=False):
    x = model.wte(seq) + model.wpe(torch.arange(96, device=dev))[None]
    keep = {}
    for li, blk in enumerate(model.blocks):
        if patchfn is not None and li == at:
            x = patchfn(x.clone())
        x = blk(x)
        keep[li] = x[:, 39].detach()
    lp = torch.log_softmax(model.head(model.lnf(x))[:, :, :P.V].double(), -1)
    return (lp, keep) if keeps else lp

lp0, keep0 = fwd(keeps=True)
lcl, lcf = clean["pred"].clamp_min(1e-9).log(), cf["pred"].clamp_min(1e-9).log()

def score(lp, tag):
    dn = (lp[:, Q] - lp0[:, Q]).cpu()[ok]; do = (lcf - lcl)[ok]
    dn, do = dn - dn.mean(-1, keepdim=True), do - do.mean(-1, keepdim=True)
    sl = float((dn * do).sum() / (do * do).sum())
    co = float((dn.flatten() @ do.flatten()) / (dn.flatten().norm() * do.flatten().norm() + 1e-12))
    p = cf["pred"][ok].clamp_min(1e-9)
    k = float((p * (p.log() - lp[:, Q].cpu()[ok])).sum(-1).mean())
    print(f"{tag}: slope {sl:.3f}  corr {co:.3f}  KL(BPcf||·) {k:.4f}")

p = cf["pred"][ok].clamp_min(1e-9)
print(f"oracle gap on subset: KL(BPcf||net_clean) {float((p*(p.log()-lp0[:,Q].cpu()[ok])).sum(-1).mean()):.4f}")
seq_cf = seq.clone()
y = P.TABLES[3][donor]
seq_cf[:, 36] = y[:, 0, 0].to(dev); seq_cf[:, 37] = y[:, 0, 1].to(dev)
x = model.wte(seq_cf) + model.wpe(torch.arange(96, device=dev))[None]
for blk in model.blocks: x = blk(x)
score(torch.log_softmax(model.head(model.lnf(x))[:, :, :P.V].double(), -1), "tape swap (ceiling)   ")

R2 = resids[2].to(dev)
didx = torch.zeros(B, dtype=torch.long)
pool = {v: (pair45 == v).nonzero().flatten() for v in range(4)}
g2 = torch.Generator().manual_seed(5)
for b in range(B):
    c = pool[int(donor[b])]
    didx[b] = c[torch.randint(0, len(c), (1,), generator=g2)]
def transplant(x):
    x[:, 36] = R2[didx, 36]; x[:, 37] = R2[didx, 37]; return x
lpT, keepT = fwd(transplant, at=2, keeps=True)
score(lpT, "transplant@L2         ")
mus = {c: torch.stack([resids[2][pair45 == v, c].mean(0) for v in range(4)]) for c in (36, 37)}
def colmean(x):
    for c in (36, 37):
        x[:, c] += (mus[c][donor] - mus[c][pair45]).to(dev)
    return x
lpM, keepM = fwd(colmean, at=2, keeps=True)
score(lpM, "per-col class-mean@L2 ")

# propagation, consistent subset only (probe trained on all clean, eval on ok half)
for tag, keepP in (("transplant", keepT), ("class-mean", keepM)):
    ell = 5
    X = keep0[ell].cpu().double()
    ntr = B // 2
    mu_, sd_ = X[:ntr].mean(0), X[:ntr].std(0) + 1e-6
    Xb = torch.cat([(X - mu_) / sd_, torch.ones(B, 1)], 1)
    A = Xb[:ntr].T @ Xb[:ntr] + 10 * torch.eye(Xb.shape[1])
    W = torch.linalg.solve(A, Xb[:ntr].T @ clean["root"][:ntr, :3])
    Xp = torch.cat([(keepP[ell].cpu().double() - mu_) / sd_, torch.ones(B, 1)], 1)
    m = ok.clone(); m[:ntr] = False
    d_dec = ((Xp - Xb) @ W)[m]
    d_orc = (cf["root"] - clean["root"])[m, :3]
    sl = float((d_dec * d_orc).sum() / (d_orc * d_orc).sum())
    co = float((d_dec.flatten() @ d_orc.flatten()) /
               (d_dec.flatten().norm() * d_orc.flatten().norm() + 1e-12))
    print(f"root-belief propagation @L5 ({tag}): slope {sl:.3f}  corr {co:.3f}")

print("\nlayer sweep (per-col class-mean, consistent subset):")
for ell in (0, 1, 2, 3, 4, 5):
    mus_l = {c: torch.stack([resids[ell][pair45 == v, c].mean(0) for v in range(4)]) for c in (36, 37)}
    def cm(x, m=mus_l):
        for c in (36, 37):
            x[:, c] += (m[c][donor] - m[c][pair45]).to(dev)
        return x
    score(fwd(cm, at=ell), f"  patch@L{ell}")
