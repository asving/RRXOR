"""Disambiguate partial transfer: tape ceiling, class-mean layer sweep, transplant."""
import glob
import torch
from train_multi import GPTV
from pcfg2 import V, TABLES
from pcfg3 import soft_tables
exec(open("pcfg3_causal.py").read().split("model = GPTV")[0])

model = GPTV(L=96, nl=6, vocab=V).to(dev)
ck = sorted(glob.glob("runs/pcfg3_dense_seed0/ckpts/*.pt"))
model.load_state_dict(torch.load(ck[-1], map_location=dev)); model.eval()
with torch.no_grad():
    _, resids = model(seq, return_resid=True)

def fwd(patchfn=None, at=None, s=None):
    x = model.wte(seq if s is None else s) + model.wpe(torch.arange(96, device=dev))[None]
    for li, blk in enumerate(model.blocks):
        if patchfn is not None and li == at:
            x = patchfn(x.clone())
        x = blk(x)
    return torch.log_softmax(model.head(model.lnf(x))[:, :, :V].double(), -1)

lp0 = fwd()
lcl, lcf = clean["pred"].clamp_min(1e-9).log(), cf["pred"].clamp_min(1e-9).log()
def score(lp, tag):
    m = canon
    dn = (lp[:, Q] - lp0[:, Q]).cpu()[m]; do = (lcf - lcl)[m]
    dn, do = dn - dn.mean(-1, keepdim=True), do - do.mean(-1, keepdim=True)
    sl = float((dn * do).sum() / (do * do).sum())
    co = float((dn.flatten() @ do.flatten()) / (dn.flatten().norm() * do.flatten().norm() + 1e-12))
    p = cf["pred"][m].clamp_min(1e-9)
    print(f"{tag}: slope {sl:.3f}  corr {co:.3f}  KL {float((p*(p.log()-lp[:,Q].cpu()[m])).sum(-1).mean()):.4f}")

seq_cf = seq.clone()
y = TABLES[3][donor]
seq_cf[:, 36] = y[:, 0, 0].to(dev); seq_cf[:, 37] = y[:, 0, 1].to(dev)
score(fwd(s=seq_cf), "tape swap (ceiling)")

didx = torch.zeros(B, dtype=torch.long)
pool = {v: ((mapsym == v) & canon).nonzero().flatten() for v in range(V)}
g2 = torch.Generator().manual_seed(5)
for b in range(B):
    c = pool[int(donor[b])]
    didx[b] = c[torch.randint(0, len(c), (1,), generator=g2)]
for ell in (0, 1, 2):
    Rl = resids[ell].to(dev)
    def tr(x, R=Rl):
        x[:, 36] = R[didx, 36]; x[:, 37] = R[didx, 37]; return x
    score(fwd(tr, at=ell), f"transplant@L{ell}")
for ell in (0, 1, 2, 3, 4):
    mus_l = {c: torch.stack([resids[ell][canon & (mapsym == v), c].mean(0) for v in range(V)]) for c in (36, 37)}
    def cm(x, m=mus_l):
        for c in (36, 37):
            x[:, c] += (m[c][donor] - m[c][mapsym]).to(dev)
        return x
    score(fwd(cm, at=ell), f"class-mean@L{ell}")
