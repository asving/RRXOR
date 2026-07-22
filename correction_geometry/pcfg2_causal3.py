"""Rung 3: full-column donor TRANSPLANT at L2 entry (replacement, not addition) +
per-column class-mean patch. Donor sequence chosen so its pair(4,5) symbol = the
counterfactual target; its columns carry contaminants about its own unrelated
context, so high transfer here also certifies the reader consumes only the symbol."""
import glob, os
import torch
from train_multi import GPTV
import pcfg2 as P
exec(open("pcfg2_causal.py").read().split("model = GPTV")[0])

model = GPTV(L=96, nl=6, vocab=P.V).to(dev)
ck = sorted(glob.glob(os.path.join(base, "runs", run, "ckpts", "*.pt")))
model.load_state_dict(torch.load(ck[-1], map_location=dev)); model.eval()
with torch.no_grad():
    _, resids = model(seq, return_resid=True)

def fwd(patchfn=None, at=None):
    x = model.wte(seq) + model.wpe(torch.arange(96, device=dev))[None]
    for li, blk in enumerate(model.blocks):
        if patchfn is not None and li == at:
            x = patchfn(x.clone())
        x = blk(x)
    return torch.log_softmax(model.head(model.lnf(x))[:, :, :P.V].double(), -1)

def score(lp, tag):
    dn = (lp[:, Q] - lp0[:, Q]).cpu(); do = (lcf - lcl)
    dn, do = dn - dn.mean(-1, keepdim=True), do - do.mean(-1, keepdim=True)
    sl = float((dn * do).sum() / (do * do).sum())
    co = float((dn.flatten() @ do.flatten()) / (dn.flatten().norm() * do.flatten().norm() + 1e-12))
    k = float((cf["pred"].clamp_min(1e-9) * (cf["pred"].clamp_min(1e-9).log() - lp[:, Q].cpu())).sum(-1).mean())
    print(f"{tag}: transfer slope {sl:.3f}  corr {co:.3f}  KL(BPcf||·) {k:.4f}")

lp0 = fwd()
lcl, lcf = clean["pred"].clamp_min(1e-9).log(), cf["pred"].clamp_min(1e-9).log()

# donor index: same pair symbol as the counterfactual target
didx = torch.zeros(B, dtype=torch.long)
pool = {v: (pair45 == v).nonzero().flatten() for v in range(4)}
g2 = torch.Generator().manual_seed(5)
for b in range(B):
    c = pool[int(donor[b])]
    didx[b] = c[torch.randint(0, len(c), (1,), generator=g2)]

R2 = resids[2].to(dev)          # state entering block 2
def transplant(x):
    x[:, 36] = R2[didx, 36]; x[:, 37] = R2[didx, 37]
    return x
score(fwd(transplant, at=2), "transplant@L2 (cols 36-37)")

# per-column class-mean patch at L2 (addition, but column-resolved)
mus = {c: torch.stack([resids[2][pair45 == v, c].mean(0) for v in range(4)]) for c in (36, 37)}
def colmean(x):
    for c in (36, 37):
        x[:, c] += (mus[c][donor] - mus[c][pair45]).to(dev)
    return x
score(fwd(colmean, at=2), "per-col class-mean@L2")

# transplant of col 37 only (the completion column)
def t37(x):
    x[:, 37] = R2[didx, 37]; return x
score(fwd(t37, at=2), "transplant@L2 (col 37 only)")
