"""Refined causal suite: (i) tape-swap ceiling, (ii) encoder-image patch with
per-sequence logit-shift transfer coefficient, (iii) PROPAGATION: does the decoded
root belief at late layers shift as BP-cf predicts when we patch the pair rep?"""
import glob, os
import torch
from train_multi import GPTV
import pcfg2 as P
exec(open("pcfg2_causal.py").read().split("model = GPTV")[0])   # reuse setup/oracles

model = GPTV(L=96, nl=6, vocab=P.V).to(dev)
ck = sorted(glob.glob(os.path.join(base, "runs", run, "ckpts", "*.pt")))
model.load_state_dict(torch.load(ck[-1], map_location=dev)); model.eval()

def fwd(s, patch_layer=None, delta=None, cols=(36, 37)):
    x = model.wte(s) + model.wpe(torch.arange(96, device=dev))[None]
    keep = {}
    for li, blk in enumerate(model.blocks):
        if patch_layer is not None and li == patch_layer:
            x = x.clone()
            for c in cols:
                x[:, c] += delta
        x = blk(x)
        keep[li] = x[:, 39].detach()
    return torch.log_softmax(model.head(model.lnf(x))[:, :, :P.V].double(), -1), keep

def kl(p, logq):
    return float((p.clamp_min(1e-9) * (p.clamp_min(1e-9).log() - logq.cpu())).sum(-1).mean())

lp0, keep0 = fwd(seq)
lcl, lcf = clean["pred"].clamp_min(1e-9).log(), cf["pred"].clamp_min(1e-9).log()

# rung 1: tape swap = ceiling (yield of donor symbol, production k=0)
seq_cf = seq.clone()
y = P.TABLES[3][donor]                     # (B, M, 2); take k=0
seq_cf[:, 36] = y[:, 0, 0].to(dev); seq_cf[:, 37] = y[:, 0, 1].to(dev)
lpT, _ = fwd(seq_cf)
print(f"tape-swap ceiling: KL(BPcf||net) {kl(cf['pred'], lpT[:, Q]):.4f} "
      f"(clean ref {kl(cf['pred'], lp0[:, Q]):.4f})")

# rung 2: encoder patch at L2 + transfer coefficient on per-seq logit shifts
with torch.no_grad():
    _, resids = model(seq, return_resid=True)
H = resids[2][:, [36, 37]].mean(1)
mu = torch.stack([H[pair45 == v].mean(0) for v in range(4)])
delta = (mu[donor] - mu[pair45]).to(dev)
lp2, keep2 = fwd(seq, patch_layer=2, delta=delta)
dn = (lp2[:, Q] - lp0[:, Q]).cpu(); do = (lcf - lcl)
dn, do = dn - dn.mean(-1, keepdim=True), do - do.mean(-1, keepdim=True)
slope = float((dn * do).sum() / (do * do).sum())
corr = float((dn.flatten() @ do.flatten()) / (dn.flatten().norm() * do.flatten().norm() + 1e-12))
print(f"encoder patch@L2: transfer slope {slope:.3f}  shift corr {corr:.3f}  "
      f"KL(BPcf||patched) {kl(cf['pred'], lp2[:, Q]):.4f}")

# rung 3: PROPAGATION — decoded root belief at col 39, late layers, patched vs BP-cf
ntr = B // 2
for ell in (3, 4, 5):
    X = keep0[ell].cpu().double()
    Ycl = clean["root"][:, :3]
    mu_, sd_ = X[:ntr].mean(0), X[:ntr].std(0) + 1e-6
    Xb = torch.cat([(X - mu_) / sd_, torch.ones(B, 1)], 1)
    A = Xb[:ntr].T @ Xb[:ntr] + 10 * torch.eye(Xb.shape[1])
    W = torch.linalg.solve(A, Xb[:ntr].T @ Ycl[:ntr])
    Xp = torch.cat([(keep2[ell].cpu().double() - mu_) / sd_, torch.ones(B, 1)], 1)
    d_dec = ((Xp - Xb) @ W)[ntr:]
    d_orc = (cf["root"] - clean["root"])[ntr:, :3]
    sl = float((d_dec * d_orc).sum() / (d_orc * d_orc).sum())
    co = float((d_dec.flatten() @ d_orc.flatten()) /
               (d_dec.flatten().norm() * d_orc.flatten().norm() + 1e-12))
    r2c = float(1 - ((Ycl[ntr:] - Xb[ntr:] @ W) ** 2).sum() / ((Ycl[ntr:] - Ycl[:ntr].mean(0)) ** 2).sum())
    print(f"root-belief probe @L{ell},col39 (clean R2 {r2c:.2f}): "
          f"propagation slope {sl:.3f}  corr {co:.3f}")
# control: does the decoded root shift under the random-direction patch?
gr = torch.Generator(device=dev).manual_seed(11)
rd = torch.randn(B, 128, generator=gr, device=dev)
rd = rd / rd.norm(dim=1, keepdim=True) * delta.norm(dim=1, keepdim=True)
_, keepR = fwd(seq, patch_layer=2, delta=rd)
ell = 5
X = keep0[ell].cpu().double()
mu_, sd_ = X[:ntr].mean(0), X[:ntr].std(0) + 1e-6
Xb = torch.cat([(X - mu_) / sd_, torch.ones(B, 1)], 1)
A = Xb[:ntr].T @ Xb[:ntr] + 10 * torch.eye(Xb.shape[1])
W = torch.linalg.solve(A, Xb[:ntr].T @ clean["root"][:ntr, :3])
XR = torch.cat([(keepR[ell].cpu().double() - mu_) / sd_, torch.ones(B, 1)], 1)
d_dec = ((XR - Xb) @ W)[ntr:]
d_orc = (cf["root"] - clean["root"])[ntr:, :3]
print(f"rand-ctrl propagation @L5: slope {float((d_dec*d_orc).sum()/(d_orc*d_orc).sum()):.3f}  "
      f"corr {float((d_dec.flatten()@d_orc.flatten())/(d_dec.flatten().norm()*d_orc.flatten().norm()+1e-12)):.3f}")
