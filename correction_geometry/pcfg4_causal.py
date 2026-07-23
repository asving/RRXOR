"""P4 centerpiece: encoder-instrumented interchange + dose-response on PCFG-4.
Geometry: tree 2, pair(4,5)=cols 36-37, query col 39 (predict leaf 8).
Every tuple is a valid counterfactual; messages take 16 graded interior values."""
import glob
import torch
from train_multi import GPTV
from pcfg2 import V
from pcfg4 import ALPHA, dirichlet_tables

P = dirichlet_tables(ALPHA)
dev = "cuda"
ev = torch.load("runs/pcfg4_dense_seed0/evalset.pt")
seq = ev["eseq"][:2048].to(dev)
B, Q = seq.shape[0], 39
lv = seq[:, 32:48].cpu()
tid = (lv[:, 4] * V + lv[:, 5]).long()

def bp_forced(force):
    lam = {4: torch.ones(B, 16, V, dtype=torch.double)}
    for j in range(8):
        lam[4][:, j] = 0.0
        lam[4][torch.arange(B), j, lv[:, j]] = 1.0
    for lev in range(4, 0, -1):
        lam[lev-1] = torch.einsum("abc,nib,nic->nia", P[lev-1], lam[lev][:, 0::2], lam[lev][:, 1::2])
        if lev - 1 == 3:
            lam[3] = lam[3].clone(); lam[3][:, 2] = force
            for l2 in range(3, 0, -1):
                lam[l2-1] = torch.einsum("abc,nib,nic->nia", P[l2-1], lam[l2][:, 0::2], lam[l2][:, 1::2])
            break
    pi = torch.full((B, V), 1/V, dtype=torch.double)
    root = pi * lam[0][:, 0]
    out = {"root": root / root.sum(-1, keepdim=True).clamp_min(1e-30)}
    tgt = 8
    for lev in range(1, 5):
        child = tgt >> (4 - lev); sib = child ^ 1
        ls = lam[lev][:, sib]
        pi = torch.einsum("na,abc,nc->nb", pi, P[lev-1], ls) if child % 2 == 0 else \
             torch.einsum("na,abc,nb->nc", pi, P[lev-1], ls)
        pi = pi / pi.sum(-1, keepdim=True).clamp_min(1e-30)
    out["pred"] = pi
    return out

MSGS = torch.stack([P[3][:, t // V, t % V] for t in range(16)])     # 16 x V
lam_nat = MSGS[tid]
g = torch.Generator().manual_seed(3)
tid_don = (tid + 1 + torch.randint(0, 15, (B,), generator=g)) % 16
lam_don = MSGS[tid_don]
clean, cf = bp_forced(lam_nat), bp_forced(lam_don)
lcl, lcf = clean["pred"].clamp_min(1e-9).log(), cf["pred"].clamp_min(1e-9).log()

model = GPTV(L=96, nl=6, vocab=V).to(dev)
ck = sorted(glob.glob("runs/pcfg4_dense_seed0/ckpts/*.pt"))
model.load_state_dict(torch.load(ck[-1], map_location=dev)); model.eval()
with torch.no_grad():
    _, resids = model(seq, return_resid=True)

def fwd(patchfn=None, at=None, s=None, keeps=False):
    x = model.wte(seq if s is None else s) + model.wpe(torch.arange(96, device=dev))[None]
    keep = {}
    for li, blk in enumerate(model.blocks):
        if patchfn is not None and li == at:
            x = patchfn(x.clone())
        x = blk(x)
        keep[li] = x[:, 39].detach()
    lp = torch.log_softmax(model.head(model.lnf(x))[:, :, :V].double(), -1)
    return (lp, keep) if keeps else lp

lp0, keep0 = fwd(keeps=True)
def score(lp, tag):
    dn = (lp[:, Q] - lp0[:, Q]).cpu(); do = (lcf - lcl)
    dn, do = dn - dn.mean(-1, keepdim=True), do - do.mean(-1, keepdim=True)
    sl = float((dn * do).sum() / (do * do).sum())
    co = float((dn.flatten() @ do.flatten()) / (dn.flatten().norm() * do.flatten().norm() + 1e-12))
    p = cf["pred"].clamp_min(1e-9)
    print(f"{tag}: slope {sl:.3f}  corr {co:.3f}  KL {float((p*(p.log()-lp[:,Q].cpu())).sum(-1).mean()):.4f}")

p = cf["pred"].clamp_min(1e-9)
print(f"oracle gap: KL(BPcf||net_clean) {float((p*(p.log()-lp0[:,Q].cpu())).sum(-1).mean()):.4f}")
seq_cf = seq.clone()
seq_cf[:, 36] = (tid_don // V).to(dev); seq_cf[:, 37] = (tid_don % V).to(dev)
score(fwd(s=seq_cf), "tape swap (ceiling)")

# instruments: 16-tuple class means + lambda-linear encoder, per layer
ntr = B // 2
Xlam = (lam_nat / lam_nat.sum(-1, keepdim=True))[:, :3]
for ell in (0, 1, 2, 3):
    mus = {c: torch.stack([resids[ell][tid == t, c].mean(0) for t in range(16)]) for c in (36, 37)}
    def tm(x, m=mus):
        for c in (36, 37):
            x[:, c] += (m[c][tid_don] - m[c][tid]).to(dev)
        return x
    score(fwd(tm, at=ell), f"tuple-mean@L{ell}")
# encoder gate: lambda-linear vs tuple-identity ceiling at L1
for ell in (0, 1, 2):
    r2s = []
    for c in (36, 37):
        Y = resids[ell][:, c].cpu().double()
        Xb = torch.cat([Xlam, torch.ones(B, 1)], 1)
        W = torch.linalg.solve(Xb[:ntr].T @ Xb[:ntr] + 1e-3 * torch.eye(4), Xb[:ntr].T @ Y[:ntr])
        r2l = 1 - ((Y[ntr:] - Xb[ntr:] @ W) ** 2).sum() / ((Y[ntr:] - Y[:ntr].mean(0)) ** 2).sum()
        mus_c = torch.stack([Y[:ntr][tid[:ntr] == t].mean(0) for t in range(16)])
        r2t = 1 - ((Y[ntr:] - mus_c[tid[ntr:]]) ** 2).sum() / ((Y[ntr:] - Y[:ntr].mean(0)) ** 2).sum()
        r2s.append((float(r2l), float(r2t)))
    print(f"encoder gate @L{ell}: col36 lam-linear R2 {r2s[0][0]:.3f} / tuple ceiling {r2s[0][1]:.3f}; "
          f"col37 {r2s[1][0]:.3f} / {r2s[1][1]:.3f}")

# dose-response with the tuple-mean instrument at L1 (interpolate delta) vs BP(lam_alpha)
mus1 = {c: torch.stack([resids[1][tid == t, c].mean(0) for t in range(16)]) for c in (36, 37)}
D = {c: (mus1[c][tid_don] - mus1[c][tid]).to(dev) for c in (36, 37)}
print("dose-response (tuple-mean@L1):")
for a in (0.25, 0.5, 0.75, 1.0):
    cfa = bp_forced((1 - a) * lam_nat + a * lam_don)
    lcfa = cfa["pred"].clamp_min(1e-9).log()
    def da(x, a=a):
        for c in (36, 37):
            x[:, c] += a * D[c]
        return x
    lpa = fwd(da, at=1)
    dn = (lpa[:, Q] - lp0[:, Q]).cpu(); do = (lcfa - lcl)
    dn, do = dn - dn.mean(-1, keepdim=True), do - do.mean(-1, keepdim=True)
    dof = (lcf - lcl); dof = dof - dof.mean(-1, keepdim=True)
    print(f"  a={a}: slope {float((dn*do).sum()/(do*do).sum()):.3f}  "
          f"corr {float((dn.flatten()@do.flatten())/(dn.flatten().norm()*do.flatten().norm()+1e-12)):.3f}  "
          f"mag BP {float(do.norm()/dof.norm()):.2f} | net {float(dn.norm()/dof.norm()):.2f}")

# P5: root-belief probe at L5 col39, entropy-quantile split
X = keep0[5].cpu().double()
mu_, sd_ = X[:ntr].mean(0), X[:ntr].std(0) + 1e-6
Xb = torch.cat([(X - mu_) / sd_, torch.ones(B, 1)], 1)
W = torch.linalg.solve(Xb[:ntr].T @ Xb[:ntr] + 10 * torch.eye(129), Xb[:ntr].T @ clean["root"][:ntr, :3])
ent = -(clean["root"] * clean["root"].clamp_min(1e-12).log()).sum(-1)
te = ent[ntr:]
for tag, m in (("low-entropy half ", te < te.median()), ("high-entropy half", te >= te.median())):
    Yh = (Xb[ntr:] @ W)[m]; Yt = clean["root"][ntr:, :3][m]
    r2 = float(1 - ((Yt - Yh) ** 2).sum() / ((Yt - clean["root"][:ntr, :3].mean(0)) ** 2).sum())
    print(f"P5 root probe, {tag}: R2 {r2:.3f} (n={int(m.sum())}, mean H {float(ent[ntr:][m].mean()):.3f})")
