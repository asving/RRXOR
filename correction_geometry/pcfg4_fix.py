"""Honest re-grade: (a) position-MARGINAL baselines per class (Dirichlet tables
skew marginals; contextual gain = marginalCE - floor, not ln4 - floor);
(b) causal suite re-targeted to one-fusion range: patch pair(4,5) cols 36-37,
query col 37 predicts leaf 6 (b=1, one fusion away)."""
import glob
import torch
from train_multi import GPTV
from pcfg2 import V, NTREE
from pcfg4 import ALPHA, dirichlet_tables, bp4

P = dirichlet_tables(ALPHA)
dev = "cuda"
ev = torch.load("runs/pcfg4_dense_seed0/evalset.pt")
seq = ev["eseq"][:2048].to(dev)
B = seq.shape[0]
lv = seq[:, 32:48].cpu()

def marg(i):
    pi = torch.full((1, V), 1 / V, dtype=torch.double)
    for lev in range(1, 5):
        child = i >> (4 - lev)
        t = P[lev - 1]
        pi = torch.einsum("na,abc->nb", pi, t) if child % 2 == 0 else \
             torch.einsum("na,abc->nc", pi, t)
        pi = pi / pi.sum(-1, keepdim=True)
    return pi[0]

evfull = ev["eseq"]
print("class: floor | marginalCE | contextual gain (marg-floor) | ln4-marg (skew)")
for b in (0, 1, 2, 3):
    idxs = [i for i in range(1, 16) if (i & -i).bit_length() - 1 == b]
    fs, ms = [], []
    for i in idxs:
        for t in range(NTREE):
            lvt = evfull[:, 16 * t:16 * (t + 1)]
            fs.append(-bp4(lvt, i, P).clamp_min(1e-12).log().gather(1, lvt[:, i:i+1]))
            m = marg(i).clamp_min(1e-12).log()
            ms.append(-m[lvt[:, i]])
    f, m_ = float(torch.cat(fs).mean()), float(torch.cat(ms).mean())
    print(f"  b{b}: {f:.4f} | {m_:.4f} | {m_-f:.4f} | {1.3863-m_:.4f}")

# ---- causal, one-fusion range: query col 37 -> leaf 6 ----
Q, TGT = 37, 6
tid = (lv[:, 4] * V + lv[:, 5]).long()
MSGS = torch.stack([P[3][:, t // V, t % V] for t in range(16)])
def bp_forced(force):
    lam = {4: torch.ones(B, 16, V, dtype=torch.double)}
    for j in range(TGT):
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
    for lev in range(1, 5):
        child = TGT >> (4 - lev); sib = child ^ 1
        ls = lam[lev][:, sib]
        pi = torch.einsum("na,abc,nc->nb", pi, P[lev-1], ls) if child % 2 == 0 else \
             torch.einsum("na,abc,nb->nc", pi, P[lev-1], ls)
        pi = pi / pi.sum(-1, keepdim=True).clamp_min(1e-30)
    return pi

lam_nat = MSGS[tid]
g = torch.Generator().manual_seed(3)
tid_don = (tid + 1 + torch.randint(0, 15, (B,), generator=g)) % 16
lam_don = MSGS[tid_don]
clean, cf = bp_forced(lam_nat), bp_forced(lam_don)
lcl, lcf = clean.clamp_min(1e-9).log(), cf.clamp_min(1e-9).log()

model = GPTV(L=96, nl=6, vocab=V).to(dev)
ck = sorted(glob.glob("runs/pcfg4_dense_seed0/ckpts/*.pt"))
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
def score(lp, tag, do_ref=None):
    dn = (lp[:, Q] - lp0[:, Q]).cpu(); do = (lcf - lcl) if do_ref is None else do_ref
    dn, do = dn - dn.mean(-1, keepdim=True), do - do.mean(-1, keepdim=True)
    sl = float((dn * do).sum() / (do * do).sum())
    co = float((dn.flatten() @ do.flatten()) / (dn.flatten().norm() * do.flatten().norm() + 1e-12))
    p = (cf if do_ref is None else None)
    k = float((cf.clamp_min(1e-9) * (cf.clamp_min(1e-9).log() - lp[:, Q].cpu())).sum(-1).mean()) if do_ref is None else float("nan")
    print(f"{tag}: slope {sl:.3f}  corr {co:.3f}" + (f"  KL {k:.4f}" if do_ref is None else ""))

pq = cf.clamp_min(1e-9)
print(f"\noracle gap @ (pair45 -> leaf6): KL(BPcf||net_clean) {float((pq*(pq.log()-lp0[:,Q].cpu())).sum(-1).mean()):.4f}")
seq_cf = seq.clone()
seq_cf[:, 36] = (tid_don // V).to(dev); seq_cf[:, 37] = (tid_don % V).to(dev)
score(fwd(s=seq_cf), "tape swap (ceiling)")
for ell in (0, 1, 2, 3):
    mus = {c: torch.stack([resids[ell][tid == t, c].mean(0) for t in range(16)]) for c in (36, 37)}
    def tm(x, m=mus):
        for c in (36, 37):
            x[:, c] += (m[c][tid_don] - m[c][tid]).to(dev)
        return x
    score(fwd(tm, at=ell), f"tuple-mean@L{ell}")
gr = torch.Generator(device=dev).manual_seed(11)
mus1 = {c: torch.stack([resids[1][tid == t, c].mean(0) for t in range(16)]) for c in (36, 37)}
nrm = (mus1[36][tid_don] - mus1[36][tid]).norm(dim=1, keepdim=True).to(dev)
rd = torch.randn(B, 128, generator=gr, device=dev); rd = rd / rd.norm(dim=1, keepdim=True) * nrm
score(fwd(lambda x: (x.__setitem__((slice(None), 36), x[:, 36] + rd), x.__setitem__((slice(None), 37), x[:, 37] + rd), x)[-1], at=1), "rand ctrl@L1")
print("dose-response (tuple-mean@L1):")
D = {c: (mus1[c][tid_don] - mus1[c][tid]).to(dev) for c in (36, 37)}
for a in (0.25, 0.5, 0.75, 1.0):
    cfa = bp_forced((1 - a) * lam_nat + a * lam_don)
    lcfa = cfa.clamp_min(1e-9).log()
    def da(x, a=a):
        for c in (36, 37):
            x[:, c] += a * D[c]
        return x
    lpa = fwd(da, at=1)
    dn = (lpa[:, Q] - lp0[:, Q]).cpu(); do = lcfa - lcl
    dn, do = dn - dn.mean(-1, keepdim=True), do - do.mean(-1, keepdim=True)
    dof = (lcf - lcl); dof = dof - dof.mean(-1, keepdim=True)
    print(f"  a={a}: slope {float((dn*do).sum()/(do*do).sum()):.3f}  "
          f"corr {float((dn.flatten()@do.flatten())/(dn.flatten().norm()*do.flatten().norm()+1e-12)):.3f}  "
          f"mag BP {float(do.norm()/dof.norm()):.2f} | net {float(dn.norm()/dof.norm()):.2f}")
