"""Patch battery with support-safe donors (tuples with count>=8 in eval set)."""
import glob
import torch
from train_multi import GPTV
from pcfg2 import V
from pcfg4 import dirichlet_tables
exec(open("pcfg4b_grade.py").read().split("model = GPTV")[0].split("g = torch.Generator")[0])

cnt = torch.bincount(tid, minlength=16)
ok_t = (cnt >= 8).nonzero().flatten()
print(f"supported tuples: {len(ok_t)}/16 (counts min {int(cnt.min())} max {int(cnt.max())})")
g = torch.Generator().manual_seed(3)
pick = ok_t[torch.randint(0, len(ok_t), (B,), generator=g)]
clash = pick == tid
while clash.any():
    pick[clash] = ok_t[torch.randint(0, len(ok_t), (int(clash.sum()),), generator=g)]
    clash = pick == tid
tid_don = pick
lam_nat, lam_don = MSGS[tid], MSGS[tid_don]

model = GPTV(L=96, nl=6, vocab=V).to(dev)
ck = sorted(glob.glob("runs/pcfg4a0.05_dense_seed0/ckpts/*.pt"))
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
def safe_mus(ell):
    out = {}
    for c in (36, 37):
        m = torch.zeros(16, 128, dtype=resids[ell].dtype)
        for t in ok_t.tolist():
            m[t] = resids[ell][tid == t, c].mean(0)
        out[c] = m
    return out

for TGT, Q, tag in ((6, 37, "1-fusion leaf6"), (8, 39, "MULTI-fusion leaf8")):
    clean, root_cl = bp_forced(lam_nat, TGT)
    cf, root_cf = bp_forced(lam_don, TGT)
    lcl, lcf = clean.clamp_min(1e-9).log(), cf.clamp_min(1e-9).log()
    p = cf.clamp_min(1e-9)
    print(f"\n[{tag}] oracle gap {float((p*(p.log()-lp0[:,Q].cpu())).sum(-1).mean()):.4f}")
    def score(lp, t2):
        dn = (lp[:, Q] - lp0[:, Q]).cpu(); do = lcf - lcl
        dn, do = dn - dn.mean(-1, keepdim=True), do - do.mean(-1, keepdim=True)
        print(f"  {t2}: slope {float((dn*do).sum()/(do*do).sum()):.3f} "
              f"corr {float((dn.flatten()@do.flatten())/(dn.flatten().norm()*do.flatten().norm()+1e-12)):.3f} "
              f"KL {float((p*(p.log()-lp[:,Q].cpu())).sum(-1).mean()):.4f}")
    seq_cf = seq.clone()
    seq_cf[:, 36] = (tid_don // V).to(dev); seq_cf[:, 37] = (tid_don % V).to(dev)
    score(fwd(s=seq_cf), "tape ceiling")
    for ell in (0, 1, 2, 3):
        mus = safe_mus(ell)
        def tm(x, m=mus):
            for c in (36, 37):
                x[:, c] += (m[c][tid_don] - m[c][tid]).to(dev)
            return x
        score(fwd(tm, at=ell), f"tuple-mean@L{ell}")
    if TGT == 8:
        mus1 = safe_mus(1)
        D = {c: (mus1[c][tid_don] - mus1[c][tid]).to(dev) for c in (36, 37)}
        for a in (0.25, 0.5, 0.75, 1.0):
            cfa, _ = bp_forced((1 - a) * lam_nat + a * lam_don, TGT)
            lcfa = cfa.clamp_min(1e-9).log()
            def da(x, a=a):
                for c in (36, 37):
                    x[:, c] += a * D[c]
                return x
            lpa = fwd(da, at=1)
            dn = (lpa[:, Q] - lp0[:, Q]).cpu(); do = lcfa - lcl
            dn, do = dn - dn.mean(-1, keepdim=True), do - do.mean(-1, keepdim=True)
            dof = lcf - lcl; dof = dof - dof.mean(-1, keepdim=True)
            print(f"  dose a={a}: slope {float((dn*do).sum()/(do*do).sum()):.3f} "
                  f"corr {float((dn.flatten()@do.flatten())/(dn.flatten().norm()*do.flatten().norm()+1e-12)):.3f} "
                  f"mag BP {float(do.norm()/dof.norm()):.2f}|net {float(dn.norm()/dof.norm()):.2f}")
        ntr = B // 2
        X = keep0[5].cpu().double()
        mu_, sd_ = X[:ntr].mean(0), X[:ntr].std(0) + 1e-6
        Xb = torch.cat([(X - mu_)/sd_, torch.ones(B, 1)], 1)
        W = torch.linalg.solve(Xb[:ntr].T @ Xb[:ntr] + 10*torch.eye(129), Xb[:ntr].T @ root_cl[:ntr, :3])
        mus0 = safe_mus(0)
        def tm0(x, m=mus0):
            for c in (36, 37):
                x[:, c] += (m[c][tid_don] - m[c][tid]).to(dev)
            return x
        _, keepP = fwd(tm0, at=0, keeps=True)
        Xp = torch.cat([(keepP[5].cpu().double() - mu_)/sd_, torch.ones(B, 1)], 1)
        d_dec = ((Xp - Xb) @ W)[ntr:]
        d_orc = (root_cf - root_cl)[ntr:, :3]
        print(f"  root-belief propagation: slope {float((d_dec*d_orc).sum()/(d_orc*d_orc).sum()):.3f} "
              f"corr {float((d_dec.flatten()@d_orc.flatten())/(d_dec.flatten().norm()*d_orc.flatten().norm()+1e-12)):.3f}")
