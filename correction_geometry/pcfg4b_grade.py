"""PCFG-4b grading: evalset-exact floors/marginals, finals, causal suites at
one-fusion (leaf 6) and multi-fusion (leaf 8) ranges, dose-response, root probe."""
import glob, json
import torch
from train_multi import GPTV
from pcfg2 import V, NTREE
from pcfg4 import dirichlet_tables, bp4

P = dirichlet_tables(0.05)
dev = "cuda"
ev = torch.load("runs/pcfg4a0.05_dense_seed0/evalset.pt")
evfull = ev["eseq"]
def marg(i):
    pi = torch.full((1, V), 1 / V, dtype=torch.double)
    for lev in range(1, 5):
        c = i >> (4 - lev)
        pi = torch.einsum("na,abc->nb", pi, P[lev-1]) if c % 2 == 0 else torch.einsum("na,abc->nc", pi, P[lev-1])
        pi = pi / pi.sum(-1, keepdim=True)
    return pi[0]
FL, MG = {}, {}
for i in range(1, 16):
    b = (i & -i).bit_length() - 1
    fs, ms = [], []
    for t in range(NTREE):
        lvt = evfull[:, 16 * t:16 * (t + 1)]
        fs.append(-bp4(lvt, i, P).clamp_min(1e-12).log().gather(1, lvt[:, i:i+1]))
        ms.append(-marg(i).clamp_min(1e-12).log()[lvt[:, i]])
    FL.setdefault(b, []).append(float(torch.cat(fs).mean()))
    MG.setdefault(b, []).append(float(torch.cat(ms).mean()))
F = {b: sum(v)/len(v) for b, v in FL.items()}; M = {b: sum(v)/len(v) for b, v in MG.items()}
print("evalset floor: " + "  ".join(f"b{b}={F[b]:.4f}" for b in range(4)))
print("evalset marg:  " + "  ".join(f"b{b}={M[b]:.4f}" for b in range(4)))
for r in ("dense", "hard"):
    x = [json.loads(l) for l in open(f"runs/pcfg4a0.05_{r}_seed0/eval.jsonl")][-1]
    print(f"{r} @{x['step']}: " + "  ".join(f"b{b}={x[f'ce_b{b}']:.4f}" for b in range(4)))

seq = evfull[:2048].to(dev)
B = seq.shape[0]
lv = seq[:, 32:48].cpu()
tid = (lv[:, 4] * V + lv[:, 5]).long()
MSGS = torch.stack([P[3][:, t // V, t % V] for t in range(16)])
def bp_forced(force, TGT):
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
    root = pi * lam[0][:, 0]
    rootp = root / root.sum(-1, keepdim=True).clamp_min(1e-30)
    for lev in range(1, 5):
        child = TGT >> (4 - lev); sib = child ^ 1
        ls = lam[lev][:, sib]
        pi = torch.einsum("na,abc,nc->nb", pi, P[lev-1], ls) if child % 2 == 0 else \
             torch.einsum("na,abc,nb->nc", pi, P[lev-1], ls)
        pi = pi / pi.sum(-1, keepdim=True).clamp_min(1e-30)
    return pi, rootp

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

g = torch.Generator().manual_seed(3)
tid_don = (tid + 1 + torch.randint(0, 15, (B,), generator=g)) % 16
lam_nat, lam_don = MSGS[tid], MSGS[tid_don]
for TGT, Q, tag in ((6, 37, "1-fusion leaf6"), (8, 39, "MULTI-fusion leaf8")):
    clean, root_cl = bp_forced(lam_nat, TGT)
    cf, root_cf = bp_forced(lam_don, TGT)
    lcl, lcf = clean.clamp_min(1e-9).log(), cf.clamp_min(1e-9).log()
    p = cf.clamp_min(1e-9)
    print(f"\n[{tag}] oracle gap: {float((p*(p.log()-lp0[:,Q].cpu())).sum(-1).mean()):.4f}")
    seq_cf = seq.clone()
    seq_cf[:, 36] = (tid_don // V).to(dev); seq_cf[:, 37] = (tid_don % V).to(dev)
    def score(lp, t2):
        dn = (lp[:, Q] - lp0[:, Q]).cpu(); do = lcf - lcl
        dn, do = dn - dn.mean(-1, keepdim=True), do - do.mean(-1, keepdim=True)
        sl = float((dn*do).sum()/(do*do).sum())
        co = float((dn.flatten()@do.flatten())/(dn.flatten().norm()*do.flatten().norm()+1e-12))
        print(f"  {t2}: slope {sl:.3f} corr {co:.3f} KL {float((p*(p.log()-lp[:,Q].cpu())).sum(-1).mean()):.4f}")
    score(fwd(s=seq_cf), "tape ceiling")
    for ell in (0, 2):
        mus = {c: torch.stack([resids[ell][tid == t, c].mean(0) for t in range(16)]) for c in (36, 37)}
        def tm(x, m=mus):
            for c in (36, 37):
                x[:, c] += (m[c][tid_don] - m[c][tid]).to(dev)
            return x
        score(fwd(tm, at=ell), f"tuple-mean@L{ell}")
    if TGT == 8:
        # dose-response at multi-fusion + root-belief propagation
        mus1 = {c: torch.stack([resids[1][tid == t, c].mean(0) for t in range(16)]) for c in (36, 37)}
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
        # root probe (clean) + propagation under full patch
        ntr = B // 2
        X = keep0[5].cpu().double()
        mu_, sd_ = X[:ntr].mean(0), X[:ntr].std(0) + 1e-6
        Xb = torch.cat([(X - mu_)/sd_, torch.ones(B, 1)], 1)
        W = torch.linalg.solve(Xb[:ntr].T @ Xb[:ntr] + 10*torch.eye(129), Xb[:ntr].T @ root_cl[:ntr, :3])
        r2 = float(1 - ((root_cl[ntr:, :3] - Xb[ntr:] @ W)**2).sum() / ((root_cl[ntr:, :3] - root_cl[:ntr, :3].mean(0))**2).sum())
        ent = -(root_cl * root_cl.clamp_min(1e-12).log()).sum(-1)[ntr:]
        Yh = Xb[ntr:] @ W
        hi = ent >= ent.median()
        r2h = float(1 - ((root_cl[ntr:, :3][hi] - Yh[hi])**2).sum() / ((root_cl[ntr:, :3][hi] - root_cl[:ntr, :3].mean(0))**2).sum())
        mus0 = {c: torch.stack([resids[0][tid == t, c].mean(0) for t in range(16)]) for c in (36, 37)}
        def tm0(x, m=mus0):
            for c in (36, 37):
                x[:, c] += (m[c][tid_don] - m[c][tid]).to(dev)
            return x
        _, keepP = fwd(tm0, at=0, keeps=True)
        Xp = torch.cat([(keepP[5].cpu().double() - mu_)/sd_, torch.ones(B, 1)], 1)
        d_dec = ((Xp - Xb) @ W)[ntr:]
        d_orc = (root_cf - root_cl)[ntr:, :3]
        slp = float((d_dec*d_orc).sum()/(d_orc*d_orc).sum())
        cop = float((d_dec.flatten()@d_orc.flatten())/(d_dec.flatten().norm()*d_orc.flatten().norm()+1e-12))
        print(f"  P5 root probe L5: R2 {r2:.3f} (high-entropy half {r2h:.3f}); propagation slope {slp:.3f} corr {cop:.3f}")
