"""P3 interchange (clean instrument), P4 dose-response linearity, P5 graded beliefs.
Same geometry as pcfg2_causal: tree 2, pair(4,5)=cols 36-37, query col 39 (leaf 8)."""
import glob
import torch
from train_multi import GPTV
from pcfg2 import V
from pcfg3 import soft_tables, DEPTH
P3 = soft_tables(0.05)
dev = "cuda"
ev = torch.load("runs/pcfg3_dense_seed0/evalset.pt")
seq = ev["eseq"][:2048].to(dev)
B, Q = seq.shape[0], 39
lv = seq[:, 32:48].cpu()

def bp_forced(force=None):
    lam = {4: torch.ones(B, 16, V, dtype=torch.double)}
    for j in range(8):
        lam[4][:, j] = 0.0
        lam[4][torch.arange(B), j, lv[:, j]] = 1.0
    for lev in range(4, 0, -1):
        lam[lev-1] = torch.einsum("abc,nib,nic->nia", P3[lev-1], lam[lev][:, 0::2], lam[lev][:, 1::2])
        if lev - 1 == 3 and force is not None:
            lam[3] = lam[3].clone(); lam[3][:, 2] = force
            for l2 in range(3, 0, -1):
                lam[l2-1] = torch.einsum("abc,nib,nic->nia", P3[l2-1], lam[l2][:, 0::2], lam[l2][:, 1::2])
            break
    pi = torch.full((B, V), 1/V, dtype=torch.double)
    root = pi * lam[0][:, 0]
    out = {"root": root / root.sum(-1, keepdim=True).clamp_min(1e-30)}
    tgt = 8
    for lev in range(1, 5):
        child = tgt >> (4 - lev); sib = child ^ 1
        ls = lam[lev][:, sib]
        pi = torch.einsum("na,abc,nc->nb", pi, P3[lev-1], ls) if child % 2 == 0 else \
             torch.einsum("na,abc,nb->nc", pi, P3[lev-1], ls)
        pi = pi / pi.sum(-1, keepdim=True).clamp_min(1e-30)
    out["pred"] = pi
    return out

lam_nat = P3[3][:, lv[:, 4], lv[:, 5]].T.permute(1, 0)  # wrong shape guard below
lam_nat = torch.stack([P3[3][:, int(lv[b, 4]), int(lv[b, 5])] for b in range(B)])
canon = lam_nat.max(-1).values / lam_nat.sum(-1) > 0.5
mapsym = lam_nat.argmax(-1)
g = torch.Generator().manual_seed(3)
donor = (mapsym + 1 + torch.randint(0, 3, (B,), generator=g)) % 4
from pcfg2 import TABLES
y = TABLES[3][donor]                                   # canonical yield of donor
lam_don = torch.stack([P3[3][:, int(y[b, 0, 0]), int(y[b, 0, 1])] for b in range(B)])
clean, cf = bp_forced(lam_nat), bp_forced(lam_don)
print(f"canonical fraction {float(canon.float().mean()):.3f} (all donors consistent by construction)")

model = GPTV(L=96, nl=6, vocab=V).to(dev)
ck = sorted(glob.glob("runs/pcfg3_dense_seed0/ckpts/*.pt"))
model.load_state_dict(torch.load(ck[-1], map_location=dev)); model.eval()
with torch.no_grad():
    _, resids = model(seq, return_resid=True)

def fwd(delta=None, at=2, keeps=False):
    x = model.wte(seq) + model.wpe(torch.arange(96, device=dev))[None]
    keep = {}
    for li, blk in enumerate(model.blocks):
        if delta is not None and li == at:
            x = x.clone()
            for ci, c in enumerate((36, 37)):
                x[:, c] += delta[ci]
        x = blk(x)
        keep[li] = x[:, 39].detach()
    lp = torch.log_softmax(model.head(model.lnf(x))[:, :, :V].double(), -1)
    return (lp, keep) if keeps else lp

lp0, keep0 = fwd(keeps=True)
lcl, lcf = clean["pred"].clamp_min(1e-9).log(), cf["pred"].clamp_min(1e-9).log()
mus = {c: torch.stack([resids[2][canon & (mapsym == v), c].mean(0) for v in range(V)]) for c in (36, 37)}
D = [ (mus[c][donor] - mus[c][mapsym]).to(dev) for c in (36, 37) ]

def score(lp, m, tag):
    dn = (lp[:, Q] - lp0[:, Q]).cpu()[m]; do = (lcf - lcl)[m]
    dn, do = dn - dn.mean(-1, keepdim=True), do - do.mean(-1, keepdim=True)
    sl = float((dn * do).sum() / (do * do).sum())
    co = float((dn.flatten() @ do.flatten()) / (dn.flatten().norm() * do.flatten().norm() + 1e-12))
    p = cf["pred"][m].clamp_min(1e-9)
    print(f"{tag}: slope {sl:.3f}  corr {co:.3f}  KL(BPcf||·) {float((p*(p.log()-lp[:,Q].cpu()[m])).sum(-1).mean()):.4f}")
    return sl

p = cf["pred"][canon].clamp_min(1e-9)
print(f"oracle gap (canon subset): {float((p*(p.log()-lp0[:,Q].cpu()[canon])).sum(-1).mean()):.4f}")
lpP, keepP = fwd(delta=D, keeps=True)
score(lpP, canon, "P3 interchange@L2 (ALL donors, canon-native)")

# P4 dose-response: delta scaled by alpha; oracle = BP at interpolated message
print("P4 dose-response (net slope vs BP at each alpha):")
bp_a, net_a = {}, {}
for a in (0.25, 0.5, 0.75, 1.0):
    lam_a = (1 - a) * lam_nat + a * lam_don
    cfa = bp_forced(lam_a)
    lcfa = cfa["pred"].clamp_min(1e-9).log()
    lpa = fwd(delta=[a * d for d in D])
    dn = (lpa[:, Q] - lp0[:, Q]).cpu()[canon]; do = (lcfa - lcl)[canon]
    dn, do = dn - dn.mean(-1, keepdim=True), do - do.mean(-1, keepdim=True)
    print(f"  alpha={a}: slope {float((dn*do).sum()/(do*do).sum()):.3f}  "
          f"corr {float((dn.flatten()@do.flatten())/(dn.flatten().norm()*do.flatten().norm()+1e-12)):.3f}")
    bp_a[a], net_a[a] = do.norm() / (lcf - lcl)[canon].sub((lcf-lcl)[canon].mean(-1,keepdim=True)).norm(), \
                        dn.norm() / (lcf - lcl)[canon].sub((lcf-lcl)[canon].mean(-1,keepdim=True)).norm()
print("  shift magnitude vs alpha (BP | net): " +
      "  ".join(f"a={a}: {float(bp_a[a]):.2f}|{float(net_a[a]):.2f}" for a in bp_a))

# P5: root-belief probe trained on canonical seqs, evaluated on NOISE-tuple seqs
ell = 5
X = keep0[ell].cpu().double()
tr = canon.clone(); tr[1024:] = False
mu_, sd_ = X[tr].mean(0), X[tr].std(0) + 1e-6
Xb = torch.cat([(X - mu_) / sd_, torch.ones(B, 1)], 1)
A = Xb[tr].T @ Xb[tr] + 10 * torch.eye(Xb.shape[1])
W = torch.linalg.solve(A, Xb[tr].T @ clean["root"][tr, :3])
for tag, m in (("canon held-out", canon & (torch.arange(B) >= 1024)),
               ("NOISE tuples   ", ~canon)):
    Yh = Xb[m] @ W; Yt = clean["root"][m, :3]
    r2 = float(1 - ((Yt - Yh) ** 2).sum() / ((Yt - clean["root"][tr, :3].mean(0)) ** 2).sum())
    print(f"P5 root probe on {tag} (n={int(m.sum())}): R2 vs soft-BP {r2:.3f}")
