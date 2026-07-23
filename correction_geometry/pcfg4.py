"""PCFG-4: fully entropic random tree process. Per level, per parent symbol a:
P(b,c|a) ~ Dirichlet(alpha * 1_16) — full support, genuinely graded (not
near-one-hot), tuple-coupled (no factorization through per-child terms).
Generic-depth BP included and used everywhere (validated vs enumeration)."""
import numpy as np
import torch
from pcfg2 import V, NTREE

ALPHA, GSEED4, DEPTH4 = 0.3, 11, 4       # alpha chosen by design sweep (see PCFG4_DESIGN.md)

def dirichlet_tables(alpha, depth=DEPTH4, gseed=GSEED4):
    rng = np.random.default_rng(gseed)
    return [torch.tensor(rng.dirichlet(alpha * np.ones(V * V), size=V),
                         dtype=torch.double).reshape(V, V, V) for _ in range(depth)]

def gen_tree4(B, P, g):
    x = torch.randint(0, V, (B, 1), generator=g)
    for lev in range(len(P)):
        flat = P[lev].reshape(V, V * V)
        idx = torch.multinomial(flat[x.reshape(-1)], 1, generator=g).reshape(B, -1)
        x = torch.stack([idx // V, idx % V], -1).reshape(B, -1)
    return x

def gen_batch4(B, P, g):
    return torch.cat([gen_tree4(B, P, g) for _ in range(NTREE)], 1)

def bp4(lv, upto, P, want_path=False):
    depth = len(P)
    B = lv.shape[0]
    lam = {depth: torch.ones(B, 2 ** depth, V, dtype=torch.double)}
    for j in range(upto):
        lam[depth][:, j] = 0.0
        lam[depth][torch.arange(B), j, lv[:, j]] = 1.0
    for lev in range(depth, 0, -1):
        lam[lev - 1] = torch.einsum("abc,nib,nic->nia",
                                    P[lev - 1], lam[lev][:, 0::2], lam[lev][:, 1::2])
    pi = torch.full((B, V), 1 / V, dtype=torch.double)
    path = []
    for lev in range(1, depth + 1):
        child = upto >> (depth - lev)
        sib = child ^ 1
        ls = lam[lev][:, sib]
        pi = torch.einsum("na,abc,nc->nb", pi, P[lev - 1], ls) if child % 2 == 0 else \
             torch.einsum("na,abc,nb->nc", pi, P[lev - 1], ls)
        pi = pi / pi.sum(-1, keepdim=True).clamp_min(1e-30)
        if want_path and lev < depth:
            pb = pi * lam[lev][:, child]
            path.append(pb / pb.sum(-1, keepdim=True).clamp_min(1e-30))
    return (pi, path) if want_path else pi

if __name__ == "__main__":
    from pcfg3 import brute
    g = torch.Generator().manual_seed(1)
    for alpha in (0.15, 0.3, 0.6, 1.0):
        P = dirichlet_tables(alpha)
        # design stats: gradedness of the single-level inversion (uniform prior)
        joint = P[3] / V
        p_bc = joint.sum(0)
        m2 = float((p_bc * (joint / joint.sum(0).clamp_min(1e-30)).max(0).values).sum())
        j_b = joint.sum(2)
        m1 = float((j_b.sum(0) * (j_b / j_b.sum(0).clamp_min(1e-30)).max(0).values).sum())
        # validation at depth 2
        P2 = dirichlet_tables(alpha, depth=2)
        lv = gen_tree4(64, P2, g)
        err = max(float((bp4(lv, u, P2) - brute(lv, u, P2, 2)).abs().max()) for u in (1, 2, 3))
        # floors at depth 4 (MC)
        gt = torch.Generator().manual_seed(9)
        lv4 = gen_tree4(8192, P, gt)
        row = {}
        for i in range(1, 16):
            b = (i & -i).bit_length() - 1
            ce = float(-bp4(lv4, i, P).clamp_min(1e-12).log().gather(1, lv4[:, i:i+1]).mean())
            row.setdefault(b, []).append(ce)
        fl = {b: sum(v) / len(v) for b, v in row.items()}
        print(f"alpha={alpha}: maxP(a|b,c)={m2:.3f} maxP(a|b)={m1:.3f} bp-err={err:.1e}  "
              + "  ".join(f"b{b}={fl[b]:.4f}(gain {1.3863-fl[b]:.3f})" for b in sorted(fl)))
