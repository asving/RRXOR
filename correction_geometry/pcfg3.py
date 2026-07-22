"""PCFG-3: RHM with FULL-SUPPORT probabilistic productions.
P(b,c|a) = (1-eps) * (1/M) sum_k 1[(b,c)=prod_k(a)] + eps/16.
Same trees/classes as pcfg2; soft BP with 4-vector messages."""
import torch
from pcfg2 import V, M, DEPTH, NLEAF, NTREE, GSEED, TABLES

EPS = None          # set by caller before use
def soft_tables(eps):
    P3 = []
    for lev in range(DEPTH):
        t = torch.full((V, V, V), eps / (V * V), dtype=torch.double)
        for a in range(V):
            for k in range(M):
                b, c = int(TABLES[lev][a, k, 0]), int(TABLES[lev][a, k, 1])
                t[a, b, c] += (1 - eps) / M
        P3.append(t)
    return P3

def gen_tree(B, P3, g):
    x = torch.randint(0, V, (B, 1), generator=g)
    for lev in range(DEPTH):
        flat = P3[lev].reshape(V, V * V)
        idx = torch.multinomial(flat[x.reshape(-1)], 1, generator=g).reshape(B, -1)
        x = torch.stack([idx // V, idx % V], -1).reshape(B, -1)
    return x

def gen_batch(B, L, P3, g):
    return torch.cat([gen_tree(B, P3, g) for _ in range(NTREE)], 1)[:, :L]

def bp_predict(lv, upto, P3, want_path=False):
    B = lv.shape[0]
    lam = {DEPTH: torch.ones(B, NLEAF, V, dtype=torch.double)}
    for j in range(upto):
        lam[DEPTH][:, j] = 0.0
        lam[DEPTH][torch.arange(B), j, lv[:, j]] = 1.0
    for lev in range(DEPTH, 0, -1):
        lamL, lamR = lam[lev][:, 0::2], lam[lev][:, 1::2]
        lam[lev - 1] = torch.einsum("abc,nib,nic->nia", P3[lev - 1], lamL, lamR)
    pi = torch.full((B, V), 1 / V, dtype=torch.double)
    path = []
    for lev in range(1, DEPTH + 1):
        child = upto >> (DEPTH - lev)
        sib, is_left = child ^ 1, (upto >> (DEPTH - lev)) % 2 == 0
        t = P3[lev - 1]
        ls = lam[lev][:, sib]
        if is_left:
            pi = torch.einsum("na,abc,nc->nb", pi, t, ls)
        else:
            pi = torch.einsum("na,abc,nb->nc", pi, t, ls)
        pi = pi / pi.sum(-1, keepdim=True).clamp_min(1e-30)
        if want_path and lev < DEPTH:
            pb = pi * lam[lev][:, child]
            path.append(pb / pb.sum(-1, keepdim=True).clamp_min(1e-30))
    return (pi, path) if want_path else pi

def brute(lv, upto, P3, depth):
    """enumeration check at small depth"""
    nl = 2 ** depth
    B = lv.shape[0]
    seqs = torch.cartesian_prod(*[torch.arange(V)] * nl).reshape(-1, nl)
    lp = torch.zeros(seqs.shape[0], dtype=torch.double)
    x = seqs
    probs = torch.full((seqs.shape[0],), 1 / V, dtype=torch.double)
    # p(leaves): sum over internal via BP on full tree (reuse lam with upto=nl)
    lam = torch.zeros(seqs.shape[0], nl, V, dtype=torch.double)
    lam[torch.arange(seqs.shape[0]).repeat_interleave(nl),
        torch.arange(nl).repeat(seqs.shape[0]), seqs.flatten()] = 1.0
    for lev in range(depth, 0, -1):
        lam = torch.einsum("abc,nib,nic->nia", P3[lev - 1], lam[:, 0::2], lam[:, 1::2])
    pseq = (lam[:, 0] / V).sum(-1)
    out = torch.zeros(B, V, dtype=torch.double)
    for b in range(B):
        m = (seqs[:, :upto] == lv[b, :upto]).all(-1)
        for v in range(V):
            out[b, v] = pseq[m & (seqs[:, upto] == v)].sum()
    return out / out.sum(-1, keepdim=True)

if __name__ == "__main__":
    g = torch.Generator().manual_seed(1)
    # validate soft BP vs enumeration at depth 2 (temporarily shrink)
    import pcfg2, sys
    for eps in (0.1, 0.2):
        P3full = soft_tables(eps)
        P3d2 = P3full[:2]
        globals()["DEPTH"] = 2
        lv = gen_tree(64, P3d2, g)  # depth-2 gen: needs DEPTH... use manual
        x = torch.randint(0, V, (64, 1), generator=g)
        for lev in range(2):
            flat = P3d2[lev].reshape(V, V * V)
            idx = torch.multinomial(flat[x.reshape(-1)], 1, generator=g).reshape(64, -1)
            x = torch.stack([idx // V, idx % V], -1).reshape(64, -1)
        lv = x
        errs = []
        for upto in (1, 2, 3):
            lam = {2: torch.ones(64, 4, V, dtype=torch.double)}
            for j in range(upto):
                lam[2][:, j] = 0.0
                lam[2][torch.arange(64), j, lv[:, j]] = 1.0
            for lev in (2, 1):
                lam[lev-1] = torch.einsum("abc,nib,nic->nia", P3d2[lev-1], lam[lev][:,0::2], lam[lev][:,1::2])
            pi = torch.full((64, V), 1/V, dtype=torch.double)
            for lev in (1, 2):
                child = upto >> (2 - lev); sib = child ^ 1; il = child % 2 == 0
                ls = lam[lev][:, sib]
                pi = torch.einsum("na,abc,nc->nb", pi, P3d2[lev-1], ls) if il else \
                     torch.einsum("na,abc,nb->nc", pi, P3d2[lev-1], ls)
                pi = pi / pi.sum(-1, keepdim=True).clamp_min(1e-30)
            bt = brute(lv, upto, P3d2, 2)
            errs.append(float((pi - bt).abs().max()))
        print(f"eps={eps}: BP-vs-enumeration max|err| = {max(errs):.2e}")
    # class x stage matrix per eps (MC, depth 4)
    globals()["DEPTH"] = 4
    for eps in (0.05, 0.1, 0.2):
        P3full = soft_tables(eps)
        gt = torch.Generator().manual_seed(9)
        lv = gen_tree(8192, P3full, gt)
        row = {}
        for i in range(1, 16):
            b = -1 if i == 0 else (i & -i).bit_length() - 1
            pred = bp_predict(lv, i, P3full)
            ce = float(-pred.clamp_min(1e-12).log().gather(1, lv[:, i:i+1]).mean())
            row.setdefault(b, []).append(ce)
        print(f"eps={eps}: floors " + "  ".join(
            f"b{b}={sum(v)/len(v):.4f}" for b, v in sorted(row.items())))
