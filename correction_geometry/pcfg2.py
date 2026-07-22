"""PCFG-2: Random Hierarchy Model (leap-hard). Binary tree, depth D=4, vocab v=4,
m=2 productions per symbol per level, random unambiguous tables (each child tuple has
a unique parent). 6 trees per sequence -> L=96, token vocab 4. Latents tape-invisible
AND compositionally deep: parent = random lookup of the child tuple; level-l parse =
l-fold composition of random tables.
"""
import torch

V, M, DEPTH, NLEAF, NTREE = 4, 2, 4, 16, 6
GSEED = 7

def make_grammar(seed=GSEED):
    g = torch.Generator().manual_seed(seed)
    tables = []
    for _ in range(DEPTH):
        perm = torch.randperm(V * V, generator=g)[:V * M]
        tup = torch.stack([perm // V, perm % V], 1)          # (V*M, 2) distinct tuples
        tab = tup.view(V, M, 2)                              # parent, choice -> (L, R)
        tables.append(tab)
    return tables

TABLES = make_grammar()

def gen_batch_tree(B, g):
    sym = torch.randint(0, V, (B, 1), generator=g)
    for lev in range(DEPTH):
        choice = torch.randint(0, M, sym.shape, generator=g)
        prod = TABLES[lev][sym, choice]                      # (B, n, 2)
        sym = prod.reshape(B, -1)
    return sym                                               # (B, 16)

def gen_batch_pcfg2(B, g):
    return torch.cat([gen_batch_tree(B, g) for _ in range(NTREE)], 1)

def boundary_class(i):
    if i == 0:
        return -1
    b = 0
    while i % 2 == 0:
        i //= 2; b += 1
    return b

CLS = [boundary_class(i) for i in range(NLEAF)]

def bp_predict(leaves, upto, trunc=None, want_path=False):
    """P(leaf_upto = x | leaves[:, :upto]) exact BP; trunc = stage l (ancestors <= l
    levels up, uniform prior at the cut). want_path -> also ancestor posteriors."""
    B = leaves.shape[0]
    l = DEPTH if trunc is None else trunc
    if l == 0 or upto == 0:
        out = torch.full((B, V), 1 / V, dtype=torch.double)
        return (out, []) if want_path else out
    anc = upto >> l
    lo = anc << l
    # lambda (upward) per level within the subtree
    lam = {l: torch.ones(B, 2 ** l, V, dtype=torch.double)}
    for j in range(lo, min(upto, lo + 2 ** l)):
        jj = j - lo
        lam[l][:, jj] = 0.0
        lam[l][torch.arange(B), jj, leaves[:, j]] = 1.0
    for lev in range(l, 0, -1):
        glev = DEPTH - l + lev                               # global table level index
        tab = TABLES[glev - 1]                               # parents at glev-1
        lamL, lamR = lam[lev][:, 0::2], lam[lev][:, 1::2]    # (B, n, V)
        up = torch.zeros(B, lamL.shape[1], V, dtype=torch.double)
        for a in range(V):
            for k in range(M):
                Lc, Rc = int(tab[a, k, 0]), int(tab[a, k, 1])
                up[:, :, a] += lamL[:, :, Lc] * lamR[:, :, Rc] / M
        lam[lev - 1] = up
    # pi (downward) along the path to the target leaf
    tgt = upto - lo
    pi = torch.full((B, V), 1 / V, dtype=torch.double)       # prior at subtree root
    path_post = []
    for lev in range(1, l + 1):
        glev = DEPTH - l + lev
        tab = TABLES[glev - 1]
        node_parent = tgt >> (l - lev + 1)
        child = tgt >> (l - lev)
        is_left = (child % 2 == 0)
        sib = child ^ 1
        bel_par = pi * lam[lev - 1][:, node_parent] if lev > 1 else pi * lam[0][:, 0]
        # parent belief EXCLUDING the child's own subtree cannot be a division here;
        # recompute: pi already excludes; use pi (excl) with sibling lambda directly:
        newpi = torch.zeros(B, V, dtype=torch.double)
        lam_sib = lam[lev][:, sib]
        for a in range(V):
            for k in range(M):
                Lc, Rc = int(tab[a, k, 0]), int(tab[a, k, 1])
                cval, sval = (Lc, Rc) if is_left else (Rc, Lc)
                newpi[:, cval] += pi[:, a] * lam_sib[:, sval] / M
        pi = newpi / newpi.sum(-1, keepdim=True).clamp_min(1e-30)
        if want_path:
            post = pi * lam[lev][:, child]
            s = post.sum(-1, keepdim=True).clamp_min(1e-30)
            path_post.append(post / s)
    out = pi                                                  # target leaf unseen
    return (out, path_post) if want_path else out

def class_ce_matrix(nsamp=4096, seed=0):
    import collections
    g = torch.Generator().manual_seed(seed)
    leaves = gen_batch_tree(nsamp, g)
    out = {}
    for l in range(0, DEPTH + 1):
        agg = collections.defaultdict(list)
        for i in range(NLEAF):
            p = bp_predict(leaves, i, trunc=None if l == DEPTH else l)
            p = p.clamp_min(1e-9)
            ce = -torch.log(p[torch.arange(nsamp), leaves[:, i]])
            agg[CLS[i]].append(float(ce.mean()))
        out[l] = {b: sum(v) / len(v) for b, v in agg.items()}
    return out

if __name__ == "__main__":
    import json, itertools
    # brute-force validation at D=2 (monkeypatch)
    import sys
    mod = sys.modules[__name__]
    savedD, savedT, savedN = DEPTH, TABLES, NLEAF
    mod.DEPTH, mod.NLEAF = 2, 4
    mod.TABLES = savedT[2:]                                   # use last 2 tables
    g = torch.Generator().manual_seed(1)
    lv = gen_batch_tree(32, g)
    worst = 0.0
    for k in (1, 2, 3):
        pbp = bp_predict(lv, k)
        for row in range(32):
            probs = torch.zeros(V, dtype=torch.double)
            for root in range(V):
                for k1 in range(M):
                    L1, R1 = mod.TABLES[0][root, k1]
                    for kL in range(M):
                        for kR in range(M):
                            l4 = [int(mod.TABLES[1][L1, kL, 0]), int(mod.TABLES[1][L1, kL, 1]),
                                  int(mod.TABLES[1][R1, kR, 0]), int(mod.TABLES[1][R1, kR, 1])]
                            w = 1 / (V * M ** 3)
                            if all(l4[j] == int(lv[row, j]) for j in range(k)):
                                probs[l4[k]] += w
            probs = probs / probs.sum()
            worst = max(worst, float((pbp[row] - probs).abs().max()))
    print("brute-force validation (D=2): max |BP - enum| =", round(worst, 10))
    mod.DEPTH, mod.TABLES, mod.NLEAF = savedD, savedT, savedN
    M_ = class_ce_matrix()
    print("class x stage CE ladder:")
    print("      first    b=0      b=1      b=2      b=3")
    for l in sorted(M_):
        print(f"l={l}:  " + "  ".join(f"{M_[l][b]:.4f}" for b in [-1, 0, 1, 2, 3]))
    W = {-1: 1, 0: 8, 1: 4, 2: 2, 3: 1}
    for l in sorted(M_):
        avg = sum(W[b] * v for b, v in M_[l].items()) / 16
        print(f"stage l={l}: avg CE = {avg:.4f}")
    json.dump({str(l): {str(b): v for b, v in M_[l].items()} for l in M_},
              open("pcfg2_ladder.json", "w"))
