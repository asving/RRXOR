"""Noisy-copy tree PCFG (broadcasting on trees): depth-5 binary trees, 32 leaves each,
3 independent trees per sequence (L=96). Node values in {0,1}; root uniform; each
child copies its parent w.p. 1-eta. Latents (internal nodes) NEVER appear on the tape.

Position classes by boundary depth b = nu_2(leaf index within tree): predicting leaf i
requires the level-b shared ancestor, inferred bottom-up (inverse Bayes) from the seen
prefix and propagated down (forward Bayes). Exact BP filter + level-truncated policies
(stage-l uses ancestors <= l levels up; prior at the cut) give the preregistered
class x stage CE ladder.
"""
import torch

ETA = 0.1
DEPTH = 5
NLEAF = 32          # per tree
NTREE = 3           # sequence length 96

def gen_tree_batch(B, g, eta=ETA):
    """Returns leaves (B, 32) and all node values dict level->(B, 2^level)."""
    nodes = {0: torch.randint(0, 2, (B, 1), generator=g)}
    for lev in range(1, DEPTH + 1):
        parent = nodes[lev - 1].repeat_interleave(2, dim=1)
        flip = (torch.rand(B, 2 ** lev, generator=g) < eta).long()
        nodes[lev] = parent ^ flip
    return nodes[DEPTH], nodes

def gen_batch_pcfg(B, g, eta=ETA):
    seqs, alln = [], []
    for _ in range(NTREE):
        lv, nodes = gen_tree_batch(B, g, eta)
        seqs.append(lv); alln.append(nodes)
    return torch.cat(seqs, 1), alln

def boundary_class(i):
    """Class of leaf index i (0..31) within its tree: first leaf = -1; else nu_2(i)."""
    if i == 0:
        return -1
    b = 0
    while i % 2 == 0:
        i //= 2; b += 1
    return b

CLS = [boundary_class(i) for i in range(NLEAF)]

def bp_predict(leaves, upto, eta=ETA, trunc=None):
    """P(leaf_upto = 1 | leaves[:, :upto]) by exact BP on one tree.
    trunc = stage l: only the subtree under the level-(DEPTH-l) ... i.e. use ancestors
    at most l levels above the target; evidence outside that subtree ignored; uniform
    prior at the subtree root. trunc=None -> full BP (l = DEPTH)."""
    B = leaves.shape[0]
    l = DEPTH if trunc is None else trunc
    if l == 0 or upto == 0:
        return torch.full((B,), 0.5, dtype=torch.double)
    # subtree root = ancestor of target at level DEPTH - l; leaves of that subtree:
    anc = upto >> l
    lo, hi = anc << l, ((anc + 1) << l)
    lo = max(lo, 0)
    seen = list(range(lo, min(upto, hi)))
    if not seen:
        return torch.full((B,), 0.5, dtype=torch.double)
    # BP inside the subtree (depth l, 2^l leaves): upward messages then downward.
    # message for node value v: m(v) proportional to P(evidence under node | v)
    flip = ETA if eta is None else eta
    def edge_pass(msg):
        # child message -> parent factor: sum_c P(c|p) m(c)
        m0, m1 = msg[..., 0], msg[..., 1]
        return torch.stack([(1 - flip) * m0 + flip * m1,
                            flip * m0 + (1 - flip) * m1], -1)
    nlv = {l: torch.ones(B, 2 ** l, 2, dtype=torch.double)}
    for j in seen:
        jj = j - lo
        obs = leaves[:, j]
        nlv[l][torch.arange(B), jj, 0] = (obs == 0).double()
        nlv[l][torch.arange(B), jj, 1] = (obs == 1).double()
    for lev in range(l, 0, -1):
        up = edge_pass(nlv[lev])
        nlv[lev - 1] = up[:, 0::2] * up[:, 1::2]
    # downward to target leaf (index upto - lo within subtree), proper BP:
    # belief at path node = message-from-parent-excluding-this-child x evidence-below
    tgt = upto - lo
    belief = nlv[0][:, 0] / nlv[0][:, 0].sum(-1, keepdim=True).clamp_min(1e-30)
    for lev in range(1, l + 1):
        child = (tgt >> (l - lev))
        below_c = nlv[lev][:, child]
        up_c = edge_pass(below_c)
        excl = (belief / up_c.clamp_min(1e-30))
        excl = excl / excl.sum(-1, keepdim=True).clamp_min(1e-30)
        down = edge_pass(excl)
        m = down * below_c
        belief = m / m.sum(-1, keepdim=True).clamp_min(1e-30)
    return belief[:, 1]

def class_ce_matrix(nsamp=8192, seed=0, eta=ETA):
    """CE[stage l, class b] over one tree, Monte Carlo with exact per-sample BP."""
    g = torch.Generator().manual_seed(seed)
    leaves, _ = gen_tree_batch(nsamp, g, eta)
    import collections
    out = {}
    for l in range(0, DEPTH + 1):
        agg = collections.defaultdict(list)
        for i in range(NLEAF):
            b = CLS[i]
            p = bp_predict(leaves, i, eta, trunc=None if l == DEPTH else l)
            p = p.clamp(1e-6, 1 - 1e-6)
            y = leaves[:, i].double()
            ce = -(y * torch.log(p) + (1 - y) * torch.log(1 - p))
            agg[b].append(float(ce.mean()))
        out[l] = {b: sum(v) / len(v) for b, v in agg.items()}
    return out

if __name__ == "__main__":
    import json
    M = class_ce_matrix()
    print("class x stage CE ladder (rows = stage l = levels used; cols = boundary class b):")
    print("      first    b=0      b=1      b=2      b=3      b=4")
    for l in sorted(M):
        row = M[l]
        print(f"l={l}:  " + "  ".join(f"{row[b]:.4f}" for b in [-1, 0, 1, 2, 3, 4]))
    json.dump({str(l): {str(b): v for b, v in M[l].items()} for l in M},
              open("pcfg_ladder.json", "w"))
