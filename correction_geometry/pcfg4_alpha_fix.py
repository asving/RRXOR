"""Corrected design sweep: CONTEXTUAL gain (marginalCE - floor) per class per alpha."""
import torch
from pcfg2 import V
from pcfg4 import dirichlet_tables, gen_tree4, bp4

def marg(i, P):
    pi = torch.full((1, V), 1 / V, dtype=torch.double)
    for lev in range(1, 5):
        child = i >> (4 - lev)
        pi = torch.einsum("na,abc->nb", pi, P[lev - 1]) if child % 2 == 0 else \
             torch.einsum("na,abc->nc", pi, P[lev - 1])
        pi = pi / pi.sum(-1, keepdim=True)
    return pi[0]

for alpha in (0.05, 0.1, 0.15, 0.3):
    P = dirichlet_tables(alpha)
    gt = torch.Generator().manual_seed(9)
    lv = gen_tree4(8192, P, gt)
    out = {}
    for i in range(1, 16):
        b = (i & -i).bit_length() - 1
        f = float(-bp4(lv, i, P).clamp_min(1e-12).log().gather(1, lv[:, i:i+1]).mean())
        m = float(-marg(i, P).clamp_min(1e-12).log()[lv[:, i]].mean())
        out.setdefault(b, []).append(m - f)
    print(f"alpha={alpha}: contextual gains " +
          "  ".join(f"b{b}={sum(v)/len(v):.4f}" for b, v in sorted(out.items())))
