"""Part 1: proper pi-weighted SDPI constants per tree edge, and the path-dilution
prediction for the measured g-ratios (pcfg4b, alpha=0.05).
Edge operator B[x,y] = sqrt(pi_par[x]) K(y|x) / sqrt(pi_child[y]); lambda1 must be
~1; eta = lambda2^2. Maximal correlation is direction-symmetric, so up and down
edges use the same eta."""
import torch
from pcfg2 import V
from pcfg4 import dirichlet_tables

P = dirichlet_tables(0.05)
# node marginals: nodes indexed (level, idx); level 0 = root
pi = {(0, 0): torch.full((V,), 0.25, dtype=torch.double)}
eta = {}
for lev in range(4):
    KL, KR = P[lev].sum(2), P[lev].sum(1)          # P(left|a), P(right|a)
    for idx in [k for (l, k) in pi if l == lev]:
        par = pi[(lev, idx)]
        for side, K in (("L", KL), ("R", KR)):
            child = 2 * idx + (0 if side == "L" else 1)
            pc = par @ K
            B = torch.diag(par.sqrt()) @ K @ torch.diag(pc.clamp_min(1e-12).rsqrt())
            s = torch.linalg.svdvals(B)
            assert abs(float(s[0]) - 1.0) < 1e-6, f"lambda1={float(s[0])}"
            eta[(lev + 1, child)] = float(s[1]) ** 2
            pi[(lev + 1, child)] = pc
lev_eta = {l: [eta[(l, i)] for i in range(2 ** l)] for l in range(1, 5)}
for l in range(1, 5):
    print(f"level {l}: eta (lambda2^2) = " + " ".join(f"{e:.3f}" for e in lev_eta[l][:4])
          + (" ..." if l > 2 else ""))

def path_edges(i):
    """edges on evidence(i-1) -> shared ancestor -> target(i), leaf indices in-tree"""
    b = (i & -i).bit_length() - 1
    anc_lev = 4 - (b + 1)
    up = [(l, (i - 1) >> (4 - l)) for l in range(anc_lev + 1, 5)]
    dn = [(l, i >> (4 - l)) for l in range(anc_lev + 1, 5)]
    return up + dn
D = {}
for b in range(4):
    i = 2 ** b
    prod = 1.0
    for e in path_edges(i):
        prod *= eta[e]
    D[b] = prod
print("\npath-dilution D(b) and predicted vs measured g-ratios:")
meas = {0: 0.5261, 1: 0.2127, 2: 0.1421, 3: 0.0582}
for b in range(3):
    print(f"  b{b}->b{b+1}: predicted D ratio {D[b]/D[b+1]:6.2f}   measured g ratio {meas[b]/meas[b+1]:5.2f}")
