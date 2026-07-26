"""Test asvin's geometric-dilution learning model on the PCFG family traces.
Per task: per-class contextual gain g_b (marginalCE - floor, evalset-exact) and
convergence times tau_b^q (contextual coverage crossing q, linear interp).
Fits: log tau ~ -kappa log g (per-position) and ~ -kappa' log G (drive: count x g).
Sequentiality: onset (q=.2) of class b vs completion (q=.8) of class b-1."""
import json, math
import torch
from pcfg2 import NTREE, V
from pcfg3 import soft_tables
from pcfg4 import dirichlet_tables, bp4

TASKS = {
    "pcfg2 (det RHM)":  ("runs/pcfg2_dense_seed0",    soft_tables(1e-9)),
    "pcfg3 (eps.05)":   ("runs/pcfg3_dense_seed0",    soft_tables(0.05)),
    "pcfg4 (a=0.3)":    ("runs/pcfg4_dense_seed0",    dirichlet_tables(0.3)),
    "pcfg4b (a=0.05)":  ("runs/pcfg4a0.05_dense_seed0", dirichlet_tables(0.05)),
}
def marg(i, P):
    pi = torch.full((1, V), 1 / V, dtype=torch.double)
    for lev in range(1, 5):
        c = i >> (4 - lev)
        pi = torch.einsum("na,abc->nb", pi, P[lev-1]) if c % 2 == 0 else torch.einsum("na,abc->nc", pi, P[lev-1])
        pi = pi / pi.sum(-1, keepdim=True)
    return pi[0]

PTS = []
for name, (run, P) in TASKS.items():
    ev = torch.load(f"{run}/evalset.pt")["eseq"]
    F, M = {}, {}
    for i in range(1, 16):
        b = (i & -i).bit_length() - 1
        fs, ms = [], []
        for t in range(NTREE):
            lv = ev[:, 16*t:16*(t+1)]
            fs.append(-bp4(lv, i, P).clamp_min(1e-12).log().gather(1, lv[:, i:i+1]))
            ms.append(-marg(i, P).clamp_min(1e-12).log()[lv[:, i]])
        F.setdefault(b, []).append(float(torch.cat(fs).mean()))
        M.setdefault(b, []).append(float(torch.cat(ms).mean()))
    rows = [json.loads(l) for l in open(f"{run}/eval.jsonl")]
    steps = [r["step"] for r in rows]
    print(f"\n=== {name} ===")
    print("b |    g_b   count  G_b   | tau20   tau50   tau80  (censored if <=100)")
    for b in range(4):
        g = sum(M[b])/len(M[b]) - sum(F[b])/len(F[b])
        if g < 0.012:
            print(f"{b} | {g:7.4f}  (no instrument, skipped)")
            continue
        fl, mg = sum(F[b])/len(F[b]), sum(M[b])/len(M[b])
        cov = [max(0.0, min(1.0, (mg - r[f"ce_b{b}"]) / g)) for r in rows]
        taus = {}
        for q in (0.2, 0.5, 0.8):
            tau = None
            for k in range(1, len(cov)):
                if cov[k] >= q and cov[k-1] < q:
                    f = (q - cov[k-1]) / (cov[k] - cov[k-1] + 1e-12)
                    tau = steps[k-1] + f * (steps[k] - steps[k-1]); break
            if tau is None and cov[0] >= q: tau = steps[0] if steps[0] > 0 else 50.0
            taus[q] = tau
        cnt = NTREE * (8 >> b)
        G = cnt * g
        cens = "*" if (taus[0.5] or 1e9) <= 100 else " "
        print(f"{b} | {g:7.4f}  {cnt:3d}  {G:6.2f} | "
              + "  ".join(f"{taus[q]:7.0f}" if taus[q] else "  never" for q in (0.2, 0.5, 0.8)) + cens)
        if taus[0.5] and taus[0.5] > 100:
            PTS.append((name, b, g, G, taus[0.5], taus[0.2], taus[0.8]))

def fit(xs, ys):
    n = len(xs); mx, my = sum(xs)/n, sum(ys)/n
    sl = sum((x-mx)*(y-my) for x, y in zip(xs, ys)) / sum((x-mx)**2 for x in xs)
    r = sl * math.sqrt(sum((x-mx)**2 for x in xs) / sum((y-my)**2 for y in ys))
    return sl, r
print("\n--- power-law fits (log tau50 vs log g / log G), uncensored points ---")
for tag, xi in (("g (per-position)", 2), ("G (count x g)", 3)):
    for name in TASKS:
        pts = [p for p in PTS if p[0] == name]
        if len(pts) >= 3:
            sl, r = fit([math.log(p[xi]) for p in pts], [math.log(p[4]) for p in pts])
            print(f"  {name:18s} vs {tag:17s}: kappa = {-sl:.2f}  (r={r:+.3f}, n={len(pts)})")
    xs = [math.log(p[xi]) for p in PTS]; ys = [math.log(p[4]) for p in PTS]
    sl, r = fit(xs, ys)
    print(f"  POOLED (no offsets)  vs {tag:17s}: kappa = {-sl:.2f}  (r={r:+.3f}, n={len(PTS)})")
print("\n--- sequentiality (renewal vs parallel): onset_b (tau20) vs done_{b-1} (tau80) ---")
for name in TASKS:
    pts = sorted([p for p in PTS if p[0] == name], key=lambda p: p[1])
    for i in range(1, len(pts)):
        print(f"  {name}: b{pts[i][1]} onset {pts[i][5]:.0f} vs b{pts[i-1][1]} done {pts[i-1][6]:.0f} "
              f"-> {'RENEWAL-consistent' if pts[i][5] >= 0.8*pts[i-1][6] else 'PARALLEL (onset precedes completion)'}")
