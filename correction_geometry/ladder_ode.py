"""Multi-path accessibility model (asvin's amendment):
dc_r/dt = eta * g_r * [w0 + (1-w0)*c_{r-1}(t)] * (1 - c_r),  c_0 = 1 (tokens).
Fit (eta, w0) per task on measured coverage curves; compare to the two limits
w0=1 (pure parallel race) and w0=0.02 (pure renewal/bootstrap)."""
import json
import torch
from pcfg2 import NTREE, V
from pcfg3 import soft_tables
from pcfg4 import dirichlet_tables, bp4

def marg(i, P):
    pi = torch.full((1, V), 1 / V, dtype=torch.double)
    for lev in range(1, 5):
        c = i >> (4 - lev)
        pi = torch.einsum("na,abc->nb", pi, P[lev-1]) if c % 2 == 0 else torch.einsum("na,abc->nc", pi, P[lev-1])
        pi = pi / pi.sum(-1, keepdim=True)
    return pi[0]

def coverage(run, P, tmax=6000):
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
    rows = [json.loads(l) for l in open(f"{run}/eval.jsonl") if json.loads(l)["step"] <= tmax]
    steps = [r["step"] for r in rows]
    g, cov = {}, {}
    for b in range(4):
        gb = sum(M[b])/len(M[b]) - sum(F[b])/len(F[b])
        if gb < 0.012: continue
        mg = sum(M[b])/len(M[b])
        g[b] = gb
        cov[b] = [max(0.0, min(1.0, (mg - r[f"ce_b{b}"]) / gb)) for r in rows]
    return steps, g, cov

def simulate(steps, g, eta, w0, dt=5.0):
    bs = sorted(g)
    c = {b: 0.0 for b in bs}
    out = {b: [] for b in bs}
    t, si = 0.0, 0
    T = steps[-1]
    while t <= T + dt:
        while si < len(steps) and steps[si] <= t:
            for b in bs: out[b].append(c[b])
            si += 1
        for b in bs:
            prereq = 1.0 if b == 0 else c.get(b - 1, 1.0)
            c[b] = min(1.0, c[b] + dt * eta * g[b] * (w0 + (1 - w0) * prereq) * (1 - c[b]))
        t += dt
    for b in bs:
        while len(out[b]) < len(steps): out[b].append(c[b])
    return out

for name, run, P in (("pcfg4b", "runs/pcfg4a0.05_dense_seed0", dirichlet_tables(0.05)),
                     ("pcfg2 ", "runs/pcfg2_dense_seed0", soft_tables(1e-9))):
    steps, g, cov = coverage(run, P)
    def err(eta, w0):
        sim = simulate(steps, g, eta, w0)
        return sum((sim[b][k] - cov[b][k])**2 for b in g for k in range(len(steps)))
    best = None
    for w0 in [0.02, 0.05] + [round(0.1*x, 2) for x in range(1, 11)]:
        lo, hi = 1e-4, 1.0
        for _ in range(40):
            m1, m2 = lo + (hi-lo)/3, hi - (hi-lo)/3
            if err(m1, w0) < err(m2, w0): hi = m2
            else: lo = m1
        e = err((lo+hi)/2, w0)
        if best is None or e < best[2]: best = (round((lo+hi)/2, 5), w0, e)
    eta_b, w0_b, e_b = best
    # the two limits with their own best eta
    def bestfix(w0):
        lo, hi = 1e-4, 1.0
        for _ in range(40):
            m1, m2 = lo + (hi-lo)/3, hi - (hi-lo)/3
            if err(m1, w0) < err(m2, w0): hi = m2
            else: lo = m1
        return err((lo+hi)/2, w0)
    npts = sum(len(cov[b]) for b in g)
    print(f"{name}: g={ {b: round(v,3) for b,v in g.items()} }")
    print(f"  MIXED  best: w0={w0_b:.2f} eta={eta_b:.4f}  rmse={ (e_b/npts)**.5:.4f}")
    print(f"  PARALLEL (w0=1):   rmse={(bestfix(1.0)/npts)**.5:.4f}")
    print(f"  RENEWAL  (w0=.02): rmse={(bestfix(0.02)/npts)**.5:.4f}")
    sim = simulate(steps, g, eta_b, w0_b)
    for b in sorted(g):
        def tq(series, q):
            for k in range(1, len(series)):
                if series[k] >= q and series[k-1] < q:
                    return steps[k]
            return None
        print(f"    b{b}: tau50 model {tq(sim[b],0.5)} vs measured {tq(cov[b],0.5)} | "
              f"tau80 {tq(sim[b],0.8)} vs {tq(cov[b],0.8)}")
