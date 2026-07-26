"""Part 2: derivative-level test of the master equation.
Probe-based latent coverage c_r(t) (order-r ancestor posteriors decoded at the
b3 query col, layer 5) across checkpoints, plus behavioral coverage from
eval.jsonl. Then regress y_r(t) = c_r'/(1-c_r) on c_{r-1}(t):
H_parallel: y_r = const. H_multipath: y_r = a + b*c_{r-1}, b>0."""
import glob, json
import torch
from train_multi import GPTV
from pcfg2 import V, NTREE
from pcfg4 import dirichlet_tables

P = dirichlet_tables(0.05)
dev = "cuda"
ev = torch.load("runs/pcfg4a0.05_dense_seed0/evalset.pt")
seq = ev["eseq"][:1024].to(dev)
B = seq.shape[0]
lv = seq[:, 32:48].cpu()
# BP targets at query col 39 (predict leaf 8): ancestors order 1..3 + root (order 4)
lam = {4: torch.ones(B, 16, V, dtype=torch.double)}
for j in range(8):
    lam[4][:, j] = 0.0
    lam[4][torch.arange(B), j, lv[:, j]] = 1.0
for lev in range(4, 0, -1):
    lam[lev-1] = torch.einsum("abc,nib,nic->nia", P[lev-1], lam[lev][:, 0::2], lam[lev][:, 1::2])
pi = torch.full((B, V), 1/V, dtype=torch.double)
root = pi * lam[0][:, 0]
TARG = {4: root / root.sum(-1, keepdim=True).clamp_min(1e-30)}
for lev in range(1, 4):
    child = 8 >> (4 - lev); sib = child ^ 1
    ls = lam[lev][:, sib]
    pi = torch.einsum("na,abc,nc->nb", pi, P[lev-1], ls) if child % 2 == 0 else \
         torch.einsum("na,abc,nb->nc", pi, P[lev-1], ls)
    pi = pi / pi.sum(-1, keepdim=True).clamp_min(1e-30)
    pb = pi * lam[lev][:, child]
    TARG[4 - lev] = pb / pb.sum(-1, keepdim=True).clamp_min(1e-30)   # order 4-lev

model = GPTV(L=96, nl=6, vocab=V).to(dev)
cks = sorted(glob.glob("runs/pcfg4a0.05_dense_seed0/ckpts/*.pt"))
steps = [int(c.split("ckpt")[-1][:6]) for c in cks]
use = [k for k, s in enumerate(steps) if 100 <= s <= 3000 and s % 100 == 0] + [len(cks) - 1]
ntr = B // 2
def probe_r2(X, Y):
    mu, sd = X[:ntr].mean(0), X[:ntr].std(0) + 1e-6
    Xb = torch.cat([(X - mu) / sd, torch.ones(B, 1)], 1)
    A = Xb[:ntr].T @ Xb[:ntr] + 10 * torch.eye(Xb.shape[1])
    W = torch.linalg.solve(A, Xb[:ntr].T @ Y[:ntr])
    return float(1 - ((Y[ntr:] - Xb[ntr:] @ W) ** 2).sum() / ((Y[ntr:] - Y[:ntr].mean(0)) ** 2).sum())
CURVES = {r: [] for r in (1, 2, 3, 4)}
TS = []
for k in use:
    model.load_state_dict(torch.load(cks[k], map_location=dev)); model.eval()
    with torch.no_grad():
        _, res = model(seq, return_resid=True)
    X = res[5][:, 39].cpu().double()
    for r in (1, 2, 3, 4):
        CURVES[r].append(max(0.0, probe_r2(X, TARG[r][:, :3])))
    TS.append(steps[k])
final = {r: CURVES[r][-1] for r in CURVES}
print("probe R2 final (order 1..4):", " ".join(f"{final[r]:.3f}" for r in (1, 2, 3, 4)))
cpro = {r: [min(1.0, v / max(final[r], 1e-6)) for v in c[:-1]] for r, c in CURVES.items()}
tpro = TS[:-1]

rows = [json.loads(l) for l in open("runs/pcfg4a0.05_dense_seed0/eval.jsonl") if json.loads(l)["step"] <= 3000]
gB = {0: 0.5261, 1: 0.2127, 2: 0.1421, 3: 0.0582}
MRG = {0: 0.9384, 1: 0.9654, 2: 1.2610, 3: 1.2378}
cbeh = {b: [max(0.0, min(1.0, (MRG[b] - r[f"ce_b{b}"]) / gB[b])) for r in rows] for b in gB}
tbeh = [r["step"] for r in rows]

def smooth(v):
    return [v[max(0, i-1):i+2] and sum(v[max(0, i-1):i+2]) / len(v[max(0, i-1):i+2]) for i in range(len(v))]
def deriv_reg(c_r, c_prev, ts, tag):
    cs, cp = smooth(c_r), smooth(c_prev)
    ys, xs = [], []
    for i in range(1, len(cs) - 1):
        if 0.05 < cs[i] < 0.92:
            dc = (cs[i+1] - cs[i-1]) / (ts[i+1] - ts[i-1])
            ys.append(dc / max(1 - cs[i], 0.08)); xs.append(cp[i])
    if len(ys) < 5:
        print(f"  {tag}: n={len(ys)} too few"); return
    n = len(ys); mx, my = sum(xs)/n, sum(ys)/n
    vx = sum((x-mx)**2 for x in xs); vy = sum((y-my)**2 for y in ys)
    b = sum((x-mx)*(y-my) for x, y in zip(xs, ys)) / max(vx, 1e-12)
    r = b * (vx / max(vy, 1e-12)) ** 0.5
    ss_res = sum((y - (my + b*(x-mx)))**2 for x, y in zip(xs, ys))
    print(f"  {tag}: n={n}  slope b={b:+.2e}  corr(y, c_prev)={r:+.2f}  "
          f"{'MULTIPATH (b>0)' if b > 0 and abs(r) > 0.5 else 'parallel-consistent (no positive c_prev dependence)'}")

print("\nderivative test y_r = dc_r/dt/(1-c_r) vs c_{r-1}(t):")
print(" behavioral (classes b1..b3 on b0..b2):")
for b in (1, 2, 3):
    deriv_reg(cbeh[b], cbeh[b-1], tbeh, f"b{b} on b{b-1}")
print(" probe-based (orders 2..4 on 1..3):")
for r in (2, 3, 4):
    deriv_reg(cpro[r], cpro[r-1], tpro, f"r{r} on r{r-1}")
json.dump({"steps": tpro, "probe": {str(r): cpro[r] for r in cpro}}, open("deriv_curves.json", "w"))
