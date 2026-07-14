"""Which eigensector? Answer: sectors are proposed by the spectrum's arithmetic
(subgroup lattice of the unit-circle angles) and certified by SECTOR ABLATION:
project the k=45 machine onto {1}+nil, {1,3rd}+nil, {1,3rd,9th}+nil, full — roll
each ablated machine and compare CE to the preregistered ladder rungs
0.6931 / 0.6758 / 0.6386 / 0.5328 / 0.4621."""
import glob, json, math, os
import torch
import numpy as np
from train import GPT
from train27 import NESTED

base = os.path.dirname(os.path.abspath(__file__))
dev = "cuda"
run = "m27_nl6_seed1"
Q0, B, NLONG, TLEN, K = 120, 1024, 240, 24, 45
ev = torch.load(os.path.join(base, "runs", run, "evalset.pt"))
seq = ev["eseq"][:B].to(dev)
L = seq.shape[1]
prefix = seq[:, :Q0]
rule = torch.tensor(NESTED)

TESTS = [tuple((b >> i) & 1 for i in range(ln - 1, -1, -1))
         for ln in (1, 2, 3) for b in range(2 ** ln)]
g = torch.Generator().manual_seed(5)
seen = set(TESTS)
while len(TESTS) < 14 + NLONG:
    ph0 = int(torch.randint(0, 27, (1,), generator=g))
    toks, hist = [], []
    for k in range(TLEN):
        c = (ph0 + k) % 27
        if c % 3 == 2 and len(hist) >= 2:
            t = ((hist[-1] ^ hist[-2]) + int(rule[((c - 2) // 3) % 9])) % 2
        else:
            t = int(torch.randint(0, 2, (1,), generator=g))
        toks.append(t); hist.append(t)
    tt = tuple(toks)
    if tt not in seen:
        seen.add(tt); TESTS.append(tt)
PATHS = sorted(set([t for t in TESTS] + [(s,) + t for s in (0, 1) for t in TESTS]))

model = GPT(L=L, d=128, nl=6).to(dev)
ck = sorted(glob.glob(os.path.join(base, "runs", run, "ckpts", "*.pt")))
model.load_state_dict(torch.load(ck[-1], map_location=dev)); model.eval()
tab = {}
with torch.no_grad():
    for p in PATHS:
        x = torch.cat([prefix, torch.tensor(p, device=dev)[None].expand(B, -1)], 1)
        lg = model(x)
        p1 = torch.softmax(lg[:, :, :2].double(), -1)[:, :, 1]
        tab[p] = p1[:, Q0 - 1:Q0 - 1 + len(p)].cpu().clamp(1e-6, 1 - 1e-6)

def string_probs(shift=None):
    cols = []
    for t in TESTS:
        p = tab[t if shift is None else (shift,) + t]
        off = 0 if shift is None else 1
        lp = torch.zeros(B, dtype=torch.double)
        for k, tok in enumerate(t):
            c = p[:, k + off]
            lp = lp + torch.log(c if tok == 1 else 1 - c)
        cols.append(torch.exp(lp))
    return torch.stack(cols, 1)

Yr = string_probs()
D = 1.0 / Yr.norm(dim=0).clamp_min(1e-12)
Y = Yr * D[None]
U, S, Vh = torch.linalg.svd(Y, full_matrices=False)
Z = Y @ Vh[:K].T
ops = {}
psig1 = tab[(1,)][:, 0]
for sig in (0, 1):
    Zs = (string_probs(shift=sig) * D[None]) @ Vh[:K].T
    psig = psig1 if sig == 1 else 1 - psig1
    ops[sig] = torch.linalg.lstsq(Z, Zs * psig[:, None]).solution.T

T = (ops[0] + ops[1]).numpy()
lam, R = np.linalg.eig(T)
Linv = np.linalg.inv(R)
LAT = 2 * np.pi / 27
def kclass(v):
    kk = int(round(np.angle(v) / LAT)) % 27
    ok = abs(np.angle(v) - round(np.angle(v) / LAT) * LAT) < 0.35 * LAT and abs(v) > 0.8
    return kk if ok else None

nil = np.array([abs(v) < 0.8 for v in lam])
kcls = [kclass(v) for v in lam]
def sector_mask(root_ks):
    m = nil.copy()
    for i, kc in enumerate(kcls):
        if kc is not None and kc in root_ks:
            m[i] = True
    return m

SECTORS = [("{1} + transients            (bet rung 0.6758?)", {0}),
           ("{1, 3rd roots} + transients (mod-3 rung 0.6386?)", {0, 9, 18}),
           ("{1, 9th roots} + transients (mod-9 rung 0.5328?)", {0, 3, 6, 9, 12, 15, 18, 21, 24}),
           ("full spectrum               (floor 0.4621?)", set(range(27)))]

i1 = TESTS.index((1,)); i0 = TESTS.index((0,))
v1 = (Vh[:K, i1] / D[i1]).numpy(); v0 = (Vh[:K, i0] / D[i0]).numpy()
toks = seq.cpu().numpy()

for name, ks in SECTORS:
    m = sector_mask(ks)
    P = (R[:, m] @ Linv[m]).real
    A = {s: P @ ops[s].numpy() @ P for s in (0, 1)}
    z = (Z.numpy() @ P.T)
    ces = []
    for t in range(Q0, min(Q0 + 60, L)):
        p1 = z @ v1; p0 = z @ v0
        tot = p0 + p1
        tot = np.where(np.abs(tot) < 1e-9, 1.0, tot)
        p = np.clip(p1 / tot, 1e-4, 1 - 1e-4)
        x = toks[:, t]
        ces.append(float(-(x * np.log(p) + (1 - x) * np.log(1 - p)).mean()))
        psig = np.where(x == 1, p, 1 - p)
        Anext = np.where(x[:, None, None] == 1, A[1][None], A[0][None])
        z = np.einsum("bij,bj->bi", Anext, z) / psig[:, None]
        zt = z @ (v0 + v1)
        zt = np.where(np.abs(zt) < 0.05, 1.0, zt)
        z = z / zt[:, None]
    print(f"{name:52s} modes {int(m.sum()):2d}  CE {np.mean(ces):.4f}")
print("\nladder rungs: uniform .6931 | bet .6758 | mod-3 .6386 | mod-9 .5328 | full .4621")
