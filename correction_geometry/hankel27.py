"""Spectral HMM reconstruction for matryoshka: the root-of-unity tower.

Behavioral machines of the stage policies are cycles of length 3 / 9 / 27 (FLIP emits
like xor, so stage-3's all-exceptions-alike policy has period 9, the true pattern 27).
Prediction: the transfer operator's unit-circle spectrum fills 1 -> 3rd -> 9th -> 27th
roots of unity across drops 2/3/4. Tests: exhaustive |s|<=3 plus generator-sampled
length-24 suffixes (short tests cannot separate phases 9 apart).
"""
import glob, json, math, os
import torch
import numpy as np
from train import GPT
from train27 import NESTED

base = os.path.dirname(os.path.abspath(__file__))
dev = "cuda"
run = "m27_nl6_seed1"
Q0, B, NLONG, TLEN = 120, 1024, 240, 24
ev = torch.load(os.path.join(base, "runs", run, "evalset.pt"))
seq = ev["eseq"][:B].to(dev)
L = seq.shape[1]
prefix = seq[:, :Q0]
rule = torch.tensor(NESTED)

# ---- tests: exhaustive short + generator-sampled long ----
TESTS = [tuple((b >> i) & 1 for i in range(ln - 1, -1, -1))
         for ln in (1, 2, 3) for b in range(2 ** ln)]
g = torch.Generator().manual_seed(5)
seen = set(TESTS)
while len(TESTS) < 14 + NLONG:
    ph0 = int(torch.randint(0, 27, (1,), generator=g))       # phase of first test token
    toks, hist = [], []
    for k in range(TLEN):
        c = (ph0 + k) % 27
        if c % 3 == 2 and len(hist) >= 2:
            u = hist[-1] ^ hist[-2]
            t = (u + int(rule[((c - 2) // 3) % 9])) % 2
        else:
            t = int(torch.randint(0, 2, (1,), generator=g))
        toks.append(t); hist.append(t)
    tt = tuple(toks)
    if tt not in seen:
        seen.add(tt); TESTS.append(tt)
PATHS = sorted(set([t for t in TESTS] + [(s,) + t for s in (0, 1) for t in TESTS]))

def net_table(model):
    """path -> (B, len(path)) conditionals p(token_k = 1 | prefix + path[:k])."""
    out = {}
    with torch.no_grad():
        for p in PATHS:
            x = torch.cat([prefix, torch.tensor(p, device=dev)[None].expand(B, -1)], 1)
            lg = model(x)
            p1 = torch.softmax(lg[:, :, :2].double(), -1)[:, :, 1]
            out[p] = p1[:, Q0 - 1:Q0 - 1 + len(p)].cpu().clamp(1e-6, 1 - 1e-6)
    return out

def true_table():
    """Exact filter conditionals along each path (27-hypothesis elimination)."""
    sc = seq.cpu()
    classes = torch.arange(27)
    det = classes % 3 == 2
    crule = rule[(classes // 3) % 9]
    def step(pi, xm1, xm2):
        pi2 = pi.roll(1, dims=1)
        u = (xm1 ^ xm2).double()
        pred = (u[:, None] + crule[None].double()) % 2
        p1 = (pi2 * torch.where(det[None], pred, torch.full_like(pred, 0.5))).sum(1)
        return pi2, p1, pred
    pi = torch.full((B, 27), 1 / 27, dtype=torch.double)
    for t in range(1, Q0):
        if t < 2:
            pi = pi.roll(1, dims=1)
            continue
        pi2, p1, pred = step(pi, sc[:, t - 1], sc[:, t - 2])
        x = sc[:, t].double()
        lik = torch.where(det[None], (pred == x[:, None]).double(),
                          torch.full_like(pred, 0.5))
        pi = pi2 * lik
        pi = pi / pi.sum(1, keepdim=True).clamp_min(1e-300)
    out = {}
    for p in PATHS:
        cur = pi.clone()
        xm1, xm2 = sc[:, Q0 - 1], sc[:, Q0 - 2]
        conds = []
        for tok in p:
            pi2, p1, pred = step(cur, xm1, xm2)
            conds.append(p1.clamp(1e-9, 1 - 1e-9))
            xv = torch.full((B,), float(tok), dtype=torch.double)
            lik = torch.where(det[None], (pred == xv[:, None]).double(),
                              torch.full_like(pred, 0.5))
            cur = pi2 * lik
            cur = cur / cur.sum(1, keepdim=True).clamp_min(1e-300)
            xm2, xm1 = xm1, torch.full((B,), tok).long()
        out[p] = torch.stack(conds, 1)
    return out

def string_probs(tab, shift=None):
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

def spectral(tab, k):
    Yr = string_probs(tab)
    D = 1.0 / Yr.norm(dim=0).clamp_min(1e-12)          # per-test scale (fixed map)
    Y = Yr * D[None]
    U, S, Vh = torch.linalg.svd(Y, full_matrices=False)
    Z = Y @ Vh[:k].T
    ops = {}
    fitr2 = []
    psig1 = tab[(1,)][:, 0] if (1,) in tab else tab[(1, 0)][:, 0]
    for sig in (0, 1):
        Ys = string_probs(tab, shift=sig) * D[None]
        Zs = Ys @ Vh[:k].T
        psig = psig1 if sig == 1 else 1 - psig1
        T = Zs * psig[:, None]
        A = torch.linalg.lstsq(Z, T).solution.T
        fitr2.append(float(1 - ((T - Z @ A.T) ** 2).sum() / ((T - T.mean(0)) ** 2).sum()))
        ops[sig] = A
    return S, ops, min(fitr2)

def eigs(ops):
    e = np.linalg.eigvals((ops[0] + ops[1]).numpy())
    return e[np.argsort(-np.abs(e))]

model = GPT(L=L, d=128, nl=6).to(dev)
ck = sorted(glob.glob(os.path.join(base, "runs", run, "ckpts", "*.pt")))
steps = [int(os.path.basename(c)[4:10]) for c in ck]
def load(step):
    i = int(np.argmin(np.abs(np.array(steps) - step)))
    model.load_state_dict(torch.load(ck[i], map_location=dev)); model.eval()
    return steps[i]

def kpick(S, thr=1e-3):
    return min(int((S / S[0] > thr).sum()), 48)

print("=== true generator ===")
tab = true_table()
Yt = string_probs(tab)
St, _, _ = spectral(tab, 5)
kt = kpick(St)
print("sv/sv1:", np.round((St / St[0])[:50].numpy(), 4))
print("k_eff:", kt)
St, ops_t, fr = spectral(tab, kt)
print('operator fit R2:', round(fr, 5))
et = eigs(ops_t)
nroots = int((np.abs(et) > 0.75).sum())
print(f"unit-circle eigenvalues: {nroots}")
angs = sorted(np.angle(et[np.abs(et) > 0.75]))
print("angles/(2pi/27):", np.round(np.array(angs) / (2 * np.pi / 27), 2))

STAGES = [("post-drop1 (1000)", 1000), ("post-drop2 (2600)", 2600),
          ("post-drop3 (3400)", 3400), ("final (16000)", 16000)]
stage_eigs = {}
for tag, s in STAGES:
    load(s)
    tab = net_table(model)
    S, _, _ = spectral(tab, 5)
    ke = kpick(S)
    S, ops, fr = spectral(tab, ke)
    e = eigs(ops)
    stage_eigs[tag] = e
    nr = int((np.abs(e) > 0.75).sum())
    print(f"{tag:20s} k_eff {ke:2d}  |λ|>0.75: {nr:2d}  "
          f"top angles/(2π/27): {np.round(sorted(np.angle(e[np.abs(e)>0.75])/(2*np.pi/27)), 1)}")

print("\n=== trajectory: number of unit-circle eigenvalues (the root tower) ===")
traj = []
TSTEPS = [50, 200, 500, 850, 1150, 1550, 2000, 2200, 2400, 2600, 2800, 3000, 3200,
          3400, 3800, 4400, 5000, 5600, 6800, 8400, 12000, 16000]
for tstep in TSTEPS:
    load(tstep)
    tab = net_table(model)
    S, _, _ = spectral(tab, 5)
    ke = kpick(S)
    S, ops, fr = spectral(tab, ke)
    e = eigs(ops)
    # snapped count: distinct 27th-root classes hit by a near-unit eigenvalue
    hit = set()
    for v in e:
        if 0.8 < abs(v) < 1.15:
            kk = round(np.angle(v) / (2 * np.pi / 27))
            if abs(np.angle(v) - kk * 2 * np.pi / 27) < 0.25 * 2 * np.pi / 27:
                hit.add(int(kk) % 27)
    nr = len(hit)
    traj.append({"step": tstep, "k": ke, "nroots": nr, "fitr2": round(fr, 5),
                 "eigs": [[float(v.real), float(v.imag)] for v in e]})
    print(f"  step {tstep:6d}  k_eff {ke:2d}  root classes hit {nr:2d}  fitR2 {fr:.4f}")
json.dump({"traj": traj,
           "stages": {t: [[float(v.real), float(v.imag)] for v in e]
                      for t, e in stage_eigs.items()},
           "true": [[float(v.real), float(v.imag)] for v in et]},
          open(os.path.join(base, "runs", run, "hankel27.json"), "w"))
print("saved hankel27.json")
