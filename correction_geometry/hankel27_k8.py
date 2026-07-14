"""Jasmina's k=8 question: what ARE the states of a rank-8 truncation of the
matryoshka behavioral machine? Fit k=8, read its transfer eigenvalues, and regress
its states against the oracle battery (mod-3 quotient, candidate, rule bits, fine
phase). Prediction: states = the coarse correction statistics (high R2 for mod-3 +
cand + rule-ish), fine phase absent, dynamics broken (eigen junk, low fit R2)."""
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
cls = ev["ecls"][:B]
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
        for k in range(len(p) + 1):
            key = tuple(p[:k])
            if key not in tab and k < len(p) + 1:
                tab[key] = p1[:, Q0 - 1:Q0 - 1 + len(p)].cpu().clamp(1e-6, 1 - 1e-6) \
                    if False else p1[:, Q0 - 1 + k - 1 if k else Q0 - 1].cpu()
# cleaner: store per-path conditional arrays as in hankel27
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

def spectral(k):
    Yr = string_probs()
    D = 1.0 / Yr.norm(dim=0).clamp_min(1e-12)
    Y = Yr * D[None]
    U, S, Vh = torch.linalg.svd(Y, full_matrices=False)
    Z = Y @ Vh[:k].T
    ops, frs = {}, []
    psig1 = tab[(1,)][:, 0]
    for sig in (0, 1):
        Zs = (string_probs(shift=sig) * D[None]) @ Vh[:k].T
        psig = psig1 if sig == 1 else 1 - psig1
        T = Zs * psig[:, None]
        A = torch.linalg.lstsq(Z, T).solution.T
        frs.append(float(1 - ((T - Z @ A.T) ** 2).sum() / ((T - T.mean(0)) ** 2).sum()))
        ops[sig] = A
    return S, Z, ops, min(frs)

# oracle battery at query Q0-1 (phase = ecls at Q0-1; synced/one-hot at 120)
phase = cls[:, Q0 - 1].long()
i_next = ((phase // 3) + ((phase % 3) == 2).long()) % 9
mod3 = torch.nn.functional.one_hot(phase % 3, 3).double()
g9 = (i_next % 3 == 2).double()[:, None]
g27 = (i_next == 8).double()[:, None]
u = (seq.cpu()[:, Q0 - 1] ^ seq.cpu()[:, Q0 - 2]).double()[:, None]
fine = torch.nn.functional.one_hot(phase // 3, 9).double()   # block index (fine phase)

def r2(X, Y):
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    if Y.ndim == 1: Y = Y[:, None]
    n = len(X); ntr = int(0.7 * n)
    Xb = np.concatenate([X, np.ones((n, 1))], 1)
    W, *_ = np.linalg.lstsq(Xb[:ntr], Y[:ntr], rcond=None)
    P = Xb[ntr:] @ W
    return float(1 - ((Y[ntr:] - P) ** 2).sum() / ((Y[ntr:] - Y[:ntr].mean(0)) ** 2).sum())

for k in (8, 11, 45):
    S, Z, ops, fr = spectral(k)
    e = np.linalg.eigvals((ops[0] + ops[1]).numpy())
    e = e[np.argsort(-np.abs(e))]
    Zn = Z.numpy()
    print(f"\nk={k}: operator fit R2 {fr:.4f}")
    print("  transfer eigen:", "  ".join(f"{v.real:.2f}" if abs(v.imag) < 0.02 else
          f"{abs(v):.2f}∠{abs(np.angle(v)):.2f}" for v in e[:10]))
    print(f"  states ← [mod3, u, g9, g27]:      R2 {r2(np.concatenate([mod3, u, g9, g27], 1), Zn):.3f}")
    print(f"  mod3 quotient ← states:           R2 {r2(Zn, mod3.numpy()[:, :2]):.3f}")
    print(f"  candidate u ← states:             R2 {r2(Zn, u.numpy()):.3f}")
    print(f"  rule bit g9 ← states:             R2 {r2(Zn, g9.numpy()):.3f}")
    print(f"  FINE phase (block id) ← states:   R2 {r2(Zn, fine.numpy()[:, :8]):.3f}")
