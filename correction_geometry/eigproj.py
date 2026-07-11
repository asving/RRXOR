"""Extract the phase-belief simplex from BEHAVIOR by projecting the reconstructed
machine's state onto the transfer operator's rotation eigenspace.

State z(h) from the Hankel; transfer T = A0+A1 eigendecomposed; phase coordinate
w(h) = (left-eigvec_omega . z) / (left-eigvec_1 . z) — a projective coordinate on the
clock plane. Prediction: w(h) is an affine image of the exact offset posterior
(vertices = synced, interior = the MSP sync tree), and the nilpotent sector carries
the pending bits instead. Certify with cross-regressions against the exact filter.
"""
import glob, json, os
import torch
import numpy as np
from train import GPT
import speclib as S

base = os.path.dirname(os.path.abspath(__file__))
dev = "cuda"
B, K = 1024, 5
TESTS = S.make_tests(4)
POSITIONS = [4, 5, 6, 7, 8, 10, 12, 60]

ev = torch.load(os.path.join(base, "runs", "seed0", "evalset.pt"))
seq = ev["eseq"][:B].to(dev)
L = seq.shape[1]
m = GPT(L=L, d=128, nl=4).to(dev)
ck = sorted(glob.glob(os.path.join(base, "runs", "seed0", "ckpts", "*.pt")))
ss = np.array([int(os.path.basename(c)[4:10]) for c in ck])
m.load_state_dict(torch.load(ck[int(np.argmin(abs(ss - 8000)))], map_location=dev))
m.eval()

# ---- fit the machine at P=60, eigendecompose the transfer ----
tab60 = S.net_table(m, seq, 60, dev=dev)
oom = S.fit_oom(tab60, TESTS, K)
T = (oom["ops"][0] + oom["ops"][1]).numpy()
lam, R = np.linalg.eig(T)
Linv = np.linalg.inv(R)                      # rows = left eigenvectors
order = np.argsort(-np.abs(lam))
lam = lam[order]; R = R[:, order]; Linv = Linv[order]
print("transfer eigenvalues:", " ".join(f"{v.real:.3f}" if abs(v.imag) < 0.02 else
      f"{abs(v):.3f}∠{np.angle(v):+.3f}" for v in lam))
i1 = 0                                        # stationary (lam ~ 1)
iw = next(i for i in range(K) if lam[i].imag > 0.02)      # the clock (+ angle)
inil = [i for i in range(K) if abs(lam[i]) < 0.5]         # nilpotent sector
print(f"modes: stationary idx {i1} (λ={lam[i1].real:.3f}), clock idx {iw} "
      f"(λ={abs(lam[iw]):.3f}∠{np.angle(lam[iw]):.4f}), nilpotent {inil}")

# ---- exact offset posterior (elimination filter), all positions ----
sc = seq.cpu()
v = torch.zeros(B, L); v[:, 2:] = (sc[:, 2:] != (sc[:, 1:-1] ^ sc[:, :-2])).double()
tt = torch.arange(L)
post = torch.zeros(B, L, 3, dtype=torch.double)
for o in range(3):
    det = ((o + tt) % 3 == 2) & (tt >= 2)
    killed = ((v.bool() & det[None]).cumsum(1) > 0).double()
    post[:, :, o] = (1 - killed) * torch.pow(2.0, det[None].double().expand(B, -1).cumsum(1))
post = post / post.sum(-1, keepdim=True).clamp_min(1e-30)

# ---- per-position states, phase coordinates, targets ----
rows = []
for P in POSITIONS:
    tab = S.net_table(m, seq, P, dev=dev)
    y = S.string_probs(tab, TESTS)
    z = (y @ oom["Vh"].T).numpy()                        # B x K, same basis
    c = Linv @ z.T                                       # K x B complex
    w = c[iw] / c[i1]
    pi_rel = torch.zeros(B, 3, dtype=torch.double)
    for o in range(3):
        r = (o + P) % 3
        pi_rel[:, r] += post[:, P - 1, o]
    u = (sc[:, P - 1] ^ sc[:, P - 2]).double()
    znil = np.stack([f(c[i]) for i in inil for f in (np.real, np.imag)], 1)
    rows.append(dict(P=P, w=w, pi=pi_rel.numpy(), u=u.numpy(), znil=znil,
                     c1=np.abs(c[i1])))
print("c1 (stationary coord) spread:",
      " ".join(f"P{r['P']}:{np.std(r['c1'])/np.mean(r['c1']):.3f}" for r in rows))

def affine_r2(X, Y, ntr=None):
    X = np.asarray(X, dtype=float); Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1: Y = Y[:, None]
    n = len(X); ntr = ntr or int(0.7 * n)
    Xb = np.concatenate([X, np.ones((n, 1))], 1)
    Wm, *_ = np.linalg.lstsq(Xb[:ntr], Y[:ntr], rcond=None)
    Pd = Xb[ntr:] @ Wm
    return float(1 - ((Y[ntr:] - Pd) ** 2).sum() /
                 ((Y[ntr:] - Y[:ntr].mean(0)) ** 2).sum()), Wm

print("\n=== certification: phase sector <-> exact offset posterior ===")
W2 = np.concatenate([np.stack([r["w"].real, r["w"].imag], 1) for r in rows])
PI = np.concatenate([r["pi"] for r in rows])
UU = np.concatenate([r["u"] for r in rows])
ZN = np.concatenate([r["znil"] for r in rows])
r2_fwd, Wmap = affine_r2(W2, PI[:, :2])          # 2 free dims of the simplex
r2_bwd, _ = affine_r2(PI, W2)
print(f"pooled over positions {POSITIONS}:")
print(f"  posterior <- clock plane (Re w, Im w):  R2 {r2_fwd:.4f}")
print(f"  clock plane <- posterior:               R2 {r2_bwd:.4f}")
for r in rows:
    rr, _ = affine_r2(np.stack([r["w"].real, r["w"].imag], 1), r["pi"][:, :2])
    print(f"  per-position P={r['P']:2d}: R2 {rr:.4f}")

print("\n=== factorization: what lives in which spectral sector ===")
r2_nil_pi, _ = affine_r2(ZN, PI[:, :2])
r2_nil_u, _ = affine_r2(ZN, UU)
r2_w_u, _ = affine_r2(W2, UU)
pre_det = PI[:, 2] > 0.9                          # synced, next slot det
r2_nil_u_pd, _ = affine_r2(ZN[pre_det], UU[pre_det])
r2_w_u_pd, _ = affine_r2(W2[pre_det], UU[pre_det])
print(f"  posterior <- nilpotent sector:          R2 {r2_nil_pi:.4f}   (phase should NOT live here)")
print(f"  u (candidate) <- nilpotent sector:      R2 {r2_nil_u:.4f}  | pre-det only {r2_nil_u_pd:.4f}")
print(f"  u (candidate) <- clock plane:           R2 {r2_w_u:.4f}  | pre-det only {r2_w_u_pd:.4f}")

json.dump({"lam": [[v.real, v.imag] for v in lam],
           "rows": [{"P": r["P"], "w": np.stack([r["w"].real, r["w"].imag], 1).tolist(),
                     "pi": r["pi"].tolist()} for r in rows],
           "r2": {"fwd": r2_fwd, "bwd": r2_bwd, "nil_pi": r2_nil_pi,
                  "nil_u": r2_nil_u, "nil_u_predet": r2_nil_u_pd,
                  "w_u": r2_w_u, "w_u_predet": r2_w_u_pd}},
          open(os.path.join(base, "eigproj.json"), "w"))
print("saved eigproj.json")
