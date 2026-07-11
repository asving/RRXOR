"""Leaky RRXOR: extract the FRACTAL phase-belief attractor from behavior via the
transfer's (damped) rotation eigenspace. Same construction as eigproj.py; prediction:
the clock plane is an affine image of the exact leaky class posterior (A/B/C), whose
reachable set is a fractal contracted toward center by the leak."""
import glob, json, os
import torch
import numpy as np
from train import GPT
import speclib as S

base = os.path.dirname(os.path.abspath(__file__))
dev = "cuda"
B, K, MX = 1024, 5, 5
TESTS = S.make_tests(MX)
POSITIONS = [40, 50, 60, 70]

out = {}
for run, eps in [("fx0.03_seed0", 0.03), ("fx0.2_seed0", 0.2)]:
    ev = torch.load(os.path.join(base, "runs", run, "evalset.pt"))
    seq = ev["eseq"][:B].to(dev)
    m = GPT(L=seq.shape[1], d=128, nl=4).to(dev)
    ck = sorted(glob.glob(os.path.join(base, "runs", run, "ckpts", "*.pt")))
    ss = np.array([int(os.path.basename(c)[4:10]) for c in ck])
    m.load_state_dict(torch.load(ck[-1], map_location=dev)); m.eval()

    tab = S.net_table(m, seq, 60, maxlen=MX, dev=dev)
    oom = S.fit_oom(tab, TESTS, K, colnorm=True)
    T = (oom["ops"][0] + oom["ops"][1]).numpy()
    lam, R = np.linalg.eig(T)
    Linv = np.linalg.inv(R)
    order = np.argsort(-np.abs(lam))
    lam = lam[order]; Linv = Linv[order]
    i1 = 0
    iw = next(i for i in range(K) if lam[i].imag > 0.02)
    inil = [i for i in range(K) if abs(lam[i]) < 0.5]
    print(f"{run}: fit R2 {oom['fitr2']:.4f}  eigen: " +
          " ".join(f"{v.real:.3f}" if abs(v.imag) < 0.02 else
                   f"{abs(v):.3f}∠{abs(np.angle(v)):.3f}" for v in lam) +
          f"   [predicted clock radius {1 - eps:.3f}]")

    PI = S.fx_pi_run(seq, eps)
    Ws, Qs, Ps = [], [], []
    for P in POSITIONS:
        tabP = S.net_table(m, seq, P, maxlen=MX, dev=dev)
        y = S.string_probs(tabP, TESTS) * oom["D"][None]
        z = (y @ oom["Vh"].T).numpy()
        c = Linv @ z.T
        w = c[iw] / c[i1]
        q = torch.stack([PI[:, P, 0], PI[:, P, 1] + PI[:, P, 2],
                         PI[:, P, 3] + PI[:, P, 4]], 1).numpy()
        pend = torch.stack([PI[:, P, 2] - PI[:, P, 1],
                            PI[:, P, 4] - PI[:, P, 3]], 1).numpy()
        Ws.append(np.stack([w.real, w.imag], 1)); Qs.append(q)
        Ps.append(np.stack([f(c[i]) for i in inil for f in (np.real, np.imag)], 1
                           ).astype(float) if inil else np.zeros((B, 1)))
    W = np.concatenate(Ws); Q = np.concatenate(Qs); ZN = np.concatenate(Ps)
    PEND = np.concatenate([np.stack([
        (PI[:, P, 2] - PI[:, P, 1]).numpy(),
        (PI[:, P, 4] - PI[:, P, 3]).numpy()], 1) for P in POSITIONS])

    def r2(X, Y):
        X = np.asarray(X, float); Y = np.asarray(Y, float)
        if Y.ndim == 1: Y = Y[:, None]
        n = len(X); ntr = int(0.7 * n)
        Xb = np.concatenate([X, np.ones((n, 1))], 1)
        Wm, *_ = np.linalg.lstsq(Xb[:ntr], Y[:ntr], rcond=None)
        Pd = Xb[ntr:] @ Wm
        return float(1 - ((Y[ntr:] - Pd) ** 2).sum() /
                     ((Y[ntr:] - Y[:ntr].mean(0)) ** 2).sum()), Wm

    r_fwd, _ = r2(W, Q[:, :2])
    r_bwd, _ = r2(Q, W)
    r_nil_q, _ = r2(ZN, Q[:, :2])
    r_nil_p, _ = r2(ZN, PEND)
    r_w_p, _ = r2(W, PEND)
    print(f"  class posterior <- clock plane: R2 {r_fwd:.4f} | reverse {r_bwd:.4f}")
    print(f"  class <- nilpotent: {r_nil_q:.4f} | pending <- nilpotent: {r_nil_p:.4f}"
          f" | pending <- clock: {r_w_p:.4f}")
    out[run] = {"eps": eps, "lam": [[v.real, v.imag] for v in lam],
                "W": W.tolist(), "Q": Q.tolist(),
                "r2": {"fwd": r_fwd, "bwd": r_bwd, "nil_q": r_nil_q,
                       "nil_p": r_nil_p, "w_p": r_w_p},
                "fitr2": oom["fitr2"]}
json.dump(out, open(os.path.join(base, "eigproj_fx.json"), "w"))
print("saved eigproj_fx.json")
