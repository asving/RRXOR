"""Sector-ablation ladders on rrxor3 and fx0.03: does {1}+transients land on the
ungated-bet policy value, and full on the floor? (fx = the no-ladder contrast:
the sector ladder exists as POLICIES even though training never plateaus there.)"""
import glob, math, os
import torch
import numpy as np
from train import GPT
import speclib as S

base = os.path.dirname(os.path.abspath(__file__))
dev = "cuda"
Q0, B, K = 60, 1024, 5

def run(name, nl, mx, colnorm):
    ev = torch.load(os.path.join(base, "runs", name, "evalset.pt"))
    seq = ev["eseq"][:B].to(dev)
    m = GPT(L=seq.shape[1], d=128, nl=nl).to(dev)
    ck = sorted(glob.glob(os.path.join(base, "runs", name, "ckpts", "*.pt")))
    m.load_state_dict(torch.load(ck[-1], map_location=dev)); m.eval()
    TESTS = S.make_tests(mx)
    tab = S.net_table(m, seq, Q0, maxlen=mx, dev=dev)
    oom = S.fit_oom(tab, TESTS, K, colnorm=colnorm)
    T = (oom["ops"][0] + oom["ops"][1]).numpy()
    lam, R = np.linalg.eig(T)
    Linv = np.linalg.inv(R)
    i1t, i0t = TESTS.index((1,)), TESTS.index((0,))
    v1 = (oom["Vh"][:, i1t] / oom["D"][i1t]).numpy()
    v0 = (oom["Vh"][:, i0t] / oom["D"][i0t]).numpy()
    toks = seq.cpu().numpy()

    # references: optimal ungated-bet CE on this eval band + net's own CE
    cand = (seq.cpu()[:, 1:] ^ seq.cpu()[:, :-1]).numpy()      # cand[:, q] for t=q+1
    band = np.arange(Q0, Q0 + 30)
    y = (toks[:, band] == cand[:, band - 2]).astype(float)
    qh = y.mean()
    bet_ce = float(-(y * np.log(qh) + (1 - y) * np.log(1 - qh)).mean())

    def ablate(keep_clock):
        mmask = np.abs(lam) < 0.8
        mmask |= np.abs(np.angle(lam)) < 0.05                   # stationary
        if keep_clock:
            mmask |= np.abs(lam) >= 0.5                         # everything big
        P = (R[:, mmask] @ Linv[mmask]).real
        A = {s: P @ oom["ops"][s].numpy() @ P for s in (0, 1)}
        z = oom["Z"].numpy() @ P.T
        ces = []
        for t in band:
            p1 = z @ v1; p0 = z @ v0
            tot = np.where(np.abs(p0 + p1) < 1e-9, 1.0, p0 + p1)
            p = np.clip(p1 / tot, 1e-4, 1 - 1e-4)
            x = toks[:, t]
            ces.append(float(-(x * np.log(p) + (1 - x) * np.log(1 - p)).mean()))
            psig = np.where(x == 1, p, 1 - p)
            An = np.where(x[:, None, None] == 1, A[1][None], A[0][None])
            z = np.einsum("bij,bj->bi", An, z) / psig[:, None]
            zt = z @ (v0 + v1)
            z = z / np.where(np.abs(zt) < 0.05, 1.0, zt)[:, None]
        return float(np.mean(ces))
    print(f"\n{name}: eigen " + " ".join(f"{v.real:.2f}" if abs(v.imag) < 0.02 else
          f"{abs(v):.2f}∠{abs(np.angle(v)):.2f}" for v in lam[np.argsort(-np.abs(lam))]))
    print(f"  {{1}} + transients : CE {ablate(False):.4f}   (optimal ungated bet = {bet_ce:.4f})")
    print(f"  + clock (full)    : CE {ablate(True):.4f}")

run("seed0", 4, 4, False)          # rrxor3: floor 0.4621
run("fx0.03_seed0", 4, 5, True)    # leaky:  floor 0.5516
