"""Eigenvalue flow: transfer-operator spectrum at EVERY rrxor3 checkpoint (dense)."""
import glob, json, os
import torch
import numpy as np
from train import GPT

base = os.path.dirname(os.path.abspath(__file__))
dev = "cuda"
run = "seed0"
Q0, B = 60, 1024
ev = torch.load(os.path.join(base, "runs", run, "evalset.pt"))
seq = ev["eseq"][:B].to(dev)
L = seq.shape[1]
prefix = seq[:, :Q0]
SUF5 = [[(b >> i) & 1 for i in (4, 3, 2, 1, 0)] for b in range(32)]
TESTS = [tuple((b >> i) & 1 for i in range(ln - 1, -1, -1))
         for ln in (1, 2, 3, 4) for b in range(2 ** ln)]

model = GPT(L=96, d=128, nl=4).to(dev)

def table(m):
    out = {}
    with torch.no_grad():
        for s4 in SUF5:
            x = torch.cat([prefix, torch.tensor(s4, device=dev)[None].expand(B, -1)], 1)
            lg = m(x)
            p1 = torch.softmax(lg[:, :, :2].double(), -1)[:, :, 1]
            for k in range(5):
                key = tuple(s4[:k])
                if key not in out:
                    out[key] = p1[:, Q0 - 1 + k].cpu().clamp(1e-6, 1 - 1e-6)
    return out

def string_probs(tab, shift=None):
    cols = []
    for t in TESTS:
        pre = () if shift is None else (shift,)
        lp = torch.zeros(B, dtype=torch.double)
        for k, tok in enumerate(t):
            c = tab[pre + t[:k]]
            lp = lp + torch.log(c if tok == 1 else 1 - c)
        cols.append(torch.exp(lp))
    return torch.stack(cols, 1)

def spectral(tab, k=5):
    Y = string_probs(tab)
    U, S, Vh = torch.linalg.svd(Y, full_matrices=False)
    Z = Y @ Vh[:k].T
    ops, fr = {}, []
    psig1 = tab[(1,)]
    for sig in (0, 1):
        Zs = string_probs(tab, shift=sig) @ Vh[:k].T
        psig = psig1 if sig == 1 else 1 - psig1
        T = Zs * psig[:, None]
        A = torch.linalg.lstsq(Z, T).solution.T
        fr.append(float(1 - ((T - Z @ A.T) ** 2).sum() / ((T - T.mean(0)) ** 2).sum()))
        ops[sig] = A
    e = np.linalg.eigvals((ops[0] + ops[1]).numpy())
    return S, e[np.argsort(-np.abs(e))], min(fr)

ck = sorted(glob.glob(os.path.join(base, "runs", run, "ckpts", "*.pt")))
steps = [int(os.path.basename(c)[4:10]) for c in ck]
flow = []
for c, st in zip(ck, steps):
    model.load_state_dict(torch.load(c, map_location=dev)); model.eval()
    S, e, fr = spectral(table(model))
    flow.append({"step": st, "eigs": [[float(v.real), float(v.imag)] for v in e],
                 "fitr2": round(fr, 5),
                 "sv": [round(float(x), 5) for x in (S / S[0])[:8]]})
    if st % 500 == 0 or st in (25, 75, 125):
        cl = [v for v in e if abs(v.imag) > 0.02]
        best = min(cl, key=lambda v: abs(abs(np.angle(v)) - 2 * np.pi / 3)) if cl else None
        print(f"  step {st:5d}  fitR2 {fr:.4f}  clock "
              f"{abs(best):.2f}∠{abs(np.angle(best)):.2f}" if best else
              f"  step {st:5d}  fitR2 {fr:.4f}  clock none")
json.dump(flow, open(os.path.join(base, "runs", run, "hankel_flow.json"), "w"))
print(f"saved hankel_flow.json ({len(flow)} checkpoints)")
