"""Spectral reconstruction of the behavioral HMM from the net's conditionals (rrxor3).

H[h, s] = P_net(suffix s | history h) for all tests |s| <= 3, rows = eval histories
truncated at a fixed position q0 (net is position-dependent). SVD -> rank spectrum;
OOM operators A_sigma by shifted-state regression; eigenvalues of A0+A1 = the process
clock. References: the exact generator (5-state machine -> {1, w, w~, 0, 0}, w=e2pi/3)
and the pre-drop checkpoint. Trajectory: effective rank + dominant eigen-angle across
all checkpoints.
"""
import glob, json, math, os
import torch
import numpy as np
from train import GPT

base = os.path.dirname(os.path.abspath(__file__))
dev = "cuda"
run = "seed0"
Q0 = 60
ev = torch.load(os.path.join(base, "runs", run, "evalset.pt"))
seq = ev["eseq"].to(dev)
B, L = seq.shape
prefix = seq[:, :Q0]

SUF5 = [[(b >> i) & 1 for i in (4, 3, 2, 1, 0)] for b in range(32)]

def net_cond_table(model):
    """P(next token = 1 | prefix + any suffix<=3) for every length-4 suffix path.
    Returns dict: tuple(s_partial) -> tensor(B,) of p1 after that partial suffix."""
    out = {}
    with torch.no_grad():
        for s4 in SUF5:
            x = torch.cat([prefix, torch.tensor(s4, device=dev)[None].expand(B, -1)], 1)
            lg = model(x)
            p1 = torch.softmax(lg[:, :, :2], -1)[:, :, 1]
            for k in range(5):                      # conditional after k suffix tokens
                key = tuple(s4[:k])
                if key not in out:
                    out[key] = p1[:, Q0 - 1 + k].cpu()
    return out

def true_cond_table():
    """Exact generator conditionals via the elimination filter (offset posterior)."""
    sc = seq.cpu()
    out = {}
    def roll(tokens):
        # posterior over offsets after prefix + tokens, plus last two bits
        v = torch.zeros(B, Q0 + len(tokens))
        full = torch.cat([sc[:, :Q0]] +
                         [torch.full((B, 1), t) for t in tokens], 1).long()
        v[:, 2:] = (full[:, 2:] != (full[:, 1:-1] ^ full[:, :-2])).float()
        w = torch.ones(B, 3)
        T = full.shape[1]
        for o in range(3):
            det = torch.tensor([(o + t) % 3 == 2 and t >= 2 for t in range(T)])
            kill = (v[:, det].sum(1) > 0)
            w[kill, o] = 0
            w[:, o] *= 2.0 ** det.sum()
        w = w / w.sum(1, keepdim=True).clamp_min(1e-30)
        cand = (full[:, -1] ^ full[:, -2]).float()
        qdet = sum(w[:, o] * float((o + T) % 3 == 2) for o in range(3))
        return qdet * cand + (1 - qdet) / 2
    for s4 in SUF5:
        for k in range(5):
            key = tuple(s4[:k])
            if key not in out:
                out[key] = roll(list(key))
    return out

TESTS = [tuple((b >> i) & 1 for i in range(ln - 1, -1, -1))
         for ln in (1, 2, 3, 4) for b in range(2 ** ln)]

def string_probs(tab, shift=()):
    """P(test | history + shift) for all 14 tests, from the conditional table."""
    cols = []
    for s in TESTS:
        p = torch.ones(B)
        for k, tok in enumerate(s):
            p1 = tab[tuple(shift) + s[:k]]
            p = p * (p1 if tok == 1 else 1 - p1)
        cols.append(p)
    return torch.stack(cols, 1)                     # B x 14

def spectral(tab, k=5):
    Y = string_probs(tab)
    U, S, Vh = torch.linalg.svd(Y - 0 * Y.mean(0), full_matrices=False)
    Z = Y @ Vh[:k].T                                # B x k states
    ops = {}
    for sig in (0, 1):
        Ys = string_probs(tab, shift=(sig,))        # tests after emitting sig
        Zs = Ys @ Vh[:k].T
        psig = tab[()] if sig == 1 else 1 - tab[()]
        target = Zs * psig[:, None]                 # OOM: A_sig z(h) = P(sig|h) z(h sig)
        A = torch.linalg.lstsq(Z, target).solution.T
        ops[sig] = A
    return S, ops, Z, Vh

def eig_summary(ops):
    Asum = (ops[0] + ops[1]).numpy()
    ev_ = np.linalg.eigvals(Asum)
    ev_ = ev_[np.argsort(-np.abs(ev_))]
    return ev_

model = GPT(L=96, d=128, nl=4).to(dev)
ck = sorted(glob.glob(os.path.join(base, "runs", run, "ckpts", "*.pt")))
steps = [int(os.path.basename(c)[4:10]) for c in ck]

def load(step):
    i = int(np.argmin(np.abs(np.array(steps) - step)))
    model.load_state_dict(torch.load(ck[i], map_location=dev)); model.eval()
    return steps[i]

print("=== singular value spectra (top 8, normalized to sv1) ===")
tab_true = true_cond_table()
S_true, ops_true, _, _ = spectral(tab_true)
print("true generator :", np.round((S_true / S_true[0])[:8].numpy(), 4))
res = {}
for tag, step in [("pre-drop2 (1000)", 1000), ("post-drop2 (3000)", 3000),
                  ("final (8000)", 8000)]:
    load(step)
    tab = net_cond_table(model)
    S, ops, Z, Vh = spectral(tab)
    res[tag] = (S, ops)
    print(f"{tag:18s}:", np.round((S / S[0])[:8].numpy(), 4))

print("\n=== eigenvalues of A0+A1 (the transfer operator; clock = e^(2pi i/3), angle 2.094) ===")
for tag in ("true",):
    e = eig_summary(ops_true)
    print(f"true generator : " + "  ".join(f"{v:.3f}" if abs(v.imag) < 1e-9 else
          f"{abs(v):.3f}∠{abs(np.angle(v)):.3f}" for v in e))
for tag, (S, ops) in res.items():
    e = eig_summary(ops)
    print(f"{tag:18s}: " + "  ".join(f"{v.real:.3f}" if abs(v.imag) < 0.02 else
          f"{abs(v):.3f}∠{abs(np.angle(v)):.3f}" for v in e))

print("\n=== trajectory: effective rank + clock angle across training ===")
traj = []
for i in sorted(set(np.geomspace(1, len(ck) - 1, 18).astype(int))):
    load(steps[i])
    tab = net_cond_table(model)
    S, ops, _, _ = spectral(tab)
    erank = int((S / S[0] > 0.02).sum())
    e = eig_summary(ops)
    cplx = [v for v in e if abs(v.imag) > 0.05]
    ang = abs(np.angle(cplx[0])) if cplx else 0.0
    mag = abs(cplx[0]) if cplx else 0.0
    traj.append({"step": steps[i], "erank": erank, "angle": round(float(ang), 4),
                 "mag": round(float(mag), 3)})
    print(f"  step {steps[i]:6d}  eff-rank {erank}  clock angle {ang:.3f} (|.| {mag:.2f})")
json.dump(traj, open(os.path.join(base, "runs", run, "hankel_traj.json"), "w"))
