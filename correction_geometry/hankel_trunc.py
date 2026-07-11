"""Jasmina's truncation puzzle: rank-k cutoff of the net's Hankel vs the true
generator's — evaluated against ground truth. Diagnosis: subspace closure (fit R2),
truncated-transfer spectra, principal angles."""
import glob, json, os
import torch
import numpy as np
from train import GPT
import speclib as S

base = os.path.dirname(os.path.abspath(__file__))
dev = "cuda"
Q0, B = 60, 1024
TESTS = S.make_tests(4)

ev = torch.load(os.path.join(base, "runs", "seed0", "evalset.pt"))
seq = ev["eseq"][:B].to(dev)
m = GPT(L=seq.shape[1], d=128, nl=4).to(dev)
ck = sorted(glob.glob(os.path.join(base, "runs", "seed0", "ckpts", "*.pt")))
ss = np.array([int(os.path.basename(c)[4:10]) for c in ck])
def load(step):
    m.load_state_dict(torch.load(ck[int(np.argmin(abs(ss - step)))], map_location=dev))
    m.eval()

load(8000)
tab_net = S.net_table(m, seq, Q0, dev=dev)
tab_true = S.rrxor3_table(seq, Q0)
load(1000)
tab_pre = S.net_table(m, seq, Q0, dev=dev)

p_true = S.rrxor3_p1_run(seq)                 # col t-1 -> P(token t = 1 | truth)
HORIZ = 30

def eval_vs_truth(oom):
    """Roll the machine teacher-forced on real sequences; CE and KL(true||machine)."""
    z = oom["Z"].clone()
    ces, kls = [], []
    for t in range(Q0, Q0 + HORIZ):
        p, z = S.oom_step(oom, z, seq[:, t].cpu())
        p = p.clamp(1e-4, 1 - 1e-4)
        x = seq[:, t].cpu().double()
        ces.append(float(-(x * torch.log(p) + (1 - x) * torch.log(1 - p)).mean()))
        pt = p_true[:, t - 1].clamp(1e-6, 1 - 1e-6)
        kls.append(float((pt * torch.log(pt / p) +
                          (1 - pt) * torch.log((1 - pt) / (1 - p))).mean()))
    return np.mean(ces), np.mean(kls), np.mean(kls[:5]), np.mean(kls[-5:])

def top_subspace(tab, k):
    Y = S.string_probs(tab, TESTS)
    U, Sv, Vh = torch.linalg.svd(Y, full_matrices=False)
    return Vh[:k]

print(f"{'machine':22s} {'k':>2s} {'fitR2':>8s} {'specrad':>8s} {'CE':>7s} "
      f"{'KL(true||m)':>11s} {'KL@1-5':>8s} {'KL@26-30':>9s}")
res = {}
for name, tab in [("true generator", tab_true), ("net final (8000)", tab_net),
                  ("net pre-drop2 (1000)", tab_pre)]:
    for k in (2, 3, 4, 5, 6):
        oom = S.fit_oom(tab, TESTS, k)
        e = S.eig_transfer(oom)
        ce, kl, kl0, kl1 = eval_vs_truth(oom)
        res[(name, k)] = dict(fit=oom["fitr2"], sr=float(abs(e[0])), ce=ce, kl=kl)
        print(f"{name:22s} {k:2d} {oom['fitr2']:8.4f} {abs(e[0]):8.3f} {ce:7.4f} "
              f"{kl:11.5f} {kl0:8.5f} {kl1:9.5f}")
    print()

pt = p_true[:, Q0-1:Q0+HORIZ-1].clamp(1e-6, 1 - 1e-6)
xs = seq[:, Q0:Q0+HORIZ].cpu().double()
print(f"Bayes floor CE (truth on itself): {float(-(xs*torch.log(pt)+(1-xs)*torch.log(1-pt)).mean()):.4f}")

# principal angles between top-3 subspaces (net vs true)
V3n, V3t = top_subspace(tab_net, 3), top_subspace(tab_true, 3)
sv = torch.linalg.svdvals(V3n @ V3t.T)
print("principal angles (deg) net-top3 vs true-top3:",
      np.round(np.degrees(np.arccos(sv.clamp(-1, 1).numpy())), 2))
# truncated transfer eigenvalues at k=3
for name, tab in [("true", tab_true), ("net", tab_net)]:
    oom = S.fit_oom(tab, TESTS, 3)
    e = S.eig_transfer(oom)
    print(f"k=3 transfer eigenvalues [{name}]: " +
          "  ".join(f"{v.real:.3f}" if abs(v.imag) < 0.02 else
                    f"{abs(v):.3f}∠{abs(np.angle(v)):.3f}" for v in e))
