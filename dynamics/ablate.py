"""Prediction 2c — the time-travel test.

Mean-ablate a concept subspace (estimated from class means) at EVERY residual point
(embedding output + every block output), then measure per-type CE.

Preregistered: phase-ablated -> stage-1 plateau (ce2 ~ 0.405, ce0/1 ~ 0.752, pxor ~ 2/3
everywhere). xor-ablated -> baseline (~0.693 everywhere). random-subspace control -> no change.
"""
import glob, json, os, sys
import torch
import torch.nn.functional as F
from train import GPT

seed = sys.argv[1] if len(sys.argv) > 1 else "0"
dev = "cuda"
base = os.path.dirname(os.path.abspath(__file__))
run = os.path.join(base, "runs", f"seed{seed}")
ev = torch.load(os.path.join(run, "evalset.pt"))
seq, types = ev["eseq"].to(dev), ev["etypes"].to(dev)
B, L = seq.shape
MIN = 33

cp = sorted(glob.glob(os.path.join(run, "ckpts", "*.pt")))[-1]
print("final ckpt:", os.path.basename(cp))
model = GPT().to(dev)
model.load_state_dict(torch.load(cp)); model.eval()

inp = seq[:, :-1]
with torch.no_grad():
    _, resid = model(inp, return_resid=True)  # 5 x (B, L-1, d)

late = torch.arange(inp.shape[1], device=dev) >= MIN

def class_subspace(r, labels, ncls):
    """span of centered class means over late positions; returns orthonormal V (d,k)."""
    X = r[:, late, :].reshape(-1, r.shape[-1]).float()
    y = labels[:, late].reshape(-1)
    mus = torch.stack([X[y == c].mean(0) for c in range(ncls)])
    C = mus - mus.mean(0, keepdim=True)
    U, S, Vt = torch.linalg.svd(C, full_matrices=False)
    k = ncls - 1
    return Vt[:k].T.contiguous()  # (d, k)

# labels at position t (of the residual): phase = types[:, t]; xor feat = seq[t]^seq[t-1]
ph_lab = types[:, : inp.shape[1]]
xor_lab = torch.zeros_like(inp)
xor_lab[:, 1:] = inp[:, 1:] ^ inp[:, :-1]

subspaces = {}
g = torch.Generator(device=dev).manual_seed(0)
for name, lab, ncls in [("phase", ph_lab, 3), ("xor", xor_lab, 2)]:
    subspaces[name] = [class_subspace(r, lab, ncls) for r in resid]
Q, _ = torch.linalg.qr(torch.randn(resid[0].shape[-1], 2, generator=g, device=dev))
subspaces["random"] = [Q for _ in resid]

def perp(A, B):
    """component of subspace A orthogonal to subspace B, re-orthonormalized."""
    P = A - B @ (B.T @ A)
    Q, R = torch.linalg.qr(P)
    keep = R.diagonal().abs() > 1e-3
    return Q[:, keep].contiguous()

subspaces["phase_perp"] = [perp(p, x) for p, x in zip(subspaces["phase"], subspaces["xor"])]
subspaces["xor_perp"] = [perp(x, p) for p, x in zip(subspaces["phase"], subspaces["xor"])]

# overlap check between phase and xor subspaces (max principal angle cosine per layer)
for li in range(5):
    ov = torch.linalg.svd(subspaces["phase"][li].T @ subspaces["xor"][li])[1].max().item()
    print(f"layer {li}: max cos principal angle phase/xor = {ov:.3f}")

mean_by_pos = [r.mean(0, keepdim=True) for r in resid]  # (1, L-1, d) per layer

def forward_clamped(x_in, cond):
    Vs = subspaces.get(cond)
    def clamp(x, li):
        if Vs is None:
            return x
        V = Vs[li]
        dx = x - mean_by_pos[li]
        return x - (dx @ V) @ V.T
    x = model.wte(x_in) + model.wpe(torch.arange(x_in.shape[1], device=dev))[None]
    x = clamp(x, 0)
    for i, blk in enumerate(model.blocks):
        x = clamp(blk(x), i + 1)
    return model.head(model.lnf(x))

def per_type(logits):
    tgt = seq[:, 1:]
    logp = F.log_softmax(logits, dim=-1)
    ce = -logp.gather(-1, tgt[..., None]).squeeze(-1)
    xorval = seq[:, 1:-1] ^ seq[:, :-2]
    pxor = logp[:, 1:].gather(-1, xorval[..., None]).squeeze(-1).exp()
    tt = types[:, 1:]
    pos = torch.arange(1, L, device=dev)
    out = {}
    for ty in range(3):
        m = (tt == ty) & (pos >= MIN)[None, :]
        out[f"ce{ty}"] = ce[m].mean().item()
        m2 = (tt[:, 1:] == ty) & (pos[1:] >= MIN)[None, :]
        out[f"pxor{ty}"] = pxor[m2].mean().item()
    return out

res = {}
with torch.no_grad():
    for cond in ["none", "random", "phase", "xor", "phase_perp", "xor_perp"]:
        res[cond] = per_type(forward_clamped(inp, cond))
        r = res[cond]
        print(f"{cond:7s} ce=({r['ce0']:.4f}, {r['ce1']:.4f}, {r['ce2']:.4f})  "
              f"pxor=({r['pxor0']:.3f}, {r['pxor1']:.3f}, {r['pxor2']:.3f})")

json.dump(res, open(os.path.join(run, "ablate_results.json"), "w"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
fig, ax = plt.subplots(figsize=(8, 4.5))
conds = ["none", "random", "phase", "xor", "phase_perp", "xor_perp"]
xs = np.arange(len(conds))
w = 0.25
for i, (ty, lab, c) in enumerate([(0, "random pos 0", "tab:orange"),
                                  (1, "random pos 1", "tab:red"),
                                  (2, "xor pos", "tab:blue")]):
    ax.bar(xs + (i - 1) * w, [res[c_]["ce" + str(ty)] for c_ in conds], w, label=lab, color=c)
for y, lab in [(0.6931, "ln2"), (0.7520, "stage-1 random"), (0.4055, "stage-1 xor")]:
    ax.axhline(y, ls=":", color="gray"); ax.text(len(conds) - 0.4, y, lab, fontsize=7)
ax.set_xticks(xs); ax.set_xticklabels(["no ablation", "random\nsubspace", "phase\nablated",
                                       "xor\nablated", "phase perp\n(xor kept)",
                                       "xor perp\n(phase kept)"], fontsize=8)
ax.set_ylabel("CE (nats)"); ax.legend()
ax.set_title(f"Ablation time-travel test, seed {seed} (pred: phase-ablated = stage-1 plateau)")
fig.tight_layout()
out = os.path.join(run, "ablate.png")
fig.savefig(out, dpi=140)
print(out)
