"""Rollout validation + figure for the spectral HMM reconstruction (hankel.py)."""
import glob, json, os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from train import GPT
import hankel as H                      # reuses tables/spectral (runs its prints)

base = os.path.dirname(os.path.abspath(__file__))
dev = "cuda"

# ---- rollout: drive the reconstructed OOM with the real continuation tokens ----
H.load(8000)
tab = H.net_cond_table(H.model)
S, ops, Z, Vh = H.spectral(tab)
seq = H.seq
B, Q0 = H.B, H.Q0
with torch.no_grad():
    lgf = H.model(seq)
p1_net = torch.softmax(lgf[:, :, :2], -1)[:, :, 1].cpu()
z = Z.clone()                            # state at position Q0 (from tests)
errs = []
e1 = torch.zeros(len(H.TESTS)); e1[1] = 1.0   # test '1' -> P(next=1)
for k in range(12):
    yhat = z @ Vh[:5]
    p1 = (yhat @ e1).clamp(1e-4, 1 - 1e-4)
    tok = seq[:, Q0 + k].cpu()
    pn = p1_net[:, Q0 + k - 1]
    errs.append(float((p1 - pn).abs().mean()))
    psig = torch.where(tok.bool(), p1, 1 - p1)
    Anext = torch.where(tok.bool()[:, None, None], ops[1][None], ops[0][None])
    z = torch.einsum("bij,bj->bi", Anext, z) / psig[:, None]
print("rollout |p_OOM - p_net| per step:", [round(e, 4) for e in errs])

tab_true = H.tab_true if hasattr(H, "tab_true") else H.true_cond_table()
S_true, ops_true, _, _ = H.spectral(tab_true)
H.load(1000); S_pre, ops_pre, _, _ = H.spectral(H.net_cond_table(H.model))
H.load(3000); S_post, ops_post, _, _ = H.spectral(H.net_cond_table(H.model))
traj = json.load(open(os.path.join(base, "runs", "seed0", "hankel_traj.json")))

fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0), facecolor="white")
fig.subplots_adjust(left=0.05, right=0.985, top=0.78, bottom=0.12, wspace=0.26)

ax = axes[0]
W = 0.2
for i, (S_, lab, c) in enumerate([(S_pre, "net @1000 (pre-drop 2)", "#b8bcc6"),
                                  (S_post, "net @3000 (post-drop 2)", "#7fb3e8"),
                                  (S, "net @8000 (final)", "#2f6fce"),
                                  (S_true, "true generator", "#d63f4f")]):
    v = (S_ / S_[0])[:8].numpy()
    ax.bar(np.arange(8) + (i - 1.5) * W, v, W, color=c, label=lab)
ax.set_xticks(range(8)); ax.set_xticklabels([f"σ{i+1}" for i in range(8)])
ax.set_ylabel("singular value / σ1")
ax.set_title("Hankel spectrum: behavior becomes EXACTLY rank 5\n(net final vs true: match to 3 decimals)",
             fontsize=10.5, loc="left")
ax.legend(fontsize=8)
ax.spines[["top", "right"]].set_visible(False)

ax = axes[1]
th = np.linspace(0, 2 * np.pi, 200)
ax.plot(np.cos(th), np.sin(th), color="#ddd", lw=1)
for a in (0, 2 * np.pi / 3, -2 * np.pi / 3):
    ax.plot(np.cos(a), np.sin(a), "x", color="#d63f4f", ms=12, mew=2)
for ops_, lab, c, m in [(ops_pre, "pre-drop 2", "#b8bcc6", "o"),
                        (ops_post, "post-drop 2", "#7fb3e8", "o"),
                        (ops, "final", "#2f6fce", "o")]:
    e = np.linalg.eigvals((ops_[0] + ops_[1]).numpy())
    ax.plot(e.real, e.imag, m, color=c, ms=7, label=lab, alpha=0.9)
ax.set_xlim(-1.25, 1.25); ax.set_ylim(-1.25, 1.25); ax.set_aspect("equal")
ax.set_title("eigenvalues of the transfer operator A0+A1\n(red ×: cube roots of unity — the phase clock)",
             fontsize=10.5, loc="left")
ax.legend(fontsize=8, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)

ax = axes[2]
st = np.array([t["step"] for t in traj], dtype=float); st[st < 25] = 25
mag = [t["mag"] if abs(t["angle"] - 2.094) < 0.45 else 0.0 for t in traj]
ax.plot(st, mag, "o-", color="#2f6fce", label="|clock eigenvalue| (angle ≈ 2π/3 only)")
ax.axhline(1.0, color="#d63f4f", ls=":", lw=1)
ax.axvspan(550, 2500, color="#3f9e58", alpha=0.08, lw=0)
ax.text(1150, 0.04, "drop 2 era", fontsize=8, color="#3a7d4c", ha="center")
ax.set_xscale("log"); ax.set_xlabel("training step")
ax.set_ylabel("|clock eigenvalue|", color="#2f6fce")
ax2 = ax.twinx()
ax2.plot(st, [t["erank"] for t in traj], "s--", color="#3f9e58", ms=4)
ax2.set_ylabel("effective Hankel rank", color="#3f9e58")
ax2.set_ylim(0, 10); ax2.axhline(5, color="#3f9e58", ls=":", lw=0.8)
ax.set_title("the clock is installed across drop 2\n(rank overshoots 9 mid-transition, compresses to 5)",
             fontsize=10.5, loc="left")
ax.legend(fontsize=8, loc="center left")
ax.spines[["top"]].set_visible(False)

fig.suptitle("Rebuilding the HMM from behavior alone: spectral (Hankel/OOM) reconstruction from the net's conditionals — "
             "RRXOR-3, histories at position 60, tests ≤ 4",
             fontsize=13, fontweight="bold", x=0.05, ha="left", y=0.97)
fig.text(0.05, 0.885,
         f"Final net: clock eigenpair 0.998∠2.0944 (true: 1.000∠2.0944 = e^(2πi/3)); rank-5 OOM rolled autoregressively tracks the net at "
         f"|Δp| = {errs[0]:.3f}…{errs[-1]:.3f} over 12 steps beyond its 4-step fitting horizon.",
         fontsize=9.3, color="#444", va="top")
fig.savefig(os.path.join(base, "hankel_fig.png"), dpi=150)
print("saved hankel_fig.png")
