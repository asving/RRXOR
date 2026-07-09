"""Correction spectroscopy: the likelihood-ratio adjustment across each sharp loss drop,
and the epsilon-machine (history quotient) it factors through.

Top row (rrxor3): delta(x) = lam_post - lam_pre conditioned on ALL 16 length-4 suffixes
(no hypothesis). Drop 1 must collapse onto xor(prev two) = the 2-state machine; drop 2
must NOT collapse on any short suffix but must sit on the stage-2 statistic
(offset posterior x candidate). Bottom row (m27): 4 drops x 4 theoretical stage
corrections, R^2 matrix -> diagonal dominance.
"""
import glob, json, math, os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from train import GPT

base = os.path.dirname(os.path.abspath(__file__))
dev = "cuda"
CLIP = 8.0

def lam_at(run, step, seq, L, nl):
    ck = sorted(glob.glob(os.path.join(base, "runs", run, "ckpts", "*.pt")))
    steps = np.array([int(os.path.basename(c)[4:10]) for c in ck])
    i = int(np.argmin(np.abs(steps - step)))
    model = GPT(L=L, d=128, nl=nl).to(dev)
    model.load_state_dict(torch.load(ck[i], map_location=dev)); model.eval()
    with torch.no_grad():
        lg = model(seq)
    return (lg[:, :, 1] - lg[:, :, 0]).clamp(-CLIP, CLIP).cpu(), int(steps[i])

def r2_feat(X, y, ntr):
    Xb = torch.cat([X, torch.ones(len(X), 1)], 1)
    W = torch.linalg.lstsq(Xb[:ntr], y[:ntr]).solution
    p = Xb[ntr:] @ W
    return float(1 - ((y[ntr:] - p) ** 2).sum() / ((y[ntr:] - y[:ntr].mean()) ** 2).sum())

# ================= rrxor3 =================
run = "seed0"
ev = torch.load(os.path.join(base, "runs", run, "evalset.pt"))
seq = ev["eseq"][:512].to(dev)
B, L = seq.shape
band = torch.arange(33, 94)          # targets t; query q = t-1
q = band - 1
u_arr = (seq[:, 1:] ^ seq[:, :-1]).cpu()   # u_arr[:, i] = x_{i+1} ^ x_i
cand = u_arr[:, q - 1]                      # x_q ^ x_{q-1} -> candidate for t = q+1
# exact offset posterior (elimination filter) -> q_det at query
v = torch.zeros_like(seq); v[:, 2:] = (seq[:, 2:] != (seq[:, 1:-1] ^ seq[:, :-2])).long()
tt = torch.arange(L, device=dev)
post = torch.zeros(B, L, 3, device=dev)
for o in range(3):
    det = ((o + tt) % 3 == 2) & (tt >= 2)
    killed = ((v.bool() & det[None]).cumsum(1) > 0).float()
    post[:, :, o] = (1 - killed) * torch.pow(2.0, det[None].float().expand(B, -1).cumsum(1))
post = post / post.sum(-1, keepdim=True).clamp_min(1e-30)
qdet = torch.zeros(B, len(band))
for o in range(3):
    qdet += post[:, q, o].cpu() * (((o + band) % 3) == 2).float()[None]

DROPS3 = [(200, 400), (1500, 3000)]
deltas, labels3 = [], []
for pre, post_s in DROPS3:
    l0, s0 = lam_at(run, pre, seq, L, 4)
    l1, s1 = lam_at(run, post_s, seq, L, 4)
    deltas.append((l1 - l0)[:, q])
    labels3.append(f"steps {s0}→{s1}")

suffix = (seq.cpu()[:, q - 3] * 8 + seq.cpu()[:, q - 2] * 4 +
          seq.cpu()[:, q - 1] * 2 + seq.cpu()[:, q])      # 0..15
usuf = (seq.cpu()[:, q] ^ seq.cpu()[:, q - 1])
ntr = int(0.6 * B)

fig = plt.figure(figsize=(17, 9.8), facecolor="white")
gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.26, left=0.06, right=0.98,
                      top=0.84, bottom=0.07)

def suffix_panel(ax, d, title):
    means, sems, cols, xs_ = [], [], [], []
    order = sorted(range(16), key=lambda s: (((s >> 1) ^ s) & 1, s))
    for xi, sfx in enumerate(order):
        m = (suffix == sfx)
        vals = d[m]
        means.append(float(vals.mean())); sems.append(float(vals.std()))
        cols.append("#e0a63b" if ((sfx >> 1) ^ sfx) & 1 else "#5a6acf")
        xs_.append(xi)
    for xi, mu, sd, c in zip(xs_, means, sems, cols):
        ax.errorbar(xi, mu, yerr=sd, fmt="o", ms=6, color=c, ecolor=c, alpha=0.85,
                    elinewidth=1.2, capsize=2)
    # between-suffix variance explained by u alone
    mu_arr = np.array(means)
    uo = np.array([((s >> 1) ^ s) & 1 for s in order], dtype=float)
    pred = np.where(uo > 0, mu_arr[uo > 0].mean(), mu_arr[uo == 0].mean())
    r2u = 1 - ((mu_arr - pred) ** 2).sum() / ((mu_arr - mu_arr.mean()) ** 2).sum()
    ax.axhline(0, color="#ccc", lw=0.8)
    ax.set_xticks(range(16))
    ax.set_xticklabels([format(s, "04b") for s in order], rotation=90, fontsize=7)
    ax.set_xlabel("last 4 tokens before the prediction (all 16 suffixes)", fontsize=9)
    ax.set_title(title, fontsize=10.5, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    return r2u

ax = fig.add_subplot(gs[0, 0])
r2u1 = suffix_panel(ax, deltas[0],
                    f"drop 1 ({labels3[0]}): E[δ | suffix] collapses onto ⊕(last two)")
ax.set_ylabel("δ = λ_post − λ_pre  (log-odds adjustment)", fontsize=9)
ax.text(0.03, 0.90, f"u explains {r2u1:.0%} of between-suffix variance\n"
        "→ the first correction machine IS the 2-state ⊕ machine",
        transform=ax.transAxes, fontsize=8.6, color="#333")
ax.text(0.03, 0.10, "amber: ⊕(last two)=1\nblue: ⊕(last two)=0", transform=ax.transAxes,
        fontsize=8, color="#666")

ax = fig.add_subplot(gs[0, 1])
r2u2 = suffix_panel(ax, deltas[1],
                    f"drop 2 ({labels3[1]}): no short suffix explains δ")
d2 = deltas[1].reshape(-1)
Xsfx = torch.nn.functional.one_hot(suffix.reshape(-1).long(), 16).float()
r2_sfx = r2_feat(Xsfx, d2, int(0.6 * len(d2)))
r2_sfx1 = r2_feat(Xsfx, deltas[0].reshape(-1), int(0.6 * len(d2)))
ax.text(0.03, 0.86, f"whole 16-suffix table: R² = {r2_sfx:.2f} (drop 1: {r2_sfx1:.2f})\n"
        "(error bars = within-suffix spread — the\ncorrection depends on LONG-range state)",
        transform=ax.transAxes, fontsize=8.6, color="#333")

# stage-2 statistic fit: features from (cand, q_det)
sgn = (2 * cand.float() - 1).reshape(-1)
qd = qdet.reshape(-1)
X2 = torch.stack([sgn, sgn * qd, qd], 1)
Xu = sgn[:, None]
n = len(d2); ntr2 = int(0.6 * n)
r2_u_d2 = r2_feat(Xu, d2, ntr2)
r2_s2_d2 = r2_feat(X2, d2, ntr2)
d1 = deltas[0].reshape(-1)
r2_u_d1 = r2_feat(Xu, d1, ntr2)
r2_s2_d1 = r2_feat(X2, d1, ntr2)

ax = fig.add_subplot(gs[0, 2])
Xb = torch.cat([X2, torch.ones(n, 1)], 1)
W = torch.linalg.lstsq(Xb[:ntr2], d2[:ntr2]).solution
pred = (Xb @ W)[ntr2:]
det_next = qd[ntr2:] > 0.5
idx = torch.randperm(len(pred))[:4000]
ax.scatter(pred[idx][det_next[idx]], d2[ntr2:][idx][det_next[idx]], s=4, alpha=0.25,
           color="#3f9e58", label="next slot det (under posterior)")
ax.scatter(pred[idx][~det_next[idx]], d2[ntr2:][idx][~det_next[idx]], s=4, alpha=0.25,
           color="#5a6acf", label="next slot random")
lo, hi = float(pred.min()), float(pred.max())
ax.plot([lo, hi], [lo, hi], color="#30343c", lw=1.2, ls="--")
ax.set_xlabel("stage-2 statistic fit:  a·(2u−1) + b·(2u−1)·q̂_det + c·q̂_det", fontsize=8.6)
ax.set_ylabel("measured δ (drop 2)", fontsize=9)
ax.set_title("drop 2 sits on the offset-posterior × candidate quotient", fontsize=10.5,
             loc="left")
ax.legend(fontsize=7.5, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
ax.text(0.97, 0.06,
        f"held-out R²:  u-only {r2_u_d2:.2f} → +offset-posterior {r2_s2_d2:.2f}\n"
        f"(drop 1 for contrast: u-only {r2_u_d1:.2f} → {r2_s2_d1:.2f}, gain ≈ 0)",
        transform=ax.transAxes, fontsize=8.4, ha="right", color="#333")

# ================= m27: 4 drops x 4 theory corrections =================
run = "m27_nl6_seed1"
ev = torch.load(os.path.join(base, "runs", run, "evalset.pt"))
seq = ev["eseq"][:512].to(dev)
cls = ev["ecls"][:512]
B, L = seq.shape
band = torch.arange(100, L - 1)
qm = band - 1
from train27 import NESTED
rule = torch.tensor(NESTED)
c_t = cls[:, band]                                    # phase of target position
is_det = (c_t % 3 == 2)
slot = torch.where(is_det, ((c_t - 2) // 3) % 9, torch.zeros_like(c_t))
r_t = rule[slot]                                      # 0/1/2 ; correct = cand ^ (r==1)
uc = (seq.cpu()[:, 1:] ^ seq.cpu()[:, :-1])   # uc[:, i] = x_{i+1} ^ x_i
candm = uc[:, qm - 1]                          # x_qm ^ x_{qm-1} -> candidate for target

def pol_lam(k, eta=0.02):
    """log-odds of token=1 under closed-form stage-k policy (PREREG3 table)."""
    s = (2 * candm.float() - 1)
    if k == 0:
        return torch.zeros_like(s)
    if k == 1:
        return s * math.log((16 / 27) / (11 / 27))
    if k == 2:
        return torch.where(is_det, s * math.log((7 / 9) / (2 / 9)), torch.zeros_like(s))
    if k == 3:
        exc = is_det & (slot % 3 == 2)
        lam = torch.where(is_det, s * math.log((1 - eta) / eta), torch.zeros_like(s))
        return torch.where(exc, -s * math.log((2 / 3) / (1 / 3)), lam)
    if k == 4:
        corr_sign = torch.where(r_t == 1, -s, s)
        return torch.where(is_det, corr_sign * math.log((1 - eta) / eta),
                           torch.zeros_like(s))

DSTEPS = [(400, 1000), (2000, 2600), (2600, 3400), (3600, 5600)]
dnet, dlabs = [], []
for pre, post_s in DSTEPS:
    l0, s0 = lam_at(run, pre, seq, L, 6)
    l1, s1 = lam_at(run, post_s, seq, L, 6)
    dnet.append((l1 - l0)[:, qm].reshape(-1))
    dlabs.append(f"{s0}→{s1}")
dth = [(pol_lam(k) - pol_lam(k - 1)).clamp(-CLIP, CLIP).reshape(-1) for k in (1, 2, 3, 4)]
TH_LAB = ["δ*₁ bet on ⊕cand\n(2-state: u)", "δ*₂ gate to det slots\n(mod-3 phase)",
          "δ*₃ XNOR the exceptions\n(mod-9 rule)", "δ*₄ resolve FLIP\n(mod-27 identity)"]

n = len(dnet[0]); ntrm = int(0.6 * n)
M = np.zeros((4, 4))
for i, dn in enumerate(dnet):
    for j, dt in enumerate(dth):
        M[i, j] = max(r2_feat(dt[:, None], dn, ntrm), 0)

axL = fig.add_subplot(gs[1, 0])
evl = [json.loads(l) for l in open(os.path.join(base, "runs", run, "eval.jsonl"))]
st = np.array([e["step"] for e in evl], dtype=float); st[st < 25] = 25
axL.plot(st, [e["ce_avg"] for e in evl], color="#30343c", lw=1.8)
for i, ((a, b), lab) in enumerate(zip(DSTEPS, dlabs)):
    axL.axvspan(a, b, color=f"C{i}", alpha=0.12, lw=0)
    axL.text(math.sqrt(a * b), [0.71, 0.715, 0.685, 0.71][i], f"drop {i+1}",
             fontsize=8, ha="center", color=f"C{i}")
axL.set_xscale("log"); axL.set_xlim(25, 19000); axL.set_ylim(0.44, 0.73)
axL.set_xlabel("training step", fontsize=9); axL.set_ylabel("eval CE (nats)", fontsize=9)
axL.set_title("matryoshka Z27: the four sharp drops", fontsize=10.5, loc="left")
axL.spines[["top", "right"]].set_visible(False)

axM = fig.add_subplot(gs[1, 1:])
im = axM.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
for i in range(4):
    for j in range(4):
        axM.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=11,
                 color="white" if M[i, j] > 0.55 else "#333",
                 fontweight="bold" if i == j else "normal")
axM.set_xticks(range(4)); axM.set_xticklabels(TH_LAB, fontsize=8.6)
axM.set_yticks(range(4))
axM.set_yticklabels([f"drop {i+1}\n({lab})" for i, lab in enumerate(dlabs)], fontsize=8.6)
axM.set_xlabel("theoretical stage correction (closed-form policy differences)", fontsize=9)
axM.set_title("each measured drop is ONE stage's correction — held-out R² of "
              "δ_net(drop i) on δ*(stage j)", fontsize=10.5, loc="left")

fig.suptitle("Correction spectroscopy: what likelihood-ratio adjustment did each loss "
             "drop implement, and what machine does it factor through?",
             fontsize=14.5, fontweight="bold", x=0.06, ha="left", y=0.965)
fig.text(0.06, 0.925,
         "δ(x) = λ_post − λ_pre measured across each sharp loss drop, per held-out history. Top: RRXOR-3 — drop 1's correction is constant on ⊕(last two) classes (the\n"
         "2-state ε-machine, read off with no hypothesis); drop 2's correction is not a function of ANY 4-token suffix — it factors through the offset posterior. Bottom:\n"
         "matryoshka — each drop's adjustment matches exactly its own stage's closed-form correction and no other (diagonal dominance).",
         fontsize=9.3, color="#444", va="top")
fig.savefig(os.path.join(base, "corrspec.png"), dpi=150)
print("saved corrspec.png")
print("rrxor3: suffix-collapse R2(u) drop1", round(r2u1, 3), "drop2", round(r2u2, 3),
      "| suffix-table R2 drop1", round(r2_sfx1, 3), "drop2", round(r2_sfx, 3))
print("rrxor3 R2: drop1 u-only", round(r2_u_d1, 3), "stage2", round(r2_s2_d1, 3),
      "| drop2 u-only", round(r2_u_d2, 3), "stage2", round(r2_s2_d2, 3))
print("m27 matrix:\n", np.round(M, 3))
