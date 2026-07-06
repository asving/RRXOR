"""Whitebox step 3: synthetic program + stitching.

Program A (filter): lambda_t = (2u_t-1) * (a*Lambda_t + b)   [a,b fitted on train seqs]
Program B (comb):   lambda_t = (2u_t-1) * (w0 + sum_l w_l m_{t-l})  [81-tap ridge comb]
Behavioral: mean KL(net || program) per token on held-out seqs, vs KL(net || Bayes).

Stitch at L4: reconstruct resid4 from named bits (x_t, x_{t-1}, x_{t-2}, u_t, u_{t-1},
m_t, m_{t-1}) + model's own wpe via a fitted affine embed; run REAL blocks 5-6 + head.
If KL(net || stitched) ~ 0, the named variables are sufficient at the cut.
"""
import glob, json, math, os, sys
import torch
import torch.nn.functional as F
from train import GPT
from train27 import NESTED

name = sys.argv[1] if len(sys.argv) > 1 else "m27_nl6_seed1"
dev = "cuda"
base = os.path.dirname(os.path.abspath(__file__))
run = os.path.join(base, "runs", name)
cfg = json.load(open(os.path.join(run, "config.json")))
ev = torch.load(os.path.join(run, "evalset.pt"))
seq, cls = ev["eseq"].to(dev), ev["ecls"].to(dev)
B, L = seq.shape
rule = torch.tensor(NESTED, device=dev)
TR = int(0.7 * B)                                       # train/test split by sequence

# ---- ground truth variables ----
u = torch.zeros_like(seq); u[:, 1:] = seq[:, 1:] ^ seq[:, :-1]
m = torch.zeros_like(seq); m[:, 2:] = seq[:, 2:] ^ seq[:, 1:-1] ^ seq[:, :-2]

@torch.no_grad()
def beliefs27(seq):
    pi = torch.full((B, 27), 1 / 27, device=dev)
    out = torch.zeros(B, L, 27, device=dev)
    classes = torch.arange(27, device=dev)
    det = classes % 3 == 2
    crule = rule[(classes // 3) % 9]
    out[:, 0] = pi
    for t in range(1, L):
        pi = pi.roll(1, dims=1)
        if t >= 2:
            uu = (seq[:, t - 1] ^ seq[:, t - 2]).float()
            x = seq[:, t].float()
            pred = (uu[:, None] + crule[None, :].float()) % 2
            lik = torch.where(det[None, :], (pred == x[:, None]).float(),
                              torch.full_like(pred, 0.5))
            pi = pi * lik
        pi = pi / pi.sum(1, keepdim=True).clamp_min(1e-30)
        out[:, t] = pi
    return out

PI = beliefs27(seq)
classes = torch.arange(27, device=dev)
nxt_det = (classes + 1) % 3 == 2
nrule = rule[(((classes + 1) % 27) // 3) % 9]
qR = PI[:, :, ~nxt_det].sum(-1)
qX = PI[:, :, nxt_det & (nrule == 0)].sum(-1)
qN = PI[:, :, nxt_det & (nrule == 1)].sum(-1)
CLIP = 12.0
Lam = (torch.log((qX + qR / 2).clamp_min(1e-9)) -
       torch.log((qN + qR / 2).clamp_min(1e-9))).clamp(-CLIP, CLIP)
p_bayes1 = torch.where(u.bool(), (qX + qR / 2), (qN + qR / 2))  # P(next=1)

# ---- net forward (manual, exposing resid) ----
model = GPT(L=cfg["L"], d=cfg["d"], nl=cfg["nl"]).to(dev)
cp = sorted(glob.glob(os.path.join(run, "ckpts", "*.pt")))[-1]
model.load_state_dict(torch.load(cp)); model.eval()
nh = 4

@torch.no_grad()
def forward_from(x, first_block):
    for blk in model.blocks[first_block:]:
        h = blk.ln1(x)
        q, k, v = blk.qkv(h).chunk(3, -1)
        D = q.shape[-1]; hd = D // nh
        q, k, v = (z.view(B, L, nh, hd).transpose(1, 2) for z in (q, k, v))
        att = ((q @ k.transpose(-1, -2)) / math.sqrt(hd) +
               torch.triu(torch.full((L, L), float("-inf"), device=dev), 1)).softmax(-1)
        x = x + blk.proj((att @ v).transpose(1, 2).reshape(B, L, D))
        x = x + blk.mlp(blk.ln2(x))
    return model.head(model.lnf(x))

with torch.no_grad():
    _, resid = model(seq, return_resid=True)
logits_net = model(seq)
lam_net = (logits_net[:, :, 1] - logits_net[:, :, 0])

def kl_binary(lam_p, lam_q, mask):
    p = torch.sigmoid(lam_p[mask]); q = torch.sigmoid(lam_q[mask])
    p, q = p.clamp(1e-6, 1 - 1e-6), q.clamp(1e-6, 1 - 1e-6)
    return (p * (p / q).log() + (1 - p) * ((1 - p) / (1 - q)).log()).mean().item()

pos = torch.arange(L, device=dev)
test = torch.zeros(B, L, dtype=torch.bool, device=dev)
test[TR:, 4:L - 1] = True

# ---- Program A: filter + fitted scalar calibration ----
s_net = ((2 * u.float() - 1) * lam_net).clamp(-CLIP, CLIP)
trm = torch.zeros_like(test); trm[:TR, 4:L - 1] = True
Xc = torch.stack([Lam[trm], torch.ones_like(Lam[trm])], 1)
ab = torch.linalg.lstsq(Xc, s_net[trm][:, None]).solution.squeeze()
lam_progA = (2 * u.float() - 1) * (ab[0] * Lam + ab[1])
print(f"calibration fit: s ~ {ab[0].item():.3f} * Lambda + {ab[1].item():.3f}")

# ---- Program B: 81-tap violation comb (fit on train seqs, pre-det + all positions) ----
NL = 81
tpos = torch.arange(NL + 2, L, device=dev)
Xl = torch.stack([m[:, tpos - l].float() for l in range(NL)], -1)
Ytr = s_net[:, tpos][:TR].reshape(-1)
Xtr = Xl[:TR].reshape(-1, NL)
Xb = torch.cat([Xtr, torch.ones_like(Xtr[:, :1])], 1)
Wc = torch.linalg.solve(Xb.T @ Xb + 10.0 * torch.eye(NL + 1, device=dev), Xb.T @ Ytr)
s_progB = torch.zeros_like(s_net)
s_progB[:, tpos] = (Xl @ Wc[:-1] + Wc[-1])
lam_progB = (2 * u.float() - 1) * s_progB
testB = test.clone(); testB[:, :NL + 2] = False

predet_test = testB & (cls % 3 == 1)
print("\n=== behavioral KL(net || program), held-out seqs, per token (nats) ===")
print(f"KL(net || Bayes-filter program A) = {kl_binary(lam_net, lam_progA, test):.4f}")
print(f"KL(net || 81-tap comb program B)  = {kl_binary(lam_net, lam_progB, testB):.4f}"
      f"   [pre-det positions only: {kl_binary(lam_net, lam_progB, predet_test):.4f}]")
lam_bayes = (2 * u.float() - 1) * Lam
print(f"KL(net || raw Bayes, no calib)    = {kl_binary(lam_net, lam_bayes, test):.4f}")
ce_net = F.cross_entropy(logits_net[TR:, 108:-1].reshape(-1, 2),
                         seq[TR:, 109:].reshape(-1)).item()
print(f"(net late CE {ce_net:.4f}; Bayes optimum 0.4621)")

# ---- Stitch at L4: PER-POSITION affine embed of named bits -> resid4 ----
x0, x1, x2 = seq.float(), seq.roll(1, 1).float(), seq.roll(2, 1).float()
u1, m1, m2 = u.roll(1, 1).float(), m.roll(1, 1).float(), m.roll(2, 1).float()
feats = torch.stack([x0, x1, x2, u.float(), u1, m.float(), m1, m2], -1)  # (B,L,8)
nf = feats.shape[-1]
recon4 = resid[4].clone()
for t in range(4, L):
    Ftr = torch.cat([feats[:TR, t], torch.ones(TR, 1, device=dev)], 1)
    Wt = torch.linalg.solve(Ftr.T @ Ftr + 1.0 * torch.eye(nf + 1, device=dev),
                            Ftr.T @ resid[4][:TR, t])
    recon4[:, t] = torch.cat([feats[:, t], torch.ones(B, 1, device=dev)], 1) @ Wt
vm = torch.zeros(B, L, dtype=torch.bool, device=dev); vm[:, 4:] = True
r2 = (1 - ((resid[4] - recon4)[TR:][vm[TR:]] ** 2).sum() /
      ((resid[4][TR:][vm[TR:]] - resid[4][TR:][vm[TR:]].mean(0)) ** 2).sum()).item()
logits_st = forward_from(recon4, 4)
print(f"\n=== stitch at L4 (per-position affine embed of 8 named bits; REAL blocks 5-6) ===")
print(f"resid4 reconstruction R2 (held-out) = {r2:.3f}")
print(f"KL(net || stitched) = {kl_binary(lam_net, logits_st[:, :, 1] - logits_st[:, :, 0], test):.4f}")
ce_st = F.cross_entropy(logits_st[TR:, 108:-1].reshape(-1, 2),
                        seq[TR:, 109:].reshape(-1)).item()
print(f"stitched late CE = {ce_st:.4f}  (net {ce_net:.4f}, Bayes 0.4621)")
