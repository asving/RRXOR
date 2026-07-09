"""PREREG5: correction-conditional geometry — probe-trajectory D_l(t) over checkpoints.

For each run: at every checkpoint, ridge-probe the latent battery from all-layer concat
residuals at query positions; subtract the (t-independent) window-16 baseline; save
trajectories + stage eras. Usage: corrprobe.py <run> <kind>   kind: fx | m27 | rrxor3
"""
import glob, json, math, os, sys
import torch
import numpy as np
from train import GPT

name, kind = sys.argv[1], sys.argv[2]
dev = "cuda"
base = os.path.dirname(os.path.abspath(__file__))
run = os.path.join(base, "runs", name)
cfgp = os.path.join(run, "config.json")
cfg = json.load(open(cfgp)) if os.path.exists(cfgp) else {"nl": 4, "d": 128}
ev = torch.load(os.path.join(run, "evalset.pt"))
seq = ev["eseq"][:512].to(dev)
B, L = seq.shape
NL, D = cfg.get("nl", 4), cfg.get("d", 128)
CLIP = 8.0

u_lab = torch.zeros_like(seq); u_lab[:, 1:] = seq[:, 1:] ^ seq[:, :-1]
m_lab = torch.zeros_like(seq); m_lab[:, 2:] = seq[:, 2:] ^ seq[:, 1:-1] ^ seq[:, :-2]

# ---------------- oracles + latent batteries ----------------
def filter_pi(seq, eps):
    sc = seq.cpu()
    Bs, Ls = sc.shape
    pi = torch.full((Bs, 5), 1 / 5)
    PI = torch.zeros(Bs, Ls, 5)
    for t in range(Ls):
        PI[:, t] = pi
        tok = sc[:, t]
        lik = torch.stack([torch.full((Bs,), .5)] * 3 +
                          [(tok == 0).float(), (tok == 1).float()], 1)
        w = pi * lik
        w = w / w.sum(1, keepdim=True).clamp_min(1e-30)
        nxt = torch.zeros_like(w)
        t0 = (tok == 0).float(); t1 = (tok == 1).float()
        nxt[:, 1] += w[:, 0] * t0; nxt[:, 2] += w[:, 0] * t1
        nxt[:, 3] += w[:, 1] * t0 + w[:, 2] * t1
        nxt[:, 4] += w[:, 1] * t1 + w[:, 2] * t0
        nxt[:, 0] += w[:, 3] + w[:, 4]
        pi = (1 - eps) * nxt + eps / 5
    return PI

if kind == "fx":
    eps = cfg["eps"]
    PI = filter_pi(seq, eps).to(dev)                       # belief BEFORE token t
    band = torch.arange(45, 94, device=dev); qpos = band - 1
    p1 = PI[:, :, 4] + PI[:, :, :3].sum(-1) / 2
    cand = torch.zeros_like(seq); cand[:, 2:] = seq[:, 1:-1] ^ seq[:, :-2]
    pc = torch.where(cand.bool(), p1, 1 - p1).clamp(1e-4, 1 - 1e-4)
    Lam = torch.log(pc / (1 - pc)).clamp(-CLIP, CLIP)
    Q = torch.stack([PI[:, :, 3] + PI[:, :, 4], PI[:, :, 4] - PI[:, :, 3]], -1)
    Fib = torch.stack([PI[:, :, 0] - PI[:, :, 1] - PI[:, :, 2],
                       PI[:, :, 1] - PI[:, :, 2]], -1)
    latents = {"u": u_lab[:, qpos, None].float(), "m": m_lab[:, qpos, None].float(),
               "Lam": Lam[:, band, None], "Q": Q[:, band], "F": Fib[:, band]}
elif kind == "rrxor3":
    # 3-hypothesis offset posterior with exact elimination; band 33..94
    band = torch.arange(33, 94, device=dev); qpos = band - 1
    v = torch.zeros_like(seq); v[:, 2:] = (seq[:, 2:] != (seq[:, 1:-1] ^ seq[:, :-2])).long()
    t_idx = torch.arange(L, device=dev)
    post = torch.zeros(B, L, 3, device=dev)
    for o in range(3):
        det = ((o + t_idx) % 3 == 2) & (t_idx >= 2)
        killed = ((v.bool() & det[None]).cumsum(1) > 0).float()   # any det violation so far
        cnt = det[None].float().expand(B, -1).cumsum(1)
        post[:, :, o] = (1 - killed) * torch.pow(2.0, cnt)
    post = post / post.sum(-1, keepdim=True).clamp_min(1e-30)
    # q_det at query q for target t=q+1: P(t is det) under posterior at q
    qdet = torch.zeros(B, L - 1, device=dev)
    for o in range(3):
        qdet += post[:, :-1, o] * (((o + t_idx[1:]) % 3) == 2).float()[None]
    Lam3 = torch.log((1 + qdet.clamp(0, 1 - 1e-6)) / (1 - qdet.clamp(0, 1 - 1e-6))).clamp(-CLIP, CLIP)
    latents = {"u": u_lab[:, qpos, None].float(), "Q3": post[:, qpos],
               "Lam": Lam3[:, qpos, None]}
else:  # m27
    from train27 import NESTED
    rule = torch.tensor(NESTED, device=dev)
    pi = torch.full((B, 27), 1 / 27, device=dev)
    PI27 = torch.zeros(B, L, 27, device=dev)
    classes = torch.arange(27, device=dev)
    det = classes % 3 == 2
    crule = rule[(classes // 3) % 9]
    PI27[:, 0] = pi
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
        PI27[:, t] = pi
    band = torch.arange(100, L - 2, device=dev); qpos = band
    # orthonormal quotient bases in R^27 (constant removed)
    def orth_against(vecs, prev):
        Vs = []
        for v in vecs:
            v = v.clone().float()
            for p in prev + Vs:
                v -= (v @ p) * p
            n = v.norm()
            if n > 1e-6:
                Vs.append(v / n)
        return Vs
    const = [torch.ones(27, device=dev) / math.sqrt(27)]
    ind3 = [((classes % 3) == c).float() for c in range(3)]
    B3 = orth_against(ind3, const)                              # 2 dims
    ind9 = [((classes % 9) == c).float() for c in range(9)]
    B9 = orth_against(ind9, const + B3)                         # 6 dims
    cell7 = [7, 16, 25]                                         # pre-exception queries
    exc_vecs = [torch.zeros(27, device=dev).index_put_((torch.tensor([7, 16], device=dev),),
                torch.tensor([1., -1.], device=dev)),
                torch.zeros(27, device=dev).index_put_((torch.tensor(cell7, device=dev),),
                torch.tensor([1., 1., -2.], device=dev))]
    BE = orth_against(exc_vecs, const + B3 + B9)                # 2 dims
    use_vecs = []
    for c in range(7):                                          # cells 0..6 only
        mem = [p for p in range(27) if p % 9 == c]
        v1 = torch.zeros(27, device=dev); v1[mem[0]], v1[mem[1]] = 1., -1.
        v2 = torch.zeros(27, device=dev); v2[mem[0]], v2[mem[1]], v2[mem[2]] = 1., 1., -2.
        use_vecs += [v1, v2]
    BU = orth_against(use_vecs, const + B3 + B9 + BE)           # 14 dims
    P = PI27[:, qpos]
    # 1-bit correction statistics (functions of current phase): rule of the NEXT det
    cls = ev["ecls"][:512].to(dev)
    phase = cls[:, qpos]                                        # current phase 0..26
    nxt_det_slot = ((phase // 3) + ((phase % 3) == 2).long()) % 9  # index i of next det
    g9 = (nxt_det_slot % 3 == 2).float()                        # next det is exception-class
    g27 = (nxt_det_slot == 8).float()                           # next det is FLIP
    latents = {"u": u_lab[:, qpos, None].float(),
               "Q3": P @ torch.stack(B3, 1), "Q9x": P @ torch.stack(B9, 1),
               "EXC": P @ torch.stack(BE, 1), "USE": P @ torch.stack(BU, 1),
               "g9": g9[:, :, None], "g27": g27[:, :, None]}
    # pre-det restricted variants for the fine quotients (JIT clause)
    predet = (cls % 3 == 2)[:, qpos + 1]

# ---------------- probes ----------------
NP = len(qpos)
ntr = int(0.6 * B)

def ridge_multi(X, Y, mask=None):
    """X: (B,P,dx)  Y: (B,P,dy) -> held-out R2 (pooled over dims)."""
    if mask is None:
        Xtr, Ytr = X[:ntr].reshape(-1, X.shape[-1]), Y[:ntr].reshape(-1, Y.shape[-1])
        Xte, Yte = X[ntr:].reshape(-1, X.shape[-1]), Y[ntr:].reshape(-1, Y.shape[-1])
    else:
        Xtr, Ytr = X[:ntr][mask[:ntr]], Y[:ntr][mask[:ntr]]
        Xte, Yte = X[ntr:][mask[ntr:]], Y[ntr:][mask[ntr:]]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    if float(((Ytr - Ytr.mean(0)) ** 2).sum()) < 1e-8:
        return float("nan")                                   # degenerate target
    Xtr = torch.cat([(Xtr - mu) / sd, torch.ones_like(Xtr[:, :1])], 1)
    Xte = torch.cat([(Xte - mu) / sd, torch.ones_like(Xte[:, :1])], 1)
    nv = int(0.75 * Xtr.shape[0])
    A = Xtr[:nv].T @ Xtr[:nv]; bb = Xtr[:nv].T @ Ytr[:nv]
    Av = Xtr[nv:]; best = (-1e9, None)
    for lam in (1.0, 10.0, 100.0, 1000.0):
        W = torch.linalg.solve(A + lam * torch.eye(A.shape[0], device=dev), bb)
        r2 = 1 - ((Ytr[nv:] - Av @ W) ** 2).sum() / ((Ytr[nv:] - Ytr[:nv].mean(0)) ** 2).sum()
        if r2 > best[0]:
            best = (float(r2), W)
    W = best[1]
    return float(1 - ((Yte - Xte @ W) ** 2).sum() / ((Yte - Ytr.mean(0)) ** 2).sum())

def concat_feats(resids):
    return torch.cat([r[:, qpos] for r in resids], -1).float()

# window baseline (t-independent)
wf = []
for lag in range(16):
    for src in (seq, u_lab, m_lab):
        f = src.roll(lag, dims=1).float(); f[:, :lag] = 0; wf.append(f)
WX = torch.stack(wf, -1)[:, qpos]
baseline = {k: ridge_multi(WX, Y) for k, Y in latents.items()}
print("window baselines:", {k: round(v, 3) for k, v in baseline.items()})
PD_KEYS = ("Q9x", "EXC", "USE", "g9", "g27")
if kind == "m27":
    base_pd = {k: ridge_multi(WX, latents[k], mask=predet) for k in PD_KEYS}
    print("  (pre-det restricted):", {k: round(v, 3) for k, v in base_pd.items()})

model = GPT(L=L, d=D, nl=NL).to(dev)
ckpts = sorted(glob.glob(os.path.join(run, "ckpts", "*.pt")))
steps = [int(os.path.basename(c)[4:10]) for c in ckpts]
traj = {k: [] for k in latents}
traj_pd = {k: [] for k in PD_KEYS} if kind == "m27" else None
for ci, cp in enumerate(ckpts):
    model.load_state_dict(torch.load(cp, map_location=dev)); model.eval()
    with torch.no_grad():
        _, resids = model(seq, return_resid=True)
    X = concat_feats(resids)
    for k, Y in latents.items():
        traj[k].append(ridge_multi(X, Y) - baseline[k])
    if kind == "m27":
        for k in traj_pd:
            traj_pd[k].append(ridge_multi(X, latents[k], mask=predet) - base_pd[k])
    if ci % 10 == 0:
        print(f"  ckpt {steps[ci]:6d}: " +
              " ".join(f"{k} {traj[k][-1]:+.3f}" for k in latents))

out = {"steps": steps, "traj": traj, "baseline": baseline,
       "traj_predet": traj_pd,
       "eval_loss": [json.loads(l) for l in open(os.path.join(run, "eval.jsonl"))]
       if os.path.exists(os.path.join(run, "eval.jsonl")) else None}
json.dump(out, open(os.path.join(run, "corrprobe.json"), "w"))
print(f"saved {run}/corrprobe.json")
