"""Whitebox step 1: characterize all heads of the fast seed (L4-L6 focus).

Per head: (i) attention mass vs lag (1..54) at late pre-det queries; (ii) violation
selectivity per lag: E[att | m_src=1] / E[att | m_src=0]; (iii) source-class profile;
(iv) single-head zero-ablation effects (per-class CE + gating split).
"""
import glob, json, math, os, sys
import torch
import torch.nn.functional as F
from train import GPT
from train27 import class_masks

name = sys.argv[1] if len(sys.argv) > 1 else "m27_nl6_seed1"
dev = "cuda"
base = os.path.dirname(os.path.abspath(__file__))
run = os.path.join(base, "runs", name)
cfg = json.load(open(os.path.join(run, "config.json")))
ev = torch.load(os.path.join(run, "evalset.pt"))
seq, cls = ev["eseq"][:1024].to(dev), ev["ecls"][:1024].to(dev)
B, L = seq.shape
nl, nh = cfg["nl"], 4
MIN = 108

model = GPT(L=cfg["L"], d=cfg["d"], nl=nl).to(dev)
cp = sorted(glob.glob(os.path.join(run, "ckpts", "*.pt")))[-1]
model.load_state_dict(torch.load(cp)); model.eval()

@torch.no_grad()
def manual_forward(head_zero=(), want_attn=False):
    x = model.wte(seq) + model.wpe(torch.arange(L, device=dev))[None]
    attns = {}
    for li, blk in enumerate(model.blocks, 1):
        h = blk.ln1(x)
        q, k, v = blk.qkv(h).chunk(3, -1)
        D = q.shape[-1]; hd = D // nh
        q, k, v = (z.view(B, L, nh, hd).transpose(1, 2) for z in (q, k, v))
        att = (q @ k.transpose(-1, -2)) / math.sqrt(hd)
        mask = torch.triu(torch.full((L, L), float("-inf"), device=dev), 1)
        att = (att + mask).softmax(-1)
        if want_attn:
            attns[li] = att                              # (B, nh, L, L) keep on gpu
        o = att @ v
        for (zl, zh) in head_zero:
            if zl == li:
                o[:, zh] = 0
        x = x + blk.proj(o.transpose(1, 2).reshape(B, L, D))
        x = x + blk.mlp(blk.ln2(x))
    return model.head(model.lnf(x)), attns

logits0, attns = manual_forward(want_attn=True)
assert (logits0 - model(seq)).abs().max().item() < 1e-3

m_lab = torch.zeros_like(seq); m_lab[:, 2:] = seq[:, 2:] ^ seq[:, 1:-1] ^ seq[:, :-2]

# query positions: pre-det, late
Q = ((cls % 3 == 1) & (torch.arange(L, device=dev)[None, :] >= MIN)).nonzero()  # (n, 2)
qb, qt = Q[:, 0], Q[:, 1]
LAGS = 54

print("=== lag profiles + violation selectivity at pre-det queries (L4-L6) ===")
prof = {}
for li in range(4, nl + 1):
    for h in range(nh):
        A = attns[li][qb, h][torch.arange(len(qt)), qt]     # rows (n, L): attention of each query
        lagmass, vsel = [], []
        idx = torch.arange(len(qt), device=dev)
        for lag in range(1, LAGS + 1):
            src = qt - lag
            ok = src >= 2
            a = A[idx[ok], src[ok]]
            mv = m_lab[qb[ok], src[ok]].bool()
            lagmass.append(a.mean().item())
            if mv.any() and (~mv).any():
                vsel.append((a[mv].mean() / a[~mv].mean().clamp_min(1e-9)).item())
            else:
                vsel.append(float("nan"))
        prof[f"L{li}h{h}"] = {"lagmass": lagmass, "vsel": vsel}
        top = sorted(range(LAGS), key=lambda i: -lagmass[i])[:5]
        print(f"L{li}h{h}: top lags {[(t+1, round(lagmass[t],3)) for t in top]}  "
              f"| vsel@top {[round(vsel[t],2) for t in top]}")

json.dump(prof, open(os.path.join(run, "wb_heads_prof.json"), "w"))

def metrics(logits):
    logp = F.log_softmax(logits[:, :-1], -1)
    ce = -logp.gather(-1, seq[:, 1:, None]).squeeze(-1)
    xorv = seq[:, 1:-1] ^ seq[:, :-2]
    pxor = logp[:, 1:].gather(-1, xorv[..., None]).squeeze(-1).exp()
    mm = class_masks(cls[:, 1:], False)
    pos = torch.arange(1, L, device=dev)
    late = (pos >= MIN)[None, :]
    out = {}
    for kk in ("rnd", "xor", "xnor", "flip"):
        out[f"ce_{kk}"] = ce[mm[kk] & late].mean().item()
        m2 = mm[kk][:, 1:] & late[:, 1:]
        out[f"px_{kk}"] = pxor[m2].mean().item()
    out["split"] = out["px_xor"] - out["px_rnd"]
    return out

print("\n=== single-head zero-ablations, L4-L6 ===")
res = {"none": metrics(logits0)}
o = res["none"]
print(f"none    ce=({o['ce_rnd']:.3f},{o['ce_xor']:.3f},{o['ce_xnor']:.3f},"
      f"{o['ce_flip']:.3f}) split {o['split']:.3f}")
for li in range(4, nl + 1):
    for h in range(nh):
        o = metrics(manual_forward(head_zero=[(li, h)])[0])
        res[f"L{li}h{h}"] = o
        print(f"L{li}h{h}  ce=({o['ce_rnd']:.3f},{o['ce_xor']:.3f},{o['ce_xnor']:.3f},"
              f"{o['ce_flip']:.3f}) split {o['split']:.3f}")
json.dump(res, open(os.path.join(run, "wb_heads_ablate.json"), "w"))
