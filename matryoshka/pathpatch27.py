"""Tests (a) and (c): who consumes which xor copy, and is the transport shared?

(a) Clean the class-mean direction of u (xor value, x_t^x_{t-1}) vs m (violation bit,
    x_t^x_{t-1}^x_{t-2}) from block deltas of layer groups {1-3},{4},{5,6}; measure
    per-class CE + gating split. Locates the evidence channel vs the answer channel.
(c) Attention-offset profiles for every head + single-head zero-ablations at L1-3;
    if one head's removal damages both value and phase, transport is shared.
"""
import glob, json, os, sys, math
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
MIN = 108
nl, nh = cfg["nl"], 4

model = GPT(L=cfg["L"], d=cfg["d"], nl=nl).to(dev)
cp = sorted(glob.glob(os.path.join(run, "ckpts", "*.pt")))[-1]
model.load_state_dict(torch.load(cp)); model.eval()

@torch.no_grad()
def manual_forward(head_zero=(), clean=None, want_attn=False):
    """head_zero: iterable of (layer1idx, head). clean: {layer1idx: (d,k) orthonormal}."""
    x = model.wte(seq) + model.wpe(torch.arange(L, device=dev))[None]
    resid = [x]
    attns = {}
    for li, blk in enumerate(model.blocks, 1):
        x_in = x
        h = blk.ln1(x)
        q, k, v = blk.qkv(h).chunk(3, -1)
        D = q.shape[-1]; hd = D // nh
        q, k, v = (z.view(B, L, nh, hd).transpose(1, 2) for z in (q, k, v))
        att = (q @ k.transpose(-1, -2)) / math.sqrt(hd)
        mask = torch.triu(torch.full((L, L), float("-inf"), device=dev), 1)
        att = (att + mask).softmax(-1)
        if want_attn:
            attns[li] = att.mean(0)                      # (nh, L, L)
        o = att @ v                                       # (B, nh, L, hd)
        for (zl, zh) in head_zero:
            if zl == li:
                o[:, zh] = 0
        x = x + blk.proj(o.transpose(1, 2).reshape(B, L, D))
        x = x + blk.mlp(blk.ln2(x))
        if clean and li in clean:
            V = clean[li]
            d = x - x_in
            x = x_in + d - (d @ V) @ V.T
        resid.append(x)
    logits = model.head(model.lnf(x))
    return logits, resid, attns

# verify harness
logits0, resid0, attns = manual_forward(want_attn=True)
ref = model(seq)
print(f"recompute max|diff| = {(logits0 - ref).abs().max().item():.2e}")
assert (logits0 - ref).abs().max().item() < 1e-3

# ---------- metrics ----------
def metrics(logits):
    logp = F.log_softmax(logits[:, :-1], -1)
    ce = -logp.gather(-1, seq[:, 1:, None]).squeeze(-1)
    xorv = seq[:, 1:-1] ^ seq[:, :-2]
    pxor = logp[:, 1:].gather(-1, xorv[..., None]).squeeze(-1).exp()
    m = class_masks(cls[:, 1:], cfg.get("flat", False))
    pos = torch.arange(1, L, device=dev)
    late = (pos >= MIN)[None, :]
    out = {}
    for kk in ("rnd", "xor", "xnor", "flip"):
        mm = m[kk] & late
        out[f"ce_{kk}"] = ce[mm].mean().item()
        mm2 = m[kk][:, 1:] & late[:, 1:]
        out[f"px_{kk}"] = pxor[mm2].mean().item()
    out["split"] = out["px_xor"] - out["px_rnd"]
    return out

def show(tag, o):
    print(f"{tag:24s} ce(rnd,xor,xnor,flip)=({o['ce_rnd']:.3f},{o['ce_xor']:.3f},"
          f"{o['ce_xnor']:.3f},{o['ce_flip']:.3f})  px_xor {o['px_xor']:.3f}  "
          f"px_rnd {o['px_rnd']:.3f}  split {o['split']:.3f}")

results = {"none": metrics(logits0)}
show("none", results["none"])

# ---------- (a) variable x layer-group cleaning ----------
u_lab = torch.zeros_like(seq); u_lab[:, 1:] = seq[:, 1:] ^ seq[:, :-1]
m_lab = torch.zeros_like(seq); m_lab[:, 2:] = seq[:, 2:] ^ seq[:, 1:-1] ^ seq[:, :-2]
valid = torch.zeros(B, L, dtype=torch.bool, device=dev); valid[:, 2:] = True

def dir_of(labels, li):
    D = (resid0[li] - resid0[li - 1])[valid].float()
    y = labels[valid]
    vvec = D[y == 1].mean(0) - D[y == 0].mean(0)
    return (vvec / vvec.norm().clamp_min(1e-8))[:, None]   # (d,1)

# where is each variable written? (delta probe via class-mean projection acc)
for lab, labels in [("u", u_lab), ("m", m_lab)]:
    accs = []
    for li in range(1, nl + 1):
        D = (resid0[li] - resid0[li - 1])[valid].float()
        y = labels[valid]
        w = dir_of(labels, li)[:, 0]
        s = D @ w
        thr = s.median()
        accs.append(max(((s > thr).long() == y).float().mean().item(),
                        ((s <= thr).long() == y).float().mean().item()))
    print(f"class-mean readout of {lab} in each delta: " +
          " ".join(f"L{i+1}:{a:.2f}" for i, a in enumerate(accs)))

GROUPS = {"L1-3": [1, 2, 3], "L4": [4], "L5-6": [5, 6]}
g = torch.Generator(device=dev).manual_seed(0)
rnd = torch.linalg.qr(torch.randn(cfg["d"], 1, generator=g, device=dev))[0]
for var, labels in [("u", u_lab), ("m", m_lab), ("rand", None)]:
    for gname, layers in GROUPS.items():
        clean = {li: (rnd if labels is None else dir_of(labels, li)) for li in layers}
        o = metrics(manual_forward(clean=clean)[0])
        results[f"{var}@{gname}"] = o
        show(f"{var}@{gname}", o)

# ---------- (c) attention profiles + per-head knockouts at L1-3 ----------
print("\nattention mass by offset (late positions), layers 1-3:")
pos_idx = torch.arange(MIN, L - 1, device=dev)
for li in range(1, 4):
    for h in range(nh):
        A = attns[li][h][pos_idx]                          # (P, L)
        offs = []
        for o_ in range(0, 5):
            offs.append(A[torch.arange(len(pos_idx)), pos_idx - o_].mean().item())
        print(f"  L{li}h{h}: self {offs[0]:.2f}  -1 {offs[1]:.2f}  -2 {offs[2]:.2f}  "
              f"-3 {offs[3]:.2f}  -4 {offs[4]:.2f}")

print("\nsingle-head zero-ablations, layers 1-3:")
for li in range(1, 4):
    for h in range(nh):
        o = metrics(manual_forward(head_zero=[(li, h)])[0])
        results[f"zero_L{li}h{h}"] = o
        show(f"zero L{li}h{h}", o)

json.dump(results, open(os.path.join(run, "pathpatch.json"), "w"))
print("\nsaved", os.path.join(run, "pathpatch.json"))
