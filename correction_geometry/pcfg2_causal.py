"""Causal test of compositionality (PREREG10-lite): patch the level-3 (pair)
constituent representation along its ENCODER image (class-mean map, the validated
causal-subspace method) and test whether deeper beliefs + output move as BP predicts
under the swapped constituent symbol.

Setup: tree 2 (global cols 32..47), intervene on pair(4,5) symbol (in-tree pair
index 2), readout at query col 39 predicting leaf 8 (the b=3 boundary, which needs
root inference consuming the parsed left half). BP-cf oracle: force that pair's
upward message to the donor symbol.
"""
import glob, json, os
import torch
import numpy as np
from train_multi import GPTV
import pcfg2 as P

base = os.path.dirname(os.path.abspath(__file__))
dev = "cuda"
run = "pcfg2_dense_seed0"
ev = torch.load(os.path.join(base, "runs", run, "evalset.pt"))
seq = ev["eseq"][:2048].to(dev)
B = seq.shape[0]
T, Q = 2, 39                       # tree 2; query col 39 -> target = global col 40 (leaf 8)
lv = seq[:, 32:48].cpu()

# ---- deterministic bottom-up parse of tree 2 (unambiguous grammar) ----
inv = []
for lev in range(P.DEPTH):
    d = {}
    for a in range(P.V):
        for k in range(P.M):
            d[(int(P.TABLES[lev][a, k, 0]), int(P.TABLES[lev][a, k, 1]))] = a
    inv.append(d)
def parse_up(x):                    # x: (B, 16) -> dict level -> (B, n)
    out = {4: x}
    for lev in range(4, 0, -1):
        L, R = out[lev][:, 0::2], out[lev][:, 1::2]
        par = torch.zeros_like(L)
        for i in range(L.shape[1]):
            for b in range(B):
                par[b, i] = inv[lev - 1][(int(L[b, i]), int(R[b, i]))]
        out[lev - 1] = par
    return out
PARSE = parse_up(lv)
pair45 = PARSE[3][:, 2]             # symbol of pair (4,5)

# ---- BP-cf oracle: force pair(4,5) upward message to symbol s ----
def bp_forced(force_sym=None):
    lam = {4: torch.ones(B, 16, P.V, dtype=torch.double)}
    for j in range(8):              # seen leaves 0..7
        lam[4][:, j] = 0.0
        lam[4][torch.arange(B), j, lv[:, j]] = 1.0
    beliefs = {}
    for lev in range(4, 0, -1):
        tab = P.TABLES[lev - 1]
        lamL, lamR = lam[lev][:, 0::2], lam[lev][:, 1::2]
        up = torch.zeros(B, lamL.shape[1], P.V, dtype=torch.double)
        for a in range(P.V):
            for k in range(P.M):
                Lc, Rc = int(tab[a, k, 0]), int(tab[a, k, 1])
                up[:, :, a] += lamL[:, :, Lc] * lamR[:, :, Rc] / P.M
        lam[lev - 1] = up
        if lev == 4 and force_sym is not None:
            fs = torch.zeros(B, P.V, dtype=torch.double)
            fs[torch.arange(B), force_sym] = 1.0
            lam[3] = lam[3].clone()
            # pair(4,5) lives at level-3 index 2; wait: lam[3] built from lam[4]
        # apply force after building the pair level
        if lev - 1 == 3 and force_sym is not None:
            fs = torch.zeros(B, P.V, dtype=torch.double)
            fs[torch.arange(B), force_sym] = 1.0
            lam[3][:, 2] = fs
    # downward to leaf 8: path root -> half(8..15) idx1 -> quad(8..11) idx2 -> pair(8,9) idx4 -> leaf8
    pi = torch.full((B, P.V), 1 / P.V, dtype=torch.double)
    post = {}
    root_post = pi * lam[0][:, 0]
    post["root"] = root_post / root_post.sum(-1, keepdim=True).clamp_min(1e-30)
    tgt = 8
    for lev in range(1, 5):
        tab = P.TABLES[lev - 1]
        child = tgt >> (4 - lev)
        sib = child ^ 1
        is_left = (child % 2 == 0)
        newpi = torch.zeros(B, P.V, dtype=torch.double)
        lam_sib = lam[lev][:, sib]
        for a in range(P.V):
            for k in range(P.M):
                Lc, Rc = int(tab[a, k, 0]), int(tab[a, k, 1])
                cval, sval = (Lc, Rc) if is_left else (Rc, Lc)
                newpi[:, cval] += pi[:, a] * lam_sib[:, sval] / P.M
        pi = newpi / newpi.sum(-1, keepdim=True).clamp_min(1e-30)
        if lev < 4:
            pb = pi * lam[lev][:, child]
            post[f"path{lev}"] = pb / pb.sum(-1, keepdim=True).clamp_min(1e-30)
    post["pred"] = pi
    return post

clean = bp_forced(None)
g = torch.Generator().manual_seed(3)
donor = (pair45 + 1 + torch.randint(0, 3, (B,), generator=g)) % 4
cf = bp_forced(donor)

model = GPTV(L=96, nl=6, vocab=P.V).to(dev)
ck = sorted(glob.glob(os.path.join(base, "runs", run, "ckpts", "*.pt")))
model.load_state_dict(torch.load(ck[-1], map_location=dev)); model.eval()

def forward_patch(patch_layer=None, delta=None, cols=(36, 37)):
    x = model.wte(seq) + model.wpe(torch.arange(96, device=dev))[None]
    for li, blk in enumerate(model.blocks):
        if patch_layer is not None and li == patch_layer:
            for c in cols:
                x = x.clone()
                x[:, c] += delta
        x = blk(x)
    return torch.log_softmax(model.head(model.lnf(x))[:, :, :P.V].double(), -1), x

def kl(p, logq):
    return float((p.clamp_min(1e-9) * (p.clamp_min(1e-9).log() - logq.cpu())).sum(-1).mean())

lp0, _ = forward_patch()
print(f"clean:  KL(BPclean||net)@Q {kl(clean['pred'], lp0[:, Q]):.4f}   "
      f"KL(BPcf||net) {kl(cf['pred'], lp0[:, Q]):.4f}   "
      f"(cf-vs-clean oracle gap {kl(cf['pred'], torch.log(clean['pred'].clamp_min(1e-9))):.4f})")

# encoder (class-mean) maps for pair(4,5) at each layer, cols 36-37
with torch.no_grad():
    _, resids = model(seq, return_resid=True)
for ell in (0, 1, 2, 3, 4):
    H = resids[ell][:, [36, 37]].mean(1)                    # constituent span mean
    mu = torch.stack([H[pair45 == v].mean(0) for v in range(4)])
    delta = (mu[donor] - mu[pair45]).to(dev)
    lp1, _ = forward_patch(patch_layer=ell, delta=delta)
    k_cf = kl(cf["pred"], lp1[:, Q])
    k_cl = kl(clean["pred"], lp1[:, Q])
    # random-direction control at matched norm
    gr = torch.Generator(device=dev).manual_seed(11)
    rd = torch.randn(B, 128, generator=gr, device=dev)
    rd = rd / rd.norm(dim=1, keepdim=True) * delta.norm(dim=1, keepdim=True)
    lpr, _ = forward_patch(patch_layer=ell, delta=rd)
    print(f"patch@L{ell}: KL(BPcf||net_patched) {k_cf:.4f}  KL(BPclean||net_patched) {k_cl:.4f}"
          f"   [rand ctrl: KL(BPcf||·) {kl(cf['pred'], lpr[:, Q]):.4f}]")
