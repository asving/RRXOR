"""Developmental tomography v2.

Fixes over v1: (1) trapezoid attribution a = <(g_t + g_{t+1})/2, dtheta> per block;
(2) units from JOINT window read/write maps — stack all resid-reading rows (qkv, fc1,
unembed) and resid-writing cols (per-head proj, fc2) of the window displacement; SVD
gives window read/write directions in the shared residual basis, fixing cross-block
fragmentation; socket-plug wiring = cos(write dirs of earlier window, read dirs of
later); (3) validation by ROLLBACK: rewind a window's top block-set to window-start
weights inside the final model, measure per-cluster CE (+ random-block control).

Usage: python devtomo2.py seed0 rrxor3   |   python devtomo2.py m27_nl6_seed1 m27
"""
import glob, json, math, os, sys
import torch
import torch.nn.functional as F
import numpy as np
from train import GPT

name = sys.argv[1] if len(sys.argv) > 1 else "seed0"
kind = sys.argv[2] if len(sys.argv) > 2 else "rrxor3"
dev = "cuda"
base = os.path.dirname(os.path.abspath(__file__))
run = os.path.join(base, "runs", name)
ev = torch.load(os.path.join(run, "evalset.pt"))
seq = ev["eseq"][:512].to(dev)
B, L = seq.shape
NL = 4 if kind == "rrxor3" else 6
TPOS = torch.arange(33, L - 1, device=dev) if kind == "rrxor3" else \
       torch.arange(108, 191, device=dev)
NH, HD, D = 4, 32, 128
model = GPT(L=L, d=D, nl=NL).to(dev)
ckpts = sorted(glob.glob(os.path.join(run, "ckpts", "*.pt")))
steps = np.array([int(os.path.basename(c)[4:10]) for c in ckpts])

# ---------------- clusters ----------------
if kind == "m27":
    d_ = torch.load(os.path.join(run, "taskdisc_assign.pt"))
    sub, assign = d_["sub"], d_["assign"]
    K = int(assign.max()) + 1
    cnames = []
    truth = d_["truth"]
    for j in range(K):
        names4 = ["rnd", "xor", "xnor", "flip"]
        cnames.append(names4[int(np.bincount(truth[assign == j].numpy()).argmax())]
                      + f"_c{j}")
    inst_b = (sub // len(TPOS)).to(dev)
    inst_p = TPOS[(sub % len(TPOS))].to(dev)
else:
    CE = torch.zeros(len(ckpts), B, len(TPOS))
    for i, cp in enumerate(ckpts):
        model.load_state_dict(torch.load(cp, map_location=dev)); model.eval()
        with torch.no_grad():
            logp = F.log_softmax(model(seq[:, :-1]), -1)
            CE[i] = (-logp.gather(-1, seq[:, 1:, None]).squeeze(-1))[:, TPOS - 1].cpu()
    X = CE.permute(1, 2, 0).reshape(-1, len(ckpts))
    X = F.avg_pool1d(F.pad(X[:, None], (1, 1), mode="replicate"), 3, stride=1).squeeze(1)
    def kmeans(X, k, iters=60, restarts=10, seed=0):
        gg = torch.Generator().manual_seed(seed)
        best = (None, 1e18)
        for r in range(restarts):
            C = X[torch.randperm(X.shape[0], generator=gg)[:k]].clone()
            for _ in range(iters):
                dd = torch.cdist(X, C); a = dd.argmin(1)
                C = torch.stack([X[a == j].mean(0) if (a == j).any() else C[j]
                                 for j in range(k)])
            inert = (X - C[a]).pow(2).sum().item()
            if inert < best[1]:
                best = (a.clone(), inert)
        return best[0]
    assign = kmeans(X, 3)
    CMs = torch.stack([X[assign == j].mean(0) for j in range(3)])
    det_c = int(CMs[:, -1].argmin()); mis_c = int(CMs[:, len(steps) // 3].argmax())
    match_c = [j for j in range(3) if j not in (det_c, mis_c)][0]
    perm = [det_c, match_c, mis_c]
    assign = torch.tensor([perm.index(int(a)) for a in assign])
    K = 3; cnames = ["det", "match", "mis"]
    ab = torch.arange(B, device=dev)[:, None].expand(B, len(TPOS)).reshape(-1)
    ap = TPOS[None, :].expand(B, len(TPOS)).reshape(-1)
    inst_b, inst_p = ab, ap

# cluster mean curves (for windows) — recompute cheaply for both kinds
CM = np.zeros((K, len(ckpts)))
for i, cp in enumerate(ckpts):
    model.load_state_dict(torch.load(cp, map_location=dev)); model.eval()
    with torch.no_grad():
        logp = F.log_softmax(model(seq[:, :-1]), -1)
        ce = -logp.gather(-1, seq[:, 1:, None]).squeeze(-1)
        cev = ce[inst_b, inst_p - 1].cpu()
    for j in range(K):
        CM[j, i] = float(cev[assign == j].mean())

def find_windows():
    cuts = set()
    for j in range(K):
        dl = np.abs(np.diff(CM[j])) / np.maximum(np.diff(steps), 1) * 1000
        act = dl > 0.03
        for i in range(1, len(act)):
            if act[i] != act[i - 1]:
                cuts.add(int(steps[i]))
    merged = []
    for c in sorted(cuts):
        if not merged or c - merged[-1] > 400:
            merged.append(c)
    bounds = [0] + merged + [int(steps[-1])]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)
            if bounds[i + 1] - bounds[i] > 200]

WINS = find_windows()
print("clusters:", cnames)
print("windows:", WINS)

# ---------------- blocks ----------------
def block_slices():
    blocks = {"emb": [("wte.weight", None, None), ("wpe.weight", None, None)]}
    for l in range(NL):
        p = f"blocks.{l}"
        for h in range(NH):
            ent = []
            for third in range(3):
                r0 = third * D + h * HD
                ent.append((f"{p}.qkv.weight", slice(r0, r0 + HD), None))
                ent.append((f"{p}.qkv.bias", slice(r0, r0 + HD), None))
            ent.append((f"{p}.proj.weight", None, slice(h * HD, (h + 1) * HD)))
            blocks[f"L{l+1}h{h}"] = ent
        blocks[f"L{l+1}mlp"] = [(f"{p}.mlp.0.weight", None, None),
                                (f"{p}.mlp.0.bias", None, None),
                                (f"{p}.mlp.2.weight", None, None),
                                (f"{p}.mlp.2.bias", None, None),
                                (f"{p}.ln1.weight", None, None), (f"{p}.ln1.bias", None, None),
                                (f"{p}.ln2.weight", None, None), (f"{p}.ln2.bias", None, None),
                                (f"{p}.proj.bias", None, None)]
    blocks["unemb"] = [("head.weight", None, None), ("lnf.weight", None, None),
                       ("lnf.bias", None, None)]
    return blocks

BL = block_slices(); bnames = list(BL.keys())
def take(td, spec):
    out = []
    for pn, rs, cs in spec:
        t = td[pn]
        if rs is not None: t = t[rs]
        if cs is not None: t = t[:, cs]
        out.append(t.flatten())
    return torch.cat(out)

def cluster_grads():
    out = []
    for j in range(K):
        m = (assign == j).to(dev)
        logp = F.log_softmax(model(seq[:, :-1]), -1)
        ce = -logp.gather(-1, seq[:, 1:, None]).squeeze(-1)
        loss = ce[inst_b[m], inst_p[m] - 1].mean()
        gs = torch.autograd.grad(loss, list(model.parameters()))
        out.append({n: g for (n, _), g in zip(model.named_parameters(), gs)})
    return out

# ---------------- trapezoid attribution ----------------
T = len(ckpts) - 1
A = np.zeros((T, K, len(bnames)))
sd_prev = torch.load(ckpts[0], map_location=dev)
model.load_state_dict(sd_prev); model.eval()
G_prev = cluster_grads()
for i in range(T):
    sd_next = torch.load(ckpts[i + 1], map_location=dev)
    model.load_state_dict(sd_next); model.eval()
    G_next = cluster_grads()
    dth = {n: sd_next[n] - sd_prev[n] for n in sd_prev}
    for bi, bn in enumerate(bnames):
        dv = take(dth, BL[bn])
        for a in range(K):
            gv = 0.5 * (take(G_prev[a], BL[bn]) + take(G_next[a], BL[bn]))
            A[i, a, bi] = float(gv @ dv)
    sd_prev, G_prev = sd_next, G_next
    if i % 20 == 0:
        print(f"interval {i}/{T}", flush=True)

print("\ncompleteness (trapezoid):")
for a in range(K):
    print(f"  {cnames[a]:10s} attributed {A[:, a, :].sum():+.4f}   "
          f"actual {CM[a, -1] - CM[a, 0]:+.4f}")

np.savez(os.path.join(run, "devtomo2.npz"), A=A, steps=steps,
         bnames=np.array(bnames), wins=np.array(WINS), CM=CM,
         cnames=np.array(cnames))
print("\nper-window top blocks (− helps, + hurts):")
for (w0, w1) in WINS:
    m = (steps[:-1] >= w0) & (steps[:-1] < w1)
    Aw = A[m].sum(0)
    print(f"  [{w0},{w1}):")
    for a in range(K):
        idx = np.argsort(Aw[a])
        helps = [(bnames[k], round(float(Aw[a][k]), 3)) for k in idx[:3] if Aw[a][k] < -0.01]
        hurts = [(bnames[k], round(float(Aw[a][k]), 3)) for k in idx[::-1][:3] if Aw[a][k] > 0.01]
        print(f"    {cnames[a]:10s} helps {helps}  hurts {hurts}")

# ---------------- joint read/write maps per window ----------------
def window_maps(w0, w1):
    i0 = int(np.searchsorted(steps, w0)); i1 = min(int(np.searchsorted(steps, w1)),
                                                   len(ckpts) - 1)
    s0 = torch.load(ckpts[i0], map_location=dev); s1 = torch.load(ckpts[i1], map_location=dev)
    d = {n: (s1[n] - s0[n]).float() for n in s0}
    reads, writes = [], []
    rlab, wlab = [], []
    for l in range(NL):
        p = f"blocks.{l}"
        q = d[f"{p}.qkv.weight"]
        for h in range(NH):
            for third, tn in enumerate("qkv"):
                reads.append(q[third * D + h * HD: third * D + (h + 1) * HD])
                rlab += [f"L{l+1}h{h}.{tn}"] * HD
            writes.append(d[f"{p}.proj.weight"][:, h * HD:(h + 1) * HD])
            wlab += [f"L{l+1}h{h}"] * HD
        reads.append(d[f"{p}.mlp.0.weight"]); rlab += [f"L{l+1}mlp"] * (4 * D)
        writes.append(d[f"{p}.mlp.2.weight"]); wlab += [f"L{l+1}mlp"] * (4 * D)
    reads.append(d["head.weight"]); rlab += ["unemb"] * 2
    Rm = torch.cat(reads, 0)                          # (N, d): rows read resid
    Wm = torch.cat(writes, 1)                         # (d, M): cols write resid
    return Rm, Wm, rlab, wlab

# semantic directions for annotation
model.load_state_dict(torch.load(ckpts[-1], map_location=dev)); model.eval()
with torch.no_grad():
    _, resid = model(seq, return_resid=True)
u_lab = torch.zeros_like(seq); u_lab[:, 1:] = seq[:, 1:] ^ seq[:, :-1]
m_lab = torch.zeros_like(seq); m_lab[:, 2:] = seq[:, 2:] ^ seq[:, 1:-1] ^ seq[:, :-2]
valid = torch.zeros(B, L, dtype=torch.bool, device=dev); valid[:, 2:] = True
def cmdir(lab, li):
    Dl = (resid[li] - resid[li - 1])[valid].float(); y = lab[valid]
    v = Dl[y == 1].mean(0) - Dl[y == 0].mean(0)
    return v / v.norm()
SEM = {}
for li in range(2, NL + 1):
    SEM[f"u@L{li}"] = cmdir(u_lab, li)
    SEM[f"m@L{li}"] = cmdir(m_lab, li)

print("\nwindow read/write directions (joint SVD across blocks):")
WDIRS, RDIRS = {}, {}
for (w0, w1) in WINS:
    Rm, Wm, rlab, wlab = window_maps(w0, w1)
    Ur, Sr, Vr = torch.linalg.svd(Rm, full_matrices=False)   # read dirs: rows of Vr
    Uw, Sw, Vw = torch.linalg.svd(Wm, full_matrices=False)   # write dirs: cols of Uw
    WDIRS[(w0, w1)] = Uw[:, :3]; RDIRS[(w0, w1)] = Vr[:3].T
    for r in range(2):
        wv = Uw[:, r]
        ann = max(((k, float((wv @ v).abs())) for k, v in SEM.items()), key=lambda x: x[1])
        share = float(Sw[r] ** 2 / (Sw ** 2).sum())
        # participating writer blocks
        wl = np.array(wlab)
        contrib = (Vw[r].abs().cpu().numpy())
        blocks_c = {}
        for lab, c in zip(wl, contrib):
            blocks_c[lab] = blocks_c.get(lab, 0) + c ** 2
        topb = sorted(blocks_c.items(), key=lambda x: -x[1])[:3]
        print(f"  [{w0},{w1}) write-dir {r}: share {share:.2f}  best-sem {ann[0]}={ann[1]:.2f}"
              f"  writers {[(b, round(v,2)) for b, v in topb]}")
    for r in range(1):
        rv = Vr[r]
        ann = max(((k, float((rv @ v).abs())) for k, v in SEM.items()), key=lambda x: x[1])
        print(f"  [{w0},{w1}) read-dir {r}: best-sem {ann[0]}={ann[1]:.2f}")

WIRE = np.zeros((len(WINS), len(WINS)))
print("\nsocket-plug wiring (cos of earlier write dirs vs later read dirs):")
for i, wa in enumerate(WINS):
    for wb in WINS[i + 1:]:
        C = (WDIRS[wa].T @ RDIRS[wb]).abs().max().item()
        WIRE[i, WINS.index(wb)] = C
        print(f"  {wa} -> {wb}: max |cos| = {C:.2f}")
np.save(os.path.join(run, "devtomo2_wire.npy"), WIRE)

# ---------------- rollback validation ----------------
print("\nrollback knockouts (rewind block-set to window start, in final model):")
final_sd = torch.load(ckpts[-1], map_location=dev)
gg = torch.Generator().manual_seed(0)
def eval_cluster_ce():
    model.eval()
    with torch.no_grad():
        logp = F.log_softmax(model(seq[:, :-1]), -1)
        ce = -logp.gather(-1, seq[:, 1:, None]).squeeze(-1)
        cev = ce[inst_b, inst_p - 1].cpu()
    return [round(float(cev[assign == j].mean()), 3) for j in range(K)]

model.load_state_dict(final_sd)
RB = {"none": eval_cluster_ce()}
print(f"  none: {RB['none']}   ({cnames})")
for (w0, w1) in WINS:
    m = (steps[:-1] >= w0) & (steps[:-1] < w1)
    Aw = np.abs(A[m].sum(0)).sum(0)
    top = [bnames[k] for k in np.argsort(-Aw)[:4]]
    i0 = int(np.searchsorted(steps, w0))
    sd0 = torch.load(ckpts[i0], map_location=dev)
    for label, blocks_sel in [("top4", top),
                              ("rand4", [bnames[k] for k in
                                         torch.randperm(len(bnames), generator=gg)[:4]])]:
        sd = {n: v.clone() for n, v in final_sd.items()}
        for bn in blocks_sel:
            for pn, rs, cs in BL[bn]:
                if rs is None and cs is None:
                    sd[pn] = sd0[pn].clone()
                elif rs is not None:
                    sd[pn][rs] = sd0[pn][rs]
                else:
                    sd[pn][:, cs] = sd0[pn][:, cs]
        model.load_state_dict(sd)
        r = eval_cluster_ce()
        RB[f"{w0}-{w1}:{label}"] = {"blocks": blocks_sel, "ce": r}
        print(f"  rollback [{w0},{w1}) {label} {blocks_sel if label=='top4' else ''}: {r}")
model.load_state_dict(final_sd)
json.dump(RB, open(os.path.join(run, "devtomo2_rollback.json"), "w"))
