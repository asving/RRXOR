"""P-B3: probe exact ancestor posteriors (levels 1..5) from each network layer,
at early checkpoints. Targets from the validated BP oracle (path beliefs)."""
import glob, json, os
import torch
import numpy as np
from train_multi import GPTV
import pcfg

base = os.path.dirname(os.path.abspath(__file__))
dev = "cuda"
run = "pcfg_dense_seed0"
ev = torch.load(os.path.join(base, "runs", run, "evalset.pt"))
seq = ev["eseq"][:1024].to(dev)
B, L = seq.shape

def path_beliefs(leaves, upto, eta=pcfg.ETA):
    """Posteriors P(ancestor_j = 1 | leaves[:, :upto]) for j = 5..1 levels above
    the target leaf (path from root to target), exact BP."""
    Bs = leaves.shape[0]
    l = pcfg.DEPTH
    lo = 0
    seen = list(range(0, upto))
    def edge_pass(msg):
        m0, m1 = msg[..., 0], msg[..., 1]
        return torch.stack([(1 - eta) * m0 + eta * m1,
                            eta * m0 + (1 - eta) * m1], -1)
    nlv = {l: torch.ones(Bs, 2 ** l, 2, dtype=torch.double)}
    for j in seen:
        nlv[l][torch.arange(Bs), j, 0] = (leaves[:, j] == 0).double()
        nlv[l][torch.arange(Bs), j, 1] = (leaves[:, j] == 1).double()
    for lev in range(l, 0, -1):
        up = edge_pass(nlv[lev])
        nlv[lev - 1] = up[:, 0::2] * up[:, 1::2]
    tgt = upto
    belief = nlv[0][:, 0] / nlv[0][:, 0].sum(-1, keepdim=True).clamp_min(1e-30)
    out = [belief[:, 1]]                                   # root = 5 levels above
    for lev in range(1, l):
        child = (tgt >> (l - lev))
        below_c = nlv[lev][:, child]
        up_c = edge_pass(below_c)
        excl = (belief / up_c.clamp_min(1e-30))
        excl = excl / excl.sum(-1, keepdim=True).clamp_min(1e-30)
        down = edge_pass(excl)
        m = down * below_c
        belief = m / m.sum(-1, keepdim=True).clamp_min(1e-30)
        out.append(belief[:, 1])
    return out                                             # levels 5,4,3,2,1 above

# targets: for each tree t and leaf i>=2, ancestor posteriors given the prefix
leavesT = {t: seq[:, 32 * t:32 * (t + 1)].cpu() for t in range(3)}
POS, ANC = [], {j: [] for j in (1, 2, 3, 4, 5)}
for t in range(3):
    for i in range(2, 32):
        pb = path_beliefs(leavesT[t], i)
        p = 32 * t + i - 1                                 # query column predicts i
        POS.append(p)
        for j in (1, 2, 3, 4, 5):
            ANC[j].append(pb[5 - j])
Y = {j: torch.stack(ANC[j], 1) for j in ANC}               # B x P
POSt = torch.tensor(POS)

def r2(X, y):
    n = X.shape[0]; ntr = int(0.7 * n)
    mu, sd = X[:ntr].mean(0), X[:ntr].std(0) + 1e-6
    Xb = torch.cat([(X - mu) / sd, torch.ones(n, 1)], 1)
    A = Xb[:ntr].T @ Xb[:ntr] + 10 * torch.eye(Xb.shape[1])
    W = torch.linalg.solve(A, Xb[:ntr].T @ y[:ntr])
    p = Xb[ntr:] @ W
    return float(1 - ((y[ntr:] - p) ** 2).sum() / ((y[ntr:] - y[:ntr].mean()) ** 2).sum())

# window baseline (8 tokens)
wf = torch.stack([seq.cpu().roll(k, dims=1).double() for k in range(1, 9)], -1)
model = GPTV(L=L, nl=6, vocab=2).to(dev)
ck = sorted(glob.glob(os.path.join(base, "runs", run, "ckpts", "*.pt")))
steps = [int(os.path.basename(c)[4:10]) for c in ck]
res = {}
for st in (100, 200, 300, 600, 3000):
    model.load_state_dict(torch.load(ck[steps.index(st)], map_location=dev))
    model.eval()
    with torch.no_grad():
        _, resids = model(seq, return_resid=True)
    row = {}
    for j in (1, 2, 3, 4, 5):
        yj = Y[j].reshape(-1)
        Xw = wf[:, POSt].reshape(-1, 8)
        bw = r2(Xw, yj)
        per_layer = []
        for li in range(7):
            X = resids[li][:, POSt].reshape(-1, 128).cpu().double()
            per_layer.append(round(r2(X, yj) - bw, 3))
        row[j] = per_layer
    res[st] = row
    print(f"step {st:5d}: " + " | ".join(
        f"lvl{j}: " + ",".join(f"{v:+.2f}" for v in row[j]) for j in (1, 3, 5)))
json.dump(res, open(os.path.join(base, "pcfg_layergrid.json"), "w"))
print("cols per level = layers emb..L6 (excess over 8-token window)")
