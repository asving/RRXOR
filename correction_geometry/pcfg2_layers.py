"""PCFG-2 P3/P4: endpoint conditional match vs full BP + ancestor-posterior probe
grid (levels x network layers x checkpoints)."""
import glob, json, os
import torch
import numpy as np
from train_multi import GPTV
import pcfg2 as P

base = os.path.dirname(os.path.abspath(__file__))
dev = "cuda"
run = "pcfg2_dense_seed0"
ev = torch.load(os.path.join(base, "runs", run, "evalset.pt"))
seq = ev["eseq"][:1024].to(dev)
B, L = seq.shape

# oracle targets: predictive dist + ancestor posteriors per (tree, leaf>=2)
POS, PRED, ANC = [], [], {j: [] for j in (1, 2, 3, 4)}
for t in range(P.NTREE):
    lv = seq[:, 16 * t:16 * (t + 1)].cpu()
    for i in range(2, 16):
        pred, path = P.bp_predict(lv, i, want_path=True)
        POS.append(16 * t + i - 1)
        PRED.append(pred)
        ANC[1].append(path[2]); ANC[2].append(path[1]); ANC[3].append(path[0])
        # root posterior: uniform prior x lambda at root
        lam_root_needed = True
        ANC[4].append(None)
    # recompute root posteriors via full-subtree lambda
for t in range(P.NTREE):
    lv = seq[:, 16 * t:16 * (t + 1)].cpu()
    for idx, i in enumerate(range(2, 16)):
        lam = {4: torch.ones(1024, 16, P.V, dtype=torch.double)}
        for j in range(i):
            lam[4][:, j] = 0.0
            lam[4][torch.arange(1024), j, lv[:, j]] = 1.0
        for lev in range(4, 0, -1):
            tab = P.TABLES[lev - 1]
            lamL, lamR = lam[lev][:, 0::2], lam[lev][:, 1::2]
            up = torch.zeros(1024, lamL.shape[1], P.V, dtype=torch.double)
            for a in range(P.V):
                for k in range(P.M):
                    Lc, Rc = int(tab[a, k, 0]), int(tab[a, k, 1])
                    up[:, :, a] += lamL[:, :, Lc] * lamR[:, :, Rc] / P.M
            lam[lev - 1] = up
        r = lam[0][:, 0]
        ANC[4][t * 14 + idx] = r / r.sum(-1, keepdim=True).clamp_min(1e-30)
POSt = torch.tensor(POS)
Yanc = {j: torch.stack(ANC[j], 1) for j in ANC}            # B x P x V
Ypred = torch.stack(PRED, 1)

model = GPTV(L=L, nl=6, vocab=P.V).to(dev)
ck = sorted(glob.glob(os.path.join(base, "runs", run, "ckpts", "*.pt")))
steps = [int(os.path.basename(c)[4:10]) for c in ck]

# P4 endpoint: net conditionals vs full BP
model.load_state_dict(torch.load(ck[-1], map_location=dev)); model.eval()
with torch.no_grad():
    lg = model(seq)
    lpn = torch.log_softmax(lg[:, :, :P.V].double(), -1)
pn = lpn[:, POSt].exp().cpu()
kl = (Ypred.clamp_min(1e-9) * (Ypred.clamp_min(1e-9).log() - pn.clamp_min(1e-9).log())).sum(-1)
print(f"P4 endpoint: KL(BP || net) mean {float(kl.mean()):.5f} nats over {len(POS)} positions")

def r2(X, Y):
    n = X.shape[0]; ntr = int(0.7 * n)
    mu, sd = X[:ntr].mean(0), X[:ntr].std(0) + 1e-6
    Xb = torch.cat([(X - mu) / sd, torch.ones(n, 1)], 1)
    A = Xb[:ntr].T @ Xb[:ntr] + 10 * torch.eye(Xb.shape[1])
    W = torch.linalg.solve(A, Xb[:ntr].T @ Y[:ntr])
    Pd = Xb[ntr:] @ W
    return float(1 - ((Y[ntr:] - Pd) ** 2).sum() / ((Y[ntr:] - Y[:ntr].mean(0)) ** 2).sum())

wf = torch.stack([seq.cpu().double().roll(k, dims=1) for k in range(1, 9)], -1)
res = {}
for st in (100, 200, 300, 600, 1000, 1500, 2500, 4000):
    model.load_state_dict(torch.load(ck[steps.index(st)], map_location=dev))
    model.eval()
    with torch.no_grad():
        _, resids = model(seq, return_resid=True)
    row = {}
    for j in (1, 2, 3, 4):
        yj = Yanc[j][:, :, :3].reshape(-1, 3)              # 3 free dims
        Xw = wf[:, POSt].reshape(-1, 8)
        bw = r2(Xw, yj)
        row[j] = [round(r2(resids[li][:, POSt].reshape(-1, 128).cpu().double(), yj) - bw, 3)
                  for li in range(7)]
    res[st] = row
    print(f"step {st:5d}: " + " | ".join(
        f"anc{j}: " + ",".join(f"{v:+.2f}" for v in row[j]) for j in (1, 2, 4)))
json.dump(res, open(os.path.join(base, "pcfg2_layergrid.json"), "w"))
print("cols = layers emb..L6, excess over 8-token window")
