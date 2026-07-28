"""Per-interface demand/supply trajectories on m27 (PREREG14 modular pilot).
At each ckpt: delta_l = grad of det-position CE w.r.t. layer-l residuals (demand
arriving at the interface); y_k = ground-truth part variables (u=xor, phase9,
flip). demand(k,l) = normalized ||E[delta x y]||^2; supply(k,l) = probe R2 of
y from resid. Prediction: demand-for-phase9 rises at the xnor drop."""
import glob, json
import torch
import torch.nn.functional as F
from train import GPT
from train27 import gen_batch27, NESTED

dev = "cuda"
g = torch.Generator().manual_seed(7)
seq, cls = gen_batch27(768, 192, g)
seq, cls = seq.to(dev), cls
tgt_cls = cls[:, 1:]
det_q = ((tgt_cls % 3) == 2) & (torch.arange(191)[None, :] >= 100)   # late det queries
bq, pq = det_q.nonzero(as_tuple=True)
blk = ((tgt_cls[bq, pq] // 3) % 9)
Y = {"u": (seq.cpu()[bq, pq] ^ seq.cpu()[bq, pq - 1]).double()[:, None] * 2 - 1,
     "phase9": F.one_hot(blk, 9).double(),
     "flip": (torch.tensor(NESTED)[blk] == 1).double()[:, None] * 2 - 1}
model = GPT(L=192, nl=6).to(dev)
cks = sorted(glob.glob("runs/m27_nl6_seed1/ckpts/*.pt"))
steps = [int(c.split("step")[-1].split(".")[0]) for c in cks]
use = [i for i, s in enumerate(steps) if s in (0, 100, 200, 300, 400, 500, 600, 800, 1000, 1200, 1600, 2000, 3000, 4000, 6000, 8000, 12000, 16000)]
ntr = len(bq) // 2

def ridge_r2(X, Yv):
    mu, sd = X[:ntr].mean(0), X[:ntr].std(0) + 1e-6
    Xb = torch.cat([(X - mu) / sd, torch.ones(len(X), 1)], 1)
    W = torch.linalg.solve(Xb[:ntr].T @ Xb[:ntr] + 10 * torch.eye(Xb.shape[1]), Xb[:ntr].T @ Yv[:ntr])
    P = Xb[ntr:] @ Yv.new_tensor(W)
    return float(1 - ((Yv[ntr:] - P) ** 2).sum() / ((Yv[ntr:] - Yv[:ntr].mean(0)) ** 2).sum())

out = []
for i in use:
    model.load_state_dict(torch.load(cks[i], map_location=dev)); model.eval()
    x = model.wte(seq[:, :-1]) + model.wpe(torch.arange(191, device=dev))[None]
    resids = []
    for b3 in model.blocks:
        x = b3(x)
        x.retain_grad(); resids.append(x)
    lg = model.head(model.lnf(x))
    ce = F.cross_entropy(lg[bq, pq], seq[:, 1:][bq, pq])
    ce.backward()
    rec = {"step": steps[i], "ce_det": float(ce)}
    for li in (1, 3, 5):
        dlt = resids[li].grad[bq, pq].cpu().double()
        dn = dlt / (dlt.norm(dim=1, keepdim=True) + 1e-9)
        h = resids[li][bq, pq].detach().cpu().double()
        for k, y in Y.items():
            yn = (y - y.mean(0)) / (y.std(0) + 1e-9)
            C = dn.T @ yn / len(bq)
            rec[f"dem_{k}_L{li+1}"] = float((C ** 2).sum())
            if li == 3:
                rec[f"sup_{k}"] = ridge_r2(h, y)
    out.append(rec)
    model.zero_grad()
json.dump(out, open("m27_parts.json", "w"))
for r in out:
    print(f"s{r['step']:6d} ce{r['ce_det']:.3f} | dem_u_L4 {r['dem_u_L4']:.4f} dem_p9_L4 {r['dem_phase9_L4']:.4f} "
          f"dem_flip_L4 {r['dem_flip_L4']:.4f} | sup_u {r['sup_u']:.2f} sup_p9 {r['sup_phase9']:.2f}")
