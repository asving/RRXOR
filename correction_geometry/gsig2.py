"""Init gradient-signal G(f|theta0) per route (drift^2 and noise) + tau from runs."""
import glob, json
import torch
import torch.nn.functional as F
from train_multi import GPTV
from train_route2 import gen

dev = "cuda"
print("route | tau95 (s0/s1) | Gdisc=|E grad margin|^2 | noise | SNR")
for arm, key, m in (("ll1", "acc1", (4, 0)), ("lx", "acc2", (0, 4)), ("p3", "acc2", (0, 4))):
    taus = []
    for s in (0, 1):
        rows = [json.loads(l) for l in open(f"runs/r2_{arm}_m{m[0]}x{m[1]}_s{s}_nl1_ns/eval.jsonl")]
        taus.append(next((r["step"] for r in rows if r[key] >= 0.95), None))
    torch.manual_seed(0)
    model = GPTV(L=44, nl=1, vocab=16).to(dev)
    g = torch.Generator().manual_seed(9)
    grads, sqn = None, 0.0
    NB = 40
    for _ in range(NB):
        seq, meta = gen(64, m[0], m[1], arm, g, noshared=True)
        seq = seq.to(dev)
        lg = model(seq[:, :-1])
        akey = "aux1" if arm == "ll1" else "aux2"
        margins = []
        for b in range(64):
            for (p, a) in meta[akey][b]:
                wrong = 6 + 7 - a
                margins.append(lg[b, p - 1, a] - lg[b, p - 1, wrong])
        loss = torch.stack(margins).mean()
        model.zero_grad(); loss.backward()
        flat = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
        grads = flat.clone() if grads is None else grads + flat
        sqn += float(flat.pow(2).sum())
    G = float((grads / NB).pow(2).sum())
    noise = sqn / NB
    print(f"{arm:4s} | {taus[0]}/{taus[1]} | {G:.3e} | {noise:.3e} | {G/noise:.4f}")
