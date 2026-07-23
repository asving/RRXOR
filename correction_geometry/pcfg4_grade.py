"""P1/P2: evalset-exact floors vs finals; onset timing from curves."""
import json
import torch
from pcfg2 import NLEAF, NTREE
from pcfg4 import ALPHA, dirichlet_tables, bp4
P = dirichlet_tables(ALPHA)
ev = torch.load("runs/pcfg4_dense_seed0/evalset.pt")["eseq"]
fl = {}
for i in range(1, 16):
    b = (i & -i).bit_length() - 1
    ces = []
    for t in range(NTREE):
        lv = ev[:, 16 * t:16 * (t + 1)]
        ces.append(-bp4(lv, i, P).clamp_min(1e-12).log().gather(1, lv[:, i:i+1]))
    fl.setdefault(b, []).append(float(torch.cat(ces).mean()))
print("evalset floors: " + "  ".join(f"b{b}={sum(v)/len(v):.4f}" for b, v in sorted(fl.items())))
for r in ("dense", "hard"):
    rows = [json.loads(l) for l in open(f"runs/pcfg4_{r}_seed0/eval.jsonl")]
    for s in (200, 500, 1000, 2000, 5000, 15000, 60000):
        x = [q for q in rows if q["step"] == s][0]
        print(f"{r} {s:6d}: " + "  ".join(f"b{b}={x[f'ce_b{b}']:.4f}" for b in range(4)))
