"""P1/P2 grading: evalset-exact BP floors vs final net CEs, dense and hard."""
import json
import torch
from pcfg2 import CLS, NLEAF, NTREE
from pcfg3 import soft_tables, bp_predict
P3 = soft_tables(0.05)
ev = torch.load("runs/pcfg3_dense_seed0/evalset.pt")["eseq"]
floors = {}
for i in range(1, 16):
    b = (i & -i).bit_length() - 1
    ces = []
    for t in range(NTREE):
        lv = ev[:, 16 * t:16 * (t + 1)]
        pred = bp_predict(lv, i, P3)
        ces.append(-pred.clamp_min(1e-12).log().gather(1, lv[:, i:i+1]))
    floors.setdefault(b, []).append(float(torch.cat(ces).mean()))
print("evalset-exact floors: " + "  ".join(f"b{b}={sum(v)/len(v):.4f}" for b, v in sorted(floors.items())))
for r in ("dense", "hard"):
    x = [json.loads(l) for l in open(f"runs/pcfg3_{r}_seed0/eval.jsonl")][-1]
    print(f"{r} @60k: " + "  ".join(f"b{b}={x[f'ce_b{b}']:.4f}" for b in range(4)))
