"""PREREG6 analysis: parallel factor installation (factored) + smooth mess3.

Per checkpoint: fit the vocab-general machine, snap unit-circle eigenvalues to the
6th-root lattice (factored) / track |lam2| (mess3); birth times vs per-factor loss
eras; sector-ablation grid at the final checkpoint vs the preregistered policy grid.
"""
import glob, json, math, os
import torch
import numpy as np
from train_multi import GPTV
import speclib_v as SV

base = os.path.dirname(os.path.abspath(__file__))
dev = "cuda"
Q0, B, HOR = 60, 1024, 30

def load_run(name):
    run = os.path.join(base, "runs", name)
    cfg = json.load(open(os.path.join(run, "config.json")))
    ev = torch.load(os.path.join(run, "evalset.pt"))
    seq = ev["eseq"][:B].to(dev)
    ck = sorted(glob.glob(os.path.join(run, "ckpts", "*.pt")))
    steps = [int(os.path.basename(c)[4:10]) for c in ck]
    model = GPTV(L=cfg["L"], vocab=cfg["vocab"]).to(dev)
    evl = [json.loads(l) for l in open(os.path.join(run, "eval.jsonl"))]
    return run, cfg, ev, seq, ck, steps, model, evl

# ================= factored =================
run, cfg, ev, seq, ck, steps, model, evl = load_run("factored_seed0")
V = 4
MX = 4                                 # tests <= 4, paths = 4^5 = 1024 forwards
TESTS = SV.make_tests_v(MX, V)
LAT6 = 2 * np.pi / 6

def cohorts(e):
    """Snapped max-|lam| per 6th-root class: {0:1, 1&5:prim6, 2&4:cube, 3:-1}."""
    out = {"one": 0.0, "cube": 0.0, "minus": 0.0, "prim6": 0.0}
    for v in e:
        if abs(v) < 0.5:
            continue
        kk = int(round(np.angle(v) / LAT6)) % 6
        if abs(np.angle(v) - round(np.angle(v) / LAT6) * LAT6) > 0.3 * LAT6:
            continue
        key = {0: "one", 1: "prim6", 5: "prim6", 2: "cube", 4: "cube",
               3: "minus"}[kk]
        out[key] = max(out[key], abs(v))
    return out

print("=== factored: eigenvalue cohorts across training ===")
traj = []
TSTEPS = sorted(set(s for s in steps if s % 50 == 0 and s <= 1500 or s % 250 == 0))
for st in TSTEPS:
    i = steps.index(st) if st in steps else None
    if i is None:
        continue
    model.load_state_dict(torch.load(ck[i], map_location=dev)); model.eval()
    tab = SV.net_table_v(model, seq, Q0, MX, V, dev=dev)
    oom = SV.fit_oom_v(tab, TESTS, MX, V, k=15, colnorm=False)
    e = SV.eig_transfer_v(oom)
    co = cohorts(e)
    traj.append({"step": st, "fitr2": round(oom["fitr2"], 4),
                 **{k: round(v, 3) for k, v in co.items()},
                 "eigs": [[float(v.real), float(v.imag)] for v in e]})
    print(f"  step {st:5d}  fit {oom['fitr2']:.3f}  |1| {co['one']:.2f} "
          f"cube {co['cube']:.2f}  -1 {co['minus']:.2f}  prim6 {co['prim6']:.2f}")

# final: sector-ablation grid
model.load_state_dict(torch.load(ck[-1], map_location=dev)); model.eval()
tab = SV.net_table_v(model, seq, Q0, MX, V, dev=dev)
oom = SV.fit_oom_v(tab, TESTS, MX, V, k=15, colnorm=False)
e = SV.eig_transfer_v(oom)
print("final eigen:", " ".join(f"{v.real:.2f}" if abs(v.imag) < 0.02 else
      f"{abs(v):.2f}∠{abs(np.angle(v)):.2f}" for v in e))
def near(v, ang):
    return abs(v) > 0.75 and abs(abs(np.angle(v)) - ang) < 0.3
GRID = [("{1}+transients        (both-bets 1.199?)", lambda v: False),
        ("+cube (A clock)       (A-gated+B-bet 1.024?)", lambda v: near(v, 2 * np.pi / 3)),
        ("+{-1} (B clock)       (A-bet+B-gated 0.983?)", lambda v: near(v, np.pi)),
        ("full                  (floor 0.809?)", lambda v: abs(v) > 0.5)]
print("\n=== factored: sector-ablation grid ===")
grid_res = {}
for nm, fn in GRID:
    ce, nm_modes = SV.sector_rollout_v(oom, fn, seq, Q0, HOR)
    grid_res[nm] = ce
    print(f"  {nm:44s} modes {nm_modes:2d}  CE {ce:.4f}")

json.dump({"traj": traj, "grid": grid_res},
          open(os.path.join(run, "prereg6_factored.json"), "w"))

import sys
if "--factored-only" in sys.argv:
    sys.exit(0)

# ================= mess3 =================
run3, cfg3, ev3, seq3, ck3, steps3, model3, evl3 = load_run("mess3_seed0")
V3, MX3 = 3, 4
TESTS3 = SV.make_tests_v(MX3, V3)
print("\n=== mess3: |lam2| across training (theory {1, 0.4, 0.4}) ===")
traj3 = []
for st in [s for s in steps3 if s % 100 == 0 and s <= 1500 or s % 500 == 0]:
    if st not in steps3:
        continue
    model3.load_state_dict(torch.load(ck3[steps3.index(st)], map_location=dev))
    model3.eval()
    tab3 = SV.net_table_v(model3, seq3, Q0, MX3, V3, dev=dev)
    oom3 = SV.fit_oom_v(tab3, TESTS3, MX3, V3, k=3)
    e3 = SV.eig_transfer_v(oom3)
    traj3.append({"step": st, "fitr2": round(oom3["fitr2"], 4),
                  "eigs": [[float(v.real), float(v.imag)] for v in e3]})
    print(f"  step {st:5d}  fit {oom3['fitr2']:.3f}  eigen " +
          " ".join(f"{v.real:.3f}" if abs(v.imag) < 0.02 else
                   f"{abs(v):.3f}∠{abs(np.angle(v)):.2f}" for v in e3))
model3.load_state_dict(torch.load(ck3[-1], map_location=dev)); model3.eval()
tab3 = SV.net_table_v(model3, seq3, Q0, MX3, V3, dev=dev)
oom3 = SV.fit_oom_v(tab3, TESTS3, MX3, V3, k=3)
print("mess3 sector ablation:")
for nm, fn in [("{1} only", lambda v: abs(v) > 0.9), ("full", lambda v: abs(v) > 0.1)]:
    ce, nmm = SV.sector_rollout_v(oom3, fn, seq3, Q0, HOR, trans=0.05)
    print(f"  {nm:16s} modes {nmm}  CE {ce:.4f}")
json.dump({"traj": traj3}, open(os.path.join(run3, "prereg6_mess3.json"), "w"))
print("saved prereg6 jsons")
