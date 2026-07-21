"""PREREG7 training: same-seed pairs with loss at the LAST target position only.

Usage: train_pos.py --task rrxor3|m27|mess3 --seed S --steps N
Matches each baseline's architecture/init/data conventions; loss = CE at the final
target only. Logs ce_last (+ per-class for m27) and ce_avg every 50 steps.
"""
import argparse, json, os
import torch
import torch.nn.functional as F
from train_multi import GPTV, gen_mess3
from train import gen_batch as gen_rrxor3
from train27 import gen_batch27, NESTED

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["rrxor3", "m27", "mess3"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=24000)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = "cuda"
    CFG = {"rrxor3": dict(L=96, nl=4, vocab=2), "m27": dict(L=192, nl=6, vocab=2),
           "mess3": dict(L=96, nl=4, vocab=3)}[args.task]
    L, nl, vocab = CFG["L"], CFG["nl"], CFG["vocab"]
    Bt = 256
    name = f"{args.task}_last_seed{args.seed}"
    base = os.path.dirname(os.path.abspath(__file__))
    run = os.path.join(base, "runs", name)
    os.makedirs(os.path.join(run, "ckpts"), exist_ok=True)
    g_ev = torch.Generator().manual_seed(4242)
    if args.task == "rrxor3":
        eseq, etypes = gen_rrxor3(2048, L, g_ev)
        torch.save({"eseq": eseq, "etypes": etypes}, os.path.join(run, "evalset.pt"))
        cls_last = None
    elif args.task == "m27":
        eseq, ecls = gen_batch27(2048, L, g_ev)
        torch.save({"eseq": eseq, "ecls": ecls}, os.path.join(run, "evalset.pt"))
        cl = ecls[:, L - 1]
        det = cl % 3 == 2
        blk = torch.where(det, ((cl - 2) // 3) % 9, torch.zeros_like(cl))
        rule = torch.tensor(NESTED)[blk]
        groups = {"rnd": ~det, "xor": det & (rule == 0) & (blk != 8),
                  "xnor": det & (rule == 1), "flip": det & (blk == 8)}
        cls_last = groups
    else:
        eseq, est = gen_mess3(2048, L, g_ev)
        torch.save({"eseq": eseq, "estates": est}, os.path.join(run, "evalset.pt"))
        cls_last = None
    eseq_d = eseq.to(dev)
    json.dump({"task": args.task, "seed": args.seed, "L": L, "nl": nl,
               "vocab": vocab, "steps": args.steps, "batch": Bt, "lr": 3e-4,
               "loss": "last"}, open(os.path.join(run, "config.json"), "w"))
    model = GPTV(L=L, nl=nl, vocab=vocab).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.0)
    g = torch.Generator().manual_seed(args.seed + 777)
    logf = open(os.path.join(run, "eval.jsonl"), "w")

    def evaluate(step):
        model.eval()
        with torch.no_grad():
            lg = model(eseq_d)
            lp = F.log_softmax(lg[:, :-1].double(), -1)
            ce = -lp.gather(-1, eseq_d[:, 1:, None]).squeeze(-1)
            rec = {"step": step, "ce_last": float(ce[:, -1].mean()),
                   "ce_avg": float(ce[:, 20:].mean())}
            if cls_last is not None:
                for k, m in cls_last.items():
                    rec[f"ce_last_{k}"] = float(ce[m, -1].mean())
            logf.write(json.dumps(rec) + "\n"); logf.flush()
        model.train()

    for step in range(args.steps + 1):
        if args.task == "rrxor3":
            seq, _ = gen_rrxor3(Bt, L, g)
        elif args.task == "m27":
            seq, _ = gen_batch27(Bt, L, g)
        else:
            seq, _ = gen_mess3(Bt, L, g)
        seq = seq.to(dev)
        lg = model(seq[:, :-1])
        loss = F.cross_entropy(lg[:, -1], seq[:, -1])       # LAST target only
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0:
            evaluate(step)
        if (step <= 2000 and step % 100 == 0) or step % 500 == 0:
            torch.save(model.state_dict(),
                       os.path.join(run, "ckpts", f"ckpt{step:06d}.pt"))
    print(f"done {name}")

if __name__ == "__main__":
    main()
