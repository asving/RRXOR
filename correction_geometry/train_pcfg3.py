"""PCFG-3 (full-support RHM, eps=0.05) training: dense vs hard-only (class b >= 2)."""
import argparse, json, os
import torch
import torch.nn.functional as F
from train_multi import GPTV
from pcfg2 import CLS, NLEAF, NTREE, V
from pcfg3 import soft_tables, gen_batch
EPS = 0.05
P3 = soft_tables(EPS)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask", required=True, choices=["dense", "hard"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=60000)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev, L, nl, Bt = "cuda", NLEAF * NTREE, 6, 256
    name = f"pcfg3_{args.mask}_seed{args.seed}"
    base = os.path.dirname(os.path.abspath(__file__))
    run = os.path.join(base, "runs", name)
    os.makedirs(os.path.join(run, "ckpts"), exist_ok=True)
    g_ev = torch.Generator().manual_seed(4242)
    eseq = gen_batch(4096, NLEAF * NTREE, P3, g_ev)
    torch.save({"eseq": eseq}, os.path.join(run, "evalset.pt"))
    eseq_d = eseq.to(dev)
    cls_seq = torch.tensor([CLS[i] for _ in range(NTREE) for i in range(NLEAF)])
    tgt_cls = cls_seq[1:]
    G = {f"b{b}": (tgt_cls == b) for b in (0, 1, 2, 3)}
    G["first"] = (tgt_cls == -1)
    hard = (tgt_cls >= 2)
    json.dump({"mask": args.mask, "seed": args.seed, "L": L, "nl": nl, "vocab": V,
               "steps": args.steps, "batch": Bt, "lr": 3e-4, "gseed": 7, "eps": EPS},
              open(os.path.join(run, "config.json"), "w"))
    model = GPTV(L=L, nl=nl, vocab=V).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.0)
    g = torch.Generator().manual_seed(args.seed + 777)
    logf = open(os.path.join(run, "eval.jsonl"), "w")
    hard_d = hard.to(dev)

    def evaluate(step):
        model.eval()
        with torch.no_grad():
            lp = F.log_softmax(model(eseq_d)[:, :-1].double(), -1)
            ce = -lp.gather(-1, eseq_d[:, 1:, None]).squeeze(-1)
            rec = {"step": step, "ce_avg": float(ce.mean())}
            for k, m in G.items():
                rec[f"ce_{k}"] = float(ce[:, m].mean())
            logf.write(json.dumps(rec) + "\n"); logf.flush()
        model.train()

    for step in range(args.steps + 1):
        seq = gen_batch(Bt, NLEAF * NTREE, P3, g).to(dev)
        lg = model(seq[:, :-1])
        lp = F.log_softmax(lg, -1)
        ce = -lp.gather(-1, seq[:, 1:, None]).squeeze(-1)
        loss = ce[:, hard_d].mean() if args.mask == "hard" else ce.mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0:
            evaluate(step)
        if (step <= 3000 and step % 100 == 0) or step % 250 == 0:
            torch.save(model.state_dict(),
                       os.path.join(run, "ckpts", f"ckpt{step:06d}.pt"))
    print(f"done {name}")

if __name__ == "__main__":
    main()
