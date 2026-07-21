"""PREREG8 Part A: m27 with loss masked by token TYPE (det-only or exceptions-only)."""
import argparse, json, os
import torch
import torch.nn.functional as F
from train_multi import GPTV
from train27 import gen_batch27, NESTED

def class_bits(cls):
    det = cls % 3 == 2
    blk = torch.where(det, ((cls - 2) // 3) % 9, torch.zeros_like(cls))
    rule = torch.tensor(NESTED, device=cls.device)[blk]
    return det, (det & ((rule == 1) | (blk == 8)))          # det, exception

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask", required=True, choices=["det", "exc"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=24000)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev, L, nl, Bt = "cuda", 192, 6, 256
    name = f"m27_{args.mask}only_seed{args.seed}"
    base = os.path.dirname(os.path.abspath(__file__))
    run = os.path.join(base, "runs", name)
    os.makedirs(os.path.join(run, "ckpts"), exist_ok=True)
    g_ev = torch.Generator().manual_seed(4242)
    eseq, ecls = gen_batch27(2048, L, g_ev)
    torch.save({"eseq": eseq, "ecls": ecls}, os.path.join(run, "evalset.pt"))
    eseq_d, ecls_d = eseq.to(dev), ecls.to(dev)
    edet, eexc = class_bits(ecls_d)
    blk = torch.where(edet, ((ecls_d - 2) // 3) % 9, torch.zeros_like(ecls_d))
    rule = torch.tensor(NESTED, device=dev)[blk]
    G = {"rnd": ~edet, "xor": edet & (rule == 0) & (blk != 8),
         "xnor": edet & (rule == 1), "flip": edet & (blk == 8)}
    json.dump({"mask": args.mask, "seed": args.seed, "L": L, "nl": nl,
               "steps": args.steps, "batch": Bt, "lr": 3e-4},
              open(os.path.join(run, "config.json"), "w"))
    model = GPTV(L=L, nl=nl, vocab=2).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.0)
    g = torch.Generator().manual_seed(args.seed + 777)
    logf = open(os.path.join(run, "eval.jsonl"), "w")

    def evaluate(step):
        model.eval()
        with torch.no_grad():
            lp = F.log_softmax(model(eseq_d)[:, :-1].double(), -1)
            ce = -lp.gather(-1, eseq_d[:, 1:, None]).squeeze(-1)
            rec = {"step": step, "ce_avg": float(ce[:, 20:].mean())}
            for k, m in G.items():
                mm = m[:, 1:]; mm[:, :19] = False
                rec[f"ce_{k}"] = float(ce[mm].mean())
            logf.write(json.dumps(rec) + "\n"); logf.flush()
        model.train()

    for step in range(args.steps + 1):
        seq, cls = gen_batch27(Bt, L, g)
        seq, cls = seq.to(dev), cls.to(dev)
        det, exc = class_bits(cls)
        mask = (det if args.mask == "det" else exc)[:, 1:]
        lg = model(seq[:, :-1])
        lp = F.log_softmax(lg, -1)
        ce = -lp.gather(-1, seq[:, 1:, None]).squeeze(-1)
        loss = ce[mask].mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0:
            evaluate(step)
        if (step <= 2000 and step % 100 == 0) or step % 500 == 0:
            torch.save(model.state_dict(),
                       os.path.join(run, "ckpts", f"ckpt{step:06d}.pt"))
    print(f"done {name}")

if __name__ == "__main__":
    main()
