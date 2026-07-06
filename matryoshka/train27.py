"""Matryoshka RRXOR (PREREG3): period-27 exception tower over the RRXOR det slot.

Nested pattern (per 9-block super-period): XNOR at blocks {2,5}, block 8 flips back to XOR.
Flat control (--flat): XNOR at blocks {0,4}, no flip (same match rates, no Z9 coherence).

Usage: CUDA_VISIBLE_DEVICES=<gpu> python train27.py --nl 6 --seed 0 [--flat] [--full_ckpt]
"""
import argparse, json, os
import torch
import torch.nn.functional as F
from train import GPT

NESTED = [0, 0, 1, 0, 0, 1, 0, 0, 0]   # 1 = XNOR; block 8 is the fake exception (FLIP)
FLAT = [1, 0, 0, 0, 1, 0, 0, 0, 0]     # XNOR at blocks {0,4}; no flip class

def gen_batch27(B, L, g, flat=False):
    pattern = torch.tensor(FLAT if flat else NESTED)
    nblk = L // 3 + 11
    r1 = torch.randint(0, 2, (B, nblk), generator=g)
    r2 = torch.randint(0, 2, (B, nblk), generator=g)
    rule = pattern[torch.arange(nblk) % 9]
    det = r1 ^ r2 ^ rule[None, :]
    blocks = torch.stack([r1, r2, det], dim=2).reshape(B, -1)
    off = torch.randint(0, 27, (B,), generator=g)
    t = torch.arange(L)
    seq = torch.gather(blocks, 1, off[:, None] + t[None, :])
    cls27 = (off[:, None] + t[None, :]) % 27
    return seq, cls27

def class_masks(cls27, flat=False):
    """Boolean masks over positions: rnd / xor-det / xnor-det / flip-det."""
    pattern = torch.tensor(FLAT if flat else NESTED, device=cls27.device)
    is_det = cls27 % 3 == 2
    blk = cls27 // 3
    is_xnor = is_det & (pattern[blk] == 1)
    if flat:
        is_flip = torch.zeros_like(is_det)
    else:
        is_flip = is_det & (blk == 8)
    is_xor = is_det & ~is_xnor & ~is_flip
    return {"rnd": ~is_det, "xor": is_xor, "xnor": is_xnor, "flip": is_flip}

@torch.no_grad()
def evaluate27(model, eseq, ecls, flat=False, min_pos=108):
    model.eval()
    logits = model(eseq[:, :-1])
    tgt = eseq[:, 1:]
    logp = F.log_softmax(logits, dim=-1)
    ce = -logp.gather(-1, tgt[..., None]).squeeze(-1)
    xorval = eseq[:, 1:-1] ^ eseq[:, :-2]
    pxor = logp[:, 1:].gather(-1, xorval[..., None]).squeeze(-1).exp()
    masks = class_masks(ecls[:, 1:], flat)
    pos = torch.arange(1, eseq.shape[1], device=eseq.device)
    late = (pos >= min_pos)[None, :]
    out = {}
    for k, m in masks.items():
        mm = m & late
        out[f"ce_{k}"] = ce[mm].mean().item() if mm.any() else float("nan")
        mm2 = m[:, 1:] & late[:, 1:]
        out[f"pxor_{k}"] = pxor[mm2].mean().item() if mm2.any() else float("nan")
    out["ce_avg"] = ce[late.expand_as(ce)].mean().item()
    model.train()
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nl", type=int, default=6)
    ap.add_argument("--steps", type=int, default=16000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--L", type=int, default=192)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--flat", action="store_true")
    ap.add_argument("--full_ckpt", action="store_true")
    args = ap.parse_args()

    dev = "cuda"
    torch.manual_seed(args.seed)
    name = f"m27{'flat' if args.flat else ''}_nl{args.nl}_seed{args.seed}"
    run = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", name)
    os.makedirs(os.path.join(run, "ckpts"), exist_ok=True)
    json.dump(vars(args), open(os.path.join(run, "config.json"), "w"))

    g_train = torch.Generator().manual_seed(50_000 + args.seed)
    g_eval = torch.Generator().manual_seed(4321)
    eseq, ecls = gen_batch27(2048, args.L, g_eval, args.flat)
    eseq, ecls = eseq.to(dev), ecls.to(dev)
    torch.save({"eseq": eseq.cpu(), "ecls": ecls.cpu()}, os.path.join(run, "evalset.pt"))

    model = GPT(L=args.L, d=args.d, nl=args.nl).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)

    def eval_now(s):
        return s % 20 == 0 if s <= 2000 else s % 100 == 0

    def ckpt_now(s):
        if args.full_ckpt:
            return s % 50 == 0 if s <= 2000 else s % 400 == 0
        return s % 2000 == 0 or s == args.steps

    logf = open(os.path.join(run, "eval.jsonl"), "w")
    for step in range(args.steps + 1):
        if eval_now(step):
            m = evaluate27(model, eseq, ecls, args.flat)
            m["step"] = step
            logf.write(json.dumps(m) + "\n"); logf.flush()
            if step % 2000 == 0:
                print(f"step {step:6d}  rnd {m['ce_rnd']:.4f}  xor {m['ce_xor']:.4f}  "
                      f"xnor {m['ce_xnor']:.4f}  flip {m['ce_flip']:.4f}  "
                      f"avg {m['ce_avg']:.4f}", flush=True)
        if ckpt_now(step):
            torch.save(model.state_dict(), os.path.join(run, "ckpts", f"step{step:06d}.pt"))
        if step == args.steps:
            break
        seq, _ = gen_batch27(args.batch, args.L, g_train, args.flat)
        seq = seq.to(dev)
        for gp in opt.param_groups:
            gp["lr"] = args.lr * min(1.0, (step + 1) / 100)
        logits = model(seq[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, 2), seq[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    logf.close()
    print("done", flush=True)

if __name__ == "__main__":
    main()
