"""Training for PREREG6: factored product process (vocab 4) and mess3 (vocab 3).

Same recipe as train.py (GPT d=128 nl=4 L=96, batch 256, lr 3e-4, 8000 steps,
ckpts every 25 to 1500 then every 250) with a vocab-general GPT and new generators.
Usage: train_multi.py --task factored|mess3 --seed 0
"""
import argparse, json, math, os
import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, d, nh):
        super().__init__()
        self.nh = nh
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        B, L, D = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).chunk(3, -1)
        q, k, v = (z.view(B, L, self.nh, D // self.nh).transpose(1, 2)
                   for z in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, L, D))
        x = x + self.mlp(self.ln2(x))
        return x

class GPTV(nn.Module):
    def __init__(self, L=96, d=128, nl=4, nh=4, vocab=4):
        super().__init__()
        self.vocab = vocab
        self.wte = nn.Embedding(vocab, d)
        self.wpe = nn.Embedding(L, d)
        self.blocks = nn.ModuleList([Block(d, nh) for _ in range(nl)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, seq, return_resid=False):
        B, L = seq.shape
        x = self.wte(seq) + self.wpe(torch.arange(L, device=seq.device))[None]
        resid = [x]
        for blk in self.blocks:
            x = blk(x)
            resid.append(x)
        logits = self.head(self.lnf(x))
        return (logits, resid) if return_resid else logits

def gen_factored(B, L, g):
    """Factor A: rrxor3 (offset mod 3). Factor B: repeat-2 (offset mod 2).
    Token = 2*a + b. Returns seq, typeA ((offA+t)%3, det when 2), typeB ((offB+t)%2,
    copy when 1)."""
    nblk = L // 3 + 2
    r1 = torch.randint(0, 2, (B, nblk), generator=g)
    r2 = torch.randint(0, 2, (B, nblk), generator=g)
    blocks = torch.stack([r1, r2, r1 ^ r2], 2).reshape(B, -1)
    offA = torch.randint(0, 3, (B,), generator=g)
    t = torch.arange(L)
    a = torch.gather(blocks, 1, offA[:, None] + t[None])
    typeA = (offA[:, None] + t[None]) % 3
    nb2 = L // 2 + 2
    r = torch.randint(0, 2, (B, nb2), generator=g)
    blocks2 = torch.stack([r, r], 2).reshape(B, -1)
    offB = torch.randint(0, 2, (B,), generator=g)
    b = torch.gather(blocks2, 1, offB[:, None] + t[None])
    typeB = (offB[:, None] + t[None]) % 2
    return 2 * a + b, typeA, typeB

def gen_mess3(B, L, g, x=0.15, a=0.6):
    seq = torch.zeros(B, L, dtype=torch.long)
    states = torch.zeros(B, L, dtype=torch.long)
    s = torch.randint(0, 3, (B,), generator=g)
    for t in range(L):
        states[:, t] = s
        u = torch.rand(B, generator=g)
        emit = s.clone()
        other = (torch.rand(B, generator=g) < 0.5).long() + 1
        emit = torch.where(u < 2 * x, (s + other) % 3, s)
        emit = torch.where(u < x, (s + 1) % 3, emit)
        emit = torch.where((u >= x) & (u < 2 * x), (s + 2) % 3, emit)
        seq[:, t] = emit
        v = torch.rand(B, generator=g)
        sw = (torch.rand(B, generator=g) < 0.5).long() + 1
        s = torch.where(v < a, s, (s + sw) % 3)
    return seq, states

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["factored", "mess3"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=8000)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = "cuda"
    L, Bt = 96, 256
    vocab = 4 if args.task == "factored" else 3
    name = f"{args.task}_seed{args.seed}"
    base = os.path.dirname(os.path.abspath(__file__))
    run = os.path.join(base, "runs", name)
    os.makedirs(os.path.join(run, "ckpts"), exist_ok=True)
    g_ev = torch.Generator().manual_seed(12345)
    if args.task == "factored":
        eseq, etA, etB = gen_factored(2048, L, g_ev)
        torch.save({"eseq": eseq, "etypeA": etA, "etypeB": etB},
                   os.path.join(run, "evalset.pt"))
    else:
        eseq, est = gen_mess3(2048, L, g_ev)
        torch.save({"eseq": eseq, "estates": est}, os.path.join(run, "evalset.pt"))
    eseq_d = eseq.to(dev)
    json.dump({"task": args.task, "seed": args.seed, "L": L, "d": 128, "nl": 4,
               "vocab": vocab, "steps": args.steps, "batch": Bt, "lr": 3e-4},
              open(os.path.join(run, "config.json"), "w"))
    model = GPTV(L=L, vocab=vocab).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.0)
    g = torch.Generator().manual_seed(args.seed + 777)
    logf = open(os.path.join(run, "eval.jsonl"), "w")

    def evaluate(step):
        model.eval()
        with torch.no_grad():
            lg = model(eseq_d)
            lp = F.log_softmax(lg[:, :-1].double(), -1)
            tgt = eseq_d[:, 1:]
            ce = -lp.gather(-1, tgt[:, :, None]).squeeze(-1)
            rec = {"step": step, "ce_avg": float(ce[:, 20:].mean())}
            if args.task == "factored":
                pa = torch.logsumexp(
                    torch.stack([lp[..., 2], lp[..., 3]], -1), -1)
                pa0 = torch.logsumexp(torch.stack([lp[..., 0], lp[..., 1]], -1), -1)
                la = torch.where((tgt // 2).bool(), pa, pa0)
                pb = torch.logsumexp(torch.stack([lp[..., 1], lp[..., 3]], -1), -1)
                pb0 = torch.logsumexp(torch.stack([lp[..., 0], lp[..., 2]], -1), -1)
                lb = torch.where((tgt % 2).bool(), pb, pb0)
                rec["ce_A"] = float(-la[:, 20:].mean())
                rec["ce_B"] = float(-lb[:, 20:].mean())
            logf.write(json.dumps(rec) + "\n"); logf.flush()
        model.train()

    for step in range(args.steps + 1):
        if args.task == "factored":
            seq, _, _ = gen_factored(Bt, L, g)
        else:
            seq, _ = gen_mess3(Bt, L, g)
        seq = seq.to(dev)
        lg = model(seq[:, :-1])
        loss = F.cross_entropy(lg.reshape(-1, vocab), seq[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 20 == 0:
            evaluate(step)
        if (step <= 1500 and step % 25 == 0) or step % 250 == 0:
            torch.save(model.state_dict(),
                       os.path.join(run, "ckpts", f"ckpt{step:06d}.pt"))
    print(f"done {name}")

if __name__ == "__main__":
    main()
