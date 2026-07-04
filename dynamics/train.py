"""RRXOR hierarchy test — see PREREG.md / PREREG2.md. Self-contained.

Usage: CUDA_VISIBLE_DEVICES=<gpu> python train.py --seed 0 [--xorbit_p P] [--bias_eps E]
  --xorbit_p P: feed b_t = x_t^x_{t-1} (flipped w.p. P) as an extra input channel (Test A)
  --bias_eps E: type-0 tokens ~ Bernoulli(0.5+E) — flat phase cue (Test B)
"""
import argparse, json, math, os
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- data ----------------

def gen_batch(B, L, g, bias_eps=0.0):
    """RRXOR: repeating blocks [r1, r2, r1^r2] with random phase offset per sequence.
    Returns seq (B,L) int64 and types (B,L) where type = (off + t) % 3; type 2 = xor token."""
    nblk = L // 3 + 2
    if bias_eps > 0:
        r1 = (torch.rand(B, nblk, generator=g) < 0.5 + bias_eps).long()
    else:
        r1 = torch.randint(0, 2, (B, nblk), generator=g)
    r2 = torch.randint(0, 2, (B, nblk), generator=g)
    blocks = torch.stack([r1, r2, r1 ^ r2], dim=2).reshape(B, -1)  # (B, 3*nblk)
    off = torch.randint(0, 3, (B,), generator=g)
    t = torch.arange(L)
    idx = off[:, None] + t[None, :]
    seq = torch.gather(blocks, 1, idx)
    types = (off[:, None] + t[None, :]) % 3
    return seq, types

def make_aux(seq, p, g):
    """b_t = seq[t] ^ seq[t-1], flipped with prob p (b_0 random)."""
    b = torch.zeros_like(seq)
    b[:, 1:] = seq[:, 1:] ^ seq[:, :-1]
    b[:, 0] = torch.randint(0, 2, (seq.shape[0],), generator=g)
    if p > 0:
        b = b ^ (torch.rand(seq.shape, generator=g) < p).long()
    return b

# ---------------- model ----------------

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
        q, k, v = (z.view(B, L, self.nh, D // self.nh).transpose(1, 2) for z in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, L, D))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab=2, L=96, d=128, nh=4, nl=4, aux=False):
        super().__init__()
        self.wte = nn.Embedding(vocab, d)
        self.wpe = nn.Embedding(L, d)
        self.waux = nn.Embedding(2, d) if aux else None
        self.blocks = nn.ModuleList([Block(d, nh) for _ in range(nl)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, seq, aux=None, return_resid=False):
        B, L = seq.shape
        x = self.wte(seq) + self.wpe(torch.arange(L, device=seq.device))[None]
        if self.waux is not None and aux is not None:
            x = x + self.waux(aux)
        resid = [x]
        for b in self.blocks:
            x = b(x)
            resid.append(x)
        logits = self.head(self.lnf(x))
        return (logits, resid) if return_resid else logits

# ---------------- eval ----------------

@torch.no_grad()
def evaluate(model, eseq, etypes, eaux=None, min_pos=33):
    model.eval()
    logits = model(eseq[:, :-1], None if eaux is None else eaux[:, :-1])
    tgt = eseq[:, 1:]
    logp = F.log_softmax(logits, dim=-1)
    ce = -logp.gather(-1, tgt[..., None]).squeeze(-1)  # (B, L-1), entry j is target pos j+1
    xorval = eseq[:, 1:-1] ^ eseq[:, :-2]              # candidate for target pos >= 2
    pxor = logp[:, 1:].gather(-1, xorval[..., None]).squeeze(-1).exp()
    ttypes = etypes[:, 1:]
    out = {}
    pos = torch.arange(1, eseq.shape[1], device=eseq.device)
    late = pos >= min_pos
    for ty in range(3):
        m = (ttypes == ty) & late[None, :]
        out[f"ce{ty}"] = ce[m].mean().item()
        m2 = (ttypes[:, 1:] == ty) & (pos[1:] >= min_pos)[None, :]
        out[f"pxor{ty}"] = pxor[m2].mean().item()
    out["ce_avg"] = (out["ce0"] + out["ce1"] + out["ce2"]) / 3
    model.train()
    return out

# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--L", type=int, default=96)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--nl", type=int, default=4)
    ap.add_argument("--xorbit_p", type=float, default=-1.0)  # <0 = no aux channel
    ap.add_argument("--bias_eps", type=float, default=0.0)
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--light", action="store_true")  # sparse checkpoints
    args = ap.parse_args()

    dev = "cuda"
    torch.manual_seed(args.seed)
    use_aux = args.xorbit_p >= 0
    run = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs",
                       f"{args.tag}seed{args.seed}")
    os.makedirs(os.path.join(run, "ckpts"), exist_ok=True)
    json.dump(vars(args) | {"aux": use_aux}, open(os.path.join(run, "config.json"), "w"))

    g_train = torch.Generator().manual_seed(10_000 + args.seed)
    g_eval = torch.Generator().manual_seed(1234)          # same eval set for all seeds
    eseq, etypes = gen_batch(2048, args.L, g_eval, args.bias_eps)
    eaux = make_aux(eseq, max(args.xorbit_p, 0.0), g_eval).to(dev) if use_aux else None
    eseq, etypes = eseq.to(dev), etypes.to(dev)
    torch.save({"eseq": eseq.cpu(), "etypes": etypes.cpu(),
                "eaux": None if eaux is None else eaux.cpu()},
               os.path.join(run, "evalset.pt"))

    model = GPT(L=args.L, d=args.d, nl=args.nl, aux=use_aux).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)

    def eval_now(s):
        return s % 10 == 0 if s <= 1500 else s % 50 == 0

    def ckpt_now(s):
        if args.light:
            return s % 1000 == 0 or s == args.steps
        return s % 25 == 0 if s <= 1500 else s % 250 == 0

    logf = open(os.path.join(run, "eval.jsonl"), "w")
    for step in range(args.steps + 1):
        if eval_now(step):
            m = evaluate(model, eseq, etypes, eaux)
            m["step"] = step
            logf.write(json.dumps(m) + "\n"); logf.flush()
            if step % 1000 == 0:
                print(f"step {step:5d}  ce0 {m['ce0']:.4f}  ce1 {m['ce1']:.4f}  "
                      f"ce2 {m['ce2']:.4f}  avg {m['ce_avg']:.4f}", flush=True)
        if ckpt_now(step):
            torch.save(model.state_dict(), os.path.join(run, "ckpts", f"step{step:06d}.pt"))
        if step == args.steps:
            break
        seq, _ = gen_batch(args.batch, args.L, g_train, args.bias_eps)
        aux = make_aux(seq, max(args.xorbit_p, 0.0), g_train).to(dev) if use_aux else None
        seq = seq.to(dev)
        for gp in opt.param_groups:
            gp["lr"] = args.lr * min(1.0, (step + 1) / 100)
        logits = model(seq[:, :-1], None if aux is None else aux[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, 2), seq[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    logf.close()
    print("done", flush=True)

if __name__ == "__main__":
    main()
