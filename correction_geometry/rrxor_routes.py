"""Route A (full belief simplex, K states, direct) vs Route B (xor + phase-3).
Enumerate exact belief states by BFS on the filter; train supervised heads for
(A) state-ID K-way, (B1) xor 2-way, (B2) phase 3-way; measure tau95."""
import argparse, json, os
import torch
import torch.nn.functional as F
from train_multi import GPTV
from train import gen_batch

def filter_states():
    """belief = (weights over next-phase hypotheses, last two bits). BFS."""
    import itertools
    def norm(w):
        s = sum(w); return tuple(round(x / s, 6) for x in w)
    states, todo = {}, []
    start = (norm((1, 1, 1)), None, None)
    todo.append(start); states[start] = 0
    while todo:
        w, b1, b2 = todo.pop()
        for x in (0, 1):
            nw = []
            for h in range(3):
                wh = dict(zip((0, 1, 2), w))[h] if isinstance(w, dict) else w[h]
                if wh == 0:
                    nw.append(0.0); continue
                if h == 2:
                    if b1 is None or b2 is None:
                        nw.append(wh * 0.5)
                    else:
                        nw.append(wh if x == (b1 ^ b2) else 0.0)
                else:
                    nw.append(wh * 0.5)
            if sum(nw) == 0:
                continue
            nxt = (norm(tuple(nw[(h - 1) % 3] for h in range(3))), b2, x)
            if nxt not in states:
                states[nxt] = len(states); todo.append(nxt)
    return states

def minimize(states):
    """bisimulation: partition by (emission dist, successor classes) to fixpoint."""
    idx = {st: i for st, i in states.items()}
    trans, emit = {}, {}
    for st in states:
        w, b1, b2 = st
        p1 = sum(w[h] * (0.5 if (h != 2 or b1 is None or b2 is None) else float((b1 ^ b2) == 1)) for h in range(3))
        emit[st] = round(p1, 6)
        for x in (0, 1):
            nw = []
            for h in range(3):
                wh = w[h]
                if wh == 0: nw.append(0.0); continue
                if h == 2:
                    if b1 is None or b2 is None: nw.append(wh * 0.5)
                    else: nw.append(wh if x == (b1 ^ b2) else 0.0)
                else: nw.append(wh * 0.5)
            if sum(nw) == 0:
                trans[(st, x)] = None; continue
            sm = sum(nw)
            nxt = (tuple(round(v / sm, 6) for v in (nw[(h - 1) % 3] for h in range(3))), b2, x)
            trans[(st, x)] = nxt
    cls = {st: emit[st] for st in states}
    for _ in range(60):
        sig = {st: (cls[st], cls.get(trans[(st, 0)]), cls.get(trans[(st, 1)])) for st in states}
        uniq = {v: i for i, v in enumerate(sorted(set(sig.values()), key=str))}
        new = {st: uniq[sig[st]] for st in states}
        if new == cls: break
        cls = new
    return cls, len(set(cls.values()))

def state_seq(seq, states):
    B, L = seq.shape
    out = torch.zeros(B, L, dtype=torch.long)
    for b in range(B):
        cur = ((1/3, 1/3, 1/3), None, None)
        key0 = (tuple(round(x, 6) for x in cur[0]), None, None)
        for t in range(L):
            x = int(seq[b, t])
            w, b1, b2 = cur
            nw = []
            for h in range(3):
                wh = w[h]
                if wh == 0: nw.append(0.0); continue
                if h == 2:
                    if b1 is None or b2 is None: nw.append(wh * 0.5)
                    else: nw.append(wh if x == (b1 ^ b2) else 0.0)
                else: nw.append(wh * 0.5)
            s = sum(nw)
            nw = tuple(round(v / s, 6) for v in (nw[(h - 1) % 3] for h in range(3)))
            cur = (nw, b2, x)
            out[b, t] = states.get(cur, 0)
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", choices=["A", "B1", "B2", "enum"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=6000)
    args = ap.parse_args()
    states = filter_states()
    cls, K = minimize(states)
    states = {st: cls[st] for st in states}
    if args.route == "enum":
        print(f"raw filter states: {len(cls)}; minimized belief states: {K}")
        raise SystemExit
    torch.manual_seed(args.seed)
    dev = "cuda"
    g = torch.Generator().manual_seed(args.seed + 777)
    g_ev = torch.Generator().manual_seed(4242)
    eseq, etypes = gen_batch(512, 96, g_ev)
    def targets(seq, types):
        if args.route == "A":
            return state_seq(seq, states)
        if args.route == "B1":
            t = seq[:, 1:] ^ seq[:, :-1]
            return torch.cat([torch.zeros(seq.shape[0], 1, dtype=torch.long), t], 1)
        return types.long() % 3
    etgt = targets(eseq, etypes).to(dev)
    eseq_d = eseq.to(dev)
    Kout = {"A": K, "B1": 2, "B2": 3}[args.route]
    model = GPTV(L=96, nl=4, vocab=max(Kout, 2)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.0)
    run = f"runs/route_rrxor_{args.route}_s{args.seed}"
    os.makedirs(run, exist_ok=True)
    logf = open(f"{run}/eval.jsonl", "w")
    for step in range(args.steps + 1):
        seq, types = gen_batch(128, 96, g)
        tgt = targets(seq, types).to(dev)
        seq = seq.to(dev)
        lg = model(seq)
        loss = F.cross_entropy(lg[:, 10:].reshape(-1, max(Kout, 2)), tgt[:, 10:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 25 == 0:
            model.eval()
            with torch.no_grad():
                acc = float((model(eseq_d)[:, 10:].argmax(-1) == etgt[:, 10:]).float().mean())
            logf.write(json.dumps({"step": step, "acc": acc}) + "\n"); logf.flush()
            model.train()
    print("done", args.route, args.seed)
