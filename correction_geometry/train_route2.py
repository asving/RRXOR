"""PREREG13 v2: shuffled shared-source order (instrument fix), complexity-reversal
arm (xl: route-1=XOR, route-2=lookup), and the Codex crossover (phase 1: lookup
masked at shared + m1=0 so XOR owns the slot; phase 2: lookup enabled)."""
import argparse, json, os
import torch
import torch.nn.functional as F
from train_multi import GPTV

M_ITEMS, L = 10, 44
A1, A2, XA, ANS = [0, 1], [2, 3], [4, 5], [6, 7]
MK1, MK2, MKS, MKD, DSRC, PAD = 8, 9, 10, 11, 12, 13

def gen(B, m1, m2, arm, g, conflict=False, mask1=False, noshared=False):
    seqs = torch.full((B, L), PAD, dtype=torch.long)
    meta = {"sh_ans": torch.zeros(B, dtype=torch.long), "sh_pos": torch.zeros(B, dtype=torch.long),
            "z2": torch.zeros(B, dtype=torch.long), "aux1": [[] for _ in range(B)], "aux2": [[] for _ in range(B)]}
    for b in range(B):
        items = []
        def r1blk(z):
            if arm == "xl":
                t1 = int(torch.randint(0, 2, (1,), generator=g)); return [XA[t1], XA[t1 ^ z]]
            return [A1[z]]
        def r2blk(z):
            if arm == "lx":
                t1 = int(torch.randint(0, 2, (1,), generator=g)); return [XA[t1], XA[t1 ^ z]]
            if arm == "p3":
                t1 = int(torch.randint(0, 2, (1,), generator=g)); t2 = int(torch.randint(0, 2, (1,), generator=g))
                return [XA[t1], XA[t2], XA[t1 ^ t2 ^ z]]
            return [A2[z]]
        for _ in range(m1):
            z = int(torch.randint(0, 2, (1,), generator=g))
            items.append(("a1", r1blk(z) + [MK1, ANS[z]]))
        for _ in range(m2):
            z = int(torch.randint(0, 2, (1,), generator=g))
            items.append(("a2", r2blk(z) + [MK2, ANS[z]]))
        z = int(torch.randint(0, 2, (1,), generator=g))
        z2 = 1 - z if conflict else z
        b1 = [DSRC] if mask1 else r1blk(z)
        b2 = r2blk(z2)
        blocks = [b1, b2] if int(torch.randint(0, 2, (1,), generator=g)) else [b2, b1]
        if not noshared:
            items.append(("sh", blocks[0] + blocks[1] + [MKS, ANS[z]]))
        while len(items) < M_ITEMS:
            items.append(("d", [DSRC, MKD, ANS[int(torch.randint(0, 2, (1,), generator=g))]]))
        order = torch.randperm(len(items), generator=g)
        pos = 0
        for oi in order:
            tag, toks = items[oi]
            for tk in toks:
                seqs[b, pos] = tk; pos += 1
            if tag == "sh":
                meta["sh_ans"][b] = toks[-1]; meta["sh_pos"][b] = pos - 1; meta["z2"][b] = ANS[z2]
            elif tag == "a1":
                meta["aux1"][b].append((pos - 1, toks[-1]))
            elif tag == "a2":
                meta["aux2"][b].append((pos - 1, toks[-1]))
    return seqs, meta

def acc_at(lp, meta, key):
    hits, tot = 0, 0
    for b in range(lp.shape[0]):
        for (p, a) in meta[key][b]:
            hits += int(lp[b, p - 1].argmax() == a); tot += 1
    return hits / max(tot, 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m1", type=int, required=True)
    ap.add_argument("--m2", type=int, required=True)
    ap.add_argument("--arm", choices=["ll1", "lx", "xl", "p3"], required=True)
    ap.add_argument("--nl", type=int, default=4)
    ap.add_argument("--noshared", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--cross", type=int, default=0, help="phase-1 steps with lookup masked")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = "cuda"
    name = f"r2_{args.arm}_m{args.m1}x{args.m2}_s{args.seed}" + (f"_x{args.cross}" if args.cross else "") + (f"_nl{args.nl}" if args.nl != 4 else "") + ("_ns" if args.noshared else "")
    run = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", name)
    os.makedirs(run, exist_ok=True)
    g_ev = torch.Generator().manual_seed(4242)
    ev, evm = gen(512, args.m1, args.m2, args.arm, g_ev, noshared=args.noshared)
    gc = torch.Generator().manual_seed(4243)
    evc, evcm = gen(512, args.m1, args.m2, args.arm, gc, conflict=True)
    ev, evc = ev.to(dev), evc.to(dev)
    model = GPTV(L=L, nl=args.nl, vocab=16).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.0)
    g = torch.Generator().manual_seed(args.seed + 777)
    logf = open(os.path.join(run, "eval.jsonl"), "w")
    json.dump(vars(args), open(os.path.join(run, "config.json"), "w"))
    for step in range(args.steps + 1):
        ph1 = step < args.cross
        seq, _ = gen(128, 0 if ph1 else args.m1, args.m2, args.arm, g, mask1=ph1, noshared=args.noshared)
        seq = seq.to(dev)
        lg = model(seq[:, :-1])
        loss = F.cross_entropy(lg.reshape(-1, 16), seq[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 25 == 0:
            model.eval()
            with torch.no_grad():
                lp = model(ev[:, :-1])
                rec = {"step": step, "acc1": acc_at(lp, evm, "aux1"), "acc2": acc_at(lp, evm, "aux2")}
                if args.noshared:
                    logf.write(json.dumps(rec) + "\n"); logf.flush(); model.train(); continue
                sp = evm["sh_pos"].to(dev)
                rec["acc_sh"] = float((lp[torch.arange(512), sp - 1].argmax(-1) == evm["sh_ans"].to(dev)).float().mean())
                lpc = model(evc[:, :-1])
                plc = lpc[torch.arange(512), evcm["sh_pos"].to(dev) - 1]
                p1 = plc.gather(1, evcm["sh_ans"][:, None].to(dev)).squeeze(1)
                p2 = plc.gather(1, evcm["z2"][:, None].to(dev)).squeeze(1)
                rec["pref1"] = float((p1 > p2).float().mean())
                logf.write(json.dumps(rec) + "\n"); logf.flush()
            model.train()
    torch.save(model.state_dict(), os.path.join(run, "final.pt"))
    print("done", name)

if __name__ == "__main__":
    main()
