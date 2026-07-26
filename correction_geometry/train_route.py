"""PREREG13 (Test 2): route-competition task.
Items (order shuffled per sequence): [source token(s), marker, answer].
Route-1 = lookup in alphabet {0,1} (perm p1); Route-2 = lookup {2,3} (p2) in arms
ll1/ll2, or XOR of two tokens {4,5} in arm lx. One shared item (both sources,
same z). Decoys pad every sequence to M=10 items. Answers in {6,7}; markers
8/9/10/11 (r1/r2/shared/decoy); decoy source 12. z binary. Dense CE.
Eval: aux accuracies, shared acc, conflict preference (sources disagree)."""
import argparse, json, os
import torch
import torch.nn.functional as F
from train_multi import GPTV

M_ITEMS, L = 10, 44
A1, A2, XA, ANS = [0, 1], [2, 3], [4, 5], [6, 7]
MK1, MK2, MKS, MKD, DSRC, PAD = 8, 9, 10, 11, 12, 13

def gen(B, m1, m2, arm, g, conflict=False, corrupt=None, noshared=False):
    seqs = torch.full((B, L), PAD, dtype=torch.long)
    meta = {"sh_ans": torch.zeros(B, dtype=torch.long), "sh_pos": torch.zeros(B, dtype=torch.long),
            "z2": torch.zeros(B, dtype=torch.long),
            "aux1": [[] for _ in range(B)], "aux2": [[] for _ in range(B)]}
    p1, p2 = ((0, 1), (0, 1)) if arm != "ll2" else ((1, 0), (1, 0))
    for b in range(B):
        items = []
        for _ in range(m1):
            z = int(torch.randint(0, 2, (1,), generator=g))
            items.append(("a1", [A1[p1[z]], MK1, ANS[z]]))
        for _ in range(m2):
            z = int(torch.randint(0, 2, (1,), generator=g))
            if arm == "lx":
                t1 = int(torch.randint(0, 2, (1,), generator=g))
                items.append(("a2", [XA[t1], XA[t1 ^ z], MK2, ANS[z]]))
            else:
                items.append(("a2", [A2[p2[z]], MK2, ANS[z]]))
        z = int(torch.randint(0, 2, (1,), generator=g))
        z2 = z if not conflict else 1 - z
        z1c = (1 - z) if corrupt == 1 else z
        z2c = (1 - z2) if corrupt == 2 else z2
        if arm == "lx":
            t1 = int(torch.randint(0, 2, (1,), generator=g))
            sh = ("sh", [A1[p1[z1c]], XA[t1], XA[t1 ^ z2c], MKS, ANS[z]])
        else:
            sh = ("sh", [A1[p1[z1c]], A2[p2[z2c]], MKS, ANS[z]])
        if not noshared:
            items.append(sh)
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
    ap.add_argument("--arm", choices=["ll1", "ll2", "lx"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--noshared", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = "cuda"
    name = f"route_{args.arm}_m{args.m1}x{args.m2}_s{args.seed}" + ("_ctl" if args.noshared else "")
    base = os.path.dirname(os.path.abspath(__file__))
    run = os.path.join(base, "runs", name)
    os.makedirs(run, exist_ok=True)
    g_ev = torch.Generator().manual_seed(4242)
    ev, evm = gen(512, args.m1, args.m2, args.arm, g_ev, noshared=args.noshared)
    gc = torch.Generator().manual_seed(4243)
    evc, evcm = gen(512, args.m1, args.m2, args.arm, gc, conflict=True)
    ev, evc = ev.to(dev), evc.to(dev)
    model = GPTV(L=L, nl=4, vocab=16).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.0)
    g = torch.Generator().manual_seed(args.seed + 777)
    logf = open(os.path.join(run, "eval.jsonl"), "w")
    json.dump(vars(args), open(os.path.join(run, "config.json"), "w"))
    for step in range(args.steps + 1):
        seq, _ = gen(128, args.m1, args.m2, args.arm, g, noshared=args.noshared)
        seq = seq.to(dev)
        lg = model(seq[:, :-1])
        loss = F.cross_entropy(lg.reshape(-1, 16), seq[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0:
            model.eval()
            with torch.no_grad():
                lp = model(ev[:, :-1])
                rec = {"step": step, "acc1": acc_at(lp, evm, "aux1"), "acc2": acc_at(lp, evm, "aux2")}
                if args.noshared:
                    logf.write(json.dumps(rec) + "\n"); logf.flush(); model.train(); continue
                sp = evm["sh_pos"].to(dev)
                pl = lp[torch.arange(512), sp - 1]
                rec["acc_sh"] = float((pl.argmax(-1) == evm["sh_ans"].to(dev)).float().mean())
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
