"""Recompute ce_last (+ m27 classes) per checkpoint for the dense baselines."""
import glob, json, os
import torch
import torch.nn.functional as F

base = os.path.dirname(os.path.abspath(__file__))
dev = "cuda"

def run(name, nl, L, vocab, m27=False):
    from train_multi import GPTV
    ev = torch.load(os.path.join(base, "runs", name, "evalset.pt"))
    eseq = ev["eseq"][:2048].to(dev)
    if m27:
        from train27 import NESTED
        cl = ev["ecls"][:2048][:, L - 1]
        det = cl % 3 == 2
        blk = torch.where(det, ((cl - 2) // 3) % 9, torch.zeros_like(cl))
        rule = torch.tensor(NESTED)[blk]
        groups = {"rnd": ~det, "xor": det & (rule == 0) & (blk != 8),
                  "xnor": det & (rule == 1), "flip": det & (blk == 8)}
    # baselines were trained with train.py/train27.py GPT (identical arch, vocab-2
    # via wte(2)) or train_multi GPTV; load state dicts into GPTV — key names match.
    model = GPTV(L=L, nl=nl, vocab=vocab).to(dev)
    out = open(os.path.join(base, "runs", name, "celast.jsonl"), "w")
    for c in sorted(glob.glob(os.path.join(base, "runs", name, "ckpts", "*.pt"))):
        st = int(os.path.basename(c)[4:10])
        model.load_state_dict(torch.load(c, map_location=dev)); model.eval()
        with torch.no_grad():
            lp = F.log_softmax(model(eseq)[:, :-1].double(), -1)
            ce = -lp.gather(-1, eseq[:, 1:, None]).squeeze(-1)
        rec = {"step": st, "ce_last": float(ce[:, -1].mean())}
        if m27:
            for k, m in groups.items():
                rec[f"ce_last_{k}"] = float(ce[m, -1].mean())
        out.write(json.dumps(rec) + "\n")
    out.close()
    print("done", name)

run("seed0", 4, 96, 2)
run("m27_nl6_seed1", 6, 192, 2, m27=True)
run("mess3_seed0", 4, 96, 3)
