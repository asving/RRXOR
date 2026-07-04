"""
Test the circuit hypothesis by progressive pruning:
  - kill L0 attn, L0 MLP, L1 attn, L1 MLP, L2 head 1
  - keep L2 head 0, L2 MLP, L3 attn, L3 MLP, embeddings, LN, unembed
If the circuit is correct, the pruned model should retain phase-2 accuracy ≈ 1.0
and flat phase-0/1 predictions.
"""

import argparse, json, os
import numpy as np
import torch
import torch.nn.functional as F

from rrxor_belief_experiment import (
    rrxor_tensors, stationary, sample_sequence, beliefs_from_emissions,
    MiniTransformer, STATE_PHASE,
)


@torch.no_grad()
def forward_with_skips(model, x, *, skip_attn=(), skip_mlp=(), kill_l2_head=None):
    """skip_attn/skip_mlp: iterables of layer indices whose attn/mlp output is set to 0
    (block becomes identity for that sublayer). kill_l2_head: head index to zero."""
    B, L = x.shape
    pos_idx = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
    h = model.tok(x) + model.pos(pos_idx)
    mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)

    for k, blk in enumerate(model.blocks):
        x_norm = blk.norm1(h)
        if k in skip_attn:
            attn_out = torch.zeros_like(h)
        elif k == 2 and kill_l2_head is not None:
            attn = blk.self_attn
            E, H = attn.embed_dim, attn.num_heads
            Dh = E // H
            qkv = F.linear(x_norm, attn.in_proj_weight, attn.in_proj_bias)
            q, kk_, v = qkv.chunk(3, dim=-1)
            q = q.view(B, L, H, Dh).transpose(1, 2)
            kk_ = kk_.view(B, L, H, Dh).transpose(1, 2)
            v = v.view(B, L, H, Dh).transpose(1, 2)
            scores = q @ kk_.transpose(-2, -1) / (Dh ** 0.5)
            scores = scores.masked_fill(mask, float("-inf"))
            attn_w = scores.softmax(dim=-1)
            attn_o = attn_w @ v
            attn_o[:, kill_l2_head, :, :] = 0
            attn_o = attn_o.transpose(1, 2).contiguous().view(B, L, E)
            attn_out = attn.out_proj(attn_o)
        else:
            attn_out, _ = blk.self_attn(x_norm, x_norm, x_norm,
                                        attn_mask=mask, need_weights=False)
        h_attn = h + attn_out

        if k in skip_mlp:
            mlp_out = torch.zeros_like(h_attn)
        else:
            x_norm2 = blk.norm2(h_attn)
            mlp_out = blk.linear2(blk.dropout(blk.activation(blk.linear1(x_norm2))))
        h = h_attn + mlp_out

    h_final = model.ln(h)
    logits = model.head(h_final)
    return logits


def eval_run(model, T, pi, rng, n_seqs, seq_len, **kwargs):
    ld, ph, xr = [], [], []
    ce_total, ce_n = 0.0, 0
    for _ in range(n_seqs):
        ems, _ = sample_sequence(T, pi, seq_len, rng)
        b = beliefs_from_emissions(T, pi, ems)
        phase = np.array([STATE_PHASE[np.argmax(bt)] for bt in b[1:]])
        xor = np.full_like(ems, -1, dtype=np.int64)
        xor[1:] = ems[1:] ^ ems[:-1]
        x = torch.from_numpy(ems).long().unsqueeze(0)
        logits = forward_with_skips(model, x, **kwargs).squeeze(0)  # (L, 2)
        # CE on next-token (predict ems[t+1] from position t, positions 0..L-2)
        target = torch.from_numpy(ems[1:]).long()
        ce = F.cross_entropy(logits[:-1], target).item()
        ce_total += ce * len(target); ce_n += len(target)
        ld_arr = (logits[:, 1] - logits[:, 0]).numpy()
        ld.append(ld_arr); ph.append(phase); xr.append(xor)
    ld = np.concatenate(ld); ph = np.concatenate(ph); xr = np.concatenate(xr)
    valid = xr >= 0
    out = {"ce_per_token": ce_total / max(ce_n, 1)}
    for p in (0, 1, 2):
        for v in (0, 1):
            m = valid & (ph == p) & (xr == v)
            if m.sum() == 0: continue
            out[f"ph{p}_xor{v}_mean"] = float(ld[m].mean())
            out[f"ph{p}_xor{v}_std"] = float(ld[m].std())
    m2 = valid & (ph == 2)
    pred = (ld[m2] > 0).astype(int)
    out["phase2_accuracy"] = float((pred == xr[m2]).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="results/rrxor_belief/naive_ctx")
    ap.add_argument("--outdir", default="results/rrxor_belief/naive_ctx/circuit")
    ap.add_argument("--n-seqs", type=int, default=120)
    ap.add_argument("--seq-len", type=int, default=128)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    prior = np.load(os.path.join(args.source, "results.npz"), allow_pickle=True)
    ta = prior["args"].item()
    model = MiniTransformer(
        vocab=2, d_model=ta["d_model"], n_heads=ta["n_heads"],
        n_layers=ta["n_layers"], max_len=ta["seq_len"] + 2,
    )
    model.load_state_dict(torch.load(os.path.join(args.source, "model.pt"), map_location="cpu"))
    model.eval()
    T = rrxor_tensors(); pi = stationary(T)

    configs = [
        ("clean", {}),
        ("kill_L0_attn", {"skip_attn": (0,)}),
        ("kill_L0_mlp", {"skip_mlp": (0,)}),
        ("kill_L0_all", {"skip_attn": (0,), "skip_mlp": (0,)}),
        ("kill_L1_all", {"skip_attn": (1,), "skip_mlp": (1,)}),
        ("kill_L01_all", {"skip_attn": (0, 1), "skip_mlp": (0, 1)}),
        ("kill_L01_all_and_L2head1",
            {"skip_attn": (0, 1), "skip_mlp": (0, 1), "kill_l2_head": 1}),
        # additional ablations of supposedly-needed parts as falsification controls
        ("kill_L2_mlp", {"skip_mlp": (2,)}),
        ("kill_L3_mlp", {"skip_mlp": (3,)}),
        ("kill_L2_head0_only", {"kill_l2_head": 0}),
    ]

    results = {}
    for name, kw in configs:
        rng = np.random.default_rng(42)
        res = eval_run(model, T, pi, rng, args.n_seqs, args.seq_len, **kw)
        results[name] = res
        print(f"\n[{name}]")
        for k, v in res.items():
            print(f"  {k} = {v}" if isinstance(v, float) else f"  {k} = {v}")

    with open(os.path.join(args.outdir, "circuit_prune.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.outdir}/circuit_prune.json")


if __name__ == "__main__":
    main()
