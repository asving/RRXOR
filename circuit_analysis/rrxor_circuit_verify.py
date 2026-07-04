"""
Follow-up causal verifications for the RRXOR circuit hypothesis:
  V1: patch the XOR-encoding direction at L2_post_mlp -> should flip the prediction at phase-2
  V2: ablate L2 head 0 (sharp t-1 attender) -> XOR computation should collapse
  V3: ablate L3 attention (phase resolver) -> phase-2 predictions should become uniform
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
def custom_forward(model, x, *, kill_l2_head=None, kill_l3_attn=False,
                    intervene_l2_post_mlp_fn=None, intervene_l3_post_attn_fn=None,
                    return_attn=False, return_cache=False):
    """Run the model with optional surgical interventions.

    kill_l2_head: int in {0,1} or None — zero out the contribution of one head in layer 2's attention.
    kill_l3_attn: bool — bypass layer 3 attention entirely (set its contribution to 0).
    intervene_l2_post_mlp_fn(h): callable that mutates h after layer 2's MLP.
    intervene_l3_post_attn_fn(h): callable that mutates h after layer 3's attention.
    """
    B, L = x.shape
    pos_idx = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
    h = model.tok(x) + model.pos(pos_idx)
    mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
    cache = {}

    for k, blk in enumerate(model.blocks):
        x_norm = blk.norm1(h)
        if k == 2 and kill_l2_head is not None:
            # Manually compute per-head attention so we can zero one head's contribution.
            attn = blk.self_attn
            E = attn.embed_dim
            H = attn.num_heads
            Dh = E // H
            qkv_w = attn.in_proj_weight
            qkv_b = attn.in_proj_bias
            qkv = F.linear(x_norm, qkv_w, qkv_b)
            q, kk_, v = qkv.chunk(3, dim=-1)
            # (B, L, H, Dh) -> (B, H, L, Dh)
            q = q.view(B, L, H, Dh).transpose(1, 2)
            kk_ = kk_.view(B, L, H, Dh).transpose(1, 2)
            v = v.view(B, L, H, Dh).transpose(1, 2)
            scores = q @ kk_.transpose(-2, -1) / (Dh ** 0.5)
            scores = scores.masked_fill(mask, float("-inf"))
            attn_w = scores.softmax(dim=-1)
            attn_out = attn_w @ v  # (B, H, L, Dh)
            # Zero target head
            attn_out[:, kill_l2_head, :, :] = 0
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, E)
            attn_out = attn.out_proj(attn_out)
        else:
            attn_out, _ = blk.self_attn(x_norm, x_norm, x_norm,
                                        attn_mask=mask, need_weights=False)
        if k == 3 and kill_l3_attn:
            attn_out = torch.zeros_like(attn_out)

        h_attn = h + attn_out
        if k == 3 and intervene_l3_post_attn_fn is not None:
            h_attn = intervene_l3_post_attn_fn(h_attn)
        if return_cache:
            cache[f"L{k}_post_attn"] = h_attn.clone()

        x_norm2 = blk.norm2(h_attn)
        mlp_out = blk.linear2(blk.dropout(blk.activation(blk.linear1(x_norm2))))
        h = h_attn + mlp_out
        if k == 2 and intervene_l2_post_mlp_fn is not None:
            h = intervene_l2_post_mlp_fn(h)
        if return_cache:
            cache[f"L{k}_post_mlp"] = h.clone()

    h_final = model.ln(h)
    logits = model.head(h_final)
    if return_cache:
        cache["logits"] = logits
        return logits, cache
    return logits


def find_xor_direction(model, T, pi, rng, n_seqs=80, seq_len=128):
    """Linear probe of L2_post_mlp -> XOR. Returns weight vector and bias for binary class."""
    feats, labels = [], []
    for _ in range(n_seqs):
        ems, _ = sample_sequence(T, pi, seq_len, rng)
        x = torch.from_numpy(ems).long().unsqueeze(0)
        _, cache = custom_forward(model, x, return_cache=True)
        h = cache["L2_post_mlp"].squeeze(0).numpy()
        xor = (ems[:-1] ^ ems[1:])  # XOR at positions 1..L-1, length L-1
        feats.append(h[1:])  # match length
        labels.append(xor)
    X = np.concatenate(feats, axis=0)
    y = np.concatenate(labels, axis=0)
    # one-vs-rest with two classes -> just (logit difference). Fit:
    # solve for w such that X w + b ≈ (2y-1)
    target = (2 * y - 1).astype(np.float32)
    Xaug = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    w_full, *_ = np.linalg.lstsq(Xaug, target, rcond=None)
    w, b = w_full[:-1], w_full[-1]
    pred = (X @ w + b > 0).astype(np.int64)
    acc = float((pred == y).mean())
    return w, b, acc


def find_phase_direction(model, T, pi, rng, n_seqs=80, seq_len=128):
    """Linear probe of L3_post_attn -> phase ∈ {0,1,2}. Returns weights (D, 3) and bias (3,)."""
    feats, phases = [], []
    for _ in range(n_seqs):
        ems, _ = sample_sequence(T, pi, seq_len, rng)
        b = beliefs_from_emissions(T, pi, ems)
        ph = np.array([STATE_PHASE[np.argmax(bt)] for bt in b[1:]])
        x = torch.from_numpy(ems).long().unsqueeze(0)
        _, cache = custom_forward(model, x, return_cache=True)
        feats.append(cache["L3_post_attn"].squeeze(0).numpy())
        phases.append(ph)
    X = np.concatenate(feats, axis=0)
    y = np.concatenate(phases, axis=0)
    Y_oh = np.eye(3)[y]
    Xaug = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    W, *_ = np.linalg.lstsq(Xaug, Y_oh, rcond=None)
    pred = (Xaug @ W).argmax(axis=1)
    acc = float((pred == y).mean())
    return W[:-1], W[-1], acc


def behavioral_xor(logits_per_seq, phase_per_seq, xor_per_seq):
    """Return per-phase mean and std of logit_diff[1]-logit_diff[0], plus phase-2 accuracy."""
    out = {}
    ld = np.concatenate([l[:, 1] - l[:, 0] for l in logits_per_seq])
    ph = np.concatenate(phase_per_seq)
    xr = np.concatenate(xor_per_seq)
    valid = xr >= 0
    for p in (0, 1, 2):
        for v in (0, 1):
            m = valid & (ph == p) & (xr == v)
            if m.sum() > 0:
                out[f"ph{p}_xor{v}"] = {
                    "n": int(m.sum()),
                    "mean": float(ld[m].mean()),
                    "std": float(ld[m].std()),
                }
    m2 = valid & (ph == 2)
    pred = (ld[m2] > 0).astype(int)
    out["phase2_accuracy"] = float((pred == xr[m2]).mean()) if m2.sum() else None
    return out


def run_clean(model, T, pi, rng, n_seqs=80, seq_len=128):
    logits, phases, xors = [], [], []
    for _ in range(n_seqs):
        ems, _ = sample_sequence(T, pi, seq_len, rng)
        b = beliefs_from_emissions(T, pi, ems)
        ph = np.array([STATE_PHASE[np.argmax(bt)] for bt in b[1:]])
        xr = np.full_like(ems, -1, dtype=np.int64)
        xr[1:] = ems[1:] ^ ems[:-1]
        x = torch.from_numpy(ems).long().unsqueeze(0)
        l = custom_forward(model, x).squeeze(0).numpy()
        logits.append(l); phases.append(ph); xors.append(xr)
    return logits, phases, xors


def run_with_intervention(model, T, pi, rng, n_seqs, seq_len, **kwargs):
    logits, phases, xors = [], [], []
    for _ in range(n_seqs):
        ems, _ = sample_sequence(T, pi, seq_len, rng)
        b = beliefs_from_emissions(T, pi, ems)
        ph = np.array([STATE_PHASE[np.argmax(bt)] for bt in b[1:]])
        xr = np.full_like(ems, -1, dtype=np.int64)
        xr[1:] = ems[1:] ^ ems[:-1]
        x = torch.from_numpy(ems).long().unsqueeze(0)
        l = custom_forward(model, x, **kwargs).squeeze(0).numpy()
        logits.append(l); phases.append(ph); xors.append(xr)
    return logits, phases, xors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="results/rrxor_belief/naive_ctx")
    ap.add_argument("--outdir", default="results/rrxor_belief/naive_ctx/circuit")
    ap.add_argument("--n-seqs", type=int, default=80)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    prior = np.load(os.path.join(args.source, "results.npz"), allow_pickle=True)
    train_args = prior["args"].item()

    model = MiniTransformer(
        vocab=2, d_model=train_args["d_model"], n_heads=train_args["n_heads"],
        n_layers=train_args["n_layers"], max_len=train_args["seq_len"] + 2,
    )
    model.load_state_dict(torch.load(os.path.join(args.source, "model.pt"), map_location="cpu"))
    model.eval()

    rng = np.random.default_rng(args.seed)
    T = rrxor_tensors()
    pi = stationary(T)

    # Find the XOR readout direction at L2_post_mlp and the phase readout at L3_post_attn.
    print("fitting probe directions...", flush=True)
    w_xor, b_xor, acc_xor = find_xor_direction(model, T, pi,
        np.random.default_rng(args.seed + 10), n_seqs=args.n_seqs, seq_len=args.seq_len)
    print(f"  XOR @ L2_post_mlp linear probe acc = {acc_xor:.3f}", flush=True)
    Wp, bp, acc_ph = find_phase_direction(model, T, pi,
        np.random.default_rng(args.seed + 11), n_seqs=args.n_seqs, seq_len=args.seq_len)
    print(f"  phase @ L3_post_attn linear probe acc = {acc_ph:.3f}", flush=True)

    # --- Clean baseline ---
    print("\n=== clean baseline ===", flush=True)
    logits, ph, xr = run_clean(model, T, pi, np.random.default_rng(args.seed + 20),
                                n_seqs=args.n_seqs, seq_len=args.seq_len)
    clean = behavioral_xor(logits, ph, xr)
    print(json.dumps(clean, indent=2), flush=True)

    # --- V1: flip the XOR direction at L2_post_mlp ---
    print("\n=== V1: project residual to flip XOR direction at L2_post_mlp ===", flush=True)
    w = torch.tensor(w_xor, dtype=torch.float32)
    bbias = float(b_xor)
    norm_w_sq = float((w * w).sum())

    def flip_xor_intervention(h):
        # h: (B, L, D). Compute score = h @ w + bias. To flip the sign, set
        # h_new = h - 2 * (score / ||w||^2) * w.
        # This reflects h across the hyperplane perpendicular to w that passes through
        # the decision boundary. After this, the new score = -score (mod 2*b adjustment).
        score = h @ w + bbias            # (B, L)
        coef = (score / norm_w_sq).unsqueeze(-1)
        return h - 2.0 * coef * w

    logits_v1, ph_v1, xr_v1 = run_with_intervention(
        model, T, pi, np.random.default_rng(args.seed + 20),
        n_seqs=args.n_seqs, seq_len=args.seq_len,
        intervene_l2_post_mlp_fn=flip_xor_intervention,
    )
    v1 = behavioral_xor(logits_v1, ph_v1, xr_v1)
    print(json.dumps(v1, indent=2), flush=True)

    # --- V2: ablate L2 head 0 ---
    print("\n=== V2: ablate L2 head 0 (sharp t-1 attender) ===", flush=True)
    logits_v2a, ph_v2a, xr_v2a = run_with_intervention(
        model, T, pi, np.random.default_rng(args.seed + 20),
        n_seqs=args.n_seqs, seq_len=args.seq_len, kill_l2_head=0,
    )
    v2a = behavioral_xor(logits_v2a, ph_v2a, xr_v2a)
    print("  kill head 0:", json.dumps(v2a, indent=2), flush=True)

    print("\n=== V2b: ablate L2 head 1 (control) ===", flush=True)
    logits_v2b, ph_v2b, xr_v2b = run_with_intervention(
        model, T, pi, np.random.default_rng(args.seed + 20),
        n_seqs=args.n_seqs, seq_len=args.seq_len, kill_l2_head=1,
    )
    v2b = behavioral_xor(logits_v2b, ph_v2b, xr_v2b)
    print("  kill head 1:", json.dumps(v2b, indent=2), flush=True)

    # --- V3: zero out L3 attention (phase resolver) ---
    print("\n=== V3: ablate L3 attention (phase resolver) ===", flush=True)
    logits_v3, ph_v3, xr_v3 = run_with_intervention(
        model, T, pi, np.random.default_rng(args.seed + 20),
        n_seqs=args.n_seqs, seq_len=args.seq_len, kill_l3_attn=True,
    )
    v3 = behavioral_xor(logits_v3, ph_v3, xr_v3)
    print(json.dumps(v3, indent=2), flush=True)

    out = {
        "probe_acc_xor": acc_xor, "probe_acc_phase": acc_ph,
        "clean": clean,
        "v1_flip_xor_direction_at_L2_post_mlp": v1,
        "v2a_ablate_L2_head0": v2a,
        "v2b_ablate_L2_head1_control": v2b,
        "v3_ablate_L3_attention": v3,
    }
    with open(os.path.join(args.outdir, "circuit_verify.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.outdir}/circuit_verify.json", flush=True)


if __name__ == "__main__":
    main()
