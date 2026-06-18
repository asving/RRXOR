"""
Diagnostic probes on the already-trained naive_ctx / paired_ctx RRXOR models:
  (1) nonlinear (MLP) probe: activations -> belief.  Does naive encode belief nonlinearly?
  (2) linear probe: activations -> last-2-bits one-hot.  Does the model carry raw history?
  (3) linear probe: activations -> model's own logits.  Logit-space shortcut?

Uses the saved model weights; no retraining of the transformer.
"""

import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rrxor_belief_experiment import (
    rrxor_tensors, stationary, sample_sequence, beliefs_from_emissions,
    MiniTransformer, STATE_PHASE,
)


def collect_features(model, T, pi, mode, n_seqs, seq_len, device, rng):
    """Gather activations + three kinds of labels at every position."""
    model.eval()
    all_resids_per_layer = None
    all_final = []
    all_beliefs = []
    all_last2 = []       # label: (bit_t, bit_{t-1}) as 4-way id, -1 for t<1
    all_logits = []      # the model's own next-token log-softmax at each position
    with torch.no_grad():
        for _ in range(n_seqs):
            if mode == "naive":
                ems, _ = sample_sequence(T, pi, seq_len, rng)
                b = beliefs_from_emissions(T, pi, ems)
                tokens = ems
                beliefs_at_pos = b[1:]
                # last-2 underlying bits at position t = (ems[t], ems[t-1])
                last2 = np.full(seq_len, -1, dtype=np.int64)
                last2[1:] = ems[1:] * 2 + ems[:-1]
            else:
                underlying, _ = sample_sequence(T, pi, 2 * seq_len, rng)
                b = beliefs_from_emissions(T, pi, underlying)
                tokens = underlying[0::2] * 2 + underlying[1::2]
                beliefs_at_pos = b[2::2]
                # last-2 underlying bits at pair-pos k = (e_{2k+2-1}, e_{2k+2-2})
                #  = (underlying[2k+1], underlying[2k])  [0-indexed]
                last2 = underlying[1::2] * 2 + underlying[0::2]
            x = torch.from_numpy(tokens).long().unsqueeze(0).to(device)
            logits, resids, final = model(x, return_resid=True)
            logprobs = F.log_softmax(logits, dim=-1).squeeze(0).cpu().numpy()  # (L, V)
            if all_resids_per_layer is None:
                all_resids_per_layer = [[] for _ in resids]
            for k, r in enumerate(resids):
                all_resids_per_layer[k].append(r.squeeze(0).cpu().numpy())
            all_final.append(final.squeeze(0).cpu().numpy())
            all_beliefs.append(beliefs_at_pos)
            all_last2.append(last2)
            all_logits.append(logprobs)
    acts = {f"layer{k}": np.concatenate(v, axis=0) for k, v in enumerate(all_resids_per_layer)}
    acts["final"] = np.concatenate(all_final, axis=0)
    return {
        "acts": acts,
        "belief": np.concatenate(all_beliefs, axis=0),
        "last2": np.concatenate(all_last2, axis=0),
        "logits": np.concatenate(all_logits, axis=0),  # log-softmax
    }


def linear_probe(X, Y, train_frac=0.8, mask=None):
    N = X.shape[0]
    idx = np.arange(N)
    if mask is not None:
        idx = idx[mask]
    rng = np.random.default_rng(0)
    idx = rng.permutation(idx)
    ntr = int(train_frac * len(idx))
    tr, te = idx[:ntr], idx[ntr:]
    Xtr = np.concatenate([X[tr], np.ones((len(tr), 1))], axis=1)
    W, *_ = np.linalg.lstsq(Xtr, Y[tr], rcond=None)
    Xall = np.concatenate([X, np.ones((N, 1))], axis=1)
    Yhat = Xall @ W
    # R^2 on held-out
    Yte = Y[te]
    Yhat_te = Yhat[te]
    if Yte.ndim == 1:
        Yte = Yte[:, None]
        Yhat_te = Yhat_te[:, None]
    ss_res = ((Yte - Yhat_te) ** 2).sum(axis=0)
    ss_tot = ((Yte - Yte.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1.0 - ss_res / np.clip(ss_tot, 1e-12, None)
    return float(r2.mean())


def linear_probe_classify(X, y, n_classes, train_frac=0.8, mask=None):
    """Linear probe for categorical y via one-vs-rest LSQ. Returns accuracy."""
    N = X.shape[0]
    idx = np.arange(N)
    if mask is not None:
        idx = idx[mask]
    rng = np.random.default_rng(0)
    idx = rng.permutation(idx)
    ntr = int(train_frac * len(idx))
    tr, te = idx[:ntr], idx[ntr:]
    Y_oh = np.eye(n_classes)[y]
    Xtr = np.concatenate([X[tr], np.ones((len(tr), 1))], axis=1)
    W, *_ = np.linalg.lstsq(Xtr, Y_oh[tr], rcond=None)
    Xall = np.concatenate([X, np.ones((N, 1))], axis=1)
    pred = (Xall @ W)[te].argmax(axis=1)
    return float((pred == y[te]).mean())


class MLPProbe(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x):
        return self.net(x)


def nonlinear_probe(X, Y, hidden=128, steps=600, lr=3e-3, batch=1024, train_frac=0.8, device="cuda"):
    N, D = X.shape
    _, O = Y.shape
    rng = np.random.default_rng(0)
    idx = rng.permutation(N)
    ntr = int(train_frac * N)
    tr, te = idx[:ntr], idx[ntr:]
    Xt = torch.from_numpy(X).float().to(device)
    Yt = torch.from_numpy(Y).float().to(device)
    # normalize X for stable training
    mu, sd = Xt[tr].mean(0, keepdim=True), Xt[tr].std(0, keepdim=True).clamp_min(1e-6)
    Xt = (Xt - mu) / sd
    model = MLPProbe(D, hidden, O).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(steps):
        b = torch.from_numpy(rng.choice(tr, size=batch, replace=True)).to(device)
        xb, yb = Xt[b], Yt[b]
        pred = model(xb)
        loss = F.mse_loss(pred, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        Yhat = model(Xt).cpu().numpy()
    Yte = Y[te]
    Yhat_te = Yhat[te]
    ss_res = ((Yte - Yhat_te) ** 2).sum(axis=0)
    ss_tot = ((Yte - Yte.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1.0 - ss_res / np.clip(ss_tot, 1e-12, None)
    return float(r2.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="results/rrxor_belief/naive_ctx",
                    help="Dir with model.pt + args")
    ap.add_argument("--outdir", default="results/rrxor_belief/naive_ctx/circuit")
    ap.add_argument("--n-seqs", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    # Load training args from the source
    prior = np.load(os.path.join(args.source, "results.npz"), allow_pickle=True)
    train_args = prior["args"].item()
    mode = train_args["mode"]
    vocab = 2 if mode == "naive" else 4
    seq_len = train_args["seq_len"]
    d_model = train_args["d_model"]
    n_layers = train_args["n_layers"]
    n_heads = train_args["n_heads"]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[probe {mode} seq_len={seq_len}] device={device}", flush=True)

    model = MiniTransformer(
        vocab=vocab, d_model=d_model, n_heads=n_heads, n_layers=n_layers, max_len=seq_len + 2,
    ).to(device)
    model.load_state_dict(torch.load(os.path.join(args.source, "model.pt"), map_location=device))

    T = rrxor_tensors()
    pi = stationary(T)
    rng = np.random.default_rng(1)

    feats = collect_features(model, T, pi, mode, n_seqs=args.n_seqs, seq_len=seq_len,
                             device=device, rng=rng)
    print(f"N total positions = {len(feats['belief'])}", flush=True)

    layer_names = [f"layer{k}" for k in range(n_layers)] + ["final"]
    results = {"mode": mode, "seq_len": seq_len}
    for k in layer_names:
        X = feats["acts"][k]
        print(f"-- {k} (D={X.shape[1]}) --", flush=True)
        # 1. Linear belief (baseline to confirm consistency)
        r2_belief_lin = linear_probe(X, feats["belief"])
        # 2. Nonlinear belief (MLP)
        r2_belief_nl = nonlinear_probe(X, feats["belief"], device=device)
        # 3. Last-2-bits linear classification
        mask = feats["last2"] >= 0
        acc_last2 = linear_probe_classify(X, feats["last2"], n_classes=4, mask=mask)
        # 4. Logits linear regression
        r2_logits = linear_probe(X, feats["logits"])
        results[k] = {
            "belief_linear_R2": r2_belief_lin,
            "belief_nonlinear_R2": r2_belief_nl,
            "last2_linear_acc": acc_last2,
            "logits_linear_R2": r2_logits,
        }
        print(f"  belief_lin={r2_belief_lin:.4f}  belief_nl={r2_belief_nl:.4f}  "
              f"last2_acc={acc_last2:.4f}  logits_lin={r2_logits:.4f}", flush=True)

    with open(os.path.join(args.outdir, "shortcut_probes.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved {args.outdir}/shortcut_probes.json", flush=True)


if __name__ == "__main__":
    main()
