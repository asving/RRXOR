"""
RRXOR belief-geometry experiment, naive vs paired tokenization.

RRXOR: 3-phase cycle. Phase 0 (R1) emits random bit, phase 1 (R2) emits random bit,
phase 2 (X) emits XOR of previous two bits. Minimal epsilon-machine has 5 causal states:
  0 = P0        (about to emit R1)
  1 = P1_0      (just emitted R1=0; about to emit R2)
  2 = P1_1      (just emitted R1=1; about to emit R2)
  3 = X_0       (will emit XOR=0)
  4 = X_1       (will emit XOR=1)
Stationary: (1/3, 1/6, 1/6, 1/6, 1/6).

Paired tokenization (vocab=4) has period 3 in pair-time because underlying period 3
vs pair period 2 are coprime — so the pair-token distribution cycles through three
statistical regimes. This creates a phase-carrying representation problem absent in
the naive view.

We train a small transformer, then probe residual stream for:
  (a) belief over 5 causal states (linear regression, R^2)
  (b) phase mod 3 (one-hot, classification via linear probe)
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# RRXOR HMM
# ---------------------------------------------------------------------------

def rrxor_tensors():
    """5-state minimal epsilon-machine. Returns T[i, o, j]."""
    S, O = 5, 2
    T = np.zeros((S, O, S))
    # P0 -> P1_{emit}
    T[0, 0, 1] = 0.5
    T[0, 1, 2] = 0.5
    # P1_0 -> X_{bit}: emit 0 => X_0 (XOR 0⊕0), emit 1 => X_1 (XOR 0⊕1)
    T[1, 0, 3] = 0.5
    T[1, 1, 4] = 0.5
    # P1_1 -> X_{1⊕bit}: emit 0 => X_1, emit 1 => X_0
    T[2, 0, 4] = 0.5
    T[2, 1, 3] = 0.5
    # X_0 emits 0 → P0; X_1 emits 1 → P0
    T[3, 0, 0] = 1.0
    T[4, 1, 0] = 1.0
    return T


# Map each causal state to its phase (0=P0, 1=P1_*, 2=X_*)
STATE_PHASE = np.array([0, 1, 1, 2, 2])


def stationary(T):
    P = T.sum(axis=1)  # P[s, s'] = sum over emissions
    evals, evecs = np.linalg.eig(P.T)
    i = np.argmin(np.abs(evals - 1))
    pi = np.real(evecs[:, i])
    pi = pi / pi.sum()
    return pi


def sample_sequence(T, pi, length, rng):
    S = T.shape[0]
    OBS = T.shape[1]
    trans = T.reshape(S, OBS * S)
    out = np.empty(length, dtype=np.int64)
    states = np.empty(length, dtype=np.int64)
    s = rng.choice(S, p=pi)
    for t in range(length):
        idx = rng.choice(OBS * S, p=trans[s])
        o, s_next = idx // S, idx % S
        out[t] = o
        states[t] = s_next  # state AFTER emission (at time t+1)
        s = s_next
    return out, states


def beliefs_from_emissions(T, pi, emissions):
    L = len(emissions)
    S = T.shape[0]
    b = np.empty((L + 1, S))
    b[0] = pi
    for t, o in enumerate(emissions):
        unnorm = b[t] @ T[:, o, :]
        Z = unnorm.sum()
        if Z < 1e-30:
            b[t + 1] = b[t]
        else:
            b[t + 1] = unnorm / Z
    return b


# ---------------------------------------------------------------------------
# Transformer (identical to Mess3 script)
# ---------------------------------------------------------------------------

class MiniTransformer(nn.Module):
    def __init__(self, vocab, d_model=64, n_heads=2, n_layers=4, max_len=256, dropout=0.0):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
                dropout=dropout, batch_first=True, activation="gelu", norm_first=True,
            )
            for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab)
        self.max_len = max_len

    def forward(self, x, return_resid=False):
        B, L = x.shape
        pos_idx = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        h = self.tok(x) + self.pos(pos_idx)
        mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
        resids = []
        for blk in self.blocks:
            h = blk(h, src_mask=mask)
            if return_resid:
                resids.append(h.detach().clone())
        h = self.ln(h)
        logits = self.head(h)
        if return_resid:
            return logits, resids, h.detach().clone()
        return logits


def train_model(model, data, steps, batch_size, seq_len, lr, device, log_every=200):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    N = data.shape[0]
    losses = []
    for step in range(steps):
        idx = np.random.randint(0, N - seq_len - 1, size=batch_size)
        batch = np.stack([data[i:i + seq_len + 1] for i in idx])
        batch = torch.from_numpy(batch).to(device)
        x, y = batch[:, :-1], batch[:, 1:]
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        if (step + 1) % log_every == 0:
            print(f"  step {step+1:5d}  loss {np.mean(losses[-log_every:]):.4f}", flush=True)
    return losses


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------

def collect_activations_and_labels(model, T, pi, mode, n_seqs, seq_len, device, rng):
    model.eval()
    all_resids_per_layer = None
    all_final = []
    all_beliefs = []
    all_phases = []   # underlying phase (0=R1, 1=R2, 2=X) of the NEXT emission site
    with torch.no_grad():
        for _ in range(n_seqs):
            if mode == "naive":
                ems, _ = sample_sequence(T, pi, seq_len, rng)
                b = beliefs_from_emissions(T, pi, ems)  # (seq_len+1, 5)
                tokens = ems
                beliefs_at_pos = b[1:]
                # phase at position t: phase mod 3 of the state we're in after t emissions
                # which equals (initial_phase + t) mod 3. Instead, derive from the belief-support directly.
                phase_at_pos = np.array([STATE_PHASE[np.argmax(bt)] for bt in beliefs_at_pos])
            else:
                underlying, _ = sample_sequence(T, pi, 2 * seq_len, rng)
                b = beliefs_from_emissions(T, pi, underlying)
                tokens = underlying[0::2] * 2 + underlying[1::2]   # pair token id ∈ [0, 4)
                beliefs_at_pos = b[2::2]
                phase_at_pos = np.array([STATE_PHASE[np.argmax(bt)] for bt in beliefs_at_pos])
            x = torch.from_numpy(tokens).long().unsqueeze(0).to(device)
            logits, resids, final = model(x, return_resid=True)
            if all_resids_per_layer is None:
                all_resids_per_layer = [[] for _ in resids]
            for k, r in enumerate(resids):
                all_resids_per_layer[k].append(r.squeeze(0).cpu().numpy())
            all_final.append(final.squeeze(0).cpu().numpy())
            all_beliefs.append(beliefs_at_pos)
            all_phases.append(phase_at_pos)
    acts = {}
    for k, lst in enumerate(all_resids_per_layer):
        acts[f"layer{k}"] = np.concatenate(lst, axis=0)
    acts["final"] = np.concatenate(all_final, axis=0)
    beliefs = np.concatenate(all_beliefs, axis=0)
    phases = np.concatenate(all_phases, axis=0)
    return acts, beliefs, phases


def linear_probe(X, Y, train_frac=0.8):
    N = X.shape[0]
    idx = np.random.permutation(N)
    ntr = int(train_frac * N)
    tr, te = idx[:ntr], idx[ntr:]
    Xtr = np.concatenate([X[tr], np.ones((len(tr), 1))], axis=1)
    W, *_ = np.linalg.lstsq(Xtr, Y[tr], rcond=None)
    Xall = np.concatenate([X, np.ones((N, 1))], axis=1)
    Yhat = Xall @ W
    Yte = Y[te]
    Yhat_te = Yhat[te]
    ss_res = ((Yte - Yhat_te) ** 2).sum(axis=0)
    ss_tot = ((Yte - Yte.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1.0 - ss_res / np.clip(ss_tot, 1e-12, None)
    return {"r2_per_output": r2, "r2_overall": float(r2.mean()), "predictions": Yhat, "W": W}


def phase_classify(X, phases, train_frac=0.8):
    """One-vs-rest linear probe accuracy for phase ∈ {0,1,2}."""
    N = X.shape[0]
    idx = np.random.permutation(N)
    ntr = int(train_frac * N)
    tr, te = idx[:ntr], idx[ntr:]
    Y_oh = np.eye(3)[phases]
    Xtr = np.concatenate([X[tr], np.ones((len(tr), 1))], axis=1)
    W, *_ = np.linalg.lstsq(Xtr, Y_oh[tr], rcond=None)
    Xall = np.concatenate([X, np.ones((N, 1))], axis=1)
    Yhat = Xall @ W
    pred = Yhat[te].argmax(axis=1)
    true = phases[te]
    acc = float((pred == true).mean())
    return {"acc": acc, "W": W}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["naive", "paired"], required=True)
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--n-heads", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--train-seqs", type=int, default=400)
    ap.add_argument("--probe-seqs", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[RRXOR {args.mode}] device = {device}", flush=True)

    T = rrxor_tensors()
    pi = stationary(T)
    print(f"stationary = {pi}", flush=True)

    if args.mode == "naive":
        vocab = 2
        buf, _ = sample_sequence(T, pi, args.train_seqs * args.seq_len, rng)
        ctx = args.seq_len
    else:
        vocab = 4
        under, _ = sample_sequence(T, pi, args.train_seqs * args.seq_len * 2, rng)
        buf = under[0::2] * 2 + under[1::2]
        ctx = args.seq_len

    print(f"vocab = {vocab}, train-buffer tokens = {len(buf)}", flush=True)

    model = MiniTransformer(
        vocab=vocab, d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, max_len=ctx + 2,
    ).to(device)
    print(f"params = {sum(p.numel() for p in model.parameters())}", flush=True)

    losses = train_model(
        model, buf, steps=args.steps, batch_size=args.batch_size,
        seq_len=ctx, lr=args.lr, device=device,
    )
    np.save(os.path.join(args.outdir, "losses.npy"), np.array(losses))

    # Eval CE
    model.eval()
    with torch.no_grad():
        if args.mode == "naive":
            eval_under, _ = sample_sequence(T, pi, 20000, rng)
            eval_tok = eval_under
        else:
            eval_under, _ = sample_sequence(T, pi, 40000, rng)
            eval_tok = eval_under[0::2] * 2 + eval_under[1::2]
        eval_tok_t = torch.from_numpy(eval_tok).long().to(device)
        n_chunks = len(eval_tok_t) // (ctx + 1)
        eval_loss = 0.0
        cnt = 0
        for i in range(n_chunks):
            chunk = eval_tok_t[i * (ctx + 1):(i + 1) * (ctx + 1)].unsqueeze(0)
            x, y = chunk[:, :-1], chunk[:, 1:]
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))
            eval_loss += loss.item() * y.numel()
            cnt += y.numel()
        eval_loss /= cnt
        print(f"eval CE = {eval_loss:.4f}  ppl = {np.exp(eval_loss):.4f}", flush=True)

    # Probing
    acts, beliefs, phases = collect_activations_and_labels(
        model, T, pi, args.mode, n_seqs=args.probe_seqs, seq_len=args.seq_len,
        device=device, rng=rng,
    )
    probe_results = {}
    phase_results = {}
    for k, X in acts.items():
        res = linear_probe(X, beliefs)
        probe_results[k] = res
        phcl = phase_classify(X, phases)
        phase_results[k] = phcl
        print(f"  probe {k:>7}: belief R^2 = {res['r2_overall']:.4f}  phase acc = {phcl['acc']:.4f}", flush=True)

    save_payload = {
        "beliefs_true": beliefs,
        "phases_true": phases,
        "eval_ce": eval_loss,
        "losses": np.array(losses),
        "pi": pi,
        "T": T,
        "args": vars(args),
    }
    for k, res in probe_results.items():
        save_payload[f"probe_{k}_r2"] = res["r2_per_output"]
        save_payload[f"probe_{k}_pred"] = res["predictions"]
        save_payload[f"phase_{k}_acc"] = phase_results[k]["acc"]
    np.savez(os.path.join(args.outdir, "results.npz"), **save_payload)
    torch.save(model.state_dict(), os.path.join(args.outdir, "model.pt"))
    print(f"saved to {args.outdir}", flush=True)


if __name__ == "__main__":
    main()
