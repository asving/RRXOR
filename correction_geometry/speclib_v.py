"""Vocab-general spectral reconstruction (extends speclib.py beyond binary)."""
import itertools
import torch
import numpy as np

def make_tests_v(maxlen, vocab):
    return [t for ln in range(1, maxlen + 1)
            for t in itertools.product(range(vocab), repeat=ln)]

@torch.no_grad()
def net_table_v(model, seq, Q0, maxlen, vocab, dev="cuda", chunk=512):
    """path (len maxlen+1) -> (B, len, vocab) conditional dists; dist k conditions on
    prefix + path[:k]. Shared shorter-prefix dists are consistent across paths."""
    B = seq.shape[0]
    prefix = seq[:, :Q0]
    out = {}
    for s in itertools.product(range(vocab), repeat=maxlen + 1):
        x = torch.cat([prefix, torch.tensor(s, device=dev)[None].expand(B, -1)], 1)
        lg = model(x)
        p = torch.softmax(lg[:, Q0 - 1:Q0 + maxlen, :vocab].double(), -1)
        out[s] = p.cpu().clamp_min(1e-9)
    return out

def string_probs_v(tab, TESTS, maxlen, shift=None):
    B = next(iter(tab.values())).shape[0]
    cols = []
    off = 0 if shift is None else 1
    for t in TESTS:
        full = ((shift,) if shift is not None else ()) + t
        pad = full + (0,) * (maxlen + 1 - len(full))
        p = tab[pad]
        lp = torch.zeros(B, dtype=torch.double)
        for k, tok in enumerate(t):                      # conditionals AFTER the shift
            lp = lp + torch.log(p[:, k + off, tok])
        cols.append(torch.exp(lp))
    return torch.stack(cols, 1)

def fit_oom_v(tab, TESTS, maxlen, vocab, k, colnorm=True):
    Yr = string_probs_v(tab, TESTS, maxlen)
    D = (1.0 / Yr.norm(dim=0).clamp_min(1e-12)) if colnorm \
        else torch.ones(Yr.shape[1], dtype=torch.double)
    Y = Yr * D[None]
    U, S, Vh = torch.linalg.svd(Y, full_matrices=False)
    Z = Y @ Vh[:k].T
    p0 = next(iter(tab.values()))[:, 0, :]                # P(sigma | h), shared
    ops, frs = {}, []
    for sig in range(vocab):
        Ys = string_probs_v(tab, TESTS, maxlen, shift=sig) * D[None]
        Zs = Ys @ Vh[:k].T
        T = Zs * p0[:, sig][:, None]
        A = torch.linalg.lstsq(Z, T).solution.T
        frs.append(float(1 - ((T - Z @ A.T) ** 2).sum() / ((T - T.mean(0)) ** 2).sum()))
        ops[sig] = A
    return {"S": S, "ops": ops, "Vh": Vh[:k], "D": D, "Z": Z, "fitr2": min(frs),
            "TESTS": TESTS, "maxlen": maxlen, "vocab": vocab}

def eig_transfer_v(oom):
    T = sum(oom["ops"].values()).numpy()
    e = np.linalg.eigvals(T)
    return e[np.argsort(-np.abs(e))]

def sector_rollout_v(oom, mask_fn, seq, Q0, horizon, trans=0.8):
    """Project onto the eigensector selected by mask_fn(lam) (transients |lam|<0.8
    always kept), roll teacher-forced, return mean CE (joint nats)."""
    vocab = oom["vocab"]
    T = sum(oom["ops"].values()).numpy()
    lam, R = np.linalg.eig(T)
    Linv = np.linalg.inv(R)
    m = np.array([abs(v) < trans or bool(mask_fn(v)) for v in lam])
    P = (R[:, m] @ Linv[m]).real
    A = {s: P @ oom["ops"][s].numpy() @ P for s in range(vocab)}
    idx1 = [oom["TESTS"].index((s,)) for s in range(vocab)]
    Vs = np.stack([(oom["Vh"][:, i] / oom["D"][i]).numpy() for i in idx1], 1)  # k x V
    z = oom["Z"].numpy() @ P.T
    toks = seq.cpu().numpy()
    ces = []
    for t in range(Q0, Q0 + horizon):
        w = z @ Vs                                            # B x V
        tot = w.sum(1, keepdims=True)
        tot = np.where(np.abs(tot) < 1e-9, 1.0, tot)
        p = np.clip(w / tot, 1e-4, 1.0)
        p = p / p.sum(1, keepdims=True)
        x = toks[:, t]
        ces.append(float(-np.log(p[np.arange(len(x)), x]).mean()))
        psig = p[np.arange(len(x)), x]
        An = np.stack([A[s] for s in x])
        z = np.einsum("bij,bj->bi", An, z) / psig[:, None]
        zt = (z @ Vs).sum(1)
        z = z / np.where(np.abs(zt) < 0.05, 1.0, zt)[:, None]
    return float(np.mean(ces)), int(m.sum())
