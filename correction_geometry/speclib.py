"""Spectral (Hankel/OOM) reconstruction library — no side effects.

Pipeline: H[h, s] = P_f(s | h) filled exactly from a model's conditionals by
teacher-forcing test suffixes; SVD -> state vectors z(h) (the belief state up to an
invertible linear map); shifted-state regression -> observable operators A_sigma with
A_sigma z(h) = P(sigma|h) z(h sigma); the reconstructed machine runs autoregressively:
p(sigma) from z via the length-1 test coordinates, z <- A_sigma z / p(sigma).
"""
import torch
import numpy as np

def make_tests(maxlen=4):
    return [tuple((b >> i) & 1 for i in range(ln - 1, -1, -1))
            for ln in range(1, maxlen + 1) for b in range(2 ** ln)]

def make_sufs(maxlen=4):
    n = maxlen + 1
    return [[(b >> i) & 1 for i in range(n - 1, -1, -1)] for b in range(2 ** n)]

@torch.no_grad()
def net_table(model, seq, Q0, maxlen=4, dev="cuda"):
    """(partial-suffix tuple) -> p(next=1 | prefix + partial) for all |partial| <= maxlen."""
    B = seq.shape[0]
    prefix = seq[:, :Q0]
    out = {}
    for s in make_sufs(maxlen):
        x = torch.cat([prefix, torch.tensor(s, device=dev)[None].expand(B, -1)], 1)
        lg = model(x)
        p1 = torch.softmax(lg[:, :, :2].double(), -1)[:, :, 1]
        for k in range(maxlen + 1):
            key = tuple(s[:k])
            if key not in out:
                out[key] = p1[:, Q0 - 1 + k].cpu().clamp(1e-6, 1 - 1e-6)
    return out

def string_probs(tab, TESTS, shift=None):
    B = tab[()].shape[0]
    cols = []
    for t in TESTS:
        pre = () if shift is None else (shift,)
        lp = torch.zeros(B, dtype=torch.double)
        for k, tok in enumerate(t):
            c = tab[pre + t[:k]]
            lp = lp + torch.log(c if tok == 1 else 1 - c)
        cols.append(torch.exp(lp))
    return torch.stack(cols, 1)

def fit_oom(tab, TESTS, k, colnorm=False):
    """Returns dict with S (svs), ops {0,1}, Vh, D (col scales), fit R2."""
    Yr = string_probs(tab, TESTS)
    D = (1.0 / Yr.norm(dim=0).clamp_min(1e-12)) if colnorm \
        else torch.ones(Yr.shape[1], dtype=torch.double)
    Y = Yr * D[None]
    U, S, Vh = torch.linalg.svd(Y, full_matrices=False)
    Z = Y @ Vh[:k].T
    ops, frs = {}, []
    psig1 = tab[()]                                   # P(next=1 | h)
    for sig in (0, 1):
        Ys = string_probs(tab, TESTS, shift=sig) * D[None]
        Zs = Ys @ Vh[:k].T
        psig = psig1 if sig == 1 else 1 - psig1
        T = Zs * psig[:, None]
        A = torch.linalg.lstsq(Z, T).solution.T
        frs.append(float(1 - ((T - Z @ A.T) ** 2).sum() / ((T - T.mean(0)) ** 2).sum()))
        ops[sig] = A
    return {"S": S, "ops": ops, "Vh": Vh[:k], "D": D, "Z": Z, "fitr2": min(frs),
            "TESTS": TESTS}

def oom_p1(oom, z):
    """p(next=1) from state via the length-1 test coordinates."""
    i1 = oom["TESTS"].index((1,))
    return (z @ oom["Vh"][:, i1]) / oom["D"][i1]

def oom_step(oom, z, tok):
    p1 = oom_p1(oom, z).clamp(1e-4, 1 - 1e-4)
    psig = torch.where(tok.bool(), p1, 1 - p1)
    A = torch.where(tok.bool()[:, None, None], oom["ops"][1][None], oom["ops"][0][None])
    z2 = torch.einsum("bij,bj->bi", A, z) / psig[:, None]
    # renormalize onto the probability manifold: p(0)+p(1) = 1
    i0, i1 = oom["TESTS"].index((0,)), oom["TESTS"].index((1,))
    tot = (z2 @ oom["Vh"][:, i0]) / oom["D"][i0] + (z2 @ oom["Vh"][:, i1]) / oom["D"][i1]
    return p1, z2 / tot.clamp(0.2, 5.0)[:, None]

def eig_transfer(oom):
    e = np.linalg.eigvals((oom["ops"][0] + oom["ops"][1]).numpy())
    return e[np.argsort(-np.abs(e))]

# ---- exact reference filters ----
def rrxor3_p1_run(seq):
    """Exact offset-elimination filter: p(next=1) at every position (B, L-1)."""
    B, L = seq.shape
    sc = seq.cpu()
    v = torch.zeros(B, L)
    v[:, 2:] = (sc[:, 2:] != (sc[:, 1:-1] ^ sc[:, :-2])).double()
    tt = torch.arange(L)
    post = torch.zeros(B, L, 3, dtype=torch.double)
    for o in range(3):
        det = ((o + tt) % 3 == 2) & (tt >= 2)
        killed = ((v.bool() & det[None]).cumsum(1) > 0).double()
        post[:, :, o] = (1 - killed) * torch.pow(2.0, det[None].double().expand(B, -1).cumsum(1))
    post = post / post.sum(-1, keepdim=True).clamp_min(1e-30)
    p1 = torch.zeros(B, L - 1, dtype=torch.double)
    cand = torch.zeros(B, L - 1, dtype=torch.double)
    cand[:, 1:] = (sc[:, 1:-1] ^ sc[:, :-2]).double()   # cand[:, q] = x_q ^ x_{q-1}
    for o in range(3):
        isdet = (((o + tt[1:]) % 3) == 2).double()
        p1 += post[:, :-1, o] * (isdet[None] * cand + (1 - isdet[None]) * 0.5)
    return p1

def fx_p1_run(seq, eps):
    """Exact 5-state leaky filter: p(next=1) at every position (B, L)."""
    B, L = seq.shape
    sc = seq.cpu()
    pi = torch.full((B, 5), 1 / 5, dtype=torch.double)
    P1 = torch.zeros(B, L, dtype=torch.double)
    for t in range(L):
        P1[:, t] = pi[:, 4] + (pi[:, 0] + pi[:, 1] + pi[:, 2]) / 2
        tok = sc[:, t]
        lik = torch.stack([torch.full((B,), .5, dtype=torch.double)] * 3 +
                          [(tok == 0).double(), (tok == 1).double()], 1)
        w = pi * lik
        w = w / w.sum(1, keepdim=True).clamp_min(1e-30)
        nxt = torch.zeros_like(w)
        t0 = (tok == 0).double(); t1 = (tok == 1).double()
        nxt[:, 1] += w[:, 0] * t0; nxt[:, 2] += w[:, 0] * t1
        nxt[:, 3] += w[:, 1] * t0 + w[:, 2] * t1
        nxt[:, 4] += w[:, 1] * t1 + w[:, 2] * t0
        nxt[:, 0] += w[:, 3] + w[:, 4]
        pi = (1 - eps) * nxt + eps / 5
    return P1

def fx_table(seq, Q0, eps, maxlen=4):
    """Exact leaky-filter conditional table along all suffix paths."""
    B, L = seq.shape
    sc = seq.cpu()
    def roll_path(tokens):
        full = torch.cat([sc[:, :Q0]] + [torch.full((B, 1), t).long() for t in tokens], 1)
        P1 = fx_p1_run(full, eps)
        return P1
    out = {}
    for s in make_sufs(maxlen):
        P1 = roll_path(s)
        for k in range(maxlen + 1):
            key = tuple(s[:k])
            if key not in out:
                out[key] = P1[:, Q0 + k].clamp(1e-9, 1 - 1e-9)
    return out

def rrxor3_table(seq, Q0, maxlen=4):
    """Exact elimination-filter conditional table along all suffix paths."""
    B = seq.shape[0]
    sc = seq.cpu()
    out = {}
    for s in make_sufs(maxlen):
        full = torch.cat([sc[:, :Q0]] + [torch.full((B, 1), t).long() for t in s], 1)
        P1 = rrxor3_p1_run(full)
        for k in range(maxlen + 1):
            key = tuple(s[:k])
            if key not in out:
                out[key] = P1[:, Q0 - 1 + k].clamp(1e-9, 1 - 1e-9)
    return out
