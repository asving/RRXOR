"""Jasmina's questions: (1) KL-per-position — is the reconstructed machine at least as
good a model of the NET as the true epsilon-machine? (2) leaky RRXOR — prediction: the
teleport pulls the clock inside the unit circle to radius exactly 1-eps. (3) the
correction-ratio Hankel: spectrally factor c = P_post/P_pre itself.
"""
import glob, json, os
import torch
import numpy as np
from train import GPT
import speclib as S

base = os.path.dirname(os.path.abspath(__file__))
dev = "cuda"
Q0, B, K = 60, 1024, 5
TESTS = S.make_tests(4)

def load_net(run, step, nl=4):
    ev = torch.load(os.path.join(base, "runs", run, "evalset.pt"))
    seq = ev["eseq"][:B].to(dev)
    m = GPT(L=seq.shape[1], d=128, nl=nl).to(dev)
    ck = sorted(glob.glob(os.path.join(base, "runs", run, "ckpts", "*.pt")))
    ss = np.array([int(os.path.basename(c)[4:10]) for c in ck])
    m.load_state_dict(torch.load(ck[int(np.argmin(abs(ss - step)))], map_location=dev))
    m.eval()
    return m, seq

def kl(p, q):
    p = p.clamp(1e-6, 1 - 1e-6); q = q.clamp(1e-6, 1 - 1e-6)
    return p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))

# ================= Q1: KL per context position (rrxor3 + fx0.03) =================
print("=== Q1: KL(net || model) per position, OOM vs true epsilon-machine ===")
results_kl = {}
for run, nl, mx, true_p1_fn in [("seed0", 4, 4, lambda sq: S.rrxor3_p1_run(sq)),
                                ("fx0.03_seed0", 4, 5, lambda sq: S.fx_p1_run(sq, 0.03)[:, 1:])]:
    m, seq = load_net(run, 8000, nl)
    tab = S.net_table(m, seq, Q0, maxlen=mx, dev=dev)
    Tst = S.make_tests(mx)
    oom = S.fit_oom(tab, Tst, K, colnorm=(mx > 4))
    print(f"  [{run} oom fit R2 {oom['fitr2']:.5f}]")
    with torch.no_grad():
        lg = m(seq)
    p_net = torch.softmax(lg[:, :, :2].double(), -1)[:, :, 1].cpu()  # col t-1 -> token t
    p_true = true_p1_fn(seq)                                          # col t-1 -> token t
    z = oom["Z"].clone()
    kls_oom, kls_true = [], []
    for t in range(Q0, min(Q0 + 32, seq.shape[1] - 1)):
        p_o, z = S.oom_step(oom, z, seq[:, t].cpu())
        kls_oom.append(float(kl(p_net[:, t - 1], p_o).mean()))
        kls_true.append(float(kl(p_net[:, t - 1], p_true[:, t - 1]).mean()))
    results_kl[run] = {"oom": kls_oom, "true": kls_true}
    print(f"{run:14s}: mean KL(net||OOM) {np.mean(kls_oom):.6f}   "
          f"mean KL(net||true-machine) {np.mean(kls_true):.6f}   "
          f"(rollout {len(kls_oom)} positions)")

# ================= Q2: leaky RRXOR — the clock's radius =================
print("\n=== Q2: leaky RRXOR — clock radius vs 1 - eps ===")
fx_eigs = {}
for run, eps in [("fx0.01_seed0", 0.01), ("fx0.03_seed0", 0.03), ("fx0.2_seed0", 0.2)]:
    m, seq = load_net(run, 8000, 4)
    tab = S.net_table(m, seq, Q0, dev=dev)
    oom = S.fit_oom(tab, TESTS, K)
    e = S.eig_transfer(oom)
    tab_t = S.fx_table(seq, Q0, eps)
    oom_t = S.fit_oom(tab_t, TESTS, K)
    et = S.eig_transfer(oom_t)
    def clock(ee):
        cl = [v for v in ee if abs(v.imag) > 0.02]
        return min(cl, key=lambda v: abs(abs(np.angle(v)) - 2 * np.pi / 3)) if cl else None
    cn, ct = clock(e), clock(et)
    fx_eigs[run] = {"eps": eps, "net": [[v.real, v.imag] for v in e],
                    "true": [[v.real, v.imag] for v in et],
                    "sv": [float(x) for x in (oom["S"] / oom["S"][0])[:8]]}
    print(f"eps={eps:4.2f}  predicted radius {1-eps:.3f}   "
          f"net clock {abs(cn):.3f}∠{abs(np.angle(cn)):.3f}   "
          f"true-machine clock {abs(ct):.3f}∠{abs(np.angle(ct)):.3f}")

# ================= Q3: the correction-ratio Hankel (rrxor3 drop 2) =================
print("\n=== Q3: spectral factorization of the correction c = P_post/P_pre itself ===")
m_pre, seq3 = load_net("seed0", 1000, 4)
tab_pre = S.net_table(m_pre, seq3, Q0, dev=dev)
m_post, _ = load_net("seed0", 3000, 4)
tab_post = S.net_table(m_post, seq3, Q0, dev=dev)

def fit_ratio_wfa(tab_p, tab_q, k):
    Yp, Yq = S.string_probs(tab_p, TESTS), S.string_probs(tab_q, TESTS)
    Yc = Yp / Yq
    D = 1.0 / Yc.norm(dim=0).clamp_min(1e-12)
    Y = Yc * D[None]
    U, Sv, Vh = torch.linalg.svd(Y, full_matrices=False)
    Z = Y @ Vh[:k].T
    ops, frs = {}, []
    for sig in (0, 1):
        Yps = S.string_probs(tab_p, TESTS, shift=sig)
        Yqs = S.string_probs(tab_q, TESTS, shift=sig)
        Zs = (Yps / Yqs * D[None]) @ Vh[:k].T
        rp = tab_p[()] if sig == 1 else 1 - tab_p[()]
        rq = tab_q[()] if sig == 1 else 1 - tab_q[()]
        w = (rp / rq)
        T = Zs * w[:, None]
        A = torch.linalg.lstsq(Z, T).solution.T
        frs.append(float(1 - ((T - Z @ A.T) ** 2).sum() / ((T - T.mean(0)) ** 2).sum()))
        ops[sig] = A
    return Sv, ops, min(frs)

def bet_table(seq, q):
    """Idealized stage-1 policy: p(next = cand) = q, cand = xor of last two tokens."""
    sc = seq.cpu()
    out = {}
    for suf in S.make_sufs(4):
        for k in range(5):
            key = tuple(suf[:k])
            if key in out:
                continue
            if k == 0:
                cand = (sc[:, Q0 - 1] ^ sc[:, Q0 - 2]).double()
            elif k == 1:
                cand = (key[-1] ^ sc[:, Q0 - 1]).double()
            else:
                cand = torch.full((sc.shape[0],), float(key[-1] ^ key[-2]), dtype=torch.double)
            out[key] = (q * cand + (1 - q) * (1 - cand)).clamp(1e-6, 1 - 1e-6)
    return out

pc = torch.where((seq3.cpu()[:, Q0 - 1] ^ seq3.cpu()[:, Q0 - 2]).bool(),
                 tab_pre[()], 1 - tab_pre[()])
qhat = float(pc.mean())
print(f"[idealized bet: q fitted from pre-net = {qhat:.3f}]")
tab_bet = bet_table(seq3, qhat)
Sv, ops_c, fr = fit_ratio_wfa(tab_post, tab_bet, k=8)
print("-- correction vs IDEALIZED bet policy --")
print("correction-Hankel sv/sv1:", np.round((Sv / Sv[0])[:12].numpy(), 4), f" fitR2 {fr:.4f}")
Sv2, ops_c2, fr2 = fit_ratio_wfa(tab_post, tab_pre, k=8)
print("-- correction vs actual pre-net @1000 (for contrast) --")
print("correction-Hankel sv/sv1:", np.round((Sv2 / Sv2[0])[:12].numpy(), 4), f" fitR2 {fr2:.4f}")
ec = np.linalg.eigvals((ops_c[0] + ops_c[1]).numpy())
ec = ec[np.argsort(-np.abs(ec))]
print("correction-WFA transfer eigenvalues: " +
      "  ".join(f"{v.real:.3f}" if abs(v.imag) < 0.02 else
                f"{abs(v):.3f}∠{abs(np.angle(v)):.3f}" for v in ec))

json.dump({"kl": results_kl, "fx": fx_eigs,
           "corr": {"sv": [float(x) for x in (Sv / Sv[0])[:12]],
                    "eigs": [[float(v.real), float(v.imag)] for v in ec]}},
          open(os.path.join(base, "hankel_qa.json"), "w"))
print("saved hankel_qa.json")
