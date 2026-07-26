# RESULTS 14 (pilot) — the within-G complexity grading, made quantitative

Question (asvin): what grades circuits WITHIN the one-block reachable class —
why does lookup beat XOR beat 3-parity, quantitatively, for a fixed architecture?

## Candidates considered
(1) input degree — right ordering, architecture-blind; (2) multiplicative chain
length (saddle-to-saddle) — mechanistic but gauge-dependent counting;
(3) COMMITTED: the correction's gradient signal in the network's tangent space,

    G_disc(f | theta) = || E_data[ grad_theta margin_f ] ||^2   (vs its noise
    E||grad||^2; SNR = drift-to-diffusion of the route's DISCRIMINATION signal)

where margin_f = logit(correct) - logit(incorrect) at f's answer slots. This is
the honest generalization of the accessibility potential A: the degree-2 class G
was a cartoon of the tangent space; G_disc IS the tangent space, grades within
G automatically, is recomputable at any theta_t (iteration = tangent-space
rotation; the selection law's ongoing competition = drift under G(theta_t)),
and degree/information-exponent/chain-length become theorems about its scaling.
PITFALL (measured): the raw full-CE gradient does NOT discriminate routes
(SNR 0.99 for all) — it is dominated by the route-independent marginal signal
("answers live in the answer alphabet"); the margin/discrimination projection
is essential.

## Pilot (1-layer GPTV, single-route tasks, degrees 1/2/3)
route      tau95 (2 seeds)   G_disc      SNR      tau x SNR
lookup     125 / 150         2.24e-1     0.162    ~22
XOR        650 / 525         9.35e-2     0.063    ~37
3-parity   2250 / 2250       1.60e-2     0.013    ~30

- tau ladder: each added input degree multiplies time by ~4 (137 -> 587 -> 2250).
- tau * SNR constant to within a factor < 2 across a 16x range of tau:
  **tau ∝ 1/G_disc with a shared constant** — the grading is quantitative.
- Parity drift does not vanish exactly at init (finite init scale + finite
  softmax temperature give continuous degree-attenuation, ~x2.4 and ~x5.8 per
  degree) — the information-exponent hard-zero story appears as smooth
  attenuation, which is better for a flow theory.

## Next (PREREG14 proper)
(i) scaling: G_disc and tau vs width d, seq length L, vocab — extract the
attenuation base (predicted ~ d_eff for IE steps); (ii) G_disc(theta_t)
trajectories along training (drift growth as prerequisites form = gating,
continuously measured); (iii) arbitration follows G_disc ordering at theta_t
(replaces PREREG13-P3 with the tangent-space law); (iv) fold into the master
potential: A(f|Phi) := G_disc(f|theta) — one object for Tests 1 and 2.
