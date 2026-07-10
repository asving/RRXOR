# Correction-conditional geometry & spectral reconstruction

**Claim tested here**: a network's representations are not the belief geometry of the
*task* but the minimal sufficient statistics of its stagewise *corrections* — at each
sharp loss drop, what forms is the ε-machine of the likelihood-ratio adjustment
c(x) = f_post(·|x)/f_pre(·|x) given the scaffold built so far, and refinements of the
task's belief simplex that belong to no correction never form at all. Preregistered in
`PREREG5.md` (with one documented amendment), graded in `RESULTS5.md`.

## The three analyses

**1. Probe trajectories** (`corrprobe.py`, `corrprobe_plots.py`, `corrfig.py` →
`corrprobe.png`, `corrfig.png`). Decodability of every oracle latent at every
checkpoint, in excess of a 16-token window baseline. Each correction statistic rises
inside its own loss-ladder era (u → mod-3 → exception identity); the never-consumed
14-dim refinement of the *perfectly inferable* 27-way phase stays below baseline at
all 76 checkpoints (the kill shot). Two refinements: correction statistics can live in
attention *wiring* rather than stream codes (matryoshka's mod-9 alignment), and linear
codes consolidate ~0.5–1k steps *after* behavior (they keep sharpening long after the
loss converges).

**2. Correction spectroscopy** (`corrspec.py` → `corrspec.png`). Hypothesis-light:
measure δ(x) = λ_post − λ_pre across each drop and ask what history quotient it
factors through. RRXOR drop 1: E[δ | all 16 length-4 suffixes] collapses onto
⊕(last two) — the first correction machine IS the 2-state xor machine, read off with
no hypothesis. Drop 2: no suffix function explains δ (table R² 0.25); the offset
posterior × candidate reaches R² 0.85. Matryoshka: each of the four measured drops
matches exactly its own stage's closed-form correction (4×4 R² matrix, diagonal
0.90/0.74/0.86/0.84, off-diagonal ≤ 0.34).

**3. Spectral (Hankel/OOM) reconstruction** (`hankel.py`, `hankel_fig.py`,
`hankel27.py`, `hankel27_fig.py` → `hankel_fig.png`, `hankel27_fig.png`). Rebuild the
HMM from the net's behavior alone: H[h,s] = P_net(s|h) teacher-forced, SVD, shifted-
state regression → observable operators. RRXOR: behavioral Hankel converges to the
true generator's rank-5 spectrum (match to 3 decimals); the transfer operator carries
the phase clock at 0.998∠2.0944 (e^{2πi/3} = ∠2.0944); a 12-step autoregressive
rollout of the rank-5 machine tracks the net at |Δp| ≤ 0.003. Matryoshka: true machine
rank exactly 45 with all 27 eigenvalues on the 27th-root lattice; the net's unit-circle
spectrum fills **1 → 3rd roots → 9th roots → 27th roots** across the drops — the
Pontryagin dual of the Z27 ⊃ Z9 ⊃ Z3 quotient tower appears rung-by-rung in pure
behavior, and the operator fit R² dips mid-drop (behavior between machines) and
returns to ≥0.9996 on every plateau.

Method pitfalls documented in RESULTS5: tests too short → the Hankel sees a rank-5
shadow and the rotation pair collapses onto its real part (−0.5 = cos 2π/3); fitting
k above the true rank injects null-space garbage; long-test columns must be
per-test normalized or they are numerically invisible; count root classes by snapping
near-unit eigenvalues to the lattice.

Data: `runs/*_corrprobe.json` (probe trajectories), `runs/seed0_hankel_traj.json`,
`runs/m27_hankel27.json` (all eigenvalues per checkpoint), `corrprobe_summary.json`.
