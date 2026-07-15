# RESULTS 6 — Parallel factors and smooth fractals (grading PREREG6)

Runs: factored_seed0 (vocab 4: rrxor3 factor A × repeat-2 factor B, composite tokens),
mess3_seed0 (vocab 3, x=0.15, a=0.6). Training per PREREG6 recipe. Artifacts:
train_multi.py, speclib_v.py (vocab-general spectral library), prereg6_analysis.py,
per-run prereg6_*.json. Pipeline validated on exact oracles first (fit R² 1.00000,
spectra exact: factored {1, cube pair, −1, prim-6th pair 1.000∠1.047, 0⁹} at rank
exactly 15; mess3 {1, 0.4, 0.4}) after fixing a porting bug (see Pitfalls).

**P1 (parallel installation) — PASS, dramatic.** The B-clock (−1) is born at step 250,
exactly inside factor B's own loss drop (ce_B 0.554 → 0.352 across steps 200–300), and
stays at 1.00 thereafter. The A-clock (cube pair) is born ~2500–3000 (0 → 0.83 → 0.92),
inside factor A's gradual gated descent — ten times later in training. The factors
never interact: no counterphase structure, B flat at its floor while A keeps learning.
Fully parallel, independent installations, timed by per-factor gradient drive
(B simpler → earlier), as predicted. Contrast m27's strictly serial, manufactured-
gradient-ordered eras.

**P2 (cross-terms are free) — PASS via the factorization route; direct spectral
detection is noise-limited.** The net's predictions factorize exactly: |ce_avg − ce_A −
ce_B| ≤ 2.4e-4 at every eval from step 100 on (0.022 only at the step-0 calibration
transient). Factor machines extracted from the net's MARGINALS are essentially exact —
A: {1, 0.995∠2.097, 0, 0} fit 0.993; B: {1, −0.999, 0} fit 0.9993; full-machine
rollouts at the factor floors (0.4681 / 0.3485 vs 0.4621 / 0.3466). Tensor structure +
factor clocks ⇒ the primitive-6th cross modes exist as products by construction, at
every point where both factor clocks exist — never separately learned. Direct
detection in the joint extraction fails for a documented reason: the cross modes'
singular values (0.036–0.058·σ₁, measured on the exact table) sit at the net's
behavioral noise shoulder (0.02–0.035·σ₁) — with the exact table the same tests and k
recover the full 6th-root lattice perfectly, so this is an instrument-resolution
limit, not physics.

**P3 (grid rollouts) — NOT CERTIFIED (instrument failure).** Sector-ablated rollouts
of the joint machine (and of clock-ablated factor machines) diverge — oblique spectral
projectors are ill-conditioned when the retained transients are near-degenerate
(|λ| ≈ 0.01 with noisy eigenvectors). The grid's endpoints are supported indirectly
(factor floors + additivity), but the intermediate rungs were not certified by
rollout. Open: a better-conditioned ablation (e.g., Schur-basis projection).

**P4 (additivity) — PASS** (see P2; ≤ 2.4e-4 across training).

**P5 (mess3 smooth) — PASS.** The real eigenvalue pair ramps 0.16 → 0.25 → 0.35 →
0.41 and settles ≈ 0.39/0.36 (theory 0.4, double), fit R² 0.999, no angle-locking, no
plateaus; loss descends smoothly. Sector check: {1}-only machine gives CE 1.0987 = ln 3
(iid), full machine 1.0836 = the net's own CE. The two-level ladder exists as machines;
training never sits on it.

**P6 (cross-alphabet validity) — PASS** (oracles exact at vocab 3 and 4; net fits
0.99+).

## Pitfalls logged

1. **Shift-symbol double count** (vocab-general port): including the shifted symbol's
   probability in the shifted string product AND as the regression weight shrinks ALL
   operators by E[P(σ|h)] — signature: the entire spectrum scaled down uniformly, top
   eigenvalue ≪ 1 and constant across training. The stationary eigenvalue pinned at
   E[P(σ|h)] (0.344 ≈ 1/3 at vocab 3) is the tell.
2. **Column normalization + exhaustive tests + hard zeros**: forbidden test strings
   (softmax leakage columns) get inflated to unit norm and pollute the SVD subspace —
   the column-side twin of Jasmina's garbage-row mode. Use colnorm only with
   generator-sampled (allowed-only) tests.
3. **Oracle-vs-net indexing conventions differ** (filter tables index by target
   position, net logits by source position) — validate any new table builder against
   an exact oracle before touching net data.
4. Marginalization as a noise filter: factor machines extracted from marginal
   conditionals are much cleaner than the same modes inside the joint extraction
   (A-clock 0.995 vs 0.92) — sums over the other factor average its noise away.
