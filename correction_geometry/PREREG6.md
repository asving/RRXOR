# PREREG 6 — Parallel factors and smooth fractals: completing the 2×2

Date: 2026-07-14. Written BEFORE training. Extends the correction-conditional +
spectral program to the two untested cells: {parallel (factored product process)} and
{smooth non-lattice (mess3)}, joining {serial laddered (m27)} and {smooth leaky (fx)}.

## Tasks

**F (factored)**: two INDEPENDENT factors emitted jointly as composite tokens
(vocab 4, token = 2·x_A + x_B):
- Factor A = classic rrxor3 (blocks [r1, r2, r1⊕r2], hidden offset mod 3).
- Factor B = repeat-2 (blocks [r, r], hidden offset mod 2: every second token copies
  its predecessor).
Exact factor machines: A = 5 states, transfer {1, e^±2πi/3, 0, 0}; B = 3 states,
transfer {1, −1, 0}. Joint machine = tensor product, 15 states; unit-circle spectrum =
products = the **6th-root lattice**: {1}, {−1}, {e^±2πi/3}, {−e^±2πi/3 = primitive
6th roots}.

**M (mess3)**: 3 symbols, 3 hidden states; emit own symbol w.p. 1−2x else x each other
symbol; stay w.p. a else switch uniformly. x = 0.15, a = 0.6. Transfer spectrum
{1, (3a−1)/2 = 0.4 (double)} — all real: pure timescale bands, no clock, fractal-ish
belief mixing.

Training recipe identical to fx/rrxor3: GPT d=128, nl=4, L=96, batch 256, lr 3e-4,
8000 steps, checkpoints every 25 to 1500 then every 250, eval every 20 steps logging
joint CE and (for F) per-factor marginal CEs.

## Closed-form reference grid (task F, nats, joint = sum of independent factors)

Factor policies: A-bet 0.6365 / A-gated 0.4621; B-bet Hb(3/4) = 0.5623 / B-gated
0.3466. Grid: uniform 1.3863 · both-bets 1.1988 · A-gated+B-bet 1.0244 ·
A-bet+B-gated 0.9832 · floor 0.8087.

## Predictions

- **P1 (parallel installation).** Each factor's clock is born inside its own factor's
  loss-drop era (read from ce_A(t), ce_B(t) separately), the two eras are set by
  per-factor gradient drive (B is simpler → expect τ(−1) ≤ τ(ω)), and the eras MAY
  OVERLAP — no manufactured-gradient ordering between factors. Contrast m27: strictly
  ordered non-overlapping eras. Operationalized: birth τ(sector) = first checkpoint
  whose snapped cohort has |λ| > 0.75.
- **P2 (cross-terms are free).** The primitive-6th-root sector (the A×B cross
  products) is born at max(τ(ω), τ(−1)) within one checkpoint spacing — it is the
  automatic tensor product of the factor clocks, NOT a separately learned correction.
  Falsifier: τ(6th) lagging max(τ_A, τ_B) by ≥ 3 checkpoints (cross-structure needing
  its own drop would break the modularity account).
- **P3 (the ladder is a grid).** Sector ablations of the final F machine land near the
  2D policy grid: {1}+transients ≈ both-bets 1.199; +cube pair ≈ 1.024; +{−1} ≈ 0.983;
  full ≈ 0.809 (tolerance: at-or-below each rung by ≤ 0.04, per the rrxor3 lesson that
  clock-blind machines exploit short-window info).
- **P4 (loss additivity).** The joint loss curve ≈ ce_A(t) + ce_B(t) with each factor
  curve resembling its solo counterpart; drops may superpose in the joint curve.
- **P5 (mess3 smooth).** No plateaus in the loss beyond the initial calibration; the
  net's transfer eigenvalues converge to {1, 0.4, 0.4} (real — magnitude ramps, no
  angle-locking); the two-level sector ladder ({1} → iid ≈ entropy of the marginal;
  full → floor) exists as machines but the training curve never sits on the
  intermediate value.
- **P6 (cross-alphabet validity).** The spectral pipeline (conditional rows, exhaustive
  tests, operator regression) reproduces theory machines for vocab 3 and 4 with
  operator-fit R² > 0.999 at convergence — first non-binary check of our
  implementation.

## Falsifiers

Serial ordering of the two factor clocks with counterphase per-factor losses
(capacity-forced seriality — would refute parallel gradient concentration for
independent factors at this scale); cross-sector born late (P2); mess3 developing a
plateau at the sector-machine value (would suggest sector ladders constrain SGD paths
even without aligned corrections); pipeline failing at vocab ≠ 2.
