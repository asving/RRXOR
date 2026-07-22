# PREREG 10 — PCFG-3: full-support probabilistic RHM (asvin's isolation of the fusion axis)

Date: 2026-07-22. Written BEFORE training. PCFG-3 = the PREREG9 RHM (V=4, M=2,
DEPTH=4, grammar seed 7) with productions blurred to full support:
P(b,c|a) = 0.95 * (1/2) sum_k 1[(b,c)=prod_k(a)] + 0.05/16. This makes PCFG-1 and
PCFG-3 the same object (probabilistic tree, full support, same topology/classes)
differing in ONE property: whether child-evidence fusion factorizes through
per-child sufficient statistics (PCFG-1: children conditionally independent given
parent, scalar-additive tanh rule) or not (PCFG-3: random bilinear map on message
4-vectors). It also removes the deterministic-parse manifold: every counterfactual
constituent value is grammatically consistent (finite likelihood ratios), fixing
the 18%-consistency trap of the PREREG9 causal test. Soft BP oracle validated vs
enumeration (max err 1.7e-16). At eps=0.05, >half of trees contain a noise event,
so soft inference is load-bearing; full-BP floors (MC 8192): b0 .5854, b1 1.1163,
b2 1.2654, b3 1.3664 (available deep signal: b2 0.121, b3 0.020 nats).

Runs: dense + hard-only (loss at b>=2, within-tree indices 4/8/12), GPTV nl=6
d=128, 60k steps, seed 0, same-seed pair.

- P1 (ladder): ordered overlapping onsets by level persist (leap preserved,
  possibly smoother); endpoints at the soft floors above (tol 0.01 for b0-b2).
- P2 (bootstrap, 4th attempt): hard-only matches dense at b2/b3 within 0.02 by
  60k (we now EXPECT the falsifier to fire, and preregister that expectation).
- P3 (causal replication, clean instrument): interchange on the pair(4,5)
  constituent with soft-belief encoder maps; 100% consistent donors; transfer and
  root-propagation slopes ~1 as in PREREG9-addendum.
- P4 (dose-response linearity — the NEW sharp test): BP fusion is LINEAR in the
  patched child message holding the sibling fixed; patching the belief to points
  along a segment in the simplex should move the net's output logits linearly
  (R^2 >= 0.9 of net shift vs interpolation coefficient). Falsifier: saturating or
  step-like response = the net consumes a hardened symbol, not the soft belief.
- P5 (graded beliefs): interior-node posteriors on noisy trees are decodable as
  CONTINUOUS values (probe R^2 vs soft targets exceeding the hard-parse-target
  fit); the net carries a belief manifold, not a symbol table. Spectral
  expectation (leaky-RRXOR parallel): parse-sector transfer eigenvalues contract
  by ~(1-eps) per level.
