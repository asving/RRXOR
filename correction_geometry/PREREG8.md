# PREREG 8 — Type-masked loss (m27) and the tape-invisible bootstrap (PCFG)

Date: 2026-07-21. Written BEFORE training. Two parts: (A) the class-sparsity version
of PREREG7 that asvin actually intended; (B) the PCFG where intermediate variables
never appear on the tape — the sharp test of Cagnetta–Wyart's bootstrap — plus its
developmental ladder and circuit-breakdown plan.

## Part A — m27, loss masked by token TYPE (same-seed vs dense m27_nl6_seed1)

- **det-only** (loss at all det slots, 1/3 of positions; 24k steps): the rule tower
  stays necessary (det target = u ⊕ rule needs phase mod 9/27), only the mod-3
  gating pressure is removed. Predictions: tower climbed; the staircase and the
  xnor manufactured-gradient excursion SURVIVE (the cascade lives at det slots);
  random-slot outputs become garbage (never trained).
- **exceptions-only** (loss at xnor+flip slots only, 3/27 of positions; 32k steps):
  removes the easy-rung (xor-class) supervision whose circuit historically funded the
  exception rungs. Predictions: (i) solvable but with a substantially larger
  sequence-factor than dense (PREREG7 precedent: attention credit assignment
  suffices; the load-bearing question is the factor size); (ii) the free-
  generalization split: the u/rule LATENTS form (shared circuitry) but xor-class
  OUTPUTS stay uncalibrated (never trained — cf. PREREG7 P3); xor-class CE does not
  approach the dense floor.

## Part B — PCFG: noisy-copy trees (broadcasting on trees)

Task: 3 independent depth-5 binary trees per sequence (L = 96, vocab 2); root
uniform; each child copies its parent w.p. 1−η, η = 0.1; only leaves are emitted —
all internal nodes are tape-invisible. Position class = boundary depth b = ν₂(leaf
index within tree) ∈ {first, 0, 1, 2, 3, 4}: predicting a class-b leaf requires
inferring the level-b shared ancestor from the prefix (inverse/upward Bayes) and
propagating down (forward Bayes). Oracle: exact BP, validated against brute-force
enumeration (max |Δ| = 0.0).

**The preregistered class × stage CE matrix** (stage ℓ = truncated BP using ancestors
≤ ℓ levels up; Monte Carlo, 8192 trees):

```
        first   b=0     b=1     b=2     b=3     b=4     stage avg
l=0:    .6931   .6931   .6931   .6931   .6931   .6931   .6931
l=1:    .6931   .4706   .6931   .6931   .6931   .6931   .5819
l=2:    .6931   .4565   .5872   .6931   .6931   .6931   .5483
l=3:    .6931   .4515   .5762   .6330   .6931   .6931   .5355
l=4:    .6931   .4491   .5712   .6251   .6578   .6931   .5299
l=5:    .6931   .4484   .5689   .6222   .6531   .6731   .5277
```

Runs (GPTV nl=6 d=128, batch 256, 60k steps, same seed): **dense** (all targets) and
**hard-only** (loss only at class b ≥ 3 positions: within-tree indices 8, 16, 24 — 9
targets/sequence).

- **P-B1 (tape-invisible bootstrap — the crux).** Hard-only training fails to reach
  the deep-class floors or is severely slowed: unlike m27/rrxor3 (tape-visible),
  the level-1/2 representations needed for upward inference receive no direct
  supervision and never appear as tokens. Quantified falsifier: hard-only within
  0.005 nats of the dense b=3 / b=4 class values (0.6531 / 0.6731) by 60k steps ⇒
  the bootstrap confers no trainability advantage even for tape-invisible latents,
  and the Cagnetta–Wyart mechanism is about sample-routing only.
- **P-B2 (developmental ladder, dense run).** Average CE descends through the stage
  averages with visible drops at least for stages 1–2 (rungs 3–5 are ≤ 6 mnat and
  may blur into a ramp — smoothness there is NOT a falsifier); per-class CE curves:
  class b improves only when its stage arrives, and at each plateau the per-class
  values match row ℓ of the matrix (tolerance 0.01); at each plateau the net's
  CONDITIONALS match truncated-BP-ℓ per position (correction-spectroscopy style —
  "predicting what the network guesses at every loss-drop moment").
- **P-B3 (circuit breakdown; analysis after training).** Probe battery = exact
  upward posteriors of the level-j ancestor of the next leaf (j = 1..5), probed per
  NETWORK LAYER per checkpoint, window-baseline-subtracted (corrprobe machinery).
  Predictions: (i) time-order: level-j posterior becomes decodable in sync with drop
  j / its class's CE drop; (ii) layer-order (EXPLORATORY, weak): deeper ancestor
  levels formatted at deeper network layers — inverse-Bayes built layer by layer;
  honest alternative from m27: correction statistics may live in attention wiring,
  invisible to pooled stream probes; (iii) secondary: downward path-node beliefs.

## Falsifiers

A: exceptions-only stalling forever at the u-formation stage would (interestingly)
revive the within-tower bootstrap for class-sparse loss. B1 falsifier stated above.
B2 dies if classes improve out of order (deep before shallow) or plateau values miss
the matrix by ≫ 0.01. B3(i) dies if ancestor posteriors are decodable long before
their class's behavioral drop (representation-first would contradict the correction-
conditional account here).
