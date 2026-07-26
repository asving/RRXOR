# RESULTS 13 — Test 2 (selection uniformity), v1 + v2 with Codex as critic

Task recap: sequences of shuffled [source(s), marker, answer] items; a shared
item answerable from either route-1 or route-2 (redundant); m1/m2 auxiliary
items exercise each route alone; arbitration read out by eval-only conflict
trials (pref1 = P(shared answer follows route 1)); single-route controls
measure each route's standalone speed (lookup tau95 ~ 100 steps, XOR ~ 375).

## v1 grid (PREREG13)
- Frequency-only null DEAD: lookup-vs-XOR arm gives pref1 = 1.00 at EVERY
  (m1,m2) incl. (1,8), both seeds, locked by ~step 200 — while the XOR circuit
  exists (aux at ceiling). Weight cannot buy back an accessibility deficit up
  to 8x at these scales.
- Preregistered smooth shifted-curve form REFUTED (no compensation).
- Symmetric lookup-lookup arms: weight acts monotonically but v1's instrument
  was confounded (route-2 source always marker-adjacent) — found post-hoc,
  fixed in v2 by randomizing shared-source order.

## v2 (Codex-designed discriminators)
- CROSSOVER (phase 1: lookup masked at shared + m1=0 until XOR owns the slot,
  pref1 pinned 0.00; phase 2: lookup enabled): lookup TAKES OVER the owned
  slot — pref1 crawls 0.00 -> 0.52 (seed 0) / 0.29 rising (seed 1) over 2500
  steps. ARRIVAL-ORDER LOCK-IN FALSIFIED: correction-zeroing does not freeze
  routing; aux gradients on shared parameters keep reshaping it. But
  incumbency FRICTION is real: takeover is ~50x slower than fresh acquisition
  (fresh: locked by 200 steps).
- COMPLEXITY REVERSAL (XOR moved to route-1's alphabet+position, lookup to
  route-2's): lookup still wins totally (pref1 -> 0.00 = follows lookup, both
  seeds). Simplicity selection survives swapping labels, alphabets, positions.
- LL WEIGHT CURVE (fixed instrument, shared-source order randomized): monotone
  and soft — m4x1 0.66, m2x1 0.70, m1x1 0.62 (4 seeds 0.47-0.91,
  near-bistable), m1x2 0.36, m1x4 0.39. Weight biases arbitration but
  saturates quickly and never approaches lock-in.

## The selection law (replaces PREREG13's P3)
Consumption sites drift toward the most ACCESSIBLE adequate supplier:
1. accessibility ordering dominates (overrides >=8x weight and position);
2. weight is a secondary, soft, quickly-saturating bias among
   accessibility-matched routes;
3. incumbency adds friction (hysteresis ~50x) but not permanence.
Theory consequence: the potential A needs a WITHIN-G complexity grading (max
over G treats copy and parity as equal; selection does not), and selection is
ongoing competition under that grading — not arrival-order lock-in, not
frequency counting. Connects to repgauge (there the weight axis was varied at
matched accessibility -> reuse won; here accessibility varies -> it dominates).

Falsifier discipline: PREREG13-P3's quantitative form (shifted universal
curve) is FALSE as written; the direction (accessibility enters selection) is
confirmed at maximal strength. Codex critiques and both discriminating designs
are recorded in PREREG13/this file.
