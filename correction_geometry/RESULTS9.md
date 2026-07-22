# RESULTS 9 — PCFG-2 / Random Hierarchy Model (grading PREREG9)

**P1 (staged ladder) — ordering CONFIRMED, discreteness REFUTED, and the refinement
matters.** Class onsets are cleanly ordered by level (b0 descends 0–300, b1 100–1500,
b2 600–2500, b3 a slow tail to ~2500) and endpoints match the full-Bayes row to 2–7
mnat (b0 .3767/.3743, b1 .9351/.9341, b2 1.1596/1.1666, b3 1.3479/1.3473). But the
descents OVERLAP — the average CE blows through the stage values (1.0397, 0.8668, …)
without pausing; no plateaus. Refinement: leap-hardness raises per-level sample
complexity, which orders the onsets, but does NOT create serial GATING. Sharp rungs
(m27) come from dependency — the later correction's gradient literally does not exist
until the earlier circuit forms. RHM levels are learnable in parallel from their own
statistics (leap costs (vm)^l = 8, 64, 512 — all far below the budget), so the ladder
smears. Two distinct sources of staging: manufactured-gradient gating (sharp rungs)
vs complexity separation (ordered ramps).

**P2 (bootstrap crux) — falsifier FIRED, third and strongest time.** Hard-only (loss
at b ≥ 2 only, 18/95 targets) matches dense EXACTLY at the deep classes — b2 1.1596
vs 1.1596, b3 1.3476 vs 1.3479 — on the same timescale (b2 near floor by ~2000 in
both). The level-1/2 parse tables were tape-invisible, leap-hard, and never
supervised; attention credit assignment built them anyway. Untrained shallow classes
are anti-calibrated (b0 ~1.41, b1 ~1.48, worse than ln 4 — the type-conditional
signature, as in m27, unlike the type-uniform PCFG-1). The cumulative verdict across
PREREG7–9: position-sparse, class-sparse, tape-invisible, and tape-invisible+leap-hard
supervision ALL solve their tasks — dense supervision buys speed, position/type-
uniform calibration, and stage discreteness, never possibility. Caveat honestly
stated: our leap costs are tiny relative to budget; a quantitative barrier could
reappear when leap cost approaches the training budget — but the qualitative
"cannot learn without the dense scaffold" story is refuted in every version we could
construct.

**P3 (layer × level × time) — CONFIRMED, now with resolved time.** The ancestor-
posterior grid is monotone in layer at every checkpoint (parent decodable from L1–3;
root ONLY from layers 3–6, exactly zero at shallow layers even at step 4000) AND
ordered in time in sync with the class drops: anc1 rises 100→300, anc2 200→1000,
root 600→4000 — the inverse-Bayes pipeline is built level by level in time and
stacked level by level in depth. All levels keep consolidating to 4000 while the
loss is flat after ~2500 (the consolidation lag, fourth appearance).

**P4 (guess matrix) — endpoint version CONFIRMED spectacularly:** KL(full-BP ‖ net)
= 0.00009 nats over all positions — the net implements exact 4-level random-table
tree inference. The plateau version is moot (no plateaus).

## Addendum — the causal test of compositionality (asvin's pipeline-vs-parallel question)

Probing depth-stratification cannot distinguish (a) a PIPELINE (level j+1 computed
from level-j representations) from (b) PARALLEL-FINISH (each level computed directly
from tokens, deeper functions merely needing more layers). Interchange test on the
dense net: patch the pair(4,5) constituent symbol (cols 36-37, tree 2) along its
encoder image (per-column class-mean swap, the validated causal-subspace method);
query col 39 predicts leaf 8 (needs root inference consuming the parsed left half).
Oracle = BP with that pair's upward message forced to the donor symbol.

TRAP (cost us a first misread): a random donor symbol leaves a grammatical parse
only 18% of the time; inconsistent counterfactuals have zero-vector BP targets
(~0 KL contribution but garbage slopes). All metrics on the consistent subset
(N=366, oracle gap 0.052 nats):

- tape swap (ceiling): slope 1.004, corr 1.000 — and yield-invariant (only the
  symbol matters to the net, as BP requires).
- encoder patch: slope 0.99, corr 0.95, at L0-L3; collapses at L4 (0.23) and L5
  (0.10) — the symbol is read from its span in blocks <=3, then it is too late.
- full-column donor TRANSPLANT @L2 (replacement, donor context contaminants
  included): slope 0.958, corr 0.981 — the reader consumes only the symbol from
  that span.
- PROPAGATION (the requested test): decoded root belief at L5 shifts under the
  L2 pair-patch with slope 0.96-0.99, corr 0.94-0.96 vs the BP-predicted shift;
  random-direction control slope 0.004.

Verdict: causal (Gold-grade) compositionality — editing an intermediate
constituent's representation rewrites the deeper beliefs and the output exactly as
one step of BP predicts, with unity transfer. The depth stratification is a real
pipeline, not parallel-finish. (PCFG-1 contrast proposed but confounded: there the
node posterior ~ weighted sum of its own leaves, so the encoder direction overlaps
raw-leaf directions and even a flat circuit would respond; needs a sharper design.)
