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
