# RESULTS 8 — Type-masked m27 and the PCFG (grading PREREG8)

## Part A — m27 type-masked loss

**det-only — CONFIRMED in full.** The tower is climbed from det-slot loss alone, with
the staircase and BOTH manufactured-gradient excursions intact and sharpened: xnor
spikes to 1.495 at step 1000 (dense twin: 1.355), flip rides the stage-3 signature
(1.01 at 3k, heading for the closed-form −ln(1/3) = 1.0986 before resolving); all det
classes reach 0.0000 by ~6k (dense: 5.6k) with a third of the loss terms. Random-slot
outputs are anti-calibrated garbage (CE ~6.6) — the class-sparse analog of
machine-at-one-position.

**exceptions-only — no stall, and a new rung.** With loss at 3/27 of positions and NO
xor-class supervision, the net still solves both trained classes (xnor 0.021, flip
0.045 by 32k; ~5× dense in steps). The within-tower bootstrap is not load-bearing
either. Two discoveries: (i) at step 2000 the net sits exactly on a rung that does
NOT exist in the dense ladder — bet-the-XNOR-value-everywhere at its own training
distribution (xnor 0.4064 ≈ −ln(2/3), flip 1.0967 ≈ −ln(1/3)) — reinforcing PREREG7:
the ladder is a property of task × supervision, and each supervision regime
manufactures its own correction tower; (ii) free generalization FAILS with the
anti-calibrated signature: at the untrained xor class the net confidently bets XNOR
(CE 6.3–6.7, ~10× worse than uniform) even though its latents must distinguish the
classes to solve flip.

## Part B — PCFG (noisy-copy trees): both preregistered falsifiers fired, same root cause

**P-B2 (developmental ladder) — REFUTED.** No stages: ALL boundary classes reach
their Bayes floors nearly simultaneously within ~200–1000 steps (b0 0.4534 at step
200 vs floor 0.4484; b4 0.6769 vs 0.6731), no drops, no rung ordering.

**P-B1 (tape-invisible bootstrap) — falsifier TRIGGERED.** Hard-only training (loss
at b ≥ 3 positions only) reaches the deep-class floors (b3 0.6554 vs 0.6531; b4
0.6712 vs 0.6731 — within tolerance), and even transfers partially to untrained
shallow classes (b0 0.6233, well below ln 2, unlike m27's anti-calibration — because
the optimal circuit here is type-uniform).

**Diagnosis (the informative negative).** The noisy-copy tree at η = 0.1 is
compositionally SHALLOW: the exact posterior is approximately a monotone weighted
vote over pairwise-correlated seen leaves — linearly accessible, no leap structure —
so every level is gradient-accessible immediately and neither stages nor bootstrap
barriers exist. Sharpened conclusion for the Cagnetta–Wyart program: TAPE-INVISIBILITY
ALONE IS NOT SUFFICIENT for the bootstrap to matter; the hidden latents must also be
computationally deep (nonlinear/leap-hard composition, as in RHM-style random
discrete productions or parity trees). That is the redesign for PCFG-2.

**P-B3 (circuit breakdown) — CONFIRMED, including the exploratory half.** Probing the
exact level-j ancestor posteriors (validated BP oracle) from each network layer:
the grid is monotone in (level, layer) — level-1 decodable from layer 1 (+0.43 excess
over an 8-token window), levels 3 and 5 (root) decodable only from layers 3–6
(rising to +0.26/+0.27 at L6) — the inverse-Bayes upward pass is literally laid out
along network depth, even though learning was not staged. Time-order appears in
compressed form: level-1 present at step 100, deep levels jump between 100→200 (in
sync with the loss transient), then consolidate 200→3000 while the loss is flat —
the consolidation-lag pattern again.
