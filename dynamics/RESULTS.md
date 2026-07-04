# Results vs preregistered predictions (see PREREG.md — written before any run)

3 seeds, 4L/128d transformer, RRXOR with random phase offset, online data, 8k steps.
All numbers at late target positions (>= 33). Figures: runs/seed*/curves.png,
probe_heatmap.png, ablate.png.

## Prediction 1 — non-monotonic subtask loss: CONFIRMED, quantitatively

All 3 seeds: random-position CE descends to ln 2 (~step 100), is pushed BACK UP above ln 2
while the xor circuit forms, plateaus near the predicted stage-1 value 0.7520
(type-0 peaks 0.749/0.749/0.751 — vs pred 0.752; type-1 peaks 0.768/0.775/0.775, slight
overshoot), then relaxes to ln 2 as phase gating completes. Final averages 0.4621–0.4623
(pred 0.4621).
p(xor of prev two): rises to ~2/3 at ALL position types during the plateau (ungated stage-1
policy, as predicted), then splits to 1.0 at xor positions / 0.5 at random positions.

The rise above ln 2 at positions the xor feature cannot help with is the direct signature of
"the early circuit manufactures the gradient for the later circuit": the miscalibration pain
IS the phase circuit's training signal.

## Prediction 2a — chronological order: CONFIRMED

xor-position loss drop onset ~step 200–400; phase gating (random-position relief +
xor-position confidence -> 1.0) completes ~step 2500–3000. Strict ordering, all seeds.

## Prediction 2b — depth order: CONFIRMED

Linear-probe heatmaps over (layer x checkpoint), all 3 seeds: xor feature decodable at
L2–L4 from ~step 300–400 (early, shallow-ish); phase decodable ONLY at L4 (L3 stays at
0.46–0.59), and only from ~step 1500–2500 (late, deep, strictly past the xor feature's
write layer). At step 400: xor probe 0.84–0.97 at L3–L4 while phase is at chance at every
layer. Time order = depth order = dependency order, at the level of features.

## Prediction 2c — ablation time travel: FAILED as stated; informative refinement

Preregistered: ablating the phase subspace reverts behavior to the full stage-1 plateau
(0.405 / 0.752 / 0.752, pxor ~ 2/3 everywhere).

Observed (subspaces from class means, mean-ablated at every residual point; seed 0 numbers,
cross-seed notes below):
- naive phase ablation: ce = (0.694, 0.694, 0.649) — looked like total fusion, but the
  phase and xor class-mean subspaces overlap (max cos principal angle 0.66 at L2/L4),
  so this also cut the xor value wire. Confound.
- phase_perp (xor direction preserved): ce = (0.741, 0.737, 0.586). Random-position
  miscalibration of stage-1 MAGNITUDE reappears (0.74 vs 0.752 pred), but the xor-position
  hedge returns at pxor ~ 0.56, not the stage-1 value 2/3, and the random-position bets are
  dispersed (mean pxor 0.54 with CE 0.74 implies noisy, not clean, hedging).
- xor ablation: catastrophic and WORSE than chance at xor positions (ce2 = 1.21): the gate
  survives and bets confidently on a corrupted value. Random positions also miscalibrate
  (0.77) because phase evidence (violation detection) runs through the same feature.
- random-subspace control: no effect.

Cross-seed: same qualitative anatomy (gate on top at L4, xor feature load-bearing below;
xor ablation catastrophic at xor positions: ce2 = 1.21 / 0.64 / 3.17), but the DEGREE of
reversion under phase ablation varies with how entangled the gate directions are with the
value direction at L4 (max cos principal angle 0.90 / 0.66 / 0.46 across seeds). Seed 1 —
the most entangled — gives a nearly exact stage-1 reversion at the xor position under full
phase ablation (ce2 = 0.408 vs pred 0.4055, pxor2 = 0.694 vs 2/3), while its random
positions overshoot (0.82–0.84, dispersed bets). Seeds 0/2 revert partially (hedge ~0.55).
No seed cleanly recovers the full stage-1 policy at random positions.

Refined conclusion: stage 2 does not stack on top of a preserved stage-1 circuit. It
PRESERVES the stage-1 REPRESENTATION (the xor feature — still linearly decodable, causally
load-bearing for both output and phase evidence) and DISMANTLES the stage-1 POLICY (the
unconditional 2/3 bet — visible in-flight as p(xor) at random positions decaying 0.65 -> 0.50
during stage 2, and post-hoc as the ~0.56 rather than 2/3 hedge under phase ablation).

Frame update: hierarchical learning reuses the scaffold's representation and refactors its
output head. The fossil record of training history is in the features and their depth
placement, not in behaviorally recoverable strata.
