# Preregistration 2 — scaffold seeding, route selection, coarsening trajectory

Date: 2026-07-03. Written before running any of these experiments. Baseline numbers
referenced from RESULTS.md (seeds 0-2): tau_xor ~ 350-450 (p(xor)@type2 crosses 0.6),
gating completes ~2500-3000, type-1 excursion peak ~0.77.

## Test A — scaffold seeding / dose-response (move the scaffold, the consumer moves)

Give the model the xor bit b_t = x_t XOR x_{t-1} as an auxiliary INPUT channel (extra
embedding added at the embedding layer), flipped with probability p (noise dose).
p = 0.5 is an uninformative control (= baseline); p = 0 is a perfect free scaffold.
Sweep p in {0, 0.1, 0.2, 0.3, 0.4, 0.5}, 2 seeds each.

Metrics (from eval.jsonl): tau_gate = first step where pxor2 - mean(pxor0, pxor1) >= 0.15
(gating signature, independent of value quality); tau_solve = first step ce2 < 0.2;
excursion = max ce1 after calibration (first ce1 < 0.70).

Predictions:
A1. tau_gate is monotone increasing in p; tau_gate(p=0.5) ~ baseline; tau_gate(p=0) <<
    baseline. The excursion ONSET also shifts much earlier for informative bits (no
    waiting for the internal xor circuit to form).
A2. DEPTH MOVES WITH THE SCAFFOLD: in the p=0 run, phase becomes linearly decodable at
    SHALLOWER layers than L4-only (baseline had phase at L4 only, xor computed at L2-3).
    Moving the prerequisite from L2-3 to the input should pull the phase computation down.
    This is the "depth is a fossil record of dependency, not of the data" claim.
A3. tau_bind-law additivity check (exploratory, from group_comp law
    tau_composite ~ max(parts) + binding): compare tau_gate(baseline) vs
    tau_xor(baseline) + tau_gate(p=0). If the binding interval is intrinsic, these match.
    I expect partial failure: the baseline scaffold is weak/noisy while it forms, so
    baseline binding should be SLOWER than tau_gate(p=0); the dose-response curve
    tau_gate(p) quantifies how binding time scales with scaffold quality.

## Test B — two-route selection (flat cue vs hierarchical violations)

Add an independent flat route to phase: bias type-0 tokens to Bernoulli(0.5 + eps)
(type-1 and xor tokens remain unbiased — the cue lives ONLY at type-0 positions, so
phase becomes inferable from positional token statistics without any xor feature).
Sweep eps in {0.02, 0.05, 0.1, 0.2}, 2 seeds; baseline runs serve as eps = 0.

Diagnosis of which route the final net uses for phase: evaluate the trained net on
OOD eps = 0 sequences. Cue-reliant net: phase collapses -> ce2 rises well above ~0.1
and pxor2 drops; violation-reliant net: unaffected (ce2 ~ 0).

Predictions:
B1. Route choice flips with eps: small eps -> violation route (ce2_ood ~ 0);
    large eps -> cue route (ce2_ood elevated). The transition is SHARP in eps
    (winner-take-all preemption), not a gradual proportional mixture.
B2. The prediction-1 excursion SHRINKS with eps and disappears for large eps:
    if the cue route gives phase before/with the xor feature, there is no ungated
    intermediate stage, so type-1 loss never rises above ln 2. The non-monotonicity is a
    signature of learning ORDER, and the cue reverses the order.

## Test C — coarsening trajectory (belief geometry unfolds coarse-to-fine)

Ground-truth belief states from the 5-state generator HMM {A, B0, B1, C0, C1} via Bayes
filter (prior = stationary with random offset). For each existing baseline seed-0
checkpoint: affine (ridge) regression from the layer-4 residual stream (all positions
t >= 2, so interior beliefs are represented) onto (i) the full 5-dim belief and (ii) the
3-dim phase marginal (A, B, C). Report R^2; scatter the regression-projected residuals
on the phase 2-simplex at selected checkpoints.

Predictions:
C1. During the stage-1 plateau (~steps 400-1500) phase-marginal R^2 stays low while the
    xor-value feature is already decodable: the network represents the phase-MARGINALIZED
    quotient of the belief geometry, not the full simplex.
C2. Phase-marginal R^2 rises in lockstep with the random-position loss relief
    (~steps 1500-3000), and the simplex scatter visibly unfolds from a collapsed blob to
    the full 2-simplex fan. Learning = coarse-to-fine refinement of the MSP geometry.

Falsifiers: tau_gate flat in p (A1); phase decodable only at L4 even with input scaffold
(A2); ce2_ood varying linearly with eps with no sharp transition (B1); excursion amplitude
unchanged at large eps (B2); phase-marginal R^2 high during the stage-1 plateau (C1).
