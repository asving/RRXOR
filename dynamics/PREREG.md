# Preregistered predictions — RRXOR hierarchy test

Date: 2026-07-03. Written BEFORE running any experiment or looking at any prior result.

Frame: circuits are learned in order of gradient accessibility given current features.
Hierarchy = serial gradient concentration: an earlier circuit's latent output turns on the
gradient for a later circuit, which must therefore sit at greater depth (the latent is only
readable past the layer that writes it, at the moment the later circuit crystallizes), and
preemption freezes that placement in.

Task: RRXOR — repeating blocks [r1, r2, r1 XOR r2], r1,r2 ~ Bern(1/2), random phase offset
per sequence (uniform over {0,1,2}), binary vocab, next-token prediction, online data.

Position types (of the TARGET token): type 0 = first random, type 1 = second random,
type 2 = xor (deterministic given previous two tokens). All losses in nats, measured at
late target positions (>= 33) so phase is resolvable from context.

## Prediction 1 — non-monotonic subtask loss (the smoking gun)

Stage 0 (init/baseline): all types at ln 2 = 0.6931.
Stage 1 (ungated xor feature, optimal hedge q = 2/3 on the xor value):
  - type 2 loss: -ln(2/3) = 0.4055
  - type 0/1 loss: -(1/2)ln(2/3) - (1/2)ln(1/3) = 0.7520  <-- RISES ABOVE ln 2
  - average: 0.6365
Stage 2 (phase-gated, converged):
  - type 2 loss: ~0
  - type 0/1 loss: back DOWN to ln 2 = 0.6931
  - average: (2/3) ln 2 = 0.4621

The loud signal: random-position loss goes UP above 0.6931 (toward 0.7520) during an
intermediate window, then comes back down. The early circuit (xor feature) manufactures the
gradient for the later circuit (phase inference): the miscalibration pain at random
positions IS the phase circuit's training signal.

Secondary check: p(model assigns to xor of prev two) by type — during stage 1, ≈ 2/3 at all
three types (ungated); after stage 2, ≈ 1 at type 2 and ≈ 1/2 at types 0/1 (gated).

Falsifier: if type-0/1 loss never rises above ln 2 during training (at any eval, fine-grained),
the "circuits manufacture successors' gradients" mechanism is wrong for this task.

## Prediction 2 — three orders coincide (time = depth = causal dependency)

(a) Chronological: xor-feature emergence strictly precedes phase-geometry emergence
    (visible as ordered drops in the per-type loss curves).
(b) Spatial: linear-probe heatmaps over (layer x training step): xor feature (seq[t]^seq[t-1])
    becomes decodable EARLY in training and SHALLOW in depth; phase ((off+t) mod 3) becomes
    decodable LATER in training and only DEEPER than the xor feature's write layer.
(c) Causal, the time-travel demo: at the final checkpoint, mean-ablating the phase subspace
    (clamped at all layers from where it first appears) reverts per-type losses to the
    STAGE-1 plateau values (~0.405 / ~0.752 / ~0.752). Ablating the xor-feature subspace
    reverts everything to baseline (~0.693). Peeling circuits off the top of the depth
    hierarchy walks the network back through its own training history.

Falsifiers: phase decodable at or above the xor write layer before xor is decodable;
phase-ablated losses landing at baseline (0.693 everywhere) instead of the stage-1 values
(which would mean no separable stage-1 circuit survives in the final net).
