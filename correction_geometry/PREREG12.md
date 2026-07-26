# PREREG 12 — Two uniformity tests for the accessibility-potential theory

Date: 2026-07-26. Theory under test (MODEL_LADDER + brainstorm): circuits form by
greedy flow under A(f|Phi) = max_{g in G(Phi)} Corr^2(g(Phi), c_f), with G fixed by
the ARCHITECTURE (one block ~ degree-2 composition + content/position selection),
rate ∝ w_f * A (w_f = token count x nats at stake), learned circuits join Phi
(iteration), floor tau^2 from batch statistics. Asvin's falsifiable consequences:

## Test 1 — G-uniformity: "correlational structure present => circuit forms,
## on every task, with the same threshold"
Battery of (task, function) cells with BOTH coordinates measured: theory side
(A at Phi_0 and at Phi_0 + ground-truth prerequisites; w from the task), outcome
side (formation verdicts from PREREG1-11 gradings where available, fresh probes
where not). Prediction: formation <=> [w*A > tau^2 somewhere on an A-increasing
chain], ONE threshold across tasks; no task may need its own exception.
Critical cells: (A high, w ~ 0) must be ABSENT (reachability without weight);
(A ~ 0 at Phi_0, A high after prerequisite, w > 0) must form LATE (gating).
Falsifier: any cell formed with w*A below the shared threshold, or absent with
w*A above it at every stage; or threshold varying by task beyond noise.

## Test 2 — selection uniformity: "which circuit dominates is set by token
## weights, uniformly across content"
Transformer version of repgauge: target computable from EITHER of two redundant
source routes; vary the number of auxiliary predictions consuming each route
(m1, m2). Measure causal reliance (source patching) on each route at
convergence. Prediction: reliance ratio is a single function of the WEIGHT
ratio (m1/m2 in token counts), the same curve for different task contents
(e.g. xor-based vs lookup-based routes) and alphabets. Falsifier: content-
dependent selection at matched weights. (Runs to be launched after Test-1
pilot; grid m1/m2 in {1,2,4,8} x 2 contents, small GPTV.)

## Discipline
G fixed a priori (degree-2 + selection; no post-hoc enlargement); tau shared;
w computed from the generative model, never fitted. Pilot slice (this round):
cells rrxor3/xor, rrxor3/dead-pair-function (fresh probe), m27/xor-part,
m27/xnor-correction at Phi_0 and given phase-mod-9 (fresh regressions),
m27/never-consumed-quotient (PREREG5 verdict), pcfg4b/pair-posterior,
pcfg4b/root-posterior (eta-product A).

## Pilot slice results (same day)

| cell | A (theory side) | w | outcome | verdict |
|---|---|---|---|---|
| rrxor3 / xor(prev2) | 1 (identically in G) | large | formed, first | ✓ |
| rrxor3 / dead pair-fn (lags 3,7) | 1 (identically in G) | ~0 | probe R2 .033 vs .000 rand-init: ABSENT (fresh) | ✓ |
| m27 / xor-part | 0.19 at Phi0 (pooled-sign analytic .31) | large | formed, first | ✓ |
| m27 / xnor-correction | 0.000 at Phi0 -> 1.000 given phase-mod-9 | >0 | formed LATE, after mod-9 (PREREG5) | ✓ gating = A crossing |
| m27 / never-consumed quotient | reachable via chain | 0 | absent all 76 ckpts (PREREG5) | ✓ |
| pcfg4b / pair posterior | 1 (deg-2 of two tokens) | >0 | formed (R2 ~.99) | ✓ |
| pcfg4b / root posterior | small >0 (eta-products) | >0 | formed, slowest | ✓ |

Both critical negative cells hold: reachability without weight => absent (two
independent instances, one freshly measured). Gating = A discontinuous in Phi,
measured at 0.000 -> 1.000. No task-specific exceptions needed in this slice.
Next: (i) more cells (mess3 belief, factored cross-modes, leaky fiber — the
fx0.03 fiber FORMED, so its w must be shown >0, a sharp cell); (ii) the shared-
threshold/timing quantification (w*A vs tau pooled); (iii) Test-2 grid runs.
