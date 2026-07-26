# MODEL_LADDER — asvin's geometric-dilution learning model, formalized and fit

## The model (as formalized from asvin's description + mid-analysis amendment)

Level-r latents (order r = class b+1) are learned at a rate set by (i) the
per-token contextual information available for their consuming class, g_b =
marginalCE - floor (evalset-exact, marginal = position-specific BP with zero
evidence — the alpha=0.3 lesson baked in), and (ii) an ACCESSIBILITY factor
carrying the multi-path structure (deep latents receive signal from tokens AND
from every lower-order latent as it consolidates). Data-processing constraint:
latent routes cannot add information (all information enters through tokens),
so the routes live in accessibility, not the budget:

    dc_r/dt = eta * g_r * [w0 + (1-w0) * c_{r-1}(t)] * (1 - c_r),   c_0 = 1.

w0 = direct-from-token accessibility. w0=1: pure parallel race paced by
dilution. w0->0: asvin's original renewal picture ("functionally less deep"
after each stage). Token-cost prediction: tau_b ∝ g_b^(-kappa); asvin's
"square factor" = kappa=2 if g is read in nats, OR kappa=1 with the square
already inside g (information ~ amplitude^2; per-level amplitude contraction
lambda => per-level token cost multiplies by lambda^(-2kappa)).

## Data: (g_b, tau_b) across the family (dense runs, batch 256, same arch)

pcfg2 (deterministic RHM): g = .766/.402/.118/.022; tau50 = 62*/229/737/1082
pcfg3 (eps=.05):           g = .591/.231/.050/.011; tau50 = 53*/246/1067/(b3 no instrument)
pcfg4 (alpha=.3):          g = .249/.031/0/0;       tau50 = 50*/155/-/-
pcfg4b (alpha=.05):        g = .526/.213/.142/.058; tau50 = 49*/190/293/598
(* = censored: at/below first eval at step 100.)

## Findings

1. **Within-task power laws are clean; kappa is ~0.5-1.2, decisively NOT 2.**
   pcfg4b: kappa(tau50 vs g) = 0.87, r = -0.997; pcfg2: 0.52, r = -0.93. The
   square-in-nats version is excluded; the info-budget version (kappa=1) fits
   the entropic task at onset/midpoint. Asvin's square survives as
   amplitude->information: tokens ∝ 1/g and g contracts as amplitude^2 per
   level, so the per-level token multiplier is contraction^-2 — geometric
   series in tokens, ratio = (g-dilution ratio)^kappa. Measured tau ratios
   compound multiplicatively exactly as g-ratios^kappa (pcfg4b: g-ratios
   1.5/2.4 -> tau ratios 1.5/2.0 at kappa .87).
2. **kappa is quantile-dependent — the fingerprint of the multi-path
   amendment.** pcfg4b kappa = 1.17 (onset, q=.2) -> 0.87 (q=.5) -> 0.51
   (completion, q=.8); pcfg2: .63/.52/.43. Onsets are paced by the diluted
   direct signal; completions are COMPRESSED relative to any single-exponent
   law — deep classes finish faster than g-pacing predicts (b3 tau80 measured
   1000 vs parallel-model 1500), consistent with late help through
   consolidated lower latents.
3. **Sequentiality: parallel, not renewal, in 4/5 task-class comparisons**
   (deep-class onsets precede shallow-class completion everywhere except
   pcfg2 b2-vs-b1, which is marginal). Strict renewal is wrong for this
   family.
4. **The ODE fit splits the family along w0 = the leap axis.** pcfg4b
   (single-child shadows exist, maxP(a|b)=.52): best w0 = 1.00 — pure
   parallel, rmse .042, tau50 predictions 100/200/300/700 vs measured
   100/200/300/600. pcfg2 (leap-hard): best w0 = 0.02 — bootstrap-dominated,
   BUT the fit is poor in both limits (rmse .11; predicted b1/b2 far too
   fast): for leap-hard features, rate ∝ g overestimates the direct route —
   the information exists but not in gradient-accessible low-order form; the
   gated case needs a structural accessibility discount (the
   manufactured-gradient mechanism), not a scalar w0.
5. Synthesis with RESULTS9's two-mechanism taxonomy: parallel-race-paced-by-
   dilution (w0~1, kappa~1) = full-support/shadowed tasks; renewal/gating
   (w0~0 + thresholds) = leap-hard tasks. Asvin's original renewal story and
   the parallel race are the two ends of the measured w0 axis.

## Caveats and the decisive follow-up

Eval cadence (100 steps) censors b0; coverage-as-consolidation is a
behavioral proxy for latent quality; rmse underweights the tails where the
multi-path signature lives; cross-task pooling needs per-task offsets.
Decisive version: measure c_k(t) DIRECTLY with ancestor-posterior probes on
the dense checkpoints (every 100 steps) and test rate_r(t) against
w0 + (1-w0)c_{r-1}(t) pointwise rather than via integrated curve fits.

Files: ladder_model.py (extraction+fits), ladder_model2.py (quantile
robustness, contraction check — crude SVD contraction inconclusive, needs a
proper chi^2 metric), ladder_ode.py (ODE fits), ladder_model.png.
