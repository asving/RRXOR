# PREREG 5 — Correction-conditional belief geometry (the "shortest path" test)

Date: 2026-07-09. Written BEFORE running any probe-trajectory analysis.

## Claim under test

A net's representations are the minimal sufficient statistics of its stagewise
CORRECTIONS, not of the task: at each developmental stage k→k+1, the geometry that
becomes decodable is the causal-state quotient of the residual process
c_k(x) = f*(·|x) / f_k(·|x) given the scaffold f_k — and refinements of the belief
simplex that belong to NO stage's correction never form, at ANY point in training
(all prior fiber tests were final-checkpoint only; the "never" clause is new).

The stage policies f_k and their corrections are already preregistered objects
(PREREG3 stage table for m27; PREREG for rrxor3): each stage's correction statistic is
derivable by hand and is a specific quotient of the full belief simplex.

## Method (fixed in advance)

D_ℓ(t) = held-out R² of ridge probe (all-layer concat residuals, query positions,
train/test 60/40 over sequences, λ ∈ {1,10,100,1000} chosen on a val quarter) for
latent ℓ at checkpoint t, MINUS the window baseline (same probe from 16 lags of
(x,u,m) raw features; t-independent). B=512 eval sequences. All checkpoints.
"Formatted" = excess > 0.10. "At baseline" = excess < 0.05.
Rise time τ_ℓ = first checkpoint with D_ℓ(t) > 0.5 · D_ℓ(final), defined only for
latents with final excess > 0.10.
Stage eras from each run's eval.jsonl: the step interval where the average CE
traverses [rung_{k−1} − 0.01, rung_k + 0.01] between preregistered ladder rungs.

## Latent batteries (oracle-computable; bases fixed in advance)

**m27** (belief = exact 27-hypothesis phase posterior π₂₇; bases orthonormal in R²⁷,
constant removed; band = positions 100..190):
- u — the xor candidate (local bit).
- Q3 — mod-3 quotient of π₂₇ (2 dims): stage-2 correction statistic (gate bet to det slots).
- Q9⊥ — mod-9 refinement ⊖ Q3 (6 dims): stage-3 correction statistic (xor-vs-exception rule).
- EXC — refinement inside the mod-9 cell {8,17,26} (2 dims): stage-4 correction
  statistic (XNOR vs FLIP disambiguation — det-slot rule = NESTED[i] needs i mod 3,
  plus i=8 vs {2,5} only within this cell).
- USE — refinements inside the other 8 mod-9 cells (16 dims): belongs to NO stage's
  correction. The "nothing more" latent.

**rrxor3** (belief = exact 3-hypothesis offset posterior with elimination; band 33..94):
- u; Q3 — offset posterior (3 dims); Λ — gate log-odds log((1+q_det)/(1−q_det)).

**fx0.03 / fx0.2** (belief = exact 5-state leaky filter; band 45..94):
- u, m; Λ — candidate log-odds; Q — decision quotient (qC, dC); F — fiber (A-vs-B sync
  split), the stage-free control: under leak the correction changes continuously and
  the comb's placement consumes sync, so F belongs to the (continuous) correction.

## Predictions

- **P1 (m27 order):** τ_u < τ_Q3 < τ_Q9⊥ < τ_EXC, and each τ falls inside its own
  stage era (u in 0.6931→0.6758; Q3 in →0.6386; Q9⊥ in →0.5328; EXC in →0.4621).
  Before its era, each latent sits at excess < 0.05.
- **P2 (m27 never-clause, the kill shot):** USE stays < 0.05 excess at EVERY
  checkpoint. (If EXC fails to rise in the pooled probe, re-run Q9⊥/EXC restricted to
  pre-det queries before judging — known just-in-time formatting; this re-restriction
  is allowed for P1, not for P2, which must hold in both.)
- **P3 (rrxor3 order):** τ_u inside the bet era; τ_Q3, τ_Λ inside the gated era;
  Q3 and Λ at < 0.05 excess throughout the bet era.
- **P4 (fx continuity):** D_Λ, D_Q, D_F rise smoothly (rise from 0.1→0.9 of final
  excess spans ≥ 4 checkpoints, no local drop > 0.1·final inside it) — the ramp is the
  continuous limit of the staircase; and F co-forms with Q (at every checkpoint where
  D_Q > 0.20, D_F > 0.05): no stage where the quotient exists without its
  implementational fiber.
- **P5 (endpoint consistency):** final excesses reproduce the known endpoint results
  (m27: EXC formatted, USE absent — matching the 0.36 ≈ 1/3 27-way ceiling; fx0.03:
  F excess ≈ 0.2–0.35 matching fiber 0.928 vs window 0.650).

## Amendment (2026-07-09, before any m27 net trajectory was computed)

The original EXC basis (refinement of mod-9 cell {8,17,26}) was mis-aligned: at
pre-det queries the current phase is ≡1 mod 3, so that target is identically zero on
its consumption sites (found via a zero-variance crash; only window baselines had been
seen). Corrected battery, same spirit: **EXC = refinement within the pre-exception
cell {7,16,25}** (2 dims — "XNOR-det next (q=7,16) vs FLIP-det next (q=25)", the
stage-4 correction statistic at the query that consumes it); **USE = refinements
within cells 0..6** (14 dims — exception identity carried 2–8 steps in advance, plus
the informationally-empty refinements of the xor-class pre-det cells 1 and 4; belongs
to no correction). Cell-8 refinement (post-exception trivia, window-recoverable) is
excluded from both. Also: for latents that are deterministic short-window functions
(u, m) the window-excess metric is degenerate (baseline = 1.0); those are graded on
raw R² relative to the step-0 checkpoint instead. All other predictions unchanged.

## Falsifiers

Any correction quotient formatted a full era before its stage (screens off the
correction-conditional account in favor of "task geometry forms wholesale");
USE > 0.10 at any checkpoint (net formats un-consumed refinements — the Simplex-canon
reading); fx showing a staircase in D(t) (stages are in the learner, not the task);
EXC never forming even query-restricted (final interp said the rule bit IS formatted
just-in-time — would indicate probe methodology failure, not theory failure).
