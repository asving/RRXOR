# PREREG 7 — Loss placement and the bootstrap: last-token-only vs dense supervision

Date: 2026-07-21. Written BEFORE training. Motivated by Cagnetta & Wyart (arXiv
2406.00048): hierarchical latents live in progressively longer-range correlations, and
easy-position supervision can build intermediate variables that serve as compressed
context for harder positions. Asvin's proposed test: train with loss ONLY at the last
token position vs at all positions, matched for the number of per-position training
instances, and ask whether last-position prediction is learned worse without the dense
scaffold.

## Design

Same-seed controlled pairs — identical architecture, init, and data stream; only the
loss mask differs:
- rrxor3 (2-stage tower): baseline runs/seed0 (all-position, 8k steps) vs
  rrxor3_last_seed0 (24k steps).
- m27 nl6 (4-stage tower): baseline runs/m27_nl6_seed1 (16k) vs m27_last_seed1 (48k).
- mess3 (no tower): baseline runs/mess3_seed0 (8k) vs mess3_last_seed0 (24k).
Loss mask "last" = CE at the final target position only (95 / 191 / 95). Metrics: CE
at the last position (recomputed per checkpoint for the baselines), per-class at the
last position for m27 (rnd / xor / xnor / flip by the last position's phase). Curves
compared on two x-axes: cumulative position-instances (steps × batch × #targets:
dense = 95 or 191 per sequence, last-only = 1) and steps (≈ FLOPs, since forward cost
is identical).

## Predictions

- **P1 (penalty ordering).** At matched position-instances, last-only lags dense
  supervision at the last position, with the gap ordered by correction-tower depth:
  m27 ≫ rrxor3 > mess3 ≈ 0. The dense losses' role: they build the per-position
  scoreboard codes and stage circuits whose corrections manufacture the later
  gradients; last-only must construct all source-side structure through attention-
  mediated credit assignment from a single query position.
- **P2 (rung stalling, m27).** Within the 48k budget the last-only net stalls at an
  earlier rung of the ladder AT the last position (classes resolve in order; the later
  corrections are delayed disproportionately). Honest counter-hypothesis, stated in
  advance: the manufactured-gradient cascade operates AT the last position itself
  (bet → gate → rules are all consumable there), and attention credit assignment may
  suffice to build source-side codes — in which case matched-instance curves coincide
  and the bootstrap-via-dense-loss story is NOT load-bearing for tape-visible tasks.
  Either outcome is informative; the falsifier discipline below adjudicates.
- **P3 (machine-ness).** The last-only net is not a stationary behavioral machine:
  its conditionals at never-trained positions are uncalibrated, so the spectral
  pipeline's operator-fit R² at mid-sequence positions collapses relative to the
  trained position. Dense supervision is what makes the net a machine at every
  position.
- **P4 (mess3 null).** No matched-instance penalty for mess3 beyond noise (shallow
  filter, no compositional scaffold to bootstrap).

## Falsifiers

P1 dies if the matched-instance gaps are unordered or ~equal across tasks (penalty
would be generic sparse-supervision optimization cost, not compositional). The
bootstrap story dies if m27-last-only matches its dense twin at matched instances.
P4 dies if mess3 shows a gap comparable to rrxor3's.

## Note for follow-up (not this round)

Our tasks are tape-visible (every useful intermediate value appears as a token), so
they test the credit-assignment half of the paper's story. The sharp test of
"context added at depth" needs a task whose intermediate variables NEVER surface on
the tape (PCFG nonterminals proper) — planned as the next round if P1 confirms.
