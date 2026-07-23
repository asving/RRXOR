# RESULTS 10 — PCFG-3 (full-support probabilistic RHM, eps=0.05), grading PREREG10

**P1 (ladder) — CONFIRMED.** Dense endpoints sit on the evalset-exact soft-BP floors
to <= 0.4 mnat at every class (b0 .5865/.5862, b1 1.1192/1.1189, b2 1.2614/1.2610,
b3 1.3632/1.3630). Onsets again ordered by level and overlapping, visibly smoother/
slower than PCFG-2 (b2 descends 100-8000 vs 600-2500) — noise-weakened gradients,
as predicted.

**P2 (bootstrap, 4th firing) — CONFIRMED as preregistered.** Hard-only (loss at
b>=2) matches dense at b2 to 0.3 mnat and b3 to 0.0 mnat. NEW: untrained shallow
classes are NOT anti-calibrated this time — b0 1.342, b1 1.379 vs uniform 1.386
(mild positive transfer, ~0.04 nats), vs PCFG-2's 1.41/1.48 above uniform. Full
support appears to force more type-uniform machinery (consistent with the rule:
type-uniform circuits transfer, type-conditional ones anti-calibrate).

**P3 (interchange, clean instrument) — CONFIRMED at the replacement level, with a
mechanistic surprise.** All 2048/2048 counterfactuals now grammatical (vs 366/2048
in PCFG-2 — the consistency trap is gone as designed). Tape ceiling slope 0.969;
donor-column transplant at L0/L1 = 0.965 (ceiling-exact). But the profile SHIFTED
vs PCFG-2: transplant@L2 drops to 0.582 (PCFG-2: 0.958) — a large share of the
constituent's causal influence leaves its span during blocks 0-1 (earlier
transport); and symbol-class-mean patches under-transfer even at L0 (0.838 vs
replacement 0.965) — the interface consumed downstream is FINER than the symbol
class (tuple/yield-level), because with noise the reader must map raw tuples to
soft messages itself (a hard symbol cache cannot handle non-canonical tuples).
Anti-hardening: noise robustness makes the pipeline keep finer information in
play, earlier — the opposite of collapsing beliefs to symbols.

**P4 (dose-response linearity) — PARTIAL, instrument-limited; falsifier NOT
triggered.** Scaling the class-mean patch by alpha: net response is graded and
direction-correct at every dose (slopes .09/.45/.66/.47, corrs .57-.84), NOT
step-like — a hardened-symbol consumer would show nothing until a class boundary
then jump. But the preregistered R^2>=0.9 linearity was not demonstrated: the
net/BP magnitude ratio is non-monotone (.16/.65/.86/.56), and the instrument
itself is the prime suspect — class-mean deltas are not a calibrated encoder of
the message manifold, and natural data gives almost no mid-simplex support at
this node (2.1% noise tuples) to fit one. Design note: a clean dose-response
needs a task variant with entropic constituent messages (larger eps at one
designated level, or ambiguous productions).

**P5 (graded beliefs) — CONFIRMED.** Root-posterior probe (L5, query col) trained
ONLY on canonical-parse sequences: R^2 0.988 held-out, and 0.953 on noise-tuple
sequences where NO hard parse exists — the net computes and carries the continuous
soft-BP belief, extrapolating correctly off the deterministic-parse manifold.
