# RESULTS 11 — PCFG-4/4b (entropic Dirichlet trees), grading the amended PREREG

Task recap: depth-4 trees, vocab 4, productions P(b,c|a) ~ Dirichlet(alpha) per
parent per level — full support, genuinely graded inversions, tuple-coupled
fusion. alpha=0.3 round: see PCFG4_DESIGN.md CORRECTION (design-metric bug;
depth-1 task; one-fusion interchange unity; tuple-indexed span code).
PCFG-4b = alpha 0.05: graded inversions (maxP(a|bc)=.78, maxP(a|b)=.52) WITH
multi-fusion contextual signal (marginal-baselined gains .53/.22/.14/.06).

**P1' (endpoints + ordering) — CONFIRMED.** Dense at evalset-exact floors to
<= 0.3 mnat every class (b0 .4125/.4123, b1 .7530/.7527, b2 1.1190/1.1189, b3
1.1799/1.1796). Onsets: weak-but-real level ordering (fraction of the
marginal-to-floor range covered at step 200: b0 93%, b1 55%, b2 46%, b3 26%;
all ~converged by 2000-4000) — between PCFG-1 co-descent and PCFG-2/3 clean
sequencing, as the amended prereg predicted for hierarchical contextual gains
with single-child shadows present.

**P2' (bootstrap, 6th firing — first non-trivial multi-fusion version) —
CONFIRMED.** Hard-only (b>=2 loss) matches dense at b2 EXACTLY (1.1189 = floor)
and b3 to 0.4 mnat (1.1800/1.1796), where b2/b3 now genuinely require composed
soft inference (0.14/0.06 nats of contextual signal). Untrained classes learn
only part of the marginal skew (b0 1.25 vs marginal 0.94; b1 1.21 vs 0.97),
none of the contextual signal, and drift (unconstrained outputs).

**P4' (causal composition of graded inversions) — CONFIRMED at unity; the
dose-response dichotomy is now a replicated finding.** Support-safe donors
(tuples with eval count >= 8):
- 1-fusion (pair(4,5)->leaf6, oracle gap 0.336): tape ceiling slope 0.925/corr
  0.992; tuple-mean patch = ceiling at L0-L3.
- MULTI-fusion (->leaf8 across the tree root, gap 0.094): ceiling 0.972/0.993;
  patch = ceiling at L0-L2, decays at L3 (0.699) — the constituent is
  transported off-span by block 3 for the long-range consumer.
- ROOT-BELIEF PROPAGATION: patch the pair columns at L0 -> decoded root
  posterior at L5 shifts at slope 0.944 / corr 0.977 vs the BP-prescribed
  shift. One graded message edit propagates through three soft fusions to the
  root belief and the output at ~unity. This is the interventional proof that
  the net inverts the channel and COMPOSES the inversions (asvin's spec).
- Dose-response: sigmoid again (net mag .03/.49/.96/.98 vs BP .22/.35/.50/1.00)
  — replicating alpha=0.3. Combined with P5, this yields the round's sharpest
  statement: REPRESENTATIONAL FORMAT FOLLOWS NATURAL SUPPORT. Variables whose
  natural distribution is dense in the simplex (root posterior, ~4^8 values)
  get metric linear belief codes (probe R2 0.986; 0.939 on the high-entropy
  half; linear propagation 0.944); variables with discrete natural support
  (single-constituent messages: <=16 values) get discrete tuple-indexed codes
  with attractor cleanup between code points (dose suppressed at alpha=1/4,
  overshoot at 3/4, exact at endpoints). A metric-linear test for the latter
  is unfalsifiable at this design — dense-support interventions are the only
  clean probe of belief-metric structure.

**P5' — CONFIRMED:** root posterior (now high-variance at alpha=0.05) decodes
at R2 0.986, and 0.939 on the high-entropy half — graded across the interior,
not vertex-clustered.
