# PREREG 13 — Test 2 (selection uniformity), Codex-hardened design

Date: 2026-07-26. Written before grid completion (launch same day; smoke test at
(1,1) only — all accuracies observed, no grid outcomes seen). Codex adversarial
review (~/tmp/test2_design.txt exchange) reshaped the design; key adopted points:
(i) two lookup contents related by exact dataset symmetry are a SANITY CONTROL,
not a test — any theory predicts their collapse; (ii) the DISCRIMINATOR is the
lookup-vs-XOR arm: frequency-only theories predict NO shift at matched item
counts, the accessibility theory predicts reliance shifted toward the faster
route by its standalone-speed ratio; (iii) operational A_i = single-route
learning speed from control runs (measured, not assumed); (iv) trajectories
logged (eval/50 steps), not just endpoints; (v) shuffled item order, disjoint
vocabularies, decoy items padding all sequences to 10 items; (vi) conflict
trials at EVAL only (never trained); (vii) competence checks via aux accuracies.
Deferred (noted): integrated-drive path dependence; >2 seeds; held-out-content
curve prediction.

## Task
Sequences = 10 shuffled items of [source(s), marker, answer]; z binary.
Route-1 = lookup {0,1}; Route-2 = lookup {2,3} (arms ll1/ll2, different perms) or
XOR of two tokens {4,5} (arm lx). One shared item (both sources, same z; both
routes individually sufficient). Answers {6,7}; markers 8/9/10/11; decoy src 12.
GPTV nl=4 d=128 L=44 vocab16, 4k steps, batch 128, dense CE.
Grid: (m1,m2) in {(1,1),(1,2),(1,4),(1,8),(4,1)} x arms {ll1,ll2,lx} x seeds {0,1}
+ single-route controls (ll:4 items, x:4 items) x 2 seeds = 34 runs.

## Measurements
pref1(t) = P(shared answer follows route-1 | conflicting sources), trajectory;
corruption deltas at end; aux accuracies (competence gates: both routes must be
at ceiling for a run to count).

## Predictions
- P1 (symmetry sanity): ll1 and ll2 pref1-vs-(m1/m2) curves coincide; pref1 at
  (1,1) = 0.5 averaged over seeds.
- P2 (weight monotonicity): pref1 decreases in m2/m1, same curve in both ll arms.
- P3 (THE DISCRIMINATOR): in arm lx at (1,1), pref1 > 0.5 (reliance biased
  toward the lookup route) by an amount consistent with the measured single-route
  speed ratio tau_x/tau_l from the control runs; frequency-only null predicts
  pref1 = 0.5. At larger m2 the xor route's aggregate drive compensates:
  the lx curve = the ll curve SHIFTED in effective weight by the speed ratio.
- Falsifiers: ll1/ll2 curves differing beyond seed noise (content-dependence at
  exact symmetry = instrument broken); lx pref1 at (1,1) = 0.5 with a real
  measured speed gap (accessibility does not enter selection); non-monotone
  pref1 in m2 (weights do not enter selection).
