# Results vs PREREG2.md (Tests A, B, C)

20 new runs (Test A: p in {0,.1,.2,.3,.4,.5} x 2 seeds; Test B: eps in {.02,.05,.1,.2} x 2
seeds) + Test C on existing baseline seed-0 checkpoints. Figures: tauA.png, routeB.png,
runs/xp00_seed0/probe_heatmap.png, runs/seed0/msp_r2.png, runs/seed0/msp_unfold.png.
Analysis tables: analysis_tauA.json, analysis_routeB.json.

## Test A — scaffold dose-response

Data (per-seed, from analysis_tauA.json):
- excursion peak (ce1): p<=0.3: 0.697-0.704 (NO excursion above ln2); p>=0.4: 0.766-0.777
  (full baseline-size excursion; baseline 0.768-0.775).
- tau_solve (ce2<0.2): p=0: 920-1050; p>=0.1: 1550-1850 (~= baseline 1350-1650).
- tau_gate (pxor split >= 0.15): 830-1200 across ALL p including baseline — roughly flat.
- additivity: tau_gate(baseline)=943 vs tau_xor(baseline)+tau_gate(p=0)=343+865=1208.

Verdicts:
A1 as stated (excursion onset shifts earlier, tau_gate monotone in p): WRONG, in a way that
STRENGTHENS the frame. With an informative input bit (p<=0.3) the excursion doesn't shift —
it VANISHES. The miscalibrated stage-1 only exists when the value arrives well before any
usable gate evidence. The input bit provides shallow gate evidence (violation indicators one
attention hop away), closing the serial gap, so the pathology never forms. At p>=0.4 the bit
is too noisy to serve as evidence and the full baseline pathology returns.

NEW DISSOCIATION (not preregistered): the scaffold's two roles separate in the dose-response.
As EVIDENCE for the gate it tolerates heavy noise (works at p=0.3); as the VALUE wire it
tolerates none (tau_solve only accelerates at p=0 — for p>=0.1, ce2<0.2 exceeds the bit's
information (H(p)>=0.325), so the net rebuilds the internal xor circuit on the baseline
schedule; all runs end at avg 0.4621-0.4625).

A2 (depth moves with the scaffold): PARTIAL. Phase becomes decodable earlier (~step 900 vs
~1500-2500) and one layer shallower (L3 reaches 0.97; baseline L3 maxed ~0.5), but does NOT
sink to L1 even though one attention layer could aggregate the given bits. In both runs
phase appears at the LAST layer first and spreads backward. Amendment: dependency sets a
LOWER bound on depth; placement gravitates toward the consumer (just-in-time before the
unembed), not toward the inputs.

A3 (tau additivity): FAILS in the overlap direction — baseline gate forms FASTER than
xor-time + seeded-gate-time (943 < 1208). Stages PIPELINE: the consumer starts binding to
the scaffold's partial signal while the scaffold is still forming. Strict seriality
(tau_composite = max(parts) + binding) is an overestimate when the part's output is usable
mid-formation.

## Test B — two-route selection (flat cue vs hierarchical violations)

Data: OOD (eps=0 at test) ce2 = 0.0001-0.0002 and pxor2 = 1.000 for EVERY training eps
including 0.2. Excursion peak: eps<=0.1: 0.773-0.785 (>= baseline); eps=0.2: 0.698-0.702
(vanished).

B1 (sharp route flip in eps): NOT REACHED in this range — every net, even eps=0.2, retains
a fully functional violation route (perfect phase OOD with the cue removed). The violation
route is asymptotically dominant here (resolves phase in ~5-10 tokens vs ~30+ for the cue),
so the cue never displaces it. Inconclusive on the flip itself; flipping likely needs
crippled violations or a far stronger cue.

B2 (excursion shrinks with eps): CONFIRMED, in threshold form — unchanged (slightly up) for
eps<=0.1, gone at eps=0.2.

DISCOVERY: at eps=0.2 the cue changes the learning ORDER (phase available early -> no
miscalibrated stage) without changing the FINAL mechanism (violation route still fully
learned; cue redundant at late positions). Path changed, endpoint preserved: early phase
access removes the stage-1 pathology, and the violation evidence is later hooked into the
already-existing phase variable. The "manufactured gradient" (miscalibration pain) is ONE
bootstrap path to phase, not the only one.

## Test C — coarsening trajectory: CONFIRMED

Ridge regression resid(L4) -> ground-truth Bayes beliefs, held-out seqs. Phase-marginal R^2:
~0.00 at step 400 (mid-plateau, xor already probe-decodable at 0.88-0.97) -> 0.20 at 1500
-> 0.48 at 2500 -> 0.85 at 8000. The simplex scatter (msp_unfold.png) unfolds from a
color-mixed central blob (the phase-marginalized quotient) to three clean phase clusters.
C1 confirmed (quotient during plateau); C2 confirmed with two refinements: R^2 starts
creeping ~step 500-800 while the loss is still plateaued (hidden progress precedes
behavioral change), and the geometry keeps consolidating (0.6 -> 0.85) after the loss has
converged.

## Frame updates from this round

1. The non-monotonic excursion is the signature of the SERIAL GAP between a feature and its
   gate. Close the gap from either side (early evidence via input bits or via a cue) and the
   intermediate pathology never exists. It is not a necessary stage of hierarchy — it is a
   symptom of asynchronous arrival.
2. Serial enablement is PIPELINED, not sequential: consumers bind to half-built scaffolds.
3. Depth placement: dependency is a lower bound; circuits otherwise form just-in-time near
   their consumer.
4. A scaffold's evidence role and value role dissociate under noise (evidence robust to 30%
   corruption; value substitution requires exactness).
5. Belief geometry is learned literally coarse-to-fine (quotient -> full simplex), with
   representational progress leading behavioral progress.

Caveats: 2 seeds per sweep config; Test B never reached the route boundary; tau_gate is a
noisy metric (flat across p within seed noise).
