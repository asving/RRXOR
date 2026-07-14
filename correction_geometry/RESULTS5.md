# RESULTS 5 — Correction-conditional belief geometry (grading PREREG5)

Runs: seed0 (rrxor3), m27_nl6_seed1, fx0.03_seed0, fx0.2_seed0. Artifacts:
`corrprobe.py` (trajectories), `corrprobe_plots.py`, `corrprobe.png`,
`corrprobe_summary.json`, per-run `corrprobe.json`. Metric: D_ℓ(t) = held-out R² of
all-layer-concat ridge probe minus window-16 baseline (u, m graded on raw R² per the
preregistered amendment). Stage eras: m27 from ladder-rung traversals of eval CE
(era1 ~620; era2 2200–2400; era3 2600–3200; era4 3600–5200); rrxor3 from the devtomo
change-point windows (bet 50–550; gated 550–1450).

## Grades

**P1 (m27 order) — PASS on order and onsets, with a mechanistic qualification.**
u: τ=650, in era1. Q3: excess jumps −0.39 → −0.07 → +0.19 across ckpts 2000/2400/2800,
i.e. the rise begins exactly on the era-2 traversal (2200–2400) at checkpoint
resolution; absolute 0.10-crossing at 2800 (era3's first ckpt). EXC (which exception
comes next, the stage-4 statistic): pooled onset 5600, one checkpoint after era4 ends;
final +0.36. Order u < Q3 < EXC holds strictly. Qualification: the 6-dim mod-9
refinement Q9x NEVER becomes pooled-decodable (final −0.03 pooled, −0.24 pre-det), and
the 1-bit stage-3 statistic g9 ("next det is exception-class") reaches only +0.12 at
step 12000+ — yet the behavior is solved by 3200 and the earlier whitebox found the
rule bit at 0.957 *per-position at L6*. So the stage-3 correction is implemented in
attention WIRING (the mod-9 lag lattice) plus a just-in-time position-local code —
essentially invisible to pooled linear probes. Correction statistics can be absorbed
into weights rather than formatted as stream variables; probe trajectories see only
the latter.

**P2 (m27 never-clause) — PASS, decisively.** USE (14-dim refinement belonging to no
stage's correction: exception identity carried 2–8 steps in advance + the empty
refinements of the xor-class pre-det cells) stays BELOW baseline at every one of 76
checkpoints: max excess −0.095 pooled, −0.57 pre-det-restricted. The phase is one-hot
inferable from context throughout (any of these quotients is decodable-in-principle);
the net formats none of the unconsumed ones, ever. Caveat: pooled + pre-det probes
only; a per-position sweep was not run.

**P3 (rrxor3 order) — PASS on order, FAIL on the timing clause, informatively.**
u: τ=325, inside the bet era. But Q3 onset (>0.10) is 2000 and Λ onset 2500 — after
the behavioral gated era (550–1450). The net *behaves* gated ~500–1000 steps before
its offset posterior becomes linearly decodable in excess of the window baseline, and
the code keeps sharpening long after behavior converges (Q3 +0.19 at 2800 → +0.63 at
8000; same pattern in m27's Q3 +0.19 → +0.58). Linear stream-formatting LAGS behavior:
circuits work distributed/nonlinear first and consolidate into linear codes during the
post-convergence plateau.

**P4 (fx continuity) — smoothness PASS, co-formation NARROW FAIL.** Both fx runs rise
smoothly (0.1→0.9 of final spans 9 and 14 checkpoints; zero local drops > 0.1·final):
no staircase — stage discreteness is a property of the task's uncertainty structure,
not the learner. Co-formation clause: 4 checkpoints (~2750–3500, fx0.03) have Q > 0.20
with F ≤ 0.05 — the decision quotient leads the sync fiber by ~750–1000 steps before
both climb (F final +0.297). Consistent with the fiber being the *implementational*
demand of the lengthening comb, though the horizon-tracking version of that claim was
not cleanly testable (early W90 is noise-inflated).

**P5 (endpoint consistency) — PASS.** fx0.03 fiber final excess +0.297 ∈ predicted
[0.2, 0.35]; m27 EXC formatted (+0.36) while USE absent, matching the known 27-way
probe ceiling ≈ 1/3. Side-prediction confirmed: fiber excess orders with leak,
F(ε=0.03)=+0.29 > F(ε=0.2)=+0.08.

**Falsifiers: none triggered.** No quotient formed early (the anti-claim "task
geometry forms wholesale" is dead: every latent sits at baseline until its era); USE
never exceeded 0.10 (Simplex-canon reading dead on this net); no fx staircase; EXC did
form.

## Reading

The correction-conditional account survives its kill shot and gains two refinements:

1. **Which belief geometries exist = which corrections needed them.** Every formatted
   statistic maps to a stage's correction; every never-needed refinement of a
   perfectly-inferable latent stays unformatted across all of training.
2. **"Formatted" has two substrates.** A correction statistic can live as a linear
   stream variable (Q3, EXC, fx quotient/fiber) or be absorbed into attention wiring +
   just-in-time local codes (m27's mod-9 alignment). The shortest-path claim should be
   stated over both substrates; probe trajectories alone under-count.
3. **Consolidation lag.** Behavior precedes linear decodability by ~0.5–1k steps and
   the codes keep sharpening long after the loss converges — the "geometry of the
   space of predictive networks" has a second, slower timescale (linearization /
   consolidation) on top of the loss descent, and endpoint-only probing conflates the
   two.

## Addendum: correction spectroscopy (corrspec.py / corrspec.png)

Direct, hypothesis-light version suggested by asvin: measure the likelihood-ratio
adjustment δ(x) = λ_post − λ_pre across each sharp loss drop and ask what history
quotient it factors through. RRXOR-3 drop 1 (steps 200→400): E[δ | each of the 16
length-4 suffixes] collapses onto two bands indexed by ⊕(last two) — u explains 94% of
between-suffix variance; the entire suffix table matches u-only R² (0.70 vs 0.67):
the first correction ε-machine IS the 2-state ⊕ machine, read off the data. Drop 2
(1500→3000): no suffix function explains it (16-suffix table R² 0.25); the stage-2
statistic (candidate × offset posterior) reaches held-out R² 0.85 (u-only 0.20).
Matryoshka: regressing each of the four measured drops on each stage's closed-form
theoretical correction gives a diagonal-dominant 4×4: diag 0.90 / 0.74 / 0.86 / 0.84,
off-diagonal ≤ 0.34 (drop-3-on-δ*₂ = residual mod-3 gating completing in the adjacent
window). Each drop implements exactly one stage's correction and no other.

## Addendum 2: spectral reconstruction of the HMM from behavior (hankel.py / hankel_fig.png)

Asvin's proposal: rebuild the machine from the net's conditionals via the Hankel route
(spectral learning of observable-operator models). H[h,s] = P_net(s | h) computed
exactly by teacher-forcing all test suffixes |s| ≤ 4 over 4096 histories at position
60; SVD → states; shifted-state regression → operators A₀, A₁. Results (rrxor3):
the true generator has Hankel rank exactly 5 and transfer eigenvalues
{1, e^±2πi/3, 0, 0}; the final net's behavioral Hankel matches the true singular-value
spectrum to 3 decimals and its transfer operator carries the clock at 0.998∠2.0944
(target 2π/3 = 2.0944); the pre-drop-2 net has neither (different spectrum, no clock).
The rank-5 OOM rolled autoregressively tracks the net at |Δp| 0.000–0.003 over 12
steps beyond its 4-step fitting horizon. Trajectory across checkpoints: effective rank
1–2 (iid) → 3–4 (order-2 bet) → overshoot to 9 mid-transition → compression to exactly
5; the clock eigenvalue (angle-filtered) is installed across drop 2, magnitude → 1.000.
Method pitfalls hit and fixed: tests ≤ 3 are too short — the rank-4 shadow collapses
the rotation pair onto its real part (eigenvalue −0.5 = cos 2π/3); fitting k above the
true rank injects a null-space dimension that scrambles the spectrum.

## Addendum 3: the root-of-unity tower (hankel27.py / hankel27_fig.png)

Same spectral reconstruction on matryoshka. True generator: Hankel rank exactly 45
(the predicted 9×(1+2+2) machine) with ALL 27 transfer eigenvalues on the 27th-root
lattice (angles k·2π/27, k=−13..13, integer-exact). The net's behavioral machine fills
the unit circle as the nested tower predicts: after drop 1, no clock (near-unit mass
at angle 0 only); after drop 2, the CUBE roots (clusters at 0, ±2π/3); after drop 3,
exactly the NINE ninth roots {0,±3,±6,±9,±12}·(2π/27); final, all 27 — matching truth.
Dense trajectory staircase: 1 root class (steps ≤2200) → 3 (2400, on the era-2
traversal) → 7 (mid-era-3) → 9 (3200–5000) → 27 (5600+), with the shifted-state
operator fit R² dipping mid-drop (0.946–0.965: behavior between machines) and
returning to 0.9996–1.0000 on every plateau. The Pontryagin dual of the Z27⊃Z9⊃Z3
quotient tower appears rung-by-rung in the spectrum of pure behavior. Method notes:
long tests (24 tokens, generator-sampled) are mandatory — short tests see only a
rank-5 shadow; per-test column normalization is required or the tiny long-test
probabilities are numerically invisible; count root classes by snapping near-unit
eigenvalues to the lattice.

## Addendum 4: how a rung is installed — the continuous picture (hankel_flow.py / eigflow.py → eigflow.png)

Transfer-operator spectrum at EVERY rrxor3 checkpoint (25-step resolution through the
transition). The installation of the clock has two phases: during the behavioral drop,
transient complex pairs flicker at wandering angles and magnitudes ≤ 0.4 (the fitted
operators are between machines); then the pair LOCKS ONTO THE ANGLE 2π/3 (~step 2400)
and ramps in MAGNITUDE 0.4 → 1.00 over steps 2500–5000 — mostly AFTER det-CE has
converged (~2600). In the complex plane the route is radial: the pair appears on the
2π/3 ray at small radius and walks straight out to the unit circle. Since |λ| is the
per-step survival of phase information (memory horizon −1/ln|λ|), the ramp is the
net's phase-memory horizon diverging from ~1 step to ∞ — the spectral form of the
consolidation lag: frequency is learned with the behavior, fidelity is consolidated
after it. Matryoshka cohorts (from stored eigenvalues): each root cohort's magnitude
rises during its own drop (cube → ninth → fine 27th), with the cube cohort transiently
OVERSHOOTING |λ| > 1 mid-drop (non-contractive fitted operator; fit R² dips there —
reorganization is briefly not a stationary machine at all).

### Correction to Addendum 4 (2026-07-10)

A convention bug in `hankel_flow.py` (operator regression weighted by the conditional
AFTER σ instead of P(σ|h); `hankel.py`/`hankel27.py` were unaffected — verified by
grep and by their independently-checked endpoint numbers) inflated the apparent
post-behavioral share of the magnitude ramp. Corrected flow: flicker phase unchanged
(transient pairs, fit R² 0.93–0.97); angle locks ~2400; |λ| reaches 0.94 by 2500 and
0.98 by 3000 (i.e. the bulk of the magnitude installs WITH the drop's completion);
the final 0.98 → 1.000 approach over 3000–6000, with fit R² 0.994 → 1.0000, is the
post-convergence consolidation. The two-phase structure and the flicker→lock→ramp
sequence stand; the consolidation-lag component is the final approach to losslessness,
not the bulk of the ramp.

## Addendum 5: Jasmina's questions (hankel_qa.py, SPECTRAL_METHOD.md)

**Q1 (does the reconstructed machine predict the net at least as well as the true
ε-machine?)** KL(net‖model) per context position, teacher-forced 32-position rollout:
at the FIRST position the reconstructed OOM beats the true machine (rrxor3 1.8e-5 vs
1.6e-4; fx0.03 2.1e-4 vs 1.6e-3) — it captures the net's own miscalibrations, which
the true machine cannot; autoregressive drift crosses over at ~5 positions; 32-step
means 7.5e-4 vs 2.6e-4 (rrxor3), 6.1e-3 vs 1.4e-3 (fx). All ≪ the ~0.5-nat entropy
scale. Rollouts require per-step state renormalization (p(0)+p(1)=1).

**Q2 (leaky RRXOR):** closed-form prediction confirmed exactly — the teleport makes
the transfer spectrum {1, (1−ε)e^{±2πi/3}, 0, 0}: reconstructed true-table clocks at
0.990/0.970/0.800∠2.094 for ε=.01/.03/.2. The NETS sit slightly inside their true
radii (0.983/0.947/0.737): they forget marginally faster than Bayes — the spectral
form of their CE gap. The forgetting-comb envelope (F1) and the clock radius are the
same quantity in two presentations. Rank stays 5: the fractal belief SET is a
continuum but its linear span is 5-dim, and the Hankel sees the span.

**Q3 (the correction as its own machine):** the ratio c = P_post/P_pre is
multiplicative along paths, so the same spectral pipeline applies with the one-step
ratio as the weight. Against the IDEALIZED bet policy (q̂ = 0.665 fitted) the
correction-WFA fits at R² 0.95 and its dominant eigen-triple is {1.97, 1.955∠2.095} =
a common growth factor × the cube-root clock: the clock lives inside the correction.
Against the ACTUAL mid-training pre-net it degrades to R² 0.83 — reciprocals of
non-local policies are not low-rank (documented caveat).

## Addendum 6: the truncation puzzle (hankel_trunc.py)

Jasmina's report: rank-3-truncated Hankel of the trained net reconstructs a much worse
machine (vs ground truth) than the rank-3-truncated true generator, despite the net
being at the Bayes floor. **Does not reproduce with exact-conditional Hankels**: at
every cutoff k ∈ {2..6}, net-truncated and true-truncated agree to 3–4 decimals — CE
vs truth 0.6853/0.6852 (k=2), 0.5967/0.5967 (k=3), 0.5394/0.5393 (k=4), 0.4629/0.4621
(k=5); KL(true‖machine) likewise; identical fit R² (0.875 at k=3); principal angles
between top-3 subspaces ≤ 1.7°; both truncated transfers stable. Likely sources of the
observed asymmetry in other pipelines: Hankels ESTIMATED from finite samples (noise
fills the tail and rotates the top-k subspace — especially if the generator's Hankel
is analytic while the net's is sampled), FREE-RUNNING generation from truncated
machines (they can emit unnormalized probabilities and diverge; ours is teacher-forced
with per-step renormalization), or short tests.

Two conceptual byproducts. (1) Truncation quality is a property of the behavioral
function alone: the net implements the machine via the xor-trick + JIT combs and never
carries the 5-dim recurrent state internally, yet its truncation ladder is identical
to the generator's — internal circuitry is invisible to this operation. (2) SVD
truncation is NOT belief-coarsening: at k=3 the truncated transfer has NO clock
({0.999, 0, 0} — the rotation lives substantially in the discarded directions), and
the truncated machine's CE (0.597) BEATS the developmental mod-3-gate rung (0.639).
The L2-optimal rank-3 machine is a better predictor than SGD's rank-3-ish stage:
gradient descent's ladder is built from gradient-accessible corrections, not from
optimal low-rank approximants, and the truncation ladder of the final machine
(0.685/0.597/0.539/0.462) does not retrace the developmental ladder.

## Addendum 7: extracting the phase-belief simplex by transfer-eigenspace projection (eigproj.py → eigproj.png)

SVD truncation kills the clock (Addendum 6); spectral projection keeps it. Construction
(no probes, no activations, no ground truth in the loop): fit the behavioral machine at
one position, eigendecompose the transfer T = A₀+A₁, and for every history form the
projective clock coordinate w(h) = ⟨l_ω, z(h)⟩ / ⟨l_1, z(h)⟩ (left eigenvectors of the
rotation and stationary modes). Certification against the exact elimination filter:
ONE affine map takes the offset posterior onto the clock plane at ALL context lengths
— pooled R² 0.9987 forward / 0.9989 reverse, per-position 0.992–1.000 including
pre-sync lengths 4–8 where beliefs are interior mixtures. The plane displays the MSP
sync tree: vertices (synced), uniform center, (1:1) and (2:1) edge points, (2:1:1)
interior points, populated exactly at the filter's reachable beliefs. The spectral
factorization is exact: posterior from the nilpotent sector R² 0.0003; candidate bit
from the clock plane R² 0.000; candidate bit from the nilpotent sector at pre-det
queries R² 1.0000. Eigenvalue magnitude = memory lifetime: the |λ|=1 rotation pair
carries the phase simplex (kept forever), the λ≈0 modes carry the pending bits (die
within a block), disjointly.

## Addendum 8: the fractal attractor from behavior (eigproj_fx.py → eigproj_fx.png)

Same eigenspace projection on the leaky nets. Clock eigenvalues: 0.958∠2.095 (ε=0.03,
1−ε=0.970) and 0.764∠2.090 (ε=0.2, 1−ε=0.800) — damped clocks, nets slightly lossier
than Bayes as before. The clock plane is an affine image of the exact leaky CLASS
posterior (A/B/C): R² 0.9889/0.9885 both directions at ε=0.03, 0.9308/0.9155 at ε=0.2;
factorization exact again (class←nilpotent ≤0.001; pending←nilpotent 0.99;
pending←clock ≤0.002). Plotted in the barycentric frame, the extracted set reproduces
the leaky filter's FRACTAL attractor: near-vertex filaments at ε=0.03; the contracted
self-similar spike structure at ε=0.2 — with the finest filaments blurred at ε=0.2
(the net's behavioral calibration noise sets a resolution floor on attractor
recovery). Forgetting = spectral contraction: more leak ⇒ smaller clock radius ⇒ the
attractor hugs the uniform center.

## Addendum 9: which eigensector? — sector ablation recovers the loss ladder (hankel27_sector.py)

Answer to "how do you know which eigensector to project onto" without ground truth:
the spectrum PROPOSES (conjugate closure; magnitude bands = memory timescales;
arithmetic closure of unit-circle angles = the subgroup lattice of the rotation group,
here {1} ⊂ 3rd ⊂ 9th ⊂ 27th roots), and sector-ablation behavior DISPOSES. Projecting
the k=45 matryoshka machine onto each candidate sector (always retaining the |λ|<0.8
transients, which carry the pending bits every stage needs) and rolling the ablated
machine teacher-forced:

  {1} + transients          → CE 0.6672   (bet rung 0.6758)
  + 3rd-root sector         → CE 0.6325   (mod-3 rung 0.6386)
  + 9th-root sector         → CE 0.5304   (mod-9 rung 0.5328)
  full spectrum             → CE 0.4624   (floor 0.4621)

The developmental loss ladder is recoverable from the FINAL machine by spectral-sector
ablation, each ablation landing within 0.002–0.009 of its rung (slightly below the
closed-form policy values — the L2-fitted sector machines are marginally better than
the idealized stage policies). Contrast Addendum 6: SVD truncation gives
0.685/0.597/0.539 — NOT the rungs. Eigensectors are the rungs' geometries; singular
directions are not. With training checkpoints the identification is even more direct:
each sector is BORN at its loss drop (Addendum 3 trajectory), so the developmental
record labels the sectors by their corrections.
