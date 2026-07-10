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
