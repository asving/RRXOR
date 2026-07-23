# PCFG-4 DESIGN — entropic Bayesian tree inversion (complete design doc)

Date: 2026-07-23. Written after design, before training results. Status: runs
launched (dense GPU 0, hard GPU 3, tmux `pcfg4`).

## Question

Asvin's specification: production rules with (a) full support on all child tuples
AND (b) a genuinely entropic (not near-one-hot) distribution of children given
parent — so that the network must INVERT a soft Bayesian channel given one or both
children, and COMPOSE these graded inversions across levels, to solve the task
perfectly. PCFG-2 was deterministic (inversion = lookup); PCFG-3's noise was an
exception channel (98% of messages still near-vertex). PCFG-4 makes gradedness the
regime, not the exception: every message is interior to the simplex, everywhere.
This also gives the dose-response test (PREREG10-P4, previously instrument-limited)
natural mid-simplex support to fit a real encoder.

## Generative process (exact)

- Alphabet V = 4 at every level; binary tree of DEPTH = 4; 16 leaves per tree;
  NTREE = 6 independent trees concatenated -> L = 96 tokens; only leaves emitted.
- Per level l in {1..4}, per parent symbol a: the production distribution over the
  16 ordered child tuples is a single Dirichlet draw,
      P_l(b, c | a) ~ Dirichlet(alpha * 1_16),  alpha = 0.3,
  sampled once with numpy default_rng(seed 11) (4 levels x 4 parents = 16 draws)
  and FIXED for the whole experiment. Full support with probability 1;
  tuple-coupled (P(b,c|a) != P(b|a)P(c|a) generically): the fusion of child
  evidence does not factorize through per-child sufficient statistics.
- Root symbol uniform on {0..3}. Generation: ancestral sampling (multinomial per
  node). Code: pcfg4.py (dirichlet_tables, gen_tree4/gen_batch4).

## Position classes

Within a tree, leaf index i in 0..15: i = 0 is class "first" (fresh tree,
unpredictable beyond marginals); otherwise class b = nu_2(i) = number of tree
levels crossed between leaf i-1 and leaf i (counts per tree: b0 x8, b1 x4, b2 x2,
b3 x1). Predicting a class-b leaf requires inferring the level-b shared ancestor
from the seen prefix (upward/inverse Bayes) and propagating down (forward Bayes).
Training masks: dense = all targets; hard-only = loss at b >= 2 positions only
(within-tree indices 4, 8, 12 -> 18 of 95 targets per sequence).

## Design constants (measured before launch)

Alpha sweep (grammar seed 11; gradedness of one-level inversion under uniform
prior; MC floors, 8192 trees):

```
alpha  maxP(a|b,c)  maxP(a|b)   b0 gain  b1 gain  b2 gain  b3 gain
0.15     0.783        0.515      0.599    0.115    0.097    0.080
0.30     0.757        0.493      0.315    0.124    0.113    0.116   <- chosen
0.60     0.626        0.401      0.140    0.070    0.028    0.028
1.00     0.527        0.380      0.095    0.039    0.037    0.046
```

("gain" = ln 4 - full-BP floor for that class = learnable signal above uniform,
in nats.) alpha = 0.3: inversion graded even with both children observed (mean
best-guess posterior 0.757; one child: 0.493), while ALL deep classes keep
>= 0.11 nats of signal — 3-6x more than PCFG-2/3, because entropic tables
preserve information through composition better than peaky ones. MC floors at
alpha=0.3: b0 1.0716, b1 1.2619, b2 1.2735, b3 1.2698 (evalset-exact floors to be
recomputed at grading time on the fixed evalset).

## Oracle and validation

bp4 (pcfg4.py): generic-depth lambda/pi belief propagation with 4-vector messages
through the Dirichlet tensors; upward lam_parent(a) = sum_{b,c} P(b,c|a)
lam_L(b) lam_R(c), downward analog with sibling messages. VALIDATED against
brute-force enumeration over all latent configurations at depth 2:
max |error| = 1.1e-16 (machine precision). House rule: no probe target or
counterfactual is trusted until its oracle passes enumeration.

## Model and training

GPTV (train_multi.py): 6 layers, d = 128, 4 heads, L = 96, vocab 4, learned
positional embeddings. AdamW lr 3e-4, weight decay 0, batch 256, 60k steps,
seed 0 (same seed for both runs — only the loss mask differs). Eval every 100
steps on a fixed 4096-sequence evalset (per-class CE: ce_b0..b3, ce_first,
ce_avg); checkpoints every 100 steps to 3000, then every 250. Runs:
runs/pcfg4_dense_seed0, runs/pcfg4_hard_seed0. Data generated on the fly
(fresh sequences every step; no reuse).

## Preregistered predictions and falsifiers

- P1 (endpoints): dense reaches the evalset-exact BP floors at every class
  (tolerance 0.01 nats). Onset ordering is NOT preregistered this time: alpha=0.3
  leaves real single-child information (maxP(a|b) = 0.493), i.e. low-order
  statistical shadows exist, so we PREDICT weaker temporal ordering than
  PCFG-2/3 (possibly co-descent). Falsifier for the frame: deep classes reaching
  floor while shallow ones lag (inverted order) would break the
  gradient-accessibility story.
- P2 (bootstrap, 5th attempt): hard-only matches dense at b2/b3 within 0.02 nats
  by 60k. Expected to fire (no barrier), per PREREG7-10. Untrained shallow
  classes at or below uniform (positive transfer, the PCFG-3 full-support
  pattern), NOT anti-calibrated above uniform.
- P3 (circuit layout): layer x level ancestor-posterior probe grid on the dense
  net: decodability monotone in network layer with the root readable only in the
  deep half; if per-class descents are temporally resolvable, level-j
  decodability rises in sync with the classes that need it.
- P4 (dose-response — the centerpiece, now properly instrumented): natural pair
  messages fill the simplex interior, so a regression ENCODER E: normalized pair
  message lambda (3 free dims) -> span activations (per column, per layer) can be
  fitted on natural data with real support. Interchange: h += E(lambda') -
  E(lambda_native), donor lambda' from another sequence. Dose: lambda_alpha
  interpolated along natural segments, alpha in {1/4, 1/2, 3/4, 1}. Predictions:
  (i) interchange transfer slope >= 0.8 (vs BP-forced-message oracle) at the
  earliest patch layer; (ii) per-dose tracking: net output shift vs BP(lambda_
  alpha) shift has slope in [0.8, 1.2] at EVERY alpha. Falsifier: saturating or
  step-like dose curve (net consumes a hardened statistic, not the graded
  belief). Instrument sanity gates (must pass before P4 is graded): encoder
  R^2 >= 0.9 on held-out natural data; tape-swap ceiling slope ~1; transplant
  at the earliest layer ~ ceiling (else the class of patch is wrong, not the
  net).
- P5 (belief manifold): pair/quad/root posterior probes decode as continuous
  values (R^2 >= 0.9) uniformly across the simplex interior — graded on the
  high-entropy quantile of positions, not just near vertices (quantile-split
  R^2 reported).

## Analysis plan (at completion)

1. Evalset-exact floors (bp4 on the evalset); grade P1/P2 from eval.jsonl.
2. Layer x level x checkpoint probe grid (adapt pcfg2_layers.py to bp4
   want_path): grade P3.
3. Causal suite (adapt pcfg3_causal*.py): encoder fit + sanity gates, tape
   ceiling, transplant/class-mean/encoder-patch layer sweeps, interchange
   scoring, dose-response curves: grade P4.
4. Belief-manifold probes with entropy-quantile splits: grade P5.
5. RESULTS11.md; push everything to RRXOR/correction_geometry/; memory update.

## File map

- pcfg4.py — process, tables, generation, generic-depth BP (validated).
- train_pcfg4.py — training (dense|hard), eval logging, checkpoints.
- PCFG4_DESIGN.md — this document.
- runs/pcfg4_{dense,hard}_seed0/ — evalset.pt, config.json, eval.jsonl, ckpts/.
- Repo mirror: RRXOR/correction_geometry/ (design doc + code pushed at launch;
  results after grading).
