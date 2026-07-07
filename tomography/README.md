# Developmental tomography: tasks and circuits from checkpoints alone

An automated pipeline that recovers a network's task decomposition AND its circuit
anatomy purely from training-time data — no labels, no human guesses about how the
data decomposes. Validated by recovering independently hand-decoded circuits on two
ground-truth testbeds, then applied out-of-sample to leaky (fractal-belief) RRXOR.

## The pipeline

0. **Tasks** = k-means clusters of per-instance loss trajectories (an instance = one
   (sequence, target-position) pair; its trajectory = exact CE at every checkpoint).
   On RRXOR this *rejects the designer's taxonomy* for the loss-relevant one (merges
   task-identical classes; discovers the unlabeled match/mismatch luck split at
   purity 1.000).
1. **Windows** = change points of the cluster mean curves (formation eras).
2. **Attribution** = Adam-effective path-integrated gradients: per checkpoint interval
   and per parameter block (per-head attention, per-layer MLP, emb, unembed),
   a_c,b(t) = ⟨(ḡ_c(θ_t)+ḡ_c(θ_{t+1}))/2, Δθ_b(t)⟩, where ḡ_c is the cluster-mean
   gradient and Δθ the actual checkpoint diff (optimizer effects included for free).
   Negative = that block's realized updates helped cluster c. Completeness audit:
   sums reproduce each cluster's total loss change.
3. **Units** = SVD of joint window read/write maps (all resid-reading rows / all
   resid-writing cols of the window displacement, stacked) → read/write directions in
   the shared residual basis + per-block writer participation; auto-annotated against
   semantic directions. **Wiring** = socket–plug matching: cos(write dirs of earlier
   windows, read dirs of later ones).
4. **Validation** = rollback: rewind only a window's top-attributed blocks to their
   window-start values inside the final model; measure per-cluster CE against a
   random-block control. Weight-space time travel.

Figures per run: `tomo_constructionlog.png` (blocks × time attribution-rate heatmap
per task, loss overlaid — the "which blocks build which task, when" picture) and
`tomo_validation.png` (rollback ladder vs stage reference values + wiring matrix).

## Results across four testbeds (data in `runs/<name>/`)

| run | testbed | headline |
|---|---|---|
| `seed0` | RRXOR-3 (4L) | Tasks = {det, rnd-match, rnd-mismatch} (luck taxonomy, unlabeled). Completeness: det attributed −0.769 vs actual −0.776. Unit recovery: gate-window write dir cos 0.87 with the known u direction (writer 96% L4mlp); m-dir 0.83–0.94. Wiring 0.75–0.87 consecutive vs ~0.3 distant. Rollback of gate-window blocks restores the ungated stage [det .11, match .46, mis 1.47] vs null controls. |
| `m27_nl6_seed1` | matryoshka (Z27⊃Z9⊃Z3, 6L; independently whitebox-decoded) | Attribution names the decoded cast per era (L4h3+L4mlp evidence chain; L5 poolers; L6h0/h1/h2 lattice; L6mlp gate). Write dirs: m@L6 cos 0.93–0.98, written 94–98% by L6mlp (the just-in-time rule machinery). Rollback of mod-9-era blocks lands on the preregistered mod-3 ladder rung: xnor 1.61 (stage value 1.50), xor 0.27 (0.25); controls null. |
| `fx0.03_seed0` | leaky RRXOR ε=0.03 (fractal beliefs) | Completeness ±3% including a cluster whose loss RISES +1.53 nats permanently (attributed +1.59): the manufactured hedging pain never relieved — under leak it is Bayes-optimal and permanent. Same early construction cast as classic RRXOR (L4mlp/L3mlp build the bet; L3h0/L4h0 counter-flows). "Surprise" cluster mean CE 2.33 suggests the natural taxonomy under leak is by BET OUTCOME across slot classes (rnd-mismatch + det-teleport-surprises), not by slot class — to be verified. |
| `fx0.2_seed0` | leaky RRXOR ε=0.2 | Completeness ±5% (det −0.300 vs −0.280; mis +0.298 vs +0.295). 14 windows, graded rollbacks with weak top-vs-random contrast: no crisp crystallization under heavy leak — equilibrium is a maintained tug-of-war, and the rollback landscape is correspondingly smooth. |

## Usage

```bash
python devtomo2.py <run_name> <kind>      # kind: rrxor3 | m27 (see script for cluster source)
python devtomo_plots.py <run_name>        # the two standard figures
# stage-0 task discovery standalone: taskdisc.py (m27) / taskdisc3.py (rrxor3-family)
# auxiliary gradient analyses: gradalign2.py (M(t) decomposition), gradsupport.py (BC/σ/PR)
```

Requirements: a run directory with dense checkpoints (`ckpts/step*.pt`) and a fixed
`evalset.pt`. Checkpoint density is the binding constraint: no loss transition should
span fewer than ~2 checkpoint gaps, or completeness degrades (trapezoid integration).

Method guardrails and pitfalls (six of them, each earned): see the `devtomo` skill
(`~/.claude/skills/devtomo/SKILL.md`).

## Positioning

Single-checkpoint circuit discovery (ACDC 2304.14997, EAP 2310.10348/2403.17806, edge
pruning 2406.16778, sparse feature circuits 2403.19647) has no temporal axis; trajectory
clustering exists without circuits (POLCA 2506.15872); training-time gradients exist for
data attribution (TracIn 2002.08484). This pipeline combines all three — trajectory-
discovered tasks, training-path-integrated (Adam-effective) gradient attribution, weight-
displacement units — and adds rollback as the causal instrument matched to developmental
claims (it removes learned *content*; activation cleaning removes a *variable* the net
can re-derive — different counterfactuals, both informative).
