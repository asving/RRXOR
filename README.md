# Reverse-engineering the RRXOR circuit in a small transformer

A complete mechanistic account of how a 4-layer transformer solves the **RRXOR**
process: it does **not** track the belief simplex that comp-mech would predict.
Instead it learns an *unconditional XOR shortcut* gated by a *Bayes-calibrated
phase posterior* that it infers from the bit history. This repo contains the
trained model and every script needed to reproduce the analysis.

## The process

RRXOR is a 5-state hidden Markov process with a 3-phase cycle and binary emissions:

- **phase 0** emits a uniformly random bit `R1`
- **phase 1** emits a uniformly random bit `R2`
- **phase 2** emits the deterministic `XOR(R1, R2)`

Sequences start from the stationary distribution, so the phase at position `t`
has a **random per-sequence offset** — phase is *not* a function of absolute
position and must be inferred from the emitted bits.

A `d_model=64`, 2-head, 4-layer transformer trained on the raw bit stream reaches
phase-2 accuracy 1.00 (mean phase-2 logit difference ±10).

## What we found

The model implements

```
logit_diff(next=1 vs 0)  ≈  α(phase posterior) · (−1)^(1 ⊕ XOR(bit_{t−1}, bit_t))
```

where `α` is **quantitatively equal to the Bayes log-odds given the inferred phase**
(corr 0.995 overall; 0.974 even on positions where the phase is still uncertain).
The pipeline:

| stage | role | evidence |
|---|---|---|
| **L0–L1** | transport current + previous bit into the residual stream | causally necessary (zeroing kills the circuit) |
| **L2 head 0** | attends 98% to offset −1 → fetches `bit_{t−1}` | ablating this *one head* drops phase-2 acc to chance (0.54) |
| **L2 MLP** | computes `XOR(bit_{t−1}, bit_t)`, **unconditionally at every position** | XOR linearly decodable (1.00) from here on |
| **L3 attention** | **infers the phase** by pooling XOR-*violation* evidence | mod-3 "comb" heads, 2–4× weight on violation tokens |
| **L3 MLP** | gates the XOR signal via the LayerNorm denominator | floods the residual with a readout-orthogonal vector at phases 0/1, cancels it at phase 2 |

### Three results worth highlighting

1. **Phase is inferred, not read off position.** A position `s` with
   `b_s ≠ b_{s−1} ⊕ b_{s−2}` cannot be a phase-2 emission. The two L3 heads form
   *offset-parity combs* (one per non-self phase class) and up-weight violation
   tokens. The gate ramps up on exactly the Bayes synchronization schedule and
   **never fires on the all-zeros stream** (which never synchronizes). Shifting
   the positional indices by any `k` leaves behavior unchanged — refuting a
   "reads `t mod 3` from positional embeddings" story.

2. **The model carries a graded belief — but only over the task-relevant
   quotient.** The Bayes log-odds of "next slot is the XOR slot" is linearly
   decodable at R² ≈ 0.8–0.9 at L3 (absent at L2). The full 5-state simplex is
   *not* represented; the R1/R2 content is handled by the unconditional XOR
   shortcut. Belief geometry survives exactly on the phase quotient, in log-odds
   coordinates, and nowhere else. (The standard "centroid fit to belief vectors"
   reports R²=1.00 at *every* layer including layer 0 — a degenerate artifact of
   regressing 12 centroids in 64 dimensions; per-token probes are the right
   instrument and tell the real story.)

3. **L3 MLP is inverse gain control through LayerNorm.** Its dominant output is a
   norm-108, readout-*orthogonal* `b_prev` vector, with a `b_prev × phase`
   interaction anti-parallel to it (cos −0.998). Net effect: a huge orthogonal
   flood at phases 0/1 that inflates the final LayerNorm's σ and squashes the XOR
   signal, cancelled at phase 2 to let it through. Removing it at a phase-2 cell
   collapses |logit| from 12.9 to 2.4 (sign still correct).

The model is **behaviorally equivalent to a pipeline of 12-cell lookup tables**
indexed by `(bit_{t−1}, bit_t, phase)` — replacing all 8 residual deltas with their
conditional means reproduces the model to sub-decibel logit drift. The caveat: the
phase index is *inferred* by the subcircuit above, not given.

## Repro

```bash
pip install -r requirements.txt        # numpy + torch only
```

A trained checkpoint is included at `results/rrxor_belief/naive_ctx/model.pt`, so
the analysis runs out of the box. Each script writes a JSON into
`results/rrxor_belief/naive_ctx/circuit/` (precomputed copies are included for
reference).

```bash
# core circuit dissection (run in this order)
python rrxor_shortcut_probes.py     # per-token belief / last-2-bits / logit probes
python rrxor_circuit_analysis.py    # fine-grained layer probes, attention patterns, activation patching
python rrxor_circuit_verify.py      # head ablations, XOR-direction patch, phase-gating ablation
python rrxor_circuit_prune.py       # component knockouts

# the phase-inference story
python rrxor_phase_sync.py          # sync curve + gate-vs-Bayes-log-odds + all-zeros stream
python rrxor_posemb_shift.py        # causal test: positional shift leaves behavior unchanged
python rrxor_l3_attn_phase.py       # L3 heads pool XOR-violation evidence (mod-3 combs)

# lookup-table equivalence + LayerNorm mechanism
python rrxor_lookup_substitute.py   # 144-cell conditional-mean substitution
python rrxor_lookup_decompose.py    # 12-cell lookups + orthogonal-contrast decomposition (writes lookup_tables_12.npz)
python rrxor_ln_propagate.py        # propagate contrasts through LN+unembed (needs lookup_tables_12.npz)
python rrxor_lookback_sweep.py      # R² of each delta by lookback window
```

To retrain from scratch (≈16k steps; GPU or CPU):

```bash
python rrxor_belief_experiment.py --mode naive \
    --outdir results/rrxor_belief/naive_ctx --seq-len 256 \
    --d-model 64 --n-layers 4 --n-heads 2 --batch-size 64 \
    --steps 16000 --lr 3e-4 --train-seqs 800 --probe-seqs 400 \
    --seed 0 --device cpu
```

Note: circuits are seed-dependent. A retrained model reaches the *same
Bayes-calibrated behavior* but may realize it with a different circuit (e.g.
diffuse vs sharp L3 attention). The algorithm — calibrated violation-accounting
gating an unconditional XOR — is the reproducible invariant; the exact wiring is
not.

## File guide

| file | what it provides |
|---|---|
| `rrxor_belief_experiment.py` | RRXOR HMM, `MiniTransformer`, training, base probes |
| `rrxor_reconstruct_probe.py` | `hooked_forward` (residual deltas), linear probes, variance-by-condition |
| `rrxor_circuit_analysis.py` | `hooked_forward` with attention weights; stage probes, patterns, patching |
| `rrxor_circuit_verify.py` / `rrxor_circuit_prune.py` | causal ablations |
| `rrxor_phase_sync.py` / `rrxor_posemb_shift.py` / `rrxor_l3_attn_phase.py` | phase-inference mechanism |
| `rrxor_lookup_substitute.py` / `rrxor_lookup_decompose.py` / `rrxor_ln_propagate.py` / `rrxor_lookback_sweep.py` | lookup equivalence + LayerNorm gain control |
