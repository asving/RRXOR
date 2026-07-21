# RESULTS 7 — Loss placement (grading PREREG7)

Same-seed pairs (identical init and data stream; only the loss mask differs):
rrxor3 seed0 / m27_nl6 seed1 / mess3 seed0, dense baselines vs last-token-only twins
(24k / 48k / 24k steps). Artifacts: train_pos.py, baseline_celast.py, per-run
eval.jsonl + celast.jsonl, prereg7_posprofile.json, prereg7.png.

**P1 (penalty ordering at matched instances) — REFUTED, with the sign reversed.**
Matched on total loss terms, last-only is far MORE efficient per term: at 6.1M terms,
dense (step ~252, its terms spread over 95–191 positions) is still at uniform
(ce_last 0.692 / 0.694) while last-only has solved the task (0.467 / 0.467).
Per-sequence (= per step = per FLOP), dense wins by a constant factor ~3–6×: rrxor3
dense converges by ~3k steps, last-only by ~15–24k; m27 dense by 5.6k, last-only by
~24–32k. No qualitative barrier anywhere: last-only reaches the SAME endpoint on all
three tasks, including the full 4-rung tower from loss at a single position (final
ce_last: rrxor3 0.4674 vs dense 0.4676; m27 0.4673 vs 0.4696; mess3 1.0829 vs 1.0880).
The bootstrap-via-dense-loss prediction is dead for tape-visible tasks: attention-
mediated credit assignment from one query position suffices to build all source-side
scaffold codes.

**P2 (rung stalling) — REFUTED, and replaced by a better finding: supervision density
reshapes the developmental geometry.** Dense m27 at the last position shows the sharp
staircase with the manufactured-gradient signature (xnor CE spikes 0.88 → 1.355 at
step ~2400, then all det classes snap to ~0 in strict rung order by 5600). Last-only
shows a MUTED excursion (xnor max 0.914 at step 1000) and then all three det classes
co-descend smoothly together (0.24/0.42/0.41 at 4k → 0.03/0.07/0.05 at 48k) — no
staircase, no strict ordering. The ladder is not intrinsic to the task: it is a
property of task × supervision density. Concentrated single-position supervision
parallelizes/blurs the stage structure (the corrections are learned near-
simultaneously, each more slowly).

**P3 (machine-ness) — CONFIRMED, dramatically.** The last-only nets are calibrated
ONLY at the trained position (CE 0.467 at position 95/191, matching dense) and are
actively miscalibrated elsewhere — CE 1.3–1.75 at untrained positions, WORSE than
uniform (ln 2 = 0.693). The net is a behavioral machine at exactly one position;
dense supervision is what buys position-uniform machine-ness (and hence the validity
of stationary spectral extraction).

**P4 (mess3 null) — CONFIRMED** (gaps ≤ 0.005 nats everywhere on mess3).

## Reading

Dense next-token supervision, on tasks whose intermediate variables are tape-visible,
buys three things — none of them possibility: (i) a constant-factor (~3–6×) speedup
in sequences, (ii) calibration at every position (machine-ness), (iii) the staircase
itself, i.e. the sharp correction-cascade dynamics with manufactured-gradient
excursions. The Cagnetta–Wyart bootstrap ("easy positions build the latents hard
positions consume") is therefore NOT load-bearing here — but our tasks put every
useful intermediate value on the tape. The sharp follow-up: a task whose intermediate
variables never surface as tokens (PCFG nonterminals proper), where the only route to
the latents is through other positions' supervision; the prediction that last-only
training hits a qualitative barrier there remains open and now well-posed.
