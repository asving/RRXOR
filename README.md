# RRXOR: how a transformer learns and implements a belief-gated circuit

Two complementary studies of small transformers trained on the **RRXOR** process
(repeating blocks `random, random, XOR(r1, r2)` with a random per-sequence phase offset):

| directory | question | headline |
|---|---|---|
| [`circuit_analysis/`](circuit_analysis/) | **What does the final network compute?** (post-training mechanistic dissection) | An *unconditional XOR shortcut* gated by a *Bayes-calibrated phase posterior*; the belief geometry survives only on the task-relevant phase quotient. |
| [`dynamics/`](dynamics/) | **How does training build it?** (preregistered learning-dynamics experiments) | Circuits are learned in order of *gradient accessibility*: the XOR shortcut forms first, its miscalibration **manufactures the gradient** for the phase circuit (a subtask loss provably rises then falls), emergence order = depth order = causal dependency, and the belief geometry unfolds coarse-to-fine from the phase-marginalized quotient. |
| [`matryoshka/`](matryoshka/) | **Does it scale to nested hierarchy?** (3-level exception tower, Z27 ⊃ Z9 ⊃ Z3, + full whitebox decode) | The manufactured-gradient cascade fires at every level (per-class losses spike at the preregistered values); depth is not the constraint (2 layers solve it); and the decoded circuit shows **sync is a matched filter** — the Bayes log-odds is exactly linear in past violation bits, the net's comb matches Bayes at r = 1.000, and the 27-point belief simplex is never built, only its decision quotient. |

The two halves were produced independently (different model configs, different seeds) and
converge: the gated-shortcut anatomy found by the dissection is exactly the endpoint the
learning-dynamics account predicts — the shortcut is the retained first-stage feature, the
gate is the second-stage addition trained on the pain the shortcut itself created, and the
quotient-only belief matches where the coarsening trajectory stops.

Start with `dynamics/README.md` for the narrative, `circuit_analysis/README.md` for the
mechanism. `pip install -r requirements.txt` covers both (numpy, torch, matplotlib).
