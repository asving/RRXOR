# RRXOR: how a transformer learns and implements a belief-gated circuit

Four complementary studies of small transformers trained on the **RRXOR** process
(repeating blocks `random, random, XOR(r1, r2)` with a random per-sequence phase offset):

| directory | question | headline |
|---|---|---|
| [`circuit_analysis/`](circuit_analysis/) | **What does the final network compute?** (post-training mechanistic dissection) | An *unconditional XOR shortcut* gated by a *Bayes-calibrated phase posterior*; the belief geometry survives only on the task-relevant phase quotient. |
| [`dynamics/`](dynamics/) | **How does training build it?** (preregistered learning-dynamics experiments) | Circuits are learned in order of *gradient accessibility*: the XOR shortcut forms first, its miscalibration **manufactures the gradient** for the phase circuit (a subtask loss provably rises then falls), emergence order = depth order = causal dependency, and the belief geometry unfolds coarse-to-fine from the phase-marginalized quotient. |
| [`matryoshka/`](matryoshka/) | **Does it scale to nested hierarchy?** (3-level exception tower, Z27 ⊃ Z9 ⊃ Z3, + full whitebox decode) | The manufactured-gradient cascade fires at every level (per-class losses spike at the preregistered values); depth is not the constraint (2 layers solve it); and the decoded circuit shows **sync is a matched filter** — the Bayes log-odds is exactly linear in past violation bits, the net's comb matches Bayes at r = 1.000, and the 27-point belief simplex is never built, only its decision quotient. |
| [`tomography/`](tomography/) | **Can tasks AND circuits be discovered automatically, from checkpoints alone?** (developmental tomography, validated on the decoded testbeds) | Yes: trajectory-clustered tasks, Adam-effective path-integrated attribution, weight-displacement units with socket–plug wiring — and **rollback** as weight-space time travel: rewinding one window's four blocks drops the net onto the exact earlier rung of the closed-form loss ladder (xnor 1.61 vs stage value 1.50). |
| [`correction_geometry/`](correction_geometry/) | **Which belief geometries get built at all — and can the machine be rebuilt from behavior?** (preregistered probe trajectories + correction spectroscopy + spectral Hankel/OOM reconstruction) | Representations are sufficient statistics of the net's stagewise **corrections**, not of the task: every quotient sits at baseline until its own stage's loss-ladder era, then rises (u → mod-3 → exception-identity); a never-consumed 14-dim refinement of the *perfectly inferable* phase stays below baseline at **all 76 checkpoints**; leaky nets ramp smoothly (discreteness lives in the task, not the learner). Two refinements: correction statistics can be absorbed into attention **wiring** instead of stream codes, and linear codes **consolidate ~0.5–1k steps after behavior**. Spectral reconstruction from the net's conditionals recovers the generator exactly (RRXOR: rank-5 Hankel, clock eigenpair 0.998∠2.0944 vs e^{2πi/3}; 12-step rollout |Δp| ≤ 0.003) and the **root-of-unity tower**: matryoshka's behavioral transfer operator fills the unit circle **1 → 3rd → 9th → 27th roots** across the drops — the dual of the Z27⊃Z9⊃Z3 tower, rung by rung. |

The two halves were produced independently (different model configs, different seeds) and
converge: the gated-shortcut anatomy found by the dissection is exactly the endpoint the
learning-dynamics account predicts — the shortcut is the retained first-stage feature, the
gate is the second-stage addition trained on the pain the shortcut itself created, and the
quotient-only belief matches where the coarsening trajectory stops.

Start with `dynamics/README.md` for the narrative, `circuit_analysis/README.md` for the
mechanism. `pip install -r requirements.txt` covers both (numpy, torch, matplotlib).
