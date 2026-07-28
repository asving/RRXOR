# RESULTS 15 — per-interface demand/supply: three instantiations

## 1. m27: the manufactured-gradient cascade as measured demand curves
At the L4 interface (late det queries), demand(k) = normalized ||E[delta x y_k]||^2
(the drift a virtual adapter supplying variable k would feel), supply(k) = probe R2.
- demand(u): 0.303 at step 0 (coherent from init) -> collapses to 0.005 once
  supplied (sup_u 0 -> 0.99): SATISFIED DEMAND STOPS DRIVING.
- demand(phase9): 0.0003 at init (gated) -> MANUFACTURED as u's supply forms
  (0.004@600, 0.022@800, peak 0.036@6000 during the exception-stage CE drop)
  -> decays to 0.003 once satisfied.
- Anomaly (replicates PREREG5 wiring-vs-stream): phase9 supply never linearly
  decodable in the L4 stream at these positions even at solution — satisfied
  through attention wiring from other columns.

## 2. RRXOR route race: full simplex (36 states) vs modular xor->phase
Exact filter enumeration: 51 raw sync states, bisimulation-minimized = 36
(asvin's number confirmed). Supervised races, same arch/data (2 seeds):
  route B1 (xor, 2-way):          tau95 = 175 / 175
  route B2 (phase, 3-way):        tau95 = 1250 / 1975
  route A  (36-way simplex, DIRECT supervision): tau95 = 1600 / 2625
Unsupervised reference (runs/seed0 next-token): xor stage onset ~200-300,
full det resolution ~2500-3000 — the unsupervised cascade pays the same
stage prices as direct supervision.
Reading: the dramatic separation is first-rung vs anything sync-gated
(175 vs 1250-2625, ~10x); the full simplex costs only ~1.3-1.6x the phase
circuit even with the best possible supervision (both are gated by the same
sync bottleneck; the fiber detail adds ~30-60%). The modular route's real
advantages: (i) the free first rung immediately reduces loss and MANUFACTURES
the demand financing the next part (measured in #1); (ii) unsupervised, the
full-simplex route is UNFINANCED — next-token demand asks only for the
decision quotient (w(fiber) ~ 0; the old just-in-time-quotients finding), so
route A is not merely slower, it is never paid for. Direct-supervision times
are its FLOOR.

## 3. comp_icl transience: binding-credit decoherence (recovered + extended)
credit_circuits over all 21 ckpts of the eps=0.8 trajectory (numbers persisted
this time: runs/metrics/credit_traj_full.log). comp peaks 0.375@200, erodes to
~0.25 by 4000; M_gate 0.77 -> 0.53. Credit side: cross-composition binding-
credit coherence (bind_cos) 0.05@200 -> brief peak 0.156@600 -> decays to
~0.03-0.1 fluctuating (dips negative); credit eff_rank 4.35 -> ~1.5
(concentration/specialization); factor-subspace share stays tiny (0.015-0.05)
throughout — the credit is binding-directed, its COHERENCE across
compositions is what dies. Reading: transience = DECOHERENCE of co-financing
(the masks stop agreeing on how to move the shared binding machinery; the
aggregate drift ||sum_c g_c||^2 shrinks while per-mask magnitudes persist).
Directional, single seed — needs the seed battery before it hardens.
