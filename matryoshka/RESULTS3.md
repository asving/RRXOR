# Results vs PREREG3.md — matryoshka RRXOR (nested exception tower, depth sweep)

Runs: nl in {2,3,4,6,8} x 2 seeds at 16k steps; six configs extended to 48k
(deterministic replay); flat control (nl=6, 2 seeds). Figures:
runs/m27_nl6_seed1/cascade.png (flagship), quotient_ev.png, distmat.png,
probe27_heatmap.png; depth27.png; runs/m27_nl6_seed0_16k/* (slow-seed contrast).

## P1 cascade — CONFIRMED (solving runs), quantitative

nl6_seed1: XNOR2 walks 0.693 -> 0.898-plateau (pred 0.8979) -> spike 1.495 (pred 1.5041)
-> 0.405 shoulder -> 0. FLIP1 rides DOWN with XOR6 (0.52 -> 0.25) then spikes to 1.046
(pred 1.0986) exactly as the mod-9 circuit forms, then -> 0. RND bumps to ~0.71 (pred
0.7106), relieved at mod-3. XOR6 steps 0.523 -> 0.251 -> 0 (preds 0.5232/0.2513).
Counterphase M/W shapes as preregistered. The manufactured-gradient mechanism fires at
every level of the tower.

## P2 order & plateau ladder — CONFIRMED conditionally

Solving runs: tau(ungated) 640 < tau(mod3) 2400 < tau(mod9) 3000 < tau(mod27) 4500; the
average walks all five rungs (0.6931/0.6758/0.6386/0.5328/0.4621). The "touches all
plateaus" clause holds only when level timescales separate: slow seeds (e.g. nl6_seed0,
stage-1 exit at 8500) then slide through stages 2-4 continuously — pipelining smears the
ladder. Where runs pause mid-training, they pause ON the preregistered rungs
(0.6467/0.6431 at stage 2; 0.5495/0.5227 near stage 3).

## P3 depth truncation — REFUTED in the strong form

No permanent truncation at any depth. The 2-LAYER net fully solves the 3-level tower
(0.4624; crosses 0.470 at ~17.6k steps). Extended runs all escape their plateaus (nl6_s0
solves at 43k, nl8_s1 at 38k; nl4_s0 still in stage-3 transit at 48k but descending).
Depth weakly affects speed; SEED variance dominates (nl3_s0 solves at 6.5k while nl4_s0
hasn't at 48k). What survives of P3: the mid-training parking spots are quantized to the
preregistered ladder. The rungs are saddle-like waypoints, not depth-forced endpoints.
Implication: inferential depth of the task does not translate into required architectural
depth — attention pooling collapses the tower (cf. depth-vs-recurrence: a
contractive/forgetting filter can be depth-flat).

## P4 geometry — top-down resolution CONFIRMED, plus unpredicted re-coarsening

Centroid explained-variance (final layer): EV(mod3) rises first (~700), the mod9
increment rises second (~1500-2000), matching the loss transitions; distance matrices
resolve the 3-block structure first. UNPREDICTED: at convergence the final layer
RE-COARSENS to pure mod-3 block structure (EV(mod3) -> 1.0); the finer distinctions are
transient there. Resolution: they are consumed mid-network. See P6.

## P5 flat control — suggestive, underpowered

Flat (no coherent Z9 quotient), nl=6: neither seed solved at 16k (0.5697/0.5645); the
48k extension reached 0.4873 and was still descending — slower than both nested nl6
seeds' solve points (5.3k / 43k)... but with n=2 per arm and huge seed variance, call the
stepping-stone acceleration suggestive, not established.

## P6 depth placement — CONFIRMED, with a sharpening (sufficiency)

Probe wave over (layer x step), solving run: xor at L3+ from ~600; mod3 at L5-L6 from
~2400; mod9 at L5-L6 from ~3000 (max 0.64); mod27 faint at L6 late (max 0.36). Time
order = depth order at all four levels, everything crowded toward the consumer end.
Sharpening: the full 27-way phase is NEVER well decodable anywhere. The decision-relevant
binary — rule of the NEXT det slot, probed at pre-det positions — sits at the 0.778 base
rate through L0-L5 and jumps to 0.957 at L6: the network computes the coarsest sufficient
statistic, just in time, at the point of consumption. Reproduces the circuit_analysis
finding ("belief survives only on the task-relevant quotient") at 3 levels of hierarchy.

## Frame updates

1. Manufactured gradients are generic and recursive: each level's circuit creates the
   miscalibration that trains the next; random-position loss stays elevated (~0.71-0.72)
   for the entire transit and relaxes only at full solution.
2. The plateau ladder is an attractor set of saddle-like waypoints. Escape times are
   wildly seed-dependent and grow with tower height; hierarchy shows up in TRAINING TIME
   distributions more than in final loss or required depth.
3. Depth >= 2 suffices for a 3-level inference tower here — dependency depth is about
   gradient ORDER (what can be learned when), not about layer count.
4. Converged networks do not keep the hierarchy as a legible tower: they keep
   task-sufficient quotients, computed just-in-time near the consumer, and the final
   layer re-coarsens once finer distinctions have been absorbed into the computation.

Caveats: 2 seeds per config; stage-time metrics smeared for pipelined seeds; extended
runs assume deterministic replay of the first 16k (verified on the solved run).
