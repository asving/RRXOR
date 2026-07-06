# Preregistration 3 — matryoshka RRXOR (nested exception tower, depth sweep)

Date: 2026-07-04. Written before running any of these experiments.

## Task

Blocks [r1, r2, f(r1,r2)] with f = XOR except an exception tower: XNOR-candidates at
blocks ≡ 2 (mod 3); among candidates, every 3rd (block 8 of the 9-block super-period)
flips back to XOR. Rule pattern per super-period: [X,X,N, X,X,N, X,X,X*] (X* = the fake
exception). One hidden offset uniform over 27 tokens. Quotient tower over the phase:
Z27 ⊃ Z9 ⊃ Z3 ⊃ point. Slot classes per 27 tokens: RND x18, XOR6 (plain xor det slots),
XNOR2 (true exceptions), FLIP1 (exception-to-the-exception; matches xor).

L=192, eval at target positions >= 108 (phase resolvable). d=128, 4 heads, 16k steps.
Depth sweep: nl in {2, 3, 4, 6, 8}, 2 seeds each.
Control: FLAT pattern — XNOR at blocks {0, 4}, no flip (same det match rate 7/9, same
final optimum, but NO coherent Z9 intermediate quotient), nl=6, 2 seeds.

## Closed-form stage table (nats; q = bet on xor of prev two)

Pooled det-slot match rate 7/9; pooled overall 16/27.

| stage (policy)            | XOR6   | XNOR2  | FLIP1  | RND    | average |
|---------------------------|--------|--------|--------|--------|---------|
| 0 calibrated              | 0.6931 | 0.6931 | 0.6931 | 0.6931 | 0.6931  |
| 1 ungated (q=16/27)       | 0.5232 | 0.8979 | 0.5232 | 0.7106 | 0.6758  |
| 2 phase mod 3 (q=7/9 det) | 0.2513 | 1.5041 | 0.2513 | 0.6931 | 0.6386  |
| 3 phase mod 9 (q=1/3 exc) | ~0     | 0.4055 | 1.0986 | 0.6931 | 0.5328  |
| 4 phase mod 27            | ~0     | ~0     | ~0     | 0.6931 | 0.4621  |

## Predictions

P1 CASCADE: per-class losses are non-monotone in counterphase.
   XNOR2: 0.693 -> up (~0.9) -> UP (~1.50) -> down (0.41) -> 0  (double rise)
   FLIP1: 0.693 -> down -> down (0.25) -> UP (~1.10) -> 0      (rise at stage 3 only)
   RND:   small bump (~0.71) relieved at stage 2, then flat ln 2.
   Each rise is manufactured by the PREVIOUS stage's circuit (same mechanism as RRXOR,
   firing at successive levels).

P2 ORDER: emergence strictly ordered tau(xor) < tau(mod3) < tau(mod9) < tau(mod27), all
   seeds at sufficient depth; average loss touches (possibly briefly) the plateau ladder
   0.6931 / 0.6758 / 0.6386 / 0.5328 / 0.4621.

P3 DEPTH TRUNCATION (the loud one): there is a minimal depth d* (guess: 3-5). Networks
   with nl < d* converge to a final average ON one of the preregistered intermediate
   plateaus (within ~0.01 of 0.6758 / 0.6386 / 0.5328), not to 0.4621 and not to an
   arbitrary value. Deeper-than-needed nets (nl=8) reach the same final solution.

P4 DENDROGRAM / QUOTIENT GEOMETRY: pairwise-distance structure of the 27 phase-class mean
   representations (final layer, late positions) develops TOP-DOWN: first 3 clusters
   (mod 3), then 9 (mod 9), then 27 — quotient separability scores for Z3, Z9, Z27 cross
   threshold in that order, matching the loss-plateau exits.

P5 FLAT CONTROL: same marginal statistics but no coherent Z9 quotient -> no stage-3
   plateau at 0.5328 and no intermediate 9-way dendrogram stage (3 -> 27 directly);
   reaching the final solution is SLOWER than the nested version at equal depth
   (stepping stones accelerate).

P6 DEPTH PLACEMENT (probes, nl=6): decodability layers ordered
   layer(xor) <= layer(mod3) <= layer(mod9) <= layer(mod27), with the finest level
   nearest the output (consumer-pull, as in round 2).

Falsifiers: any pair of stages emerging out of order at sufficient depth; truncated nets
landing between plateaus; dendrogram splitting 9-before-3; flat control faster than
nested; a class trajectory missing its predicted rise (e.g. XNOR2 monotone).
