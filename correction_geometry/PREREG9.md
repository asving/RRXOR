# PREREG 9 — PCFG-2: the leap-hard Random Hierarchy Model

Date: 2026-07-22. Written BEFORE training. PCFG-1 (RESULTS8) showed tape-invisibility
alone is insufficient — the broadcast tree was compositionally shallow. PCFG-2 adds
leap-hardness: random UNAMBIGUOUS production tables (v=4 symbols, m=2 productions per
symbol per level, depth 4, binary branching; parent = random lookup of the child
tuple, so level-l parsing is an l-fold composition of random tables with no
pairwise-correlation shortcut). 6 trees per sequence, L=96, token vocab 4; grammar
fixed (seed 7); exact BP oracle validated against enumeration (max |Δ| = 0.0).

## Preregistered class × stage CE matrix (stage l = BP truncated at l ancestor levels,
uniform prior at the cut; MC 4096 trees)

```
        first    b=0      b=1      b=2      b=3     stage avg
l=0:    1.3863   1.3863   1.3863   1.3863   1.3863   1.3863
l=1:    1.3863   0.6931   1.3863   1.3863   1.3863   1.0397
l=2:    1.3863   0.4567   1.1835   1.3541   1.3873   0.8668
l=3:    1.3863   0.4031   0.9971   1.2480   1.3873   0.7802
l=4:    1.3863   0.3743   0.9341   1.1666   1.3473   0.7374
```

(Note: rows l < b show ≈ ln 4 at class b — at those boundaries the target's own
subtree has no seen leaves, so the truncated policy is the marginal; the 1.3873
entries reflect near-uniform leaf marginals under the uniform-cut convention.)

Runs: **dense** and **hard-only** (loss at classes b ≥ 2: within-tree indices 4, 8,
12 → 18 targets/sequence), same seed, GPTV nl=6 d=128 vocab=4, 60k steps, batch 256,
eval every 100 (per-class CE), checkpoints every 100 to 3000 then every 250.

## Predictions

- **P1 (the staged ladder returns).** Unlike PCFG-1, per-class CE drops are now
  separated in time and ordered by level (b=0 first, then 1, 2, 3), with plateau
  values matching matrix rows (tolerance 0.02); the average CE visits the stage
  ladder. Falsifier: all classes reaching floors within one eval window of each
  other again (would mean leap-hardness at this size still doesn't separate).
- **P2 (the bootstrap crux, now with teeth).** Hard-only training is qualitatively
  impaired at the deep classes: the level-1/2 parse tables receive no direct
  supervision, are tape-invisible AND leap-hard. Prediction: hard-only fails to
  reach the dense b=2/b=3 floors (1.1666/1.3473) within 60k steps, with a residual
  gap > 0.05 nats, OR requires ≥ 5× the dense step count to close. Falsifier: gap
  < 0.02 by 60k — which would mean even leap-hard hidden scaffolds are learnable
  from sparse deep loss, and the bootstrap story needs yet another ingredient.
- **P3 (layer × level grid, now with time).** Ancestor-posterior probes per network
  layer per checkpoint: level-j decodability rises in sync with its class's drop
  (time-order, now resolvable), monotone in layer (deeper levels deeper — the
  inverse-Bayes pipeline, as already seen statically in PCFG-1).
- **P4 (guess matrix).** At each dense plateau ℓ, the net's per-class CEs match row
  ℓ AND its conditionals match truncated-BP-ℓ per position (held-out R² ≥ 0.9 on
  log-odds, correction-spectroscopy style) — "predicting what the network guesses
  at every drop."
