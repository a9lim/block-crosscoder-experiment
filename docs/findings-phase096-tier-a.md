# Phase 0.9.6 tier A — consolidation & robustness matrix (2026-07-17)

12 runs on the 1b 0.9 store, all at the ratified winner optimizer
(lr 1.2e-3 cosine, λ=1e-3, enc-wd 0) unless tagged. Measurement:
`scripts/analysis/tier_a_ring_tests.py` — encodes the saved calendar-probe
activations (18k month tokens, capitalized-only labels) through every
checkpoint; consolidation = per-class top-1/top-2 block maps by class-mean
selection score; ring = calendar-adjacent hits in the top plane of the
consensus block's class means, 20k-draw permutation null (p floors at
5.0e-5). Same screen replayed over the 0.9.5 reference checkpoints for
apples-to-apples (`ring_tests_ref095.json`); it reproduces the interim
findings' numbers (winner b23 12/12, seed-1 b60 7/12, renorm@3e-4 b140
10/12, G4096 b244 10/12). Raw JSON: `data/analysis/tier_a_ring_tests.json`
(Mac mirror) / `/data/runs/bcc-analysis/` (jobe).

## The headline: capture always consolidates; calendar *order* is a lottery

Every tier-A run — every seed, epoch count, G, and the renorm arm — ends
with **a single block claiming the month family** (top-1 ≥ 11/12, most
12/12; top-2 always ≥ 11/12), with 93–99 % of class-mean variance in a
plane. What varies is whether that block's code plane arranges the months
in **calendar order**:

| run | best block | top-1 | ring | perm p |
|---|---|---|---|---|
| seed 0 (=0.9.5 winner) | 23 | 12/12 | **12/12** | 5.0e-5 |
| seed 1 | 60 | 11/12 | 7/12 | 2.4e-3 |
| seed 2 | 895 | 12/12 | 10/12 | 5.0e-5 |
| seed 3 | 843 | 12/12 | 2/12 | 0.67 |
| seed 4 | 298 | 12/12 | 3/12 | 0.37 |
| seed 5 | 937 | 12/12 | **12/12** | 5.0e-5 |

**A1 verdict:** 2/6 seeds perfect, 2/6 near (9–10), 2/6 chance. The
runbook criterion (≥3/5 additional seeds at 12/12) **fails** →
`PILOT_EXTRA_SEED=1`. But single-block ring order is a real minority
outcome, not seed-0 luck — the paper's consolidation row is a
distribution, and "block captures the family" is the invariant part.

## A2: epochs don't buy order at the winner lr

- Seed 0: 12/12 at ep2, ep4, ep8 — stable, same block 23.
- Seed 1: 7/12 (ep2, b60) → 6/12 (ep4, b60, top-1 split to 6/12) →
  4/12 (ep8, capture **migrated to b371**, top-1 back to 12/12).
  Optimization time reshuffles ownership; it does not order the plane.
- lr 3e-4 seed 0: ep2 5/12 → ep8 9/12 (same b23). At a starved lr more
  epochs *do* help — order tracks total effective optimization, and the
  winner lr is already order-converged at 2 epochs.

**Verdict:** `PILOT_EPOCHS=2`; the pilot store's 6M unique train tokens
carry the more-data insurance instead.

## A3: renorm @ winner lr — inconclusive under the lottery

b140 (the same block its lr-3e-4 sibling chose), top-1 12/12, ring 4/12
vs matched baseline's 12/12. One draw can't separate "renorm hurts order"
from "renorm re-rolls the lottery." The F7 analysis-primary designation
is **deferred to the 4b pilot's own arms** (both run regardless).

## A4: G ladder — 2048 healthy, 8192 not tame

- G2048: b480, top-1 12/12, ring 9/12 — healthy.
- G8192 k=32: trains stably (16k tok/s) *after two infrastructure
  fixes* (below); b5840, top-1 12/12, ring 9/12, plane 93 %; pooled FVU
  0.522; **dead_frac 3.58 %** — 36× the G4096 stress arm's 0.098 %.

**Verdict:** dead dynamics not tame → `PILOT_G8192=0`. The stretch config
stays a Phase-1 decision on the production store (the arm can be run
standalone against the pilot store later — it's an env knob).

## Block identity is init-determined

Seed 0's month block is b23 across lr {3e-4, 6e-4, 1.2e-3}, λ {0, 1e-3},
schedule {cosine, linear_fifth}, wd, and epochs {2,4,8}; renorm's is b140
at both lrs; seed 1's is b60 at ep2/ep4. Which block wins is set at init;
optimization sets the geometry inside it (and can, rarely, migrate
ownership — seed 1 ep8).

## Weekday null holds

Max 3/7 everywhere (p 0.43) except seed-1 ep8's 5/7 (p 0.04) — one
nominal hit among ~19 tests, dies under any multiple-comparisons
correction. Capture-without-ring stays the weekday story.

## Infrastructure findings (both fixed same-day)

1. **cusolver batched-syev batch limit** (`efb4f9b`): G=8192 × S=6 =
   49,152 stacked 4×4 eigenproblems; `cusolverDnXsyevBatched_bufferSize`
   rejects ≥ ~32k (INVALID_VALUE on finite input; 24,576 passes).
   `site_singular_values` now chunks at 16,384 — bit-identical,
   differentiable, and required for the 4b G8192 config (65,536).
2. **θ-calibration host RAM at G=8192**: 128 calib batches → 62 GB
   anon-rss → OOM-killed on 61 GB jobe (training had completed; resume
   + `--calib-batches 64` recovered without retraining). Sharper
   evidence for the Phase-1 streaming-quantile carry item. The tier-A
   script's G8192 line now carries `--calib-batches 64`.

## Disposition

Tier B launched 18:59 same day with `PILOT_EPOCHS=2 PILOT_EXTRA_SEED=1
PILOT_G8192=0`. Carry into Phase-1 eval: the known-ring-consolidation
probe should report *both* statistics — capture consolidation (robust,
the architecture's claim) and calendar order (a per-seed distribution,
the geometry's claim) — and never conflate them.
