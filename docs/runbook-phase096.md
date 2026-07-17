# Runbook — Phase 0.9.6 (2026-07-17, a9-directed)

**What this is.** a9 bundled two tiers into one pre-NVMe campaign:
**tier A**, a 1b consolidation/robustness matrix on the existing 0.9
store, and **tier B**, the design-mandated **D13 4b exact-config pilot**,
which fits on jobe's existing `/data` disk (~290 GB stored vs ~600 GB
usable above the ShardWriter floor — only the 53M-token production store
needs the 4 TB NVMe). Everything is staged; **nothing has been launched.**
Motivating context: `docs/findings-interim-artifact-analysis.md` (the
2026-07-17 interim sweep — block-23 month ring, consolidation-vs-optimizer
open question, F7 mechanism, renorm/lr confound).

**Session note:** this runbook is written to survive a context compaction —
it carries everything needed to launch, monitor, and measure.

## Launch commands (jobe)

Both scripts are `nohup`-able and idempotent-ish (tier B skips a completed
harvest; a failed run exits the script — relaunch after diagnosis,
completed runs are skipped only by deleting/renaming their `run()` line).
**One 4090: run the tiers sequentially, not concurrently.** Suggested
order — tier B first (a9 priority; harvest is the long pole), tier A
after:

```bash
ssh jobe
cd ~/Work/transformer-experiments/block-crosscoder-experiment
mkdir -p /data/runs/bcc-pilot4b /data/runs/bcc-phase096
nohup bash scripts/run_phase096_pilot4b.sh > /data/runs/bcc-pilot4b/pilot.log 2>&1 &
# ... after PILOT DONE:
nohup bash scripts/run_phase096_tier_a.sh > /data/runs/bcc-phase096/matrix.log 2>&1 &
```

## Tier B — the D13 4b pilot (`scripts/run_phase096_pilot4b.sh`)

| stage | what | expected |
|---|---|---|
| harvest | `harvest_pilot4b_store.py`: gemma-3-4b, **sites (9,12,15,18,21,24,27,30)**, 2M whitener + 2M calib + 1M eval + 4M train, whitened bf16 → `/data/stores/bcc-pilot4b/gemma3_4b_8site_fineweb` | hours (4b forward, ~290 GB written); watch tok/s + io-wait lines |
| verify | `verify_phase09_store.py --store …` | checksums + whitening round trip green |
| bsc_primary | G=4096×b=4, k=32, lr 1.2e-3 cosine, λ=1e-3, SASA AuxK; split `--max-steps 900` → `--resume` (checkpoint/resume gate at production config) | 1953 steps total; resume must continue bit-compatibly |
| bsc_renorm | same + `--site-renorm` (F7 arm) | — |
| scalar | matched baseline, 16384 latents, k=128, `--calib-batches 64` (θ score matrix ≈17 GB host RAM; 128 would be ~34 GB of jobe's 61) | — |

**D13 gates to check in the logs/reports** (`/data/runs/bcc-pilot4b/`):
whitener stability (halves/quarters rel ΔW — if 4b needs the production
5M slice, these numbers are the evidence), held-out spectrum vs shrinkage
prediction, resume bit-compatibility, θ transfer (threshold-mode
avg_active_blocks ≈ 32), eval determinism `true`, dead_frac, per-site FVU
(8 sites, depth allocation — compare the late-heavy pattern + renorm
reversal seen at 1b), train tok/s + data-wait (store-reader prefetch is a
known Phase-1 carry item).

**Flagged for a9 ratification:** the site-list resolution
(9,12,15,18,21,24,27,30) — design says "8 in 25–90% band (≈ layers 9–30,
resolved at harvest, frozen in config)"; step-3 spacing lands exactly on
the band edges and brackets the Phase-0 probed depths 9/17/22/29. The
production harvest should freeze the same list if the pilot passes.
**D13 risks are cleared only by this pilot passing, not by the 1b
rehearsal** (design language).

## Tier A — 1b consolidation matrix (`scripts/run_phase096_tier_a.sh`)

All at the ratified winner optimizer (lr 1.2e-3 cosine, λ=1e-3) unless
tagged. Out-root `/data/runs/bcc-phase096`. ~12 runs, most 4–8 min;
epoch-8 runs ~16 min; total well under 2 h.

| arm | runs | question |
|---|---|---|
| A1 seeds | seed 2,3,4,5 | is single-block ring consolidation (block-23-style, 12/12) the rule or seed-0 luck? Interim data: seed 0 → 12/12 one block; seed 1 → 7/12 |
| A2 epochs | ep4/ep8 × seed 0/1, ep8 @ lr 3e-4 | does more optimization consolidate seed 1 / merge lr-3e-4's two-block split (5/12 alone, 12/12 top-2)? |
| A3 renorm @ 1.2e-3 | 1 run | F7 deconfound — interim renorm arm ran only at lr 3e-4 (10/12) |
| A4 G ladder | G=2048 (k=16), G=8192 (k=32) | consolidation vs splitting vs packing-clique onset (interim: 4-clique Jaccard >0.9 at G=4096; 8192 = Phase-1 stretch ratio) |

## Analysis pass (after either tier; Mac or jobe)

The measurement scripts were parameterized for 0.9.6 (`--store`, `--runs
name=path`, `--model`, `--tag`):

```bash
# 1b tier A (on jobe; store/model defaults are the 1b ones):
python scripts/analysis/eval_activation_stats.py \
  --runs seed2=/data/runs/bcc-phase096/bsc_lam0.001_seed2_lr0.0012 ... \
# calendar probe codes for new checkpoints (reuses the saved probe acts —
# just encode: see the base_lr3e-4/winner_seed1 pattern in the session, or
# rerun the full probe with --runs; ring test = fig_probe.py machinery)
python scripts/analysis/calendar_probe.py --runs name=path ...

# 4b pilot (model + store + tag matter; site list rides with the whitener):
python scripts/analysis/calendar_probe.py \
  --model google/gemma-3-4b-pt \
  --store /data/stores/bcc-pilot4b/gemma3_4b_8site_fineweb \
  --runs pilot=/data/runs/bcc-pilot4b/bsc_lam0.001_seed0_lr0.0012_G4096_k32 \
         pilot_renorm=... pilot_scalar=... \
  --tag _pilot4b --batch-rows 8
python scripts/analysis/eval_activation_stats.py \
  --store /data/stores/bcc-pilot4b/gemma3_4b_8site_fineweb \
  --runs pilot=... --out /data/runs/bcc-analysis

# unsupervised discovery (new: planarity screen — rank planar blocks from
# evalstats, then decode top-activating contexts; READ the output JSON):
python scripts/analysis/planarity_screen.py --stage rank \
  --evalstats /data/runs/bcc-analysis/evalstats_pilot.npz
python scripts/analysis/planarity_screen.py --stage decode \
  --checkpoint /data/runs/bcc-pilot4b/bsc_lam0.001_seed0_lr0.0012_G4096_k32 \
  --model google/gemma-3-4b-pt \
  --store /data/stores/bcc-pilot4b/gemma3_4b_8site_fineweb
```

Geometry extraction (`extract_geometry.py`) has a hardcoded 1b RUNS map —
extend it with the new run dirs when regenerating figures. Ring-test
methodology (capitalized-only labels, adjacency-hit statistic,
permutation null) is documented in the interim findings doc, Methods.

**What "tier A success" looks like:** ≥3 of 5 additional seeds showing
single-block 12/12 at 16M tokens, OR epoch-8 consolidating seed 1 —
either turns the paper's consolidation row into a distribution. **What
"pilot success" looks like:** all D13 gates green + any month-selective
block at 4b (ring order a bonus at 8M optimizer tokens — the 4b stream's
ring band sits early per Phase 0.5, and the pilot's site list covers it).

## Parked (a9, 2026-07-17)

- **Color-word hue-circle probe** — parked until a proper 4b run exists;
  revisit post-Phase-1. (The rest of the manifold zoo — digits, number
  words, abbreviations, compass, seasons, ordinals, alphabet, geography —
  also waits; the label-map machinery in `phase0/labels.py` +
  `calendar_probe.py` generalizes directly when wanted.)
- Publication path (discussed 2026-07-17): 0.9.6 results → zoo breadth +
  planarity screen → block-23 real export + saklas steering demo → draft.
  The 0.9.6 outcome decides whether the consolidation row is ready.
