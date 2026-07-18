# Runbook — Phase 0.9.6 (2026-07-17, a9-directed)

**What this is.** a9 bundled two tiers into one pre-NVMe campaign:
**tier A**, a 1b consolidation/robustness matrix on the existing 0.9
store, and **tier B**, the design-mandated **D13 4b exact-config pilot**,
which fits on jobe's existing `/data` disk (~290 GB stored vs ~600 GB
usable above the ShardWriter floor — only the 53M-token production store
needs the 4 TB NVMe).

**LAUNCHED 2026-07-17.** Tier A completed same day (results:
[`findings-phase096-tier-a.md`](findings-phase096-tier-a.md) — capture
consolidation universal, calendar order a 2/6 seed lottery, epochs
don't help at winner lr, G8192 not tame at 3.6 % dead). Decision map
resolved: **`PILOT_EPOCHS=2 PILOT_EXTRA_SEED=1 PILOT_G8192=0`**; tier B
launched 18:59 with those knobs. Two same-day fixes: cusolver batched-
eigvalsh chunking (`efb4f9b`, G8192 step-0 crash) and `--calib-batches
64` on the tier-A G8192 line (θ-calibration OOM at 128 batches on 61 GB
host RAM; recovered via `--resume` without retraining).
Motivating context: `docs/findings-interim-artifact-analysis.md` (the
2026-07-17 interim sweep — block-23 month ring, consolidation-vs-optimizer
open question, F7 mechanism, renorm/lr confound).

**Session note:** this runbook is written to survive a context compaction —
it carries everything needed to launch, monitor, and measure.

## Launch commands (jobe)

Both scripts are `nohup`-able and idempotent-ish (tier B skips a completed
harvest; a failed run exits the script — relaunch after diagnosis,
completed runs are skipped only by deleting/renaming their `run()` line).
**One 4090: run the tiers sequentially, not concurrently.**

**Recommended order (a9 + fable, 2026-07-17): tier A first.** Tier A is
~2 h and its results set tier-B training knobs (decision map below); the
pilot *store* doesn't depend on tier A at all, so the only cost is ~2 h
of pilot latency:

```bash
ssh jobe
cd ~/Work/transformer-experiments/block-crosscoder-experiment
mkdir -p /data/runs/bcc-pilot4b /data/runs/bcc-phase096
nohup bash scripts/run_phase096_tier_a.sh > /data/runs/bcc-phase096/matrix.log 2>&1 &
# ... read tier A (consolidation ring tests), set the knobs, then:
PILOT_EPOCHS=2 PILOT_EXTRA_SEED=0 PILOT_G8192=0 \
  nohup bash scripts/run_phase096_pilot4b.sh > /data/runs/bcc-pilot4b/pilot.log 2>&1 &
```

## Tier-A → tier-B decision map

Read tier A's ring tests (calendar-probe encode + the adjacency-hit
statistic on each new checkpoint), then set the pilot knobs:

| tier-A observation | knob |
|---|---|
| A2: ep4/ep8 consolidates seed 1 or merges the lr-3e-4 two-block split | `PILOT_EPOCHS=4` (or 8); the cosine schedule stretches automatically |
| A2: epochs don't help (unique tokens were the binding factor) | leave `PILOT_EPOCHS=2` — the store already carries 6M unique train tokens as insurance |
| A1: consolidation is seed-dependent (fewer than ~3/5 new seeds at 12/12) | `PILOT_EXTRA_SEED=1` (second seed of the BSC primary) |
| A3: renorm @ 1.2e-3 consolidates as well as baseline | treat the renorm arm as the *analysis* primary for the F7 pin (both arms always run; no script change) |
| A4: G=8192 at 1b healthy (dead dynamics tame, no packing bloom) | `PILOT_G8192=1` (adds the Phase-1 stretch config) |

## Tier B — the D13 4b pilot (`scripts/run_phase096_pilot4b.sh`)

| stage | what | expected |
|---|---|---|
| harvest | `harvest_pilot4b_store.py`: gemma-3-4b, **sites (9,12,15,18,21,24,27,30)**, 2M whitener + 2M calib + 1M eval + 6M train, whitened bf16 → `/data/stores/bcc-pilot4b/gemma3_4b_8site_fineweb` | hours (4b forward, ~373 GB written); watch tok/s + io-wait lines |
| verify | `verify_phase09_store.py --store …` | checksums + whitening round trip green |
| bsc_primary | G=4096×b=4, k=32, lr 1.2e-3 cosine, λ=1e-3, SASA AuxK; split `--max-steps 900` → `--resume` (checkpoint/resume gate at production config) | 2929 steps at ep2 (1464/epoch over 6M); resume must continue bit-compatibly |
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

## Site density (asked and answered, 2026-07-17)

a9 asked whether the pilot should sample **all layers** instead of the
spaced 8. Decision: no for tier B — (i) D13's purpose is rehearsing the
*exact* Phase-1 config, and 8 sites is part of that config; (ii) the
bytes fail twice: all-34 at 4b is 174 KB/token (~1.6 TB pilot store; the
production 53M store would be ~6 TB, exceeding even the new NVMe), and
34-site parameter stacks push past the 4090; (iii) adjacent-layer frames
are already ~0.93-aligned at gap 2 in the trained 1b dictionary — gap-1
sites mostly buy near-duplicate frames while thinning the Gram budget.
The underlying question (is the L13→L17 shear zone a wall or a
gradient?) is worth answering as a **1b all-26-layer side-study**:
60 KB/token, so a 2–3M-token store is 120–180 GB — fits on /data even
after the pilot, trivial post-NVMe. Parked behind 0.9.6.

## Parked (a9, 2026-07-17)

- **Color-word hue-circle probe** — parked until a proper 4b run exists;
  revisit post-Phase-1. (The rest of the manifold zoo — digits, number
  words, abbreviations, compass, seasons, ordinals, alphabet, geography —
  also waits; the label-map machinery in `phase0/labels.py` +
  `calendar_probe.py` generalizes directly when wanted.)
- **1b all-layer depth-resolution store** (above) — behind 0.9.6.
- Publication path (discussed 2026-07-17): 0.9.6 results → zoo breadth +
  planarity screen → block-23 real export + saklas steering demo → draft.
  The 0.9.6 outcome decides whether the consolidation row is ready.
