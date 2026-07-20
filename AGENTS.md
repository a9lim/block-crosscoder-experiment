# AGENTS.md

Research repository for **block-sparse crosscoders (BSCs)**: sparse
dictionary learning whose unit is a subspace with one code shared across
layers. This is a phased experiment with explicit go/no-go gates, not a
general-purpose library.

**Phase 0** is the completed pilot program (2026-07-15 through 2026-07-19).
**Phase 1** is the production Gemma 3 4B run.

## Read first

- [`docs/findings-phase0.md`](docs/findings-phase0.md): the authoritative,
  pilot result.
- [`docs/design.md`](docs/design.md): the complete normative architecture.
- [`docs/literature.md`](docs/literature.md): intellectual lineage.
- [`figures/README.md`](figures/README.md): the current winner-scoped figure
  inventory and qualification status.

## Status (2026-07-19)

Phase 0 is complete. At matched 24M optimizer tokens, optimizer budget does
the work and fresh data changes the clean comparison by only 0.0013 FVU. The
promoted site-renormalized epoch cell reaches 0.3997 pooled FVU. The current
winner arm qualifies month, weekday, and country; its matched primary-gauge
counterpart qualifies cardinal and ordinal. A 2M-token whitener is adequate,
site-renorm scalars are stable to roughly 1%, whitened bf16 is validated,
resume/checksum drills passed, and production harvest forecasts at about
three hours plus one hour verification.

Phase 1 waits only on installation of the purchased 4 TB NVMe in jobe.
Record its mount point in this file, `docs/design.md`, and the workspace-root
`AGENTS.md` before harvest. The pinned stack is `lr=3e-4` cosine,
`λ=1e-3`, site-renormalized shrinkage whitening, SASA C.1-style AuxK with
gradient-ratio cap 1.0, mandatory spike guard with skip rate at most 0.1%,
full-split streaming threshold fitting, and prefetch 4.

[`data/winner.json`](data/winner.json) is the only current checkpoint
pointer. Block identities are checkpoint-specific and must be derived from
winner artifacts. [`data/showcase.json`](data/showcase.json) records the
two-gauge Phase-0 election; do not hard-code it into current figures.

## Code and commands

All implementations live inside `block_crosscoder_experiment/`. The unified
entry point is `bsc`; do not recreate a parallel `scripts/` package.

```bash
bsc --help
bsc harvest --help
bsc train --help
bsc train-single-site --help
bsc verify-store --help
bsc validate-codec --help
bsc capture-zoo --help
bsc probe-families --help
bsc extract-geometry --help
bsc refresh-analysis --help
```

`block_crosscoder_experiment/discovery/` owns SAE discovery, topology/null
tests, and the sealed panel. `analysis/` owns the family registry, probes,
winner-scoped extraction, and figures. `cli/` owns operational entry points.
Core model/trainer/store/codec modules remain at package root.

The canonical figure contract is one consolidated index plus exactly four
winner-arm diagnostics per zoo family:

```text
figures/<family>/{frames,flow,stream,code}.html
figures/summary/*.png
figures/index.html
```

Every family page must stamp winner run, block, top-1 count, order statistic,
qualification, and FVU. An unqualified candidate is a diagnostic, not a
captured manifold. HTMLs share `figures/assets/plotly.min.js`; never embed a
fresh Plotly runtime in every file.

## Data and repository conventions

- Use plain shared Python 3.12; no project venv and no `uv`. Install the
  workspace root once and this package editable.
- `data/evidence/` is committed compact evidence. `data/analysis/` and
  `logs/` are ignored regenerated artifacts. `data/winner.json` and
  `data/showcase.json` are committed control pointers.
- Current figure artifacts and their manifest are committed. Run
  `bsc refresh-analysis` on jobe after promoting a winner; use `bsc figures`
  only for a render-only pass over existing compact artifacts.
- The Phase-2 consumer bridge is deferred until publication and lands in
  saklas, not here. Do not import sibling experiments. Local saklas
  development may use `python -m pip install -e ../../saklas`.

## Hardware

- Training and the production harvest run on jobe's RTX 4090. Mac MPS is
  appropriate for smaller harvest/analysis. Long MPS loops need periodic
  `torch.mps.synchronize()` and zero-row guards.
- fp16 is banned in harvest and store; use raw-model bf16, transformed bf16
  storage, and fp32/fp64 statistics as designed.
- Never load checkpoints concurrently with training on jobe. Restoring fp32
  masters and Adam beside the training residency reaches the 24 GB limit.
- Sequential buffered store reads only; no token-random mmap. Every shard
  carries the frozen transform hash.
- Exact store rate is 40,960 bytes/token. Production is 2.171 TB. Host RAM is
  61 GB, so calibration and eval must stream.
- Primary config is `G=4096 × b=4 × 8 sites`, about 671M parameters and
  about 9 GB train VRAM with 8-bit Adam. `G=8192` remains reserved.
