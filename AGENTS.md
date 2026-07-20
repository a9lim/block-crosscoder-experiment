# AGENTS.md

Research repository for **block-sparse crosscoders (BSCs)**: sparse
dictionary learning whose unit is a subspace with one code shared across
sites. This is a phased experiment with explicit go/no-go gates, not a
general-purpose library.

**Phase 0.5** is the active evidentiary reset (opened 2026-07-20). Phase-0
results, winner pointers, evidence, and figures are withdrawn. Do not describe
Phase 1 as ready or use a former block/checkpoint identity.

## Read first

- [`docs/audit_2026-07-20.md`](docs/audit_2026-07-20.md): flaw/remediation
  ledger.
- [`docs/paper_comparison.md`](docs/paper_comparison.md): exact paper bridges,
  hyperparameter comparison, and staged matrix.
- [`docs/design.md`](docs/design.md): normative Phase-0.5 architecture and
  operating contract.
- [`docs/findings-phase0.md`](docs/findings-phase0.md): withdrawal notice.

## Status (2026-07-20)

The training, checkpoint, codec, normalization, provenance, and trained-
endpoint stacks have been remediated. The old large runs/stores on `jobe` and
all committed evidence/figures derived from them were removed. The executable
Phase-0.5 matrix declares an 80-cell paper-bridge screen followed, only if the
screen has no failures, by 68,220 recipe-valid factorial cells.

The active screen uses five aligned normalization stores (`none`, `scalar`,
token `layer`, shrinkage `whiten`, and `whiten_renorm`) and explicitly includes
`lr=1e-4`, longer training, lambda, selector, encoder-bias, constraint,
regularizer, Aux, dead-window, and cap factors. No configuration is promoted
until saved-codec, raw-FVU, shared-code, used-span, safety, and provenance
gates pass.

The purchased 4 TB NVMe is still not installed. The current `/data` disk can
hold the reduced Phase-0.5 stores and sequential one-checkpoint-at-a-time
matrix, but not the former 2.171 TB production plan. Record the new mount point
here, `docs/design.md`, and the workspace-root `AGENTS.md` before a later
production harvest.

## Code and commands

All implementations live inside `block_crosscoder_experiment/`; `bsc` is the
only executable surface.

```bash
bsc --help
bsc harvest --help
bsc train --help
bsc train-single-site --help
bsc verify-store --help
bsc validate-codec --help
bsc reproduce-papers --help
bsc phase05-matrix --help
bsc trained-endpoints --help
bsc refresh-analysis --help
```

`discovery/` owns SAE discovery, topology/null tests, and the sealed panel.
`analysis/` owns trained endpoints, family probes, winner-scoped extraction,
and figures. `cli/` owns entry points. Core model/trainer/store/codec modules
remain at package root. Do not recreate a parallel `scripts/` package.

## Data and repository conventions

- Use shared plain Python 3.12; no project venv and no `uv`.
- `data/evidence/` stays empty until corrected Phase-0.5 evidence qualifies.
  `data/analysis/` and `logs/` are ignored regenerated artifacts.
- Absence of `data/winner.json` and `data/showcase.json` is the valid
  pre-promotion state; analysis commands must fail clearly or require an
  explicit run rather than import-time crash.
- Generated figures remain absent until promotion. After a valid winner,
  `bsc refresh-analysis` regenerates source artifacts and figures; `bsc
  figures` is render-only.
- Decoder capacity must never be labeled used/effective dimension. Figures use
  activation-weighted contribution/centered spans and explicit eligibility.
- The Phase-2 consumer bridge remains deferred and belongs in saklas.

## Campaign operations

```bash
bsc phase05-matrix status --root /data/runs/bcc-phase05
bsc phase05-matrix campaign \
  --root /data/runs/bcc-phase05 \
  --store-root /data/stores/bcc-phase05
```

The campaign is resumable. `campaign_state.json`, `harvest_state.json`, and
`state.json` are authoritative. Screen failure stops the full factorial.
Successful non-promoted cells retain reports/logs/manifests and discard only
their large `latest.pt`. Never delete a non-final checkpoint unless the job
report is complete.

## Hardware

- Training/harvest run on `jobe`'s RTX 4090. Mac MPS is suitable for smaller
  analysis; long loops need synchronization and zero-row guards.
- fp16 is banned in harvest/store. Use model bf16, store bf16, transform fp32,
  statistics fp64.
- Never load checkpoints concurrently with training on the 24 GB GPU.
- Sequential buffered reads only; no token-random mmap. Every shard carries a
  frozen transform hash.
- `prefetch=4` is queue depth, with producer/current-shard residency in
  addition; early exit must close the iterator.
- Exact eight-site bf16 store rate is 40,960 bytes/token. Host RAM is 61 GB;
  calibration and eval stream.
