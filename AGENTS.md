# AGENTS.md

Research repo: **block-sparse crosscoders (BSC)** — dictionary
learning whose unit is a *subspace* with one shared code across
layers, filling the {block} × {cross-site} cell of the literature's
2×2 on gemma. Not a library — phased experiments with explicit
go/no-go gates. Naming: **Phase 0** = the completed pilot program
(2026-07-15 → 07-19; internally it used a −1 → 0.9.9 ladder — its
one-shot scripts live in `scripts/archive/` with a rename map);
**Phase 1** = the production run. Live tools are phase-neutral:
`train_bsc.py`, `train_single_site.py`, `harvest_store.py`,
`extend_store.py`, `verify_store.py`, `validate_{codec,theta,revival}.py`,
`run_battery.py`, `sweep_{bundle,capture}.py`; analysis probes are
`scripts/analysis/probe_*.py`, figures `fig_*.py`, regenerated via
`regen_figures.sh` from the winner pointer.

## Read first

- [`docs/findings-phase0.md`](docs/findings-phase0.md) — what Phase 0
  established, paper-shaped (claims C1–C10, H1–H5 status, standing
  rules). Read before re-deriving any result.
- [`docs/design.md`](docs/design.md) (v3.0) — the forward design:
  settled architecture + pinned training stack, open parameters, the
  Phase-1 plan. Where it is silent,
  [`docs/archive/design-v2.4.md`](docs/archive/design-v2.4.md)
  governs (full algebra, gates, decision log).
- [`docs/archive/`](docs/archive/README.md) — verbatim primary
  sources. Read the relevant one before re-litigating a settled
  choice.
- [`docs/runbook-phase099.md`](docs/runbook-phase099.md) — still
  live: the sealed probe panel (tranche 0) and the open closeout
  tranches (5, 6, 7). Archives when the campaign closes.

## Status (2026-07-19)

Phase 0 complete except closeout: **tranche 6 done** (epochs-vs-fresh:
budget does the work, fresh ≈ epochs within 0.0013; findings §C10;
winner promoted to the 24M-token renorm epochs cell, FVU 0.3997;
canonical figures regenerated from it — 5 showcase families qualify:
renorm month/weekday/country, primary cardinal/ordinal, §C10),
**tranche 7 done** (2M whitener adequate — held-out dev flat in fit
size; F7 gauge slice-stable ≤ 1%; fp16 ban quantified, whitened bf16
validated; drills passed, ~3 h production-harvest forecast; findings
§C11), **tranche 5** (guarded lr recovery, last — the
re-ratification bar is a9's). **Phase 1 store commit waits only on
the 4 TB NVMe install** (record its mount point here and in the
workspace root when live). Training stack pinned and a9-ratified
(design v3 §Settled parameters): lr 3e-4 cosine, λ=1e-3, site-renorm
gauge, SASA C.1 + aux-ratio-cap 1.0, mandatory loss-spike guard with
skip-rate ≤ 0.1% as a run gate, streaming full-split θ, prefetch 4.
The current best checkpoint is `data/phase0/winner.json` — the
dynamic pointer figure regeneration reads (block identities are
checkpoint-specific: `derive_showcase.py` re-derives them per winner).

## Standing rules (binding)

- **Sealed panel stays sealed**: the six-family probe panel
  (runbook §Tranche 0) opens only at Phase-1 config freeze or by an
  explicit a9 unsealing. Never set `BCC_PANEL_UNSEALED`; no
  stream-side availability checks either.
- **Burned families**: calendar/zoo/atlas are descriptive probes
  only, never selection criteria — three analysis passes tuned on
  them. Confirmatory capture routes through the sealed panel.
- **Mega-block rule**: top-1 capture is never read without ring
  order and FVU beside it (healthy runs produce
  consolidation-without-order too).
- **Norm-CV is never a ring detector by itself** — ring evidence is
  span-level and gate-conditional.
- **Contribution-energy shares, never Frobenius** — decoder spectra
  are frame capacity, not used dimension.
- **Capitalization filtering for token-class probes** (the May
  lesson: lowercase 'may' is 88% modal).
- **Verify the effective config in the report artifact**
  (`battery_config` / `model_cfg`), never trust the intended CLI —
  two silent config-shadowing incidents.
- **Never judge structure through a single site's dictionary**
  (the layer-17 cautionary tale).
- A null result is informative at every gate; don't chase a
  positive. Phase discipline: each phase gates the next.
- Reserved to a9: lr re-ratification (tranche-5 bar), panel
  unsealing, store purges, gate-semantics rulings.

## The saklas seam (post-publication)

This experiment is a producer; saklas is the consumer (manifold
folder contract). **Deferred until the Phase-0/1 research is
published** (a9, 2026-07-19) — the Phase-2 import bridge lands in the
saklas repo, not here. Consumer-side machinery still imported where
useful (`LayerWhitener` convention, model loading); develop against a
local saklas via `pip install -e ../../saklas` when needed. Do not
import sibling experiments (workspace rule).

## Conventions

- Workspace rules apply: shared base Python 3.12, plain `python`, no
  venvs, no `uv`. Install the workspace root once, then this package
  editable.
- Results and run logs: `data/analysis/` and `logs/` are regenerated
  artifacts, out of git; committed compact evidence lives in
  `data/phase0/`; findings prose goes to `docs/`. Figures are
  regenerated from the winner pointer into `figures/phase0/`
  (`scripts/analysis/winner.py`); force-add (`git add -f`) the
  canonical set — `figures/**` is gitignored.

## Hardware

- **Training on the 4090 (`ssh jobe`), harvest + analysis on the M5
  Max (MPS)** — activations and checkpoints are portable.
- **MPS async-OOM discipline**: long unsynced MPS loops need periodic
  `torch.mps.synchronize()` + zero-row guards; never trust an
  all-zeros block on MPS.
- **fp16 is banned in the harvest/store path** (gemma-3 late-layer
  channels overflow it); whitened bf16 store, fp32 stats.
- **No concurrent checkpoint loads with training on jobe**:
  `Trainer.load_checkpoint` restores fp32 master + Adam (~8–9 GB)
  next to a ~15 GB training residency — 23.5 GB OOMs. Eval/codec
  passes run strictly after training drains.
- Store discipline: disk-backed whitened store, sequential buffered
  shuffling only, whitener hash in every shard header. Store facts:
  40,960 B/token; pilot store 348 GB + 6M extension on `/data`;
  production 2.171 TB on the 4 TB NVMe (purchased, not yet
  installed). Host RAM 61 GB — streaming θ is mandatory.
- Phase-1 primary config G=4096 × b=4 × 8 sites (~671M params, ~9 GB
  train VRAM, 8-bit Adam); G=8192 stretch is an open decision.
