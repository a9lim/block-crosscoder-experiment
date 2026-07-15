# AGENTS.md

Research repo: **block-sparse crosscoders (BSC)** — dictionary learning whose
unit is a *subspace* with one shared code across layers, i.e. the unsupervised
generator of saklas's manifold artifact (shared code = discover coords,
per-layer frames = `LayerSubspace.basis`, per-layer decoder norms = baked
`share`). The {block} × {cross-site} cell of the literature's 2×2 is empty as
of 2026-07-15; this experiment fills it on gemma. Not a library — phased
experiments with explicit go/no-go gates.

## Read first

- [`docs/design.md`](docs/design.md): hypotheses H1–H5, the architecture sketch
  (BatchTopK block selection in whitened space, per-site nuclear norm), the
  phase ladder 0 → 0.5 → 1 → 2 → 3 with gates, out-of-scope list.
- [`docs/research/block-sparse-crosscoders-2026-07.md`](docs/research/block-sparse-crosscoders-2026-07.md):
  the canonical research digest — parent papers (Fel BSF, SASA, Anthropic
  crosscoders + Minder artifacts), gap sweep, the synergy argument, full
  source provenance. Migrated from the saklas repo 2026-07-15; it is the
  literature ground truth for this project.

**Status: pre-Phase-0 scaffold.** No experiment code yet. Next action: the
Phase-0 blockification script (cluster an existing SAELens SAE's decoder
directions on gemma, PCA within-cluster codes over a token stream, hunt
weekday/month rings).

## The saklas seam

This experiment is a **producer**; saklas is the **consumer**. The contract is
the manifold folder (`manifold.json` + per-model safetensors) — the same
producer/consumer shape saklas already has with SAELens. Discovery and
training happen here; the Phase-2 import bridge (a `discovered` manifold
source) is a **saklas feature and lands in the saklas repo**, not here.

Imports from saklas (consumer-side machinery used producer-side): model
loading, `LayerWhitener` (whitened block selection is non-negotiable — raw-L2
selection on a residual stream is massive-activation bait), the `sae` runtime
for Phase-0 harvest, and `experiment naturalness` for Phase-2 eval. Develop
against a local saklas when needed: `pip install -e ../../saklas`.

Do not import sibling experiments (workspace rule). Shared model registry /
chat-template fixups come from the workspace-root `transformer_experiments`
package.

## Conventions

- Workspace rules apply: shared base Python 3.12, plain `python`, no venvs,
  no `uv`. Install the workspace root once, then this package editable.
- Phase discipline: each phase gates the next (gates in `docs/design.md`). Don't
  start Phase-1 training before Phase-0/0.5 have verdicts — the whole design
  is built so the expensive step is the *third* thing, not the first.
- A null result is informative at every gate (no-rings sharpens the
  flattening line; rank-1-everywhere corroborates it; cross-layer incoherence
  is a finding). Don't chase a positive.
- Results and run logs: `data/` and `logs/` are regenerated artifacts, out of
  git; findings prose goes to `docs/`.

## Hardware

- **Harvest + analysis on the M5 Max (MPS), training on the 4090 (CUDA)** —
  workspace convention, and the artifact rides over (activations/checkpoints
  are portable; fit on CUDA, analyze on MPS).
- **MPS async-OOM discipline** (learned the hard way in saklas's lens fit):
  Metal reports queue exhaustion as an *asynchronous* command-buffer error
  that silently zeroes work instead of raising. Any long unsynced MPS loop —
  the harvest loop especially — needs periodic `torch.mps.synchronize()`
  backpressure plus output validation (zero-row guards). Never trust an
  all-zeros block on MPS; suspect the queue first.
- Streamed activation harvest only (~40 KB/token at 8 sites on gemma-3-4b —
  200M tokens ≈ 8 TB raw; storage was never on the table).
- Phase-1 full config (~1.3B params untied) is tight on the 4090 with
  mixed-precision AdamW (≈16–21 GB): drop to G=4k / 6 sites / tied
  (Grassmannian) encoder / 8-bit Adam as needed. The matched scalar baseline
  is the same size — budget for both runs.
