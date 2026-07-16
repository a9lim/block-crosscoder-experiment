# AGENTS.md

Research repo: **block-sparse crosscoders (BSC)** — dictionary learning whose
unit is a *subspace* with one shared code across layers, i.e. the unsupervised
generator of saklas's manifold artifact (shared code = discover coords,
per-layer frames = `LayerSubspace.basis`, per-layer decoder norms = baked
`share`). The {block} × {cross-site} cell of the literature's 2×2 is empty as
of 2026-07-15; this experiment fills it on gemma. Not a library — phased
experiments with explicit go/no-go gates.

## Read first

- [`docs/design.md`](docs/design.md) (v2.2, post-round-3, frozen): hypotheses H1–H5, the
  architecture spec (Gram-constrained decoders — Σ_s D_g^s D_g^sᵀ = I_b —
  BatchTopK block selection by exact whitened contribution ‖z_g‖, per-site
  nuclear norm on a fixed spectrum budget), the phase ladder
  −1 → 0 → 0.5 → 0.9 → 1 → 2 → 3 with gates, configs, the rate–distortion
  protocol, out-of-scope list, decision log.
- [`docs/design-review-2026-07-15.md`](docs/design-review-2026-07-15.md):
  the adversarial review disposition (Codex sol-tier + parallel pass, 35
  findings) that produced v2 — the *why* behind every load-bearing spec
  choice. Read before re-litigating any of them.
- [`docs/design-review-2026-07-16.md`](docs/design-review-2026-07-16.md):
  round 3 — two fresh-context sol passes (deployment/design D1–D14 on
  measured jobe hardware; paper-fidelity P1–P25 over all 13 reference
  full texts) + fable parallel pass. Produced v2.2: the deployment
  re-plan (4 TB NVMe, whitened store, calibration split), the
  gauge-corrected Phase −1 generator, the SASA-based AuxK respec, and
  the pinned Phase-0 positive control.
- [`docs/research/block-sparse-crosscoders-2026-07.md`](docs/research/block-sparse-crosscoders-2026-07.md):
  the canonical research digest — parent papers (Fel BSF, SASA, Anthropic
  crosscoders + Minder artifacts), gap sweep, the synergy argument, full
  source provenance. Migrated from the saklas repo 2026-07-15; carries
  bracketed 07-15 review amendments; it is the literature ground truth for
  this project.

**Status: Phase −1 harness complete and battery run twice on jobe
(2026-07-16) — primitives, trainer, gauge-correct generator, recovery
metrics, seven-scenario battery, all unit-tested (69 tests on CUDA).
Verdicts in
[`docs/findings-phase-minus1-battery.md`](docs/findings-phase-minus1-battery.md):
λ=0-primary confirmed (empty admissible set; the run-1 "share
concentration" was largely a Frobenius parked-capacity artifact — the
real λ harm is overlap collapse); rank ≤3 gaussians recover exactly,
shells tile, spare capacity splits — one capture/tiling/splitting
lottery behind every red gate. The 8-bit-Adam retraction-ordering
check (0.9 gate) passed early.**
Blocking decision (a9's): Phase −1 core gate semantics — capture-only
vs capture-where-achievable + span-recovery for shells/high-rank
(findings §3). Then: Phase 0 (positive control first — pinned to
Bloom's 2024 GPT-2-small layer-7 SAE, observational only — then
`google/gemma-scope-2-4b-pt` blockification: cosine+spectral clustering
plus the activation-dependence branch, PCA within-cluster codes over a
token stream, hunt weekday/month rings with the full Engels battery —
noting the tiled-ring caveat: norm-CV only flags *captured* rings, so
the ring hunt needs span-level evidence too).

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
- **Disk-backed whitened activation store, not streaming** (decision
  2026-07-15; re-planned 2026-07-16 — jobe's stock disks are 2×1 TB and
  could not hold the store; a dedicated **4 TB NVMe** is purchased for it):
  harvest once on the 4090 (whitener slice first, then whitened bf16,
  8 sites, ~40 KB/token) into a 53M-token store — 38M train + 2M eval +
  13M calibration ≈ 2.17 TB — and train from the store with gemma out of
  VRAM. Sequential buffered shuffling only (no token-random mmap); whitener
  hash in every shard header. Interleaved streaming is the documented
  escalation/fallback. **fp16 is banned in the harvest/store path** —
  gemma-3 late-layer channels overflow it.
- Phase-1 primary config is G=4096 × b=4 × 8 sites untied (~671M params,
  ~9 GB train VRAM with 8-bit Adam); G=8192 (~1.34B, ~11 GB) is the stretch
  config the store makes possible. The matched scalar baseline is the same
  size — budget for both runs, 2 seeds each. A ≥3M-token exact-config
  pilot (long enough to exercise AuxK, checkpoint/resume, and threshold
  calibration — a separate mandatory gate; the 1b rehearsal cannot stand
  in for it) precedes the store commit.
