# AGENTS.md

Research repo: **block-sparse crosscoders (BSC)** — dictionary learning whose
unit is a *subspace* with one shared code across layers, i.e. the unsupervised
generator of saklas's manifold artifact (shared code = discover coords,
per-layer frames = `LayerSubspace.basis`, per-layer decoder norms = baked
`share`). The {block} × {cross-site} cell of the literature's 2×2 is empty as
of 2026-07-15; this experiment fills it on gemma. Not a library — phased
experiments with explicit go/no-go gates.

## Read first

- [`docs/design.md`](docs/design.md) (v2.3, post-Phase−1 consolidation, frozen): hypotheses H1–H5, the
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

**Status: Phase −1 PASSED (2026-07-16) — battery run 6 on jobe, all
hard gates green under a9's strict capture-as-written ruling.** The
harness (primitives, trainer, gauge-correct generator, recovery
metrics, seven-scenario battery; 69 tests on CUDA) plus the capture
campaign (sweep rounds 1–8, battery runs 3–6) are documented in
[`docs/findings-phase-minus1-battery.md`](docs/findings-phase-minus1-battery.md).
Load-bearing outcomes:

- **Operating point** (findings §1): budget ratio 0.8, 10k steps × batch 1024
  (≈10M tokens), spare capacity G ≈ 2.5×F (all zoos G16). Budget is
  the driving factor and monotone: loose budgets junk-fill/tile, 0.7
  starves. The 8-bit-Adam retraction-ordering check (0.9 gate) passed
  early. Battery runs 3–4 silently ran at 3k steps (CLI shadowing,
  fixed) — always verify the report's embedded `battery_config`.
- **λ-verdict reversed at the honest operating point** (findings §2.4): the
  admissible set opens at 10M tokens — **Phase-1 primary is λ=1e-3**
  (largest admissible, per protocol; design decision log amended). The
  earlier λ=0 fallback was a 3k-scale overlap-collapse artifact.
- **Block width is a packing budget** (findings §2.2): sub-width co-active or
  low-rank features pack losslessly into one block; merging is the
  *converged optimum*, and better convergence merges more. Decoys
  re-fixtured to rank-3 twins (a9's ruling); expect packed blocks in
  production — signature: full overlap, ≈50/50 split shares, degraded
  code-R². Phase-2 `share` export must treat that signature as a
  packing flag.
- **Ring detection needs zero budget slack** (findings §2.3): the bundle
  scenario budget is pinned to block-event demand k=0.75 — per-feature
  accounting double-counts co-active features, and any slack
  junk-fills through captured rings (junk tolerance <1% of firings),
  hiding them from all-firings norm-CV.
- **AuxK separates at 10k**: SASA keeps 12/12 rare features (1–4 dead
  of 16) vs 9–12 dead for Fel/long-horizon — the C.1 default is now
  positively justified. Frequency floor: clean to f=0.01 at 10M
  tokens; below is a per-seed lottery.

Next: Phase 0 (positive control first — pinned to Bloom's 2024
GPT-2-small layer-7 SAE, observational only — then
`google/gemma-scope-2-4b-pt` blockification: cosine+spectral clustering
plus the activation-dependence branch, PCA within-cluster codes over a
token stream, hunt weekday/month rings with the full Engels battery).
Ring-hunt caveat, now quantified in vivo: bare norm-CV misses real
rings both ways — soft phase-splits score CV ≈ 0.22, and *perfectly
captured* rings under production budget slack score 0.17–0.43 — so
ring claims need span-level plus gate-conditional evidence (circular
decoding / Fourier structure per design §Phase-0), never bare norm-CV.

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
