# block-crosscoder-experiment — design v3.0

**A block-sparse crosscoder (BSC) — dictionary learning whose atomic
unit is a subspace, with one shared code across layers.** The
literature is a 2×2, {scalar, block} × {single-site, cross-site},
whose fourth cell was empty as of 2026-07-15. Phase 0 (the pilot
program) filled it on gemma and measured that the combination earns
its parameters: the tying × blocking interaction is positive, and the
site-renormalized BSC strictly dominates the matched scalar frontier
on the honest rate–distortion axis everywhere they overlap. The
condensed evidence is [`findings-phase0.md`](findings-phase0.md);
the literature digest is
[`research/block-sparse-crosscoders-2026-07.md`](research/block-sparse-crosscoders-2026-07.md).

**This document is the forward-facing design**: the settled
architecture and parameters, what remains open, and Phase 1 — the
production run. Its ancestor (v1 → v2.4, four adversarial review
rounds, the full decision log) is preserved verbatim at
[`archive/design-v2.4.md`](archive/design-v2.4.md); bracketed
references like (R13) or (F7) index findings in the archived reviews.
Where this document is silent, v2.4 governs.

*Provenance: v3.0, 2026-07-19 — the post-Phase-0 rewrite (a9-directed
restructure). No frozen surface changed; the pinned stack, gates, and
protocol below are v2.4's, re-organized around Phase 1.*

## Hypotheses

- **H1 (rings exist).** Gemma carries irreducible multi-dimensional
  token-level features. *Phase-0 verdict: sharpened split — the month
  ring exists decoder-side in a 65k SAE (p ≈ 1e-4) below every
  clustering threshold, so post-hoc blockification cannot reach it;
  trained BSCs capture it as single blocks.*
- **H2 (cross-layer coherence).** Subspaces persist across depth
  while frames rotate, and paired tokens occupy corresponding
  positions. *Passed pre-training (code-map R² 0.83–0.90); the
  trained-BSC form routes through the shared-code evals in Phase 1.*
- **H3 (blocks earn their parameters).** The BSC's held-out
  rate–distortion frontier dominates a matched scalar crosscoder's.
  *Pilot preview strongly positive (renorm dominates across
  ~390–1,600 bits); Phase 1 at 38M tokens is the verdict.*
- **H4 (depth-resolved geometry).** Per-block per-site effective
  linear span dimension localizes where structured geometry lives.
  *Instruments validated; confirmatory numbers are Phase 1's.*
- **H5 (manifold-level diffing).** Sites = layers × models answers
  provenance questions. *Deferred — Phase 3 stub.*

Gates are decision heuristics with demonstrated power, not
falsifications: every null is reported against the positive controls
and synthetic recovery that calibrate it. A mostly-rank-1 verdict at
Phase 1 is a publishable corroboration of the flattening line.

## Architecture (settled)

The full algebra with derivations is archive/design-v2.4
§Architecture spec; this is the operative summary.

**Coordinates.** All model mathematics lives in whitened per-site
coordinates. Per site: shrinkage whitener W_s = (Σ_s + λ_s I)^{-1/2}
fit on a dedicated 5M-token slice (fp64 accumulation, TF32 off),
frozen and hashed into every shard; ridge = mean eigenvalue
(saklas `LayerWhitener` convention). The store holds whitened bf16;
**fp16 is banned everywhere in the harvest path** (gemma-3 late-layer
channels overflow it). **Gauge: site-renorm (F7, designated)** — a
per-site scalar RMS renormalization after shrinkage whitening,
folded into W_s and the whitener hash at the production harvest,
restoring equal total site power (the bare shrinkage whitener leaves
an ~8× deep tilt in pooled sq_tot; findings C7).

**Model.** G blocks of width b; per site encoder E_g^s and decoder
D_g^s; encode z_g = Σ_s E_g^s x̃^s; **Gram constraint
Σ_s D_g^s D_g^sᵀ = I_b** enforced by fp32-master retraction after
every optimizer step (batched b×b eigh, λ_min floored at 1e-6,
post-cast residual logged). The one constraint kills the scale gauge,
reduces within-block gauge to O(b), makes ‖z_g‖² the block's exact
contribution energy, and structurally blocks the decoder death
spiral. **Selection**: BatchTopK over blocks by ‖z_g‖ (training);
fixed threshold θ at inference, fit on the calibration split by
streaming log-histogram quantile and serialized with the codec.
**Loss**: L = L_rec + λ·R_rank + α·L_aux, reductions pinned (mean
over tokens/sites/dims; R_rank = per-site nuclear norm on the fixed
spectrum budget, normalized to [0, √S−1]; squared error). **Init**:
Gaussian + one retraction; E = Dᵀ at init; encoder scale
norm-calibrated. **Decoder weight decay 0** (retraction undoes it);
encoder decay 0 (measured no-op).

**Baseline.** The matched scalar crosscoder is the b=1
Gram-constrained special case of this architecture (signed latents,
energy selection) — G·b latents, matched training-average L0
(E[ℓ] = b·E[k]), identical everything else, cold start both. The
primary H3 comparison runs both at λ=0.

**Known limits, worn openly.** Decoder spectra are frame capacity,
not used dimension — rank claims come only from code-anchored
readouts (contribution second moment, truncation ablations). Block
width is a packing budget: sub-width co-active features pack
losslessly into one block as a *converged optimum* (signature:
full span overlap, ≈50/50 contribution-energy split, degraded
code-R² — the packing flag). Perfect co-activation is observationally
equivalent to a block; the coherence battery certifies only that
*curved* structure is never hallucinated. The per-site nuclear
penalty prefers site-exclusivity; the Phase −1 quantitative veto
bounds it (admissible λ = {3e-4, 1e-3}).

**Shared-code validity.** Mandatory on every trained BSC:
site-dropout encoding (raw + calibrated), leave-one-site-out,
cross-site code agreement (CCA / Procrustes-R², rotation-invariant).
Blocks that fail are "correlated bundles," never shared manifolds.

## Settled parameters

### The Phase-1 pinned training stack (v2.4, a9-ratified 2026-07-18/19)

| component | pinned value | evidence (findings-phase0 / archive) |
|---|---|---|
| model / sites | gemma-3-4b, residual-post layers **(9, 12, 15, 18, 21, 24, 27, 30)** | site list ratified 07-18; brackets the early stream (C1) |
| config | **G=4096 × b=4, k=32** (16,384 latents, ~671M params, ~9 GB train VRAM) | pilot config; G=8192 is the stretch decision (open, below) |
| optimizer | **lr 3e-4 cosine**, 1k-step warmup, enc-wd 0, 8-bit Adam, batch 4096, fp32 master + retraction | 1.2e-3 destroys at 4b, 6e-4 marginal; zero guard events at 3e-4 across every campaign run |
| λ | **1e-3** (largest admissible; H3/frontier comparisons at λ=0) | Phase −1 veto + λ ~free at 4b (Δ ≤ 0.0002 both gauges) |
| gauge | **site-renorm** | three independent lines (C7) |
| AuxK | SASA C.1 + **aux-ratio-cap 1.0** | bit-inert healthy; defuses the cascade 100×+; restores functional revival (C9) |
| loss-spike guard | **mandatory**: grad > 20× AND rec > 5× trailing accepted-median (window 50); skip advances scheduler; > 5 consecutive → refuse; **skip-rate ≤ 0.1% is a run gate** | exact failure-mode partition (C9); the guard must never make an unstable lr look stable |
| θ | streaming log-histogram quantile over the **full 13M calib split** | full-split θ closer to target k; 19.5 GB on the old 61 GB OOM case |
| data path | store-reader prefetch 4; site-subset view for single-site cells | data-wait 30% → 12% |
| seeds | 2 confirmatory at the operating points | 4b FVU endpoint seed-deterministic (spreads ≤ 0.0009); seed risk lives in capture metrics → sealed panel |
| eval | threshold-mode primary, realized counts; codec q ∈ {4, 6}; bf16 shadow | q=6 transparent; count model non-load-bearing |

Standing 4b facts: training is **bit-deterministic across runs**;
healthy dead band at G=4096 ≈ 0.1–0.15%; support amortization ≈ 4×
across k; store costs 40,960 bytes/token exactly.

### Parameters left to settle

- **Tranche 5 — the lr point, last look** (guarded ladder {4.5e-4,
  6e-4} renorm-first, no 9e-4). Re-opening 3e-4 is an a9 decision
  against the runbook's six-condition bar (replication, predeclared
  aggregate endpoint, dead/skip tolerances, revival parity,
  sealed-panel ≥ control, step-1600 clean). Until met, 3e-4 stands.
- **G=8192 stretch**: deferred to a production-store decision. Not
  tame at 1b (3.6% dead, 36× the healthy band); needs the streaming-θ
  path (landed) and a dead-dynamics look at 4b before commitment.
- **Epochs vs fresh data** — **settled** (tranche 6, 2026-07-19,
  findings §C10): at matched 24M optimizer tokens, fresh tokens beat
  epochs by only 0.0013 pooled FVU in the clean primary comparison;
  the budget itself buys −0.016 to −0.020 per doubling. Phase-1
  consequence: if the 38M budget ever binds, store passes are a
  near-free substitute for unique tokens; consolidation-vs-budget is
  a non-issue at this scale. (The renorm×fresh cell was spike-marred
  — one guarded skip — and is not read; see §C10.)
- **Frontier ends** (optional, cheap): block k=128 (the ≥2.9 kbit
  region), scalar k≈6–8 (the ultra-cheap region), one renormed joint
  scalar cell (gauge symmetry in the tying comparison).
- **Sealed-panel unsealing** happens at Phase-1 config freeze — one
  unsealing, consumed deliberately (runbook-phase099 §Tranche 0; the
  panel and its fixtures stay blind until then).

## Phase 1 — the production run

**Store commit** (waits only on the 4 TB NVMe install):

1. Mount the NVMe; record the mount point here and in the workspace
   AGENTS.md.
2. Harvest 53M tokens ≈ **2.171 TB** (38M train + 2M eval + 13M
   calibration), 8 sites, whitened bf16, seq 1024, FineWeb-Edu with
   pinned manifest + HF revision, BOS/pos-0/1 dropped. The 5M-token
   whitener slice is harvested first (never stored); **the renorm
   scalars fold into W_s and the hash**. Measured expectations:
   ≈ 3 GPU-hours at 5,000 tok/s, writer 205 MB/s, ≥15% free-space
   floor with byte-exact pre-write abort checks.
3. Shard integrity: whitener hash in every header, content checksums,
   non-zero/finiteness audits, a retained raw shard for round-trip
   verification. Sequential buffered shuffle only (no token-random
   mmap); 2–8 GiB shards, [token, site, d] bf16.
4. The D13 exact-config pilot is **discharged** (0.9.6 tier B passed
   at 3e-4); the store-commit gate is the harvest-integrity checklist
   plus one resumed-checkpoint smoke run from the new store.

**Run matrix** (staged; the 4b headline runs are not a search):

- Headline: **renorm BSC** at the pinned stack, 2 seeds; **b=1
  scalar baseline** matched, 2 seeds, λ=0 both for H3.
- λ=0 frontier k ∈ {16, 32, 64} both arms (1 seed each; positions
  are seed-stable at CI width) for the production R-D figure.
- Secondary: primary-gauge arm (1 seed) for the gauge-comparison
  line; λ ∈ {0, 3e-4} arm on renorm (the admissible-set lower arm).
- Escalation path if rare-feature starvation shows (dead fraction
  high or held-out FVU still falling at epoch end): interleaved
  harvest+train streaming for ≥100M tokens — and the baseline
  retrains on the identical extended manifest (R25).

**Eval battery** (every trained model):

- Pooled + per-site FVU (threshold mode, realized counts, bf16
  shadow); learning curves at store fractions 25/50/100% (R24 sample
  power discipline).
- The preregistered R-D codec (canonical orientation, calib-quantile
  clipping, frozen count model, enumerative support bits, sequence
  bootstrap) — the H3 verdict is frontier dominance over the shared
  region under this declared codec.
- Shared-code evals (site-dropout raw + calibrated,
  leave-one-site-out, CCA/Procrustes) — the H5-style stories are told
  only over blocks that pass.
- H4 readouts: contribution-spectrum histograms with full spectra,
  λ-sensitivity, 2-seed stability at block *and* global-subspace
  level (principal angles), rank-r truncation ablations as the final
  arbiter.
- **Sealed-panel capture score at unsealing** (the confirmatory
  capture readout); known-family consolidation probes as descriptive
  monitoring only (burned families; mega-block rule).
- Standing monitors: dead-fraction trajectory against the Phase −1
  recovery-vs-frequency calibration (clean to f=0.01 at 10M tokens),
  skip-rate gate, Gram residual, θ drift, packing-clique census.

**Gates.** H3 in either direction is informative. Proceed toward
export (Phase 2) iff blocks resolve multi-dimensional, coherent,
*and* pass the shared-code evals. Rank claims are stated conditional
on demonstrated sample power.

## Phase 2/3 — stubs (post-publication)

Deliberately deferred until the Phase-0/1 research is written up
(a9, 2026-07-19). **Phase 2 (export bridge)**: a discovered block
exports as a multi-layer manifold artifact (per-site truncated SVD of
whitened D_g^s, contribution-energy `share`, packing flag on
near-50/50 splits); the import lands in the consumer runtime (saklas)
as a manifold-folder source, not here. **Phase 3 (cross-model BSC)**:
sites = layers × models at constant site budget; paired forwards on
identical raw sequences; corpus mix and per-model init pairing pinned
at that phase's freeze; Minder-style causal diffing is the H5
toolkit. Both inherit their full specs from archive/design-v2.4
§Phases when reactivated. Causal intervention/steering demos are
likewise deferred (a9, 2026-07-18).

## Out of scope

Cross-architecture token alignment; sphere/exotic topologies;
weakly-causal crosscoder variants; steering-quality tuning of
exports; quantized activation storage for primary runs (int8 codec
admissible only after a paired-tolerance pilot); b ∈ {2, 8}
(a rank-4 ceiling is a design choice, not a discovery).

## Risks

Structurally designed out: gauge-vacuous rank penalty,
selection-proxy error, decoder death spiral (Gram constraint); rogue
dims (whitened everything); fp16 overflow (banned); pipeline-failure
nulls (Phase −1 harness + positive control + rehearsals, all passed).
Empirically mitigated, monitored: shrinkage/decoupling artifacts
(BatchTopK + blockwise Latent Scaling); AuxK cascade (ratio cap);
optimizer instability (guard + skip-rate gate). Instrumented:
site-concentration pressure (λ-veto + λ=0 control), rank-histogram
circularity (truncation ablations), multi-view leakage (site-dropout
matrix), packing (coherence + BH + the share signature), store
overfit (held-out gap + escalation path). Honestly open: the block
prior may mismatch language (b=4 is a hypothesis; H3/H4 informative
under the null); the novelty claim survived a 13-source full-text
sweep in its narrow form, with the older multi-view/group-sparse
dictionary literature still the largest owed read before external
claims.

## Settled decisions (condensed record)

Full log with rationale: archive/design-v2.4 §Decision log.

| date | decision |
|---|---|
| 07-15/16 | Gram-constrained decoders over SASA product penalty; disk-backed whitened bf16 store on a dedicated 4 TB NVMe; BatchTopK not L1; R-D framing for H3; calibration-fit θ |
| 07-16 | Phase −1 passed under strict capture-as-written (a9); **λ=1e-3 primary** (veto admissible {3e-4, 1e-3}); SASA C.1 AuxK; packing economics accepted |
| 07-17 | site band 25–90% (early stream in); shrinkage-whitener honesty + F7 question opened; 1b optimizer optimum 1.2e-3 (superseded at 4b) |
| 07-18 | **4b lr 3e-4 cosine** (1.2e-3 destroys at 4b); **site list (9…30)**; **F7 renorm designated**; AuxK cap + spike guard required; 0.9.9 campaign chartered; burned families + sealed panel; stores purged (a9 authorization) |
| 07-19 | **aux-ratio-cap 1.0 ratified**; guard/θ/prefetch mandatory; interaction +0.011; renorm frontier dominance; tying 7.8–7.9× distortion-free; stack pinned as v2.4; repo restructured, design rewritten as v3.0 |
