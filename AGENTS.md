# AGENTS.md

Research repo: **block-sparse crosscoders (BSC)** — dictionary learning whose
unit is a *subspace* with one shared code across layers, i.e. the unsupervised
generator of saklas's manifold artifact (shared code = discover coords,
per-layer frames = `LayerSubspace.basis`, per-layer decoder norms = baked
`share`). The {block} × {cross-site} cell of the literature's 2×2 is empty as
of 2026-07-15; this experiment fills it on gemma. Not a library — phased
experiments with explicit go/no-go gates.

## Read first

- [`docs/design.md`](docs/design.md) (v2.3.2, post-fidelity-audit, frozen): hypotheses H1–H5, the
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
- [`docs/design-review-2026-07-17-fidelity.md`](docs/design-review-2026-07-17-fidelity.md):
  round 4 — the paper-fidelity audit (fable three-way paper→spec→code
  pass + sol counter-review; F1–F11, S1–S7) that produced v2.3.2: the
  0.9.5 calibration addendum, the pre-4b-store site-renorm decision,
  and the S-series code fixes. The verified paper ground truth
  (hyperparameters of all four parents) lives here.
- [`docs/research/block-sparse-crosscoders-2026-07.md`](docs/research/block-sparse-crosscoders-2026-07.md):
  the canonical research digest — parent papers (Fel BSF, SASA, Anthropic
  crosscoders + Minder artifacts), gap sweep, the synergy argument, full
  source provenance. Migrated from the saklas repo 2026-07-15; carries
  bracketed 07-15 review amendments; it is the literature ground truth for
  this project.

**Status: Phases −1, 0, 0.5, 0.9, 0.9.5, and 0.9.6 (incl. the D13 4b
pilot) PASSED/COMPLETE (2026-07-16/17/18). Active: Phase 0.9.9, the
a9-ratified pre-NVMe campaign
([`docs/runbook-phase099.md`](docs/runbook-phase099.md)). Phase 1
store commit waits only on the 4 TB NVMe install.** A paper-fidelity audit (2026-07-17, fable + sol
counter-review;
[`docs/design-review-2026-07-17-fidelity.md`](docs/design-review-2026-07-17-fidelity.md))
produced the v2.3.2 amendment set, **ratified by a9 2026-07-17**,
including the **0.9.5 calibration addendum** — run same day, 31 runs on
the 1b store
([`docs/findings-phase095-calibration.md`](docs/findings-phase095-calibration.md)):
**cosine schedule, lr optimum 1.2e-3 with a cliff at 2.4e-3 on both
arms** (the {1,2,3,6}e-4 grid didn't bracket; extension rungs added
mid-campaign); linear_fifth wins below 3e-4 but turns over past 6e-4;
encoder-wd is a no-op; seed noise ~5× below the winner gap; λ=1e-3
still ~free at the optimum; dead dynamics engaged at G=4096 (0.098%
mortality, k=16 stress arm skipped per the conditional); the renorm arm
reverses the per-site FVU allocation (deep→shallow) at wash pooled FVU.
**Phase-1 optimizer defaults ratified by a9 2026-07-17: lr 1.2e-3
cosine, enc-wd 0, λ=1e-3** (6e-4 the documented fallback; the 4b pilot
carries lr-point confirmation). **F7 site-renorm: a9 leans renorm** —
pinned at 4b store build. Phase 1 now waits only on the 4 TB NVMe.
The θ-calibration OOM at G=4096 was fixed in `9bb4133` (CPU-side score
accumulation; estimator unchanged); the Phase-1 streaming-quantile
carry item stands.

An **interim artifact-analysis sweep (2026-07-17, analysis-only,
exploratory — not gate evidence)** over everything on disk is written up in
[`docs/findings-interim-artifact-analysis.md`](docs/findings-interim-artifact-analysis.md)
(figures in `figures/interim/`, regeneration scripts in
`scripts/analysis/`). Headlines: a calendar probe through the 0.9.5
checkpoints found **the month manifold captured as a single block** of the
ratified winner (block 23: 12/12 calendar order, 97% class-mean variance in
its code plane, ring held at all 6 sites by the block's rotating frames
while the raw stream's top plane loses it after L17); the matched scalar
arm carries the same information without unit-level individuation (top-1
month features collapse to 4); the late-heavy depth allocation of baseline
dictionaries is mechanistically traced to shrinkage-whitener retained power
(supports the F7 renorm lean); the 2.4e-3 lr cliff is a mid-run
instability (loss-spike guard for Phase 1); a 4-block packing clique
exists at G=4096 (Jaccard >0.9). Single-block ring consolidation varies
with lr/seed — carry a known-ring-consolidation probe into Phase-1 eval.

**Phase 0.9.6 LAUNCHED 2026-07-17**
([`docs/runbook-phase096.md`](docs/runbook-phase096.md), a9-directed).
**Tier A complete same day**
([`docs/findings-phase096-tier-a.md`](docs/findings-phase096-tier-a.md)):
capture consolidation is universal (every seed/epoch/G/renorm run ends
with one block claiming ≥11/12 months top-1), but **calendar order is a
seed lottery** — ring {12,7,10,2,3,12}/12 across seeds 0–5, epochs don't
buy order at the winner lr (they can migrate block ownership, seed-1
ep8), lr 3e-4 + ep8 partially catches up (5→9), renorm@1.2e-3
inconclusive under the lottery (F7 primary deferred to 4b), G8192 not
tame (3.6 % dead, 36× G4096). Weekday null holds. Block identity is
init-determined. Two same-day fixes: cusolver batched-eigvalsh chunk
(`efb4f9b`) and G8192 θ-calibration `--calib-batches 64` (host-RAM OOM).
**Tier B (D13 4b pilot) COMPLETE 2026-07-17/18**
([`docs/findings-phase096-pilot4b.md`](docs/findings-phase096-pilot4b.md)):
**D13 passes at lr 3e-4** — the ratified 1b optimizer point (1.2e-3) is
catastrophically unstable at 4b (warmup-peak edge-of-stability seed +
SASA-AuxK cascade amplifier, no gradient clipping in the loop; scalar
spikes identically, so it's the shared stack, not the Gram machinery);
6e-4 marginal, 3e-4 clean everywhere (BSC 0.430 / renorm 0.415 / scalar
0.368 pooled FVU). All other gates green; two θ-calibration host-RAM
OOMs make the Phase-1 streaming quantile mandatory. 348 GB store on the
existing /data disk (only the production store needs the 4 TB NVMe).
Calendar probe at 4b (next morning, from saved acts after an encode
OOM): both calendar rings available in the raw stream across nearly the
whole site list (weekday 7/7 at *every* site — the 1b weekday null does
not transfer to 4b); capture consolidation NOT universal at the 12M-token
pilot budget; **renorm is the only arm capturing both families** (month
10/12 + ring at the perm floor + weekday 7/7, at the best FVU) — F7
ring-side evidence; destroyed dictionaries show consolidation-without-
order (mega-block) — never read top-1 capture without ring order + FVU;
scalar smearing replicates (7 distinct month features, 1 weekday
feature). **Ratified by a9 2026-07-18: 4b lr 3e-4 cosine, site list
(9,12,15,18,21,24,27,30), F7 renorm designated, AuxK capped for
Phase 1 (mechanism pinned at config freeze) + spike guard required**
(design decision log). Phase 1 store commit now waits only on the
NVMe install. Post-pilot analysis pass **complete 2026-07-18**
(evalstats, planarity screen, 8 PNG + 11 interactive 3D figures,
7-family zoo probe — findings doc §manifold zoo): number-lines are
depth-pervasive in the stream and straighten with depth (ordinal rho
0.91→0.99); **the cardinal number-line is captured as a single block
in both arms** (primary b2146 17/20, code order at the perm floor);
ordinals split by frequency band; digits fully individuate (one block
per digit, no line); **renorm binds numbers across notation**
(3/third, 4/fourth, 6/sixth, 7/seventh same-block) where primary
instead binds calendar granularity (March/April/May + spring);
consolidation-without-order also occurs in healthy runs (primary
weekday 7/7 top-1, ring 2/7) — the mega-block eval rule applies to
every capture claim. A **second analysis pass (2026-07-18, geometry —
findings doc §cross-depth geometry)** closed the open items: the
mid-stack shear zone (frame rotation trough L18-L21) with captured
blocks' frames tracking their stream manifold's rotation at r ≥ 0.92;
cross-depth frame coherence as a training-health signature (destroyed
run decoheres); the two arms shown to be different gauges of the same
manifolds (paired-token code maps at the perm floor, cardinal span cos
0.80); 4b packing cliques (14+4/19+2 primary/seed1 vs 5 renorm) as
cross-block tilings of one early context-detector subspace (renorm's
decodes as citation/date scaffold); the shape-space census
(`p4b_census.png`); and the number-identity blocks decoded as numeral
blocks (197x/196x/15xx years, durations, ISBNs; b1808 spelled round
decades) — block identity persisting across the renorm toggle at
shared init (both arms' b1018 = the '197x' block). A **third pass
(2026-07-18, atlas tranche — findings doc §analysis pass 3)** probed
non-1D families (color/country/element/planet, 12M fresh tokens,
cap-only label maps for country/planet): **renorm b1781 is an atlas
block** — 36/48 countries top-1, lat/lon LOO-decodable from its 4-dim
code (lat 0.34 / lon 0.15, p 1e-3; giants individuate, tail
consolidates), on a stream that carries the Gurnee-Tegmark world map
at every site (LOO lat R² 0.57–0.66; longitude washes out at L30) with
zero continent clustering; color shows **no hue wheel in stream or
dictionary** (the pilot-scale color look a9 parked — production-run
color probe still open) but a stable non-wheel geometry; the planet
sun-distance line exists in-stream (|rho| ≤ 0.88) uncaptured; element
order dips through the shear zone. New explainer figures:
block-anatomy 3D (one shared code + rotating frames with ghost
decodes, b595/b2146), the hoverable all-block BSC atlas
(`p4b_atlas.html`), the stacked world-map decode
(`p4b_worldmap_3d.html`).

**Phase 0.9.9 CHARTERED 2026-07-18**
([`docs/runbook-phase099.md`](docs/runbook-phase099.md), a9-ratified;
design decision log) — the pre-NVMe campaign, center of gravity per
the Codex sol consult: (1) Phase-1-blocking engineering (streaming
θ-quantile → spike guard with skip-rate-as-gate → AuxK cap pinned via
cascade-suppression × Phase −1 revival retention → prefetch); (2) the
**2×2 factorial completed** — new {block, single-site} "BSF cell" and
{scalar, single-site} per-site-SAE cells, config-only at exact
per-site parameter/rate matching (S=1 deletes only code tying; the
existing b=1 Gram arm is the ordinary-crosscoder cell) — the thesis
becomes a positive interaction term; (3) the preregistered R-D codec
(currently **unimplemented**) built + validated on the pilot store,
λ=0 pilot frontiers as the H3 preview; (4) ≥3 seeds per headline
cell; (5) lr recovery demoted and guarded — **calendar/zoo/atlas
families are burned as selection criteria**; a sealed six-family
panel (zodiac/US states/ranks/SI prefixes/size adjectives/alphabet)
is preregistered blind, opened only at config freeze; (6) +6M-token
store extension on /data → epochs-vs-fresh factorial at matched 24M
optimizer tokens; (7) production-harvest derisk. Causal intervention
explicitly deferred by a9. **Purge executed under a9's 2026-07-18
authorization**: bcc-phase09 + bcc-phase0 stores deleted (findings
committed, regenerable by script); /data ≈ 507 GB free; /data/runs
kept (backs committed figures).

**Tranche 1 + factorial + codec first light COMPLETE (overnight
2026-07-18 → 19, autonomous;
[`docs/findings-phase099-tranche1.md`](docs/findings-phase099-tranche1.md))**:
E1 streaming θ 3/3 green (the 61 GB OOM case now 19.5 GB, full-split θ
*better*); E2 guard suite green (silent + 4th-digit FVU replication at
3e-4; refused 6e-4/1.2e-3 — and **4b training is bit-deterministic
across runs**, spike sites exactly reproducible; both unstable lrs blow
mid-run, not at warmup peak); E3 two-axis verdict — static α eliminated
(revival-killing), frac cap dominated (not inert when healthy —
perturbs from the first dead block — weaker suppression, FVU-neutral),
**ratio cap 1.0 RATIFIED by a9 2026-07-19 — the Phase-1 AuxK pin**
(design v2.4 carries the full pinned training stack: lr 3e-4 cosine,
λ=1e-3, renorm gauge, guard mandatory with skip-rate gate, streaming
full-split θ, prefetch 4, rcap 1.0): bit-inert on healthy
trajectories, engages at amplifier
ignition, crushes the SASA cascade 100×+ (peak grads 107.9→0.52,
527.7→2.53, renorm 220,670→36.9), converts self-defeating revival into
functional (final dead 3.08%→0.098%, the healthy band), best capped
FVU (0.524 vs 0.553) — but does NOT rescue bad operating points
(renorm 6e-4 destroyed either way): guard refuses seeds/runaways,
guard skips poison batches (step-1600 event proven **batch-locked**
across 3 divergent trajectories), cap defuses amplifiers — exact
partition, no overlap. **Codec first light**: q=6 transparent, q=4
+0.005 FVU; **support-bit amortization measured at 4.08×** (scalar
1,067 vs BSC 261 bits/tok support at equal amp); matched-L0 scalar
"win" costs 2.06× the bit-rate — H3 sharply posed, λ=0 frontiers
decide. **2×2 factorial complete at seed 0**: interaction term
**positive (+0.011 pooled FVU)** — tying helps blocks ~2.3× more than
scalars (BSC 0.4299 / BSF 0.4497 / scalar-cross 0.3682 / SAE 0.3768);
tying helps both geometries; single-site cells are the fair-allocation
control and **vindicate F7** (BSF/SAE per-site profiles land on the
renorm arm's, primary's deep tilt = whitener retained-power artifact).
E4/E5 dogfooded by every run (guard+prefetch+streaming θ, 16 site-runs
silent). **Second overnight campaign (03:47–09:35) completed the λ=0
frontier (k∈{16,32,64} × both arms + renorm k32) and tranche-4 seeds
(≥3 per headline cell, spreads 0.0002–0.0009; tying delta replicates
3/3; λ ~free at 4b both gauges)** — H3 preview verdict: support
amortization ≈4× across k, **renorm strictly dominates the scalar
frontier in the ~800-bit overlap region**, primary ties mid-region,
cross-site tying ≈ 7.8× rate cut vs per-site models at ≈equal pooled
FVU (`figures/phase099/rd_frontier.png`). Morning of 07-19 (a9):
**rcap pin ratified, design updated to v2.4 (pinned Phase-1 stack)**,
renorm k16/k64 frontier completion + E6 +6M store extension launched
(`scripts/run_phase099_ext.sh`). Remaining tranches: 6 (epochs-vs-
fresh, needs E6), 7 (harvest derisk), 5 (guarded lr-recovery, last);
single-site R-D plane placement (sq_tot weights) pending the sweep
tail.

- **Phase 0** ([`docs/findings-phase0-gemma.md`](docs/findings-phase0-gemma.md),
  control in `findings-phase0-control.md`): positive control recovered
  Engels' weekday ring on GPT-2; on gemma layer 22 the discovery
  verdict is the scoped null at 16k *and* 65k (BH-flagged: none; the
  supervised skeleton features are τ=0.5 graph singletons — max family
  cosine 0.32 structurally excludes rings from the candidate set). But
  the **month ring exists in the 65k dictionary decoder-side**
  (adjacency p 1.5e-4, Fisher–Lee angle order p 3.5e-4): splitting
  completes at 25.6× and the ring rides at cosines below every
  clustering threshold. H1's artifact statement is positive even
  though post-hoc blockification can't reach it.
- **Phase 0.5** ([`docs/findings-phase05-cross-layer.md`](docs/findings-phase05-cross-layer.md),
  layers 9/17/22/29 @ 65k, paired 4M-token stream): **gate passed in
  the BSC-native form** — month codes correspond linearly across
  depths (held-out code-map R² 0.83–0.90 for 9↔22↔29, all CCA ≥ 0.96
  for 9↔22) while raw-basis decoder spans sit at chance alignment;
  22→29 also span-matches. Frames rotate, the code persists — the
  architecture's premise observed pre-training. Depth rewrite:
  activation-space rings live *early* (layer-9 weekday: circ 0.981 on
  the top plane AND decoder |r| 0.886 — both spaces, still max
  adjacent cos 0.27 < τ); layer 22 is the ring-visibility minimum,
  not a representative depth. Layer 17's 65k SAE undersplits both
  families — never judge structure through a single site's
  dictionary. Phase-1 site selection must bracket depth including an
  early (layer-9-like) site. A bonus layer-9 discovery run nulls
  identically (BH none; 15/17 skeleton feats τ=0.5 singletons): the
  discovery gap is depth-general, which sharpens H1 at every probed
  depth.
- **Phase 0.9** ([`docs/findings-phase09-rehearsal.md`](docs/findings-phase09-rehearsal.md),
  gemma-3-1b, 6 sites {7,10,13,17,20,22}, 13M-token whitened store on
  jobe): **all plumbing gates green** — store checksums + bit-exact
  whitening round trip, whitener stability ≤0.025 (halves) at 2M fit
  tokens, held-out spectrum within 0.047 of the shrinkage prediction,
  production-scale checkpoint/resume, θ transfer within 0.13% of
  target k, 8/8 bit-deterministic eval passes, zero dead blocks, toy
  manifold export with exact Gram-invariant energy accounting. λ
  ladder ~free at 16M tokens (FVU spread 0.07%); scalar beats BSC at
  matched latent-L0 (0.353 vs 0.422 pooled FVU) — the H3 question,
  deferred to 4b. Carried to Phase 1: streaming quantile for
  `fit_threshold_` (13M calib tokens), store-reader prefetch
  (55–70% data-wait). D13 risks explicitly not cleared.

Phase −1 detail (battery run 6 on jobe, all hard gates green under
a9's strict capture-as-written ruling): the
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

Ring-hunt caveat, quantified in vivo at Phase −1: bare norm-CV misses real
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
