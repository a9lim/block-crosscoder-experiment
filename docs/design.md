# block-crosscoder-experiment — design

**A block-sparse crosscoder (BSC) — dictionary learning whose atomic unit is a
subspace, with one shared code across layers — is the unsupervised generator
of saklas's manifold artifact, and within LLM interpretability nobody has
built one.** The literature is a 2×2, {scalar, block} × {single-site,
cross-site}, with three cells occupied (SAEs; BSF/SASA; crosscoders and kin)
and the fourth empty as of 2026-07-15. This experiment fills it on gemma,
measures whether the combination earns its parameters, and lands discovered
manifolds in a real steering/probe runtime.

Full literature provenance, gap sweep, and the synergy argument live in
[`docs/research/block-sparse-crosscoders-2026-07.md`](research/block-sparse-crosscoders-2026-07.md).
One paragraph of it matters here: each parent fixes the other's failure.
Blocks absorb cross-layer within-subspace rotation that fragments scalar
crosscoder latents (SASA's splitting theorem, aggravated by depth);
crosscoders solve cross-layer block identification by construction and emit
per-layer decoder norms — the depth profile saklas bakes as `share`.

**Revision provenance.** v2, 2026-07-15: rewritten after an adversarial
design review (Codex sol-tier, thread `bsc-design-review`, 35 findings /
6 blockers) plus an independent parallel pass; the finding-by-finding
disposition is in
[`docs/design-review-2026-07-15.md`](design-review-2026-07-15.md). The v1
objective was mathematically ill-posed (gauge-defeated rank penalty); v2
fixes it structurally. SASA has now been read at full text — its regularizer
is on the product `D_k E_k`, not the decoder alone, which is how it dodges
the same gauge hole; our Gram-constraint route is different but solves
selection exactness and gauge in one move (rationale in the review doc).
**v2.1, same day**: round-2 verification PASSed the Gram-constraint algebra
(all four claimed properties) and conditioned freeze on the amendments now
folded in — objective normalization conventions, the complete quantizer
spec, the explicit b=1 Gram-constrained baseline, fp32-master retraction
ordering, calibrated shared-code evals, and exact store arithmetic.
Round-2 findings are numbered R1–R26 in the review doc.

## Hypotheses

- **H1 (rings exist).** Gemma carries irreducible multi-dimensional
  token-level features (Engels-style weekday/month rings) findable by post-hoc
  blockification of an existing SAE. The saklas-side centroid-level flattening
  results don't rule this out — pooling can average a ring into a blob; they
  measure different objects.
- **H2 (cross-layer coherence).** Ring/block subspaces at different depths are
  matchable — the subspace persists while its frame may rotate — **and paired
  tokens occupy corresponding positions** in the matched subspaces. Both
  halves are load-bearing (a shared code needs position correspondence, not
  just span overlap); both are testable before any training.
- **H3 (blocks earn their parameters at language).** On a held-out
  activation rate–distortion comparison (the Fel MDL argument, made honest —
  see *Rate–distortion protocol*), a BSC's frontier dominates a matched
  scalar crosscoder's. A mostly-rank-1 dictionary is a publishable
  *corroboration* of the flattening line, not a wasted run.
- **H4 (depth-resolved geometry).** Per-block per-site **effective linear
  span dimension** (see *Rank measurement* — deliberately not called
  "intrinsic dimension": a circle is intrinsically 1-D in a 2-D linear span,
  and decoder rank measures the span) localizes *where in depth* structured
  geometry lives, cross-checkable against J-lens depth center-of-mass.
- **H5 (manifold-level diffing).** With sites = layers × models, per-model
  block presence and rank answer provenance questions: is the persona fan
  present in base gemma or installed by chat-tuning?

Gates below are **decision heuristics, not falsifications**: each null is
reported with the demonstrated power of the pipeline that produced it
(positive controls, synthetic recovery), and what it does is downgrade the
prior and reroute effort, not refute the hypothesis.

## Architecture spec

### Sites, coordinates, whitening

Sites s = 1..S are residual-post taps at selected layers (Phase 3: layers ×
models). **All model mathematics — encoding, selection, loss, rank penalty,
rank readout — lives in whitened per-site coordinates**; raw space appears
only at harvest ingress and export egress. This is the single consistency
rule that keeps selection, regularization, and the H4 readout commensurable
(review finding 7), and it designs out massive-activation/rogue-dim bait.

Whitening protocol: per site, fit mean μ_s and covariance Σ_s on a dedicated
5M-token slice of the harvest corpus (disjoint from train and eval), fp32
accumulation, ridge per saklas's `LayerWhitener` convention (λ_s =
mean-diagonal of Σ̂_s × `DEFAULT_RIDGE_SCALE`); eigendecompose once; freeze
W_s = (Σ_s + λ_s I)^{-1/2} and export it with the run config. Training-side
whitening is materialized as a dense d×d matmul (the saklas Woodbury object
is a CPU one-shot operator, not a hot-path transform). Whitened activations:
x̃^s = W_s (x^s − μ_s).

**dtype discipline: fp16 is forbidden everywhere in the harvest and store
path** — gemma-3's late-layer channels exceed fp16 max (fact preserved in
saklas `mahalanobis.py`); harvest and store in bf16, whiten and accumulate
stats in fp32.

### Model

G blocks of width b. Per site: encoder E_g^s ∈ ℝ^{b×d}, decoder D_g^s ∈
ℝ^{b×d}, decoder bias c^s ∈ ℝ^d (init 0). Per token:

- **encode** — z_g = Σ_s E_g^s x̃^s (summed per-site maps; crosscoder
  convention; untied by default).
- **gauge constraint (load-bearing)** — per block, the concatenated decoder
  Gram is the identity:
  **Σ_s D_g^s D_g^sᵀ = I_b.**
  Enforced by retraction after every optimizer step: M_g = Σ_s D_g^s D_g^sᵀ,
  D_g^s ← M_g^{-1/2} D_g^s (batched b×b eigh, fp32). This one constraint
  simultaneously (i) kills the z↦cz, D↦D/c scale symmetry that made the v1
  rank penalty vacuous (finding 1), (ii) cuts the within-block GL(b) gauge
  down to orthogonal rotation, which spectra are invariant to (finding 2),
  (iii) makes ‖z_g‖² *exactly* the block's **individual contribution
  energy** Σ_s ‖D_g^sᵀ z_g‖² — the selection score is no longer a proxy
  (finding 3; note this is the block's isolated output energy, not its
  marginal loss reduction — blocks can overlap or cancel, same caveat as
  every SAE-family norm score), and (iv) structurally blocks the dead-block
  death spiral, since no block's decoder can shrink toward zero (finding
  29). Per-site Frobenius shares tr(D_g^s D_g^sᵀ)/b remain free — the depth
  profile survives.

  Retraction mechanics (round-2 findings R8–R11): the retraction operates on
  the **fp32 master weights** in this order — optimizer step on master →
  retract master decoders → regenerate bf16 forward copy → log post-cast
  Gram residual ‖M_g − I‖ (a training-health metric; large residual after
  init indicates trouble). λ_min(M_g) is floored (default 1e-6) before
  inversion, with floor hits logged; verify the ordering against the chosen
  8-bit-Adam implementation before Phase 0.9. **Decoder weight decay is 0**
  (uniform shrinkage is undone by retraction and only injects noise); decay
  applies to encoders only. Coordinatewise Adam is not equivariant to the
  residual O(b) rotation, so Phase −1 includes a rotation-equivariance test
  (train paired seeds from randomly rotated inits; compare recovered
  spectra/subspaces); if divergence is material, decoders move to
  projected-momentum SGD while encoders keep Adam.

  One structural consequence, worn openly (R6): the constraint forces every
  block's concatenated frame to rank exactly b — a genuinely lower-rank
  feature parks its surplus directions somewhere in site space with the
  encoder suppressing their code variance. Decoder spectra are therefore
  **frame capacity**, not used dimension; rank claims come only from the
  code-anchored readouts below.
- **select** — BatchTopK over blocks by p_g = ‖z_g‖₂ (exact contribution
  energy, comparable across blocks because of the constraint). Training:
  keep the top k·B block-activations across the batch of B tokens (per-token
  counts vary). Inference: fixed global threshold θ on p_g, estimated during
  training as an EMA of the batch-minimum selected score (BatchTopK
  convention). BatchTopK, not L1: Minder et al. (2504.02922) show L1
  manufactures Complete Shrinkage / Latent Decoupling; the block-level
  analogues would poison exactly the H5 diffing questions.
- **decode** — x̂̃^s = c^s + Σ_{g active} D_g^sᵀ z_g; raw-space export
  un-whitens (x̂^s = W_s^{-1} x̂̃^s + μ_s).
- **loss** — L = L_rec + λ_* R_rank + α L_aux, with reductions pinned so λ
  and α are meaningful across configs (round-2 blocker R12):
  L_rec = mean over tokens, sites, and dimensions of the squared whitened
  residual; R_rank = mean over blocks of (Σ_s ‖D_g^s‖_* − b)/b — under the
  constraint the per-block sum ranges over [b, b√S] (b = every code
  direction site-exclusive, b√S = flat across sites), so the normalized
  penalty lives in [0, √S−1] with 0 = fully site-concentrated; L_aux = the
  AuxK residual reconstruction under the same reduction as L_rec.

### Rank regularizer

R_rank = Σ_g Σ_s ‖D_g^s‖_* — per-site nuclear norm of the **whitened**
decoder, computed from batched b×b Gram eigenvalues (Σ√(eig+ε), fp32; no
d-dimensional SVD anywhere in the loop). Under the Gram constraint the total
spectrum budget per block is fixed (Σ_s Σ_i σ_i² = b), so the penalty can
only *reshape* spectra toward per-site low rank — it can no longer be gauged
away.

Known residual bias, accepted and instrumented (finding 4, sharpened by
round-2 R7): the penalty's unconstrained preference is **strongly
site-exclusive** — its minimum (normalized 0) is attained by assigning each
code direction to a single site, and the flat shared configuration sits at
the maximum (√S−1). Reconstruction must overpower that preference for
genuinely shared structure to survive nonzero λ. Mitigations are therefore
gates, not notes: (i) the Phase −1 harness plants *shared, flat-profile*
blocks and imposes a **quantitative veto** — λ is admissible only where the
planted flat blocks' depth-profile share error stays under a predeclared
tolerance (set when the harness lands; recorded in its config) and planted
subspace recovery holds; **if every nonzero λ fails the veto, λ = 0 becomes
primary and decoder nuclear regularization is demoted to a failed
ablation** — the veto cannot be passed trivially by λ=0 alone; (ii) headline
H4 numbers are reported across the λ sweep including λ = 0; (iii) H4's
primary readouts are code-anchored (below), not the raw decoder spectrum.

SASA's actual formulation (full-text read 2026-07-15) — penalty on the
end-to-end product ‖D_k E_k‖_*, gauge-invariant by construction, weight-decay
variational form — is the documented **ablation variant**: it needs no
retraction but leaves selection exactness unfixed and mixes encoder
conditioning into the per-site rank readout, which is why it is not primary.

### Rank measurement (H4 readout)

"Effective linear span dimension," three estimators, all reported with full
spectra (findings 5, 6; round-2 R6, R21):

1. **Decoder spectrum** = **frame capacity**, not used dimension — the
   constraint mandates aggregate rank b, so surplus directions are parked
   capacity. Reported (participation ratio (Σσ²)²/Σσ⁴, 95%-energy rank) but
   never the basis of a rank claim.
2. **Contribution second moment** (primary) — spectrum of
   D_g^sᵀ E[z_g z_gᵀ | g active] D_g^s: the span the block *actually uses*
   at that site, mean included (a large conditional mean is a real rank-1
   component that centering would delete).
3. **Contribution covariance** (centered) — spectrum of
   D_g^sᵀ Cov(z_g | g active) D_g^s: within-feature *positional* variation,
   the manifold-structure signal.

Inclusion threshold (R22): a block enters H4 histograms only above a
predeclared minimum active-token count (default 10k on the eval set);
below-threshold blocks are counted and reported separately.

Required controls: λ = 0 run, λ-sensitivity curve, seed stability (2 seeds
at primary config), and **rank-r truncation ablations** — held-out FVU as a
block is truncated to its top-r directions, which is the reconstruction-
anchored ground truth the spectra summarize and the final arbiter of any
rank claim. Curvature/ring claims are never made from rank alone; they
require the code-distribution diagnostics from Phase 0 (circular decoding,
Fourier structure).

### Shared-code validity (anti-leakage evals)

A summed multi-site encoder can achieve low loss by multi-view correlation
without a genuinely shared coordinate system, including late→early leakage
(finding 8). Two round-2 corrections to the eval battery (R19, R20): the
summed-encoder optimum may legitimately distribute a shared estimate across
sites (E^s x^s ≈ z/S), so *raw* single-site encoding under-reconstructs even
for a perfectly shared code — single-site evals are always reported **raw
and calibrated**, where the calibration is a per-(block, site) linear map
from site-only code to full code, fit on a calibration split, never on
eval. And per-coordinate correlation is not O(b)-invariant — cross-site
agreement uses rotation-invariant statistics (canonical correlations /
Procrustes-aligned R²), the same family Phase 0.5 uses.

Mandatory evals, run on every trained BSC:

- **Site-dropout encoding** — encode from single site s′ only, decode all
  sites; per-(s′, s) FVU matrix, raw + calibrated.
- **Leave-one-site-out** — encode from all sites but s, reconstruct s.
- **Cross-site code agreement** — CCA / Procrustes-R² between site-only
  codes, per block.

Blocks that fail these are reported as "correlated bundles," not shared
manifolds — the H5 diffing story is only told over blocks that pass.

### Sparsity hygiene

- **Aux loss (dead blocks)** — AuxK at block level: top-32 blocks dead for
  ≥ 500 batches reconstruct the residual; α = 1/32. The Gram constraint
  removes the decoder-shrinkage spiral; the aux loss handles encoder-side
  starvation. Block resampling kept as a documented option, off by default.
- **Latent Scaling, blockwise** (Minder) — per-site
  reconstruction-contribution regression as the shrinkage/decoupling
  diagnostic for Phase 3 diffing.
- **Init** — D_g^s Gaussian then one retraction (≈ equal site shares at
  init); E_g^s = D_g^s (transpose-tied at init only).
- **Optimizer** — AdamW, lr 3e-4 (1k-step linear warmup, cosine decay),
  β=(0.9, 0.999), fp32 master weights, 8-bit moments, bf16 params; batch
  4096 tokens. All to be recalibrated in the Phase-0.9 rehearsal.

### Identifiability caveat (worn openly)

Hard group sparsity rewards packing frequently co-active scalar features
into one block whether or not they form a manifold (finding 9). Block
discovery is *candidate generation*; the manifold claim for any block rests
on the coherence diagnostics (within-block code topology, ring tests,
truncation ablations, shared-code evals), synthetic-recovery calibration,
and BH multiple-comparison correction over the unknown-block search.
Pre-registered confirmatory targets: weekday ring, month ring.

## Configurations

| | rehearsal (0.9) | primary (Phase 1) | stretch |
|---|---|---|---|
| model | gemma-3-1b (d=1152, 26L) | gemma-3-4b (d=2560, 34L) | gemma-3-4b |
| sites | 6 in 40–90% band | 8 in 40–90% band (≈ layers 14–30, resolved at harvest, frozen in config) | 8 |
| G × b | 1024 × 4 | 4096 × 4 (16,384 latents) | 8192 × 4 |
| k (blocks/token avg) | 16 | 32 (128 active coeffs) | 32 |
| params (untied) | 57M | 671M | 1.34B |
| train VRAM est. | ~2 GB | ~9 GB | ~11 GB |
| store | ~8M tok ≈ 110 GB | 38M tok ≈ 1.56 TB | same store |

Baseline (every config): matched scalar crosscoder = **the b=1
Gram-constrained architecture** (round-2 R16): per latent,
Σ_s ‖d_j^s‖² = 1 with the same retraction — otherwise its coefficient scale
and quantization rate are gauge-dependent and the codec comparison
collapses. G·b scalar latents, BatchTopK with matched *training-average*
L0 (E[ℓ_t] = b·E[k_t]; each model uses its own realized counts at eval),
identical sites/whitening/data/order/precision/init scheme/tuning budget,
cold start both (finding 16). Same parameter count by construction. Warm
starts (from Phase-0 clusters) are exploratory runs only, never the
headline comparison.

**The primary H3 comparison runs both models at λ = 0** (at b=1 the
per-site nuclear term is a pure site-concentration penalty, not a rank
penalty, so nonzero λ is not architecture-fair); nonzero-λ BSC frontiers
are secondary.

λ_* initial grid {0, 3e-4, 1e-3, 3e-3} under the pinned reductions
(narrowed by the Phase −1 veto and the 0.9 rehearsal); k frontier
{16, 32, 64}. Staged run matrix (R26): rehearsal narrows λ to the
admissible set → one-seed frontier exploration at 4b → both confirmatory
seeds only at the preselected operating points. The 4b headline runs are
not a hyperparameter search.

## Data & training topology

**Disk-backed activation store on the 4090 box (~1.9 TB free NVMe;
decision 2026-07-15).** Harvest once on CUDA: gemma forward in bf16, 8 sites,
seq len 1024, FineWeb-Edu sample streamed with a pinned manifest (shard ids +
seed, shared verbatim by BSC and baseline runs); BOS and position-0/1
activations dropped; bf16 shards, shard-level + in-shard shuffle. Exact
bytes (round-2 R23): 40,960 B/token × 38M train = 1.557 TB; × 2M eval =
81.9 GB; **the 5M-token whitener slice is accumulated into fp32 statistics
and never stored** (only μ_s, Σ_s, W_s persist). Stored total ≈ 1.64 TB
against ~1.9 TB free — ~260 GB headroom for checkpoints (~1.3 GB each ×
2 models × snapshots), temporary shard writes, checksums, and filesystem
reserve; free space verified in bytes before the harvest commits. Training
then runs **without gemma in VRAM** — which is what re-opens the G=8192
stretch config — with 2-epoch default (per-epoch reshuffle), held-out FVU
gap monitored for store overfit.

Sample-power discipline (R24): 38M unique tokens is the demonstrated-power
envelope, not a free pass — Phase 1 reports learning curves at store
fractions (25/50/100%), block-stability across epochs, and reads rare-block
conclusions against the Phase −1 recovery-vs-frequency calibration. A
"mostly rank-1" verdict is stated as *conditional on demonstrated sample
power*, not as unqualified corroboration of flattening.

Escalation path if rare-feature starvation shows (dead fraction high or
held-out FVU still falling at epoch end): interleaved harvest+train
streaming (gemma co-resident; primary config only, ~17–18 GB) for a
≥ 100M-token budget — **and the scalar baseline retrains on the identical
extended manifest and token order** (R25); an escalated BSC is never
compared against the un-escalated baseline. Documented, not default.

Phase-0/rehearsal harvests on the M5 Max obey the MPS async-OOM discipline
(periodic `torch.mps.synchronize()`, zero-row guards — see AGENTS.md); the
big store is CUDA-side and additionally verified by content checksums and
per-shard non-zero/finiteness audits at write time.

Mandatory pre-commitment pilot (finding 31): before writing the full store,
a 1M-token pilot measures harvest tok/s, train tok/s, dead-block trajectory,
and VRAM high-water; the token budget and config are re-confirmed against
measurements, not estimates.

## Rate–distortion protocol (H3, made honest)

What is compared is **held-out activation rate–distortion**, not "total
MDL" — parameter bits are deliberately out of scope and the claim is scoped
accordingly (finding 15). Three data splits: train, **calibration** (held
out of training; all codec fitting happens here), eval (untouched until
scoring). Codec, pre-registered:

- **Canonical block orientation** (round-2 blocker R13): after training,
  each block's code space is rotated to diagonalize its calibration-set
  active-code second moment (descending), exploiting the residual O(b)
  gauge to fix a canonical frame; orientation is then frozen. Without this,
  an arbitrary rotation changes componentwise clipping and quantization
  error while the model is unchanged.
- **Quantizer** — per-coordinate uniform quantizer in the canonical frame:
  clipping range = calibration-set quantiles (0.1%/99.9%) per coordinate,
  2^q uniform levels within range, out-of-range values saturate to the
  clip; ranges and orientation are codec metadata (fit off-eval, applied
  identically in kind to both models). q swept {4, 6, 8}; distortion
  measured through the quantized codes.
- **Support bits/token** — −log₂ P(k_t) + log₂(G choose k_t), where the
  count model P is the empirical count distribution **fit on the
  calibration split and frozen** (R15 — the eval set cannot define its own
  code). The BatchTopK variable-k is thereby priced, not idealized
  (finding 13).
- **Realized counts** (R14) — each model pays its own: the BSC token pays
  for k_t active blocks (q·b·k_t amplitude bits), the scalar token for its
  own realized ℓ_t active latents (q·ℓ_t amplitude bits against
  log₂(Gb choose ℓ_t) support bits); only the *training-average* budgets
  are matched, E[ℓ_t] = b·E[k_t]. The support-term advantage is the entire
  structural bet, priced against the block's obligation to transmit b
  coordinates (finding 12).
- **Distortion** — whitened FVU, per site and pooled.
- **Comparison** — full frontiers over (k, q) for BSC vs matched scalar at
  λ = 0 both (see *Configurations*); the H3 verdict is frontier dominance
  over the shared operating region under **this declared codec** (R17 —
  the enumerative support code is deliberately usage-agnostic; an
  empirical-support-entropy sensitivity analysis accompanies it before any
  broader compression claim), not a single matched point (finding 14).
- **Uncertainty** — bootstrap over *sequences*, not tokens (R18 —
  neighboring-token activations are correlated and token bootstrap
  understates uncertainty); the two seeds displayed separately, never
  pooled.

## Phases (gates are decision heuristics with demonstrated power)

**Phase −1 — synthetic ground-truth harness. Offline; first deliverable.**
Planted dictionaries → synthetic activations → recovery. Must include:
shared blocks with cross-site frame rotation (the thing the BSC exists
for); flat-profile shared blocks under the **quantitative λ-veto** (see
*Rank regularizer* — admissible λ must keep planted flat-profile share
error under the predeclared tolerance, with the λ=0-primary fallback if
none passes); site-specific decoys; correlated scalar bundles that do *not*
form manifolds (the finding-9 test); planted ranks 1..b at controlled depth
profiles **and controlled feature frequencies** (the recovery-vs-frequency
calibration that Phase 1's rare-block claims are read against, R24); and
the **rotation-equivariance test** (R8): paired seeds from randomly
O(b)-rotated inits must recover matching spectra/subspaces, else decoders
move off coordinatewise Adam. Recovery metrics: subspace principal angles,
rank recovery, depth-profile fidelity, code correlation.
*Gate:* the pipeline recovers what was planted, does not hallucinate
structure in the nulls, and yields a nonempty admissible λ set (or the
documented λ=0 fallback). Lives in `tests/` + `scripts/`; runs on MPS.

**Phase 0 — post-hoc blockification. Zero training; days.**
First, **positive control**: replicate the Engels weekday/month rings on
GPT-2-small with a public SAELens release through our exact pipeline —
a gemma null without this is uninterpretable (finding 17/19/35). Then:
`google/gemma-scope-2-4b-pt` residual SAE at the saklas-convention analysis
depth (~65%); cluster decoder directions (cosine + spectral, Engels-style);
within-cluster PCA of code contributions over a token stream (saklas `sae`
runtime harvest). Ring criteria, quantitative: held-out circular decoding
of the known cyclic families, Fourier structure of code angle, mixture/
separability tests, label-permutation and random-cluster nulls; BH
correction over unknown-cluster search.
*Gate:* rings recoverable → live targets + warm-start candidates for
exploratory runs. Null **with the control passing** → "no rings recoverable
from this SAE at these depths at demonstrated power" — downgrades H1,
reroutes to a different SAE/depth or Qwen before any training. (What a null
does *not* establish: "gemma flattens token-level geometry" — the gate
language is scoped to what it measures.)

**Phase 0.5 — cross-layer coherence pre-test. Zero training; days.**
Verify Gemma Scope 2 layer coverage first (needs SAEs at 2–3 depths). The
gate metric is **not** per-latent cosine matching — independently-trained
per-layer SAEs can span the same rotating subspace with dissimilar
individual directions, which would fail exactly when the BSC premise holds
(finding 20). Instead: (i) principal angles between Phase-0 cluster
subspaces across depths, against a random-subspace null; (ii) paired-token
CCA/Procrustes between matched subspaces; (iii) out-of-sample coordinate
prediction — fit the layer-A→layer-B code map on half the tokens, report
residuals on the other half. Laptev cosine flow (2502.03032) demotes to a
matching bootstrap. *Gate:* spans match AND positions correspond → the
shared-code assumption has legs. Span-match without position-correspondence
is itself a finding (frames persist, codes transform) and steers toward
stage-blocks.

**Phase 0.9 — 1b dress rehearsal. Days; decision 2026-07-15.**
The entire ladder end-to-end on gemma-3-1b + `gemma-scope-2-1b-pt` at the
rehearsal config: harvest → store → whiten → train BSC + scalar baseline →
full eval suite → toy export. Plumbing gates only (harvest integrity,
training stability, eval determinism, hyperparameter sanity) — science
verdicts wait for 4b. A 4b failure after a green rehearsal is about the
science, not the code.

**Phase 1 — train the BSC on gemma-3-4b. ≈week part-time + 4090 runs.**
Primary config from the table; store-backed; cold-start BSC + b=1
Gram-constrained baseline; staged run matrix (rehearsal-narrowed λ →
one-seed frontier exploration → 2 confirmatory seeds at preselected
operating points). Headline outputs: rate–distortion frontiers at λ=0
(H3); per-block per-site effective-span histograms + spectra with the full
control battery and learning curves at store fractions (H4, stated at
demonstrated sample power); shared-code eval matrix (raw + calibrated);
block coherence diagnostics.
*Gate:* H3 in either direction is informative; proceed to Phase 2 if any
blocks resolve multi-dimensional, coherent, AND pass the shared-code evals.

**Phase 2 — the saklas bridge. Medium feature; lands in saklas, not here.**
Export a block as a multi-layer manifold artifact. The export must preserve
the per-site coordinate map, not just the span (finding 10): per site,
truncated SVD of whitened D_g^s (keep U_r, σ, and the b×r right-factor);
node positions embedded per site through the *full* map, never re-derived
from a bare basis. share = per-site whitened decoder norms **re-expressed in
saklas's consumer-side whitener** (the two whiteners — training-side
harvest-fit vs consumer-side neutral-fit — are not interchangeable; the
seam is explicit). Origin (neutral-mean projection), σ (per-site code-density
residual), and node labeling (clustered code-density modes + max-activating
contexts) are *additional estimators with their own validation burden*
(finding 22) — `experiment naturalness` validates the intervention path, not
the discovery. Import lands as a saklas manifold-folder source
(`manifold.json` + per-model safetensors, per the existing producer/consumer
contract).

**Phase 3 — cross-model BSC. New horizon.**
Sites = layers × models at **constant site budget** (4 layers × 2 models = 8
sites; the v1 claim that cross-model work "doubles the harvest, not the
decoder scaling" was wrong under layers×models — finding 23). Base vs
instruct gemma first, **paired forwards on identical raw token sequences**
(shared tokenizer removes vocabulary alignment; chat-template asymmetry is
the residual caveat, handled by running both models template-free on the
same stream and treating template tokens as a documented distribution gap —
finding 24). Manifold-level diffing + persona-fan provenance (H5), gated on
the same shared-code evals per model. Gemma vs Qwen cross-arch only after
token alignment is solved; document-level pooling changes the object of
study (it answers a pooled question, not the token-level ring question) and
is out of scope for the ring claim.

## Out of scope for v1

- Cross-architecture token alignment (Phase 3's second half gates on it).
- Sphere/exotic topologies (authored-only in saklas for good reason).
- Weakly-causal crosscoder variants (reach for them only if stage-splitting
  dominates).
- Steering-quality tuning of imported manifolds (Phase 2 proves the pipe).
- fp8 activation storage (quantization noise sits too close to the
  rate–distortion floor being measured).
- b-sweeps beyond {1 (baseline), 4}: b ∈ {2, 8} noted as follow-up; a
  rank-4 ceiling is a design choice, not a discovery (finding 33).

## Risks

Structurally designed out: gauge-vacuous rank penalty, selection-score
proxy error, decoder death spiral (all via the Gram constraint); L1
shrinkage artifacts (BatchTopK); rogue dims (whitened everything); fp16
overflow (banned); pipeline-failure nulls (Phase −1 harness + GPT-2
positive control + 0.9 rehearsal).

Instrumented, not eliminated: per-site penalty's site-concentration
pressure (synthetic gate + λ=0 control + contribution-covariance readout);
rank-histogram circularity (λ sweep, truncation ablations, seed stability);
acausal multi-view leakage (site-dropout eval matrix); block bundling
(coherence diagnostics + BH correction); store overfit at 38M tokens
(held-out FVU gap; interleaved escalation path).

Honestly open: the block prior may mismatch language (the Fel hedge) — H3/H4
are informative under the null and a flat dictionary corroborates the
flattening line. The novelty claim is scoped to LLM interpretability; the
older multi-view / coupled group-sparse dictionary-learning literature has
not been swept and should be before any external writeup.

## Decision log

- **2026-07-15** — v2 rewrite after adversarial review. Gram-constrained
  decoders chosen over SASA's product penalty (selection exactness + clean
  per-site rank readout; product form kept as ablation). Disk-backed store
  chosen (a9: ~1.9 TB free on the 4090 box) over interleaved streaming;
  interleaved kept as the >100M-token escalation. Phase 0.9 (1b rehearsal)
  added (a9 sign-off). Primary config reduced to G=4096 after VRAM
  arithmetic; G=8192 stretch re-enabled by the store. Rate–distortion
  framing replaces "MDL" for H3. Phase 0.5 gate metric replaced (subspace
  principal angles + position correspondence, not per-latent cosine flow).
- **2026-07-15 (v2.1)** — round-2 amendments after verification pass
  (algebra PASS; freeze conditioned on these): loss reductions pinned
  (λ, α now meaningful); canonical-orientation quantizer completing the
  codec; baseline made explicitly b=1 Gram-constrained with realized
  counts; primary H3 at λ=0 both models; fp32-master retraction order +
  eigenvalue floor + decoder weight decay 0; calibrated site-dropout and
  rotation-invariant code-agreement stats; uncentered contribution second
  moment added as primary H4 readout; frame-capacity naming; λ-veto made
  quantitative with λ=0 fallback; store arithmetic in exact bytes with
  whitener accumulate-and-discard; sequence-level bootstrap; staged run
  matrix; escalation retrains the baseline. Design frozen for
  implementation from this state.
