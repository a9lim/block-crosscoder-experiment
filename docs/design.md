# Block-sparse crosscoders: authoritative design

*Version 4.0, 2026-07-19. This document is self-contained and normative.
Completed evidence is summarized in
[`findings-phase0.md`](findings-phase0.md); the literature position is
[`literature.md`](literature.md).*

## 1. Objective and current state

A block-sparse crosscoder (BSC) learns sparse multidimensional features with
one code shared across sites and a distinct decoder frame at every site. It
occupies the {block} × {cross-site} cell of the dictionary-learning 2×2:

| | one site | shared across sites |
|---|---|---|
| scalar | SAE | scalar crosscoder |
| block | block-sparse featurizer | **block-sparse crosscoder** |

Phase 0 passed the architecture, instrument, factorial, rate–distortion, and
harvest-readiness gates. The promoted pilot checkpoint is
[`data/winner.json`](../data/winner.json): Gemma 3 4B, `G=4096`, `b=4`,
`k=32`, site-renormalized gauge, 24M optimizer tokens, pooled top-k FVU
0.3997. Phase 1 trains the production dictionary on a 53M-token, 2.171 TB
store. Its only external blocker is installation of the purchased 4 TB NVMe
in jobe. Record the mount point here and in the workspace root `AGENTS.md`
before harvesting.

The research hypotheses are:

- **H1:** language models contain irreducible multidimensional token-level
  features. Phase 0 found the month ring below scalar-SAE clustering scale
  and captured it with a native BSC;
- **H2:** feature coordinates persist across depth while frames rotate.
  Paired-token maps passed before training; trained shared-code validity is a
  Phase-1 endpoint;
- **H3:** blocks earn their four amplitudes on held-out activation
  rate–distortion. The pilot result is strongly positive; Phase 1 is the
  headline verdict;
- **H4:** contribution-anchored effective linear span localizes structure
  over depth. The readouts are validated; confirmatory histograms and
  truncations belong to Phase 1;
- **H5:** sites extended to layers × models permit manifold-level model
  diffing. This is deferred to the post-publication cross-model phase.

Gates are decision rules with demonstrated power, not logical
falsifications. Every null is reported with its positive controls, synthetic
recovery envelope, and sample-power limits.

## 2. Coordinates and activation store

### 2.1 Sites and whitening

Production sites are residual-post layers `(9, 12, 15, 18, 21, 24, 27,
30)` of `google/gemma-3-4b-pt`. All encoding, selection, reconstruction,
regularization, and rank readout occurs in per-site transformed coordinates.

For site `s`, fit mean `μ_s` and covariance `Σ_s` on a dedicated
calibration prefix, disjoint from train and eval. The transform is

`W_s = (Σ_s + λ_s I)^(-1/2)`,

where `λ_s` is the mean covariance eigenvalue, matching the saklas
`LayerWhitener` convention. This is **shrinkage whitening**, not full
whitening: its held-out target spectrum is `σ/(σ+λ)`, not identity.
Covariance sufficient statistics accumulate in fp64 with TF32 disabled;
the eight eigendecompositions run in fp64.

After shrinkage whitening, multiply each site by one scalar so its total RMS
power matches the other sites. The scalar is estimated on the calibration
slice and folded into `W_s`; it is not a train-time option. The Phase-0
winner records its exact pilot scalars in `data/winner.json`. Production
uses the fresh calibration fit. Phase-0 validation found pilot versus fresh
5M scalars within 0.7% and independent-slice CV at most 1.13%.

Freeze and hash the complete transform: model revision, layers, source
manifest, `μ_s`, shrinkage ridge, RMS scalar, and `W_s`. Every shard header
carries the hash; mismatches are fatal. Any refit requires reharvest or an
explicit, validated migration.

**fp16 is prohibited throughout harvest and storage.** Raw Gemma late-layer
channels routinely exceed 65,504. Model forwards use bf16, whitening and
statistics use fp32/fp64 as specified, and the store holds transformed bf16.
A small raw shard is retained for round-trip validation.

### 2.2 Store layout and I/O

Harvest FineWeb-Edu with a pinned dataset revision and explicit source
manifest. Use sequence length 1024; drop BOS and positions 0/1. Harvest the
whitener slice first and do not store it. Then write:

| split | tokens | bytes |
|---|---:|---:|
| train | 38M | 1.557 TB |
| eval | 2M | 81.9 GB |
| calibration | 13M | 532.5 GB |
| **total** | **53M** | **2.171 TB** |

The exact layout is `[token, site, d_model]` bf16: 40,960 bytes/token.
Shards are 2–8 GiB, sequence-contiguous, atomically renamed, self-describing,
and checksummed. Enforce a byte-exact pre-write capacity check and at least
15% free-space floor. Audit non-zero and finite values at write time; verify
all checksums after harvest.

Token-random mmap is forbidden. Sequentially read contiguous chunks into a
32k–128k-token shuffle buffer, shuffle shards per epoch and samples within the
buffer, and prefetch four pinned batches without reordering or swallowing
exceptions. Record the permutation seed and share it across matched models.
Eval and calibration stream from disk; neither is host-RAM-resident.

Interrupted harvest recovery never appends through a fresh writer. Recover
complete atomic shards, quarantine `.tmp`, verify contiguity, then relaunch a
new split past the recovered corpus offset and merge manifests. A new writer
restarts shard numbering at zero and would collide in place.

## 3. Model and optimization

### 3.1 BSC parameterization

For `G` blocks of width `b` and sites `s=1..S`, block `g` has encoder
`E_g^s ∈ R^(b×d)`, decoder frame `D_g^s ∈ R^(b×d)`, and each site has
decoder bias `c^s`. For transformed activation `x̃^s`:

- encode: `z_g = Σ_s E_g^s x̃^s`;
- select by `p_g = ||z_g||_2`;
- decode: `x̂̃^s = c^s + Σ_(g active) D_g^{sT} z_g`.

Every block obeys the concatenated Gram constraint

`Σ_s D_g^s D_g^{sT} = I_b`.

After each optimizer step on fp32 master weights, compute the block Gram,
floor eigenvalues at `1e-6`, left-multiply every site frame by its inverse
square root, regenerate the bf16 forward copy, and log post-cast Gram
residuals and floor hits. The order is optimizer → fp32 retraction → bf16
copy → residual. Decoder weight decay is zero because retraction would
undo uniform shrinkage; encoder decay is also zero at the pinned point.

The constraint removes the coefficient/decoder scale gauge, reduces the
within-block `GL(b)` gauge to `O(b)`, fixes every concatenated frame at rank
`b`, prevents decoder collapse, and makes `||z_g||²` exactly the block's
isolated total decoder-output energy. It does not make that energy a marginal
loss reduction; blocks can overlap and cancel.

### 3.2 Selection and inference threshold

Training uses BatchTopK over block norms, retaining `k×B` block activations
over a batch of `B` tokens. Counts per token may vary. Inference uses one
fixed global threshold `θ`, fit on the entire calibration split to the target
average count with a bounded-memory streaming log-histogram quantile.
Serialize `θ` with the codec. The training EMA of selected-score minima is a
diagnostic only; it is history-dependent and never the inference threshold.

### 3.3 Loss and rank pressure

The objective is

`L = L_rec + λ_rank R_rank + α L_aux`,

with squared reconstruction error averaged over tokens, sites, and
dimensions. The normalized per-site nuclear penalty is

`R_rank = mean_g [(Σ_s ||D_g^s||_* - b) / b]`.

Under the Gram constraint it ranges from 0 for site-exclusive directions to
`√S-1` for a flat shared frame. It can reshape the fixed spectrum budget but
cannot scale it away. Its residual bias toward site exclusivity is real. The
synthetic flat-profile veto admitted `λ_rank ∈ {3e-4, 1e-3}` and rejected
`3e-3`; `1e-3` is the primary structured arm. All architecture-fair H3
frontiers compare BSC and scalar at `λ_rank=0`.

Decoder singular values are **frame capacity**, not used dimension. Rank
claims use code-anchored contribution moments and reconstruction ablations.

### 3.4 Dead blocks, initialization, and guard

Initialize decoder frames Gaussian, retract once, set encoders from the
decoder transpose, and norm-calibrate encoder scale so initial scores are
comparable. Use the SASA C.1-style frequency-dead approximation: re-encode a
detached residual through dead blocks only, selecting the highest residual-
energy auxiliary blocks. Cap the auxiliary/main gradient norm ratio at 1.0.
Block resampling remains off by default.

The mandatory loss-spike guard compares the current step against the median
of 50 accepted steps. Trigger only when gradient norm exceeds 20× **and**
reconstruction loss exceeds 5× the reference, or on non-finite state. Skip
the batch, advance the scheduler, snapshot batch identity and component
norms, and continue. More than five consecutive skips refuses the run. Total
skip rate above 0.1% fails the run. The guard is an instrument, not permission
to operate at an unstable learning rate.

## 4. Matched models and the production stack

The scalar baseline is the signed `b=1` special case with concatenated
decoder norm one, `G×b` latents, identical sites, data, whitening, parameter
count, optimizer, and cold start. Match training-average coefficient count:
`E[ℓ_t] = b E[k_t]`; eval prices each model's realized counts. This is an
internally controlled baseline, not a published ReLU-crosscoder convention.

The single-site controls delete code tying without changing per-site tensor
shapes: eight independent `S=1` block models and eight scalar SAEs. At one
site, RMS renormalization is a global loss scale, so controls train at unit
scale and compare by per-site FVU and the R–D plane.

### Pinned Phase-1 stack

| component | value |
|---|---|
| model/sites | Gemma 3 4B; residual-post 9, 12, 15, 18, 21, 24, 27, 30 |
| primary BSC | `G=4096`, `b=4`, `k=32` (~671M params, ~9 GB train VRAM) |
| optimizer | AdamW, 8-bit moments, fp32 masters, bf16 forward, `lr=3e-4`, 1k linear warmup, cosine decay, `β=(0.9,0.999)`, batch 4096 |
| regularizer | `λ_rank=1e-3`; H3/frontiers at zero |
| coordinate gauge | shrinkage whitening plus site RMS renormalization |
| dead-block path | SASA C.1 approximation, auxiliary ratio cap 1.0 |
| guard | 20× gradient and 5× reconstruction trigger; >5 consecutive refuses; skip rate ≤0.1% |
| inference | full-calibration streaming `θ`; threshold mode primary; realized counts |
| data path | sequential buffered shuffle, prefetch 4 |
| epochs | 2 over 38M train tokens; repeats are acceptable if budget binds |
| seeds | 2 confirmatory seeds at headline operating points |
| codec | `q ∈ {4,6}` primary; bf16 shadow in top-k and threshold modes |

Training on 4B is bit-deterministic across repeated runs. The healthy dead
band at `G=4096` is roughly 0.1–0.15%. `G=8192` is not a silent stretch:
Phase-0 dead dynamics reached 3.6%, so it requires an explicit decision after
a short production-store diagnostic.

## 5. Evaluation protocol

### 5.1 Split discipline and activation codec

Train fits weights only. Calibration fits `θ`, canonical code orientation,
quantizer ranges, count model, and site-only code maps. Eval is untouched
until scoring. The 13M-token calibration split supplies about 100k active
examples to an average block at `k/G ≈ 0.78%`; blocks below a predeclared
active-count floor are excluded from codec comparisons and counted.

For each block, rotate the active calibration codes to diagonalize their
second moment in descending order. Freeze that canonical orientation. In the
canonical frame, fit 0.1%/99.9% clipping quantiles per coordinate and a
`2^q`-level uniform quantizer; saturation outside the interval is part of the
codec.

For a BSC token with `k_t` active blocks, charge

`-log2 P(k_t) + log2 C(G,k_t) + q b k_t` bits.

For a scalar token with `ℓ_t` active latents, replace `G,k_t,bk_t` by
`Gb,ℓ_t,ℓ_t`. Fit and freeze `P(count)` on calibration. Report a
usage-aware support-entropy sensitivity analysis beside the declared
enumerative code, but do not change the headline codec after seeing eval.

Distortion is transformed-coordinate FVU per site and pooled with exact eval
squared-energy weights. Compare full `(k,q)` frontiers at `λ_rank=0`, over
their shared rate region. Bootstrap sequences, not tokens, and display seeds
separately.

### 5.2 Shared-code validity

A summed encoder can exploit multi-view correlation without learning a
shared coordinate. Every trained BSC therefore reports:

- single-site encoding to all-site reconstruction, as an `(input site,
  output site)` FVU matrix;
- the same matrix after a calibration-fit per-(block, site) linear map from
  site-only code to full code;
- leave-one-site-out reconstruction;
- canonical correlations and Procrustes `R²` between site-only codes.

Raw site-only encoding is not expected to match a code whose estimate was
distributed over sites; the calibrated result separates that benign scaling
from a coordinate failure. Blocks that fail are **correlated bundles**, not
shared manifolds. No Phase-2/3 story is told over them.

### 5.3 Effective linear span

Report full spectra for three estimators:

1. decoder spectrum: capacity only, with participation ratio and 95%-energy
   rank;
2. contribution second moment
   `D_g^{sT} E[z_g z_g^T | active] D_g^s`, including the conditional mean;
3. centered contribution covariance, measuring within-feature position.

Include only blocks with at least 10k eval activations in headline
histograms; count the rest. Report the zero-regularizer control,
regularizer sensitivity, two-seed stability both at block level and global
recovered-subspace level, and rank-`r` decoder/code truncation ablations.
Truncation FVU is the final arbiter. Curvature requires separate topology
tests; rank alone is never ring evidence.

## 6. Phase-1 execution

### 6.1 Store commit gate

1. Install and mount the 4 TB NVMe; record its path here and in workspace
   guidance; verify live capacity.
2. Harvest the 5M-token statistics prefix, validate the transform, freeze its
   hash, and forecast exact bytes.
3. Harvest all three splits with the pinned manifest; retain a raw shard.
4. Run non-zero/finiteness, header/hash, content-checksum, split-contiguity,
   bf16-error, and raw round-trip checks.
5. Resume one checkpoint smoke run from the new store. No concurrent
   checkpoint loads while training: restoring fp32 masters and Adam beside a
   training residency OOMs the 24 GB GPU.

### 6.2 Run matrix

The 4B headline runs are not a search:

- site-renormalized BSC at the pinned stack, two seeds;
- matched signed scalar crosscoder, two seeds, with both headline H3 models
  at `λ_rank=0`;
- zero-regularizer frontiers at `k ∈ {16,32,64}`, one seed per off-point;
- one primary-gauge BSC seed for gauge comparison;
- site-renormalized BSC at `λ_rank ∈ {0,3e-4}` as lower regularizer arms.

Optional frontier closure is `k=128` for the BSC, scalar `k≈6–8`, and one
site-renormalized scalar joint cell. These do not delay the headline matrix.

### 6.3 Reports and gates

Every run reports pooled and per-site top-k/threshold FVU, realized counts,
bf16 shadow, learning curves at 25/50/100% of the store, R–D points, dead and
skip trajectories, Gram residual, `θ` drift, data-wait fraction, packing
census, and the applicable shared-code/effective-span battery.

H3 is the production frontier comparison over the shared declared-codec
range. Either direction is informative. Proceed to Phase 2 only if at least
one block is multidimensional under contribution/truncation readouts,
coherent under topology/null controls, and passes shared-code validity.
State any mostly-rank-one verdict conditional on the observed
recovery-vs-frequency and store-fraction power.

If dead fraction remains high or held-out FVU is still falling at epoch end,
escalate to interleaved harvest/train streaming for at least 100M tokens. The
scalar baseline must retrain on the identical extended manifest and order;
never compare an escalated BSC to an un-escalated baseline.
