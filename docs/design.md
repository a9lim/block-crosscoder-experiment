# Block-sparse crosscoders: Phase-0.5 design

*Version 5.0, 2026-07-20. This document is normative for the evidentiary
reset. The flaw ledger is [audit_2026-07-20.md](audit_2026-07-20.md); exact
paper lineage and matrix rationale are in
[paper_comparison.md](paper_comparison.md).*

## 1. Status and objective

Phase 0.5 supersedes the former Phase-0 winner and Phase-1 production pin.
There is no current winner. All earlier compact evidence, generated figures,
checkpoint pointers, and large `jobe` artifacts were removed after the audit.
No old quantitative claim may be restored without a corrected, hash-bound
rerun.

A BSC learns one sparse vector code jointly from `S` activation sites. For
block `g`, site `s` has encoder `E_g^s`, decoder `D_g^s`, optional shared
encoder bias `a_g`, and site decoder bias `c^s`:

`u_g = sum_s E_g^s x^s + a_g`,

`xhat^s = c^s + sum_(g active) D_g^{sT} z_g`.

The native BSC uses signed `z_g=u_g`, selects by `||z_g||_2`, and constrains
the concatenated decoder frame by

`sum_s D_g^s D_g^{sT} = I_b`.

Then `||z_g||²` is exactly the isolated decoded energy summed over sites. It
is not the marginal loss reduction when block contributions overlap. The
model is a new BSF/Crosscoder hybrid until the paper bridges pass.

## 2. Immutable activation identity and normalization factors

The Phase-0.5 screen uses `google/gemma-3-4b-pt` residual-post sites
`(9,12,15,18,21,24,27,30)`, FineWeb-Edu `sample-10BT`, context 1024, and
drops BOS plus position 1. Harvest resolves model and corpus aliases to commit
SHAs and records tokenizer artifact digests, tokenizer special IDs, dependency
versions, repository revision, and a source-tree hash. All transform-defining
metadata and tensors enter the normalization hash carried by every shard.

Normalization is an experimental factor, never implicit preprocessing:

| matrix name | transform |
|---|---|
| `none` | raw activations |
| `scalar` | dataset-site mean centering and one RMS scalar per site |
| `layer` | token/site LayerNorm with recorded epsilon |
| `whiten` | shrinkage covariance whitening |
| `whiten_renorm` | shrinkage whitening plus calibration-fit site RMS scaling |

Each store is harvested from the identical pinned stream prefix. Stats,
calibration, eval, and training ranges are disjoint. Model forwards and store
payloads are bf16; fp16 is forbidden; statistics are fp64 and transforms
fp32. The Phase-0.5 screen sizes are 250k stats, 500k calibration, 250k eval,
and 1M train tokens per normalization. A later confirmatory run must select
its size only after the screen.

Evaluation always reports distortion in the training coordinates and raw
coordinates. Fixed transforms are inverted. Token LayerNorm reconstructions
use the aligned raw store's token/site mean and variance. Store alignment,
site order, token count, source identity, and revision must match exactly.

Sequential buffered reads are mandatory. `prefetch=4` means queue depth four,
not total residency: the producer's current item and source shard are extra.
The wrapper is cancellation-aware and must be closed on early exit.

## 3. Executable method family

One implementation exposes the differences needed for controlled bridges:

- selector: per-token TopK, BatchTopK, fixed first-batch-calibrated threshold,
  or dense support;
- encoder: untied linear/affine or Grassmannian decoder-tied with one positive
  learned scale;
- code: signed, scalar ReLU, or learned group soft-threshold;
- selection score: block/code norm or Minder's ReLU activation times summed
  site decoder norms;
- decoder: concatenated Stiefel, per-block Frobenius ball, or free map;
- regularizer: none, site-profile, SASA end-to-end map nuclear norm,
  Anthropic decoder-norm-weighted activation L1, or group `L_(2,1)`.

The end-to-end map penalty is computed exactly as
`||Dbar_g^T Ebar_g||_*`, including for free decoders. The configuration field
is `lambda_regularizer`; the withdrawn `lambda_rank` spelling is recognized
only by historical analysis loaders. The old decoder-only term is
`site_profile`, never rank.

The paper bridges are:

- Fel Grassmannian: `S=1`, tied encoder, token TopK, Stiefel decoder, no Aux;
- Fel Vanilla: `S=1`, affine encoder, token TopK, Frobenius-ball decoder,
  no Aux;
- Fel Group Lasso: `S=1`, affine encoder, learned group threshold, free
  decoder, group `L_(2,1)`;
- original Crosscoder: affine scalar ReLU, dense support, free site decoders,
  decoder-norm-weighted L1;
- Minder Crosscoder: affine scalar ReLU, decoder-weighted BatchTopK, free
  decoder, calibrated inference threshold;
- SASA: `S=1`, signed token Top-s, free encoder/decoder, exact end-to-end map
  nuclear penalty, token-denominated dead residual Aux.

The native BSC and signed scalar special case remain internal controls. An
`S=1 BSC` is labeled as such and is not called a Fel model.

## 4. Training correctness contract

Masters and optimizer state are fp32. CUDA may use 8-bit Adam moments and a
bf16 forward copy. The accepted-step order is optimizer step, configured
decoder projection, finiteness validation, bf16 refresh, then diagnostics.
The free decoder performs no projection. Spectral Grams are formed in fp32
before contraction.

The loss-spike guard is transactional. A rejected batch cannot update
parameters, optimizer state, decoder projection state, activation/deadness
history, or accepted-step reference medians. Non-finite post-step model,
optimizer, projection, or forward-copy state aborts and requires reloading the
last atomic checkpoint. More than five consecutive rejected steps aborts;
skip rate above 0.1% refuses evaluation/promotion.

Deadness windows are denominated in accepted tokens. The matrix includes the
SASA 1k-token window, an intermediate 32,768-token window, and the former
409,600-token local horizon. Auxiliary capacity is coefficient-matched across
block and scalar controls. The Fel runner-up arm sets `s_aux=k` and
`alpha=1/k`; SASA/local arms explicitly sweep binding gradient-ratio caps.
No bridge inherits an unreported local auxiliary or guard modification.

Every run directory begins with an immutable manifest binding source-code
hash, git revision, store and transform hashes, model configuration, training
configuration, split identity, sites, shuffle seed, and postprocessing plan.
Fresh runs refuse nonempty directories. Resume and codec evaluation fail
closed on any binding mismatch; legacy unbound checkpoints are refused.
Checkpoints save model, optimizer, scheduler, dead tracker, guard state, RNG,
and run binding atomically after a free-space check.

## 5. Phase-0.5 matrix

`bsc phase05-matrix campaign` creates a stable, resumable manifest and runs:

1. five aligned normalization stores;
2. an 80-cell screen: every one of 16 recipes in every normalization at
   `lr=1e-4`, cosine schedule, four epochs, and its primary regularizer/Aux;
3. only if every screen cell completes, the complete declared factorial.

The factorial crosses every recipe-valid combination of:

- normalization: the five modes above;
- peak LR: `1e-4, 2e-4, 3e-4`;
- schedule: cosine or SASA-style final-fifth linear decay;
- epochs: `2,4,8`;
- seeds: `0,1` for every full-factorial cell (the screen uses seed 0);
- site/map coefficients: `0,3e-4,1e-3,3e-3`;
- Crosscoder L1: `1e-6,3e-6,1e-5,3e-5,1e-4`;
- Group-Lasso: `1e-4,3e-4,1e-3,3e-3,1e-2`;
- recipe-supported Aux, dead-window, and ratio-cap levels.

The generated manifest contains 68,220 factorial cells. This is the finite
declared training space, not a claim to enumerate every imaginable optimizer
or architecture. Successful non-promoted cells retain their manifest, log,
and report but delete `latest.pt` to keep disk use bounded. Screen failures
stop the campaign before the factorial.

The separate `bsc reproduce-papers` bridge first runs the clean exact-k Fel
synthetic protocol for Vanilla, Grassmannian, Group-Lasso, and the local
hybrid. Real-data bridge interpretation is blocked until this implementation
validation passes.

## 6. Evidence and promotion

Calibration fits inference thresholds, codec count/support/value models,
canonical orientations, and site-to-full affine maps. Eval is untouched until
all fits are frozen. Partial calibration is labeled diagnostic and is
promotion-ineligible.

The complete serialized codec is the only rate-evaluation input. Support
models are refit after block exclusion; zero-rate and saved-codec round trips
are regression tested. Report transformed and raw FVU, per-site FVU, full
support-count distributions, coded rate, dead/skip trajectories, FLOPs/time,
and deterministic repeat checks.

Every trained BSC also runs the calibration-fit/eval-score battery:

- operational and oracle-support site-only to all-output FVU matrices;
- calibration-fit site-to-full affine maps and held-out `R²`;
- leave-one-site-out reconstruction and support IoU;
- rank-aware CCA and held-out Procrustes `R²`;
- decoder-capacity, uncentered contribution, and centered-contribution
  spectra with explicit sample eligibility;
- origin-preserving and centered joint code/decoder truncation FVU.

Decoder spectra are capacity only. Used/effective dimension requires
activation-weighted contribution or map spectra and truncation behavior.
Generated figures must say which estimator they show and exclude or count
blocks below 10k eval activations.

No winner is promoted from FVU alone. A promotion requires all run gates,
complete calibration, saved-codec evaluation, raw-coordinate distortion,
paper-bridge context, shared-code validity, used-span evidence, and matched
budget comparisons. If a headline family is still improving at its planned
endpoint, extend every headline control to the same optimizer-token budget
before election.

## 7. Storage and execution

Training and harvest run on `jobe`'s RTX 4090 using shared plain Python. The
current `/data` disk has enough space only for the Phase-0.5 screen stores and
sequential one-checkpoint-at-a-time matrix. The purchased 4 TB NVMe is still
required before any later multi-terabyte production harvest. Record its mount
point here and in workspace guidance when installed.

Never load checkpoints concurrently with training on the 24 GB GPU. Store
writers enforce atomic shards, checksums, finiteness/zero-row audits, and a
15% free-space floor. Campaign status is recorded in `campaign_state.json`,
harvest state in `harvest_state.json`, and cell state in `state.json`; the
scheduled supervisor must inspect these and logs rather than infer progress
from process presence alone.
