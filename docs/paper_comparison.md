# BSC training compared with BSFs, crosscoders, and SASA

## Bottom line

The BSC formulation aligns with both parent ideas at the level that matters
most:

- from BSFs it takes a signed vector block as the sparse unit, selects by a
  rotation-invariant block norm, and decodes the whole vector within an active
  subspace;
- from crosscoders it takes one code inferred jointly from several sites and a
  separate decoder at every site.

It does **not** follow the complete training procedure of either parent paper.
The production model is a novel hybrid: free, biasless cross-site encoders;
concatenated orthonormal decoder frames; block BatchTopK; a site-profile
decoder penalty; and a SASA-inspired dead-block auxiliary loss. No Fel BSF has
that combination, Anthropic's original crosscoder is ReLU/L1 rather than
signed TopK, Minder's BatchTopK model is scalar and affine/ReLU, and SASA is
single-site with per-token Top-s and a nuclear norm on the end-to-end map.

That is not a reason to discard the design. It is a reason to treat each
difference as an empirical factor, add faithful bridge implementations, and
narrow every claim until the bridges have been run. Implementation defects
and claim consequences are tracked separately in
[audit_2026-07-20.md](audit_2026-07-20.md).

## Phase-0.5 implementation disposition

The comparison above describes the audited baseline. Phase 0.5 has now made
the differences executable rather than implicit. The shared model stack can
instantiate Fel Vanilla, Grassmannian, and Group-Lasso BSFs; the original
affine ReLU/L1 Crosscoder; Minder's decoder-weighted ReLU BatchTopK
Crosscoder; free-map SASA; the native BSC; and its signed scalar and `S=1 BSC`
controls. The original site-profile term has been renamed, and SASA's
end-to-end map nuclear norm is computed exactly for constrained or free
decoders.

`bsc reproduce-papers` owns the clean exact-k Fel synthetic bridge. `bsc
phase05-matrix` owns the activation-store program. Its stable manifest has 16
recipes, five normalization gauges, three learning rates, two schedules,
three epoch budgets, two full-factorial seeds, method-specific lambda grids, paper/local auxiliary
variants, three SASA dead windows, and two binding ratio caps. It produces 80
screening cells and 68,220 full-factorial cells. These are all recipe-valid
combinations of the finite declared factors; they are not a mathematically
unbounded claim about every optimizer or architecture one could invent.

The screen deliberately fixes `lr=1e-4`, four epochs, and each recipe's
primary penalty/Aux because `1e-4` is the strongest directly paper-supported
starting point (Fel/Minder) and was absent from the former preferred stack.
The full matrix then crosses `1e-4,2e-4,3e-4`, cosine versus final-fifth
linear decay, and 2/4/8 epochs. Site-profile/map coefficients are
`0,3e-4,1e-3,3e-3`; Crosscoder L1 coefficients are
`1e-6,3e-6,1e-5,3e-5,1e-4`; Group-Lasso coefficients are
`1e-4,3e-4,1e-3,3e-3,1e-2`. No old `3e-4, lambda=1e-3` winner is privileged.

Every normalization uses the same pinned token stream. Reports include both
training-coordinate and raw-coordinate FVU, saved-codec rate, support-count
distribution, and the trained shared-code/effective-span endpoints. The
campaign stops after the screen if any cell fails and retains no scientific
claim from implementation tests alone.

## Sources and naming

This comparison uses the following primary sources rather than secondary
summaries:

- Fel et al. (2026), *Structuring Sparsity: Block-Sparse Featurizers Capture
  Visual Concept Manifolds*: [paper](https://arxiv.org/abs/2606.25234),
  [official repository](https://github.com/goodfire-ai/block-sparse-featurizer),
  [local full text](../references/fel2026-block-sparse-featurizers.md).
- Lindsey et al. (2024), *Sparse Crosscoders for Cross-Layer Features and
  Model Diffing*: [primary report](https://transformer-circuits.pub/2024/crosscoders/index.html),
  [local text](../references/anthropic2024-crosscoders.md).
- Minder et al. (2025), *Overcoming Sparsity Artifacts in Crosscoders to
  Interpret Chat-Tuning*: [paper](https://arxiv.org/abs/2504.02922),
  [training code](https://github.com/jkminder/dictionary_learning),
  [local full text](../references/minder2025-overcoming-sparsity-artifacts-crosscoders.md).
- Dalili and Mahdavi (2026), *Subspace-Aware Sparse Autoencoders for Effective
  Mechanistic Interpretability*: [paper](https://arxiv.org/abs/2606.06333),
  [official repository](https://github.com/arshandalili/sasa),
  [local full text](../references/dalili2026-sasa.md).

“Fel BSF” below refers to the three paper variants—Vanilla, Grassmannian, and
Group Lasso. “Original crosscoder” refers to the Anthropic ReLU/L1 model.
“Minder crosscoder” refers specifically to the later BatchTopK variant unless
the L1 arm is named. “Current BSC” refers to the Phase-1 configuration in
[design.md](design.md) and the implementation in
[`model.py`](../block_crosscoder_experiment/model.py),
[`trainer.py`](../block_crosscoder_experiment/trainer.py), and
[`gram.py`](../block_crosscoder_experiment/gram.py) at audit baseline
`e7f9017`.

## The current BSC in one equation

For sites `s=1,...,S`, block `g` has an encoder
`E_g^s in R^(b x d)` and decoder frame `D_g^s in R^(b x d)`. Given transformed
site activations `x_s`, the shared pre-code is

`z_g = sum_s E_g^s x_s`.

The training selector retains `kB` block events with largest `||z_g||_2`
over a batch of `B` tokens. Each site reconstructs

`xhat_s = c_s + sum_(g active) D_g^{sT} z_g`,

subject to

`sum_s D_g^s D_g^{sT} = I_b` for every block.

The audited objective is

`L = L_rec + lambda_site_profile R_site_profile + alpha_aux L_aux`,

where the audited baseline called the middle coefficient `lambda_rank`, but

`R_site_profile = mean_g[(sum_s ||D_g^s||_* - b)/b]`

is not an effective-rank penalty. Inference replaces BatchTopK with a fixed
global threshold fit on the calibration split to the target average block
count.

The decoder constraint has an especially useful consequence. If the isolated
contribution of block `g` is concatenated over sites, its squared norm is

```text
sum_s ||D_g^{sT} z_g||_2^2
  = z_g^T (sum_s D_g^s D_g^{sT}) z_g
  = ||z_g||_2^2.
```

Thus the selection score is exactly isolated cross-site output energy. It is
not necessarily marginal error reduction because different blocks can overlap
and cancel.

## Architecture comparison

| Method | Code and selector | Encoder | Decoder constraint | Regularization | Aux/revival | Sites |
|---|---|---|---|---|---|---|
| Fel Vanilla BSF | Signed vector blocks; per-sample block TopK | Free affine `xW+b` | Each block in Frobenius unit ball | Reconstruction only; hard block count | Core method does not require dead-block Aux | One |
| Fel Grassmannian BSF | Signed vector blocks; per-sample block TopK | Tied `gamma xD^T`, one positive learned scale | Each block individually Stiefel; QR about every 20 steps | Reconstruction only | Core method does not require dead-block Aux | One |
| Fel Group Lasso BSF | Signed vector blocks; learned block soft threshold | Free affine `xW+b` | Free decoder with scale control | Reconstruction plus `lambda ||z||_(2,1)`, activated against target sparsity in implementation | No required dead-block Aux in core formulation | One |
| Anthropic original crosscoder | Shared nonnegative scalar code; ReLU; L1 sparsity | Free affine sum over sites | Free per-site scalar decoder vectors and biases | Activation times L1-of-per-site decoder norms | Not central to published formulation | Many |
| Minder L1 crosscoder | Same affine/ReLU scalar family | Separate base/chat encoders plus bias | Free base/chat decoders | Decoder-norm-weighted L1 | L1 arm as trained | Two models |
| Minder BatchTopK crosscoder | ReLU scores scaled by sum of decoder norms; top `Bk` globally; threshold at inference | Free affine | Free base/chat decoders | Reconstruction | Dead-latent residual Aux, usually `k_aux=512`, `alpha=1/32` | Two models |
| SASA | Signed vector blocks; per-token Top-s by encoder-block norm | Free linear block encoder | Free block decoder | Nuclear norm of the end-to-end block map `||D_k E_k||_*` | Residual re-encoding through frequency-dead blocks | One |
| Current BSC | Shared signed vector blocks; block BatchTopK; global threshold at inference | Free linear cross-site encoder, no bias | Concatenated site frames are Stiefel; per-step polar retraction | Reconstruction plus decoder site-profile penalty | SASA-style dead-block residual Aux with gradient-ratio cap | Eight layers |

### What is inherited cleanly

The following correspondences are direct rather than metaphorical.

1. **Block as the sparse event.** Fel's MAP argument charges once for switching
   on a factor and does not charge its internal coordinates separately. BSC
   does the same during selection and in its block codec.
2. **Signed internal coordinates.** Fel rejects coordinatewise ReLU because it
   restricts a subspace to a positive cone. BSC likewise preserves positive
   and negative coordinates within each selected block.
3. **Basis-invariant support score.** Both use the Euclidean norm of a block,
   so an orthogonal change of basis inside the block leaves support unchanged.
4. **Shared code, site-specific decoding.** Anthropic's crosscoder computes a
   latent from all sites and decodes it separately at each site. BSC replaces
   each shared scalar with one shared vector and each decoder direction with a
   frame.
5. **BatchTopK training/global-threshold inference.** BSC is structurally close
   to Minder here. Because the concatenated decoder frame is normalized, the
   BSC block norm already equals the decoder-scaled energy used to make
   Minder's scalar scores comparable.

### What is genuinely new or unmatched

1. **Free encoder plus Stiefel decoder.** Fel either makes both maps free with a
   decoder ball constraint or ties them under a Stiefel constraint. BSC frees
   the encoder while putting the *concatenated* multi-site decoder on Stiefel.
2. **Cross-site vector coordinates.** A scalar crosscoder can share feature
   presence. BSC also forces the within-feature position `z_g` to be common
   across sites, up to the learned site frames. That is a stronger hypothesis
   and should be tested with site-only and leave-one-site-out endpoints.
3. **Decoder site-profile pressure.** No parent method uses
   `sum_s ||D_g^s||_*` under a concatenated Gram constraint. It redistributes a
   fixed decoder spectrum across sites and can encourage coefficient-level
   site partitioning. It is neither Fel's constraint nor SASA's map nuclear
   norm.
4. **Whitening plus site RMS renormalization.** This is materially different
   from Fel's isotropic input scaling, Anthropic's independent layer
   normalization, and SASA's layer-normalized activations.
5. **The exact Aux/guard combination.** The long frequency window, auxiliary
   capacity, gradient-ratio cap, and spike guard are local engineering choices.

## Fel BSF comparison in detail

### Vanilla BSF

Fel's Vanilla model computes `z=Pi_k(xW+b)`, uses a free decoder, constrains
each decoder block to the Frobenius unit ball, and minimizes reconstruction.
The encoder is initialized from the decoder transpose with a calibrated scale.
It is the nearest Fel model to BSC's *free encoder*, but differs on four axes:

- BSC has no encoder bias;
- BSC makes the concatenated decoder block orthonormal rather than bounding
  each single-site block's Frobenius norm;
- BSC uses batch-global rather than per-token TopK;
- BSC normally adds site-profile and Aux terms.

An `S=1` BSC is therefore not Vanilla BSF: it keeps the free encoder but swaps
the decoder ball for an orthonormal frame and deletes the affine bias.

### Grassmannian BSF

Fel's Grassmannian model uses `z_g=gamma xD_g^T`, a single learned positive
`gamma`, per-sample TopK, and an orthonormal decoder frame for each block. It
minimizes pure reconstruction and reprojects by QR; the paper reports that
every 20 steps is sufficient.

An `S=1` BSC shares the frame geometry, but its encoder is a separate learned
matrix and its polar retraction runs every optimizer step. This distinction is
large: the tied model makes selection and reconstruction use the same
subspace, while the free encoder can learn an oblique discriminative routing
map unrelated to the decoder frame.

### Group Lasso BSF

Fel's Group Lasso model uses an affine encoder, learned positive block
thresholds, the group soft-threshold
`(1-theta/||u||)_+ u`, and an `L_(2,1)` penalty. It obtains variable block
counts without coupling examples in a batch. BSC's calibrated inference
threshold also yields variable counts, but it hard-keeps the whole block and
does not shrink magnitudes. These are different estimators even if their mean
support is matched.

### Fel AuxK ambiguity

The paper's per-method descriptions specify pure reconstruction for Vanilla
and Grassmannian and an additional group penalty only for Group Lasso. A
shared Appendix-D loss later includes a runner-up block AuxK term with
`alpha=1/ell`; the toy description says auxiliary loss is absent unless
prescribed, and the released implementation does not make that Appendix term
part of the core architecture. A faithful comparison should therefore use
**no Aux** as the primary Fel bridge and treat the exact Appendix runner-up
Aux as a separately named arm. The local `aux_variant="fel"` is faithful only
when its runner-up count equals the main sparsity, since the paper's coefficient
is `1/ell`, not generically `1/s_aux` for an unrelated count.

### Fel synthetic bridge

The existing synthetic battery is a useful BSC stress test, not a Fel
reproduction. Fel's controlled recovery uses one site, ambient dimension 128,
four factors active without replacement, 256 blocks of width four, top four
blocks per sample, clean generated examples, and long 300k/100k train/eval
sets. The local generator uses four sites, six independent Bernoulli factors
with expected active count one, Gaussian noise `0.02`, 16 blocks, block
BatchTopK with `k=1`, and the SASA auxiliary path. A method bridge should first
reproduce the Fel setup and expected recovery ordering, then introduce sites,
batch selection, noise, and Aux one at a time.

## Crosscoder comparison in detail

### Anthropic's original ReLU/L1 crosscoder

For site activations `a^s`, Anthropic computes

`f = ReLU(sum_s W_enc^s a^s + b_enc)`

and reconstructs each site with `W_dec^s f + b_dec^s`. Its objective sums site
MSE and

`sum_i f_i sum_s ||W_dec,i^s||_2`.

The L1-of-site-norms form is intentional: compared with treating the
concatenated decoder as one L2-normalized vector, it charges a feature for
being present at each site and makes the loss comparable with a collection of
single-site SAEs. Anthropic separately normalizes each layer and sweeps feature
count and training steps/FLOPs.

BSC agrees on joint encoding and site-specific decoding, but differs on sign,
unit dimension, selector, decoder constraint, sparsity loss, bias, and input
gauge. In particular, BSC's site-profile penalty is *not* the block analogue of
Anthropic's activation-weighted L1-of-site-norms: it does not multiply by
activation, and the Stiefel constraint fixes total decoder energy.

The original model is still a necessary bridge because its optimization can
produce model/layer-specific norm artifacts that a normalized signed model
cannot. Without it, BSC has not shown that blocks solve a problem faced by the
published crosscoder rather than by its own scalar special case.

### Minder's L1 and BatchTopK crosscoders

Minder et al. identify two L1-crosscoder artifacts:

- **Complete Shrinkage:** a feature's contribution is suppressed in one model
  even when that feature is present;
- **Latent Decoupling:** separate latents encode related base/chat content,
  making one appear model-specific.

Their Latent Scaling diagnostic fits contribution scaling against
reconstruction and reconstruction error. Their BatchTopK variant starts from
the affine/ReLU code, scales each latent by the sum of its base/chat decoder
norms, selects the top `Bk` events globally, adds dead-latent residual Aux, and
fits an inference threshold to obtain mean count `k`. On Gemma 2 2B base/chat
layer 13 it used 100M training tokens, two epochs, learning rate `1e-4`, and
`k=100`; the paper's own model has 73,728 latents. Its L1 comparator used
`mu≈0.04`, the same learning rate, and realized L0 near 100.

This is the closest published selector to BSC. The remaining differences are
not cosmetic:

- Minder scores a nonnegative scalar activation times decoder norm; BSC scores
  a signed vector norm whose cross-site decoder energy is fixed by constraint;
- Minder has an encoder bias and free decoder norms; BSC has neither;
- Minder's model-specific-feature analysis uses Latent Scaling and causal
  activation replacement with None/All/error controls; BSC does not yet route
  corresponding shared-code falsifications as required endpoints;
- BSC's block code can hide coordinate-level site exclusivity even when the
  whole block appears shared.

The proper test is not just FVU. Apply a block generalization of Latent Scaling:
fit a small linear map or scalar/profile family from each block contribution
to held-out reconstruction and error, compare site-norm classification with
that fitted map, and run causal replacement/ablation. At minimum, use the
already specified site-only FVU matrix, leave-one-site-out reconstruction,
CCA, and Procrustes as cross-layer analogues.

## SASA comparison in detail

SASA is a single-site block model. It computes `p_k=E_k h`, keeps the `s`
blocks with largest `||p_k||_2` independently for each token, reconstructs with
`sum_k D_k a_k`, and minimizes

`sum_i ||h_i-Da(h_i)||_2^2 + lambda_dim sum_k ||D_k E_k||_*`.

The nuclear norm is on the end-to-end block map. This matters: it couples
encoder and decoder and can shrink unused singular directions of the actual
linear reconstruction operator. BSC's concatenated Stiefel decoder always has
capacity rank `b`, while its current regularizer only reallocates that decoder
capacity among sites. Decoder-frame singular values cannot substitute for
SASA's map spectrum or for activation-weighted contribution rank.

SASA's stated GPT-2 setting is `(K,r,s)=(2048,6,10)` and its Mistral setting is
`(4096,8,10)`, with `s_aux=512` and `256` respectively, auxiliary coefficient
one, AdamW at `2e-4`, weight decay `1e-3`, 1k warmup, linear decay over the last
fifth of training, batch 4096, and layer-normalized inputs. It defines a dead
group by frequency at most `1e-4` over 1,000 tokens, detaches the residual,
re-encodes it through dead blocks, and selects per-token top `s_aux` residual
blocks.

Current BSC borrows the residual/dead-group logic but uses a 100-*batch*
window—409,600 tokens at production batch size—plus a local gradient-ratio cap.
It also gives both the block and scalar arms `s_aux=256`, which means up to
1,024 block coefficients versus 256 scalar coefficients. These should not be
called faithful SASA settings until the window and capacity are matched and a
binding cap test passes.

There is one additional design subtlety for a true cross-site SASA bridge. If
the full concatenated decoder `Dbar` has orthonormal rows, then
`||Dbar^T Ebar||_* = ||Ebar||_*`; the decoder no longer supplies an adaptive
singular-value factor. A useful bridge should therefore compare:

1. a faithful single-site SASA with free `D_k,E_k`;
2. a multi-site free-map SASA generalization;
3. a Stiefel-decoder BSC with an explicitly acknowledged encoder-spectrum
   penalty; and
4. the current decoder site-profile penalty.

Conflating those objectives would make a nominal “SASA regularizer” arm much
less informative than it sounds.

## Hyperparameters and scale

Raw hyperparameter similarity is weak evidence across different activation
dimensions, site counts, modalities, and objectives. The useful comparison is
both literal and normalized.

| Setting | Current BSC Phase 1 | Fel DINO BSF sweep | Minder BatchTopK crosscoder | SASA |
|---|---|---|---|---|
| Data/model | Gemma 3 4B residual-post, 8 sites | DINOv3 ViT-B final-layer patches | Gemma 2 2B base/chat, layer 13 | GPT-2 and Mistral-7B single layers |
| Input dimension | `d=2560` per site; concatenated 20,480 | `d=768` | Model hidden size at two models | Model hidden size, one site |
| Dictionary | `G=4096`, `b=4` | `G=4096..32768`, `b=1..32` | 73,728 scalar latents | GPT-2 `K=2048,r=6`; Mistral `K=4096,r=8` |
| Main sparsity | `k=32` blocks; 128 active coefficients on average | `ell=8,16,32,64` per token | `k=100` scalar events on average | `s=10` blocks per token |
| Selector | Block BatchTopK; global threshold inference | Per-token block TopK; Group Lasso variant soft-thresholds | Scalar BatchTopK; global threshold inference | Per-token block Top-s |
| Optimizer | 8-bit AdamW moments, fp32 masters/bf16 forward | Adam | Paper training library; LR reported | AdamW, weight decay `1e-3` |
| Learning rate | `3e-4` to zero cosine | `1e-4` to `1e-5` cosine | `1e-4` | `2e-4`, linear late decay |
| Warmup | 1,000 steps | 2,000 steps | Not the transferable headline variable | 1,000 steps |
| Batch | 4,096 tokens | 8,192 patches | Paper reports token corpus/epochs; implementation-specific batch | 4,096 tokens |
| Budget | 76M optimizer tokens planned, two passes over 38M | Three epochs over activation shards | 100M tokens, two epochs | Paper claims about half-token budgets vs scalar baselines; task-specific |
| Normalization | Shrinkage whitening plus site RMS renorm | Scale mean norm to `sqrt(d)` | Not specified in the paper's training-details table | Layer-normalized activations |
| Structural penalty | `1e-3` site-profile; zero for fair scalar frontier | None for TopK arms; Group Lasso tuned | None in BatchTopK; L1 arm `mu≈0.04` | Map nuclear norm `lambda_dim` |
| Auxiliary | SASA-style, `s_aux=256`, coefficient 1, cap 1.0 | Core arms no Aux; appendix runner-up `alpha=1/ell` | Dead-latent residual, usually `k_aux=512`, `alpha=1/32` | `s_aux=512/256`, coefficient 1 |

### Are `G=4096`, `b=4`, and `k=32` sensible?

Yes as a starting operating point. All three literal values lie inside Fel's
real-activation sweep, and width four agrees with the project's synthetic and
Phase-0 results. But the normalized capacity is very different:

- Fel at `d=768,G=4096,b=4,k=32` allocates `Gb/d=21.3` decoder coordinates per
  input dimension and activates `kb/d=16.7%` as many coefficients as input
  dimensions.
- BSC at one `d=2560` site allocates `Gb/d=6.4` coordinates and activates
  `kb/d=5%`.
- Relative to the eight-site concatenated dimension 20,480, BSC allocates only
  `Gb/(Sd)=0.8` shared coordinates per input coordinate and activates
  `kb/(Sd)=0.625%`.

The cross-site structure should reduce redundant capacity, so the last ratio
need not match Fel. It does show why “same `G,b,k` as the paper” is not a
capacity argument. The appropriate sweep includes block count, active
coefficient count, parameter/FLOP budget, and coded rate.

### Does `lr=3e-4` make sense?

It is plausible but aggressive relative to the nearest papers: Fel uses a
`1e-4` peak, Minder reports `1e-4`, and SASA uses `2e-4`. BSC also has far more
parameters, bf16 forward weights, an 8-bit optimizer, per-step retraction, and
a batch-coupled selector. Phase-0 spike guards and deterministic tests are
evidence that it can run, not evidence that it is optimal. The executable
matrix should compare at least the current schedule with a `1e-4` Fel/Minder
bridge and a `2e-4` SASA-like arm, holding optimizer-token budget fixed.

### Does `lambda_site_profile=1e-3` make sense?

Only as an empirically screened strength for the *site-profile* objective.
Phase 0 rejected `3e-3` and retained `3e-4`/`1e-3` under a flat-profile veto.
There is no meaningful numerical comparison to SASA's `lambda_dim`, because
the functions, normalizations, and parameter gauges differ. Architecture-fair
frontiers should remain at zero; any regularized headline needs a three-way
none/site-profile/map-spectrum ablation.

### Does the 76M-token budget make sense?

It is a reasonable forecast, not a demonstrated convergence budget. Phase 0
showed a 0.016–0.020 FVU gain from doubling optimizer tokens to 24M while fresh
data at fixed budget contributed only 0.0013. That favors repeats if storage is
binding, but it also warns that architecture comparisons can be budget-limited.
The 76M run needs a predeclared held-out slope gate and a matched 100M extension
for every surviving headline arm if still improving.

## Exhaustive conceptual factor space

The scientific object is a factorial family, not one scalar “BSC versus
papers” switch. The conceptual space below is intentionally broader than what
should be executed as a Cartesian product.

| Axis | Levels that must be distinguished |
|---|---|
| Site structure | One site; multiple layers; base/chat models; mixed layers and models |
| Shared object | No sharing/independent models; shared scalar; shared vector block; shared presence but site-specific coordinates |
| Block width | `b in {1,2,4,6,8,16}` plus task-driven widths |
| Dictionary capacity | `G`; total coefficients `Gb`; parameter matched; FLOP matched; memory matched |
| Decoder geometry | Unconstrained scalar; per-block Frobenius ball; per-block Stiefel; concatenated cross-site Stiefel; free map with weight decay |
| Encoder relation | Tied with one scale; untied linear; untied affine; partially tied/shared encoder |
| Code sign/nonlinearity | Signed linear block; ReLU scalar; block soft threshold; other cone constraints only as explicit controls |
| Training selector | Per-token TopK/Top-s; BatchTopK; learned/global threshold; L1; group lasso |
| Inference selector | Same as training; calibrated threshold; fixed per-token count; rate-targeted per-sequence code |
| Sparsity match | Mean block events; coefficient events; support entropy; full coded bits; per-token count distribution |
| Structural regularizer | None; current site-profile; SASA end-to-end map nuclear; Anthropic activation-weighted L1-of-site-norms; group `L_(2,1)` |
| Auxiliary path | None; Fel runner-up; Minder dead-latent; SASA residual dead-block; block-event matched; coefficient-capacity matched |
| Deadness definition | Token horizon; batch horizon; frequency threshold; time-since-last-fire; no revival |
| Input gauge | Raw/isotropic norm; separately normalized sites; layer norm; shrinkage whitening; whitening plus site renorm |
| Reconstruction metric | Transformed-coordinate MSE/FVU; raw-coordinate MSE/FVU; site weighted; downstream causal/behavioral loss |
| Optimizer | Adam; AdamW; 8-bit moments; full precision moments; fp32 vs bf16 forward; weight decay policy |
| Schedule | Peak LR; warmup; cosine/linear decay; floor; retraction frequency |
| Budget | Unique tokens; optimizer tokens; steps; epochs/repeats; FLOPs; early-stop rule |
| Initialization | Tied decoder transpose; independent encoder; norm calibration; bias initialization; cold versus warm bridge |
| Causality | Acausal all-site encoding; prefix/causal encoding; site-only; leave-one-site-out |
| Evaluation | FVU; deadness; support distribution; R–D; factor recovery; map/code/contribution rank; Latent Scaling; patching/intervention |

Executing the full Cartesian product would be wasteful and statistically
opaque. The correct use of this table is to prevent hidden factor changes,
then prune through staged gates.

## Staged executable matrix

### Stage A — Paper-faithfulness and implementation validation

Run small, high-precision, two- or three-seed cells. No whitening, local
regularizer, custom guard behavior, or SASA Aux should enter a bridge until the
paper-faithful version passes its own recovery/reconstruction check.

| Cell family | Fixed factors | Deliberate comparison | Pass condition |
|---|---|---|---|
| A0 Fel toy reproduction | `S=1,d=128,G=256,b=4,k=4`, clean Fel generator, per-token TopK, Adam, paper budget | Vanilla vs Grassmannian vs Group Lasso; paper constraints/bias/tie | Recovery ordering and near-oracle block recovery qualitatively reproduce paper; deterministic artifact |
| A1 Local synthetic delta | Start from A0 | Add local noise, Bernoulli supports, then BatchTopK, then Aux one at a time | Each delta's effect on assignment, contribution R², deadness, and FVU is attributable |
| A2 Original crosscoder bridge | Small multi-site activation store, `b=1`, separately normalized sites | Anthropic affine/ReLU/L1 vs independent SAEs | Published loss definition, decoder-norm profiles, L0/FVU and feature-sharing diagnostics work |
| A3 Minder bridge | Same data and width as A2 | ReLU/L1 vs ReLU/BatchTopK with exact score, threshold, and Aux | BatchTopK support and threshold calibration reproduce target L0; Latent Scaling detects injected shrinkage/decoupling controls |
| A4 SASA bridge | Single-site language activation store, paper-like `(K,r,s)` at reduced width if necessary | Map nuclear on/off; exact 1k-token dead window and Aux | Map spectrum responds to `lambda_dim`; revival behavior and per-token Top-s match source semantics |
| A5 BSC identity tests | Synthetic multi-site rotated frames | Joint vector code vs independent `S=1 BSC`; exact concatenated Stiefel | Recover shared support/coordinates up to `O(b)` gauge; site-only and leave-one-out tests pass |

Stage A is also where numerical tests belong: fp64 Gram/gradient references,
post-step rollback, checkpoint experiment mismatch refusal, saved-codec
round-trips, and zero-rate exclusion tests.

### Stage B — Isolate BSC innovations on a common pilot store

Use a moderate fixed `G,b,k`, one common activation gauge per submatrix, common
optimizer-token checkpoints, and at least two seeds for promoted comparisons.
Change one factor family at a time.

| Submatrix | Arms | Question |
|---|---|---|
| B1 Encoder/decoder | Tied Grassmannian; free linear/Stiefel; free affine/Stiefel; free affine/Frobenius-ball | Is the hybrid free-encoder/Stiefel choice better, and is encoder bias necessary? |
| B2 Selector | Per-token block TopK; block BatchTopK; learned/group threshold; calibrated hard threshold training | Does variable support help enough to justify batch coupling? |
| B3 Structural penalty | None; current site-profile at `3e-4,1e-3`; true map-spectrum matched on effective scale | Is the winner due to useful site localization or coordinate partition? |
| B4 Aux | None; Fel runner-up; SASA dead residual; Minder-style dead residual | Which revival prior improves final held-out quality rather than only early deadness? |
| B5 Aux fairness | Same block-event count; same maximum coefficient count; matched realized Aux rate | Does the block arm retain an advantage without 4× auxiliary capacity? |
| B6 Gauge | Isotropic/raw; separate layer norm; shrinkage whitening; whitening plus site renorm | Is the architecture advantage stable outside the promoted transformed metric? |
| B7 Scalar bridge | Signed normalized scalar; Anthropic ReLU/L1; Minder ReLU/BatchTopK | Is the block gain specific to the current scalar baseline? |
| B8 Sharing | Joint BSC; eight `S=1 BSC`; site-specific coordinate maps; scalar crosscoder | Does one shared vector code fit, or is shared presence with flexible coordinates better? |

Primary endpoints for every B cell are held-out transformed and raw FVU,
per-site FVU, realized support distribution, dead fraction, coded rate from a
saved codec, training FLOPs, and stability. BSC-specific endpoints are the
site-only/all-site FVU matrix, leave-one-site-out, CCA/Procrustes, centered
code/contribution/map spectra, and truncation FVU.

### Stage C — Confirmatory production shortlist

Only Stage-B winners reach the 4B production store. Keep the shortlist small
enough that all arms receive the same budget and two confirmatory seeds.

Recommended minimum shortlist:

1. current BSC without structural penalty;
2. best site-profile or map-spectrum BSC, selected without using confirmatory
   eval;
3. best faithful block bridge (`S=1` controls plus the multi-site extension);
4. signed scalar BatchTopK control;
5. Minder-style ReLU BatchTopK crosscoder;
6. independent single-site block and scalar controls.

Checkpoint every arm at common optimizer-token budgets (for example 24M,
50M, and 76M). At 76M, apply the preregistered convergence gate. If any
headline arm is still improving materially, extend **all headline arms** to at
least 100M. Winner election requires full calibration, a serialized codec,
all safety/provenance gates, all shared-code/effective-span endpoints, and raw
as well as transformed distortion.

### Stage D — Sensitivity and frontier mapping

After a model family wins, sweep only the factors that define its operating
frontier:

- `b in {1,2,4,8}` initially, adding six if comparison with SASA warrants it;
- `G` at parameter- and coefficient-capacity-matched points;
- target block rate and target coefficient rate, including variable-support
  thresholds;
- quantization `q in {4,6}` plus bf16 shadow;
- unique-data versus repeat budget at fixed optimizer tokens;
- peak LR `1e-4,2e-4,3e-4` with schedule family fixed;
- structural coefficient around the selected zero/nonzero boundary;
- causal/site-prefix versus fully acausal encoding where the scientific claim
  depends on temporal formation.

Report Pareto surfaces over distortion, coded rate, parameters/FLOPs, and
scientific recovery. Do not choose `b` from FVU alone: reconstruction is
monotone in capacity. Use coded rate and used-span/truncation evidence.

## Decision rules

The staged program should answer concrete forks rather than accumulate cells.

1. **Keep BatchTopK** only if it improves held-out distortion/rate or recovery
   over per-token TopK without unacceptable count tails or token-class harms.
2. **Add encoder bias** if it fixes offset/antipodal recovery or improves real
   held-out results without support collapse; otherwise retain the simpler
   linear encoder and document the negative result.
3. **Keep the site-profile penalty** only if it improves causal/shared-code
   endpoints after controlling for site-exclusive coordinate partition. Never
   call it rank adaptation.
4. **Prefer map-spectrum regularization** only if it reduces code-anchored used
   span or improves truncation/R–D, not merely decoder singular values.
5. **Use an auxiliary path** only if final held-out quality or recovery improves
   at matched auxiliary capacity. Early dead-count reduction alone is not
   enough.
6. **Claim a BSF extension** only after exact Fel bridges reproduce expected
   behavior and the multi-site delta is isolated.
7. **Claim a crosscoder improvement** only against both the internal signed
   scalar control and at least the Minder ReLU/BatchTopK bridge, with
   Latent-Scaling-style and causal diagnostics.
8. **Claim effective dimension** only from activation-weighted contribution or
   map spectra plus truncation, never decoder capacity alone.

## Recommended formulation language

Until the matrix is executed, the defensible concise description is:

> A BSC is a cross-site, block-sparse dictionary model with one signed vector
> code inferred jointly from several activations and a site-specific decoder
> frame. Its concatenated decoder frame is orthonormal, so block norm equals
> isolated total decoded energy. The current training procedure uses block
> BatchTopK and a free linear encoder; it is a hybrid inspired by, but not an
> exact implementation of, any one BSF, crosscoder, or SASA variant.

The following descriptions should be avoided without new evidence:

- “SASA rank regularization” for the decoder site-profile penalty;
- “the Fel training procedure extended across layers” for the current hybrid;
- “the original crosscoder baseline” for the signed normalized scalar arm;
- “effective block dimension” for raw decoder-frame singular values;
- “paper-faithful SASA Aux” for a 409,600-token dead window and unmatched
  auxiliary coefficient capacity.
