# Paper bridges and executable conditional matrix

*Normative comparison ledger, 2026-07-20. `studies.py` is authoritative for
the exact serialized cells; this document is authoritative for their scientific
interpretation.*

## 1. The methods do not form one flat grid

[BSF](https://arxiv.org/abs/2606.25234) and
[SASA](https://arxiv.org/abs/2606.06333) learn signed vector blocks at one
site. [Anthropic's original crosscoder](https://transformer-circuits.pub/2024/crosscoders/index.html)
learns one nonnegative scalar code from several sites. The native BSC combines
the cross-site topology with the vector-block ontology; no reviewed paper
publishes that combination.

The matrix is therefore conditional:

1. test source equations and invariants;
2. establish truth-known identification;
3. select one parent;
4. change one/few related decisions;
5. promote only complete-seed winners to the next round;
6. confirm the final recipe without tuning.

Every row is a whole recipe. A selector, decoder gauge, regularizer, or Aux
bundle is not transplanted invisibly.

Hard TopK tie behavior is one universal engineering contract, not a method
axis: score descending, then lowest block index within a token or lowest
row-major `(token, block)` index batch-wide at the exact cutoff. Threshold
equality is excluded by strict greater-than. No recipe inherits backend TopK
tie order.

## 2. Source-method comparison

| Source recipe | Sparse object | Encoder and selector | Decoder/gauge | Objective beyond reconstruction | Project role |
|---|---|---|---|---|---|
| BSF Vanilla | signed vector block | untied affine; per-token block TopK | per-block Frobenius ball | none | paper parent and single-site geometry anchor |
| BSF Grassmannian | signed vector block | decoder-tied with one positive scale; per-token block TopK | Stiefel block; periodic QR | none | tied-inference and subspace-gauge anchor |
| BSF Group Lasso | signed vector block | affine; learned group soft threshold | source scale control | conditional group `L2,1` | variable-support block anchor |
| SASA paper | signed vector block | free linear; per-token Top-s | free block maps | `sum_g ||D_g E_g||_*`; whole-group dead-residual Aux | effective-map-dimension and block-revival anchor |
| Anthropic original | nonnegative scalar | sum affine site encoders; dense ReLU during training | free affine decoder per site | activation times the sum of per-site L2 decoder norms | same-model cross-layer comparator family; never sparse-finalist-eligible |
| decoder-weighted BatchTopK adaptation | nonnegative scalar | sum affine site encoders; score by activation times sum of decoder norms; batch-global event allocation | free affine decoder per site | optional token-horizon residual Aux as a separate bundle | strongest scalar multi-site mechanism comparator |
| adaptation from *fmxcoders: Factorized Masked Crosscoders for Cross-Layer Feature Discovery* | signed vector block retained locally | Tucker-style low-rank site-axis encoder/decoder factors; optional stochastic encoder-site masking | compatible free decoder | clean-target reconstruction from partial layer evidence | Phase-1 capability evidence and Phase-2 real-model tuning; not a reproduction of the source tensor factorization |
| BSC | signed vector block | joint sum/mean/source encoder; block selector | site-specific block decoders with declared cross-site gauge | optional named block penalty or Aux | novel target method |

The original Anthropic training model is retained exactly as dense ReLU plus
the disclosed L1-of-site-norms objective. Its norm geometry is
`sum_s ||d_i^s||_2`, not `||(d_i^1,...,d_i^S)||_2`. Calibration fits a
deployment threshold so packet behavior can be measured. This post-training
sparsification does not make it eligible as the sparse finalist, but its
Phase-3 comparator is selected through its own Phase-2 calibration chain.

The decoder-weighted BatchTopK and token-horizon Aux rows are explicitly
`adapted`. They preserve the relevant equations while changing the task to
same-model layers. They make no claim to reproduce the source experiment.

## 3. Paper settings and disclosure gaps

### 3.1 BSF

The truth-known source generator has ambient width 128, 128 factors, four
active factors per example, 300,000 training and 100,000 held-out examples,
independent seeds, and no primary observation noise. The executable Phase-1
bridge separately binds 50,000 factor-calibration examples, 300,000 unique
training examples, and disjoint 100,000-example codec-calibration,
development, and confirmation ranges. Half the factors are
one-dimensional; the remainder span circles, disks, spheres, tori, Möbius
strips, Swiss rolls, and helices after factorwise calibration and random
orthonormal embedding.

The core anchor uses `G=256`, `b=4`, and four active blocks. Vanilla uses an
affine encoder and Frobenius-ball projection. Grassmannian ties inference to
the decoder and reports QR about every 20 steps. Group Lasso learns block
thresholds and penalizes group activity while activity exceeds target.

Appendix D adds a distinct runner-up residual auxiliary: select the next four
unselected blocks and weight their residual reconstruction by `1/4`. The live
matrix therefore separates primary and Appendix-Aux recipes. The exact toy
optimizer schedule is incompletely disclosed; transferred batch/LR/epoch
values remain adapted even when the architecture is exact.

### 3.2 SASA

The defining penalty is the nuclear norm of the end-to-end map,

\[
L=L_{\rm rec}+\lambda_{\rm dim}\sum_g\lVert D_gE_g\rVert_* ,
\]

not the decoder nuclear norm. A group is dead when its firing frequency is at
most `1e-4` over 1,000 tokens; the detached residual is re-encoded with dead
groups and the selected auxiliary groups reconstruct it at coefficient one.

The GPT-2 SASA setting is residual-pre block 7 on OpenWebText, 150M tokens,
context 128, `(groups,width,active)=(2048,6,10)`, 512 auxiliary groups, token
LayerNorm, token batch 4,096, AdamW, LR `2e-4`, WD `1e-3`, 1,000 warmup steps,
and final-fifth linear decay. In the paper's sample-efficiency comparison, that
150M SASA run is compared with a standard externally trained SAE using 300M
tokens; `300M` is not a second SASA budget. Its controlled architecture
comparison instead trains the scalar baselines on the same data and token
budget. The Mistral SASA setting uses block 8, 500M tokens, context 512,
`(4096,8,10)`, and 256 auxiliary groups with the same optimizer family.

The paper does not disclose numeric `lambda_dim`, so the live paper bridge does
not transfer an invented absolute coefficient across dimensions or reduction
conventions. After all declared initialization and encoder-scale fitting, it
measures the unweighted map penalty and reconstruction loss on the hash-bound
first training batch in fp32 and resolves the coefficient to target an initial
penalty/reconstruction ratio. The independent SASA ladder is
`0/.01/.03/.10`, centered at `.03`; the zero arm retains the map objective.
The inspected release's absolute value `100` belongs only to its decoder-only
nuclear-norm drift recipe, which is diagnostic and nonpromotable.

### 3.3 Anthropic architecture bridge

The disclosed equations are

\[
f(x)=\operatorname{ReLU}\left(\sum_sW^s_{\rm enc}x^s+b_{\rm enc}\right),
\qquad
\hat x^s=W^s_{\rm dec}f+b^s_{\rm dec},
\]

with squared reconstruction and

\[
\lambda\sum_i f_i(x)\sum_s\lVert d_i^s\rVert_2 .
\]

Separate site normalization and width/compute sweeps are described, but the
model, corpus, exact normalization, token budget, width values, optimizer,
batch, LR, coefficient, initialization, and dead-feature treatment are not
sufficiently disclosed for numerical reproduction. Every numeric runtime value
in the executable architecture bridge is therefore adapted or engineering.

### 3.4 Adapted decoder-weighted mechanisms

For dense positive activation $f_{ti}$, the carrier scores

\[
v_{ti}=f_{ti}\sum_s\lVert d_i^s\rVert_2,
\]

retains the largest `batch_tokens * target_events` scores across the batch, and
decodes the corresponding unscaled $f_{ti}$. Deployment uses a threshold fit
only on calibration to reproduce the support target.

The optional residual-Aux bundle declares death by accepted-token horizon,
scores only dead candidates with the same decoder weighting, and reconstructs
the detached residual. The implemented source-grounded pilot bundle has
coefficient `1/32`, Aux width 384, and a 10M-token dead horizon. It is not a
live Phase-2 matrix arm: the live decoder-weighted anchor isolates selection,
and `auxiliary_16m` isolates BSF and SASA auxiliaries. Activating the Minder
bundle would require a separately declared selected-parent round.

## 4. Exact live Phase-1 matrix

Default development seeds are 0, 1, and 2. Initial stages are materialized;
later stages are exact selected-parent children.

| Stage | Recipes/variants | Selection or gate |
|---|---|---|
| `paper_anchors` | BSF Vanilla/Grassmannian/Group-Lasso, each primary and Appendix Aux; SASA paper; Anthropic architecture | integrity-complete gate only |
| `representable_controls` | signed scalar token-TopK; scalar ReLU decoder-weighted BatchTopK; source-only block; source-only scalar | opens after complete anchor evidence |
| `fusion_identification` | shared-coordinate sum; shared-coordinate mean | non-promotable parity diagnostic after complete controls |
| `dgp_identification_screen` | BSC single-site; support-only stress; shared coordinates | shared-coordinate arm is the sole eligible parent |
| `capacity_identification` | widths `1,2,4,8` at fixed total/active coordinates; width-4 half/double capacity; width-4 half/double activity | capability only; fixed `width_4` carrier advances |
| `retraction_identification` | QR concatenated Stiefel; symmetric-polar concatenated Stiefel | capability only; fixed `qr_retraction` carrier advances |
| `site_factorization_identification` | exact selected parent; unfactorized full free-site weights; site ranks `1,2,4` | capability only; fixed `selected_parent_carrier` advances |
| `site_mask_fusion_control_identification` | literal sum `p=0`; literal sum `p=.10`; availability-rescaled sum `p=.10` | capability only; fixed rescaled-sum `p=.10` carrier advances; literal positive masking is diagnostic |
| `site_masking_identification` | Bernoulli clean-target masking `0,.02,.05,.10`; exactly one hidden; exactly one retained | capability only; fixed zero-mask carrier advances |
| `selection_score_identification` | code norm, exact isolated decoded energy, and exact isolated squared-loss decrease on the Stiefel equality-control carrier; the same three scores on one common free decoder | capability only; every free-decoder arm is nonpromotable and the fixed Stiefel decoded-energy provisional carrier advances |
| `selector_identification` | token block-TopK; block BatchTopK | capability only; fixed token-TopK carrier advances |
| `robustness_confirmation` | baseline; support-only; site rotation; site-scale ratio 2; noise `0.1`; rank heterogeneity; two/eight active factors; rank-two/independent site maps; one/two-site spans; Zipf-alpha-one frequency; pair forcing `.5/.9`; standardized Student-t df=3 coordinates; paired 30-degree factor subspaces | confirmation stream, no selection; support-only and one-site spans are negative controls; independent maps retain shared coordinates and stress site-axis factorization |

Every capability challenger still runs all seeds and contributes its
qualification digest and pass/fail outcome, but it is nonpromotable and cannot
replace the named carrier or prune a real-model option. No Phase-1 capability
round chooses a model-specific winner. All advancing carriers require complete
seeds and a passed scientific outcome; their metric is the minimum normalized margin
across same-block support/subspace/code recovery and aggregate pathology
guardrails.

For observed sites $O$, the new signed score is

\[
\Delta_g(O)=\lVert x_O\rVert^2-\lVert x_O-D_{g,O}^{\top}z_g\rVert^2
=2\langle x_O,D_{g,O}^{\top}z_g\rangle
-\lVert D_{g,O}^{\top}z_g\rVert^2.
\]

It excludes hidden clean targets, requires a bias-free quadratic objective,
retains negative gains, is invariant to reciprocal within-block gauge, and
reduces to $\lVert z_g\rVert^2$ under unit-scale tied concatenated Stiefel. Its
deployment threshold uses a signed deterministic histogram.
The score definition is independent of the engineering contraction identity:
the mapped free-decoder implementation evaluates the same signed quadratic by
one flattened decoder projection and mapped block-Gram products, with exact
observed-site weighting for partial and source-only views. This optimization
is a local adaptation, not a paper value or a distinct score arm.

At seeds 0, 1, and 2 the serialized variants declare and execute **198 cells**:
51 initial, 96 capability/contract cells, and 51 confirmation cells. The score
panel contributes 18 cells: three Stiefel equality controls and the same three
scores on a common free decoder, at all three seeds. Synthetic LR,
native-regularizer, and Aux tuning rounds do not exist.

### 4.1 Transfer to Phase 2

The frozen `bsc-phase1-transfer-v2` payload separates the universal method
contract, the signed-coordinate/decoded-energy provisional carrier, and all
diagnostic capability panels; it also binds claim-scope narrowing, selection
IDs, source plan/blueprint IDs, evidence hashes, and its own content ID. It
contains no synthetic numeric winner for architecture, width, rank, masking,
score, selector, optimizer, or scale, and capability failures do not filter the pilot.
The signed coordinate ontology is universal, while the literal activation
operator and score are explicitly reopened; group soft thresholding remains a
signed-coordinate Phase-2 method.
A runnable Phase-2 blueprint binds both the
Phase-1 decision ID and transfer ID; its `phase1_contract_bsc` anchor is rebuilt
from that evidence, while an unbound preview remains non-runnable.

## 5. Exact live Phase-2 matrix

Default pilot seeds are 0 and 1. Every cell uses the same pinned four-layer
GPT-2 raw task and an actual saved-codec round trip. Site count is a later
robustness question, not an architecture-search confound.
All non-smoke CUDA recipes execute their named Adam/AdamW procedure through
the explicitly bound fused kernel with `foreach=False`; CPU smoke uses the
explicit scalar kernel. This changes engineering arithmetic, not the paper
optimizer name or hyperparameters, and its standardized scalar-versus-fused
trajectory sensitivity is reported rather than treated as a matrix factor.

| Stage | Budget and split | Recipes/variants |
|---|---|---|
| `anchors_1m` | 1M, development | BSF Vanilla/Grassmannian/Group-Lasso; scalar ReLU BatchTopK; SASA; Anthropic dense-L1; adapted decoder-weighted BatchTopK; evidence-bound `phase1_contract_bsc`; nonpromotable `phase1_contract_source_only_control` |
| `architecture_4m` | 4M, development | exact selected parent; labeled parent architecture; no init preconditioning; tied Grassmann `b=4` QR; tied Grassmann `b=4` polar |
| `capacity_4m` | 4M, development | exact selected parent; widths `1,2,4,8` at fixed total/active coordinates; width-4 half/double capacity; width-4 half/double activity |
| `site_factorization_4m` | 4M, development | exact selected parent; unfactorized full free-site weights; factorized site ranks `1,2,4` |
| `site_masking_4m` | 4M, development | exact selected parent; Bernoulli clean-target masking `0,.02,.05,.10`; exactly one hidden; exactly one retained |
| `site_factorization_revisit_4m` | 4M, development | exact selected masked parent; unfactorized full free-site weights; factorized site ranks `1,2,4`; only the exact parent materializes when zero Bernoulli masking wins |
| `hard_selector_score_interaction_4m` | 4M, development | full 3-by-2 product of code norm, exact isolated decoded energy, or exact isolated loss decrease with signed token-TopK or signed block-BatchTopK; decoded-energy/token-TopK is the exact incoming control |
| `group_threshold_method_4m` | 4M, development | exact selected parent; complete affine group-soft-threshold/unit-Frobenius/conditional-L2,1 method bundles at `3e-4`, `1e-3`, or `3e-3` |
| `learning_rate_4m` | 4M, development | exact selected parent; `3e-5`, `1e-4`, `3e-4` |
| `batch_size_4m` | 4M, development | exact selected parent; 2,048; 4,096; 8,192 optimizer tokens per batch |
| `warmup_4m` | 4M, development | accepted-update fractions `.02`, selected-parent `.05`, `.10` |
| `schedule_4m` | 4M, development | exact selected parent; constant after selected warmup; cosine; final-fifth linear decay |
| `learning_rate_revisit_4m` | 4M, development | exact selected parent; revisit `3e-5`, `1e-4`, `3e-4` after batch/warmup/schedule selection |
| `regularization_16m` | 16M, development | exact selected parent; no regularizer/Aux; SASA map nuclear at initial penalty/reconstruction ratios `.01/.03/.10`; decoder-only nuclear diagnostics at absolute `30/100/300` |
| `auxiliary_16m` | 16M, development | exact selected parent; no Aux; BSF runner-up Aux; SASA source, low-weight, or long-window dead-group Aux |
| `confirmation_16m` | 16M, confirmation | scalar RMS; none; `sqrt_d`; shrinkage whiten; token LayerNorm |

Observation-site/evidence topology and missing-site fusion are absent from the
Phase-2 tuning chain; model architecture is explicitly retuned.
Availability-rescaled fusion is inherited as a universal semantic. Decoded
energy enters only as a provisional score carrier. Site-axis rank and mask
probability are retuned because their optimum is model-, hook-, scale-, and
rate-dependent, and rank is revisited after the selected mask. The pilot then
runs the complete three-score by two-hard-selector interaction rather than a
coordinate-descent score/selector/revisit sequence. Learned group thresholding
is isolated as a bundled method because it changes the affine encoder, bias,
decoder constraint, shrinkage activation, L2,1 objective, and activity schedule
together.
For a learned group-threshold arm, dense training support is the nonzero
post-shrinkage code, independent of the inherited endpoint-ranking score. The
score is reintroduced only at calibrated deployment, so isolated loss decrease
cannot silently become an extra hard training gate.
The tied-architecture QR/polar comparison is intentionally narrow: Phase 1
measured both mechanisms and advanced QR by declaration, whereas real
conditioning and dimension can change their optimization behavior. No
truth-known capability failure deletes a Phase-2 option.

The live coordinate contract is exact at every width:
`groups=8192//block_width` and
`active_blocks=32//block_width`. Architecture rows keep total latent coordinates and nominal active
coordinates explicit; the headline comparison is nevertheless made again at
achieved packet rate.

Every development stage retains one candidate using mean raw FVU at the frozen
256, 384, and 512 total-bit/token budgets. Operational packet bits and exact
serialized codec bytes amortized over 100M tokens are included. Adjacent
lower-envelope mixtures use the content-bound
`balanced_global_token_counter_u64_v1` schedule, requiring no per-token side
bits. Each selected operating point is serialized as an exact 32-byte record
in the immutable `deployment_schedules` bundle, reloaded through the consumer
path, and amortized with the deployable codec. Mixture distortion is measured
by executing those bytes on paired raw rows, not by averaging the aggregate
endpoint FVUs. The score comes only from the
lower convex envelope of measured zero-event and
2/4/6/8/12/16-bit amplitude points; no extrapolation is allowed. Seeds
aggregate by median, then worst seed, then candidate ID. `confirmation_16m`
has no selection policy. The anchor allowlist admits only the shared-coordinate
BSC to the first transition, and the source-only anchor is explicitly
nonpromotable. Every ordinary main-chain development round declares an inert
exact parent. Materialization retains one representative of each resolved
execution-value signature and records every redundant parent/center label it
elides. A child replaces the retained parent only if its fixed-rate score
improves by at least `0.002` on every seed and on the median and worst-seed
aggregates. The initial factorization round instead requires the full free-site
carrier to remain within `0.01` of the exact selected parent, then chooses the
lowest of rank `1`, rank `2`, rank `4`, and full that remains within `0.01` of
full on every seed and on the median/worst aggregates. The post-mask rank
revisit uses ordinary parent retention and emits no rank children when the
selected mask is Bernoulli zero.

The nonselectable confirmation round does not tune a gauge. Panel freeze uses
the scalar-RMS rerun and requires every seed to re-pass qualification and the
sharing guard while remaining within `0.02` fixed-rate score of its exact
development parent. This is a novel project reproducibility rule, content-
bound in the confirmation cells with a `.01/.02/.05` marginal sensitivity
report and an ungated descriptive result; it is not attributed to a paper.

Every selected-parent/revisit development policy also freezes conjunctive
sharing admission. For both site-only and leave-one-out inference, the
worst-site decoded-coordinate Lin concordance in the all-site decoder-Gram
geometry, with a mean-offset penalty, must be at least `.80`. Worst-site
support-intersection recall and full-view decoded-energy coverage on that
intersection must be at least `.75` and `.90`, respectively. Parent-relative
site-only and leave-one-out FVU degradations remain capped at `.02`, their mean
support-IoU drops at `.05`, root-relative FVU degradation at `.02`, and absolute
partial-view FVU at `1.0`. The same-candidate all-view FVU advantage is reported
only descriptively, not gated: redundant shared factors need not exhibit
positive reconstruction synergy. This is not a comparison with the separately
trained source-only anchor. Missing guard data fails before candidate
aggregation.

These winner-changing practical-effect, noninferiority, and sharing thresholds
are novel preregistered project policies, not values from any paper. Each
applicable policy content-binds the complete sensitivity grid: minimum effect
`0/.001/.002/.005`; noninferiority `.005/.01/.02`; FVU degradation
`.01/.02/.05`; support-IoU drop `.02/.05/.10`; concordance, intersection recall,
and decoded-energy coverage `.50/.80/.90`, `.50/.75/.90`, and `.75/.90/.95`,
respectively; and absolute FVU `.75/1.0/1.25`.
Each selection artifact reports marginal counterfactual pass sets from the
authenticated measurements without retuning the center policy.

Seven independent comparator-family chains branch from their own anchor
selection. Width/activity are calibrated for block families; Group Lasso
calibrates its coefficient and Appendix-Aux bundle; SASA calibrates its
`0/.01/.03/.10` initial map-penalty ratios and source Aux bundles; Anthropic
calibrates L1 coefficient; both BatchTopK scalar families calibrate
batch size; every family calibrates activity, schedule, and the exact four-rate
learning-rate ladder `3e-5/1e-4/2e-4/3e-4` as applicable. One top-two nomination
policy ranks the complete union of qualified
4M family-round candidates, deduplicates resolved non-replicate execution
signatures before outcome ranking while preserving every stage/candidate alias
and metric spread, and binds one universe hash. The earliest declared source
round is the representative, preventing best-of-repeats bias. Comparator-family policies report but do not gate on BSC sharing
admission, so a deliberately non-sharing baseline cannot disappear before the
comparison. Each chain ends with a fresh 16M-token revisit of that overall
winner and strongest distinct resolved runner-up,
followed by a one-winner family selection. This probes local path/order sensitivity but does not
claim a global optimum. Phase 3 consumes those content-addressed family
selections, never the root anchors. At seeds 0 and 1 the blueprint derives
a pre-elision ceiling of **176 main-chain cells** and **238 family cells**,
**414 total**, computed from the manifest. The main chain has 18 anchors and
158 declared cells in 15 rounds from architecture through confirmation.
Execution-equivalent parent/center cells are deterministically elided, and the
rank revisit conditionally loses four children when Bernoulli-zero masking
wins, so the realized count is lower and recorded in each materialized stage.

Scientific-outcome guardrails require calibrated mean support within `0.1`
block of target and no more than `1%` selected events excluded on calibration
or evaluation. A candidate with an oracle-only raw inverse or an ineligible
fixed budget cannot promote.

## 6. Frozen Phase-3 panel

The Phase-2 evidence producer freezes one exact selected recipe and seven
comparators:

| Slot | Role |
|---|---|
| selected finalist | exact Phase-2-derived recipe |
| shared-coordinate BSC | mechanism comparator |
| BSF Grassmannian | paper comparator |
| BSF Group Lasso | paper comparator |
| SASA | paper comparator |
| Anthropic dense L1 | independently calibrated architecture comparator |
| decoder-weighted BatchTopK | adapted mechanism comparator |
| scalar ReLU BatchTopK | controlled scalar baseline |

Every non-finalist panel entry must carry a derived family recipe, its complete
selection chain, family blueprint ID, and root lineage. If a ranked comparator
duplicates the selected finalist, its serialized slot policy advances to the
next ranked nonduplicate; a duplicate selected-finalist slot fails closed.

Before the final panel can open, all eight frozen designs run one 262,144-token
production-shape stability cell on a dedicated `stability` split. Each uses the
exact Gemma hook geometry and capacity, records an fp32-versus-bf16 initial
forward comparison, requires reconstruction relative error at most `.05` and
support IoU at least `.90`, and completes the short optimization without
nonfinite state. This is a conjunctive refusal gate, not a ranking or tuning
stage, and it never reads the final split. The frozen panel then runs five
seeds, four pinned Gemma layers, 100M optimizer tokens per cell, 25M unique
training tokens, and untouched final evaluation at 1,024, 1,536, and 2,048
total bits/token. Those budgets are the exact fourfold transfer of the pilot
`256/384/512` frontier, preregistered from the nominal active-coordinate ratio
`128/32`; the preflight also requires nonzero packet coverage and at least two
distinct nonzero frontier endpoints. Phase 3 therefore contains
eight preflight cells plus 40 final cells; no Phase-3 row has a selection
policy.
The resource envelope uses estimator schema
`dense-linear-memory-v14-q2-c512-t256-s32`. Its guarded
`stiefel_code_norm_bounded_v1` implementation is an engineering specialization,
not a paper result or a different scientific score. It uses the algebraic
decoded-energy/code-norm equality only for an unfactorized Gram/QR Stiefel
decoder with hard TopK selection and every-update retraction, under the fixed
fp32/bf16 residual and trajectory gates in `design.md`; all other cells retain
the exact decoder-Gram implementation. The implementation identity and its
residual thresholds are project engineering decisions and are content-bound in
the cell, checkpoint, run binding, and deployable codec.

Factorized arms separately derive `direct_rank_space_sparse_topk_cuda_v3`;
rank-one/two map- and decoder-nuclear cells derive the composite engineering
identity `direct_rank_space_sparse_topk_cuda_factor_regularizers_v4`. Their
factor-Gram contraction is the same scientific nuclear objective and is not an
FMX or SASA procedure claim. The materialized site tensor is retained only
under the explicit
`materialized_prepacked_core_reference_v2` oracle identity. Direct rank-space
contraction is a project execution optimization, not an FMX procedure claim or
a new experimental arm. The cell, checkpoint, run binding, and deployable
codec bind this identity. The physical encoder and decoder cores are contiguous
in their respective GEMM orders; stale v1/v2 layouts refuse rather than
migrate. Its low-density bf16 hard-TopK CUDA decoder is a bounded engineering
specialization, not a change to the factorization hypothesis.
Full unfactorized map-nuclear cells bind the separate
`batched_site_gram_reference_guard_d1e-3_e1e-4_v1` engineering identity. It
changes only the fp32 site-summation schedule of the same SASA objective and
falls back wholesale around low-rank or ill-conditioned boundaries; the
former site-reduced einsum remains its oracle rather than a scientific
comparator.
Estimator v14 conservatively retains the prior operational compute and
workspace price.

Deployment evaluation binds the engineering identity
`fused_deployable_full_view_packet_v2`: selector/shared-code endpoints,
transformed-space packet diagnostics, and paired raw-space selection endpoints
consume one paired input traversal and one deployable full-view encode while
the public codec result, raw evidence contract, and independent source-free
packet validation remain unchanged. The deployable carrier canonically
resolves formerly distinct factorized, mapped-score, and persisted-view full
paths; this is an execution-identity change, not a paper-method adaptation or
experimental condition. Evaluation schema v2, the campaign gate, and estimator
v16 bind and price the fused implementation.
Executor v6 overlaps one ordered device transfer with the current CUDA batch
and applies the same explicitly closed pipeline to all calibration traversals;
this does not alter the data stream or scientific method.

QR versus symmetric-polar retraction remains a scientific architecture choice.
Canonical QR cells derive `cholesky_qr1_positive_diagonal_cond64_v1`, polar
cells derive `symmetric_polar_site_bmm_guard_g1024_w8192_c512_f2_r1e-4_v2`, and other carriers
derive `not_applicable_v1`. The former polar einsum and positive-diagonal
Householder QR are admitted reference/test oracles, not matrix rows or tuning
choices. Root, smoke, and child cells rederive the identity, serialized
artifacts bind it, and unknown or
carrier-incompatible values fail closed. These identities and the Cholesky
condition/residual gates are project engineering decisions, not paper recipes.

## 7. Fairness and compatibility

Every headline comparison reports:

1. nominal block-event and active-coordinate matches;
2. actual fixed-width packet rate and amplitude rate;
3. exact deployable-codec side information;
4. parameter, forward/training FLOP, peak-VRAM, and peak-host-RAM estimates;
5. identical raw rows, split, and token presentations;
6. common optimizer-token checkpoints on development only;
7. all method-native endpoints and common endpoints.

Forbidden silent hybrids include a decoder-only nuclear norm called SASA, a
raw unweighted BatchTopK selector called the decoder-weighted carrier, a
concatenated decoder norm called Anthropic's L1-of-site-norms, an oracle
LayerNorm inverse called deployable, or a vector block presented as an exact
scalar-crosscoder implementation.

Smoke reductions preserve each full cell's resolved promotable intent, while
`runtime.smoke` independently makes scientific promotion impossible. A
uniformly smoke stage may emit only a `smoke_protocol_only` selection to test
the state machine; it does not consume scientific outcomes or sharing guards,
and cannot promote a cell. Smoke selections may feed only subsequent smoke
stages; a uniformly smoke Phase-2 campaign may freeze a protocol panel for
smoke Phase 3, but it cannot feed non-smoke scientific Phase 3.

## 8. Implemented derived-mechanism roles

Derived mechanisms are not a standing grid multiplied across paper anchors.
Four families are executable, with deliberately different phase roles:

1. FMX-inspired site-axis rank runs as a Phase-1 fixed-carrier capability
   panel and a Phase-2 parsimony contest over an exact parent, full free-site
   carrier, and `R={1,2,4}`, followed by a conditional post-mask rank revisit.
2. Clean-target observation masking runs as a Phase-1 capability panel with a
   fixed zero-mask carrier, then as a Phase-2 contest over
   `p={0,.02,.05,.10}`, exactly-one-hidden, and exactly-one-retained draws.
3. Availability-rescaled missing-site fusion is validated against literal sum
   in Phase 1 and then transferred as a universal semantic; Phase 2 does not
   retune it.
4. Gauge-aware support scoring compares code norm, decoded energy
   `sqrt(z_g^T (sum_s D_g^s D_g^{sT}) z_g)`, and signed isolated loss decrease
   in Phase 1 on both the Stiefel equality control and a common free decoder.
   Decoded energy on Stiefel is the fixed provisional carrier; Phase 2 runs the
   full three-score by two-hard-selector interaction on real evidence.

The Phase-1 stages are `site_factorization_identification`,
`site_mask_fusion_control_identification`, `site_masking_identification`, and
`selection_score_identification`. Factorization, masking, and score reappear as
Phase-2 tuning stages; rank is revisited after masking, followed by the full
hard score-selector interaction and the separate group-threshold bundled-method
round. The
model, trainer, saved codec, masked evaluation, exact resume, transfer object,
and blueprint tests bind this boundary. Other frontier ideas remain outside
the executable matrix until they receive the same complete contract.

Gauge-invariant partial-view coordinate concordance is now mandatory admission
evidence, not a future live cell. Its decoder-Gram Lin concordance, support
intersection recall, and decoded-energy coverage separate support drift from
coordinate drift for both site-only and leave-one-out inference. A
fixed-support restricted least-squares refit gap remains the highest-priority
contingent diagnostic; it opens a new inference round only after at least `.05`
aligned-code R2, `.01` FVU, or a method-order reversal.
Pattern-specific threshold calibration and a fixed effective-rank codec remain
contingent on reproducible missingness-rate drift and decoder-Gram anisotropy,
respectively, with every new basis/rank byte priced.

## 9. Adversarial non-paper design-space triage

No item below is source-exact for a same-model signed block crosscoder. An
`adapted` label means the nearest paper supplies a mechanism, while the block,
multi-site, or packet-budget transfer remains untried. A `novel` label means
even the mechanism-level hypothesis is local. The ranking is scientific
priority if its trigger fires, not permission to tune on confirmation.

| Rank | Candidate and lineage | Untried hypothesis, confounds, and minimal controls | Phase-1 discriminator | Phase-2 endpoint and burden | Decision |
|---:|---|---|---|---|---|
| 1 | input-adaptive block count; **adapted** from SoftSAE | predict per-token `k`, train through soft selection, deploy hard variable-count blocks; compare token-TopK, BatchTopK, and learned threshold under the same mean total packet bits, because matching mean blocks is unfair across widths; guard soft-weight information hiding and soft/hard mismatch | on mixed 2/4/8-factor examples, require better count calibration and worst-rank support/coordinate recovery without split/merge regression | exact fixed-rate FVU and sharing guards; active-count field and every selector byte priced; high burden because the source selector/MLP substantially increases training cost | highest-priority contingent round only if fixed-count robustness fails; not live now |
| 2 | separate block gate and signed coordinate head; **adapted** from Gated SAE | let a scalar head choose support and a vector head estimate coordinates; exact-parent and parameter-matched encoder controls are required because the extra head may buy reconstruction rather than better support | support precision/recall, same-block aligned-code R2, and gate/contribution disagreement on amplitude/SNR stresses | fixed-rate FVU at matched events/packets plus parameter/FLOP reporting; roughly one extra encoder head | contingent on support errors with already-good subspaces; not live now |
| 3 | heterogeneous block widths or sparse-within-block coordinates; **novel** | allocate different intrinsic dimensions in one dictionary; uniform-width winners, equal total coordinates, equal active coordinates, and exact rate are all controls; ragged amplitudes and coordinate masks must be serialized | mixed planted rank with recovery reported separately for rank 1/2/4/8, including wrong-rank splitting | fixed-rate FVU with variable packet lengths; high model/codec implementation burden | scientifically strong but reserve until uniform-width Phase 1 establishes the failure |
| 4 | all-site versus masked-site code consistency; **novel**, adjacent to FMX masking | add a detached full-view teacher/code-agreement loss after masking; zero coefficient and masking-only parent isolate the loss; guard collapse and true layer-local features | site-only aligned-code R2 and factor recovery on shared versus one-site-span controls | site-only/leave-one-out FVU and support IoU under the existing sharing guard; about two encoder passes | contingent only if masking improves reconstruction while code drift remains |
| 5 | block-subspace incoherence; **adapted** from OrtSAE | replace scalar decoder cosine with a principal-angle/projector-overlap penalty; coefficient zero and a short ladder; correlated-truth controls prevent orthogonalizing real cofeatures | duplicate/mixing fractions and recovery while independently varying true factor-subspace overlap | fixed-rate FVU plus block-overlap and feature-duplication diagnostics; stochastic pair sampling adds modest cost | contingent on observed duplication/absorption, not prophylactic |
| 6 | quantization/entropy-aware fine-tuning; **adapted** from learned compression | optimize a rate-distortion surrogate without changing the selected topology; exact post-hoc-codec parent and a frozen small multiplier ladder; price learned entropy models and reject oracle relaxation gains | not an identification selector; at most verify truth recovery is retained after fine-tuning | actual integer-packet frontier, not surrogate rate; moderate post-selection cost | finalist-only appendix candidate, never an upstream selection axis |
| 7 | nested block prefixes; **adapted** from Matryoshka SAE | one ordered dictionary serves several widths/capacities; compare every prefix with an independently trained equal-size model; prefix order, loss weights, and dictionary ID coding are confounds | broad/specific planted factor recovery at each prefix and feature-continuity mapping across prefixes | every prefix's exact packet frontier; multiple reconstructions increase training cost | broad-rate appendix candidate after one stable family wins |
| 8 | cosine/hybrid block score; **adapted** from cosine-scored SAE | define a gauge-aware directional score and compare with code norm, decoded energy, and isolated loss decrease; pure cosine is not presumed adequate | admit only if support selection remains correlated with irrelevant input/code norm on scale stresses | matched-event and matched-packet FVU plus support-tail diagnostics; low-to-moderate cost | contingent diagnostic round; the two gauge-aware scores answer nearer failures first |
| 9 | invariant site-profile smoothness/contiguity; **novel** | penalize variation of decoder-energy or Gram profiles, not raw gauge-dependent weights; compare zero/short ladder and step-change truth | smooth, abrupt, and reappearing factor site maps with identical support | sharing endpoints and fixed-rate FVU; low compute but weakly identified with only four hooks | reserve for a denser hook panel after factorization results |
| 10 | active-block/capacity curriculum or data-derived subspace initialization; **novel engineering/scientific** | each can improve optimization while silently changing effective rate or pre-solving geometry; compare exact fixed parent at equal tokens/compute and use fit/train data only | learning curves plus final recovery and seed variance; convergence speed alone cannot win | final packet frontier and frozen score, with schedule/initializer artifact bound; modest-to-medium burden | trigger only on reproducible optimization instability, never sweep speculatively |

Three broader proposals are useful falsifiers but are not members of the BSC
hyperparameter space:

- A dense low-rank scaffold beside the sparse residual is an **adapted
  alternative ontology**. Open it only after persistent dense-latent or
  residual-rank evidence, with rank, gradient-stop/co-adaptation, and complete
  dense-channel rate controls.
- SpaDE, mixtures of factor analyzers/experts, and bilinear autoencoders are
  **alternative local or nonlinear ontologies**. They require their own
  truth-association and codec contract after the linear BSC fails the frozen
  nonlinear DGP.
- Procrustes prealignment is an **adapted diagnostic**, not a primary learner.
  An alignment map fit on paired activations can perform the very cross-layer
  identification under study; any diagnostic must use calibration-only fits,
  an identity control, and fully priced dense maps.

The audit also rejects several superficially cheap axes. A learned scalar site
weight is non-identifiable with the free untied site encoders; rare-token or
high-norm oversampling changes the target distribution without truth labels;
and causal/downstream losses change the estimand from activation
factorization. Additional normalization choices are unnecessary until the
already frozen identity/RMS/`sqrt_d`/whitening/LayerNorm confirmation resolves
the scale question. Arbitrary optimizer and schedule grids are likewise not a
scientific substitute for a diagnosed failure.

Thus the executable campaign remains bounded: four implemented mechanism
families have declared phase roles, ranks 1–10 are preregistered intake with
explicit evidence triggers, and
the ontology falsifiers remain separate studies. A trigger can open one
selected-parent round on development evidence; none can inspect confirmation
or multiply across every source family.
