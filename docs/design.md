# Block-crosscoder experimental design

*Normative scientific and execution contract, 2026-07-20.*

## 1. Question and estimand

The project asks whether a single sparse block support carrying signed vector
coordinates can identify and efficiently encode coherent multidimensional
factors across layers of one language model.

For aligned activations $x=(x^1,\ldots,x^S)$, with
$x^s\in\mathbb R^{d_s}$, the native block crosscoder computes

\[
u_g=\sum_s E_g^s x^s+a_g,
\qquad z_g=m_g u_g,
\qquad \hat x^s=c^s+\sum_g D_g^{s\top}z_g .
\]

Here $g\in\{1,\ldots,G\}$ indexes blocks, $u_g,z_g\in\mathbb R^b$, and
$m_g\in\{0,1\}$ selects an entire block. One support event therefore carries
both the claim that a factor is present and a signed coordinate inside its
learned $b$-dimensional chart.

The primary estimands are:

1. **synthetic identification:** whether the same selected block recovers a
   planted factor's support, subspace, and coordinate;
2. **real compression:** raw-activation distortion at a fixed, operational
   total rate;
3. **operational sharing:** whether a code inferred from a site subset predicts
   the held-out sites and remains stable under leave-one-site-out encoding;
4. **manifold quality:** whether a factor is captured once rather than tiled,
   shattered, diluted, duplicated, or mixed with other factors.

Aggregate FVU is not an identification metric. Decoder norm is not proof of
site specificity. The nominal decoder width is capacity, not used dimension.

## 2. Scope

The experiment is deliberately same-model and cross-layer. The load-bearing
parents are [BSF](https://arxiv.org/abs/2606.25234) and
[SASA](https://arxiv.org/abs/2606.06333) for signed vector blocks, and
[Anthropic's original crosscoder](https://transformer-circuits.pub/2024/crosscoders/index.html)
for one code inferred jointly from several layers. The
[BatchTopK](https://arxiv.org/abs/2412.06410) selector and the
decoder-weighted selector and residual-Aux mechanics analyzed by
[Minder et al.](https://arxiv.org/abs/2504.02922) are adapted only as generic
multi-site mechanisms.

[fmxcoders: Factorized Masked Crosscoders for Cross-Layer Feature Discovery](https://arxiv.org/abs/2605.09438)
is also in scope as the closest frontier treatment of scalable same-model
cross-layer dictionaries. Its
full three-mode factorization motivates our narrower, adapted site-axis-only
factorization; its stochastic observation masking supplies a separate
hypothesis. Both follow the staged derived-candidate contract in Section 12.

Methods whose scientific object is allocating shared and exclusive features
between different models are outside the executable design. The project does
not reproduce that task or interpret layerwise decoder norms as
model-difference evidence.

No reviewed source publishes the complete signed vector-block crosscoder above.
The synthesis is novel; source faithfulness applies to its components, not to
the combined object.

## 3. Provenance and claim contract

Every resolved choice has exactly one lineage:

| Lineage | Meaning | Required record |
|---|---|---|
| `exact` | A disclosed equation or numeric setting used in the same source setting. | Primary citation precise enough to locate it. |
| `adapted` | A disclosed mechanism transferred across data, model, geometry, scale, or budget. | Citation, transfer rationale, and named ablation. |
| `engineering` | Execution machinery intended not to alter the scientific object. | Rationale and an invariant/parity test; an ablation if a claim could depend on it. |
| `novel` | A project hypothesis not disclosed by a source. | Mechanistic rationale and a falsifying ablation. |

An omitted source value is `undisclosed`, not a conventional default. Paper
prose, paper tables, released code, and local adaptations remain distinct
recipes. A paper bridge may establish architecture or equations even when
missing training settings prevent a numerical reproduction.

Evidence claims have four levels:

1. an implementation invariant or equation test;
2. a source bridge in a fully disclosed or explicitly adapted setting;
3. a controlled one/few-factor comparison on development evidence;
4. a frozen confirmatory result on a complete seed panel.

A report must state its level. No lower level is silently promoted to a higher
one.

## 4. Representation families

### 4.1 Paper-parent block methods

- **BSF Vanilla:** untied affine signed encoder, token block-TopK, and
  per-block Frobenius-ball decoder projection.
- **BSF Grassmannian:** decoder-tied encoder with one positive global scale,
  token block-TopK, Stiefel decoder blocks, and periodic QR retraction.
- **BSF Group Lasso:** affine encoder, group soft threshold, a scale-controlled
  decoder, and conditionally applied group $L_{2,1}$ penalty.
- **SASA:** free signed block encoder/decoder, token Top-s, the exact
  \(\sum_g\lVert D_gE_g\rVert_*\) map penalty, and whole-group frequency-dead
  residual re-encoding.

Any recipe that declares encoder-scale fitting uses the content-bound
`global_fp64_mean_postactivation_block_norm` statistic, never the selector
score. The executor replays the exact fit prefix under
`positive_bracketed_bisection_remeasure_v1`, targets mean norm `1.0`, and
refuses unless a final remeasurement is within `1e-3` in at most 32
evaluations. This is required for Group Lasso because soft thresholding breaks
the one-shot linear rescaling identity; signed isolated-loss scores are not
admissible calibration statistics.

The project retains paper and inspected-release recipes separately where their
objectives or constraints differ. Only materializable recipes enter the
executable matrix.

### 4.2 Same-model scalar multi-site controls

- **Anthropic architecture anchor:** affine site encoders summed before ReLU,
  free site decoders, squared reconstruction, and activation weighted by the
  **sum of the per-site L2 decoder norms**, not the L2 norm of a concatenated
  decoder. Its undisclosed training configuration makes
  it an architecture bridge. The L1 term is computed from the unscaled ReLU
  activation and decoder norms directly; selector scores never enter the
  regularizer, so a decoder-weighted selector cannot square the decoder cost.
  The exact dense-ReLU/L1 training rule is retained as a comparator family, and
  the calibration stage sparsifies it before
  operational packet evaluation. It cannot become the selected sparse
  finalist, but it must receive the same independent Phase-2 calibration as
  every other Phase-3 comparator.
- **Decoder-weighted BatchTopK carrier:** multiply each positive scalar
  activation by the sum of its sitewise decoder norms, allocate a batch-global
  event budget using those scores, and decode the unscaled activations. This is
  an adapted mechanism, not a reproduction of the source task.
- **Token-horizon residual Aux:** declare a latent dead after a fixed number of
  accepted token presentations, use dead-latent decoder-weighted candidates to
  reconstruct the detached residual, and normalize the residual objective as
  declared. This is tested only as an isolated adapted bundle.
- **Scalar signed, scalar ReLU, source-only block, and source-only scalar
  controls:** remove within-block geometry, joint evidence, or both while
  retaining the same data and accounting contract.

Every hard TopK recipe uses one universal engineering tie rule: rank by score
descending and, only at an exact cutoff tie, retain the lowest declared
candidate indices. The candidate index is the block index within a token for
token-TopK, row-major `(block, coordinate)` index within each token for scalar
AuxK, and row-major `(token, block)` index for BatchTopK. Threshold
selectors use strict greater-than and dense selectors retain strictly positive
codes. This is content-bound behavior, not an installed-PyTorch default or a
matrix axis.

### 4.3 Native block-crosscoder hypotheses

The native object separates ten questions:

1. Does common support exist across sites?
2. Are the within-block coordinates themselves common?
3. Should site evidence be summed or averaged?
4. Does the cross-site decoder need a Stiefel gauge, and should retraction use
   QR or the symmetric polar map?
5. Does block BatchTopK help when event counts vary across tokens?
6. Does a source-derived dead-block auxiliary improve identification without
   causing splitting or mixing?
7. What block width, total coordinate capacity, and active-coordinate budget
   best match the factor rank?
8. Can a low-rank factorization of only the site axis retain recovery while
   reducing parameters and imposing useful cross-layer regularity?
9. Under a free decoder gauge, should a block be ranked by raw code norm,
   isolated decoded contribution energy, or its exact isolated decrease in
   observed squared reconstruction error?
10. Does light site masking improve one-site-to-all-site function without
    harming full-site coding, after controlling the scale change induced by
    summing a variable number of available sites?

The staged matrix assigns universal identification and method-contract
questions to Phase 1, while Phase 2 owns choices whose optimum depends on the
model, hooks, activation scale, optimizer, or packet rate. Most rounds change
one or a few factors. Two bounded interactions are explicit: site-axis rank is
revisited after masking, and hard support selection evaluates the full
three-score by two-selector surface.

## 5. Phase 1 — truth-known synthetic identification

### 5.1 Purpose

Phase 1 answers whether the implementation can recover the object it claims to
represent and freezes only universal semantics and capability evidence. It
does not export a synthetic numerical hyperparameter winner into the pilot. It
uses stateless generators, fp32 reference training, independent
structure/train/eval/confirmation seeds, and disjoint calibration,
development, and confirmation identity ranges.

The live contract is deliberately minuscule and identifiable. It plants one
rank-two factor in a two-dimensional site, activates that factor on every row,
and fits exactly one width-two learner block with one active block. Each
site receives a complete orthogonal factor dictionary. The multisite carrier
uses four independently rotated orthogonal dictionaries with one shared signed
coordinate vector, so factor identity is identifiable from sparse support and
there is no width, activity, or capacity slack.

Every cell uses one fixed optimizer recipe, 20,000 unique training rows,
2,000,000 presentations, and four disjoint 5,000-row factor-calibration,
codec-calibration, development, and confirmation roles. Phase 1 contains no
optimizer, width, capacity, score, selector, or regularizer tournament. Those
quantities are model- and scale-dependent and are reopened on real activations
in Phase 2. The synthetic recipe exports only the shared signed-coordinate
ontology and universal execution semantics, never its numeric dimensions or
optimizer values. Multi-feature sparse allocation is deliberately absent:
Phase 1 proves the shared-vector coordinate path, while Phase 2 owns whether
real activations admit a useful sparse allocation.

### 5.2 Initial spine

The materialized prefix has two fixed stages:

1. `single_site_learnability`: the exact matched orthogonal contract at one
   site. All three seeds must pass before the multisite stage opens. This is an
   instrument check, not evidence for a cross-site feature claim.
2. `multisite_learnability`: the same latent support and coordinates rendered
   through four independent orthogonal site dictionaries. It is the sole
   selectable Phase-1 carrier, so selection authenticates a fixed candidate
   rather than choosing among synthetic hyperparameters.

The scientific gate uses qualification outcomes; an integrity-complete failed
positive control does not open the next stage.

### 5.3 Fixed carrier and confirmation

The selected multisite carrier is rerun on the untouched confirmation role.
Two one-delta negative controls accompany it:

| Variant | Required outcome |
|---|---|
| `baseline` | pass the native and saved-codec identification conjunction on every seed |
| `support_only` | fail the conjunction when sites share event presence but draw different coordinates |
| `site_span_one` | fail the shared-feature conjunction when each planted factor exists at only one site |

The negative controls test the claim boundary. They are not alternative models,
and no failed synthetic stress can delete an option from Phase 2.

The score implementations below remain part of the shared engine and Phase-2
real-model matrix; they are no longer synthetic selection arms. The
isolated-loss-decrease score for block contribution
$y_{g,O}=D_{g,O}^{\top}z_g$ on the actually observed sites $O$ is

\[
\Delta_g(O)
=\lVert x_O\rVert_2^2-\lVert x_O-y_{g,O}\rVert_2^2
=2\langle x_O,y_{g,O}\rangle-\lVert y_{g,O}\rVert_2^2.
\]

It is an exact signed score, so harmful blocks retain negative values rather
than being clipped. Hidden clean targets are excluded from both terms even
though they remain reconstruction targets for training. The arm is legal only
for a bias-free quadratic reconstruction objective. It is invariant to an
invertible reciprocal within-block encoder/decoder gauge; for a tied encoder
with an exact concatenated-Stiefel decoder and unit scale, it reduces to
$\lVert z_g\rVert_2^2$ and therefore ranks blocks identically to code norm.
The three Stiefel score arms deliberately retain this algebraic collapse as a
negative/equality control. The parallel free-decoder arms share one identical
untied, unconstrained carrier so their difference is the score alone; they are
capability evidence, never the Phase-1 winner. This makes gauge-aware score
behavior visible without pretending the synthetic generator chose the pilot's
support rule.

The scientific score is unchanged by its serialized
`implementation.isolated_loss_decrease_implementation`.  The reference
`exact_site_gram_quadratic_v1` performs the site-wise projection and direct
three-factor quadratic.  `mapped_free_decoder_quadratic_v1` is admissible only
for isolated-loss-decrease scoring on a free decoder with the same bias-free
mean-squared or squared-L2 carrier.  It never activates from device, shape, or
ambient runtime state, and an ineligible explicit declaration is refused.

The mapped implementation forms the linear term with one flattened
decoder-transpose GEMM.  It maps the code through the block Gram with BMM and
takes the coordinate dot product rather than asking a three-input einsum to
choose a contraction.  When every site is observed, it contracts the decoder
directly to one all-site Gram and performs one map.  Partial and source-only
views retain one mapped quadratic per site and weight it by the exact observed-
site mask.  Padding is applied before the projection, hidden clean targets are
excluded from both terms, negative gains are retained, and neither route
materializes `[batch, groups, sites, d_model]`.  The materialized factorized
reference uses the same path and retains gradients to both site and core
factors.  The canonical factorized implementation below instead evaluates the
quadratic directly in its rank carrier.

Release fixtures compare reference and mapped implementations from identical
state across all, partial, source-only, padded, factorized, exact-threshold, and
signed-streaming paths.  In fp32, score relative L2 drift is at most `2e-6`,
mask-element disagreement at most `1e-6`, output and maximum parameter-gradient
relative drift at most `2e-4`, and loss relative drift at most `2e-6`.  In
bf16, score relative L2 drift is at most `2e-3`, mask disagreement at most
`1e-3`, support IoU at least `.99`, output relative drift at most `.05`, loss
relative drift at most `1e-4`, and maximum parameter-gradient relative drift at
most `.06`.  Exact contribution, harmful-negative, observed-site, frozen-
geometry binding, and configuration-refusal fixtures remain hard gates.

On the retired 2026-07-22 benchmark shape (`B=8192`, four width-128 sites, 256 groups,
block width four, four active blocks, fp32), 31 post-warmup CUDA samples of the
complete forward, squared-L2 loss, and backward measure `8.218 ms` for the
reference and `2.476 ms` for the mapped implementation, a `69.9%` reduction.
Peak CUDA allocation falls from `407,346,176` to `277,142,528` bytes
(`124.2 MiB`).  The paired gate records score relative L2 drift `5.90e-7` and
exact support, reconstruction, loss, and model gradients.  These are kernel
evidence, not a scientific score change; full-Trainer planning remains bound
to the separately versioned resource estimator.

Calibration uses a deterministic signed log histogram with an explicit zero
boundary. The implementation computes one decoder-transpose-like projection
and block Gram terms without materializing a
`[batch, groups, sites, d_model]` contribution tensor.

Scientific Phase 1 authorizes the pilot only when the baseline passes the full
identification conjunction on every seed **and** both preregistered negative
controls—`support_only` and `site_span_one`—fail that conjunction on every
seed.

With seeds 0, 1, and 2, the blueprint declares and executes **15 cells**: three
single-site instrument cells, three multisite truth-contract cells, and nine
confirmation cells. All architecture, capacity, optimizer, score, selector,
regularizer, masking, and rate choices are owned by Phase 2.

### 5.4 Identification metric

For each planted factor $f$, support association chooses a learned group
$g(f)$. Every factor-level geometry score is then computed through that same
group; a method cannot receive support credit from one group and subspace credit
from another.

The native training rule and the reloaded saved-codec deployment rule must both
pass the frozen conjunction:

| Component | Threshold |
|---|---:|
| per-factor support-association F1 | at least `0.50` |
| same-group subspace overlap, when eligible | at least `0.50` |
| full-model isolated-input reconstruction R2 guard | at least `0.50` |
| same-group aligned-code R2 | at least `0.50` |
| fraction of factors satisfying the per-factor conjunction | at least `0.80` |
| aggregate support precision | at least `0.75` |
| aggregate support recall | at least `0.75` |
| duplicated/split-factor fraction | at most `0.10` |
| fraction of planted factors participating in a merged learned group | at most `0.20` |
| nonfinite values | exactly `0` |

The global isolated-input R2 is a reconstruction guardrail; it is not called
matched-block recovery. The decisive geometry evidence is support association,
same-group subspace overlap, and same-group aligned-code R2.
Aligned-code R2 uses the actual post-selection sparse code on isolated-factor
inputs. Unselected rows remain zero in the regression and selected-group
coverage is reported, so pre-selector decodability or conditioning on successful
selections cannot substitute for an operational code.

The inactive dictionary fraction is reported but is not a promotion gate or a
ranking margin. Overcomplete capacity makes unused blocks expected—a perfect
one-block-per-factor solution can leave half or more of the dictionary unused—
so rewarding occupancy would favor spurious activations and duplication.
Likewise, the descriptive merged-group fraction is normalized only over
support-associated groups. The qualification gate uses the fraction of planted
factors participating in a merge, so changing unused dictionary capacity
cannot mechanically improve the pathology score.

The raw same-block support, subspace, and aligned-code endpoints remain the
Phase-1 headline and gate even when learner block width is smaller than the
planted factor rank. Each factor reports the linear-information ceiling
`min(1, block_width / truth_rank)`; the threshold is not divided by that
ceiling and scalar controls therefore retain an honest rank-mismatch failure.
A nonpromotable precision companion freezes exactly
`ceil(truth_rank / block_width)` groups per factor from calibration association,
then reports held-out union-support confusion, concatenated decoder-subspace
overlap, aligned-code R2, and selection coverage. It diagnoses whether failure
is allocation across several scalar groups without replacing the raw gate.

Matching pathologies use primary association cutoffs `strong=.50` and
`weak=.25`. The complete reporting-only grid `strong in {.40,.50,.60}` by
`weak in {.20,.25,.30}` is content-bound and emitted without retuning or
changing the primary pass. No cutoff remains an executor constant outside the
cell contract.

Each threshold is converted to a signed margin with separate failing-side
threshold scale and passing-side available headroom to the natural unit bound;
this prevents high-threshold endpoints from saturating earlier merely because
of denominator choice. A cell's
`phase1_identification_margin` is the weakest per-factor or aggregate margin.
Candidate seeds are complete or the candidate is ineligible; ranking is
descending median margin, then descending worst-seed margin, then candidate ID.
All component values and frozen margins are reported even though the ranker is
frozen.

### 5.5 Phase-1 transfer boundary

Freezing Phase 1 produces both an authorization decision and a
`bsc-phase1-transfer-v3` object. The transfer is derived again from the complete
campaign manifest rather than copied from a winner file. It binds the source
plan and blueprint IDs, manifest hash, seed-complete baseline candidate and
cell IDs, every selection ID, confirmation scope narrowing, and two distinct
scientific payloads:

- `method_contract`: the exact universal decisions for threshold estimator and
  source, availability-rescaled encoder fusion, deterministic tie breaking,
  and quadratic reconstruction, plus its own hash;
- `provisional_carriers`: signed coordinate activation and the decoded-energy score
  used to carry the synthetic method into the pilot, plus their own hash and an
  explicit `phase2_reopened_decisions=[model.activation,
  model.selection_score]` declaration. The universal invariant is a signed
  real coordinate vector, not the literal activation operator: group soft
  thresholding also emits signed coordinates and is therefore a legal Phase-2
  method bundle;
- `capability_evidence`: every seed-level qualification digest and scientific
  pass/fail outcome from the seven capability panels, including which fixed
  carrier advanced. Its role is explicitly `diagnostic_only_no_phase2_pruning`.

The transfer's method semantics are explicitly
`shared_signed_coordinate_vector`, `clean_all_sites` targets under masked
encoding, and
`universal_semantics_and_provisional_carriers_then_real_model_tuning`.
Synthetic numeric winners for width, rank, mask probability, retraction,
score, selector, optimizer, or regularizer are absent by construction. A
failed confirmation stress must
appear in the content-bound claim-scope narrowing rather than being averaged
away.

Phase 2 can be previewed without evidence for shape and resource estimation,
but a runnable blueprint must be rebuilt from the complete authenticated
Phase-1 decision. It binds both `source_phase1_decision_id` and
`phase1_transfer_id`, and embeds those IDs in the `phase1_contract_bsc` root.
Capability outcomes remain attached evidence but never prune the pilot: full
recovery under one synthetic DGP, optimizer, and representability regime is not
an implementation-admissibility predicate for GPT-2. Any changed decision,
transfer, qualification digest, scope, plan, or blueprint changes the content
ID or is rejected as stale or forged.

## 6. Phase 2 — GPT-2 Small staged pilot

### 6.1 Data contract

Phase 2 captures one immutable raw stream from pinned
`openai-community/gpt2` residual-pre hooks at blocks 3, 5, 7, and 9 on pinned
OpenWebText. Every primary candidate consumes this same four-layer task; site
count is fixed rather than mixed into architecture selection, and any later
site-count robustness claim must be a separate declared panel. Context length
is 128, BOS position 0 is dropped, and every row has stable
`(sequence, position, token_id)` identity.

Capture is single-model-only. Before output creation, the implementation
resolves the model and corpus to immutable commits, loads the reviewed slow
tokenizer at the model commit, verifies its exact class, BOS ID, Unicode-canonical
vocabulary hash, and ordered tokenizer-file hash, and passes that tokenizer
explicitly to TransformerLens. The capture artifact separately binds the exact
source contract, ordered whole-sequence allocation, capture/store source-code
hashes, Python and dependency versions, site dimensions, and shard geometry.
The final manifest embeds that complete capture binding and its canonical
SHA-256; consumers recompute the digest, require the exact binding field set,
check every duplicated source/allocation/implementation field, and match the
producer implementation against the currently reviewed capture code. A
well-shaped digest string is not authentication. Scientific capture additionally
requires a recorded CUDA device request.
Capture-manifest v2, capture-binding v2, derived-view v2, and transform v2 are
the only accepted prelaunch shapes. Their content identities use typed,
length-prefixed streaming fields and bind each split's manifest bytes, physical
file stream, logical content stream, row-identity stream, geometry, and exact
allocation. A campaign pins one raw-content digest plus one digest per named
view; standalone split directories without the authenticated root envelope are
neither executable inputs nor storage-preflight credit.
Physical store schema v3 requires `int64` row identities in the manifest,
every shard record, metadata header, and tensor payload. It records an
incomplete self-hashed manifest after each durable shard; resume reconstructs
the prefix only from reverified shards and refuses any changed binding.
Shard persistence is a fixed one-deep pipeline: the producer may fill one CPU
staging shard while one detached shard is owned by a single persistence worker.
Before any bytes are written, that worker's first operation performs the
finite/zero-row integrity audit; detection may therefore be delayed until the
next submit or explicit synchronization barrier, but invalid payloads are never
published. The worker returns immutable shard evidence and never mutates live
writer state. The producer alone installs ordered-stream hashes, atomically
writes and directory-fsyncs the incomplete manifest, then advances capture
progress through its durable-progress callback. A crash can consequently leave
at most the one exact next canonical shard orphan already admitted by verified
resume. Capture and materialized-view derivation refuse before output creation
when two physical shard payloads (bf16 padded activations plus int64 row IDs)
exceed the configured writer-residency ceiling; shard geometry and that exact
estimate are bound into capture state.
The content-bound Phase-2 name is
`activation-store-v3-derived-views`; Phase 3 uses the distinct
`activation-store-v3-single-view` contract. No v2 alias is accepted.
Transform-only artifacts use the same durable publication discipline as store
manifests: temporary write, file fsync, atomic replacement, and parent-directory
fsync.

The raw capture contains whole-sequence-disjoint roles:

| Role | Requested rows | Permitted use |
|---|---:|---|
| normalization fit | 250,000 | fit declared activation transforms and encoder scale only |
| calibration | 250,000 | fit inference threshold, clipping, rotations, and quantizers only |
| development | 1,000,000 | staged selection endpoints |
| confirmation | 1,000,000 | final pilot confirmation only |
| train | 16,000,000 | prefix-nested optimizer data |

The initial gauge is per-site centered scalar RMS. `none`, `sqrt_d`, shrinkage
`whiten`, and token `layer` are frozen confirmation arms. All views derive from
the same raw row stream. Token LayerNorm requires per-token inverse statistics;
unless those statistics are transmitted and priced, its raw inverse is oracle
and the cell cannot be a deployable finalist.

The Phase-2 runner receives the parent view directory and dispatches each cell
to the `<normalization>` child named by its immutable manifest. Before changing
campaign state it verifies the frozen transform, complete self-hashed split
manifests, exact declared split set, and cross-view row-stream identity. The
`none` transform is currently a materialized derived view: a raw identity alias
would change the uniform derived-view contract and is not implemented or
credited implicitly.

### 6.2 Conditional pilot

The initial `anchors_1m` stage runs BSF Vanilla, BSF Grassmannian, BSF Group
Lasso, SASA, the Anthropic dense-L1 comparator root, the adapted
decoder-weighted BatchTopK carrier, the scalar ReLU BatchTopK control, the
evidence-bound `phase1_contract_bsc`, and
`phase1_contract_source_only_control`. The latter differs only by inference
from site 0 and is explicitly nonpromotable. It supplies the real source-only
descriptive control without opening a topology-tuning round. It is reported,
but it is not the later same-model partial-view guard and is not a matched-token
selection gate. Only `phase1_contract_bsc` is
eligible for the main-chain anchor selection; the other roots calibrate the
comparison surface. The similarly named unbound preview recipe is usable only
for count/resource inspection and cannot register a scientific campaign.

Subsequent selected-parent rounds are:

| Stage | Optimizer tokens | Live choices |
|---|---:|---|
| `architecture_4m` | 4M | exact selected-parent rerun; labeled parent-architecture carrier; no initialization preconditioning; tied Grassmann width 4 with QR; tied Grassmann width 4 with polar retraction |
| `capacity_4m` | 4M | exact selected parent; widths `1,2,4,8` at fixed 8,192 total/32 active coordinates; width-4 half/double capacity; width-4 half/double activity |
| `site_factorization_4m` | 4M | exact selected-parent carrier; unfactorized full free-site weights; factorized site ranks `1,2,4` |
| `site_masking_4m` | 4M | exact selected parent; Bernoulli clean-target masking `0,.02,.05,.10`; exactly one hidden; exactly one retained |
| `site_factorization_revisit_4m` | 4M | exact selected masked parent; unfactorized full free-site weights; factorized site ranks `1,2,4`; if zero Bernoulli masking wins, materialize only the exact parent because no rank–mask interaction exists |
| `hard_selector_score_interaction_4m` | 4M | full Cartesian surface of code norm, exact isolated decoded energy, or exact isolated loss decrease by signed token-TopK or signed block-BatchTopK; decoded-energy/token-TopK is the exact incoming control |
| `group_threshold_method_4m` | 4M | exact selected parent; complete affine group-soft-threshold/unit-Frobenius/conditional-L2,1 method bundles at coefficients `3e-4`, `1e-3`, or `3e-3` |
| `learning_rate_4m` | 4M | exact selected parent; peak learning rates `3e-5`, `1e-4`, `3e-4` |
| `batch_size_4m` | 4M | exact selected parent; 2,048, 4,096, or 8,192 optimizer tokens per batch |
| `warmup_4m` | 4M | accepted-update warmup fractions `.02`, selected-parent `.05`, `.10` |
| `schedule_4m` | 4M | exact selected parent; constant after selected warmup; cosine; final-fifth linear decay |
| `learning_rate_revisit_4m` | 4M | exact selected parent; revisit peak rates `3e-5`, `1e-4`, `3e-4` after batch/warmup/schedule selection |
| `regularization_16m` | 16M | exact selected parent; no regularizer/Aux; exact SASA map nuclear at initial penalty/reconstruction ratios `.01/.03/.10`; decoder-only nuclear diagnostics at absolute `30/100/300` |
| `auxiliary_16m` | 16M | exact selected parent; no Aux; BSF runner-up Aux; SASA source, low-weight, or long-window dead-group Aux |
| `confirmation_16m` | 16M | scalar RMS, none, `sqrt_d`, shrinkage whitening, token LayerNorm |

Phase 2 uses each cell's declared bf16 forward precision. It has no matrix-level
fp32/bf16 parity cell: the executable parity-and-short-run stability preflight
is deliberately Phase-3-only.

There are deliberately no Phase-2 observation-site/evidence-topology or
missing-site-fusion tuning rounds: the four hooks and their availability
semantics stay fixed. Model architecture is explicitly retuned in
`architecture_4m`, and the complete group-threshold method later changes the
encoder/decoder/activation bundle. Availability-rescaled fusion is inherited
as a universal semantic.
Mask probability and site-axis rank are Phase-2 questions because their optimum
depends on the actual hook distribution, capacity, and rate. The initial rank
round establishes a carrier; after masking, `site_factorization_revisit_4m`
tests the bounded rank–mask interaction. It is conditionally vacuous when the
selected mask is Bernoulli zero, so only the exact parent is then emitted.
Block score is also reopened: decoded energy is only the provisional incoming
carrier, and `hard_selector_score_interaction_4m` evaluates all six hard
score-selector pairs on real four-hook fixed-rate evidence. Learned group
thresholding changes the encoder bias, decoder constraint, activation,
selector, L2,1 objective, and activity schedule together; it is therefore
`group_threshold_method_4m`, a bundled method comparison, not a third selector
level in that interaction.
Learned group-threshold training support is exactly the nonzero support of the
post-shrinkage code. The inherited endpoint score is used only for calibrated
deployment ranking; it cannot add an undeclared `score > 0` training gate.

The BSF runner-up auxiliary uses the encoder carrier before fixed per-token
hard selection. For signed token-TopK cells that is the ordinary pre-selection
code. Every token must have the complete declared runner-up count; an
under-capacity row fails rather than silently changing the auxiliary width or
its fixed `1/l` coefficient. Learned Group-Lasso support has no guaranteed
unselected runner-up set—the initial threshold can activate every block—and
BatchTopK does not guarantee per-row residual capacity. Appendix D defines
neither interaction. The executable matrix therefore keeps Group Lasso
primary-only, runs paper Appendix-Aux only on the Vanilla and Grassmannian
token-TopK carriers, and conditionally elides the Phase-2 main-chain runner-up
arm unless fixed token-TopK is the selected parent.

The real architecture round retains only the narrow transfer that synthetic
evidence cannot settle: whether untied inference or a tied Stiefel carrier is
preferable at GPT-2 width, and, conditional on the tied carrier, whether QR or
symmetric-polar retraction behaves better under real conditioning and the
declared numeric regime. Phase 1 measured both retractions and advanced QR by
declaration; it did not rank them as a transferable hyperparameter winner.
The registered Phase-2 blueprint retains every declared architecture,
retraction, capacity, factorization, masking, score, and selector option and
owns their real-model ranking. Synthetic failures remain warnings and
claim-scope evidence, not filters.

Every ordinary main-chain Phase-2 development round declares an inert exact
parent control. Before execution, the materializer resolves every decision and
retains one representative of each execution-value signature; redundant
parent/center labels are recorded in `elided_execution_duplicates` rather than
trained twice. A child replaces the retained parent only when the fixed-rate
selection score improves by at least `0.002` on every seed and on both the
median and worst-seed aggregates; otherwise the parent advances. The initial
site-factorization round uses a separate two-part parsimony rule: the
unfactorized full free-site carrier must first remain within `0.01` of the exact
selected parent, then the lowest-capacity member of rank `1`, rank `2`, rank
`4`, and full that remains within `0.01` of full on every seed and on the
median/worst aggregates advances. The post-mask rank revisit uses the ordinary
parent-retention rule because it measures interaction rather than choosing the
globally smallest rank again. The confirmation round has no selection policy
and cannot feed hyperparameter tuning. Panel freeze takes only the untouched
scalar-RMS confirmation rerun. On every seed it must remain scientifically
qualified, re-pass the full sharing guard, and degrade by at most `0.02` in
fixed-rate score relative to its exact development parent. That `0.02` is a
novel preregistered reproducibility tolerance—not a paper value—and each
confirmation cell content-binds the `.01/.02/.05` sensitivity ladder. The
frozen evidence reports every counterfactual pass set plus the result with the
score tolerance removed; confirmation data never retunes the center.

The seven frozen Phase-3 comparators are not copied from these 1M anchors.
Each receives an independent, content-addressed conditional calibration chain
under the same fixed-rate policy. All branches are appended to one authenticated
Phase-2 DAG: every child names its scientific parent stage and exact parent cell
IDs even when another family's stage is most recent in the journal.

| Family | Family-specific development rounds before the top-two revisit |
|---|---|
| shared-coordinate BSC | width; activity; learning rate; schedule |
| BSF Grassmannian | width; activity; learning rate; schedule |
| BSF Group Lasso | width; activity; L2,1 coefficient; learning rate; schedule |
| SASA | width; activity; initial map-penalty ratio `0/.01/.03/.10`; dead-residual Aux; learning rate; schedule |
| Anthropic dense L1 | activity; decoder-weighted L1 coefficient; learning rate; schedule |
| decoder-weighted BatchTopK | activity; learning rate; batch size; schedule |
| scalar ReLU BatchTopK | activity; learning rate; batch size; schedule |

Every family learning-rate round has exactly four peak-rate arms:
`3e-5`, `1e-4`, `2e-4`, and `3e-4`. This is independent family calibration,
not the three-rate main-chain ladder plus its exact-parent control.

Every block family resolves `groups = 8192 // block_width` and
`active_blocks = target_active_coordinates // block_width`; the center is
exactly 8,192 total and 32 active coordinates. Each family then ranks the union
of integrity- and scientific-outcome-complete candidates from all of its 4M
rounds under one content-addressed `retain_count=2` nomination policy. The
union deduplicates the seed-independent resolved execution signature while
preserving every stage/candidate alias and metric spread in the ranked
evidence. Deduplication happens before outcome ranking; the representative is
the earliest declared source round, then candidate ID, so repeated center
configurations receive no best-of-repeats advantage. Comparator
families report the BSC sharing endpoints but do not use them as admission
gates. It freshly reruns the overall development winner and strongest distinct
resolved runner-up for 16M tokens, then selects one family comparator. This revisit
probes local order sensitivity; the staged search does not estimate every
interaction or establish a global optimum. At seeds 0 and 1, the serialized
blueprint declares a pre-elision ceiling of **176 main-chain cells** plus **234
family-calibration cells**, for **410 total**. The 176 comprise 18 anchors and
158 declared cells across the 15 rounds from `architecture_4m` through
`confirmation_16m`. The realized count is lower: execution-equivalent
parent/center variants are deterministically elided, and a zero-Bernoulli-mask
winner conditionally removes the four rank children from the revisit. A
non-token-TopK winner also conditionally removes the two-seed Appendix-D
runner-up arm. All forms of elision are content-addressed in the materialized
stage, and reports must distinguish the declared ceiling from cells actually
executed.

SASA does not disclose a numerical `lambda_dim`, and its absolute scale changes
with dimension and reduction convention. A ratio-calibrated cell therefore
stores absolute coefficient `0` in its manifest, then—after every declared
initializer and encoder-scale fit but before optimizer construction—evaluates
the unweighted map penalty and clean-target reconstruction in fp32 on the
content-bound first training batch. It resolves
`lambda = target_ratio * initial_reconstruction / initial_penalty` and records
the input digest, both raw losses, target, resolved coefficient, and achieved
ratio in the checkpoint run binding and training report. Resume reconstructs
the same binding and refuses any mismatch. Target zero is a real SASA-map arm,
not a switch to another objective. The exact zero-smoothing nuclear norm uses a
compact Cholesky/SVD evaluation so the intentional repeated decoder-Gram
eigenvalues do not create undefined eigenvector gradients. Decoder-only
nuclear norm is globally schema-forced nonpromotable: it is inspected
release-drift evidence, not SASA's paper objective.

For rank-one and rank-two site-factorized cells, the same objectives operate
directly on factor Grams. If `D[s]=sum_r A_D[s,r] C_D[r]`, the decoder row Gram
is `sum_rt (A_D.T A_D)[r,t] C_D[r] C_D[t].T`; the encoder and per-site decoder
Grams use the analogous contraction. This is the identical real-arithmetic
map or decoder nuclear norm without a rounded full bf16 site tensor. It is
serialized separately because casting the factors before contraction changes
bf16 reduction order.

Every adaptive main-chain Phase-2 selected-parent or revisit policy carries
frozen sharing guards. Root-anchor policies are deliberately ungated baselines:
the first sharing admission occurs at the first adaptive main-chain selection,
where both parent- and root-relative evidence exists. Comparator-family
policies continue to report the endpoints without using them for admission.
For both site-only and leave-one-out inference, every
seed must have worst-site decoded-coordinate Lin concordance at least `.80` on
the support intersection. Concordance is computed in the gauge-invariant
all-site decoder-Gram geometry, centers the paired coordinate streams, and adds
their mean-offset energy to the denominator so a partial-view additive offset
relative to the all-view code cannot look like agreement. The worst-site
full-view-support intersection recall and
the fraction of full-view decoded energy retained on that intersection must be
at least `.75` and `.90`, respectively.

The existing safety gates remain conjunctive. Relative to the selected parent,
mean site-only and mean leave-one-out held-out raw FVU may each degrade by at
most `.02`, and each mean support-IoU endpoint may fall by at most `.05`.
Relative to the root anchor, those two FVU means may each degrade by at most
`.02`; their absolute values must each be at most `1.0`. The same-candidate
all-view FVU advantage is still reported, but it is descriptive only: an ideal
redundantly shared factor need not reconstruct better from all sites than from
one sufficient site. This is not a comparison with the separately trained
source-only anchor. Missing or nonfinite guard data fails before
median/worst-seed aggregation. These tolerances are qualification constraints,
not extra ranker weights.

No source paper supplies the project's winner-changing minimum-effect,
noninferiority, or sharing cutoffs. They are **novel preregistered practical and
safety policies**. Every applicable selection-policy content ID binds both that
basis and the complete descriptive sensitivity surface:

- minimum effect: `0/.001/.002/.005`;
- noninferiority: `.005/.01/.02`;
- parent/root partial-view FVU degradation: `.01/.02/.05`;
- support-IoU drop: `.02/.05/.10`;
- coordinate concordance: `.50/.80/.90`;
- support-intersection recall: `.50/.75/.90`;
- decoded-energy coverage: `.75/.90/.95`;
- absolute partial-view FVU: `.75/1.0/1.25`.
- scalar-RMS confirmation score degradation: `.01/.02/.05`, center `.02`.

The preregistered center thresholds determine eligibility. Every scientific
selection artifact executes and serializes the marginal counterfactual pass
set at each threshold above, using the same authenticated measurements and
without reranking or changing the selected candidate. Phase-1
identification and other qualification thresholds are likewise versioned novel
project decisions rather than attributed to a paper.

### 6.3 Operational rate–distortion metric

Every real cell calibrates and saves an actual deployment codec, reloads it,
encodes integer packets, decodes without the source activation, and measures
raw-coordinate reconstruction. The rate includes:

- a fixed-width active-count field;
- compact included-block IDs with the transmitted rank-to-block table;
- `q * block_width * selected_events` amplitude bits for `q` in
  `{2,4,6,8,12,16}`;
- all bytes in the exact deployable codec, amortized over 100,000,000 tokens.

The optimizer checkpoint is not counted as deployment side information. The
zero-event point reconstructs the per-site calibration mean and still pays the
codec side-information rate. Blocks with insufficient calibration events are
excluded by the frozen policy, and the excluded event share is a qualification
guardrail rather than free compression.

The frozen real-cell scientific-outcome guardrails require calibration mean
support within `0.1` block of target and excluded selected-event fractions at
or below `1%` on both calibration and evaluation. Promotion additionally
requires a source-free raw inverse and eligibility at every fixed-rate budget.

The primary budgets are **at most 256, 384, and 512 total bits/token**. For each
candidate:

1. remove rate-distortion points dominated by a lower-rate point;
2. construct the lower convex envelope of the measured zero-event and
   2/4/6/8/12/16-bit points;
3. time-share only between adjacent envelope points, using the
   content-bound `balanced_global_token_counter_u64_v1` schedule so the decoder
   reproduces the mixture without per-token side bits; serialize each selected
   operating point as an exact 32-byte record in the immutable
   `deployment_schedules` bundle, reload and execute those bytes on the paired
   raw evaluation rows, and price the selected record as deployable side
   information rather than use a weighted average of aggregate endpoint FVUs;
   because budgets are
   upper bounds, retain the lower endpoint instead whenever the executed
   schedule does not improve it;
4. declare a budget below the measured envelope ineligible rather than
   extrapolating;
5. above the highest measured rate, use the best measured point within budget,
   never an extrapolated distortion.

`deployment_schedules.bin` is an audit bundle of mutually exclusive operating
points, not a file that a consumer receives wholesale. A deployment at one
fixed budget consists of the exact deployable codec plus that budget's one
32-byte record; the embedded evaluation manifest binds every alternative
record and the bundle hash, while rate accounting charges only the record that
the consumer actually needs.

Development freezes the endpoint identities selected at all three budgets in
one content-addressed operating policy. Scalar-RMS confirmation and Phase 3 do
not rebuild a hull from their own distortions: they replay the policy from the
deterministic worst-scoring source seed, recomputing only the largest integer
mixture that fits the current serialized rates. The 32-byte record is present
and charged even when both endpoints are identical. A missing endpoint,
changed quantizer identity, incomplete budget grid, or mixture that no longer
fits fails rather than selecting a replacement from holdout/final evidence.

The cell score is negative mean raw-space FVU over the three frozen budgets.
Candidates need every seed and a passed scientific outcome. Rank by descending
median score, then descending worst-seed score, then candidate ID. Transformed
FVU, native sparsity, packet frontier, per-site FVU, shared-code matrices,
support tails, runtime, memory, and uncertainty are reported diagnostics, not
substitutes for the primary metric.

Shared-code evaluation also reports a descriptive functional-dependence profile
before and after selection. For each omitted site and block, it measures the
RMS code change, max-normalizes that block's site profile, and sums the profile
to `C` in `[1,S]` when dependence is nonzero (`0` for an invariant/zero block).
Larger `C` means dependence is distributed across more sites; it is not a
monotone quality score. Both sharply local and broadly cross-layer blocks can
be valid, so `C` is never used as the selection direction.

## 7. Phase 3 — frozen publishable panel

Phase 3 consumes a self-contained panel decision emitted from complete Phase-2
evidence. It verifies the Phase-2 blueprint and final plan, the full selection
chain and ranked universe, qualification artifacts for every seed, confirmation
artifacts for the finalist, and the exact source cells. An arbitrary JSON list
of preferred recipes is not a panel decision.

Portability is stronger than checking self-consistent hashes. The envelope
replays the same reducer used by the live campaign over every embedded
qualification, reconstructing eligible and excluded candidates, exclusion
reasons, gates, ordering, sharing lineages, and the serialized threshold-
sensitivity surfaces. Comparator nominations replay the complete cross-round
union and its pre-ranking execution-signature deduplication, including the
canonical representative, aliases, and observed metric spread. Thus neither a
winner moved into an excluded list nor a forged nomination score can steer the
panel merely by rehashing it.

The frozen panel has eight slots:

1. the exact derived Phase-2 finalist;
2. independently calibrated shared-coordinate BSC mechanism comparator;
3. BSF Grassmannian;
4. BSF Group Lasso;
5. SASA;
6. independently calibrated Anthropic dense-L1 comparator;
7. adapted decoder-weighted BatchTopK mechanism comparator;
8. scalar ReLU BatchTopK control.

Every comparator slot binds the derived winner of its Phase-2 family chain,
the complete family selection IDs, its family blueprint ID, and its root
recipe lineage. A 1M root anchor is not admissible as a Phase-3 comparator.
Comparator-family calibration and revisit policies rank qualified cells by the
same fixed-rate raw-FVU metric but do **not** require the BSC sharing-admission
gate. Otherwise a baseline that intentionally lacks shared-coordinate
inference could disappear before comparison. Every comparator still reports
the identical partial-view concordance, support, energy, and FVU endpoints;
those values are Phase-3 outcomes rather than family-admission filters.
Every slot also serializes duplicate handling: the selected-finalist slot
fails closed, while a comparator that duplicates it advances to the next
ranked qualified nonduplicate in that comparator's already frozen universe.
That exception is accepted only when replay reproduces the original collision
and proves that every earlier-ranked alternative also collides. Scientific
configuration projection builds the exact five-seed production plan and then
fingerprints the seed-zero representative of each slot.
Each slot uses seeds 0–4. The model is pinned `google/gemma-3-4b-pt`; the four
ordered residual-pre sites are blocks 8, 14, 20, and 26; the corpus is pinned
FineWeb-Edu; context is 512. One raw bf16 store requests 25M unique training
rows, 250k normalization-fit rows, 250k calibration rows, 250k dedicated
stability rows, and 2M final rows; physical splits round upward to whole packed
sequences. Transform fitting consumes exactly the requested 250k-row
normalization prefix and excludes that rounding surplus.
Every cell receives 100M optimizer-token presentations, scalar-RMS
normalization, 16,384 total latent coordinates, and 128 active scalar
coordinates before achieved-rate matching.

The Phase-3 store uses the same physical schema and resumable capture contract,
but retains only raw bf16 shards. Scalar-RMS is a content-addressed transform
artifact applied and inverted in fp32 at load time; diagonal modes never
accumulate covariance or execute dense site matrices. Transform provenance
binds the capture file and normalization-fit manifest/row/content hashes.

Phase 3 has no ranking, hyperparameter tuning, or recipe substitution. Before
the final split may be read, each of the eight frozen designs runs one
262,144-token `production_stability_preflight` cell at the exact production
site geometry and capacity, using the dedicated `stability` evaluation split.
On the content-bound initial training prefix it compares fp32 and bf16 forward
paths before optimizer construction, requires reconstruction relative error at
most `.05`, support IoU at least `.90`, and finite outputs, then must complete
the declared short optimization and all ordinary qualification gates. All
eight cells must qualify; this conjunctive refusal gate does not rank them and
its thresholds cannot be retuned from observed stability data. A passed smoke
cell can test only the gate protocol, never create scientific evidence.

The 40 frozen final cells then evaluate on the untouched final split at
1,024, 1,536, and 2,048 total bits/token. This is the preregistered exact
fourfold transfer of the Phase-2 `256/384/512` budgets, matching the nominal
active-coordinate ratio `128/32`; reusing the pilot bits would mostly compare
zero-event or saturated endpoints rather than like-for-like sparse payloads.
The preflight requires every scaled budget to use a nonzero packet endpoint
and at least two distinct nonzero frontier endpoints across the three budgets.
Method-valid secondary endpoints remain unchanged. Recovery checkpoints every
5M tokens limit lost work;
they are not an invitation to choose an endpoint after seeing final
performance.

Resource-estimator schema `dense-linear-memory-v20-e8cd28faf7b38d6e64f0426000de174679f4c01413ec6647fa6b997219978e55` reports aggregate optimizer
tokens and FLOPs, maximum parameters per cell, deduplicated persistent storage,
peak training VRAM, and peak streamed-host RAM. It prices fp32 masters,
optimizer/gradient state, forward copies, dense code/score workspaces,
calibration-event materialization, and explicit runtime headroom; the activation
store is not assumed resident in host memory. Activation stores use the exact
whole-sequence-rounded split allocation and physical max-width site padding.
The 16-byte parameter price is explicitly checkpoint model plus Adam moments
plus the separately persisted deployment model; a schema-derived per-cell
envelope prices codec tensors, reports, manifests, nontrainable buffers, and
container metadata. Estimator v20 additionally
prices the one-batch CUDA copy-stream lookahead at each phase's exact input
precision and paired-stream topology; the existing host-prefetch queues remain
inside the separately declared streamed-host headroom. It prices tied-encoder
execution and the live materialized transpose map even though tying removes
independent parameters. It also reserves the explicit tensors and opaque
small-eigensolver workspaces used by symmetric-polar retraction and by map- and
decoder-nuclear regularizers. Phase 1 and Phase 2 enforce the universal peak
VRAM and streamed-host limits at construction, materialization, and launch;
Phase 3 additionally enforces its declared aggregate token, parameter,
storage, and dense-compute ceilings at all three boundaries.
Site-axis factorization reduces trainable parameters, optimizer state, and
checkpoint bytes. Canonical factorized execution contracts the site basis and
rank core directly and does not materialize full encoder or decoder site
tensors. Estimator v20 nevertheless retains the previous unfactorized FLOP and
operational-workspace prices: the measured speed and memory reduction are not
planning credit until a separately audited estimator version prices every
direct-rank lifetime.

Every cell serializes `factorized_execution_implementation`. Unfactorized
carriers derive `not_applicable_v1`; site-rank carriers derive
`direct_rank_space_sparse_topk_cuda_v3`, except rank-one/two map- and
decoder-nuclear cells, which derive
`direct_rank_space_sparse_topk_cuda_factor_regularizers_v4`. The explicit
`materialized_prepacked_core_reference_v2` identity is an oracle only. Unknown or
carrier/objective-incompatible identities refuse, and root, smoke, and child
materialization rederive the canonical identity after each effective delta.
The direct path contracts `[batch,d_model,sites]` with the site basis, applies
one flattened rank-core encoder map, evaluates decoder weight and Gram scores
in rank space, and decodes rank coordinates through the site basis. Padding,
bias centering, all four fusion rules, every hard selector, and code-norm,
decoder-weighted, decoded-energy, and signed isolated-loss-decrease scoring are
covered without an `[sites,groups,block_width,d_model]` runtime weight.

Release oracles cover ranks `1/2/4`, fp32/bf16, full and partial observation,
source-only fusion, padding, bias, activation, selector, score, forward loss,
and all factor gradients. The bf16 bounds are code/score relative L2 at most
`.008`, selected-code/reconstruction drift at most `.20`, mask disagreement at
most `.02`, support IoU at least `.95`, loss drift at most `.005`, and maximum
factor-gradient drift at most `.30`; fp32 bounds are respectively `3e-6`, exact
support, `3e-6`, and `1e-5`. On the Phase-2 rank-one stress shape
(`B=4096`, four width-768 sites, 2,048 groups, block width four, eight active
blocks/32 active coordinates, bf16 forward, fused AdamW), five warmups and 31 CUDA-event samples of a
complete Trainer step measure `10.482 ms` median / `11.360 ms` p95 for the
materialized oracle and `4.198 ms` / `4.514 ms` for direct rank space: a
`59.95%` median reduction (`2.497x`) and `130.0 MiB` lower incremental peak.
Across 24 paired steps, maximum loss drift is `6.91e-6`, terminal loss drift is
`5.14e-6`, support disagreement is `4.19e-4`, and support IoU is `.97353`.
This bounded kernel-order change is engineering evidence, not a new scientific
factorization arm.

Version 2 stores the logical encoder core physically as contiguous
`[rank*d_model,groups*block_width]` and the decoder core as contiguous
`[groups*block_width,rank*d_model]`. The hot encode/decode GEMMs therefore
consume parameters directly instead of permuting and packing them every step;
explicit logical-core adapters remain for materialization, score geometry, and
regularizers. Old v1 identities refuse rather than migrate. At the same jobe
shape, nine 31-step samples show rank one neutral (`4.1712` to `4.1708 ms`),
rank two `7.842` to `7.460 ms` (`4.87%`) with `24.0 MiB` lower peak, and rank
four `14.022` to `13.279 ms` (`5.29%`) with `96.0 MiB` lower peak. Across 24
steps, the v1 and v2 materialized master weights, bf16 forward weights,
supports, and losses are bitwise identical at ranks `1/2/4`.

The unfactorized untied encoder likewise stores its sole parameter physically
as contiguous `[sites*d_model,groups*block_width]`; `encoder_tensor()` is the
logical `[sites,groups,block_width,d_model]` view used only by explicit
objectives and oracles. The hot flattened GEMM therefore consumes the
parameter directly and backward no longer unpacks its gradient. Checkpoints
with the former four-dimensional `E` shape refuse at load rather than migrate.
A six-step mapped materialization oracle is bitwise exact for model, optimizer,
scheduler, and diagnostics. On the canonical Phase-2 bf16 shape (`B=4096`,
four width-768 sites, 2,048 groups, block width four), 31 jobe wall samples
reduce complete Trainer median from `12.469` to `11.349 ms` (`8.98%`,
`1.099x`) and p95 from `12.803` to `12.279 ms`; the later full-step workspace,
not the removed 48 MiB pack, remains the allocation peak.

For a tied unfactorized encoder, one shape-polymorphic CUDA Inductor producer
fuses eager `exp(log_gamma)` scaling with the logical-to-GEMM permutation and
writes only the packed encoder allocation. Its logical return is a zero-copy
view, so map regularizers and frozen partial-view evaluation retain the same
geometry. Calls below one million decoder elements, non-CUDA devices, and
other dtypes remain eager. The release oracle requires bitwise packed weights,
codes, losses, and decoder gradients; the scalar gamma gradient has a `2e-6`
relative bound across cold and warm Inductor caches. A paired alternating jobe
benchmark at the same canonical bf16 shape reduces complete Trainer
median from `10.675` to `10.298 ms` (`3.53%`, `1.037x`) and p95 from `11.557`
to `11.279 ms`; the later workspace still determines peak allocation.

Version 3 retains those contraction-ready layouts. The separately serialized
universal sparse-decode identity replaces dense zero-filled decode for bf16
CUDA hard-TopK batches of at least 2,048 tokens and support density at most
`1/32`, for both direct factorized carriers and native contiguous unfactorized
decoders. The native kernel reads and differentiates `[S,G,b,d]` strides
directly; packing or transposing the decoder every step is forbidden. The
batch gate retains dense Tensor Core decode for Phase-3's 256-token geometry,
where the sparse kernel is slower. A fixed-size row-major event extraction
avoids a dynamic-nonzero host synchronization; deterministic Triton kernels
reduce by row in the forward pass and by group in the decoder-weight backward.
Other dtypes, selectors, densities, and the materialized factorized reference
remain dense. The no-Aux, zero-regularizer Trainer also bypasses the otherwise
unused dense selected-code tensor; objective-bearing consumers keep it while
still receiving sparse decode.

The release kernel bounds forward, code-gradient, and decoder-gradient relative
L2 drift by `.005`, repeated sparse trajectories deterministically reproduce,
and a 24-step canonical rank-one comparison bounds maximum loss drift by
`1.30e-6`, optimizer-state relative L2 by `7.81e-7`, model-parameter relative
L2 by `.00265`, and support disagreement over the union by `.0063`. On jobe,
100 post-warmup CUDA-event samples reduce median rank-one step time from
`4.0643` to `3.3657 ms` (`17.19%`, `1.208x`), p95 from `4.1271` to
`3.5712 ms`, and peak allocation from `492.348` to `440.747 MiB`. This bounded
bf16 reduction-order change is content-bound engineering evidence, not a new
scientific arm. Old v2 identities refuse rather than migrate.

For the native unfactorized carrier, primitive output, code-gradient, and
decoder-gradient relative L2 drift are bounded by `.004`, `.004`, and `.005`;
unselected code gradients remain exactly zero and repeated execution is
bitwise deterministic. Tied and untied QR trajectory fixtures preserve at
least `.98` support IoU, `.005` per-step loss drift, `.05` global parameter
drift, and the existing Gram bound. On jobe at `B=4096`, four width-768 sites,
2,048 groups, block width four, tied bf16 QR, and fused AdamW, the complete
Trainer median falls from `11.792` to `8.820 ms` at eight active blocks
(`25.20%`, `1.337x`) and from `11.810` to `10.548 ms` at 32 active blocks
(`10.69%`, `1.120x`). Peak allocation falls from `768.31 MiB` to
`680.70 MiB` and `681.83 MiB`, respectively. The conservative estimator grants
no credit for this reduction. Token TopK additionally binds its exact constant
events per row into the sparse decoder, deriving the CSR row pointer directly
instead of counting and cumulatively summing the already row-major event list;
BatchTopK retains the generic construction. The specialized and generic
forward and backward tensors are bitwise identical. On the same jobe shape,
paired complete-step medians improve from `8.836` to `8.524 ms` at eight
active blocks and from `10.507` to `10.425 ms` at 32, with unchanged peaks.

Version 4 leaves the version-3 encode, score, selector, and decode carrier
unchanged, but rank-one/two map and decoder nuclear objectives contract the
site/core pair Grams directly. Structurally padded coordinates are masked in
both cores. Supplied materialized tensors and the v2 identity remain explicit
oracles; affected v3 checkpoints refuse rather than silently changing the
objective kernel. Across eight fp32/bf16 oracle seeds, fp32 value and maximum
factor-gradient relative errors are at most `2.28e-7` and `4.39e-7`; bf16
value and gradient errors are at most `.001499` and `.004348`, with gradient
cosine at least `.9999938`. Exact-zero smoothing, rank-deficient encoders,
padding, ratio calibration, nonzero training, and exact resume are release
fixtures. On jobe at `B=2048`, four width-768 sites, 2,048 groups, block width
four, bf16 forward, and fused AdamW, complete Trainer medians change as follows:
map nuclear rank one `11.043` to `5.716 ms` (`1.932x`, `623.7 MiB` lower peak),
map rank two `13.258` to `8.583 ms` (`1.545x`, `527.0 MiB` lower), decoder
nuclear rank one `7.214` to `3.938 ms` (`1.832x`, `311.9 MiB` lower), and
decoder rank two `8.227` to `6.993 ms` (`1.176x`, `71.0 MiB` lower). Estimator
v17 retains its conservative materialization allowance.

The full unfactorized map-nuclear carrier separately serializes
`batched_site_gram_reference_guard_d1e-3_e1e-4_v1`: it forms each site's fp32
block Gram with batched matmul and then reduces the site axis, instead of
contracting sites inside one einsum. The whole call falls back to the old
`site_reduced_einsum_reference_v1` unless every decoder Cholesky diagonal
ratio exceeds `1e-3` and every transformed-map spectral ratio exceeds `1e-4`.
This preserves exact rank-deficient-encoder behavior and near-singular-decoder
refusal rather than loosening either scientific contract. Fast-path fp32/bf16
value relative drift is at most `2e-6`; decoder/encoder gradient relative L2
drift is at most `1e-4` in fp32 and `5e-4` in bf16. Fallback tensors and errors
must match the reference exactly. The 25-step bf16 trajectory gate requires
loss drift at most `1e-5`, support IoU at least `.995`, model relative L2 drift
at most `2e-3`, and optimizer relative L2 drift at most `1e-5`; exact resume
under the selected identity is bitwise. On jobe at four width-768 sites,
2,048 groups, and block width four, complete regularizer forward/backward
including both guards falls from `7.200` to `4.353 ms` (`39.5%`) and
incremental peak allocation falls by `384 MiB`.

Frozen threshold fitting also stays in the direct rank carrier instead of
materializing full encoder and decoder site tensors. On eight canonical
4,096-token calibration batches, the streaming pass falls from `17.747` to
`10.499 ms` (`40.84%`, `1.690x`) and peak allocation falls by `96.0 MiB`; the
fitted threshold remains bitwise identical.

Codec calibration selection and trusted packet decode likewise remain in rank
space: sparse packet rows multiply `D_core` and then project through `D_site`.
At rank one, the 2,048-token threshold-select kernel falls from `2.147` to
`.878 ms` (`2.445x`); four-batch R-D evaluation falls from `22.866` to
`15.959 ms` (`30.21%`, `1.433x`), and codec-fit/evaluation peaks each fall by
`192.0 MiB`. Codec fitting batches ragged per-group coordinate quantiles in a
bounded NaN-padded workspace; it is bitwise identical to the scalar group loop
and preserves the complete artifact hash. This reduces rank-space fit time
from `67.340` to `28.712 ms` (`57.36%`, `2.345x`) and improves on the original
materialized fit's `73.311 ms` by `60.84%` (`2.553x`).

R-D evaluation queues every q-chunk and observer consumer before packing
counts, Bernoulli terms, all-q per-site errors, and centered totals into one
fp64 D2H transfer. Sequence grouping consumes the transferred count column,
never the CUDA source. A direct `row_len=1` append path preserves row and
bootstrap order without one Python/CUDA scalar synchronization per token. The
complete result JSON hash is unchanged; canonical 128-token-row evaluation
falls from `15.984` to `14.932 ms` (`6.58%`), while the Phase-1-style
one-token-row case falls from `349.824` to `37.567 ms` (`89.26%`, `9.312x`).
The paired prefetch pipeline selects floating activation leaves explicitly:
row identities remain in their original unpinned CPU storage instead of being
copied to CUDA and immediately returned for sequence grouping. At 4,096 rows
this removes `96 KiB` H2D and `32 KiB` D2H per batch. An interleaved jobe
benchmark improves a transfer-bound width-64 pair from `.3299` to `.3194 ms`
(`3.18%`) and is neutral at the Phase-2 width-768 pair (`3.7748` to
`3.7687 ms`).

Within that traversal, included support now has one trusted q-independent
carrier: its mask, per-token counts, row indices, and original block IDs feed
both rate accounting and packet rotation. The fp32 mask cast is shared by both
Bernoulli terms, and excluded events are the exact raw-minus-included integer
count. An independent-support reconstruction oracle produces the identical
complete R-D payload. In paired jobe measurements, support reuse reduces an
8,192-token evaluation from `32.634` to `31.898 ms` with 512-token batches
(`2.26%`) and from `25.582` to `25.258 ms` with 2,048-token batches (`1.26%`).

Shared-code concordance likewise constructs each blockwise full/partial
support intersection once. The same mask feeds concordance moments and exact
intersection cardinality; union cardinality is derived as
`|A| + |B| - |A intersection B|`, with all integer counts exactly representable
in fp64 at campaign limits. This removes the full-shape intersection and union
allocations without changing any code, Gram, energy, covariance, or reduction
order. On the Phase-2 jobe evaluator, median falls from `473.741` to
`470.680 ms` (`0.65%`), cumulative CUDA allocation traffic falls by `1.25 GiB`,
and the isolated set-algebra loop falls from `.891` to `.534 ms` (`40.0%`).

The blueprint enforces hard ceilings: 4,002,097,152 aggregate optimizer tokens
(4B final plus eight 262,144-token stability cells), 400M
parameters per cell, 22GB peak VRAM, 55GB peak host RAM, 850GB storage against
1TB provisioned, and the declared one-week compute envelope at a conservative
20 TFLOP/s. A plan that exceeds any ceiling is rejected before launch.
The CLI labels conditional-prefix estimates separately from complete frozen
plans. Its filesystem refusal gate compares free space to incremental bytes:
configured existing stores or transforms are credited only after manifest and
content verification, while the scientific plan's full storage estimate and
budget gate remain unchanged.

## 8. Normalization, calibration, and saved-codec contract

Normalization is a scientific factor:

- `none`: identity transform;
- `scalar_rms`: fitted mean and one centered RMS scalar per site;
- `sqrt_d`: fitted mean and scale to mean centered norm `sqrt(d_s)`;
- `whiten`: fitted shrinkage whitening with frozen eigensystem;
- `layer`: per-token LayerNorm diagnostic.

Token LayerNorm is also structurally ineligible for the linear planted-factor
subspace, isolated-input reconstruction, and aligned-code identification
conjunction. Phase-1 reports that conjunction as `applicable=false` with a
named reason and null margin/pass fields; support diagnostics remain visible.
Metric ineligibility is never encoded as a large negative scientific margin or
as a failed scientific outcome. The qualification records the exact reason in
`scientific_outcome.inapplicable_checks`; detached replay permits that neutral
check only for a resolved `data.normalization=layer` cell. Other promotion
constraints, including nondeployable inverse side information and the cell's
declared promotability, remain independent.

Dataset statistics use fp64 accumulation and fp32 transforms. Raw capture and
store payloads are never fp16. Any inverse requiring source-token information
must serialize and price that information or declare raw evaluation
ineligible.

Calibration never mutates the trained checkpoint. It emits a separate,
content-addressed codec binding the cell, checkpoint, store, transform,
threshold, included-block table, rotations, clipping bounds, quantizers, and
side-information contract. Evaluation reloads both artifacts and performs a
source-free round trip.

Codec orientation is `second_moment_ordered_event_frame_v2`. Calibration orders
active-code second-moment spectral clusters by descending eigenvalue. Separated
one-dimensional clusters retain their principal axes; repeated or near-repeated
clusters (adjacent relative eigengap at most `1e-6`) use a gauge-equivariant
projected two-pass Gram--Schmidt frame over the active mean followed by active
codes in immutable calibration-stream order. Non-null directions that those
vectors cannot identify fail closed. A direction at or below `512` times fp64
epsilon times the block's largest eigenvalue is calibration-null: its arbitrary
orthonormal storage completion is harmless only because calibration forces its
clip interval to exact `[0,0]`, so it contributes identically zero in every
gauge. The codec serializes and revalidates both tolerances, minimum relative
eigengap, near-degenerate block IDs/count, null block IDs/per-block dimensions,
null-coordinate count, and maximum cluster/null dimensions. Gauge-rotated
exact-isotropic, near-degenerate, and rank-deficient release fixtures must
produce matching rate--distortion points; generic random simple-spectrum
coverage is insufficient.

Quantizer division uses a safe denominator only to form a symbol.
Reconstruction always multiplies by the exact serialized `hi-lo` span: a
zero-span coordinate emits symbol zero and decodes exactly to its constant
floor, while a positive span smaller than `1e-12` still saturates at its actual
ceiling rather than an artificially widened interval.

The multi-quantizer CUDA decoder gathers each selected event's inverse
canonical rotation once per batch and applies every bounded quantizer chunk as
a broadcast row-vector matmul when the block width is at least two. Scalar
width one and every CPU decode retain their direct scalar/einsum reduction.
This authorized CUDA reduction-order change has a fixed standardized release
fixture at 65,536 events: against the direct einsum oracle, widths `2/4/6/8`
require maximum absolute drift at most `5e-6` and relative L2 drift at most
`3e-7`, while width one remains exact. The Phase-2 and Phase-3 campaign-shape
benchmarks additionally cover their complete quantizer chunks and decoded
predictions. Hoisting the gathered rotations replaces the repeated per-chunk
gather and does not add trusted-decoder workspace. Any kernel or bound change
requires a new clean implementation identity and fresh audit before launch.

## 9. Training and resumption

Every cell resolves optimizer, betas, epsilon, parameter-group weight decay,
batch, learning rate, warmup, schedule, precision, initialization, decoder
constraint, retraction cadence, objective reduction, regularizer, Aux bundle,
deadness unit, selector tie behavior, and inference-threshold estimator. Only
Adam and AdamW are scientific optimizer choices in the live recipes.

Every non-smoke CUDA cell executes its declared Adam or AdamW with
`foreach=False, fused=True`; CPU smoke executes `foreach=False, fused=False`.
This is a content-addressed engineering choice, not an ambient PyTorch default
or a fallback. Fused construction refuses non-CUDA or non-fp32 master
parameters. Checkpoint save, load, exact resume, and final validation bind the
optimizer kind plus each parameter group's fused/foreach flags, betas, epsilon,
weight decay, and immutable Adam controls. This is necessary because PyTorch's
optimizer loader otherwise permits serialized group flags to replace the
constructor's kernel choice silently.

Registration freezes implementation identity v2 before publishing plan or
worker state. Its execution digest covers the exact Python package bytes,
dependency versions, Python/platform/Torch/CUDA build, numerical backend flags
and environment, driver/cuDNN, and physical CUDA device identities. Git
commit/dirty state remains authenticated provenance and a clean-commit gate for
scientific work, but does not perturb the execution digest when identical bytes
are committed. Every preparation compares this pin in O(1); there is no
first-preparation adoption or legacy identity shape.

On jobe, isolated optimizer steps are `2.10x` faster for Adam and `2.37x` for
AdamW. In the final wired AdamW gate, complete Phase-2-shaped QR Trainer steps
improve from `101.296` to `99.214 ms` (`2.06%`), while polar-carrier steps
improve from `15.986` to `13.887 ms` (`13.12%`); QR peak allocation falls by
`92.8 MiB`. The standardized 20-step QR comparison records support IoU
`.997927`, loss relative difference `1.73e-4`, parameter relative difference
`.0381`, and optimizer-state relative difference `.0209`. These bounded
trajectory changes are authorized before the first experiment run. The
resource planner remains conservative and grants no fused-kernel memory credit.

For bf16 training, each leaf gradient is copied into its persistent fp32 master
buffer and immediately released. If a parameter used by an earlier graph is
absent from a later graph, the master buffer is explicitly zeroed so Adam
moment advancement and decoupled weight decay remain identical to the former
retained-gradient behavior. On jobe at the rank-one Phase-2 shape this reduces
median step time from `4.168` to `4.029 ms` (`3.32%`) and p95 from `4.187` to
`4.043 ms`; rank four falls from `13.280` to `12.823 ms` (`3.44%`). Inter-step
allocated memory falls `24.0 MiB` and `96.0 MiB`, respectively, while absolute
step peak falls `12.0 MiB` and `48.0 MiB`. Twenty-four-step model, optimizer,
support, and loss hashes remain bitwise identical at ranks one and four. The
planner retains its larger pre-optimization gradient allowance.

The post-optimizer global finite scan is also a reusable certificate for every
canonical projection composed only of finite masks, zeroing, Frobenius-ball or
unit-Frobenius scaling, and latent-row scaling. Even an overflowed finite fp32
norm produces a finite zero scale. Decoder/encoder/bias tensors mutated only by
those operations therefore skip the redundant post-projection scan; polar
eigensolve output remains uncertified and scanned. On jobe this removes
`.144 ms` for a decoder-only projection and `.251 ms` when decoder and encoder
are both normalized, roughly `1–2%` of affected every-step cells.

Decoder retraction has a separate serialized implementation identity. Canonical
QR cells derive `cholesky_qr1_positive_diagonal_cond64_v1`, polar cells derive
`symmetric_polar_site_bmm_guard_g1024_w8192_c512_f2_r1e-4_v2`, and other decoder carriers derive
`not_applicable_v1`; root, smoke, and child cells recompute that identity.
Positive-diagonal Cholesky-QR1 fails closed above condition 64 or on nonfinite,
factorization, reconstruction-residual, or post-Gram-residual failure, with no
runtime fallback. `householder_qr_positive_diagonal_v1` is an admitted
reference/test oracle but is never canonically derived. QR versus polar remains
scientific; Cholesky versus Householder within QR is engineering.

On CUDA, the polar identity applies its already computed inverse square root
to one site at a time through a single reusable block scratch. This is bitwise
equal to the former CUDA broadcast einsum for output tensors and floor counts
across one, four, and six sites; CPU retains the existing einsum. At `S=4`,
`G=2048`, `b=4`, `d=768` on jobe, the polar primitive falls from `1.558` to
`1.175 ms` (`24.6%`); the complete Phase-2 polar step is projected to improve
by about `3.0%` with no peak increase.

Version 2 also forms the fp32 multi-site Gram through sequential site BMMs;
`symmetric_polar_eigh_floor_v1` remains the explicit site-reducing-einsum
oracle. The fast branch is CUDA-only, requires at least 1,024 groups and
`S*b*d >= 8192`, and retains the fixed 512-group chunk ceiling. Before any
chunk is mutated, every eigenvalue must be finite and its minimum must exceed
both twice the eig-floor and `1e-4` of the maximum; otherwise that whole chunk
recomputes through v1. Thus small cells and floor-active or rank-deficient
chunks are bitwise reference-exact. The admitted fast primitive is bounded to
`5e-6` relative output drift, `1e-5` post-Gram-residual difference, `1e-4`
absolute post-Gram residual, and exactly zero floor hits. Its canonical
20-step bf16 gate requires loss drift at most `2e-5`, support IoU at least
`.995`, model relative L2 drift at most `3e-3`, and optimizer relative L2 drift
at most `1e-5`; same-identity resume is bitwise exact. Combined with sitewise
application, the canonical polar primitive falls from `1.567` to `.651 ms`
(`58.4%`); `G=4096` falls `54.1%`, and chunked `G=8192` falls `16–21%` without
the roughly 2.25 GiB unchunked workspace.

The fixed jobe benchmark uses five warmups and 31 separately synchronized
CUDA-event samples, with nearest-rank p95, peak reset after warmup, and all
input preparation outside the timed region. At the Phase-2 QR geometry
(`S=4`, `G=2048`, `b=4`, `d=768`), the fully guarded primitive measures
`88.784 -> 1.765 ms` median and `88.829 -> 1.777 ms` p95 (`50.00x` at p95),
while incremental workspace falls from `360.125` to `96.688 MiB`. Final
post-Gram residuals are `7.25e-7` and `5.66e-7`. The canonical in-memory
Trainer fixture uses an already CUDA-resident bf16 `[4096,4,768]` batch,
block-BatchTopK/code-norm selection at `k=32`, no auxiliary, constant-rate
fused AdamW, and `materialize_record=False`; it deliberately excludes store
I/O, H2D transfer, and on-the-fly normalization. Median step time falls from
`100.952` to `14.216 ms` (`85.92%`, `7.10x`), p95 falls from `101.002` to
`14.241 ms` (`7.09x`), and peak allocation falls by `176.125 MiB`. Across 20 paired
campaign-shape updates from one canonical state, loss relative drift is
`5.86e-6`, support disagreement `7.51e-5`, support IoU `.980960`, model-state
relative drift `.002377`, optimizer-state drift `5.69e-6`, and both terminal
Gram residuals remain below `7e-7`. These bounds authorize the engineering
replacement before the first experiment run; changing a guard or gauge
convention requires a new implementation identity.

The Trainer's global post-optimizer parameter/state scan now supplies QR's
input-finite fact directly. Both QR implementations still validate every
factor and the complete post-Gram before their transactional copy; any
nonfinite candidate necessarily makes a diagonal Gram sum-of-squares and its
post-residual nonfinite. The duplicate input, candidate, and post-projection
full-decoder scans are therefore elided without changing the refusal set.
Exact checked/prevalidated candidate tests cover Cholesky and Householder QR,
and an injected infinite candidate must still fail transactionally. On jobe
at `B=2048`, tied token-TopK/code-norm selection and the Phase-2 QR weight
geometry, 21 post-warmup complete-Trainer samples fall from `8.836` to
`8.563 ms` median (`3.09%`, `1.032x`) and from `9.768` to `9.354 ms` p95;
peak allocation is unchanged.

The same transactional boundary lets QR finish its private candidate before
reading any intermediate predicate on the admitted path. Cholesky-QR combines
the positive-definite Gram, conditioning/reconstruction, and post-Gram guards;
Householder QR combines finite-factor, full-rank, and post-Gram guards. Only
their conjunction permits the master copy. A rejected conjunction then enters
the former failure-specific diagnostics, so speculative work changes neither
the refusal predicates nor the accepted state. CUDA profiling at the same
Phase-2 geometry reduces scalar host reads per complete Trainer step from five
to three. Two 21-sample reverse-order jobe repetitions improve the median from
`8.483` to `8.426 ms` (`0.66%`, `1.007x`) and from `8.616` to `8.480 ms`
(`1.58%`, `1.016x`); peak allocation remains `656.31 MiB`.

Large no-gradient bf16 code-norm pools with contiguous block width four use a
dedicated CUDA reduction bound by the serialized identity
`bf16_b4_n4194304_tile256_n8388608_tile128_sqrt_rn_cuda_else_native_v1`.
It is admitted only from 4,194,304 output scores upward, using 256-element
tiles below 8,388,608 outputs and 128-element tiles at and above that second
threshold. Grad-enabled scoring, other widths, dtypes, layouts, devices,
smaller pools, and the explicit `native_vector_norm_v1` oracle retain
`torch.linalg.vector_norm`. The kernel accumulates four squared coordinates in
fp32 and uses CUDA `sqrt.rn`: Triton's generic square root was rejected after
changing 5--10 of 8,388,608 bf16 scores by one ULP. Fixed-scale,
tiny/large-scale, signed-zero, infinity, and NaN fixtures require bitwise
equality or paired NaNs, and the complete fallback predicate is directly
tested. Changing the carrier, either threshold, reduction tree, square-root
mode, or tile schedule requires a new implementation identity.

On jobe at `[4096,2048,4]`, the isolated norm falls from `.162--.174 ms` to
`.092 ms`. In alternating 26-step complete-Trainer gates from identical state,
rank-four site-factorized execution falls from `9.351` to `9.295 ms` median
(`.60%`) and the unfactorized control falls from `7.117` to `7.042 ms`
(`1.06%`); fp32 master parameters, bf16 forward parameters, and AdamW state
remain bitwise identical after every compared trajectory.

The shared kernels also retain guarded configuration values used only by unit
fixtures or explicitly quarantined source-release adapters. Those values are
test-only or quarantined, not latent matrix rows: only a canonical cell emitted
by `studies.py` is live, and nonmaterializable recipes fail before execution.

The direct training, calibration, codec, and all-view evaluation encoder uses
one flattened `(site, coordinate)` GEMM. Partial-view sharing evaluation keeps
one per-site contraction and fuses each declared observation mask from that
cache; this is the same linear map but has a different floating reduction
order. Both paths are bound by the clean implementation commit before prepare.
The named `flattened_encoder_reduction_sensitivity` engineering ablation uses
the superseded per-site BMM as a test oracle across fp32/bf16, every fusion and
weight topology, both hard selectors, and all four score geometries. Its fp32
gate requires exact support, relative output/loss drift below `2e-6`, and
relative gradient drift below `5e-6`. Its bf16 gate requires code and score
drift below `.006`, changed support below `.02`, loss drift below `.003`, and
maximum parameter-gradient drift below `.25` in both the complete topology
matrix and the larger selector-by-score matrix. These are pre-run release
gates, not a tunable scientific matrix row; changing either kernel or bound
requires a new clean implementation identity and a fresh audit before any
campaign starts.

Native reconstruction in the joint fp32 CUDA evaluator uses a per-site CSR
SpMM only when the complete batch support contains at most
`floor(tokens * groups / 32)` block events. Per-token counts and the scalar
event count are resolved before any dynamic event tensor is allocated; hard
token/BatchTopK instead passes its exact shape-derived event count and uses
fixed-size row-major `nonzero_static`, so full/site-only/leave-one-out TopK
views never read a CUDA scalar. Threshold views retain the dynamic count gate. Support
above the inclusive cap, non-fp32 tensors, and every non-CUDA device retain the
dense decoder. The sparse path fills one preallocated site slice at a time,
releases that temporary before the next site, and applies the identical decoder
bias and structural coordinate mask. This changes only the floating reduction
order of native reconstruction. The standardized Phase-2 and Phase-3 direct
dense-oracle gates cover both hard selectors and every full, site-only, and
leave-one-out view: prediction maximum absolute drift is at most `1e-6`,
prediction relative L2 drift at most `3e-7`, and per-site squared-error relative
drift at most `1e-9`. Repeated CSR execution is bounded, not claimed bitwise
deterministic, with maximum absolute disagreement at most `1e-6`. Zero support,
the exact density boundary, the first event above it, bias, padding, and dtype
fallbacks are release fixtures. Estimator `dense-linear-memory-v20-e8cd28fa...`
content-binds and prices the capped coordinates, values, columns, row pointer,
and one live site output. Any kernel, density, or bound change requires a new
clean implementation identity and fresh audit before launch.
On jobe, a four-site TopK profiler removes exactly nine
`aten::_local_scalar_dense` calls per batch (`59` to `50` total evaluator
calls). A 512-token TopK-only pass falls from `73.058` to `72.345 ms` (`.98%`);
the 4,096-token pass is neutral (`367.348` to `367.012 ms`).

The joint selector/shared-code evaluator retains a private lean view record,
not a training `BSCOutput`: dense selected codes are released immediately
after their selector-specific decode because every later concordance endpoint
uses the raw code and mask. A weak-reference release gate permits at most the
two current selector decode inputs and requires every prior view's selected
codes to be gone. Reconstructions are likewise reduced to per-site SSE and
released before concordance; repeated full/null semantic views cache that
small exact reduction rather than a wide prediction. On the canonical Phase-2
evaluation shape (`B=4096`, four
width-768 sites, 2,048 groups, block width four), jobe peak allocation falls
from `4,893.567` to `4,765.567 MiB` after selected-code release and then to
`4,717.567 MiB` after prediction release (`176.0 MiB` total), while median
complete shared-code evaluation remains neutral (`495.063` initially;
`494.550` before and `494.809 ms` after prediction release). No arithmetic
operation or endpoint schema changes.

Rate-distortion evaluation serializes
`evaluation_execution_implementation=fused_deployable_full_view_packet_v2`
under evaluation schema v2. One paired evaluation stream owns the deployable
full-view encode and scores, both selector masks, shared-code endpoints, the
q-independent packet events, and exactly
`ceil(number_of_quantizers / 2)` trusted sparse decodes per batch. The codec
accumulator preserves transformed-space rate arithmetic, sequence grouping,
bootstrap draw order, and its public `evaluate_rd` payload; a synchronous raw
observer consumes those same decoded chunks, applies the deployment inverse,
and accumulates paired raw-space endpoints and schedule-cache errors. The first
packet event stream is also reused for the independent public source-free
roundtrip. The selector/shared-code pass no longer rereads the normalized
split, repeats its H2D copy, or repeats the full-view encode. The unique full
carrier is deliberately the deployable path: direct-factorized cells remain
in rank space, mapped isolated-loss scoring uses the canonical unmasked full
view, and real endpoints consume the fp32 view reconstructed from deployment
bytes rather than a tolerated persisted bf16 copy. This is a new execution
identity because those formerly distinct full carriers cannot be fused while
preserving all three old arithmetic paths. Packet/rate arithmetic and every
site-only/leave-one-out re-encoding retain their prior semantics.
On jobe at the canonical Phase-2 geometry (four width-768 sites, 2,048 groups,
block width four, 512 pinned-host tokens, a 201,338,880-byte model), five
alternating end-to-end in-memory passes measured `116.836 ms` for the separate
selector/shared plus R-D traversals and `116.156 ms` for the fused traversal:
`1.0059x`, saving `0.680 ms` per batch before counting the eliminated second
store read. Direct packet-event comparison proved counts, block IDs, original
IDs, and canonical codes bit-identical when gathering raw `z` at the selected
positions. The final v20 lifetime releases encoder-site, partial-view, and
shared reducer carriers before packet/R-D consumption. It retains only the
full threshold `z`/score/mask carrier and streams the R-D consumer in exact
ordered 4,096-token microbatches without changing the declared 8,192-token
scientific batch. The estimator prices that irreducible threshold carrier
additively with the bounded R-D workspace. The largest current Phase-3 scalar
comparator projects `19,212,206,256` bytes against the 22 GB gate.
The executor composes ordered pinned-host prefetch with a one-batch-ahead
dedicated CUDA copy stream for training, ordinary metric evaluation, and the
paired rate-distortion stream. Threshold fitting, achieved-rate preflight, and
codec fitting independently use the same pipeline and explicitly close each
iterator so early return or failure drains its CUDA stream and host producer.
Per-batch events and allocator stream recording preserve consumer order without
a device-wide synchronization.

On jobe at the canonical Phase-2 throughput shape (`B=4096`, four width-768
sites, 2,048 groups, block width four, site rank one, bf16 fused AdamW), seven
alternating 31-step samples after five warmups measure `6.072 ms/batch` median
for default-stream H2D plus training and `4.315 ms/batch` with the dedicated
copy stream: a `28.93%` throughput reduction (`1.407x`). P95 falls from
`6.078` to `4.319 ms`; the deliberate one-batch lookahead adds `24.0 MiB`
incremental peak allocation. Twenty-four paired steps leave both fp32 master
and bf16 forward states bitwise equal.

For the 250,000-token Phase-2 calibration split at `B=4096`, four width-768
sites, 2,048 groups, and block width four, seven alternating jobe pairs reduce
threshold fitting from `.8958` to `.3157 s`, achieved-rate traversal from
`.8682` to `.3092 s`, and codec fitting from `1.4384` to `.5850 s`. Total
calibration falls from `3.2034` to `1.2505 s` (`60.96%`, `2.562x`); p95 falls
from `3.2176` to `1.2711 s`. Peak allocation rises by one prefetched activation
batch, `840.66` to `888.66 MiB`. Threshold, selected-event/token counts, and
all ten codec tensors remain bitwise equal.

Phase 3 prefetches the one raw bf16 view, transfers it once, and applies the
serialized normalization on CUDA. Phase 2 prefetches paired raw and persisted
bf16 views, verifies row identity, reconstructs the actual encoder input from
deployment bytes on CUDA, and releases the persisted view after an independent
comparison. Large CUDA comparisons use one dynamic compiled reduction for
maximum absolute and normalized allclose error; CPU and small tensors remain
eager. Identity, diagonal, whitening, LayerNorm oracle, synthetic, zero-event,
bias, padding, unusual quantizer order, chunk-tail, row-order, persisted-view
mismatch, and public-packet corruption cases are release fixtures. Evaluation
and campaign validators fail closed on a missing or superseded joint identity.

At `B=4096`, four width-768 sites, 2,048 groups, block width four, six
quantizers, and approximately 32 events/token on the RTX 4090, five warmups and
31 synchronized wall-clock samples measure `40.939 ms` median / `41.062 ms`
p95 for the former separate transformed/raw traversals and `25.707 ms` /
`25.748 ms` for the joint traversal: a `37.21%` median reduction (`1.593x`),
`1.595x` at p95, and `244.8 MiB` lower incremental peak. Repeated sparse
execution gives maximum transformed-FVU relative disagreement `4.99e-11` and
raw aggregate disagreement `1.25e-11`. On the same activation shape, compiled
persisted-view validation measures `0.326 ms` versus `0.695 ms` eager (`2.13x`)
and saves `96.0 MiB` peak. Estimator v20 explicitly prices paired raw and
transformed targets, both metric lifetimes, all-q raw errors, cached forward and
inverse normalization operators, and the persisted-validation peak; it grants
no speculative memory credit for the removed second traversal.

Decoded energy remains the scientific score. Its serialized
`implementation.decoded_energy_implementation` field selects either the exact
decoder-Gram quadratic or `stiefel_code_norm_bounded_v1`. The latter is derived
only for an unfactorized Gram- or QR-constrained decoder, decoded-energy
scoring, token-TopK or block-BatchTopK training selection, and decoder
retraction after every optimizer update. Under that carrier it evaluates
`||z_g||` directly and omits the selector's fp32 decoder Gram; every other
configuration retains `exact_decoder_gram_v1`. The implementation identity is
recomputed after every child-cell delta rather than inherited from its parent.

The bounded implementation refuses fp32 master Gram residual above `1e-4` or
bf16 forward-copy residual above `2e-3` at initialization before score-using
calibration, periodic diagnostics, checkpoint save, exact resume, trained-model
load, and deployable-codec load. Checkpoint, run-binding, outer deployable
model, and nested codec model configurations must agree exactly. These boundary
checks deliberately avoid rebuilding the Gram on every scored step; the
required every-update retraction preserves the carrier between them. Release
fixtures compare exact and specialized paths from identical state. In fp32
they require score relative L2 drift at most `2e-6`, exact hard support, and
output, loss, and parameter-gradient relative drift at most `2e-6`. In bf16
they require score relative L2 drift at most `2e-3`, mask-element disagreement
at most `1e-3`, support IoU at least `.99`, output relative L2 drift at most
`.05`, loss relative drift at most `1e-4`, and maximum parameter-gradient
relative drift at most `.06`. The 25-step trajectory gate additionally
requires nested model/optimizer-state
relative L2 drift at most `.05`, active-support disagreement at most `.02`, and
support IoU at least `.95`. Calibration, threshold inference, codec selection,
and partial-view selectors use the same centralized score branch. Publication
evaluation still constructs its exact fp64 site and all-site decoder Grams for
decoded-energy coverage and concordance; the specialization does not alter
those endpoints.

On the standardized bf16 CUDA fixture, explicitly casting the code to fp32
before the norm produces bitwise-identical bf16 scores and support, while the
Phase-2 score kernel is `2.681x` slower and adds `320 MiB` peak allocation
instead of `32 MiB`. That non-improvement is excluded; the direct bf16 norm is
the bound implementation.

The fixed Phase-2 campaign-shape gate (`B=2048`, four sites of width 768,
2,048 groups, block width four, eight active blocks, bf16 forward) measures the
complete Trainer step, including every-update polar retraction. Across two
idle-GPU runs the bounded implementation reduces median step time from
`16.591 ms` to `12.698 ms` (`23.47%`) and peak allocation by `352.125 MiB`.
Its identical-state one-step score/loss/maximum-gradient relative drifts are
`4.39e-4`, `4.16e-7`, and `.03193`, with support IoU `.999390`. Across 25
paired steps, combined model-plus-optimizer state drift is `.02535`, accumulated
support disagreement is `7.31e-5`, and support IoU is `.981462`. Initial and
terminal fp32/bf16 Gram residuals remain within their declared gates.

Estimator v20 conservatively credits only four fp32
`[batch_tokens, groups, block_width]` selector work buffers plus the omitted
fp32 score Gram when the explicit bounded identity and its full predicate both
hold. An otherwise eligible exact implementation receives no credit. Sparse
evaluation workspace and exact fp64 sharing-Gram residency are unchanged.
Cholesky-QR1 additionally reserves
`4 * (sites * padded_site_width * groups * block_width + 6 * groups *
block_width^2)` bytes of training workspace. The direct-output batched matmul
does not materialize a second transformed-decoder tensor. Large-CUDA finite
classification and reduction use one shape-polymorphic compiled kernel rather
than materializing a bool tensor; smaller and CPU tensors stay eager. Measured
incremental Cholesky workspace is `96.688 MiB` against a `96.750 MiB` Phase-2
estimate and `641.373 MiB` against a `641.500 MiB` Phase-3 estimate. The
planner grants no speculative retraction speed or FLOP credit.

The dominant all-observed, unpadded CUDA quadratic objective compiles the fp32
cast, residual, square, scalar sum, and declared normalization division as one
shape-polymorphic reduction. This avoids the hard eight-static-shape Dynamo
recompile limit across heterogeneous campaign cells. The parallel reduction is
an authorized implementation-order
change, not a new scientific objective. Its eager-oracle release gate requires
relative loss and prediction/target-gradient drift at most `2e-6`, multi-step
model and optimizer state relative L2 drift at most `1e-5`, and hard-support
disagreement at most `1e-4`. Masked, padded, nonquadratic, small, and non-CUDA
objectives retain the eager reduction. Any kernel or bound change requires a
new clean implementation identity and a fresh audit before launch.

Atomic checkpoints include model, optimizer, scheduler, retraction/dead-state,
attempted and accepted tokens, data cursor, Python/NumPy/Torch/accelerator RNG
states, and all content bindings. Resume validates the bindings and continues
the exact deterministic presentation stream. The campaign never
garbage-collects a recorded final checkpoint or store and emits no retention
journal event. Archival or deletion is an external operational action; any
recorded artifact that later goes missing fails verification.

The append-only journal supplies concurrency-safe operational history and its
transition/artifact consistency is replayed at freeze. Before freeze, however,
it is not an externally authenticated origin ledger: a writer with filesystem
access could replace the journal and all matching evidence. The frozen decision
content-addresses and replays the bytes it receives, but cannot prove who
produced that pre-freeze state. Host/directory access control and an externally
preserved frozen decision are therefore part of the provenance trust boundary.

## 10. Qualification, scientific outcome, and promotion

The campaign lifecycle is:

`planned -> prepared -> running -> trained -> calibrated -> evaluated -> qualified`

with explicit failure, retry, and promotion events. Qualification means the
evidence package is complete: finite evaluation, method endpoints, split
integrity, resource compliance, provenance, codec round trip, and phase-specific
endpoint completeness all verify. It does **not** mean that the scientific
hypothesis succeeded.

The qualification artifact separately records:

- `scientific_outcome`: calibration exclusions, support-target accuracy, and
  the Phase-1 conjunction where applicable;
- `promotion_eligible`: no smoke profile, no oracle raw inverse, all fixed-rate
  budgets eligible, source recipe allowed to promote, and scientific outcome
  passed;
- `selection_eligibility_mode`: `scientific_promotion`,
  `smoke_protocol_only`, or `none`, internally consistent with the resolved
  cell and promotion decision;
- `selection_metrics`: the hash-bound metrics consumed by the frozen policy.

Study manifests use `bsc-study-v3`, blueprints use `bsc-blueprint-v5`,
preparations use `bsc-preparation-v4`, evaluations use `bsc-evaluation-v3`, and
qualifications use `bsc-qualification-v4`. The v4 qualification binds the
preparation plus every
downstream input hash and repeats the preparation's complete implementation
identity. Detached Phase-1 and Phase-2 decision replay reruns the same semantic
qualification contract, including outcome/eligibility consistency, rather than
accepting a self-consistent rehash. Every seed admitted to one campaign
selection must have the same implementation-identity digest. Non-smoke cells
also require a clean committed source tree at preparation. Any non-current
study or blueprint schema is refused with fresh-root guidance; it is never
silently defaulted, migrated, or rewritten.

Operational status excludes `running` cells from default runnable work and
reports them separately as resume-required. `matrix run` repeats the live
plan-bound storage preflight immediately before execution, ignores unrelated
stores (and all stores for stateless Phase 1), credits only hash-verified inputs
and recorded campaign artifacts, and exits nonzero if any selected cell fails.

A smoke reduction preserves the underlying `qualification.promotable` intent
of its full cell so selectable methods and declared controls remain
distinguishable. `runtime.smoke=true` always makes `promotion_eligible=false`.
For a uniformly smoke stage, an otherwise-promotable cell may instead be
`smoke_protocol_only`: the campaign can deterministically exercise selection,
child materialization, and resume without reading scientific outcome values or
enforcing scientific sharing/noninferiority gates. The artifact is explicitly
non-scientific. It may feed the next smoke-only protocol stage, and a uniformly
smoke Phase-2 campaign may freeze a protocol-test panel for smoke Phase 3, but
neither artifact can authorize or feed non-smoke scientific Phase 3.

Stage integrity gates may count complete negative evidence. Selection and
promotion require a passed scientific outcome and the entire declared seed
set.

## 11. Local-choice registry

The cell manifest contains the full rationale and ablation for every local
decision. This table is the human-readable registry of the choices that can
materially affect a claim:

| Choice | Lineage and hypothesis | Falsifier or guardrail | Status |
|---|---|---|---|
| common vector coordinate | novel: one intrinsic coordinate can decode at every layer | support-only DGP, scalar controls, same-block aligned-code R2 | live |
| sum, availability-rescaled sum, and source-only evidence | novel/adapted: aggregation scale and available evidence change identifiability | Phase-1 sum/mean parity and missing-site fusion capability; Phase-2 joint/source anchors | fusion semantics transferred; source-only is a control, not a tuning round |
| concatenated Stiefel gauge and QR/polar retraction | novel/engineering: fix total decoded block energy while testing numerically distinct scientific retractions; canonical QR uses positive-diagonal Cholesky-QR1 and retains Householder QR only as a reference oracle | Gram invariant, same initialized parent, recovery/rate parity; condition and residual gates refuse with no fallback | Phase-1 capability; narrow Phase-2 architecture tuning |
| block width, capacity, and activity | novel: intrinsic rank, dictionary size, and event budget are different causes | change exactly one while holding the other two fixed | Phase-1 capability; Phase-2 tuning |
| block BatchTopK | novel adaptation: heterogeneous token event counts may improve allocation | token block-TopK at identical parent and achieved packet rate | Phase-1 capability; Phase-2 tuning |
| decoder site-profile concentration penalty | rejected local proposal: concentrating decoder energy into fewer sites is directionally opposed to the shared cross-layer object, and the raw-weight profile is gauge-dependent | removed from the executable model and matrix; any future site smoothness proposal must use invariant energy/Gram profiles and explicit smooth/step truth controls | rejected, not executable |
| map-nuclear and BSF/SASA Aux transfers | adapted: source block mechanisms may control effective rank or revive dead groups | no-term parent; full coefficient/window bundle; realized Aux rate | live |
| scalar RMS, `sqrt_d`, whitening, token LayerNorm | adapted/novel gauges: separate scale/covariance confounds from feature geometry | one raw row stream, raw-space inversion, oracle-side-info refusal | live |
| initialization Gram preconditioning | adapted engineering: start decoders well conditioned without changing the declared final gauge | no-preconditioning child and initialization invariant tests | live Phase 2 |
| calibrated deployment threshold | adapted engineering: replace batch/global training selection with source-free per-token inference | calibration target error, native/deployed endpoints, saved-codec round trip | live |
| quantizer frontier and 256/384/512-bit budgets | engineering/scientific: compare methods at actual total deployment cost rather than nominal L0 | complete zero/2/4/6/8/12/16-bit frontier, exact codec bytes, no extrapolation | live |
| deterministic row replay and checkpointing | engineering: make unique data and optimizer presentations auditable and resumable; Phase 2 writes atomically every 1,000,000 accepted tokens, bounds interruption replay below one million tokens, and prices both retained bytes and cumulative checkpoint-write traffic | uninterrupted-versus-resume equality, row-stream hashes, checkpoint count, and cumulative-write accounting | live |
| CUDA execution and sparse data movement | engineering: fuse invariant checks and bf16-gradient transfer certificates, specialize exact large bf16 four-coordinate code norms, use the flattened direct encoder plus cached partial-view contractions, reuse frozen/materialized tensors, stage pinned batches, stop capture at the last requested hook, and decode or transfer only selected events without changing the mathematical cell | bitwise native score/support and exact state trajectories; named `flattened_encoder_reduction_sensitivity` ablation and bounds; exact resume and pathological-gradient fallback under the bound clean implementation; dense/sparse packet round trips, CPU lifecycle, CUDA/CPU endpoint parity, deterministic stream order | universal pre-run implementation choice, not swept |
| deterministic selector cutoff ties | engineering: exact zero/ReLU ties are common and backend TopK tie order is not a scientific choice | score descending, then lowest block index per token or lowest row-major event index batch-wide; invalid policy refused | universal, not swept |
| smoke protocol selection | engineering: exercise the complete conditional state machine without laundering tiny runs into evidence | preserve full-cell promotable intent; `runtime.smoke` blocks promotion; mode is `smoke_protocol_only`; panel escalation refused | test-only |
| site-axis factorization | adapted from FMX at the mechanism level: a low-rank layer axis may reduce parameters and impose useful cross-layer regularity without changing the sparse block coordinate | exact selected parent, unfactorized free-weight control, ranks `1/2/4`, fixed-rate and identification endpoints; repeat after nonzero/structured masking | Phase-1 capability; Phase-2 parsimony tuning and post-mask interaction revisit |
| clean-target encoder-site masking | adapted/novel: partial layer evidence may improve one-site-to-all-site function without changing the reconstruction target; fixed-cardinality draws control removed information | Bernoulli `p=0/.02/.05/.10`, exactly-one-hidden, exactly-one-retained, all-site and every site-only evaluation | Phase-1 capability; Phase-2 tuning |
| availability-rescaled masked-site fusion | novel control: a positive-mask arm must not win because literal summation lowers encoder-score scale when sites disappear | literal-sum `p=0/.10` and rescaled-sum `p=.10`; literal positive-mask arm is diagnostic | Phase-1 capability and universal transfer; not retuned in Phase 2 |
| gauge-aware selection score | novel: decoded energy prices contribution magnitude; isolated loss decrease also prices alignment with the observed input | code norm and decoded-energy controls; reciprocal-gauge invariance; hidden-target exclusion; signed-gain diagnostics; Stiefel equality controls and a common free decoder | decoded-energy Stiefel Phase-1 carrier; full Phase-2 hard score-selector interaction |
| site-map rank, factor site span, frequency law, coactivation, amplitude tails, and factor overlap | novel truth-known axes: shared structure can vary independently from frequency, correlated occurrence, coordinate tails, and geometric separation | rank-one/rank-two/independent maps; one/two/all sites; uniform/Zipf; pair forcing `0/.5/.9`; Gaussian/standardized Student-t df=3; uncontrolled/paired-30-degree subspaces | live Phase-1 confirmation |

An engineering label does not exempt a choice from ablation when the result can
depend on it.

## 12. Derived-candidate intake and implemented frontier mechanisms

The literature suggests plausible mechanisms beyond the paper parents, but an
interesting idea is not automatically a matrix axis. A derived candidate may
be added only when all of the following are supplied:

1. a mechanism-level hypothesis tied to a measured failure mode;
2. the smallest executor-representable decision delta;
3. a nearest-parent control and a falsifying endpoint;
4. an explicit compute/rate match and coefficient ladder, if applicable;
5. a reason it cannot be answered by an existing round;
6. a declared stage before any affected evidence is read.

Candidates passing that review enter one declared role—capability panel,
provisional-carrier validation, phase-local tuning, or diagnostic—and do not
multiply every paper anchor. Four mechanism families have passed intake:

1. **FMX site-axis factorization.** From a qualified four-site lineage, first
   rerun the exact selected parent. Separately derive one common
   untied/free-decoder carrier and compare its full unfactorized tensor with site ranks
   `R={1,2,4}`. Only the site axis is factorized; block and coordinate axes
   remain intact. The factorization is a transparent Tucker-style site basis,
   not a claimed reproduction of FMX's tensor-ring implementation. `R=1,2,4`
   test
   whether layer structure reduces parameters without losing fixed-rate FVU or
   identification. The common carrier is centered, untied, unconstrained in
   the encoder, and free in the decoder; the unfactorized arm makes that
   carrier change explicit rather than silently projecting factor parameters
   onto a Stiefel constraint. Phase 1 records all options as capability
   evidence and advances its exact parent by declaration. Phase 2 independently
   requires the free carrier to remain within `0.01` of the exact parent, then
   advances the lowest rank within `0.01` of full on every seed and aggregate.
   After masking selection it reruns the exact masked parent against full and
   `R={1,2,4}` under the ordinary minimum-effect rule. When zero Bernoulli
   masking wins, those rank children are conditionally elided because the
   declared rank–mask interaction is absent.
2. **FMX observation masking.** After rank selection, compare Bernoulli
   encoder-site masking probabilities `p={0,.02,.05,.10}` with two novel
   fixed-cardinality draws: exactly one site hidden and exactly one site
   retained. At least one available site remains
   visible, every site remains a reconstruction target, and evaluation reports
   both all-site input and every site-only-to-all-site matrix. The training mask
   can only remove truly available encoder evidence, repairs an all-hidden row
   by retaining one available site, and leaves clean reconstruction and Aux
   targets unchanged. The hypothesis is functional cross-layer coherence, not
   generic regularization.
3. **Availability-rescaled missing-site fusion.** Compare literal sum at
   `p=0/.10` with sum multiplied by the ratio of total available to visible
   sites at `p=.10`. The latter becomes the universal masking semantic after
   Phase 1; its purpose is to prevent missing evidence from changing score
   scale. It is not reopened as a Phase-2 hyperparameter.
4. **Gauge-aware support scores.** For free-gauge blocks compare raw
   `||z_g||` with the isolated contribution
   $\sqrt{z_g^\top(\sum_sD_g^sD_g^{s\top})z_g}$. This score is invariant to a
   reciprocal encoder/decoder gauge and collapses to code norm under the exact
   concatenated-Stiefel gauge. Also compare the signed isolated loss decrease
   $2\langle x_O,D_{g,O}^{\top}z_g\rangle-\lVert
   D_{g,O}^{\top}z_g\rVert^2$, which adds observed-input alignment and can mark
   a block as harmful. Phase 1 fixes decoded energy as a gauge-consistent
   provisional carrier and records all three scores on both the Stiefel
   equality control and a common free decoder without ranking them. Phase 2
   evaluates the full Cartesian product of those three scores with signed
   token-TopK and signed block-BatchTopK. Learned group thresholding remains a
   separate bundled method because it changes more than support allocation.

**Partial-view coordinate concordance is implemented mandatory admission
evidence**, not a future matrix cell. On the intersection of all-view and
partial-view supports, it compares codes through the decoder Gram
$G_g=\sum_sD_g^sD_g^{s\top}$ using a centered Lin-style concordance with a
mean-offset penalty. The gate consumes the worst observation-site concordance,
support-intersection recall, and decoded-energy coverage for both site-only and
leave-one-out inference; per-block distributions remain reported diagnostics.
This separates a failure to select the same block from a failure to infer the
same coordinate once selected.

Per-block diagnostics use separate, serialized eligibility rules so support
failures cannot vanish behind a concordance sample floor. Concordance requires
at least 32 intersection events and positive decoded variance; blockwise
support-intersection recall is reported for every block/pattern with positive
full-view support count; decoded-energy coverage is reported for every
block/pattern with positive full-view decoded energy, including zero-
intersection failures as zero coverage. Each distribution records eligible and
ineligible counts per site. These distributions are diagnostic; the mandatory
gate uses the unfiltered worst-site micro aggregates above.

One related gauge-invariant diagnostic remains contingent:

- **fixed-support restricted-LS refit gap:** freeze the model-selected support
  $A$ and solve
  $z_A^*=\arg\min_z\lVert x_O-\mathcal D_{A,O}^{\top}z\rVert^2+
  \eta\lVert z\rVert^2$ for $\eta\in\{0,10^{-3},10^{-2}\}$. A refit gain of
  at least `.05` aligned-code R2, `.01` FVU, or a method-order reversal would
  justify a new inference/consistency round; otherwise more encoder machinery
  is not admitted.

Observation-pattern-specific threshold calibration and a fixed per-block
effective-rank codec remain contingent diagnostics. The former opens only if
the global availability-rescaled threshold shows reproducible support-rate
drift by missingness pattern. The latter opens only after stable decoder-Gram
anisotropy and must serialize and price every rank and basis; neither may be
used as an unpriced selection improvement.

The remaining literature-derived longlist is intentionally auditable rather
than silently forgotten:

| Intake candidate | Why it could help this project | Why it is not a live matrix axis | Smallest admission test |
|---|---|---|---|
| gated block support with a separate signed coordinate head | a scalar gate can decide whether a block exists while a vector head estimates its signed manifold coordinate, avoiding support/magnitude interference | this adapts scalar Gated SAE to a new multi-site block object, adds encoder parameters, and can make gate and decoded contribution disagree | after observing support errors despite good learned subspaces, compare exact parent, parameter-matched encoder control, and gated block arm at identical events/packets; require same-block support and aligned-coordinate gains |
| SoftSAE-derived input-adaptive block count | examples may contain different numbers of active factors, so a learned per-token block budget could spend packets where complexity is high instead of relying on incidental BatchTopK variation | the source method is scalar and trains a large dynamic-sparsity MLP through Soft Top-K; a block adaptation must survive soft-to-hard mismatch, information hiding in tiny weights, and exact variable-length packet accounting | after the frozen 2-versus-8-factor stresses show a fixed-count failure, compare token-TopK, BatchTopK, learned threshold, and adaptive block count under the same mean **total packet-bit** budget, priced active-count field, and hard deployed selector |
| heterogeneous block widths or hierarchical within-block sparsity | real factors need not share one intrinsic dimension, so a mixed-width dictionary could spend one coordinate on lines and several on higher-dimensional manifolds | ragged blocks change selector fairness, parameter capacity, active-count meaning, and packet amplitude length; coordinate-level masks add more side bits | use the truth-known mixed-rank DGP, match total and active coordinates plus exact packet rate to uniform-width winners, and report recovery separately by planted rank |
| full-view-to-masked-view code consistency | matching the code inferred from partial sites to the all-site code may enforce functional cross-layer coordinates more directly than reconstruction masking alone | an agreement loss can collapse codes or erase genuine layer-local emergence, and doubles encoding work unless approximated | only if masking improves site-only FVU but code drift remains, add detached-all-site consistency with zero/frozen coefficient ladder and retain all identification and sharing guards |
| invariant site-profile smoothness or contiguity | features that emerge and transform across adjacent layers may have smoother site-energy profiles than unconstrained per-site maps | four pilot sites offer weak resolution, raw weight differences are gauge-dependent, and abrupt appearance/disappearance may be real | with a denser hook panel, compare zero versus a short ladder on an invariant decoder-energy/Gram profile against both smooth and step-change synthetic site maps |
| OrtSAE-derived block-subspace incoherence | discouraging high overlap between decoder block subspaces could reduce duplicate blocks, absorption, and composite factors | scalar decoder cosine does not define the correct penalty for signed vector subspaces, and genuinely correlated factors must not be forced orthogonal | first observe excess duplicate/mixing pathology; then compare coefficient zero with a frozen ladder using a principal-angle/projector-overlap penalty on a truth-known DGP that independently varies true factor overlap |
| Matryoshka nested block dictionaries | one ordered dictionary could preserve broad factors while providing usable prefixes at several capacity/rate points | prefix order breaks block-permutation symmetry and simultaneously changes capacity, loss weighting, and codec semantics | finalist-only comparison of nested group-aligned prefixes against independently trained equal-size dictionaries, reporting every prefix's identification and exact packet frontier |
| quantization-aware rate-distortion training | the primary real-data metric prices actual packets, so training through a quantization/entropy surrogate could improve the measured frontier rather than only post-hoc quantization | it adds an entropy model, relaxation, and rate-distortion tradeoff that could select a different ontology before the structural question is settled | post-selection appendix arm from the exact finalist, with a frozen small multiplier ladder, identical topology, actual integer packets, and every learned side-information byte priced |
| cosine or hybrid support scoring | normalizing directional agreement may prevent activation magnitude from dominating BatchTopK support | the source finds pure cosine insufficient, and block/multi-site normalization is gauge-dependent; decoded energy and isolated loss decrease already address the nearer failures | admit only if frozen diagnostics show support probability dominated by input/code norm after normalization; compare the exact parent with a declared hybrid score at matched event and packet rate |
| dense low-rank scaffold plus sparse residual | a parallel low-rank channel could absorb ubiquitous dense computation that otherwise wastes sparse block capacity | this changes the represented object and adds a second rate-bearing code path, so crossing it with BSC hyperparameters would confound ontology with tuning | dedicated falsifier after persistent dense-latent/residual-rank evidence, with a rank ladder, stopped-gradient and co-adaptation controls, and exact pricing of dense coefficients and maps |
| Procrustes prealignment of sites | layer spaces may be approximately rotated, making a common code easier to identify after calibration-only orthogonal alignment | a dense fitted map can perform the alignment the crosscoder is meant to discover and imports a cross-model solution into a same-model question | diagnostic-only identity-versus-heldout-fit alignment, fitted on calibration data, with all maps priced and compared directly with the live site-axis factorization |
| local/nonlinear manifold families (SpaDE, mixture/expert, bilinear) | curved or locally linear features are credible failure modes for a globally additive linear block | these are alternative ontologies, not hyperparameters of one block crosscoder, and require different truth association and operational coding contracts | open a separate falsifier study only if the linear BSC fails the preregistered nonlinear/manifold DGP while the alternative has a common truth-known and rate endpoint |
| active-block or capacity curriculum | a high-capacity/less-sparse warm start followed by the target event budget could avoid early dead blocks and poor local minima | the path consumes a different effective training-rate budget and may produce transient features that disappear at the target setting | admit only after frozen early-training instability; compare fixed target activity with a content-bound schedule at equal tokens, final activity, and final exact packet frontier |
| data-derived subspace initialization | calibration-only PCA or clustered local subspaces could reduce seed variance and accelerate discovery of rotated factor planes | initialization may pre-solve the identification task, consumes a fitted dense artifact, and can bias against rare factors | compare random/preconditioned parent with a fit-split-only initializer and a parameter/compute-matched random restart; score final recovery, not just convergence speed, and price the initializer if deployment needs it |

The four implemented mechanism families preserve the sparse block ontology,
already have
an exact nearest-parent delta in the shared executor, use the existing
identification/fixed-rate endpoints, and have a bounded declared role.
The deferred candidates either require a measured trigger, introduce an
unpriced second representation or alignment map, or change the ontology and
evaluation contract. They are promising conditional follow-ups, not free
degrees of freedom to add to the present 15-cell Phase 1, 410-cell Phase-2
pre-elision ceiling, and 48-cell Phase 3 (the last count is eight refusal-gate
cells plus 40 final cells). Phase-2 execution-equivalent and conditionally
vacuous cells are removed at materialization and recorded, so 410 is not an
executed-cell claim.

The relevant Phase-1 order is `site_factorization_identification`,
`site_mask_fusion_control_identification`, `site_masking_identification`, then
`selection_score_identification`; only the declared capability carriers feed
forward, including the provisional decoded-energy score. Phase 2 reopens
factorization, masking, score, and selector as real-model tuning: rank is
revisited after masking, then the full hard score-selector interaction and the
separate group-threshold method round run; only fusion remains an inherited
method semantic. The generic model,
trainer, codec path, evaluator, resume state, transfer object, and blueprint
tests bind the boundary. Further frontier ideas remain intake items until they
satisfy the six requirements above and enter an exact emitted blueprint; an
implemented model field alone is not an executable trial.

## 13. Stop rules

- Stop a source branch if its bridge fails after an implementation audit; do
  not tune a hybrid and retain the source label.
- Stop a design delta if its named benefit fails across the complete seed set
  or crosses a guardrail.
- Narrow a claim when a robustness stress fails; do not average the failure
  away.
- Stop before launch if any data, storage, codec, compute, or provenance
  binding is unresolved.
- Do not start Phase 2 without the authenticated Phase-1 decision and exact
  transfer contract.
- Do not materialize Phase 3 without a verified frozen Phase-2 panel decision.
- Do not tune on confirmation or final evaluation under any name.
