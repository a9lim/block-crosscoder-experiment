# Launch and adversarial-review contract

*Normative verification ledger, 2026-07-20.*

An audit item closes only through an executable test, a content-addressed
validation artifact, or an explicit narrowing of the claim. A clean subprocess
exit is not evidence by itself.

## 1. Source and design audit

- Every executable source label matches the paper equation or is explicitly
  `adapted`; inspected-release behavior has a separate recipe.
- Every omitted paper value remains disclosed as missing and every local fill
  has a rationale/ablation.
- SASA's omitted `lambda_dim` is represented by the declared
  `0/.01/.03/.10` initial penalty/reconstruction-ratio ladder, never by an
  invented paper coefficient. The released absolute `100` remains bound to
  the separate decoder-only diagnostic.
- The executable scope is same-model cross-layer factorization. No
  different-model partition or difference objective can enter through a reused
  name.
- Anthropic dense L1 uses the source-exact sum of per-site L2 decoder norms.
  It cannot be selected as the sparse finalist, but its Phase-3 comparator must
  come from an independently calibrated Phase-2 family selection.
- Every live matrix row appears in `studies.py`. Executor values reached only by
  unit-test fixtures or explicitly nonmaterializable release adapters are
  marked test-only/quarantined; merely retaining a guarded config branch does
  not make it a live cell. Any other unreachable branch is deleted.
- A derived candidate has a failure-mode hypothesis, nearest-parent delta,
  falsifier, coefficient policy, rate/compute match, preregistered stage, and
  declared role: capability, provisional carrier, phase-local tuning, or
  diagnostic.
- A later child stage is byte-for-byte derivable from its blueprint and frozen
  selected parent; arbitrary child manifests are refused.
- Winner-changing minimum-effect, noninferiority, and sharing thresholds are
  explicitly novel preregistered project policies, not paper values. Every
  applicable policy content-binds its rationale and complete sensitivity grid;
  each scientific selection artifact executes the marginal counterfactual
  pass sets, and observed evidence cannot retune the center policy.
- The bound sensitivity grid is exactly: minimum effect
  `0/.001/.002/.005`; noninferiority `.005/.01/.02`; partial-view FVU
  degradation `.01/.02/.05`; support-IoU drop `.02/.05/.10`; coordinate
  concordance `.50/.80/.90`; support-intersection recall `.50/.75/.90`;
  decoded-energy coverage `.75/.90/.95`; and absolute partial-view FVU
  `.75/1.0/1.25`.

## 2. Method audit

- Forward equations, masks, reductions, constraints, initialization order,
  selector scores, thresholds, and complete Aux bundles have unit tests.
- The `flattened_encoder_reduction_sensitivity` release gate compares the
  direct flattened GEMM against the superseded per-site BMM oracle across
  fp32/bf16, all fusion and weight topologies, both hard selectors, and every
  score geometry at the fixed bounds in `design.md`. The all-view evaluator
  uses the direct kernel exactly; only declared partial views use the cached
  per-site reduction. Any kernel or bound change requires a new clean
  implementation identity before launch.
- The unfactorized untied encoder parameter is physically contiguous in the
  flattened GEMM layout. Its logical adapter must share storage with that
  parameter, exact mapped forward/backward and multi-step optimizer oracles
  must pass, and a checkpoint carrying the superseded four-dimensional shape
  must refuse. Tests that inspect scientific encoder geometry use the logical
  adapter rather than relying on parameter storage order.
- Large fp32/bf16 CUDA tied encoders fuse gamma scaling and GEMM-order packing
  in one dynamic compiled producer. The logical view must share its packed
  allocation, while an eager oracle requires bitwise weights, codes, loss,
  and decoder gradient, plus at most `2e-6` relative global-scale-gradient
  drift across cold and warm Inductor caches. Small, non-CUDA, and other-dtype
  calls remain eager; compile dispatch is shape-derived rather than
  data-dependent.
- The compiled large-CUDA fp32 quadratic reduction is compared with its eager
  oracle at the fixed loss, prediction/target-gradient, multi-step
  model/optimizer-state, and selector-support bounds in `design.md`. Masked,
  padded, nonquadratic, small, and non-CUDA paths remain eager, and exact
  checkpoint resume is tested inside the compiled path. One CUDA gate drives
  more than eight distinct tensor shapes through the same dynamic graph so a
  campaign cannot exhaust Dynamo's static recompile limit.
- Every production CUDA cell binds fused Adam/AdamW and every CPU smoke cell
  binds the scalar kernel, with `foreach=False` in both. Construction refuses
  fused non-CUDA/non-fp32 masters. Checkpoint save, post-load exact resume, and
  final-checkpoint validation compare optimizer kind and immutable parameter-
  group fields so a serialized optimizer cannot replace the declared kernel.
  Adam/AdamW, QR/polar, trajectory-drift, support, and allocation gates use the
  fixed measurements and bounds in `design.md`; the planner takes no fused
  memory credit.
- Bf16 forward-copy gradients are released immediately after transfer to fp32
  masters. A previously used parameter missing from the next graph receives an
  explicit zero master gradient, preserving Adam moment and weight-decay
  semantics. Tests require no retained forward gradients, the missing-gradient
  zero update, exact resume, and the bitwise trajectory/RTX 4090 gates in
  `design.md`; the planner takes no memory credit.
- The multi-quantizer CUDA codec rotation uses broadcast row-vector matmul only
  for block widths of at least two. Its 65,536-event standardized fixtures pass
  the fixed maximum-absolute and relative-L2 drift bounds in `design.md` for
  widths `2/4/6/8`; width one and CPU remain exact against the direct einsum
  oracle. The trusted decoder resolves selected rotations once across bounded
  quantizer chunks, and Phase-2/Phase-3 campaign-shape benchmarks include the
  complete decoded prediction.
- The joint fp32 CUDA evaluator uses per-site CSR native reconstruction only at
  or below the fixed `1/32` support-density cap. Counts gate allocation before
  `nonzero`; hard TopK passes its statically derived count and uses
  `nonzero_static`, while threshold retains the dynamic gate. Denser, non-fp32,
  and non-CUDA cases remain dense. Direct dense
  oracles cover both hard selectors, all full/site-only/leave-one-out views,
  zero support, the inclusive cap and first dense event, bias, and padding at
  the prediction, SSE, and repeated-SpMM bounds in `design.md`. The estimator
  version content-binds the density denominator and exactly prices the capped
  live tensors and one released-per-site output.
- Internal evaluator view records must never retain dense selected codes after
  reconstruction or wide predictions after their exact SSE reductions.
  Weak-reference tests require all selected codes from the preceding view and
  batch to be released and at most the two current selector predictions to
  coexist. Repeated full/null views may retain only their reduced per-site SSE;
  payload-equality, selector-sharing, and call-count tests preserve every
  endpoint and decode.
- Rate-distortion artifacts bind `fused_deployable_full_view_packet_v2` under
  evaluation schema v2. One paired stream, one deployable full-view encode,
  one packet-event construction, and `ceil(Q/2)` trusted decodes must produce
  selector/shared-code, transformed, and raw endpoints; the first event stream
  must also feed the independent public packet roundtrip. Codec payload equality,
  sequence/bootstrap order, row and persisted-view mismatch refusal,
  normalization modes, zero support, q order/tails, padding/bias, packet
  corruption, and CUDA drift use the gates and benchmark in `design.md`.
  Phase-3 normalization and Phase-2 persisted-view validation execute on CUDA.
  Estimator v20 prices the complete fused lifetime and dedicated-stream device
  lookahead without a traversal credit. Training and ordinary metric iterators
  use the same ordered event-bound transfer pipeline. All three calibration
  traversals use separately closed instances of that host/CUDA prefetch
  pipeline; exception and early-close tests require producer and stream cleanup.
- Decoded energy has an explicit serialized implementation identity. The
  bounded code-norm kernel is admitted only for decoded-energy scoring on an
  unfactorized Gram/QR Stiefel decoder, a hard token- or batch-TopK selector,
  and retraction after every update; every other carrier uses the exact
  decoder-Gram quadratic. Root, smoke, and child materialization recompute this
  identity from their final decisions. The fast frozen-score geometry contains
  no selector Gram, while exact fp64 sharing/concordance Grams remain present.
- The bounded implementation is released only after the fp32/bf16 score,
  support, loss, gradient, and 25-step state gates in `design.md`. Master and
  forward-copy Gram residuals fail closed at initialization before any
  score-consuming calibration, logged diagnostics, checkpoint save/resume,
  trained-model load, and deployable-codec load. Checkpoint model/train configs
  must match their run binding, and outer deployable and nested codec model
  configs must match exactly. Finite off-manifold states, missing identities,
  ineligible cadence/configuration, and rehashed configuration forgeries are
  refusal fixtures. Estimator v20 credits only the four bounded selector buffers
  and score Gram actually removed; explicit exact mode, sparse evaluation, and
  fp64 sharing geometry receive no such credit.
- Exact TopK cutoff ties retain the lowest block index within each token or
  lowest row-major event index batch-wide. Scalar-coordinate AuxK uses row-major
  `(block, coordinate)` order within each token. Zero/ReLU and BF16 cutoff-tie
  fixtures pass on every supported device and any undeclared tie policy is
  refused.
- Code norm has a separate serialized execution identity. Large contiguous
  no-grad bf16 CUDA pools of block width four bind the exact `sqrt.rn` kernel;
  the identity binds both the 4,194,304-output admission threshold and the
  8,388,608-output 256-to-128-element tile switch. Every other carrier and the
  explicit native identity use
  `torch.linalg.vector_norm`. Random, extreme-scale, signed-zero, Inf/NaN,
  TopK-support, complete fallback, and multi-step master/forward/optimizer
  equality gates must pass before release. Missing or unknown identities
  refuse at model, cell, checkpoint, and frozen-artifact boundaries.
- Stiefel QR and symmetric-polar retractions satisfy their declared Gram
  invariant and produce finite gradients. Canonical QR, polar, and other
  carriers respectively bind `cholesky_qr1_positive_diagonal_cond64_v1`,
  `symmetric_polar_site_bmm_guard_g1024_w8192_c512_f2_r1e-4_v2`, and `not_applicable_v1`;
  root, smoke, and child cells rederive the identity. Polar v1 and
  positive-diagonal Householder QR remain reference/test oracles. Unknown or
  mismatched identities, condition above 64,
  nonfinite state, factorization failure, or residual failure refuse without
  fallback, and all serialized artifact identities must agree. Primitive,
  complete-Trainer, 20-step trajectory, exact-resume, both-hard-selector, and
  code-norm/decoded-energy gates use the fixed bounds reported in `design.md`.
  The Trainer may pass its immediately preceding global finite result into QR;
  QR's factor and post-Gram guards must still refuse injected nonfinite input
  or candidates transactionally, and no other decoder projection may reuse
  that specialization. Canonical QR may speculatively compute a private
  candidate and combine its precondition, factor, rank, and post-Gram
  predicates into one admitted-path host fence; every failure-only diagnostic
  and the no-mutation refusal remain release gates.
  Polar v2 may enter its site-BMM Gram only above the bound workload gate and
  only while every unmutated chunk clears the bound eig-floor and spectrum
  guard; small or unsafe chunks must reproduce polar v1 exactly.
- Site-axis factorization includes an exact selected-parent carrier. Phase 1
  records full and rank `1/2/4` as nonpromotable capability evidence and
  advances the exact carrier. In Phase 2 the common free carrier must pass the
  frozen absolute noninferiority tolerance before a free/factorized winner can
  promote; full and rank `1/2/4` free-carrier arms
  have the declared parameter shapes and fail closed for rectangular sites,
  tied encoders, constrained decoders, or
  constrained untied encoders. Its optimizer roles and codec round trip remain
  operational.
- Factorized cells derive and serialize
  `direct_rank_space_sparse_topk_cuda_v3`; rank `1/2` map- or decoder-nuclear
  cells instead derive
  `direct_rank_space_sparse_topk_cuda_factor_regularizers_v4`; unfactorized
  cells derive
  `not_applicable_v1`, and `materialized_prepacked_core_reference_v2` is an
  explicit release oracle only. Canonical rank-space encode, every score
  geometry, threshold fitting, and decode must not materialize a full site
  tensor. Encoder and decoder cores use their declared contiguous physical
  layouts; stale v1/v2 identities refuse. Low-density bf16 hard-TopK decode
  uses the content-bound Triton forward/backward only at batch size at least
  2,048 and density at most `1/32`; every other factorized execution retains
  dense rank-space decode. Unknown or carrier/objective-incompatible identities
  refuse in cells, checkpoints, run bindings, and codecs. Version 4 must
  contract masked site/core pair Grams without materializing either structured
  weight; its fp32/bf16 value, every factor gradient, exact-zero,
  rank-deficient, padded, ratio-calibration, nonzero-step, and exact-resume
  gates use the bounds in `design.md`. Checkpoint loading validates every
  non-scalar Adam moment against
  the fresh parameter shape before PyTorch may accept positional optimizer
  state. Rank `1/2/4`, fp32/bf16, masking/fusion, padding/bias,
  selector/score, forward/backward, exact-resume, and paired-trajectory gates
  use the fixed bounds and RTX 4090 evidence in `design.md`. Estimator v20
  remains conservative and grants no runtime credit for this optimization.
  Unfactorized map-nuclear cells additionally bind
  `batched_site_gram_reference_guard_d1e-3_e1e-4_v1`; the old site-reducing
  einsum is the mandatory whole-call fallback and explicit oracle. Its
  value/gradient, rank-deficient, near-singular refusal, branch, trajectory,
  checkpoint, and exact-resume gates use the fixed bounds in `design.md`.
  Every carrier also serializes
  `native_or_rank_hard_topk_cuda_tile256_else_dense_v2` or the explicit
  `dense_reference_v1` sparse-decode oracle. The CUDA identity may consume
  native contiguous `[S,G,b,d]` directly but must never pack the full decoder;
  token TopK must derive its exact constant-count row pointer without a
  `bincount`/cumulative-sum pass, while BatchTopK retains the generic path;
  its primitive, tied/untied trajectory, support, bias, padding, and
  determinism bounds are the release gates in `design.md`.
  Codec fitting, packet encoding, public packet decode, and trusted multi-q
  decode must also remain in rank space. R-D metric D2H uses one packed
  transfer per batch, and fixed one-token rows must never read CUDA scalars
  inside the per-sequence loop. Paired R-D CUDA prefetch selects floating
  activation leaves only: row identities remain on CPU, retain their original
  storage through both host and CUDA prefetch, and receive neither pinning nor
  a CUDA stream record. Mixed nested-leaf and invalid-policy tests enforce the
  default and selected-copy contracts. Included support mask/count/index
  tensors are constructed once per batch and shared by rate and packet paths;
  an explicit duplicated-support oracle must produce the identical payload.
  Shared-code concordance constructs one intersection per block and derives
  union counts by exact inclusion-exclusion; legacy full-mask AND/OR oracles,
  511/512/513-group boundaries, short batches, and complete payloads must agree.
- Phase-1 masking has a preceding scale-control panel: literal sum at `p=0`,
  nonpromotable literal sum at `p=.10`, and availability-rescaled sum at
  `p=.10`. Its fixed rescaled carrier enters the subsequent
  probability/structured-mask panel and the universal transfer. Phase 2
  inherits that fusion rule and has no fusion-tuning round, so changing the
  number of visible sites cannot masquerade as a masking benefit through
  encoder-score scale alone.
- Clean-target site masking consumes no RNG at Bernoulli `p=0`, implements
  exactly-one-hidden and exactly-one-retained as fixed-cardinality draws, removes only truly
  observed encoder sites, preserves at least one available site per row, and
  leaves every clean reconstruction/Aux target visible to the loss. Source-only
  fusion with positive masking is refused, and checkpoint resume reproduces
  both mask draws and factorized parameters.
- Decoded-energy scoring equals the norm of the isolated multi-site decoded
  contribution, is invariant to reciprocal within-block encoder/decoder gauge,
  and reduces to code norm under the exact concatenated-Stiefel gauge. The
  Phase-1 score panel runs code norm, decoded energy, and isolated loss decrease
  both on that Stiefel equality control and on one common free decoder; all
  free-decoder arms are nonpromotable and Stiefel decoded energy is the fixed
  carrier.
- Isolated-loss-decrease scoring equals
  `2 <x_O, D_g,O^T z_g> - ||D_g,O^T z_g||^2` on observed sites only. It
  preserves harmful negative scores, excludes hidden clean targets, rejects
  decoder bias and nonquadratic reconstruction, is invariant to invertible
  reciprocal within-block gauge, and reduces to squared code norm for the
  unit-scale tied concatenated-Stiefel carrier. Exact and signed-streaming
  threshold calibration agree within the frozen support-rate tolerance.
- Isolated-loss decrease has an explicit serialized contraction identity. The
  mapped quadratic is admitted only for the free, bias-free quadratic carrier;
  every ineligible declaration is refused rather than falling back. Direct
  contribution oracles and paired exact/mapped gates cover full, partial,
  source-only, padded, factorized, fp32, bf16, threshold-calibration, support,
  loss, and gradient paths at the fixed bounds in `design.md`. The dominant
  all-observed route constructs one all-site Gram directly; partial/source
  routes retain exact site masking. Any predicate, contraction, or bound change
  requires a new clean implementation identity and campaign-shape benchmark.
- Synthetic subspace and aligned-code metrics use the same block chosen by
  support association.
- Synthetic map-rank, site-span, frequency, and coactivation axes alter only
  their named truth field, preserve replay and exact fixed-cardinality support,
  and serialize the realized ranks, active sites, inclusion probabilities, and
  pair groups. The standardized Student-t df=3 amplitude arm preserves unit
  marginal variance, and the paired-overlap arm records exact factor pairs,
  target 30-degree principal angles, and realized angles; both are
  confirmation-only one-delta stresses. One-site-span cells are ineligible as evidence for a shared
  cross-layer feature. Independent-map cells remain eligible when coordinates
  are shared across multiple sites; they falsify low-rank site-factorization,
  not cross-layer feature existence. Real-model cells bind explicit
  `not_applicable` sentinels for every synthetic-only truth field.
- The removed raw decoder site-profile concentration penalty fails closed. It
  was gauge-dependent and directionally favored layer concentration rather
  than the shared cross-layer object; any future smoothness proposal requires
  an invariant energy/Gram definition and separate smooth/step truth controls.
- Dense L1 training remains dense; deployment threshold calibration is
  recorded as a separate codec operation.
- Learned group-threshold training support is the nonzero post-shrinkage code,
  not `endpoint_score > 0`. In particular, an inherited signed isolated-loss
  score cannot add an undeclared hard gate before the calibrated deployment
  threshold is fitted. The Phase-2 group-threshold round changes the complete
  affine encoder/bias, activation, decoder constraint, L2,1, and schedule bundle
  at three coefficients; it is not labeled as a selector-only effect.
- Ratio-calibrated regularization runs after all declared initialization and
  encoder-scale fitting but before optimizer construction, on the hash-bound
  first training batch with true observation masks and fp32 clean targets.
  Its raw losses, target, resolved coefficient, achieved ratio, and input
  digest are identical in the checkpoint binding and training report; exact
  resume refuses any drift. The exact zero-smoothing map nuclear path has a
  finite-gradient repeated-Gram test. A rank-`1/2` factorized map cell must use
  the v4 factor-Gram implementation even when the target ratio is zero, because
  calibration temporarily evaluates the unweighted objective.
- Every decoder-only nuclear-norm cell is schema-forced diagnostic and
  nonpromotable.
- Decoder-weighted BatchTopK ranks the scaled candidate but decodes the unscaled
  activation.
- Token-horizon deadness counts accepted token presentations, not steps,
  batches, or wall time.
- Every padded coordinate is structurally masked in values and gradients.

## 3. Data and split audit

- Model, model revision, corpus, corpus revision/config/split, tokenizer-file
  hashes, loader, hook order, context, BOS/special-token policy, packing
  algorithm, dtype, and row-identity schema all match the cell.
- All hooks are captured from the same model forward and the same immutable
  token rows.
- Whole packed sequences belong to only one of normalization-fit, calibration,
  train, development/confirmation, production stability, or final roles.
- Phase-1 factor association/alignment, codec calibration, development, and
  confirmation each consume their own ordered disjoint identity range whose
  length equals its explicit manifest count.
- Every shard and split manifest binds content, row stream, source, transform,
  ordered sites, dimensions, dtype, count, and shard index.
- `capture.json` embeds the exact producer binding. Verification recomputes its
  canonical digest, rejects missing or extra binding fields, checks every
  duplicated source/allocation/implementation field and physical geometry, and
  requires every declared profile role. An empty store is a failed verification,
  not an empty successful report.
- The one-deep shard writer owns no more than one detached pending payload plus
  one producer staging payload; the exact padded bf16 activation and int64 row-ID
  residency estimate passes its pre-output refusal gate.
- The persistence worker audits finite values and zero rows before writing any
  bytes, mutates no live writer state, and a synchronization failure poisons the
  writer while `close`/`abort` still joins its executor.
- Capture progress advances only from the post-manifest-fsync durable callback;
  crash/resume accepts at most one verified next-shard orphan and reproduces
  uninterrupted row order and stream hashes.
- Physical schema v3 is named `activation-store-v3-derived-views` in Phase 2
  and `activation-store-v3-single-view` in Phase 3 across cells, capture CLI,
  source manifests, and documentation; stale v2 aliases are refused.
- Derived normalization views preserve the raw row-stream digest exactly.
- Transform-only manifests are file-fsynced, atomically replaced, and followed
  by a parent-directory fsync; crash injection never publishes partial JSON.
- Normalization and encoder-scale statistics read only their declared fit split;
  codec thresholds/quantizers read calibration only.
- Corrupt headers, shards, transforms, row order, or hashes fail closed.
- Codec orientation uses the serialized
  `second_moment_ordered_event_frame_v2` contract. Exact-isotropic and
  near-degenerate gauge-rotation fixtures match at every priced quantizer.
  Calibration-null directions are explicitly diagnosed and have exact zero
  clip bounds; forged nonzero bounds are refused. Eigengap/cluster/null
  diagnostics are present, and an unidentified non-null frame fails closed.

## 4. Checkpoint and campaign audit

- Checkpoints are atomic and contain the complete model/optimizer/scheduler,
  retraction/dead state, data cursor, attempted/accepted tokens, and every RNG
  state needed for exact resume.
- Crash injection followed by resume reproduces uninterrupted weights, state,
  cursor, and reports.
- Campaign transitions are append-only and legal; retries cannot overwrite
  earlier attempts.
- The active plan and every registered cell are derived from the initial
  journal registrations plus the ordered `plan_extension` chain and immutable
  plan histories. `plan.json`, `cell.json`, and `state.json` are repairable
  projections, never authority. A partially journaled initial registration or
  a child registered before its extension commit cannot run.
- Atomic JSON publication fsyncs both file data and the replaced directory
  entry. Every journal append fsyncs the journal fd; a possible first creator
  additionally fsyncs the campaign root, so the authoritative commit cannot
  survive without its directory entry while ordinary appends avoid redundant
  directory flushes.
- Immutable create-if-absent JSON, Torch, codec, and whitener artifacts use an
  exclusive hard-link commit, so concurrent writers cannot replace a winner.
  Recursive durable directory creation classifies an existing target from one
  `stat` result and treats an atomic `mkdir` collision as a concurrent success
  only under the declared `exist_ok`/parent semantics; separate existence/type
  probes may not manufacture a false file collision.
  Whiteners additionally require the sole current artifact/content schemas,
  typed tensor digests, CPU fp32 dtype, canonical shapes, finite values, and
  strict JSON metadata; dtype-coercing hash aliases and legacy payloads refuse.
  A torn final journal fragment is ignored by readers and repaired under the
  append lock: a complete record missing only its newline is preserved;
  otherwise the discarded bytes and SHA256 are recorded before the next event.
- A cell lock holds a never-unlinked advisory-lock guard for its whole lifetime
  and publishes heartbeat/worker ownership through a separate atomic lease.
  Reconciliation first acquires the guard nonblocking, then re-reads the stale
  lease; release waits for any in-flight heartbeat. Canonical workers run in a
  dedicated process group whose PID, PGID, and process-birth identity are bound
  into that lease and whose descendants receive TERM/KILL on shutdown or stale
  orphan reconciliation.
- Append-only is an API/filesystem discipline, not pre-freeze origin
  authentication. A writer who can replace the journal and all matching
  artifacts lies outside the in-process tamper model; freeze proves internal
  consistency of the supplied evidence, not who created its pre-freeze bytes.
  Protect the directory and preserve the frozen decision outside that boundary.
- The campaign does not garbage-collect recorded final checkpoints or stores and
  has no retention event. Archival/deletion is external; any missing recorded
  artifact fails verification.
- Qualification rehashes every prerequisite rather than trusting an artifact
  manifest's claim about itself.
- The append-only journal, not `state.json`, supplies executor prerequisites
  and frozen-parent qualifications. The runner repairs a missing/stale snapshot
  under the cell guard, so power loss after journal fsync cannot make resume
  consume an older projection or an injected artifact reference.
- Reconcile repairs missing/corrupt plan, cell, state, activation-identity, and
  unterminated-tail projections only while holding the registration/mutation/
  implementation locks in the canonical order and every active cell guard
  nonblocking. It refuses live work instead of racing a transition. A
  post-journal projection error is nonauthoritative and cannot append a
  contradictory `FAILED` event.
- Study-v3/blueprint-v5 accept only the current prelaunch schema; any older or
  noncurrent local root remains physically preserved but is not migrated or
  supported and must be replaced with a fresh root. Preparation-v4,
  evaluation-v3, and qualification-v4 bind exact versioned shapes, the
  exact eleven integrity checks, exact eight scientific checks and margins,
  cell-derived profile/thresholds, all six qualification inputs, and promotion
  reasons replayed from the bound evaluation. Detached Phase-1/Phase-2 replay
  reruns the same qualification semantics and refuses opaque, dirty, uncommitted,
  or noncanonical scientific identities. A selection refuses mixed
  implementation identities across seeds.
- Registration atomically creates one immutable campaign-wide implementation
  pin before publishing plan state or workers. Preparation performs an O(1)
  comparison with that pin, so no cell can win a first-preparation race and no
  launch pays an O(cells) scan. Every non-smoke phase requires
  the canonical executor and clean committed source; custom executor modules
  are smoke-only.
- Implementation identity v2 binds the executable package bytes, exact imported
  dependency versions, Python/platform/Torch/CUDA build, numerical backend
  flags and environment, driver/cuDNN, and physical CUDA device identities.
  Git commit/dirty state is authenticated provenance; it is excluded from the
  execution digest but still gates non-smoke work.
- Every preparation is replayed against its registered cell before the
  authoritative `PREPARED` journal append. Phase 1
  must reproduce the exact deterministic generator protocols, ranges,
  evaluation stream, and fitted normalization. Real data must reproduce the
  exact model/corpus/revision/hook/site axis, whole-sequence split allocation,
  raw/view row streams, capture and transform lineage, and training-row policy.
  Its injectively framed identity admits one raw capture and one digest per view
  key; a self-consistent alternate or orphan store cannot enter merely by
  rehashing itself.
- Direct state-machine transitions have the same boundary as runner-driven
  transitions: artifact paths stay inside the campaign root, the registered
  cell projection must equal the authoritative plan cell, and each exact stage
  manifest must name the same path/hash/size records being committed.
- Train, calibrate, and evaluate admission additionally require an unjournaled
  process-local receipt for the exact executor stage. Scientific receipts can
  be issued only for the canonical executor, which validates its durable
  checkpoint/codec/evaluation outputs before emitting the manifest; this avoids
  a second concurrent checkpoint/model load in the parent while preventing a
  caller-built byte-valid intermediate artifact from poisoning append-only
  state. Custom executors remain smoke-only.
- Persistent-worker model handoffs bind the runtime-resolved model
  configuration from the immutable training report, including fitted
  regularizer coefficients, rather than reconstructing an unresolved static
  cell configuration. On 2026-07-23 the first MAP-nuclear Phase-1 anchor
  trained and calibrated at `f1efa31` but failed closed before evaluation when
  the retained-cache key compared its resolved coefficient with the declared
  zero placeholder. Fresh same-commit workers resumed all three seed artifacts
  through evaluation, then exposed a second fail-closed defect: explicitly
  inapplicable token-LayerNorm identification evidence omitted its already
  declared margin-normalization contract, while qualification correctly
  required that contract on every endpoint. The complete failed journal and
  artifacts are preserved as an aborted campaign; they are not rewritten or
  mixed with corrected code. The corrected persistent path validates the
  report's preparation/checkpoint binding, exact resolved field set, and
  reconstructible model configuration, while inapplicable identification
  carries its named reason, null metrics, and margin-normalization contract.
- Journal validation plus extension commit share one cross-process mutation
  lock, so selected evidence cannot fail between its final replay and the
  `plan_extension` append. The hot authority cursor retains and stat-revalidates
  every extension artifact on each use; changed evidence forces a full hash and
  refusal. Public event reads are deep-detached from private caches, and
  activation/cell/plan caches reset on any changed journal prefix.
- Embedded plan, blueprint, and plan-history hashes are recomputed from their
  canonical JSON evidence. Historical journal and selection-artifact digests
  that cannot be recomputed from the detached envelope are explicitly labeled
  opaque commitments requiring a separately trusted origin; they are never
  described as detached authentication.
- Detached Phase-2 evidence uses exact campaign-manifest and
  selection-universe field sets. Every panel entry exactly covers the frozen
  blueprint seeds, and confirmation rows bind the unique same-seed parent from
  the cell's immutable `selection.parent_cell_ids` while reusing the exact
  shared sharing-guard reconstruction.
- Selection and detached replay call one cutoff/tie-policy implementation, so
  any cutoff-tied candidate admitted for advancement is also replayable at
  freeze. Runner limits are strictly positive and cannot be combined with an
  explicit cell list.
- Integrity qualification, scientific outcome, and promotion eligibility are
  separate and internally consistent.
- Structurally inapplicable Phase-1 identification endpoints carry
  `applicable=false`, a named reason, and null margin/pass fields. They remain
  integrity-complete neutral diagnostics, are named under
  `scientific_outcome.inapplicable_checks`, and are never rewritten as
  catastrophic numeric or scientific-outcome failures. Detached replay rejects
  the exemption for any cell not resolved to token LayerNorm.
- Smoke cells preserve the full cell's `qualification.promotable` intent while
  `runtime.smoke` forces `promotion_eligible=false`. A uniformly smoke stage
  may select only through qualification mode `smoke_protocol_only`, without
  consuming scientific outcomes or enforcing sharing/noninferiority gates;
  the resulting artifact may feed only another smoke stage. A smoke Phase-2
  campaign may freeze a protocol panel for smoke Phase 3, but that panel cannot
  register non-smoke scientific Phase 3.
- Selection requires the complete stage seed universe and aggregates the
  frozen metric by median, then worst seed, then candidate ID.
- Selection freezes the entire eligible/ineligible universe and all metric and
  qualification hashes.
- Status reports `running` cells as resume-required rather than default-runnable;
  default run skips them, while `--resume` opts in explicitly. A matrix run with
  any failed selected cell exits nonzero.
- Phase-1 capacity, retraction, site-factorization, missing-site-fusion,
  site-masking, score, and selector stages are fixed-carrier capability panels.
  Every challenger is seed-complete and evidence-bearing but nonpromotable;
  only the named carrier may enter the next stage. Decoded energy is the fixed
  provisional Stiefel score carrier, the three parallel free-decoder score arms
  are nonpromotable, and robustness confirmation is nonselectable.
- `freeze-phase1` rederives `bsc-phase1-transfer-v3` from the complete campaign
  manifest. The transfer binds source plan/blueprint and evidence hashes,
  baseline cells, selection IDs, the hashed universal method contract, the
  hashed provisional carrier, every diagnostic capability qualification
  digest/outcome, and claim-scope narrowing. Synthetic numeric hyperparameter
  winners are absent.
- Phase-1 registration and decision replay independently rebuild the exact
  canonical blueprint and initial plan for the bound seeds/smoke mode. A
  self-consistent reduced capability matrix cannot authorize Phase 2.
- A runnable Phase-2 blueprint and `phase1_contract_bsc` anchor bind both the
  authenticated Phase-1 decision ID and transfer ID. Unbound previews, stale
  embedded transfers, or forged scope/evidence are refused. Capability failures
  remain evidence and do not prune real-model options. The Phase-2 chain has no
  observation-site/evidence-topology or fusion-tuning stage, but it explicitly
  retunes model architecture. It revisits site rank after masking, then
  runs the full three-score by two-hard-selector interaction and the separate
  bundled group-threshold method round. Its source-only BSC is a descriptive
  nonpromotable anchor.
- Every main-chain Phase-2 selected-parent/revisit selection enforces the frozen sharing
  guards for both site-only and leave-one-out inference: worst-site
  decoded-coordinate Lin concordance in the all-site decoder-Gram geometry,
  with mean-offset penalty, is at least `.80`; worst-site support-intersection
  recall is at least `.75`; and decoded-energy coverage is at least `.90`.
  Parent- and root-relative partial-view FVU, support-IoU, and absolute-FVU
  safety gates remain conjunctive. Same-candidate all-view FVU advantage is
  descriptive only and is not compared with the separately trained source-only
  anchor. The initial factorization round additionally requires its exact
  selected-parent carrier.
- Comparator-family cells report those same sharing endpoints but family
  calibration, nomination, and revisit do not require BSC sharing admission;
  an intentionally non-sharing comparator remains available for Phase 3.
- Every main-chain development round outside the specialized initial
  factorization policy declares an exact selected parent. Materialization
  retains one cell per resolved execution-value signature and records every
  parent/center duplicate it elides. A child replaces the retained parent only
  after improving the fixed-rate score by at least `0.002` on every seed and
  on the median and worst-seed aggregates; confirmation remains nonselectable.
- Panel freeze consumes only the untouched scalar-RMS confirmation rerun. Each
  seed must re-pass scientific qualification and the sharing guard and remain
  within `0.02` fixed-rate score of its exact development parent. The cells
  label this as a novel reproducibility rule and bind the complete
  `.01/.02/.05` marginal sensitivity surface plus the ungated result.
- Phase-2 factorization first requires the full free-site carrier to remain
  within `0.01` of the exact selected parent, then advances the lowest of rank
  `1`, rank `2`, rank `4`, and full that remains within `0.01` of full on every
  seed and on the median/worst aggregates.
- The post-mask factorization revisit uses the ordinary minimum-effect policy.
  If zero Bernoulli masking wins, it emits only the exact parent and records the
  conditional elision of the four rank children.
- Warmup is serialized as a fraction of accepted optimizer updates and is
  recomputed exactly for each batch/token budget. The `.02/.05/.10` warmup
  round precedes the schedule round, and the peak learning-rate ladder is
  rerun after batch, warmup, and schedule are fixed; source-step warmups remain
  confined to exact paper anchors.
- Each of the seven Phase-3 comparator families has its own content-addressed
  root selection, conditional calibration rounds, and fresh winner/runner-up
  revisit. Counts are derived from the serialized blueprint, and reports state
  that staged ordering does not prove a global optimum.
- Every comparator-family learning-rate round has exactly four arms:
  `3e-5`, `1e-4`, `2e-4`, and `3e-4`.
- Default counts are rederived as 195 declared/executed Phase-1 cells at three
  seeds and a 410-cell Phase-2 pre-elision ceiling at two seeds: 176 main-chain
  plus 234 family-chain. Phase-2 reports separately record the smaller realized
  count after execution-signature, conditional rank-revisit, and conditional
  non-token-TopK Appendix-runner-up elision.
- Revisit nominations rank the union of every qualified 4M candidate in that
  family, deduplicate seed-independent resolved execution signatures while
  preserving aliases and metric spread before outcome ranking, retain exactly
  two distinct configurations under one nomination policy/universe hash, rerun both at 16M, and then retain one
  comparator. Stage winners alone are not mistaken for the complete runner-up
  pool.
- Comparator branches coexist in one append-only Phase-2 DAG. Every family
  stage gate, selection, and cell lineage names its exact branch-parent stage
  and cell IDs; journal adjacency cannot substitute for scientific parentage.
- Phase-3 panel production verifies the complete Phase-2 plan/blueprint,
  selection chain, confirmation evidence, independently calibrated derived
  comparator source cells and family/root lineage, and
  full source-manifest hash. Registration reconstructs and exactly matches
  both the panel-bound Phase-3 plan and panel-bound blueprint; an unbound
  preview or stale comparator lineage is refused.
- Static panel replay runs the same deterministic selection reducer as the
  live campaign over every embedded qualification. It therefore reconstructs
  the complete ranked and excluded populations, exclusion reasons, scientific
  gates, aggregate/order, sharing lineage, and threshold-sensitivity report;
  family nominations likewise replay every source round, canonical
  execution-signature representative and alias spread before ranking. An
  ordinary adaptive chain must name rank one. The sole exception is a
  producer-verified `next_ranked_nonduplicate` comparator substitution, whose
  collision fingerprint and exact first noncolliding rank are replayed.
  Rehashing a runner-up, moving the true winner into the excluded set, or
  forging a nomination metric is refused.
- Non-smoke Phase 3 requires the exact preregistered seed tuple
  `(0,1,2,3,4)`. Caller-supplied production seed reductions cannot turn the
  48-cell publishable panel into a smaller canonical campaign.
- Phase 2 uses its declared bf16 forward precision but has no matrix-level
  fp32/bf16 parity claim. The executable parity-and-short-run stability gate is
  Phase-3-only.
- Phase-3 slots serialize duplicate handling. Scientific projection uses all
  five required production seeds before fingerprinting the seed-zero member
  of each design. The selected-finalist slot fails
  on duplication; comparator slots advance only within their already frozen
  ranked qualified universe to the next nonduplicate.
- All eight exact production-shape designs must qualify a dedicated
  262,144-token stability cell before the final stage opens. The gate records
  fp32/bf16 reconstruction relative error `<= .05`, support IoU `>= .90`,
  finite forward state, exact short-run completion, and use of the disjoint
  `stability` split. It also verifies the exact fourfold active-coordinate
  transfer from pilot budgets to `1024/1536/2048` bits/token, a nonzero packet
  endpoint at every budget, and at least two distinct nonzero frontier
  endpoints. It is conjunctive and non-ranking; final data cannot be read to
  adjust it.

## 5. Codec and metric audit

- Calibration serializes a deployable codec and never mutates the checkpoint.
- Evaluation reloads the saved checkpoint and codec and performs source-free
  integer packet encode/decode.
- Packet count width covers every legal count; IDs are compact transmitted IDs,
  and the serialized rank-to-block table maps them to dictionary rows.
- Rate prices count, block IDs, amplitudes, dense payloads if any, and every
  byte of deployable side information.
- The zero-event endpoint, excluded calibration/eval shares, quantizer points,
  and bootstrap sequence units are present and finite.
- Raw FVU uses the paired raw row stream. An inverse needing unpriced per-token
  state is oracle and ineligible.
- Fixed-rate scoring removes dominated points, uses only the measured lower
  convex envelope, and never extrapolates below the envelope.
- Lower-envelope time sharing uses the serialized
  `balanced_global_token_counter_u64_v1` global schedule and prices its fixed
  32-byte header; no unpriced per-token mixture bit is permitted. Exact records
  live in the immutable, hash-bound `deployment_schedules` artifact, are
  reloaded through the consumer path, and are executed on paired raw evaluation
  rows, so distortion is not inferred from an analytic blend of aggregate
  endpoint FVUs. The bundle is an audit container for mutually exclusive
  budgets; an exact deployment ships and prices one selected 32-byte record,
  not every alternative record or the review manifest.
- Phase 1 checks native and deployed support association, same-block subspace,
  isolated-input guard, same-block aligned code, precision/recall, deadness,
  split, merge, and nonfinite thresholds.
- Width-below-truth-rank cells report their exact same-block information
  ceiling and retain the raw gate. Their calibration-frozen minimum-group
  support/subspace/code companion is reporting-only and cannot authorize
  promotion.
- Pathology association uses content-bound primary `.50/.25` strong/weak
  cutoffs and the complete declared 3-by-3 reporting-only sensitivity grid;
  qualification verifies both the cell contract and emitted evidence.
- Encoder-scale fitting measures postactivation block norm independently of
  selector geometry, replays a positive bracketed solver, and qualifies only
  a remeasured `1.0 +/- .001` result. One-shot calibration refuses Group-Lasso
  shrinkage and signed isolated-loss scores.
- Phase 2 checks the exact mean score over 256, 384, and 512 total bits/token
  and records the complete zero/2/4/6/8/12/16-bit rate-distortion surface even
  though only the frozen scalar aggregate enters selection.
- Development evidence freezes one content-addressed lower-envelope endpoint
  identity pair at every exact budget. Confirmation and Phase 3 replay the
  deterministic worst-source-seed policy and cannot use holdout/final
  distortion to choose new endpoints. Every eligible budget serializes and
  prices its fixed 32-byte operating record, including a pure endpoint.
- Functional-dependence profiles are present before and after selection, but
  their coherence sum is descriptive and has no hard-coded monotone preference.
- All-site and every site-only-to-all-site endpoints are emitted for masking
  rounds; masked training does not alter the clean target or waive fixed-rate
  codec accounting.

## 6. Resource audit

- Estimates distinguish unique rows, optimizer-token presentations, model
  parameters, checkpoint bytes, activation-store bytes, and compute FLOPs.
- Resource-estimator schema `dense-linear-memory-v20-e8cd28faf7b38d6e64f0426000de174679f4c01413ec6647fa6b997219978e55` binds peak training VRAM
  and peak host RAM in addition to persistent storage and aggregate compute;
  estimates are finite, nonnegative, and monotone under the declared scaling
  checks. Cholesky-QR1 reserves
  `4 * (sites * padded_site_width * groups * block_width + 6 * groups *
  block_width^2)` bytes of training workspace and receives no speculative
  speed or FLOP credit. Direct jobe peak probes at the Phase-2 and Phase-3
  geometries remain below those respective estimates after warmup; the roughly
  8 MiB cold Inductor finite-check compile is covered by the separate fixed
  2 GiB CUDA/PyTorch context and allocator allowance.
- Fused evaluation releases all shared-view carriers before the threshold
  packet callback and runs that callback in exact ordered 4,096-token
  microbatches. Estimator v20 adds the still-live full threshold
  `z`/score/mask carrier to the bounded R-D workspace rather than taking their
  maximum; the worst current Phase-3 scalar geometry projects
  `19,212,206,256` bytes and remains below the 22 GB refusal gate.
- The planner refuses a declared budget violation before registration.
- Local filesystem planning checks live free space unless explicitly overridden
  for planning only.
- Launch repeats the live storage check without accepting the planning override.
  It first replays the current phase resource ceilings against the complete
  loaded plan, so an older registration cannot bypass a tightened estimator or
  ceiling.
  Input credit is bound to the materialized plan's capture/allocation/view
  contract; stateless Phase 1 receives zero input-store credit, unrelated stores
  receive none, and only verified recorded campaign artifacts receive resume
  credit. Requirements are grouped by destination device, same-device roles are
  summed, and cross-device free space is never aggregated. An explicit
  `--view-root` is part of this binding. If inputs are not yet complete, their
  remainder is conservatively required on every declared input destination;
  scientific launch normally has zero such remainder, while planning alone may
  record the existing override.
- Capture, derive, and transform fitting run destination-device prewrite gates
  before their corresponding payload writes and retain the writer's 15% free
  space floor. Reports distinguish raw free bytes, the reserved floor, and
  usable bytes above the floor.
- Phase 3 remains within the frozen 4,002,097,152 optimizer-token
  (4B final plus eight short preflights), 400M-parameter,
  22GB peak-VRAM, 55GB peak-host-RAM, 850GB-storage, and one-week conservative
  compute ceilings.
- GPU execution is sequential; evaluation does not load another checkpoint
  alongside a training job on the 24GB device. A host-global per-device advisory
  lock serializes canonical CUDA workers even across distinct campaigns.
  SIGTERM during matrix dispatch unwinds the runner, closes its isolated worker
  process group, and restores the caller's prior signal handler.
- Every materialized derived mode has one exact-key root manifest embedding an
  authenticated capture contract and binding its transform plus complete split,
  content, and row streams. Standalone verification rejects a subset, foreign
  entry, rewritten capture binding, or divergent split.
- Persistent store-verification receipts live only below the authenticated
  campaign root. Arbitrary cache-root overrides are refused; reuse requires both
  an exact inode/device/size/mtime/ctime fingerprint and the exact row-allocation
  contract. A changed fingerprint forces complete checksum and row replay; the
  verifier refuses any before/after fingerprint change. The initial full pass
  also records deterministic shard probes, but unchanged receipt reuse is a
  metadata fast path and is not claimed as perpetual whole-store authentication.

## 7. Required review loop

Perform independent adversarial passes after implementation stabilizes:

1. **paper audit:** equations, settings, disclosure gaps, and release drift;
2. **scientific audit:** estimand validity, leakage, metric gaming, rate and
   compute fairness, seeds, and claims;
3. **code audit:** unreachable branches, malformed manifests, state-machine
   bypasses, corrupt artifacts, packet edge cases, resume, and resource math;
4. **fresh final audit:** review the complete diff and smoke evidence rather
   than relying on earlier partial reviews.

Every finding is closed in code/tests/docs or recorded as launch-blocking. A
review that merely agrees with the design is not an adversarial pass.

Subprocess verification must prepend the exact source tree under review rather
than silently importing an editable checkout elsewhere on the host. A test that
intentionally triggers a fatal CUDA device assertion runs in its own child
process: the production fail-closed asynchronous path remains synchronization-
free, while the shared full-suite CUDA context remains usable for later tests.

## 8. Minimum local verification

```bash
python -m pytest -q
python -m compileall -q block_crosscoder_experiment
git diff --check
bsc --help
bsc matrix --help
bsc data --help
bsc cell --help
```

Also run at least one schema-complete CPU campaign through prepare, train,
calibrate, evaluate, and qualify; advance every smoke blueprint stage through
its explicit `smoke_protocol_only` selection artifact; exercise the real-store
path with a tiny content-bound store; assert that a smoke Phase-2 panel can
register only a smoke Phase-3 protocol campaign and is refused by non-smoke
Phase 3; and deliberately corrupt each artifact class to confirm refusal.
