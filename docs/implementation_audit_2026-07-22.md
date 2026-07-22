# Implementation preflight audit — 2026-07-22

Status: first- and second-pass remediation complete; definitive verification is
recorded below. No scientific optimizer training had started when either audit
was performed. The registered Jobe Phase-1 root contained only planned and
prepared cells, so every finding below is a preflight defect rather than
evidence contamination.

Audited revision: `d84e19b6b59c4b4fbd2019066d5d1a3521b0d61d`.

## Method

The audit combined three independent implementation reviews—campaign integrity,
scientific kernels, and execution/data contracts—with live local and Jobe
verification. Findings were admitted only after a source trace and a concrete
reproduction or constructive counterexample. A separate Claude/Opus review was
attempted but unavailable because its OAuth session had expired.

Baseline verification at the audited revision:

- Mac: 787 passed, 80 skipped; Ruff, compileall, `pip check`, and diff checks
  passed.
- Jobe RTX 4090: 1,191 passed, 4 skipped.
- A fresh schema-complete smoke cell reached qualification.
- The tracked worktree was clean.

Green tests are not exculpatory for the findings below: the most severe defects
crossed otherwise-valid artifact boundaries or were semantic mismatches not
covered by the existing fixtures.

## Launch-blocking findings

### A1. Study-schema evolution broke the registered Phase-1 campaign

Severity: launch blocker. Remediation status: verified.

`SCHEMA_VERSION` remained `bsc-study-v1` after seven new
`implementation.*` decisions became mandatory. Old v1 cells therefore passed
the nominal schema check and then failed reconstruction under the new required
field set. On clean Jobe revision `d84e19b6`,

```text
bsc matrix status --root /data/runs/bsc-phase1
error: cell 'phase1.paper_anchors.bsf_vanilla_primary.s0' has unresolved
decisions: [seven implementation.* decisions]
```

Status, reconciliation, resume, and freeze were unavailable. The repair must
introduce an honest new schema and an explicit migration or purpose-built
incompatibility error. Silent defaults are forbidden because they change cell,
candidate, plan, journal, and artifact identities.

### A2. Detached Phase-1 and Phase-2 decisions did not replay qualification semantics

Severity: scientific authorization blocker. Remediation status: verified.

The live qualification gate enforced `qualified: true`, complete all-true
integrity checks, scientific-outcome consistency, eligibility, and exact input
artifact bindings. Detached Phase-1 and Phase-2 decision parsers checked the
qualification byte hash, schema, cell ID, and selection-metric hash without
replaying those semantics.

A required embedded qualification was changed to:

```python
qualification["qualified"] = False
qualification["checks"] = {"integrity": False}
qualification["inputs"] = {}
```

After consistently recomputing the qualification hash, campaign-manifest hash,
transfer, and decision ID, the Phase-1 parser accepted the envelope as `go`.
The analogous Phase-2 panel mutation also parsed successfully. This is distinct
from the documented lack of an origin signature: an internally contradictory
envelope passed the claimed internal-consistency check.

### A3. Selection could aggregate seeds produced by different implementations

Severity: scientific reproducibility blocker. Remediation status: verified.

Source and partial dependency identity was stored in preparation and enforced
between stages of one cell, but it was absent from qualification and selection.
Seed 0 could therefore qualify under commit A and seed 1 under commit B, after
which the reducer could pool their median and worst-seed metrics. Phase 1 and
Phase 2 also permitted dirty source; Phase 3 required each cell to be clean but
did not require the same clean commit across cells. The recorded dependency set
covered only NumPy, safetensors, and Torch rather than the complete runtime
environment.

### A4. Auxiliary TopK violated the universal deterministic tie rule

Severity: current-protocol scientific blocker. Remediation status: verified.

The content-bound rule retains the lowest declared candidate indices at exact
cutoff ties. Main selectors implemented that rule, while SASA release AuxK,
decoder-weighted token-horizon AuxK, Fel AuxK, and the remaining SASA/long-
horizon arms called raw `torch.topk`.

A representative BF16 probe with `B=128`, `G=4096`, `b=8`, and `k_aux=256`
had exact cutoff ties in 128/128 rows. Raw and declared support differed in
107/128 rows. The repair must share the deterministic selector interior,
declare the scalar-coordinate candidate order, and test tied CPU and CUDA cases.

### A5. Capture implementation provenance was not authenticated

Severity: real-data lineage blocker. Remediation status: verified.

The consumer recomputed the source hash but only checked that
`capture_binding_sha256` was a 64-character string. It neither recomputed the
canonical binding—which includes implementation, runtime, device, geometry,
and allocation—nor validated `capture_implementation`. A capture contract with
an arbitrary implementation schema and an all-zero digest was accepted.

### A6. `bsc matrix run` returned success when cells failed

Severity: operational launch blocker. Remediation status: verified.

The runner summarized failures as `failed_cells`, but the CLI printed that
summary and exited zero. A scheduler or systemd unit could therefore report a
failed campaign pass as successful. Locked/skipped work must remain distinct
from cell execution failure.

## Scientific correctness and interpretation findings

### B1. Codec canonicalization was not gauge-invariant in repeated eigenspaces

Severity: winner-risk; current campaign impact unknown. Remediation status: verified.

The codec used the active-code second-moment eigenvectors plus active-mean signs
as a canonical orientation. Repeated eigenvalues leave an arbitrary basis in
their eigenspace. For one two-coordinate block with calibration support
`{+e1, -e1, +e2, -e2}`, identity and 45-degree gauge representations produced
identical unquantized reconstructions but different deployed distortion:

```text
q=2 FVU: 0.111111 versus approximately 0
q=4 FVU: 0.004444 versus approximately 0
q=8 FVU: 0.0000154 versus approximately 0
```

The q=4 shift exceeds the preregistered 0.002 winner-effect threshold. The
repair must either provide an equivariant, content-bound tie-break inside
degenerate eigenspaces or make simple-spectrum eligibility and gauge-sensitivity
diagnostics explicit. Per-block eigengaps must be reported before attributing a
winner change.

### B2. `crosscoder_l1` could square decoder cost

Severity: latent algebra bug. Remediation status: verified.

The configuration allowed `crosscoder_l1` with any ReLU-compatible selection
score. The regularizer multiplied `out.scores` by decoder cost. Under
`selection_score="decoder_weighted"`, the score already contained decoder
cost, so the objective became `activation * decoder_cost**2` rather than
`activation * decoder_cost`. A constructive probe produced 144 instead of 28.
Current Anthropic cells resolve to `code_norm` and were not affected.

### B3. Token-LayerNorm metric ineligibility was reported as catastrophic failure

Severity: reporting defect. Remediation status: verified.

Token LayerNorm makes several linear recovery metrics structurally inapplicable.
The evaluator emitted absent values, but the Phase-1 gate converted them into
`-1e9` margins and a failed scientific outcome. The paper-anchor stage is
integrity-only, so this did not select a parent, but the report confused
ineligibility with empirical failure.

## Operational and hardening findings

### C1. Existing-input storage credit was not bound to the plan

Severity: hard-refusal bypass. Remediation status: verified.

Storage preflight credited any verified store named by configured environment
roots without matching it to the plan's capture/view identities. An unrelated
valid store could receive full credit against a Phase-1 plan even though Phase 1
consumes no activation store.

### C2. Launch did not repeat the storage preflight

Severity: long-run operational risk. Remediation status: verified.

Capacity was checked during plan/advance, not when `run` began. A
planning-only insufficient-storage override could therefore be followed by an
unblocked launch, and a stale free-space result could be trusted indefinitely.

### C3. Resource estimates and estimator documentation were not exact

Severity: accounting hardening. Remediation status: verified.

The estimator used requested rather than whole-sequence-rounded rows and did
not price every manifest, codec, report, nontrainable tensor, and metadata byte.
The data-side helper also used summed unequal site widths despite physical
max-width padding. Normative documentation simultaneously named estimator v14
and v16.

The initial audit's proposed 60.4 GB Phase-3 deployment-model add-on was
withdrawn: the existing 16-byte-per-parameter allowance can numerically cover
the checkpoint model, Adam moments, and deployment model copy. No Phase-3
ceiling breach was established. The remaining defect is that the estimator's
claimed scope and implemented accounting disagree.

### C4. Status and default-run semantics disagreed for `RUNNING` cells

Severity: operator-state ambiguity. Remediation status: verified.

Status counted `RUNNING` cells as runnable, while a default run skipped them
unless `--resume` was supplied. The status surface needs an explicit
resume-required count or must exclude such cells from default runnable work.

### C5. `bsc data verify` accepted an empty directory

Severity: verification false positive. Remediation status: verified.

Single-store verification returned `{}` and exit zero when the directory
contained no split manifests. It must require at least one split and, when a
capture manifest is present or a phase profile is requested, the complete
declared role set and capture binding.

### C6. Transform-only manifests were not crash-durable

Severity: artifact durability. Remediation status: verified.

The whitener was published before `transform.json`, which was written with
plain `write_text`. A crash could leave a partial manifest that a rerun then
rejected as an immutable mismatch. Publication must use atomic replacement,
file fsync, and directory fsync.

## Surfaces that held up

The audit did not find additional defects in:

- append-only journal replay, transition legality, artifact-kind gates, and
  live artifact rehashing;
- seed completeness, promotion eligibility, median/worst-seed aggregation, and
  deterministic selection ordering;
- Phase-1 conditional-Bernoulli generation, Student-t standardization, paired
  stresses, and split identity separation;
- decoded-energy and signed isolated-loss-decrease mathematics;
- observed-site masking, clean-target masking, Stiefel/QR/polar constraints,
  and guarded optimized/reference kernels;
- partial-view concordance, support recall, energy coverage, and decoder-Gram
  geometry;
- packet count/ID/amplitude accounting, executed time-sharing, raw/transformed
  row pairing, sequence bootstrap, store payload verification, and whitening.

## Remediation exit criteria

This audit closes only when every item A1–C6 has:

1. a code or explicit contract repair;
2. a regression that fails on the audited revision's reproducer;
3. updated normative documentation where behavior changes;
4. focused local verification;
5. the full Mac suite and static gates;
6. a schema-complete smoke cell; and
7. the full Jobe CUDA suite on one clean implementation identity.

The old `/data/runs/bsc-phase1` root must not be silently rewritten. Once the
new schema is final, it must either be preserved as an explicitly incompatible
preflight artifact or migrated through a separately reviewed, content-bound
procedure. A fresh campaign root is acceptable and likely preferable because
no scientific optimizer training had started.

## Remediation record

| ID | Implemented repair | Adversarial regression |
|---|---|---|
| A1 | Introduced study v2, candidate v2, and blueprint v4; legacy study-v1/blueprint-v3 manifests now fail with explicit preservation and reviewed-migration guidance. | Old plan, cell, and blueprint payloads are refused; the live Jobe preflight root produces the intended incompatibility error without mutation. |
| A2 | Qualification v2 uses one semantic validator for live gates and detached Phase-1/Phase-2 replay, including checks, outcome consistency, eligibility, exact inputs, metrics, and evaluation binding. | Consistently rehashed false qualification, failed integrity, empty inputs, outcome contradiction, forged inapplicability, and Phase-2 envelope mutations are refused. |
| A3 | Preparation v2 and qualification v2 bind the complete package-source, Git, Python, Torch/CUDA-build, and declared dependency identity; non-smoke preparation requires clean committed source; campaigns and detached decisions require one identity across seeds. | Dirty scientific preparation, sequential preparation drift, and consistently rehashed mixed seed identities are refused. |
| A4 | Every AuxK arm now uses the shared deterministic cutoff interior; scalar SASA-release order is row-major `(block, coordinate)`. | Exact BF16 cutoff ties across FEL, SASA, long-horizon, Minder, and scalar release arms retain the lowest declared indices on CPU/CUDA. |
| A5 | `capture.json` embeds the exact capture binding; producer and consumers share source/allocation derivation, recompute its digest, validate exact fields/geometry/runtime/profile roles, and match current reviewed capture code. | Arbitrary implementation metadata, zero/forged digests, reordered roles, reassigned allocation, geometry drift, and incompatible resume bindings are refused. |
| A6 | `matrix run` exits nonzero whenever any selected cell fails while preserving its complete summary. | A failing executor returns status 1; locked/skipped work remains distinct. |
| B1 | Codec frame v2 orders spectral clusters and uses gauge-equivariant ordered-event MGS. Calibration-null directions are explicitly diagnosed and forced to exact zero clip bounds; only non-null unidentified directions fail closed. | Exact isotropic, near-degenerate, and rank-deficient 45-degree gauge rotations agree at all priced quantizers; forged nonzero null bounds are refused. |
| B2 | Crosscoder L1 consumes the unscaled code activation and decoder norm directly; selector scores are detached from the objective. | Decoder-weighted selection reproduces the direct algebra rather than squaring decoder cost. |
| B3 | Token-LayerNorm identification is `applicable=false` with a named reason, null pass/margin, neutral scientific outcome, and a cell-bound inapplicability record. | Catastrophic sentinel margins are absent and a rehashed exemption on any non-layer cell is refused. |
| C1 | Existing-input credit is matched to the plan's capture, allocation, and view contract; Phase 1 always receives zero input-store credit. | Unrelated verified stores receive no credit; a matching store receives only verified physical-byte credit. |
| C2 | `matrix run` repeats the live storage preflight and never accepts the planning-only override. | Changed free space and an earlier override cannot bypass launch refusal. |
| C3 | Estimator v17 uses whole-sequence-rounded rows, physical max-width padding, explicit 16-byte parameter ownership, exact tracker storage, and a conservative schema-derived codec/report/container envelope. Documentation now names only v17 and states its conservative scope. | Rounding, unequal-width padding, fast-path eligibility, QR workspace, mapped-score geometry, monotonicity, and phase ceilings are rechecked. |
| C4 | Default runnable work excludes `RUNNING`; status reports `resume_required` and `failed_retry_required`; `--resume` opts into both states explicitly. | Running and failed cells are absent from default runnable IDs and present only under their matching opt-in. |
| C5 | Single-store verification requires a nonempty, complete, authenticated capture role set and cross-checks every split's allocation and geometry. | Empty and incomplete roots fail with nonzero CLI status. |
| C6 | Transform manifests use file fsync, atomic replacement, and parent-directory fsync. | Publication uses the atomic helper and rerun accepts the exact surviving content-addressed whitener. |

## Final verification

- Mac full suite: **811 passed, 80 skipped** in 381.18 seconds.
- Mac focused integration suite: **521 passed, 47 skipped**; data CLI suite:
  **26 passed, 2 skipped**; scientific-kernel suite after null-space hardening:
  **345 passed, 42 skipped**. Campaign replay was also run independently before
  the definitive full suite.
- Mac static/operational gates: Ruff, compileall, `pip check`,
  `git diff --check`, all CLI help surfaces, and a fresh Phase-1 schema-complete smoke
  passed. The smoke qualified one cell with zero failures and credited zero
  activation-store bytes.
- Jobe clean isolated snapshot: **1,221 passed, 4 skipped** in 487.78 seconds
  on Torch 2.8.0+cu128 / CUDA 12.8 / RTX 4090 with bf16, followed by a fresh
  schema-complete smoke that qualified one cell with zero failures and bound
  `git_dirty=false` in qualification v2.
- The real Jobe checkout and `/data/runs/bsc-phase1` were not modified. The old
  preflight root was read-only probed and refused as legacy study v1. No capture,
  optimizer training, selection, advancement, or scientific launch occurred.

All exit criteria above are satisfied for the remediated implementation. A new
scientific campaign must be registered under a fresh root after this change is
reviewed, committed, and synchronized; the old root is evidence to preserve,
not state to rewrite.

## Second adversarial pass after first-pass publication

The first remediation was committed, pushed, and synchronized to Jobe as
`25ee69637571f5f6b59d641829d669c59425ab1b`. The second pass audited that exact
committed baseline before considering any new working-tree fix. It combined
three independent campaign/science/operations reviews, a successful read-only
Claude/Opus review, direct source reconciliation, and constructive probes. The
Opus transcript was recovered from its raw event log because the wrapper's
final reply contained only a continuation marker.

Claims were admitted only after local source verification. Several reviewer
claims were rejected or narrowed; those are recorded below rather than silently
dropped. The repository is prelaunch and has no external users, so the repaired
implementation supports only the single current schema. Non-current artifacts
are refused with a fresh-root instruction; there is no migration or backward-
compatibility layer.

### Campaign authorization and lifecycle findings

| ID | Severity | Failure at `25ee696` | Implemented repair |
|---|---|---|---|
| D1 | launch-blocking | Qualification v2 required only six integrity keys and accepted arbitrary scientific-check/margin maps. A canonical executor module could emit an invented but self-consistent approval that bypassed precision and diagnostic gates. | Qualification v3 has one exact validator for eleven integrity checks, eight scientific checks, eight margins, the resolved profile and threshold map, all six inputs, implementation identity, evaluation metrics, promotion reasons, and protocol/scientific eligibility. Producer and detached replay use the same contract. |
| D2 | launch-blocking | Detached implementation identities could be opaque, dirty, uncommitted, or self-hashed with an unconstrained shape; a non-smoke custom executor could forge every stage payload. | Preparation v3 and qualification v3 require the exact versioned identity shape. Scientific cells require the canonical executor, a clean 40-hex commit, and the canonical process model. Custom executor modules are smoke-only. |
| D3 | high | Two cells could race their first preparation under disjoint cell locks and commit different implementations before either observed the other. | The first preparation creates one immutable campaign implementation pin under a campaign-wide lock, and every existing preparation is scanned before the pin/transition commits. |
| D4 | medium | Detached Phase-1 plan/blueprint/history hashes could be fabricated while unreconstructable journal and selection-file hashes were described as if they proved detached authentication. | Reconstructible hashes are recomputed. Unreconstructable historical hashes are explicitly labeled opaque commitments that require a separately trusted origin. |
| D5 | launch-blocking | Phase-3 detached replay accepted an attacker-chosen same-seed development parent for a confirmation finalist. Choosing a poor parent made the `.02` noninferiority and parent/root-relative sharing gates vacuous. | Replay now derives the unique same-seed parent from the confirmation cell's immutable `selection.parent_cell_ids` and requires exact equality with the embedded row. The shared embedded-guard implementation recomputes the complete acyclic parent/root trace. |
| D6 | launch-blocking | A detached finalist entry could omit the failing confirmation seed because comparator entries, but not the nonselectable finalist, were indirectly seed-complete. | Every frozen-panel entry must cover the reconstructed blueprint seed tuple exactly. |
| D7 | medium | Phase-2 campaign/universe objects accepted extra keys, so irrelevant fields minted distinct panel IDs over identical evidence. Separately, a policy-retained tied selection could advance but later fail replay because freeze required the first-ranked tied member. | Phase-2 manifest and universe key sets are exact. One cutoff-retention helper is used by live selection and both detached replayers, so any explicitly policy-retained cutoff tie replays consistently. |
| D8 | high | Cell locks named only the runner PID. SIGTERM or parent death could orphan a trainer, stale reconciliation could then license a second worker, and heartbeat/reconcile path races could resurrect or steal a lease. | Each cell has a stable never-unlinked guard inode held by `flock` plus atomic lease metadata. The lease binds owner and worker PID/PGID birth identities. Workers run in new sessions; close and stale reconciliation TERM/KILL the exact process group. Reconcile locks and re-reads before removing metadata, while release joins the heartbeat and serializes its final unlink. |
| D9 | operational | `limit=0` returned a successful no-op, and a limit silently truncated an explicit cell list. | Both CLI and `CampaignRunner` require a positive limit and reject limit-plus-explicit-cells. |

### Scientific and estimator findings

| ID | Severity | Failure at `25ee696` | Implemented repair |
|---|---|---|---|
| E1 | launch-blocking | Quantizer reconstruction used a `1e-12` clamped span. Exact-null coordinates reconstructed nonzero symbols as `1e-12`, and positive spans below `1e-12` reconstructed beyond their serialized high endpoint. | Every dense, packet, multi-q, and trusted decoder uses the raw serialized span for reconstruction, a safe denominator only for normalization, and symbol zero for exact-null spans. |
| E2 | launch-blocking | Group-Lasso Fel Aux ranked unselected post-shrink codes, which are exactly zero, so its auxiliary had no encoder/decoder/threshold gradient. A global keep count also silently underfilled rows with variable support. | The explicitly adapted bridge ranks and decodes the affine pre-shrink carrier, requires exactly `s_aux` unselected blocks in every row, and fails closed on insufficient capacity. Paper/source lineage and the Phase-2 family decision now name the adaptation. |
| E3 | high | Same-block factor-subspace overlap is bounded by `min(rank,b)/rank`. Rank-2 truth with scalar `b=1` can only reach the `.5` gate at a perfect boundary, so the width/control panels mixed an intended capacity negative control with an unreported structural ceiling. | Raw same-block recovery remains the primary negative-control gate rather than being waived. Evaluation now reports the exact theoretical ceiling and a calibration-frozen grouped-scalar diagnostic using exactly `ceil(rank/b)` association-selected blocks on held-out evidence, with a separate specificity companion. |
| E4 | high | Encoder-scale calibration recorded `mean_after = mean_before * multiplier` without measuring it. Group soft thresholding is not homogeneous, so live Group-Lasso cells serialized a false post-fit statistic. | The fitted statistic is the postactivation block norm, independent of selection-score geometry. A positive bracketed bisection replays the exact fit stream, checks monotonicity, remeasures every trial, and fails unless the realized mean is within `1e-3` of one in at most 32 evaluations. The training report and qualification bind the solver outcome. |
| E5 | medium | Strong/weak factor-association cutoffs `.5/.25` drove hard pathology gates but were neither decisions nor part of the threshold sensitivity surface. | Both primary cutoffs are content-bound decisions and the full strong `{.4,.5,.6}` by weak `{.2,.25,.3}` counterfactual grid is reported without changing the primary gate. |
| E6 | medium | Phase-1 margin normalization capped healthy recovered-factor fractions at `.25`, collapsing many good cells to one score. | The v2 margin contract normalizes each min-gate by its feasible range above threshold while retaining signed failure margins. The selection endpoint/profile names were bumped accordingly. |
| E7 | high | Tied encoders received a false twofold compute discount and omitted their materialized training map. Polar retraction and live nuclear-regularizer workspaces were unpriced. Registered plans were not rechecked under the current estimator at launch. | Estimator v18 counts tied execution/residency, explicit polar workspace, and regularizer cast/Gram/eigensolver lifetimes. Phase-specific budgets apply universal Jobe memory limits to Phases 1/2 and all six declared ceilings to Phase 3. The current plan is re-estimated and re-enforced at every launch. |
| E8 | medium | Phase-1 capability challengers were relabeled `protocol.hyperparameter_tuning=true` by a later stage override. A dead ReLU architecture bundle and a test-only residual-Aux executor arm remained on the executable surface. | Capability status now wins the final merge for every panel arm. Dead architecture branches and the unreachable executor mapping were removed; cell validation rejects auxiliaries absent from every live recipe. |

### Data, durability, and operator findings

| ID | Severity | Failure at `25ee696` | Implemented repair |
|---|---|---|---|
| F1 | high | Completed-capture resume omitted the embedded capture binding in its reconstructed manifest. A crash after all split manifests but before `capture.json` could therefore publish an unusable yet apparently complete capture. | Both completion paths use the same exact manifest with its full binding; complete resume is idempotent and crash-finalization is regression-tested. |
| F2 | high | Checkpoints, codecs, several JSON artifacts, and their directory entries could be renamed and journaled without file/parent fsync. A power loss could leave a durable journal pointing at missing or torn bytes. | One durable-publication primitive performs file fsync, atomic replace, and parent-directory fsync. Trainer checkpoints, final checkpoints, codecs, run-cell artifacts, campaign JSON, and whiteners use it. The journal fsyncs its parent only when first created and its file on every append. |
| F3 | medium | `bsc data verify` did not require raw split `whitener_hash == raw:<source_hash>`, canonical sequence/position identities, or `site_dims` agreement; physical storage estimation used summed widths instead of max-width padding. | Verification recomputes the exact row allocation, raw source binding, ordered sites/dimensions, and padded geometry. Alignment includes `site_dims`, and storage uses `sites * max(site_dims)` plus row IDs. |
| F4 | medium | Capture/derive/fit-transform producers could race deterministic output names. Derived views were not resumable after an interruption. | Nonblocking producer leases record PID/host ownership outside the immutable output tree, publication uses unique temporary names, and derive resume reuses only a verified complete split prefix while refusing foreign or partial state with exact cleanup guidance. |
| F5 | launch-blocking | The Phase-2 `--view-root --resume` path passed failed-cell opt-in but omitted resume-required cells, yielding a zero-work success. | View dispatch passes both resume opt-ins and has direct interrupted-cell coverage. |
| F6 | high | Storage preflight compared all bytes with the campaign-root filesystem even when stores/views lived elsewhere; explicit `--view-root` was not part of input credit; data producers had no whole-output prewrite gate. | Plan storage is decomposed into input and campaign components, credited only by verified matching artifacts, grouped by actual `st_dev`, and checked above the 15% floor without aggregating free space across devices. Explicit view roots are bound. Capture, derive, and fit-transform check their actual destination before writing and recheck only the missing remainder on resume. |
| F7 | medium | Derive/fit-transform authenticated only `source_hash`, and derived mode roots had no standalone verification manifest. | Both producers run the full capture validator. Each derived root has an exact-key `view.json` containing the complete validated source-capture evidence, canonical hashes, transform identity, ordered roles, and per-split stream identities; standalone verify replays all of them. |
| F8 | medium | A test-only environment variable could redirect executor verification receipts outside the campaign trust boundary. Persistent store receipts reused only stat fingerprints, so silent media corruption was never sampled again. | Canonical execution rejects the override and stores receipts only under the campaign root. Matrix and executor caches re-read deterministic content windows from every shard before accepting a receipt. The documentation honestly narrows this to bounded corruption detection after the initial full hash rather than claiming perpetual whole-store authentication. |
| F9 | high | Different campaign roots could run CUDA cells concurrently. Default SIGTERM killed the runner without guaranteeing worker cleanup. | A safe host/user/device `flock` covers the complete CUDA worker lifecycle, and matrix run temporarily converts SIGTERM to stack unwinding so worker-process-group cleanup executes. |

### Rejected or narrowed second-pass claims

- Phase-3 aggregate token/storage/compute ceilings were **not** imposed on
  Phases 1/2. The normative text declares those aggregate limits for Phase 3;
  applying the 4.002B-token limit to the 7.02B-token truth-known Phase 1 would
  silently invent a new protocol. Only the universal Jobe memory limits apply
  to earlier phases.
- Scalar controls were **not** declared scientifically inapplicable. Their
  capacity failure is intended evidence; the repair exposes its ceiling and a
  fair grouped diagnostic without waiving the raw negative-control gate.
- Persistent receipt probes do **not** prove the absence of arbitrary future
  bit rot. They detect metadata-changing mutations and bounded deterministic
  content corruption; the initial full hash remains the content proof.
- The campaign artifact-verification cache was not defective: it rehashes on
  each stage/process boundary and binds before/after stat fingerprints. The
  receipt issue was specific to immutable activation stores.
- Claims that current capture completion omitted its binding, that derived
  view role completeness was unchecked, that whiteners lacked fsync, and that
  producer locks were absent were rejected after reconciling against the live
  fixes already landed by the parallel operations review.
- Backward compatibility and migration were deliberately omitted at the user's
  direction: this is a prelaunch repository with no users. The sole supported
  contract is study v3, candidate v3, blueprint v5, preparation/qualification
  v3, transfer v3, and estimator v18.

### Second-pass verification

The exact required Mac integration command completed with `861 passed, 80
skipped` in 508.72 seconds. A fresh current-schema Phase-1 CPU smoke registered
17 materialized cells, selected one paper anchor, and completed prepare, train,
calibrate, evaluate, and qualify with one qualified cell, zero failures, and 23
durable journal events. The smoke used a temporary root; the registered Jobe
scientific root was not opened for mutation. Focused pre-integration evidence
was:

- operations/data: 45 passed, 2 skipped; executor hardening: 5 passed;
- executor science: 108 passed, 3 skipped;
- studies/model: 245 passed, 22 skipped;
- codec/trainer: 154 passed, 20 skipped;
- campaign authorization/lifecycle representative sets: 14/14 and 7/7;
- Ruff, compileall, and `git diff --check` green in every independently owned
  slice.

The Mac editable install exposes the `bsc` console entry point and `python -m
pip check` reports no broken requirements. The containing code revision is
published and independently rerun on Jobe before this audit is closed.

Estimator v18 default-plan checks pass at 198 Phase-1 cells, 158 realized
Phase-2 cells under the 414-cell pre-elision ceiling, and 48 Phase-3 cells.
Phase 3 estimates 21,546,492,064 peak-VRAM bytes against 22,000,000,000 and
813,332,723,968 storage bytes against 850,000,000,000.
