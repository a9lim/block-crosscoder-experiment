# Implementation preflight audit — 2026-07-22

Status: remediation complete and independently verified. No scientific
optimizer training had started when this audit was performed. The registered
Jobe Phase-1 root contained only planned and prepared cells, so every finding
below is a preflight defect rather than evidence contamination.

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
