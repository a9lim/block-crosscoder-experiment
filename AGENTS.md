# AGENTS.md

This repository runs a staged scientific experiment on **block-sparse
crosscoders**: one sparse block support with signed vector coordinates shared
across activation sites in a single model. It is not a general-purpose library.

## Read first

- `README.md` — experiment overview and ordinary operating workflow
- `docs/design.md` — scientific, data, metric, and execution contract
- `docs/paper_comparison.md` — exact bridges to source methods
- `docs/papers/` — primary-source procedure notes
- `docs/findings.md` — evidence-bound reporting format

## Operating model

This is a trusted-operator experiment. Plans, selections, cell states, and
phase decisions are readable JSON files. Treat them as editable working
artifacts, not authenticated records. Prefer direct names and paths over
checksums, content IDs, append-only journals, historical snapshots, redundant
bindings, or forgery defenses.

Keep only checks that can change a scientific conclusion or prevent an
accidental resource overrun. Ordinary existence checks, schema checks, split
separation, model/metric invariants, and resumable stage state are useful.
Security-style checks against deliberate local editing are not.

There should be one current plan, one current state file per cell, and one
ordinary output path per artifact. An explicit rerun may replace outputs. Git
is the history when history is needed.

## Three phases

1. **Phase 1:** truth-known synthetic identification. It tests universal method
   semantics and exports no numeric model or optimizer winner.
2. **Phase 2:** GPT-2 Small development and confirmation. It owns
   model-, hook-, scale-, optimizer-, masking-, and rate-dependent tuning.
3. **Phase 3:** one reviewed Phase-2 finalist plus six declared comparators,
   trained at five seeds on Gemma with no further tuning.

A reviewed decision JSON from the preceding phase opens the next phase. Smoke
decisions may open only smoke descendants.

## Scientific rules

- Mark decisions as `exact`, `adapted`, `engineering`, or `novel`.
- Keep paper recipes, inspected release behavior, and local adaptations
  distinct.
- Give adapted or novel scientific choices a rationale and named ablation.
- Use conditional one/few-factor rounds rather than an incoherent Cartesian
  product.
- Phase 1 fixes one rank-two factor in two dimensions, one width-two learner
  block, and one active factor. A one-site positive control precedes the
  four-site carrier; support-only and one-site-span controls accompany
  confirmation.
- Phase 2 may tune architecture, site-axis rank, masking, selectors,
  optimization, regularization, and rate choices. It does not tune on
  confirmation.
- Every candidate needs every declared seed. Aggregate by median, then worst
  seed, then readable candidate name.
- Phase 1 selects by worst normalized truth-identification margin. FVU is only
  a guardrail there.
- Phase 2 selects by mean raw-space FVU at exactly 256, 384, and 512
  total bits/token. Include packet overhead and deployable codec bytes. Execute
  time sharing on paired rows and never extrapolate.
- Qualification means the cell produced complete usable evidence. Scientific
  pass and promotion eligibility remain separate.
- BSC and comparators share the same real-model performance standard.
  Method-specific invariants confirm what ran; they do not create an extra
  hurdle for the proposed method.
- Report all-view, site-only, and leave-one-out endpoints. Missing-view
  robustness is a secondary diagnostic, not a promotion gate.
- Keep development, confirmation, and final evaluation disjoint.
- Decoder norm is not specificity; decoder capacity is not used dimension;
  aggregate reconstruction is not manifold recovery.

## Code surface

`bsc` is the only executable surface:

```bash
bsc matrix --help
bsc data --help
bsc cell --help
```

- `studies.py`: recipes, blueprints, selection policies, resource estimates
- `campaign.py`: current state, stage transitions, selection, phase handoffs
- `phase1.py`: truth-known generators
- `model.py`, `trainer.py`: model and training kernels
- `store.py`, `codec.py`, `evaluation.py`: data, deployable codec, metrics
- `cli/`: command entry points

Do not add paper-specific scripts or a parallel analysis package.

## Data and runtime

- Capture raw activations once with whole-sequence split allocation and stable
  `(sequence, position, token_id)` rows.
- Fit normalization, encoder scale, and codec calibration only on named splits.
  Derived views preserve row order exactly.
- Ordered hook names and site dimensions remain part of the data contract.
  Padding is structural and masked.
- Generated stores, checkpoints, evaluations, and logs are local and ignored.
  Commit compact evidence under `data/summary/` and figures under
  `docs/figures/`.
- Maintainer machines use shared plain Python 3.12. The reviewer install in
  `docs/reviewer_setup.md` declares every capture and CUDA dependency in one
  package command; do not send reviewers through a separate CPU environment.
- Run maintainer training and capture on `jobe` when it is available. The
  reviewer replication command may use any compatible Linux/NVIDIA device.
- Phase 1 uses fp32. Real-model capture uses its declared bf16 precision; fp16
  stores are unsupported.
- Recheck storage and memory headroom before a large run. Resource ceilings are
  hard refusal gates.

## Verification

Match verification to the change. Prefer focused checks of model math, data
splits, selection, ordinary resume, and the touched CLI surface. Reviewer-path
changes should validate the resolved formula, install surface, and artifact
loader; actual CUDA execution is authoritative when compatible hardware is
available. Do not substitute a CPU tour for that claim. Use `compileall` and
`git diff --check` before publication.
