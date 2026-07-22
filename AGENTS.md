# AGENTS.md

This repository runs a staged experiment on **block-sparse crosscoders**: one
sparse block support with signed vector coordinates shared across activation
sites in a single model. It is not a general-purpose library and it is not a
model-comparison project.

## Read first

- `docs/design.md` — normative scientific, data, metric, and execution contract;
- `docs/paper_comparison.md` — exact source bridges and live conditional matrix;
- `docs/papers/block_methods.md` and `docs/papers/crosscoders.md` — primary-source
  procedure ledgers;
- `docs/audit.md` — launch and adversarial-review gates;
- `docs/findings.md` — evidence-bound reporting format.

## Three phases

1. Phase 1 uses truth-known synthetic factors to test identification,
   paper/mechanism bridges, capability evidence, and the universal method
   contract. It does not export synthetic numeric hyperparameter winners.
2. Phase 2 uses a pinned GPT-2 Small four-hook capture and owns model-, hook-,
   scale-, optimizer-, and rate-dependent tuning on development evidence,
   followed by confirmation.
3. Phase 3 freezes one Phase-2 finalist plus seven declared comparators and
   trains the five-seed Gemma publication panel without further tuning.

Later conditional stages do not exist until a complete, hash-bound selection
artifact from the preceding stage is verified.
Phase 2 additionally requires the complete authenticated Phase-1 go/no-go and
`bsc-phase1-transfer-v3` envelope. Its blueprint binds both the Phase-1
decision ID and transfer ID; prose, a cell ID, a copied metric, or an unbound
preview cannot authorize registration.

## Evidence rules

- Every decision is `exact`, `adapted`, `engineering`, or `novel`.
- Paper, inspected release, and local adaptation are separate recipes.
- Adapted or novel scientific choices require a rationale and named ablation.
- Use one/few-factor conditional rounds, not an incoherent Cartesian product.
- Phase-1 width/capacity/activity, retraction, site-axis rank, missing-site
  fusion, site masking, score, and selector rounds are capability panels. Only
  their declared carriers advance; challenger outcomes are authenticated
  diagnostics and never prune real-model options.
- Decoded energy on the Stiefel carrier is the fixed provisional Phase-1 score.
  The three scores—code norm, decoded energy, and isolated squared-loss
  decrease—run both on the Stiefel equality control and on one common
  nonpromotable free decoder. Isolated loss decrease may use only observed
  sites, preserves signed negative gains, and requires a bias-free quadratic
  reconstruction carrier.
- Phase 2 does not retune observation-site/evidence topology or missing-site
  fusion. It does retune model architecture, site-axis rank, and masking on
  real evidence, revisits rank after masking, evaluates the complete three-
  score by two-hard-selector interaction, and tests learned group thresholding
  only as a bundled method at three coefficients. Its source-only model is a
  descriptive nonpromotable anchor, while the sharing guard compares all-view
  with partial-view inference on the same candidate.
- Phase-1 confirmation includes standardized Student-t df=3 coordinate tails
  and paired 30-degree factor-subspace overlap as separate one-delta stresses.
- A candidate must have every declared seed; aggregate candidates by median,
  then worst seed, then content ID.
- Phase 1 selection uses the worst normalized truth-identification margin. FVU
  is a guardrail, not proof of factor recovery.
- Phase 2 selection uses mean raw-space FVU at the exact 256, 384, and 512
  total-bit/token budgets, including fixed-width packet bits and amortized
  deployable-codec bytes. Use the lower convex envelope only, execute any
  selected time-sharing schedule on paired raw rows, and never extrapolate.
- Qualification records integrity-complete positive and negative results.
  Scientific-outcome pass and promotion eligibility are distinct gates.
- Smoke selections and frozen panels are protocol-only. They may drive the next
  stage only when that next stage is also smoke, including a smoke Phase 3;
  they can never authorize or feed non-smoke scientific Phase 3.
- Phase-2 sharing admission is conjunctive for both site-only and leave-one-out
  inference: worst-site decoded-coordinate Lin concordance in the all-site
  decoder-Gram geometry, with the mean-offset penalty, is at least `.80`;
  worst-site support-intersection recall is at least `.75`; decoded-energy
  coverage is at least `.90`; and the parent/root-relative partial-view FVU,
  support-IoU, and absolute-FVU safety gates pass. All-view FVU advantage is
  descriptive only.
- Winner-changing practical-effect, noninferiority, and sharing thresholds are
  novel preregistered project policies, not paper values. Applicable selection
  policies content-bind the complete threshold-sensitivity grid, and each
  scientific selection artifact reports the marginal counterfactual pass sets
  without retuning the center policy.
- Comparator-family calibration reports the same sharing endpoints but does
  not require the BSC sharing gate; otherwise a deliberately non-sharing paper
  or control method could be filtered out before the frozen comparison panel.
- Decoder norm is not specificity; decoder capacity is not used dimension;
  aggregate reconstruction is not manifold recovery.
- Development, confirmation, and final evaluation are disjoint. Never tune on
  confirmation or final evidence.
- The default Phase-1 blueprint declares and executes 198 cells at seeds 0/1/2.
  The Phase-2 blueprint has a 414-cell pre-elision ceiling at seeds 0/1: 176
  main-chain plus 238 family-chain cells. Materialization deterministically
  records and elides execution-equivalent parent/center cells; if zero
  Bernoulli masking wins, the rank revisit emits only its exact parent. Report
  the realized count, not the ceiling, as executed work.

## Code surface

All implementation lives under `block_crosscoder_experiment/`; `bsc` is the
only executable surface.

```bash
bsc matrix --help
bsc data --help
bsc cell --help
```

- `studies.py`: recipes, decisions, blueprints, selection policies, and resource
  estimates;
- `campaign.py`: append-only state machine, artifact verification, selection,
  and frozen-panel production;
- `phase1.py`: stateless truth-known generators;
- `model.py`, `trainer.py`: shared model and training kernels;
- `store.py`, `codec.py`, `evaluation.py`: immutable data, deployable codec,
  raw-space rate–distortion, and shared-code endpoints;
- `cli/`: the only command entry points.

Do not add paper-specific scripts, mutable promotion pointers, or a parallel
analysis package.

## Data and artifacts

- Capture raw activations once with whole-sequence split allocation and stable
  `(sequence, position, token_id)` identities.
- Fit normalization, encoder scale, and codec calibration only on their named
  splits. Derived views preserve the raw row stream exactly.
- Site dimensions and ordered hook names are part of every cell/store binding;
  padding is structural and masked.
- Checkpoints, stores, reports, generated references, logs, and evaluations are
  ignored local artifacts. Committed `data/` contains placeholders only.
- Every promotion is an immutable content-addressed decision artifact.
- The campaign never garbage-collects a recorded final checkpoint or store.
  Deletion or archival is an external operational action; missing recorded
  artifacts are detected by verification, and there is no retention journal
  event to wait for or claim.

## Runtime

- Use shared plain Python 3.12; no project venv and no `uv`.
- Training/capture run on `jobe`; use plain `python` there and pass
  `PIP_CONSTRAINT=~/.venv/constraints.txt` for installations.
- fp16 is forbidden in capture/store. Phase 1 uses fp32 reference execution and
  Phase 2 uses its declared bf16 forward precision. The executable fp32/bf16
  parity-and-stability preflight exists only in Phase 3.
- Only declared Adam/AdamW recipes are scientific cells. No automatic optimizer
  resolution is permitted.
- Do not load evaluation checkpoints concurrently with training on the 24 GB
  GPU.
- Recheck `/data` headroom before launch. Planner token, parameter, peak-VRAM,
  peak-host-RAM, storage, and compute ceilings are hard refusal gates.

## Verification

Before calling the campaign launch-ready, run:

```bash
python -m pytest -q
python -m compileall -q block_crosscoder_experiment
git diff --check
```

Also run a schema-complete CPU cell through prepare, train, calibrate, evaluate,
and qualify. Corrupt-artifact refusal, exact resume, selected-parent binding,
and frozen-panel forgery tests must remain green.
