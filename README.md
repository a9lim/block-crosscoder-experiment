# Block-sparse crosscoder experiment

This repository tests one question:

> Can a block-sparse crosscoder recover features whose signed vector
> coordinates are shared across several activation sites in one model?

The proposed representation selects a sparse set of blocks, then uses the
same coordinates for each selected block at every observation site. Each site
has its own decoder, so a shared latent feature can still render differently
at different layers.

This is an experiment repository, not a general-purpose library. The workflow
is intentionally simple: plans and selections are readable JSON, artifacts
are ordinary files, and the operator is trusted to review or edit them.

## Experimental shape

The work is staged so synthetic identification, real-model tuning, and final
evaluation do not leak into one another.

1. **Phase 1 — truth-known synthetic identification.** A rank-two factor is
   rendered through independent site dictionaries. The experiment asks whether
   the learner recovers the shared block and its signed coordinates. One-site
   positive controls and support/span negative controls make the result
   interpretable.
2. **Phase 2 — GPT-2 Small development.** Architecture, rank, masking,
   selection score, optimizer, and rate-dependent choices are tuned on four
   activation sites. Confirmation data is held out from those choices.
3. **Phase 3 — Gemma final panel.** One Phase-2 finalist and six declared
   comparators are trained at five seeds with no further tuning.

A later phase starts from a reviewed JSON decision produced by the preceding
phase. Those files are handoff notes, not security credentials.

## What is compared

The main method is a block-sparse crosscoder with:

- one sparse support shared across sites;
- signed vector coordinates within each active block;
- a site-specific decoder for each observation site;
- a deployable thresholded codec evaluated in raw activation space.

Phase 3 compares it with scalar sparse crosscoders, independent per-site
models, shared-support scalar models, block methods without the complete
shared-coordinate contract, a dense low-rank control, and a source-only
descriptive anchor. The exact panel is defined in
[`docs/design.md`](docs/design.md).

## Evidence and selection

Phase 1 selects by the worst normalized truth-identification margin. FVU is a
guardrail there, not evidence that the correct factor was recovered.

Phases 2 and 3 use mean raw-space FVU at exact 256, 384, and 512
bits/token. Fixed-width packet overhead and serialized codec bytes are included.
When a rate lies between measured endpoints, the experiment executes the
declared time-sharing schedule on paired rows; it does not extrapolate.

Candidates are aggregated by median across seeds, then worst seed, then their
readable candidate name. Development, confirmation, and final splits remain
disjoint.

All-view, site-only, and leave-one-out endpoints are reported for the main
method and comparators. Missing-view behavior is a secondary diagnostic rather
than a separate promotion hurdle.

## Repository map

- `block_crosscoder_experiment/studies.py` — scientific recipes, plans, and
  selection policies
- `block_crosscoder_experiment/phase1.py` — truth-known generators
- `block_crosscoder_experiment/model.py` — block-sparse crosscoder
- `block_crosscoder_experiment/trainer.py` — training loop
- `block_crosscoder_experiment/store.py` — activation storage and batching
- `block_crosscoder_experiment/codec.py` — deployable sparse codec
- `block_crosscoder_experiment/evaluation.py` — rate-distortion and shared-code
  metrics
- `block_crosscoder_experiment/campaign.py` — current campaign state and stage
  transitions
- `block_crosscoder_experiment/cli/` — the `bsc` command
- `docs/design.md` — complete scientific contract
- `docs/paper_comparison.md` — bridges to the source papers
- `docs/findings.md` — evidence-bound reporting surface

Generated checkpoints, stores, evaluations, and logs are local artifacts.
Compact result tables live in `data/summary/`; publication figures live in
`docs/figures/`.

## Quick start

For a fresh reviewer machine, use Python 3.12 in an isolated environment:

```bash
git clone https://github.com/a9lim/block-crosscoder-experiment.git
cd block-crosscoder-experiment
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[review]"
bsc review
```

`bsc review` is download-free and runs the real training, codec, evaluation,
campaign, selection, and confirmation code on tiny CPU data. It also loads
and estimates the Phase-2 and Phase-3 definitions. The command prints and
retains its artifact directory for inspection.

See [`docs/reviewer_setup.md`](docs/reviewer_setup.md) for native
macOS/Linux and Windows/WSL notes, the optional test suite, Hugging Face
access, CUDA installation, real activation capture, storage sizing, and the
full experiment path.

To inspect the individual commands instead:

```bash
bsc --help
bsc matrix estimate --phase phase1 --smoke
bsc matrix plan --phase phase1 --smoke --root /tmp/bsc-phase1-smoke
bsc matrix run --root /tmp/bsc-phase1-smoke --limit 1
bsc matrix status --root /tmp/bsc-phase1-smoke
```

The cell lifecycle is deliberately ordinary:

```text
planned → prepared → running → trained → calibrated → evaluated → qualified
```

Each cell directory contains `cell.json`, `state.json`, and an `outputs/`
directory. Rerunning with `--resume` continues a failed or interrupted cell
from the latest completed stage.

After a complete Phase 1:

```bash
bsc matrix freeze-phase1 --root RUN_ROOT
```

The resulting decision JSON can be reviewed, edited if necessary, and passed
to Phase 2:

```bash
bsc matrix plan \
  --phase phase2 \
  --phase1-decision RUN_ROOT/decisions/phase2-authorization.json \
  --root PHASE2_ROOT
```

Use `bsc data --help` for activation capture and aligned derived views, and
`bsc matrix --help` for selection and phase advancement commands.

## Execution notes

- Phase 1 runs in fp32.
- Phase 2 capture uses its declared bf16 forward precision; fp16 activation
  stores are not supported.
- Training and capture belong on the CUDA machine; local CPU smoke runs cover
  platform-independent behavior.
- The portable installation needs only NumPy, safetensors, and PyTorch.
  Install `.[full]` for model capture and figures, and `.[cuda]` for the
  Linux/NVIDIA kernels.
- Resource estimates remain refusal gates because they prevent an accidental
  oversized run, not because artifacts are considered hostile.
- The campaign never launches scientific training implicitly. Planning,
  running, selecting, and advancing are separate commands.

## Focused verification

The retained checks cover the claims that could change an experimental result:
model math, training behavior, codec round-trips, split separation, selection,
ordinary resume, and one end-to-end smoke cell.

## License

CC BY-SA 4.0.
