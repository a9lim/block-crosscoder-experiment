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

The reviewer path has one job: produce the selected BSC artifact on a
Linux/NVIDIA CUDA machine. With Python 3.12 installed, this single shell
command installs every runtime dependency and runs the replication:

```bash
python3.12 -m pip install "block-crosscoder-experiment @ git+https://github.com/a9lim/block-crosscoder-experiment.git" && bsc replicate
```

`bsc replicate` downloads the pinned GPT-2 Small and OpenWebText inputs,
captures residual-pre activations at blocks 3/5/7/9, fits scalar-RMS
normalization, trains the selected formula for 16M tokens, calibrates its
deployable codec, and writes:

- `bsc-16m/bsc-16m.pt` — the usable model, codec, and normalization;
- `bsc-16m/result.json` — the training and evaluation result.

The formula is the audited 4M development winner extrapolated to 16M at the
reviewer's request: LR `6e-4`, batch 512, no warmup, final-fifth decay, and
the weight-1 frequency-dead auxiliary. This exact 16M extrapolation was not
previously run or confirmed; the artifact and result say so directly.

See [`docs/reviewer_setup.md`](docs/reviewer_setup.md) for the hardware and
storage expectation, resume behavior, exact formula, and artifact-loading
example.

## Execution notes

- The command requires Python 3.12, Linux, a bf16-capable NVIDIA GPU, internet
  access, and roughly 215 GiB of temporary working data plus free-space
  headroom.
- GPT-2 Small and OpenWebText are public; no Hugging Face login is required.
- Interrupted work is retained. Continue it with `bsc replicate --resume`.
- Successful runs remove raw activations and campaign intermediates by
  default. Add `--keep-work` if those are useful.
- `bsc replicate --dry-run` prints the resolved formula and resource estimate
  without downloading data or requiring CUDA.

## License

CC BY-SA 4.0.
