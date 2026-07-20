# block-crosscoder-experiment

Block-sparse crosscoders learn one sparse **subspace code** across model
depth, with a different frame at every site. This repository owns the model,
training and store stack, synthetic and language evaluations, compact pilot
evidence, and the production Phase-1 plan.

Phase 0 established that Gemma contains coordinates that persist while their
frames rotate; native blocks capture manifolds that scalar-SAE clustering
cannot bind; cross-site tying reduces rate by 7.8–7.9× without measured
distortion cost; and the site-renormalized BSC dominates the matched scalar
pilot frontier throughout their shared rate range. The full account is
[`docs/findings-phase0.md`](docs/findings-phase0.md).

## Current state

Phase 1 is ready to harvest and train. It waits only on installation of the
dedicated 4 TB NVMe in jobe. The production stack is pinned at Gemma 3 4B,
sites 9–30, `G=4096`, `b=4`, `k=32`, `lr=3e-4` cosine,
site-renormalized shrinkage whitening, `λ=1e-3`, auxiliary gradient-ratio
cap 1.0, a mandatory spike guard, full-split streaming threshold fitting,
and prefetch 4. See the self-contained [`docs/design.md`](docs/design.md).

[`data/winner.json`](data/winner.json) is the only current checkpoint
pointer. The figure and probe stack derives block identities from winner
artifacts instead of hard-coding them.

## Install and command line

Use the workspace's shared plain Python installation:

```bash
# once from the transformer-experiments root
python -m pip install -e .

# from this repository
python -m pip install -e .
bsc --help
```

The `bsc` command is the sole executable surface. Implementations live in the
package rather than a parallel `scripts/` tree:

```bash
bsc harvest --help
bsc train --help
bsc train-single-site --help
bsc verify-store --help
bsc validate-codec --help
bsc capture-zoo --help
bsc probe-families --help
bsc extract-geometry --help
bsc refresh-analysis --help
```

Training runs on jobe's RTX 4090. Harvest analysis and offline figures can
run on the M5 Max where their artifacts fit; `--device` overrides automatic
selection. Do not load a checkpoint concurrently with training on jobe.

## Figures

[`figures/index.html`](figures/index.html) is the consolidated catalog. Every
descriptive zoo member has the same contract:

```text
figures/<family>/frames.html
figures/<family>/flow.html
figures/<family>/stream.html
```

`stream` shows raw transformed class means over depth; `frames` shows the
same means through the promoted winner's best candidate block frame; `flow`
uses one fixed frame-space gauge to show motion across depth. Every page
stamps the winner, block, top-1 capture, order statistic, qualification, and
FVU. Failed families are diagnostics, not captured-manifold claims.
Cross-family figures live in [`figures/summary/`](figures/summary/).

Refresh the zoo, checkpoint-derived artifacts, and every canonical figure on
jobe:

```bash
bsc refresh-analysis
```

For a render-only pass over existing compact artifacts, use `bsc figures`.
The catalog shares one local Plotly runtime instead of embedding a multi-MB
copy in every page.

## Repository map

```text
block_crosscoder_experiment/
  model.py, trainer.py, store.py, codec.py
  discovery/       SAE discovery, nulls, ring tests, sealed panel
  analysis/        probe, extraction, and figure implementations
  cli/             unified command dispatch and operational entry points
data/
  winner.json      promoted checkpoint and exact coordinate metadata
  showcase.json    two-gauge descriptive block election
  evidence/        compact, committed Phase-0 evidence
figures/            one indexed catalog plus cross-family summaries
docs/
  findings-phase0.md
  design.md
  literature.md
tests/              offline and synthetic verification
references/         source ledger; fetched full texts remain ignored
```

Large activation stores, checkpoints, and winner-scoped analysis caches are
regenerated artifacts under `/data` on jobe or `data/analysis/` locally.
The exact verbatim Phase-0 chronology removed from the working tree remains
recoverable at Git commit `ed5816e12d20589727e1a0cc4ec7e80e36d6ea2e`.

## License

CC-BY-SA-4.0. See [LICENSE](LICENSE).
