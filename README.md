# block-crosscoder-experiment

Block-sparse crosscoders (BSCs) learn one sparse vector-subspace code jointly
across model sites and decode it through a different frame at each site. This
repository owns the model, training/store/codec stack, paper bridges,
synthetic and language evaluations, and the Phase-0.5 comparison campaign.

## Current state

Phase 0.5 began on 2026-07-20 after an adversarial comparison with the BSF,
Crosscoder, Minder, and SASA papers exposed both implementation flaws and
uncontrolled differences from the parent methods. The old evidence,
winner/showcase pointers, generated figures, and corresponding `jobe` runs
were withdrawn. There is currently no promoted winner and no empirical
Phase-0 claim.

Read these in order:

1. [`docs/audit_2026-07-20.md`](docs/audit_2026-07-20.md) — flaw and
   remediation ledger.
2. [`docs/paper_comparison.md`](docs/paper_comparison.md) — exact method
   comparison, hyperparameters, endpoints, and staged matrix.
3. [`docs/design.md`](docs/design.md) — normative architecture and operations,
   now subordinate to the Phase-0.5 gates where older Phase-1 choices conflict.
4. [`docs/findings-phase0.md`](docs/findings-phase0.md) — withdrawal notice.

The corrected stack supports raw, dataset-scalar, token-LayerNorm, shrinkage-
whitened, and whitened-plus-site-renormalized gauges as explicit store-level
factors. It also includes per-token TopK, BatchTopK, fixed-threshold, Fel
Vanilla/Grassmannian/Group-Lasso bridges, original ReLU/L1 and Minder
BatchTopK Crosscoder bridges, a free-map SASA bridge, token-denominated AuxK,
experiment-bound checkpoints, serialized codecs, raw-coordinate FVU, and
trained shared-code/effective-span endpoints.

## Install and command line

Use the workspace's shared plain Python installation:

```bash
# once from the transformer-experiments root
python -m pip install -e .

# from this repository
python -m pip install -e .
bsc --help
```

The package owns the only executable surface:

```bash
bsc harvest --help
bsc train --help
bsc train-single-site --help
bsc verify-store --help
bsc validate-codec --help
bsc reproduce-papers --help
bsc phase05-matrix --help
bsc trained-endpoints --help
bsc refresh-analysis --help
```

Plan or run the complete declared campaign with:

```bash
bsc phase05-matrix plan --root /data/runs/bcc-phase05 --profile all
bsc phase05-matrix campaign \
  --root /data/runs/bcc-phase05 \
  --store-root /data/stores/bcc-phase05
```

The campaign is resumable and fail-closed. It harvests each normalization
from the same pinned stream prefix, runs the paper-bridge screen first, stops
before the full factorial if any screen cell fails, and deletes only
non-promoted large checkpoints after retaining manifests, logs, and reports.

## Repository map

```text
block_crosscoder_experiment/
  model.py, trainer.py, store.py, codec.py
  discovery/       SAE discovery, nulls, ring tests, sealed panel
  analysis/        trained endpoints, extraction, probes, and figures
  cli/             unified operational entry points
data/
  evidence/        empty until corrected Phase-0.5 artifacts qualify
figures/            empty catalog plus shared Plotly runtime
docs/
  audit_2026-07-20.md
  paper_comparison.md
  findings-phase0.md
  design.md
tests/              offline and synthetic verification
references/         checked-in paper text/source ledger
```

Training and harvest run on `jobe`'s RTX 4090. Do not load a checkpoint
concurrently with training there. Large stores, checkpoints, and analysis
caches live under `/data` and are regenerated artifacts.

## License

CC-BY-SA-4.0. See [LICENSE](LICENSE).
