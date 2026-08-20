# Figures

Publication-facing evidence figures are tracked here. The current Phase-2
geometry set lives in [`phase2/`](phase2/), alongside its compact
`month_geometry_metrics.json` source. Exploratory renders belong in the
ignored `work/` directory.

Both PNG and SVG versions are available for the raw residual-layer views,
joint BSC views, layerwise BSC views, and feature gallery. The figures support
the topology claim summarized in [`docs/findings.md`](../docs/findings.md):
calendar order is preserved, while the learned geometry is not a
seed-invariant ideal circle.

Machine-readable campaign headlines are in
[`data/summary/results.json`](../data/summary/results.json). Full campaign
stores, checkpoints, and evaluation bundles remain ordinary local
artifacts and are intentionally not committed.

Regenerate the Phase-2 set from two deployable artifacts with:

```bash
bsc figures \
  --deployable /path/to/seed-0.pt \
  --deployable /path/to/seed-1.pt
```

The command writes `figures/phase2/` by default. It requires the pinned GPT-2
capture dependency and a CUDA device unless `--device` selects another
compatible backend.
