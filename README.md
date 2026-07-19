# block-crosscoder-experiment

**Block-sparse crosscoders (BSC)**: dictionary learning whose atomic
unit is a *subspace* with one shared code across depth — the empty
{block} × {cross-site} cell of the sparse-dictionary 2×2, filled on
gemma. Phase 0 (the pilot program) found that trained BSCs capture
concept manifolds — the calendar ring, the cardinal number line, a
world-atlas map — as single blocks whose frames rotate with depth
while the code persists, and that on the honest rate–distortion axis
cross-site tying is a ~7.9× rate cut at zero distortion cost, with
the site-renormalized BSC strictly dominating the matched scalar
frontier everywhere they overlap.

- [`docs/findings-phase0.md`](docs/findings-phase0.md) — the findings,
  paper-shaped: claims C1–C10, hypothesis status, standing rules.
- [`docs/design.md`](docs/design.md) — the forward design: settled
  parameters, open parameters, the Phase-1 production run.
- [`docs/archive/`](docs/archive/README.md) — verbatim primary
  sources (ten findings docs, four review rounds, design v2.4).
- [`docs/research/block-sparse-crosscoders-2026-07.md`](docs/research/block-sparse-crosscoders-2026-07.md)
  — the literature digest.

## Status

**Phase 0 complete** (2026-07-19, closeout tranches in flight);
**Phase 1** — the full-size BSC on a 53M-token / 2.17 TB gemma-3-4b
store — waits on the 4 TB NVMe install. Training stack pinned and
a9-ratified: lr 3e-4 cosine, λ=1e-3, site-renorm gauge,
aux-ratio-cap 1.0, loss-spike guard, streaming θ. The current best
checkpoint is always [`data/phase0/winner.json`](data/phase0/winner.json);
figures regenerate from it.

## Install

```bash
# from the workspace root (once):
python -m pip install -e ..
# this experiment:
python -m pip install -e .
```

Training runs on the 4090 box (CUDA); harvest/analysis on the M5 Max
(MPS) — device auto-detected, `--device` overrides.

## Layout

```text
block_crosscoder_experiment/  model, trainer, store, codec, battery
scripts/                      phase entry points (ladder-era names)
scripts/analysis/             probes + figure regeneration (winner.py)
data/phase0/                  committed compact evidence (R-D payloads,
                              placement, SAE-era provenance, winner.json)
figures/phase0/               canonical figures, regenerated from winner
docs/                         findings, design, runbook, archive
tests/                        offline checks
```

## License

CC-BY-SA-4.0. See [LICENSE](LICENSE).
