# block-crosscoder-experiment

Do block-sparse crosscoders — subspace-unit dictionary learning with one
shared code across layers — recover token-level concept manifolds in gemma,
and can those discovered manifolds feed a real steering/probe runtime
(saklas)? See [`docs/design.md`](docs/design.md) for hypotheses, the architecture spec,
and the phase ladder;
[`docs/design-review-2026-07-15.md`](docs/design-review-2026-07-15.md) and
[`docs/design-review-2026-07-16.md`](docs/design-review-2026-07-16.md) for
the adversarial reviews that shaped it;
[`docs/research/block-sparse-crosscoders-2026-07.md`](docs/research/block-sparse-crosscoders-2026-07.md)
for the research digest this grew from.

## Install

```bash
# from the workspace root (once):
python -m pip install -e ..
# this experiment:
python -m pip install -e .
# developing against a local saklas:
python -m pip install -e ../../saklas
```

## Status

Design v2.2, frozen after three review rounds (two-round adversarial
review 2026-07-15; round-3 deployment + paper-fidelity pass 2026-07-16);
no runnable experiments yet. Next deliverables: the Phase −1 synthetic
ground-truth harness, then the Phase-0 blockification script (pinned
Bloom GPT-2 layer-7 positive control → gemma-scope-2).

## Layout

```text
block_crosscoder_experiment/  reusable phase implementations
scripts/                      numbered phase entry points
data/                         regenerated harvest and analysis artifacts
figures/                      regenerated figures
logs/                         local run logs
tests/                        offline checks
docs/                         design and research provenance
```

## License

CC-BY-SA-4.0. See [LICENSE](LICENSE).
