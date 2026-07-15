# block-crosscoder-experiment

Do block-sparse crosscoders — subspace-unit dictionary learning with one
shared code across layers — recover token-level concept manifolds in gemma,
and can those discovered manifolds feed a real steering/probe runtime
(saklas)? See [`DESIGN.md`](DESIGN.md) for hypotheses and the phase ladder;
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

Pre-Phase-0 scaffold (2026-07-15). No runnable experiments yet; the Phase-0
blockification script is the next deliverable.
