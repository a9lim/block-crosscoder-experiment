# Phase 0 findings — withdrawn pending reproduction

*Status changed 2026-07-20 during Phase 0.5.*

The former Phase-0 findings are not current scientific evidence. The
adversarial audit in [audit_2026-07-20.md](audit_2026-07-20.md) found defects
in training-state handling, run/checkpoint binding, codec fitting and
serialization, geometry endpoints, upstream provenance, and paper-baseline
fidelity. The affected activation stores and checkpoints on `jobe`, the
committed compact evidence, the winner/showcase pointers, and every generated
figure derived from them were removed before starting Phase 0.5.

Accordingly, this file makes no quantitative reconstruction, rate-distortion,
feature-capture, optimizer-budget, or geometry claim from Phase 0. The prior
text remains recoverable from git history for audit purposes, but must not be
cited as a passing result.

What remains supported without the withdrawn runs is the model definition: a
BSC infers one signed vector block code jointly from several sites and decodes
it through a site-specific frame. Under the concatenated Stiefel constraint,
the squared block-code norm equals that block's isolated total decoded energy.
This algebraic identity does not establish empirical superiority or paper
faithfulness.

Phase 0.5 now owns the evidentiary reset:

- [paper_comparison.md](paper_comparison.md) defines the exact BSF,
  Crosscoder, Minder, and SASA bridges and the staged comparison matrix.
- [audit_2026-07-20.md](audit_2026-07-20.md) records each flaw, its fix, and
  the remaining requirement to regenerate evidence.
- `bsc phase05-matrix campaign` harvests independent raw/scalar/layer/
  whitened/whitened-plus-site-renormalized stores, runs the paper-bridge
  screen, and only then enters the exhaustive declared optimizer matrix.

No Phase-0 finding is reinstated until it is reproduced from hash-bound
Phase-0.5 artifacts with the corrected code and required endpoints.
