# Phase-2 gate review and evidence carry-forward — 2026-07-23

This review corrects the Phase-2 promotion contract without discarding trials
whose training and evaluation contracts did not change.

## Decision

- BSC is the target method and may receive a deeper tuning chain than the
  controls.
- Promotion uses the same common performance evidence for every method:
  integrity-complete qualification and exact fixed-rate raw-space FVU, subject
  only to method-specific validity invariants.
- Site-only and leave-one-out FVU, support overlap, coordinate concordance,
  decoded-energy coverage, and functional-dependence measurements remain
  common diagnostics. They do not gate BSC while remaining optional for the
  controls.
- Phase-2 codec-exclusion fractions are diagnostic because excluded events are
  already priced in the measured deployable-codec distortion. Phase 1 and
  Phase 3 retain their declared exclusion checks.

The correction is recorded as an immutable campaign amendment. It does not
rewrite cell manifests, qualifications, checkpoints, evaluations, or prior
selection artifacts.

## Carry-forward boundary

The active campaign is `/data/runs/bsc-phase2-d84627e`.

At review time it contained 52 materialized cells:

- 47 integrity-complete qualified cells, including positive and negative
  scientific outcomes;
- 5 training failures caused by the declared Stiefel code-norm /
  decoded-energy Gram-residual invariant, not by the superseded sharing gate.

All 47 qualified cells remain usable. Nine qualifications whose only negative
scientific checks were Phase-2 codec-exclusion fractions are reinterpreted by
the amendment at selection time; their original bytes and hashes remain
unchanged. The five training failures remain negative evidence and are not
promoted or relabeled.

The already-committed anchor and comparator-root selections were recomputed
under the corrected contract. Their selected candidate identities did not
change, so the amendment ratifies those exact historical artifacts and the
4M-token trials descended from them. The incorrect BSC sharing gate had not yet
selected the architecture winner.

## Continued execution

Every post-amendment selection embeds the complete amendment manifest and
artifact hash. New cells remain descendants of the existing campaign plan and
use the amendment-pinned successor implementation identity. The runner
schedules only cells newly materialized by corrected selections; it does not
repeat an unchanged completed cell.

The compact review export is retained at
`/data/runs/archive/bsc-phase2-d84627e-pilot-evidence`. The full active campaign,
checkpoints, activation stores, and derived views remain in place.
