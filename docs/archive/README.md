# Archive — Phase 0 primary sources

Everything in this directory is a **verbatim historical record** of the
pilot program (2026-07-15 → 07-19), preserved as the evidence chain
behind the condensed, authoritative layer above it
([`../findings-phase0.md`](../findings-phase0.md),
[`../design.md`](../design.md)). Where the condensed docs and these
disagree, the condensed docs are current; where you need the raw
numbers, the campaign chronology, or the *why* behind a settled
parameter, the primary source is here.

Naming note: these documents use the original internal phase ladder
(−1 → 0 → 0.5 → 0.9 → 0.9.5 → 0.9.6 → 0.9.9). Under the 2026-07-19
renumbering, that entire ladder **is Phase 0**; the production run
(formerly "Phase 1" in these documents too — the number survives) is
Phase 1. Script names in `scripts/` keep their ladder-era names.

## Map (old ladder → what it was)

| doc | ladder phase | one line |
|---|---|---|
| [`design-v2.4.md`](design-v2.4.md) | — | the full frozen design as of 2026-07-19: hypotheses, architecture algebra, phase gates, decision log (v1→v2.4 provenance) |
| [`design-review-2026-07-15.md`](design-review-2026-07-15.md) | — | round 1+2 adversarial review (35 findings + R1–R26): the Gram constraint, the codec, the honest R-D framing |
| [`design-review-2026-07-16.md`](design-review-2026-07-16.md) | — | round 3 (D1–D14 deployment, P1–P25 paper fidelity): the 4 TB NVMe re-plan, whitened store, AuxK respec |
| [`design-review-2026-07-17-fidelity.md`](design-review-2026-07-17-fidelity.md) | — | round 4 (F1–F11, S1–S7): shrinkage-whitener honesty, the F7 renorm question, verified parent-paper ground truth |
| [`findings-phase-minus1-battery.md`](findings-phase-minus1-battery.md) | −1 | synthetic ground-truth harness: λ-veto, packing economics, ring-detection limits, AuxK three-way, operating point |
| [`findings-phase0-control.md`](findings-phase0-control.md) | 0 | GPT-2 positive control: Engels rings replicate through our pipeline (all three families at the permutation floor) |
| [`findings-phase0-gemma.md`](findings-phase0-gemma.md) | 0 | gemma discovery null at 16k+65k; the month ring exists decoder-side below every clustering threshold |
| [`findings-phase05-cross-layer.md`](findings-phase05-cross-layer.md) | 0.5 | frames rotate, code persists (R² 0.83–0.90 across depths); rings live early; L17 the cautionary tale |
| [`findings-phase09-rehearsal.md`](findings-phase09-rehearsal.md) | 0.9 | 1b end-to-end plumbing rehearsal: store integrity, bit-determinism, θ transfer, toy manifold export |
| [`findings-phase095-calibration.md`](findings-phase095-calibration.md) | 0.9.5 | 31-run optimizer calibration on the 1b store: the 1.2e-3 optimum + 2.4e-3 cliff (later overturned at 4b) |
| [`findings-phase096-tier-a.md`](findings-phase096-tier-a.md) | 0.9.6 A | consolidation is universal, calendar order is a seed lottery; G8192 dead dynamics |
| [`findings-phase096-pilot4b.md`](findings-phase096-pilot4b.md) | 0.9.6 B | the D13 4b pilot: lr 3e-4 ratified (1.2e-3 destroys at 4b), F7 evidence, calendar/zoo/atlas/geometry passes |
| [`findings-interim-artifact-analysis.md`](findings-interim-artifact-analysis.md) | interim | block 23 (the first discovered month manifold), whitener-shrinkage allocation mechanism, packing cliques |
| [`findings-phase099-tranche1.md`](findings-phase099-tranche1.md) | 0.9.9 | the pre-NVMe campaign: guard/cap/skip partition, R-D codec first light, 2×2 factorial, single-site placement, frontier |
| [`runbook-phase096.md`](runbook-phase096.md) | 0.9.6 | the pilot campaign runbook |

Still live (not archived): [`../runbook-phase099.md`](../runbook-phase099.md)
carries the sealed probe panel and the open tranches (5, 6, 7); it
archives when the campaign closes.

## Figures and heavyweight data

The figure directories these documents reference
(`figures/phase0-gemma/`, `figures/pilot4b/`, `figures/interim/`, …)
were **deleted 2026-07-19** (a9); the canonical set regenerates from
the current winner (`data/phase0/winner.json`) via
`scripts/analysis/`. Historical diagnostic figures (lr-cliff traces,
cascade anatomy) are regenerable from the run checkpoints under
`/data/runs/` on jobe plus the scripts named in each document.
Compact committed evidence lives in `data/phase0/` (R-D payloads,
SAE-era provenance under `sae/`, the battery report).
