# Phase −1 battery findings (runs 1–2, 2026-07-16)

Scenario battery per design v2.2 Phase −1, run on jobe (RTX 4090, CUDA,
8-bit Adam — the production optimizer, per the placement amendment).
Battery code: `block_crosscoder_experiment/battery.py`; reports (out of
git, regenerated): `data/phase_minus1_report.json` (run 2, current),
`data/phase_minus1_report_run1.json` (run 1). Config: 3000 steps ×
batch 1024 ≈ 3M tokens per run, d=128, S=4, b=4, seeds {0, 1}.

**Run 1 → run 2 is a pure re-measurement.** Training is fully
seed-deterministic — every recovery overlap is bit-identical across the
two runs — so run 2 isolates the three scoring fixes made after run 1:
contribution-energy shares (replacing Frobenius, which counts parked
frame capacity), gate-level bundle association (replacing per-member
Hungarian matching), and a seed-variance control arm for the rotation
gate. All three instruments demonstrably work (details below); the
gates that remain red fail on a real phenomenon, not on measurement.

## Verdicts at a glance

| Scenario | Gate | What actually happened |
|---|---|---|
| lambda_veto | **PASS** (λ=0 fallback fires) | Admissible set empty at every nonzero λ; λ=0 primary confirmed. Mechanism revised — see below. |
| core | FAIL (0.5 captured) | Rank ≤3 gaussians recover essentially perfectly; shells tile into arcs; rank-4 splits. Span-found 0.67 (seed 0) / 0.83 (seed 1). |
| decoys | FAIL | Depth-profile question answered YES (one-hot decoys come back site-exclusive under contribution shares). Fails on splitting + one homeless decoy (seed 0). |
| bundle_null | FAIL (seed 0 ring) | No hallucination on either seed (the actual null holds). Seed 1 captures the ring perfectly; seed 0 tiles it into arcs. |
| rotation_equivariance | FAIL (seed 0 spectrum) | Spans rotation-stable on both seeds (agreement ≥0.99). Spectra are basin-sensitive; the n=1 control arm can't absorb that variance. |
| frequency_ladder | report-only | Whole ladder degraded at 3M tokens (f=0.1 → overlap 0.5–0.7); below f≈0.01 nothing survives (116 active samples at f=0.001). The R24 curve's starved end. |
| auxk_comparison | report-only | SASA / long-horizon / Fel indistinguishable (6–7 dead of 16, all variants, both seeds). Keep SASA C.1 as spec default; re-run the comparison at 0.9 scale. |

## 1. The λ-verdict: λ=0 primary, confirmed — with a revised mechanism

Run 2, contribution-energy shares (seed-0 arm; overlaps identical to run 1):

| λ | share error (run 2) | share error (run 1, Frobenius) | overlap |
|---|---|---|---|
| 0 | 0.0002 | 0.0089 | 0.952 |
| 3e-4 | 0.0028 | 0.176 | 0.710 |
| 1e-3 | 0.0175 | 0.169 | 0.690 |
| 3e-3 | 0.0587 | 0.204 | 0.705 |

The run-1 reading — "nuclear penalty grossly concentrates depth
profiles" — was substantially a **Frobenius parked-capacity artifact**:
under honest contribution shares, λ=3e-4 and 1e-3 *pass* the 0.02 share
tolerance. What keeps every nonzero λ inadmissible is **overlap
retention**: recovery collapses 0.95 → ~0.71 at the smallest tested λ,
far below the 0.85-retention floor. The nuclear penalty's real harm at
this scale is subspace-recovery degradation (rank compression), not
measured depth distortion. Verdict per the v2.2 protocol: admissible
set empty → **λ=0 primary**, rank reported diagnostically. The Phase-1
λ grid should still bracket smaller values (3e-5, 1e-4) before writing
the penalty off at production scale — 3M-token battery runs can't rule
out a benign-λ window that opens with more data.

## 2. One phenomenon behind every red gate: the capture/tiling/splitting lottery

After the instrument fixes, every remaining failure traces to the same
mechanism. At harness scale with spare capacity (G/F ≈ 1.7 in core and
decoys), each (seed, dataset) lands in a per-block basin:

- **Captured** — the block comes back whole: overlap ≥0.999,
  code R² ≥0.97, rank exact, geometry regime (norm-CV) preserved.
  Reliable for rank ≤3 gaussian blocks on both seeds, in every scenario.
- **Tiled** — curved geometry (shells/rings) splits into gated arcs:
  span found (overlap 0.98–0.9996) but matched-piece code R² ~0.6,
  learned rank 1, support size 2. The union of the pieces captures the
  block (support-restricted R² 0.81–0.97). Seed-dependent: seed 1
  captured the bundle ring perfectly (R² 0.9996, norm-CV 0.095); seed 0
  tiled the same ring into two arcs. Confirmed aux-invariant in run-1
  side experiments.
- **Split** — flat blocks shed rank when spare blocks are available:
  the rank-4 core block splits (overlap 0.75) on both seeds; rank-2
  gaussians split into rank-1 pairs in the decoy zoo (matched piece
  R² 0.62, support union R² 0.97). Classic dictionary feature-splitting,
  now observed for subspace units.
- **Homeless** — a planted block not found at all. Core seed 0 lost the
  hollow shell (zero associated activations); decoy seed 0 lost one
  one-hot decoy (overlap 0.17, energy smeared). Accounting: with exact
  selection budget (k·B = E[active]), surplus learned blocks *should*
  die — dead blocks are only pathological when alive-count < planted
  count (core seed 0: 5 alive for 6 planted). AuxK (SASA, s_aux=4) was
  on; at exact budget a revived block can only survive by displacing an
  incumbent, so revival is structurally weak exactly when it's needed.

Instrument notes for later phases, from the same data:

- **Norm-CV shell detection only works on captured rings.** Arc tiles
  of a hollow ring read norm-CV 0.5–0.66 (gated pieces don't inherit
  the shell's norm concentration). A tiled ring is invisible to the
  shell signature — Phase-0 ring hunting must check span-level evidence
  (closed 2-D span, phase coverage across tiles), not per-block norm-CV
  alone.
- **Spans are the rotation-stable readout; spectra are not.** Rotated
  vs control: span agreement ≥0.99 on both seeds, while spectrum
  rel-diff is basin-dominated (0.52 vs 0.16 control on seed 0, 0.22 vs
  0.22 on seed 1). Phase-0 claims should anchor on spans and treat
  per-block spectra as basin-sensitive.
- **Contribution shares are mandatory.** Frobenius shares misread
  one-hot decoys as ~0.37-error "shared" profiles; contribution shares
  read 1e-4. The Phase-2 saklas `share` export must be
  contribution-energy based.

## 3. Open decision — Phase −1 core gate semantics (a9's call)

The battery's hard gates demand full capture per planted block. The
learner instead delivers capture for flat rank ≤3 blocks and
span-recovery-with-tiling/splitting elsewhere — a documented regime
map, not an instrument failure. Options on the table:

1. **Capture gates as written** — Phase −1 blocks the ladder on
   behavior (feature splitting under spare capacity) that the SAE
   literature documents thoroughly and that real-data phases will face
   regardless.
2. **Split gate semantics** — capture required where capture is
   achievable (flat rank ≤3), span-recovery + support-union capture
   accepted for shells and high-rank blocks, tiling reported as a
   characterized regime. Homeless blocks (span not found at all) stay a
   hard failure everywhere — different in kind from tiled.
3. **Match capacity to the zoo (G=F)** — kills the splitting basin but
   teaches to the test; production runs have massive spare capacity.

Recorded lean (fable): option 2 — the harness's job is to prove the
pipeline finds what was planted and measures it honestly, which it now
does. Not adopted unilaterally; the gate criterion is frozen design
surface.

**Ruling (a9, 2026-07-16): option 1 — strict capture-as-written.** The
gate stands; Phase −1 does not pass on tiled or split recoveries. The
consequence is a capture campaign (below), not gate surgery: find the
training conditions under which capture is seed-robust, verify the
production config sits inside that region, and re-run the battery
there. An existence proof is already pinned in the repo: the
`test_end_to_end_recovery` config achieves 5/5 full capture *including
a hollow shell* — the battery core config differs from it in six
enumerable factors (zoo composition, feature frequency, budget ratio
0.8 vs 1.0, G 8 vs 10, AdamW vs 8-bit Adam, step count), which is a
clean factorial to isolate.

## 4. Smaller confirmations

- 8-bit-Adam × retraction ordering check (pulled forward from the 0.9
  gate): passed on jobe — post-cast Gram residuals stay at bf16 noise
  under the production optimizer.
- Dead-block revival unit test: aux revives, no-aux control stays at
  exactly zero (encoder-side starvation confirmed as the death mode).
- Frequency ladder at 3M tokens is sample-starved below f≈0.01 —
  consistent with the R24 rare-feature curve; the Phase-1 store (38M
  train tokens, calibration split) is sized far above this regime, but
  the ladder should be re-read at 0.9 scale before trusting rare-block
  claims.
