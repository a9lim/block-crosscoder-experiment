# Phase −1 battery findings (runs 1–6 + capture campaign, 2026-07-16)

**Phase −1 verdict: PASSED.** Battery run 6 (10k steps × batch 1024,
seeds {0,1,2,3}, G16 zoos, rank-3 decoy fixture, bundle budget pinned
to block-event demand) passes **all hard gates** under a9's strict
capture-as-written ruling — core, λ-veto, decoys, bundle-null,
rotation-equivariance 4/4 each; bundle ring CVs 0.047–0.060,
bit-identical to sweep round 8's k=0.75 cell. The route there is
§§5–6; Phase-1 primary is λ=1e-3 (§6.1).

Scenario battery per design v2.2 Phase −1, run on jobe (RTX 4090, CUDA,
8-bit Adam — the production optimizer, per the placement amendment).
Battery code: `block_crosscoder_experiment/battery.py`; reports (out of
git, regenerated): `data/phase_minus1_report.json` (run 5, current),
`data/phase_minus1_report_run1.json` (run 1). Runs 1–2 (and, by the
CLI bug in §5.3, runs 3–4): 3000 steps × batch 1024 ≈ 3M tokens,
seeds {0, 1}; run 5: the honest operating point, §5.4/§6. d=128, S=4,
b=4 throughout.

**Run 1 → run 2 is a pure re-measurement.** Training is fully
seed-deterministic — every recovery overlap is bit-identical across the
two runs — so run 2 isolates the three scoring fixes made after run 1:
contribution-energy shares (replacing Frobenius, which counts parked
frame capacity), gate-level bundle association (replacing per-member
Hungarian matching), and a seed-variance control arm for the rotation
gate. All three instruments demonstrably work (details below); the
gates that remain red fail on a real phenomenon, not on measurement.

## Verdicts at a glance

**This table is the 3k-scale (run-2) snapshot**, kept because §§1–4
analyze it; the operating-point verdicts that supersede it are in §6.

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

## 3. Phase −1 core gate semantics — ruled: strict capture-as-written

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

## 5. The capture campaign (sweep rounds 1–6, 2026-07-16)

Driver: `scripts/run_capture_sweep.py` (basin labels per planted block:
captured / merged / tiled / partial / missing; the cell grid's comments
are the round-by-round log). Data: `data/capture_sweep*.json` on jobe.
All cells 4 seeds unless noted; battery-scale d=128, S=4, b=4.

### 5.1 The budget-regime map

- **Round 1 (OFAT, failing battery core vs pinned passing test):
  selection budget ratio k/E[active blocks/token] is the single driving
  factor.** At ratio 1.0, shells tile into arcs and spare capacity
  splits ranks — stable attractors (10k steps does not heal them). At
  0.8, tiling and splitting vanish. Optimizer (8-bit Adam vs AdamW),
  s_aux (2/4/8), and step count are non-factors; the pinned test passes
  because its config lands at effective ratio 0.8 (k=1.0, E=1.25) with
  a lucky seed.
- **Round 2 (budget curve + anneal):** the curve is asymmetric — 0.7
  starves (deaths), 0.85 re-admits partials; 0.8 with 10k steps is the
  operating point (core 0.96 mean / 0.83 min capture, zero tiles).
  Budget annealing in either direction adds nothing (implemented,
  tested, unused).
- **Round 3 (G × budget at 10k):** spare capacity at tight budget fixes
  the residual init-lottery deaths — core at G16 (≈2.5×F) captures 4/4
  seeds at ratio 0.8 *and* 0.9; G24 regresses slightly (lottery
  returns). The same move made the old decoy zoo *worse* — see 5.2.
- **Round 4 (matched budget):** ratio 1.0 refuted everywhere. Core at
  G16×1.0 tiles its shells 4/4. The shell-free decoy zoo tiles its
  *gaussians* instead — junk-fill pressure lands on the most tileable
  geometry available, shells are merely preferred prey. Budget is
  monotone over the tested range: tight (0.8) is best for every zoo.

### 5.2 Block width is a packing budget (headline finding)

The old decoy fixture — three *identical* rank-2 one-hot site decoys —
kept failing at the operating point with a distinctive signature:
overlap 1.0, depth share split ≈0.5/0.5, code-R² ≈0.86, plus one
unmatched twin. Round 5 (30k steps) turned the interpretation around:
at G10 the outcome is step-count-invariant, and at G16 with 30k steps
**all four seeds merge** — more capacity and more training make the
merge *more* reliable, not less.

Synthesis (marked as such; every observation above is data): two
rank-2 features pack losslessly into one width-4 block — the union
block's code carries both twins' coordinates and the per-site frames
route them, so reconstruction stays exact even on co-fire events —
while freeing an entire block plus selection budget. Merging is
therefore loss- *and* budget-optimal, and a better-converged estimator
merges more often. The passing seeds were init-trapped in the
less-optimal separated solution. Prediction: raising decoy rank to 3
(pairwise rank 6 > b=4) makes packing lossy and separation
objective-aligned. **Round 6 confirmed it exactly**: zero merges at
either G, and G16 flipped from worst (4/4 merged at 30k) to perfect
(4/4 seeds, all six blocks captured) — with packing impossible, spare
capacity helps decoys precisely as it helps core.

Consequences beyond the fixture:

- Production gemma is full of sub-width features; **Phase 0/1 must
  expect packed blocks** — multi-feature blocks with split
  contribution-share profiles and degraded code-R² at full overlap are
  a *converged optimum*, not noise. The signature above is the
  diagnostic.
- The Phase-2 saklas `share` export of a packed block is a composite;
  naturalness evals should treat near-50/50 split shares as a packing
  flag before trusting a discovered manifold.
- Fixture ruling (a9, 2026-07-16): decoys re-fixtured to rank-3 twins
  (`decoy_zoo`), which tests the design-spec'd property — site-exclusive
  recovery — without gating on twin discrimination the objective
  actively opposes. Residual caveat: rank-3 decoy + rank-1 shared sums
  to exactly b=4; if that pack ever appears it is a finding, not a
  fixture bug.

### 5.3 Correction: the "cross-process nondeterminism" reading is retracted

Battery runs 3 and 4 silently ran at 3000 steps, not the 10k operating
point — `run_phase_minus1.py`'s `--steps` default shadowed
`BatteryConfig` (fixed; CLI flags now fall through, and `--out` always
writes the report). Run-3 core is bit-identical to the sweep's 3k
`A_budget08` cell, so the interim theory that marginal basins were
cross-process chaotic (suspected cuBLAS) is dead: the configs differed.
Seed-determinism has held in every honest comparison (runs 1–2
bit-identical; sweep-vs-battery identical once configs match). Lesson
retained: **verify the effective config embedded in the report, not
the config you intended** — the `battery_config` block existed
precisely for this and went unread through two runs.

### 5.4 The battery operating configuration (post-campaign)

Ratio 0.8, 10k steps × batch 1024 (≈10M tokens), G ≈ 2.5×F: core /
bundle / frequency / decoy zoos at G16 (auxk already 16, deliberately
dead-prone). Established basins are seed-robust at this point; the
λ-veto, frequency-floor, and AuxK-variant verdicts quoted in §1 and §4
were measured at 3k scale and are superseded by battery run 5 below
where they differ.

## 6. Battery run 5 — the operating-point verdict (2026-07-16)

The first battery run actually executed at the operating point
(embedded `battery_config` verified: 10000 steps × batch 1024, seeds
{0,1,2,3}, ratio 0.8, all zoos G16, rank-3 decoy fixture). Six of
seven scenarios pass; the one red is `bundle_null`, and its anatomy is
a new instrument finding, not a rerun of the old lottery.

| Scenario | Gate | Operating-point result |
|---|---|---|
| core | **PASS** | recovered_fraction 1.0, all four seeds. |
| lambda_veto | **PASS** | Admissible set *opens*: λ=3e-4 and 1e-3 pass. **λ=1e-3 is the Phase-1 primary** (largest admissible, per protocol). See below. |
| decoys | **PASS** | Every planted block, every seed: overlap 1.0, code-R² 1.0, share error 0.0. No rank-3+rank-1 pack appeared. |
| bundle_null | **FAIL** → resolved | No-hallucination 4/4 (the null holds). Ring detected 1/4 — two seeds *soft phase-split* the ring (§6.2); rounds 7–8 located the fix, a block-event budget pin (§6.3). |
| rotation_equivariance | **PASS** | 4/4 seeds (the n=4 control arm absorbs the spectrum variance that failed run 2's n=1). |
| frequency_ladder | report | Clean through f=0.01 on all seeds (overlap ≥0.998). Below: per-seed lottery — 1/4 clean at f=0.003 and at f=0.001. |
| auxk_comparison | report | The variants **separate** at 10k (they were indistinguishable at 3k): SASA 1–4 dead of 16 and 12/12 rare features at overlap ≥0.998; long-horizon 11–12 dead, 7/12 rare lost; Fel 9–11 dead. SASA C.1 default now positively justified. |

### 6.1 The λ reversal: the admissible window opened with data

| λ | share error | overlap | admissible (tol 0.02, retention ≥0.850) |
|---|---|---|---|
| 0 (base) | 0.0002 | 1.0000 | — |
| 3e-4 | 0.0013 | 1.0000 | yes |
| 1e-3 | 0.0097 | 0.9548 | yes |
| 3e-3 | 0.0457 | 0.8573 | no (share error) |

§1's λ=0-primary verdict was a 3k-scale artifact: the overlap collapse
(0.95 → 0.71 at λ=3e-4) that emptied the admissible set does not
happen at 10M tokens — overlap holds ≥0.95 through λ=1e-3, and the
binding constraint at 3e-3 is genuine share concentration, not rank
compression. §1's own closing caveat ("a benign-λ window that opens
with more data") is exactly what happened, at values already in the
grid. Per the frozen v2.2 protocol (largest admissible λ = primary):
**Phase-1 primary is λ=1e-3**, λ=0 and 3e-4 ride along as the grid's
lower arm. Decision-log amendment appended to `docs/design.md`.

### 6.2 The bundle red: soft phase-splitting evades the norm-CV ring detector

Per-seed anatomy (gate needs no-hallucination AND ring-detected on
every seed):

| seed | no halluc. | ring detected | ring overlap | what the ring did |
|---|---|---|---|---|
| 0 | yes | no | 0.9995 | split across 2 blocks, norm-CV 0.22 each |
| 1 | yes | no | 0.51 | ordinary miss (establishment failure) |
| 2 | yes | yes | 1.0000 | single block, code-R² 1.000, norm-CV 0.010 |
| 3 | yes | no | 0.9995 | split across 2 blocks, norm-CV 0.22 each |

The failure mode in seeds 0/3 is **not** hard arc-tiling (which the
in-code comment assumed would stay norm-concentrated: an arc of a
constant-radius ring still has constant norm). The two blocks each
span the *full* ring plane (overlap 0.9995) and **co-fire on ~46% of
ring tokens** (conditional rates 0.73 + 0.73), splitting amplitude by
phase — each block's magnitude varies with ring angle, norm-CV lands
at ~0.22, and the 0.1 detector correctly refuses to call either block
a ring. The split forms early (visible at 300 steps) and is stable to
10k. Three consequences:

- The tiled-ring instrument caveat (norm-CV only flags *captured*
  rings) is now observed in vivo, in its soft form — and the soft form
  is *worse* than the design anticipated, because it also defeats the
  "arcs stay norm-concentrated" fallback argument. The Phase-0 ring
  hunt needs span-level evidence, not just norm-CV, as already noted.
- In all four seeds the co-active bundle stays *unpacked* (four
  separate rank-1 blocks, norm-CV ≈ 0.5 each) — which is why
  no-hallucination passes everywhere, but also means the zoo's
  E[active] stays at its unpacked value, keeping ring-vs-bundle budget
  competition high.
- Under strict capture-as-written the gate stands and the fix must be
  a training condition that makes single-block ring capture
  seed-robust. Round 7 (`scripts/run_bundle_sweep.py`) swept the
  levers: G8 (starve the split of spare blocks), 30k steps (the
  round-5 convergence-merge force, here working *for* the gate — a
  rank-2 ring fits b=4), and budget in both directions (k=0.9 prices
  the unpacked bundle out; k=1.5 is the matched-unpacked budget).

### 6.3 Rounds 7–8: capture solved at k=0.9; the detector needs zero slack

Round 7 headline (`data/capture_sweep_round7.json`; the G16/k1.2
reference cell reproduced run 5 bit-for-bit — determinism holds):
**no swept lever passes the gate as written, but k=0.9 solves
capture.** At G16/k0.9, all four seeds converge to the ideal
representation — the co-active bundle finally *packs* into one block
(conditional rate 1.0, norm-CV ≈ 0.45, so no hallucination) and the
ring lands in exactly one block, conditional rate 1.0, overlap 1.0,
code-R² 0.9996, dead blocks 0–2. The other levers fail honestly: 30k
steps *deepens* the phase-split (a seed that passed at 10k split at
30k — unlike the decoy twins, convergence here favors the split), G8
just relocates the miss lottery, and k=1.5 mixes splits with misses.

Why k=0.9 still fails the gate: the detector, not the capture. A jobe
diagnostic in the exact sweep basin (seed 0, G16/k0.9) splits the ring
block's firings by the planted ring gate:

| | ring-on firings | ring-off firings |
|---|---|---|
| rate | 1.000 of ring tokens | 0.023 of off tokens |
| mean ‖z‖ | 2.000 (planted radius) | 0.074 |
| norm-CV | **0.010** | — |

The block is a perfect ring on the ring's own tokens. But packed
demand is 0.75·B against a 0.9·B budget, and the 0.15·B slack
junk-fills through whatever blocks are handy: 6.5% of the ring block's
firings are trace-magnitude junk, and the bimodal mixture inflates the
all-firings norm-CV to 0.254 (the arithmetic is exact: CV ≈
0.97·√(junk fraction)). The all-firings detector therefore demands
**token-exclusive blocks**, which requires zero post-packing budget
slack — nothing in the objective rewards exclusivity, and junk-fill
actively erodes it whenever slack exists. Run 5's one passing seed
(CV 0.010) was the case where the *unpacked* bundle over-subscribed
k=1.2 so no slack existed anywhere.

Two control facts that keep the instrument story honest: a
CPU-basin diagnostic of a genuine phase-split shows ring-conditional
CV stays red (0.35) — conditioning does not rubber-stamp splits — and
the CV-vs-junk arithmetic says the detector tolerates junk only below
~1% of firings.

The arithmetic predicts a pass-as-written window: k just above the
packed demand 0.75 — tight enough to force packing, near-zero slack,
junk ≈ 0. Round 8 probed k ∈ {0.70, 0.75, 0.80} and **found the
window** (`data/capture_sweep_round8.json`):

| k | gate | ring CV | note |
|---|---|---|---|
| 0.70 | **4/4 PASS** | 0.010 | mild under-budget starves nothing |
| 0.75 | **4/4 PASS** | 0.047–0.060 | exact block-event demand |
| 0.80 | 0/4 | 0.17–0.20 | capture still perfect; 0.05 slack re-admits junk |

Every k ≤ 0.8 run has single-block capture at overlap 1.0 / code-R²
0.9996 — the entire k ∈ [0.70, 0.80] range solves capture; only slack
separates pass from fail, exactly as the junk arithmetic predicts.

**Resolution (training condition, per the strict-capture ruling's fix
space): the bundle scenario's budget is pinned to k = 0.75** — the
derivable value (matched *block-event* demand: the gate-grouped bundle
packs to one block, so 0.25 + 0.25 + 0.25), not a tuned number; 4/4
with 2× CV margin. The naive `budget_k` accounting (ratio 0.8 ×
per-feature ΣF = 1.2) double-counts the co-active bundle relative to
the packed format the learner correctly adopts — the packing economics
of §5.2, closing the loop. **Battery run 6 at this pin: all hard gates
pass** (bundle 4/4, ring CVs 0.047–0.060 bit-identical to the round-8
cell; every other scenario reproduces run 5). Phase −1 is green.

Phase-0 consequence, sharpened: production runs always have budget
slack, so **bare norm-CV will miss real, perfectly captured rings**
(junk tolerance is <1% of firings). The ring hunt needs span-level
plus gate-conditional evidence — the existing caveat, now with a
quantified mechanism and an in-vivo demonstration.
