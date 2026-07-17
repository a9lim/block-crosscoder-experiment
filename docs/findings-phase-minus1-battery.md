# Phase −1 findings — verdict, operating point, and what the harness taught

Synthetic ground-truth battery per design v2.3 Phase −1, run on jobe
(RTX 4090, CUDA, 8-bit Adam — the production optimizer). Battery code:
`block_crosscoder_experiment/battery.py`; campaign driver:
`scripts/run_capture_sweep.py` + `scripts/run_bundle_sweep.py`;
reports and sweep data regenerate into `data/` (out of git). This doc
is the consolidated record of battery runs 1–6 and capture-campaign
sweep rounds 1–8 (all 2026-07-16); it leads with the final state, and
§6 keeps the run-by-run history including the retractions.

**Phase −1 verdict: PASSED.** Battery run 6 passes **all hard gates**
under a9's strict capture-as-written ruling (§5). Training is
bit-deterministic per (seed, config, device) — every honest re-run
reproduced its predecessor exactly.

## 1. Final verdicts (battery run 6, operating point)

Operating configuration: 10 000 steps × batch 1024 (≈10M tokens),
d=128, S=4, b=4, seeds {0,1,2,3}; selection budget ratio 0.8 of
E[active blocks/token] (bundle scenario: block-event demand k=0.75,
§2.3); spare capacity G=16 ≈ 2.5×F in every zoo; rank-3 site decoys;
8-bit Adam; SASA AuxK s_aux=4.

| Scenario | Gate | Result |
|---|---|---|
| core | **PASS** | recovered_fraction 1.0, all four seeds. |
| lambda_veto | **PASS** | Admissible set {3e-4, 1e-3} → **Phase-1 primary λ=1e-3** (largest admissible, per protocol). §2.4. |
| decoys | **PASS** | Every planted block, every seed: overlap 1.0, code-R² 1.0, share error 0.0. No rank-3+rank-1 pack appeared. |
| bundle_null | **PASS** | No-hallucination 4/4 *and* ring detected 4/4 (ring CV 0.047–0.060) at the block-event budget pin. §2.3. |
| rotation_equivariance | **PASS** | 4/4 — spans rotation-stable; the 4-seed control arm absorbs the basin variance of spectra. |
| frequency_ladder | report | Clean through f=0.01 on all seeds (overlap ≥0.998); per-seed lottery below (1/4 clean at 0.003 and at 0.001). The R24 calibration curve. |
| auxk_comparison | report | Variants separate at 10k: **SASA C.1** 1–4 dead of 16, 12/12 rare features at overlap ≥0.998; long-horizon 11–12 dead (7/12 rare lost); Fel 9–11 dead. Spec default confirmed. |

Fidelity note (F5, audit 2026-07-17): the "Fel" arm is a **hybrid**, not
a faithful Fel App. D replication — Fel uses the next-**ℓ** runner-up
blocks with α = 1/ℓ, ℓ being the *main* block sparsity; our arm
parameterizes both as s_aux (α = 1/s_aux), and s_aux ≠ k in the runs
made. Read its loss here as "runner-up-style AuxK", not "Fel's exact
recipe". The comparison verdict is unaffected (the SASA arm won on
mechanism, and §2.1 found s_aux a non-factor over 2/4/8).

## 2. What the harness taught

### 2.1 The budget-regime map (campaign rounds 1–4)

The **selection budget ratio** k / E[active blocks/token] is the
single driving factor of recovery quality, and it is monotone over the
tested range — tight (0.8) is best for every zoo:

- **Ratio 1.0 (matched)**: junk-fill pressure tiles curved geometry
  into gated arcs — shells preferentially, gaussians when the zoo has
  no shells (junk-fill lands on the most tileable geometry available).
  Stable attractors; more steps do not heal them.
- **Ratio 0.8**: tiling and splitting vanish.
- **0.7 starves** (establishment deaths); 0.85–0.9 re-admit partials.
- **Spare capacity fixes the init lottery**: at tight budget, G ≈
  2.5×F removes the per-seed establishment deaths (core 4/4 at G16 vs
  a death lottery at G10); G24 regresses slightly. Loose budget still
  tiles at any G.
- **Non-factors**: optimizer (8-bit Adam vs AdamW), s_aux (2/4/8),
  budget annealing in either direction (implemented, tested, unused),
  and step count *for basin identity* — basins established by 10k are
  stable; more steps only deepen the converged optimum (which flips
  sign depending on whether the optimum is the one you want: §2.2 vs
  §2.3).

### 2.2 Block width is a packing budget

The original decoy fixture — three *identical* rank-2 one-hot site
decoys — kept failing at the operating point with a distinctive
signature: overlap 1.0, depth share split ≈0.5/0.5, code-R² ≈0.86,
one unmatched twin. Thirty-k-step runs turned the interpretation
around: at G16 with 30k steps **all four seeds merge** — more capacity
and more training make the merge *more* reliable.

Synthesis (marked as such; the observations above are data): two
rank-2 features pack losslessly into one width-4 block — the union
block's code carries both twins' coordinates and the per-site frames
route them, so reconstruction stays exact even on co-fire events —
while freeing an entire block plus selection budget. Merging is
therefore loss- *and* budget-optimal, and a better-converged estimator
merges more. Prediction: raising decoy rank to 3 (pairwise rank 6 >
b=4) makes packing lossy and separation objective-aligned. **Round 6
confirmed it exactly**: zero merges at either G, and G16 flipped from
worst (4/4 merged) to perfect (4/4 seeds, all blocks).

Consequences:

- Production gemma is full of sub-width features; **Phase 0/1 must
  expect packed blocks as converged optima**, not noise. Signature:
  full span overlap, near-50/50 contribution-share split, degraded
  code-R².
- The Phase-2 saklas `share` export of a packed block is a composite;
  near-50/50 splits are a packing flag before trusting a discovered
  manifold.
- Fixture ruling (a9, 2026-07-16): decoys re-fixtured to rank-3 twins
  (`decoy_zoo`) — tests the design-spec'd property (site-exclusive
  recovery) without gating on twin discrimination the objective
  actively opposes. Residual caveat: rank-3 decoy + rank-1 shared sums
  to exactly b=4; that pack has not appeared (runs 5–6), and if it
  ever does it is a finding, not a fixture bug.

### 2.3 Ring detection needs zero budget slack (rounds 7–8)

Battery run 5's only red was bundle_null's ring positive-contrast:
no-hallucination held 4/4 (the null itself), but the ring was
"detected" 1/4. The failure anatomy came in two layers:

**Layer 1 — soft phase-splitting (a real capture failure).** Two
seeds captured the ring *span* perfectly (overlap 0.9995) but split it
across two co-firing blocks — each block spans the full ring plane and
carries phase-dependent amplitude (~46% of ring tokens shared,
norm-CV ≈ 0.22 per block). This is not the hard arc-tiling the code
had assumed norm-concentration would survive; the soft split defeats
that argument outright. It forms early (visible by step 300) and
deepens with training — the convergence force that *fixed* the decoy
twins (§2.2) works *against* the gate here.

**Layer 2 — junk-fill through captured rings (an instrument limit).**
k=0.9 solves capture completely: the co-active bundle packs into one
block and the ring lands in exactly one block, conditional rate 1.0,
overlap 1.0, code-R² 0.9996, on all four seeds. It still failed the
detector. Diagnostic (exact sweep basin): the ring block fires on
100% of ring tokens at mean norm 2.000 (the planted radius) with
**ring-conditional CV 0.010** — and on 2.3% of off-ring tokens at
norm 0.074, and those trace firings inflate the all-firings CV to
0.254. The arithmetic is exact — CV ≈ 0.97·√(junk fraction) — so the
detector tolerates junk only below ~1% of a block's firings, i.e. it
requires near-token-exclusive blocks, which nothing in the objective
rewards and any budget slack erodes.

The resolution (a training condition, per the ruling's fix space):
the bundle scenario's budget is pinned to **block-event demand
k = 0.75** — the derivable value: the gate-grouped bundle packs to one
block, so E[block events/token] = 0.25 + 0.25 + 0.25. Per-feature ΣF
(=1.5) double-counts the co-active bundle relative to the packed
format the learner correctly adopts — §2.2's packing economics closing
its own loop. Round 8: k=0.70 and 0.75 pass 4/4 (ring CV ≤0.06);
k=0.80 already fails (CV 0.17–0.20 with capture still perfect); run 6
reproduced the k=0.75 cell bit-for-bit inside the battery.

Phase-0 consequence, binding: production runs always have budget
slack, so **bare norm-CV misses real rings both ways** — soft splits
score ≈0.22, perfectly captured rings under slack score 0.17–0.43.
Ring evidence must be span-level plus gate-conditional (circular
decoding, Fourier structure, evaluated on the candidate feature's own
active tokens). Now in design §Phase 0.

### 2.4 The λ window opens with data

Operating-point veto table (10M tokens; tolerance 0.02, retention
floor 0.85×base):

| λ | share error | overlap | admissible |
|---|---|---|---|
| 0 (base) | 0.0002 | 1.0000 | — |
| 3e-4 | 0.0013 | 1.0000 | yes |
| 1e-3 | 0.0097 | 0.9548 | yes |
| 3e-3 | 0.0457 | 0.8573 | no (share error) |

At 3M tokens (runs 1–2) the admissible set was empty — recovery
overlap collapsed 0.95 → ~0.71 at the smallest nonzero λ — and the
λ=0 fallback fired. At 10M tokens the collapse does not occur: overlap
holds ≥0.95 through λ=1e-3, and the binding constraint at 3e-3 is
genuine share concentration, not rank compression. Per the frozen
protocol (largest admissible λ = primary): **Phase-1 primary is
λ=1e-3**, with 0 and 3e-4 as the sweep's lower arm. The mechanism
sequence is worth keeping: run 1's "share concentration" was largely a
Frobenius parked-capacity artifact; run 2's honest shares moved the
harm to overlap collapse; run 5 showed the collapse itself was a
data-starvation artifact. Each instrument fix relocated the story —
the design's caveat that "a benign-λ window may open with more data"
is exactly what happened, at values already in the grid.

### 2.5 Instrument rules for later phases

- **Contribution-energy shares, never Frobenius.** Frobenius shares
  misread one-hot decoys as ~0.37-error "shared" profiles (parked
  frame capacity); contribution shares read 1e-4. Binding on all
  recovery/depth readouts and the Phase-2 `share` export.
- **Spans are the rotation-stable readout; spectra are not.** Rotated
  vs control inits: span agreement ≥0.99, spectrum rel-diff
  basin-dominated. Claims anchor on spans; per-block spectra are
  basin-sensitive diagnostics.
- **Norm-CV is never a ring detector by itself** (§2.3).
- **Packed blocks are ordinary** (§2.2); the share-split signature is
  the flag.
- **Verify the effective config, not the intended config** (§6.2): the
  report's embedded `battery_config` exists to be read.

## 3. The basin taxonomy

At spare capacity every (seed, planted block) lands in one of these
basins — the campaign made the taxonomy and then located the training
conditions under which "captured" is the only occupied basin:

- **Captured** — overlap ≥0.999, code R² ≥0.97, rank exact, geometry
  regime preserved.
- **Tiled** — curved geometry splits into gated arcs: span found,
  matched-piece code R² ~0.6, learned rank 1; the union of pieces
  captures the block. Loose-budget junk-fill is the driver (§2.1);
  aux-invariant.
- **Split** — flat blocks shed rank into sibling blocks (classic SAE
  feature-splitting, observed here for subspace units).
- **Merged/packed** — sub-width features union into one block (§2.2);
  a converged optimum, not an accident.
- **Soft phase-split** — the manifold-specific merge dual: one feature
  amplitude-shared across two blocks (§2.3, layer 1).
- **Homeless** — a planted block not found at all; the init lottery
  that spare capacity at tight budget eliminates (§2.1). Dead blocks
  are only pathological when alive-count < planted count; at exact
  budget a revived block can only survive by displacing an incumbent,
  so AuxK revival is structurally weak exactly when needed.

## 4. Gate semantics — ruled: strict capture-as-written

The battery's hard gates demand full capture per planted block; the
3k-era learner delivered capture only for flat rank ≤3 blocks. Options
considered: (1) capture as written; (2) split semantics — capture
where achievable, span-recovery + support-union elsewhere, homeless
still hard-fails; (3) match capacity to the zoo (G=F — kills the
splitting basin but teaches to the test). Recorded lean (fable):
option 2. **Ruling (a9, 2026-07-16): option 1 — strict
capture-as-written.** The gate stands; Phase −1 does not pass on tiled
or split recoveries; the consequence is a capture *campaign*, not gate
surgery — find the training conditions under which capture is
seed-robust and re-run the battery inside them. That campaign
(§§2.1–2.3) is what produced the operating point, and the battery
passed with gate criteria untouched.

## 5. Smaller confirmations

- 8-bit-Adam × retraction ordering (pulled forward from the 0.9
  gate): passed — post-cast Gram residuals at bf16 noise under the
  production optimizer.
- Dead-block revival unit test: aux revives, no-aux control stays at
  exactly zero (encoder-side starvation confirmed as the death mode).
- Seed determinism: bit-identical recovery across re-runs in every
  honest comparison (runs 1↔2; sweep↔battery at matched configs;
  run 6 ↔ round-8 cell).

## 6. History — the run ledger, with corrections

### 6.1 Runs 1–2 (3k steps; the instrument fixes)

Run 1 → run 2 was a pure re-measurement (bit-identical overlaps)
isolating three scoring fixes: contribution-energy shares (replacing
Frobenius), gate-level bundle association (replacing per-member
Hungarian matching, meaningless for co-active bundles), and a
seed-variance control arm for the rotation gate. Post-fix verdicts at
3k: core FAIL (0.5 captured — shells tiled, rank-4 split), decoys FAIL
(splitting + one homeless), bundle FAIL (seed lottery), rotation FAIL
(n=1 control arm), λ-veto "PASS" via empty-set fallback (λ=0
primary — later superseded, §2.4), frequency ladder degraded
everywhere. Every red traced to the basin lottery (§3), not to
instruments — which is what motivated the campaign rather than a
harness rewrite.

### 6.2 Runs 3–4 and the CLI steps bug (a retraction)

Battery runs 3 and 4 silently ran at 3000 steps: `run_phase_minus1.py`'s
`--steps` default shadowed `BatteryConfig`'s 10k operating point
(fixed — CLI flags now fall through; `--out` always writes). Run-3
core proved bit-identical to the sweep's 3k cell, so the interim
theory that marginal basins were **cross-process chaotic (suspected
cuBLAS) is retracted** — the configs differed; determinism has held in
every honest comparison. Run-3's λ/frequency/AuxK readings were
thereby 3k-scale measurements, all superseded by run 5. Lesson kept in
§2.5: verify the report's embedded config.

### 6.3 Run 5 (first honest operating-point run): 6/7

Core, λ-veto (window open, §2.4), decoys (perfect under the rank-3
fixture), rotation (4/4) pass; frequency and AuxK report cleanly
(§1 table); bundle_null red on ring detection 1/4 — dissected in §2.3
and resolved by rounds 7–8.

### 6.4 Run 6: all hard gates pass

Bundle at the block-event pin: 4/4, ring CVs 0.047–0.060,
bit-identical to the round-8 k=0.75 cell; every other scenario
reproduces run 5. **Phase −1 green**; design re-frozen as v2.3 with
all of the above folded into the body.
