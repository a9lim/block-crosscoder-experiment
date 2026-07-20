# Block-sparse crosscoders: Phase 0 findings

*The pilot program, 2026-07-15 → 07-19, condensed. Every claim links
its primary source in [`archive/`](archive/README.md); the design and
the Phase-1 plan live in [`design.md`](design.md). Status: complete
except two closeout tranches (§Open items).*

## Abstract

A **block-sparse crosscoder (BSC)** is dictionary learning whose
atomic unit is a *b*-dimensional subspace with **one shared code
across depth**: G blocks, per-site decoder frames, BatchTopK selection
on exact contribution energy under a Gram constraint
(Σ_s D_g^s D_g^sᵀ = I_b). It occupies the empty {block} × {cross-site}
cell of the sparse-dictionary literature's 2×2. Phase 0 built the
architecture, validated it on synthetic ground truth, and ran it on
gemma-3-1b and gemma-3-4b pilot stores. Findings: (1) the
architecture's premise — concept subspaces whose *frames rotate with
depth while the code persists* — is directly observable in gemma
before any training; (2) the ring structure that motivates subspace
units exists in scalar SAE dictionaries but sits **below every
clustering threshold**, so post-hoc blockification cannot reach it —
native block training can, and does: trained BSCs capture the month
ring, the cardinal number line, and a world-atlas map **as single
blocks** with code geometry at the permutation floor; (3) on the
honest rate–distortion axis, cross-site code tying is a **7.8–7.9×
rate reduction at zero distortion cost**, the tying × blocking
interaction is positive, and the site-renormalized BSC **strictly
dominates the matched scalar frontier everywhere they overlap**
(~390 → ~1,600 bits/token) — the matched-L0 "block tax" inverts once
support bits are priced; (4) the production training stack is settled,
bit-deterministic, and its failure modes are partitioned exactly
between three independent mechanisms. Phase 1 — the full-size BSC on a
53M-token store — waits only on hardware.

---

## C1. The shared-code premise holds in gemma before any training

*(Cross-layer coherence probe on pretrained gemma-scope SAEs, layers
9/17/22/29 of gemma-3-4b, paired 4M-token stream —
[`archive/findings-phase05-cross-layer.md`](archive/findings-phase05-cross-layer.md).)*

Month-family codes correspond **linearly across depths** while the
frames that carry them rotate to chance alignment: held-out
cross-layer code-map R² reaches 0.83–0.90 across the 9↔22↔29 triangle
(CCA ≥ 0.96 for 9↔22) while raw decoder-subspace principal angles sit
at their random-subspace null (p ≈ 0.33–0.52). One pair (22→29) also
span-matches (p = 0.001). **Frames rotate, the code persists** — the
exact object a shared code + per-site frames parametrizes, observed
in the wild before a single BSC gradient step.

The same probe rewrote the depth story: activation-space calendar
rings live *early* (layer 9: weekday circular score 0.981, decoder
adjacency |r| 0.886), layer 22 is the ring-visibility *minimum*, and
layer 17's dictionary undersplits both families — the origin of the
standing rule **never judge structure through a single site's
dictionary**, and the reason the production site list brackets depth
from 26% to 88% (layers 9–30).

## C2. The structure exists, but post-hoc discovery cannot reach it

*(Positive control + discovery runs at 16k/65k —
[`archive/findings-phase0-control.md`](archive/findings-phase0-control.md),
[`archive/findings-phase0-gemma.md`](archive/findings-phase0-gemma.md).)*

The pipeline replicates Engels' weekday/month/year rings on the exact
GPT-2 SAE Engels used (all three families at the permutation floor
p = 0.005), so its nulls are interpretable. On gemma-scope-2-4b layer
22, discovery is **null at both 16k and 65k**: no clustering branch
ever surfaces a multi-member ring candidate (BH-flagged: none), and
the skeleton features are τ=0.5 graph singletons. But the month ring
**exists decoder-side at 65k** — 12/12 distinct top-1 features whose
decoder vectors are cyclically ordered (adjacency p = 1.5e-4,
Fisher–Lee angle order p = 3.5e-4) at max adjacent cosine **0.32**,
structurally below every cosine-clustering threshold. A bonus layer-9
run nulls identically, so the gap is depth-general.

This is H1's sharpened form and the project's motivation in one
sentence: *the ring rides in the dictionary at cosines no post-hoc
method can cluster, so reaching it requires learning the block
jointly* — which is what a BSC does.

## C3. A trained BSC captures concept manifolds as single blocks

*(1b interim analysis + 4b pilot probes —
[`archive/findings-interim-artifact-analysis.md`](archive/findings-interim-artifact-analysis.md),
[`archive/findings-phase096-pilot4b.md`](archive/findings-phase096-pilot4b.md),
[`archive/findings-phase096-tier-a.md`](archive/findings-phase096-tier-a.md).)*

- **The month ring, 1b** (block 23 of the calibration winner): fires
  on 53% of month tokens vs 0.2% background; 12/12
  calendar-adjacent class means in its top code plane (97% of
  class-mean variance, p < 5e-5, split-half stable); the block's
  rotating per-site frames hold the ring at **all six sites**,
  including depths where the raw stream's top plane has lost it.
- **At 4b** (12M optimizer tokens): the renorm arm's b595 claims
  10/12 months with code ring 10/12 at the permutation floor *and*
  weekday 7/7 at the floor — the only arm capturing both calendar
  families. The primary arm's b2146 captures the **cardinal number
  line** (17/20 top-1, code order rho 0.90 at the floor); digits
  fully individuate (one block per digit); renorm binds numbers
  **across notation** (3/third, 4/fourth, 6/sixth, 7/seventh
  same-block). The renorm arm's b1781 is an **atlas block**: 36/48
  countries top-1 with latitude/longitude LOO-decodable from its
  4-dim code (lat 0.34 / lon 0.15, p = 1e-3).
- **The matched scalar baseline carries the same information without
  unit-level individuation**: month top-1 features collapse to 4
  distinct (1b) / 7 (4b), weekday to a single feature — the
  information is present as a population code but no unit is the
  manifold.

Two structural caveats are part of the claim, not qualifications of
it. **Capture and order are different statistics**: single-block
consolidation is near-universal at sufficient budget (every 1b
seed ≥ 11/12 top-1), but calendar *order* inside the block is a seed
lottery (ring 12/7/10/2/3/12 across six seeds), and block identity is
init-determined. And **consolidation-without-order is a failure
signature** — a destroyed dictionary produced a mega-block claiming
all 12 months with ring 6/12, and a *healthy* run claimed weekday 7/7
top-1 with ring 2/7 — hence the standing **mega-block rule**: top-1
capture is never read without ring order and FVU beside it.

## C4. Cross-site tying is a ~7.9× rate cut at zero distortion cost

*(Single-site R-D placement with exact eval-split weights —
[`archive/findings-phase099-tranche1.md`](archive/findings-phase099-tranche1.md);
figure [`../figures/phase0/rd_tying.png`](../figures/phase0/rd_tying.png).)*

The factorial's single-site cells (8 independent per-site models,
exactly parameter- and rate-matched) price what the shared code buys.
Reconstructing all 8 sites with independent per-site models costs, at
q=4: **block side 6,031 bits/token vs the joint BSC's 772.7 (7.8×);
scalar side 12,509 vs the joint crosscoder's 1,588 (7.9×)**. Pooled
with exact eval-split sq_tot weights, the joint block model
**strictly dominates** its 8-model control in the whitened gauge
(0.4360 @ 772.7 vs 0.4559 @ 6,031) and exactly ties it in the renorm
gauge (0.4207 = 0.4207) — tying is distortion-free in both gauges.
One support set and one amplitude vector serve all eight depths; the
support-bit amortization that funds this is measured at **4.08×**
joint (261 vs 1,067 support bits/token) and replicates *inside* the
single-site cells (246 vs 1,051 per site — 4.26×).

## C5. The R-D frontier: renorm strictly dominates scalar everywhere they overlap

*(Preregistered codec + λ=0 frontier, k ∈ {16, 32, 64} × three arms —
[`archive/findings-phase099-tranche1.md`](archive/findings-phase099-tranche1.md);
figure [`../figures/phase0/rd_frontier.png`](../figures/phase0/rd_frontier.png).)*

At matched latent-L0 the scalar arm "wins" FVU by 0.047 — the
apparent block tax. Priced at matched **bits** through the
preregistered codec (canonical orientation, calibration-quantile
clipped uniform quantizer, frozen count model, realized counts,
sequence bootstrap), the story inverts:

| region (q=4) | block arm | scalar arm | verdict |
|---|---|---|---|
| ~390 bits | renorm k16: 0.4869 @ 390.7 | *(none — needs k≈6)* | blocks extend the frontier into the ultra-cheap region |
| ~770–820 | renorm k32: 0.4207 @ 770.5 | scalar k16: 0.4306 @ 822.0 | **renorm strictly dominates** |
| ~1.5 kbit | renorm k64: 0.3660 @ 1491.6 | scalar k32: 0.3718 @ 1588.4 | **renorm strictly dominates** (CIs disjoint) |
| ~2.9 kbit | *(none — needs k=128)* | scalar k64: 0.3249 @ 2900 | open at the expensive end |

Quantization is nearly transparent (q=6 reproduces unquantized FVU to
the third decimal; q=4 costs +0.004–0.005), the count model is
non-load-bearing (Bernoulli within ~5%), R-D positions are seed-stable
(spreads at or below CI width), and the renorm-over-primary gap is
k-stable (−0.015 to −0.021 FVU at identical rates). This is the H3
*preview* (pilot store, 12M optimizer tokens); the 24M-token winner
already moves the k32 point to **0.4053 @ 771.2** (C10), and Phase 1's
frontier at production budget is the verdict.

## C6. The 2×2 factorial: the tying × blocking interaction is positive

*(Both single-site cells trained config-only at exact per-site
matching — [`archive/findings-phase099-tranche1.md`](archive/findings-phase099-tranche1.md).)*

| pooled FVU (topk) | cross-site (tied code) | single-site | tying effect |
|---|---|---|---|
| **block** (4096×b4, k32) | **0.4299** | 0.4497 | −0.0198 |
| **scalar** (16384×b1, k128) | **0.3682** | 0.3768 | −0.0086 |

Tying helps in *both* geometries (the shared code pools selection
signal across depth; it is not a constraint being paid for), and it
helps blocks **~2.3× more** — interaction term **+0.011 pooled FVU**,
replicating across 3 seeds (0.0198/0.0195/0.0193). The literature 2×2
that the project set out to fill is thereby not just filled but
*measured*: the combination earns more than the sum of its parts.

## C7. The gauge result: shrinkage whitening tilts, renorm restores

*(F7 lineage: fidelity audit → 0.9.5 arm → 4b pilot → factorial +
placement — [`archive/design-review-2026-07-17-fidelity.md`](archive/design-review-2026-07-17-fidelity.md),
[`archive/findings-phase099-tranche1.md`](archive/findings-phase099-tranche1.md).)*

The shrinkage whitener retains ~6% of per-dimension variance at
shallow sites vs ~29–32% deep, so the equal-per-dimension
reconstruction loss silently weights deep sites several times
heavier. Measured exactly on the 4b eval split: whitened-gauge
per-site sq_tot weights run **0.033 → 0.275** (site 30 alone carries
27% of pooled sq_tot; sites 9/12/15 together < 11%), while
**site-renorm weights are uniform to ±2%**. Three independent
evidence lines designate renorm:

1. **Fair-allocation control**: both single-site families' per-site
   FVU profiles land on the joint renorm profile at **r = 0.984**
   (vs r ≈ 0.4 against primary) — every fairly-allocated model shows
   the same intrinsic difficulty ordering; only raw-whitened joint
   training deviates.
2. **R-D dominance** (C5) plus the pooled-FVU win at the operating
   point (0.4154 vs 0.4299).
3. **Capture breadth** (C3): renorm is the only arm capturing both
   calendar families at 4b, and the cross-notation number binding is
   renorm-side.

The two arms are **different gauges of the same manifolds** — paired
token code maps between them beat the within-class permutation floor
for every captured family (cardinal R² 0.62, span cosine 0.80), and
block identity persists across the renorm toggle at shared init (both
arms' b1018 is the same '197x' block). Cost worn openly: renorm's
stability edge is lower (destroyed at lr 6e-4 where primary
recovered) — covered by the guard at the pinned 3e-4.

## C8. The stream carries far more captured-manifold supply than calendars

*(Exploratory catalog, explicitly non-gate:
[`archive/findings-phase096-pilot4b.md`](archive/findings-phase096-pilot4b.md).)*

- **Number lines are depth-pervasive and straighten with depth**
  (ordinal Spearman |rho| along PC1: 0.91 → 0.99 by layer 24;
  spacing linear, not logarithmic).
- **The Gurnee–Tegmark world map rides at every site** (country
  LOO lat R² 0.57–0.66 at every depth; longitude washes out at L30;
  zero continent clustering).
- **A mid-stack shear zone** (frame-rotation trough at L18–L21, 0.71
  vs 0.84/0.91 flanks) through which stream manifolds rotate hardest;
  captured blocks' frames track their manifold's rotation at
  r ≥ 0.92. Cross-depth frame coherence doubles as a training-health
  signature (destroyed runs decohere).
- **Uncaptured supply**: the planet sun-distance line exists
  in-stream (|rho| ≤ 0.88) with no block consolidating it at pilot
  budget; color shows a stable non-hue-wheel geometry; element order
  dips through the shear zone.
- **Packing cliques** (Jaccard > 0.9 co-firing blocks) are converged
  optima, not noise — at 4b they tile one early context-detector
  subspace (renorm's decodes as citation/date scaffold); Phase −1
  showed block width is a packing budget and merging *improves* with
  convergence. Near-50/50 share splits + degraded code-R² is the
  packing flag.

**These families (calendar, zoo, atlas) are burned as selection
criteria** — three analysis passes tuned on them; all confirmatory
capture claims route through the sealed six-family panel
(runbook-phase099 tranche 0, opened only at Phase-1 config freeze).

## C9. The training stack is settled and its failure modes partitioned

*(Phase −1 battery, 0.9 rehearsal, 0.9.5/0.9.6 calibration, 0.9.9
engineering campaign — full lineage in
[`archive/`](archive/README.md); the pinned stack table is
[`design.md §Settled parameters`](design.md).)*

- **4b training is bit-deterministic across runs** (8-bit Adam + CUDA
  included) — spike sites exactly reproducible, regression suites and
  forensics exact.
- **The failure-mode partition is exact, three mechanisms, no
  overlap**: operating-point instability → the **guard refuses** the
  run (> 5 consecutive skips); poison batches (the step-1600 event is
  batch-locked across three divergent trajectories) → the **guard
  skips** them; the SASA AuxK revival cascade (s_aux=256 slam
  amplifying to ~100% of the gradient, *self-defeating* — it re-kills
  its own revivals) → the **aux-ratio-cap 1.0 defuses** it (peak
  gradients crushed 100×+: 107.9→0.52, 527.7→2.53, 220,670→36.9;
  final dead 3.08%→0.098%; bit-inert on healthy trajectories). The
  cap does not rescue bad operating points — that separation is the
  point.
- **Zero guard events at lr 3e-4 across every campaign run**; the 1b
  optimum (1.2e-3) is catastrophically unstable at 4b — scale
  transfer of optimizer points is *not* assumed anywhere anymore.
- **θ calibration streams** (log-histogram quantile over the full 13M
  calib split; the 61 GB OOM case now runs in 19.5 GB and full-split θ
  is *closer* to target k), **prefetch 4** cuts data-wait 30%→12%,
  and the codec is validated end-to-end (C5).
- **Store/harvest facts**: 40,960 bytes/token exactly; the 53M-token
  production store is 2.171 TB (fits the 4 TB NVMe at ~45% headroom);
  harvest ≈ 3 GPU-hours at 5,000 tok/s; writer 205 MB/s; whitener
  drift on a +6M extension uniformly below the pilot's own held-out
  baseline; fp16 banned everywhere in the harvest path; healthy dead
  band at G=4096 is 0.1–0.15%.

## C10. Epochs vs fresh data: the optimizer-token budget does the work

*(Tranche 6, four cells at matched 24M optimizer tokens on the full
pinned stack — k=32, λ=1e-3, lr 3e-4 cosine, guard + streaming θ +
rcap 1.0, seed 0, 5856–5858 steps; codec passes q ∈ {4, 6, 8};
payloads `data/phase0/t6_*.json`, run reports on jobe under
`/data/runs/bcc-phase099`.)*

| pooled FVU (topk) | primary | renorm |
|---|---|---|
| 12M-token anchor (6M × 2ep) | 0.4299 | 0.4154 |
| **6M unique × 4 epochs** | 0.4102 | **0.3997** |
| **12M unique × 2 epochs** | 0.4089 | 0.4098 † |

Three clean cells give the verdict: **at this scale the optimizer-token
budget is what matters; data freshness is a refinement at the edge of
noise.** Doubling the budget by *epochs* buys −0.0197 (primary) /
−0.0157 (renorm); switching those repeats to *fresh* tokens adds only
−0.0013 more (primary, 0.4102 → 0.4089). The k16 bonus point agrees
(epochs doubling: 0.502 → 0.4832, −0.019).

† The fourth cell is not a clean read: the renorm×fresh run hit a
guarded loss-spike cluster at steps 2676–2687 (one skipped step, rec
loss transiently 2.8×, six near-misses; skip-rate 0.017% — within the
≤ 0.1% gate) and finished 0.0101 *behind* its epochs counterpart. The
same-order primary run over the same 12M store had zero guard events,
so "fresh data is worse in the renorm gauge" and "this run ate a
spike" are confounded; we do not read a freshness penalty from it.

The codec confirms at the rate axis: the epochs-renorm cell reaches
**0.4053 @ 771.2 bits/token** (q=4; 0.4003 @ 1026.3 at q=6), moving
the C5 frontier point down 0.0154 at unchanged rate versus the
12M-token champion (0.4207 @ 770.5). All four cells' support bits sit
at 261–265 bits/token — support cost is budget-invariant; the entire
improvement arrives as amplitude fidelity.

Phase-1 consequence: the 53M-token production store at 2 epochs sits
comfortably inside the regime where repeats and fresh tokens are
near-equivalent, so store size is not the binding constraint the
freshness-conservative reading feared. The epochs-renorm cell
(`bsc_lam0.001_seed0_G4096_k32_renorm_ep4_guard_rcap1`, 0.3997) is the
new program-best checkpoint and the promoted winner
(`data/phase0/winner.json`).

---

## Hypothesis status after Phase 0

| | statement | status |
|---|---|---|
| **H1** | rings exist, findable by post-hoc blockification of an SAE | **split verdict, sharpened**: the ring exists decoder-side (p ≈ 1e-4) but post-hoc discovery is structurally null at every probed depth — the positive artifact statement stands, the "findable post-hoc" clause is refuted, and native block training demonstrably reaches what clustering cannot (C2, C3) |
| **H2** | cross-layer subspace + position correspondence | **passed pre-training** (C1); the trained-BSC form (shared-code evals) is Phase-1 confirmatory |
| **H3** | blocks earn their parameters on the R-D axis | **preview strongly positive** (C4–C6): renorm dominates the scalar frontier in the full overlap region; verdict at Phase-1 scale |
| **H4** | depth-resolved effective-span geometry | instruments validated (contribution spectra, truncation ablations, shear-zone tracking); confirmatory numbers are Phase 1's |
| **H5** | manifold-level model diffing | untouched (Phase 3 stub; deferred) |

## Standing rules (binding on all future evals)

1. **Mega-block rule**: top-1 capture is never read without ring
   order and FVU beside it (C3).
2. **Burned families**: calendar/zoo/atlas are descriptive only;
   confirmatory capture goes through the sealed panel (C8).
3. **Norm-CV is never a ring detector by itself** — soft phase-splits
   score ≈ 0.22 and perfectly captured rings under budget slack score
   0.17–0.43; ring evidence is span-level and gate-conditional
   (Phase −1 §2.3).
4. **Contribution-energy shares, never Frobenius** — parked frame
   capacity poisons Frobenius readouts (Phase −1 §2.5); the Gram
   constraint makes decoder spectra *frame capacity*, not used
   dimension.
5. **Capitalization filtering for token-class probes** (the May
   contamination lesson: uncapitalized 'may' is 88% modal).
6. **Verify the effective config in the report artifact**
   (`battery_config` / `model_cfg`), not the intended CLI (two silent
   config-shadowing incidents caught this way).
7. **Never judge structure through a single site's dictionary** (C1's
   layer-17 cautionary tale).

## Open items

- **Tranche 7 tail**: whitener stability at the 5M production slice,
  late-layer bf16 tail stats, renorm-scalar stability across
  independent slices, store checksum drill (deferred for I/O quiet).
- **Tranche 5** (guarded lr recovery, {4.5e-4, 6e-4} renorm-first):
  last, judged against the complete endpoint battery; re-ratification
  bar is a9's (runbook §Tranche 5).
- **Frontier ends**: block k128 (does scalar's 2.9 kbit lead
  survive?), scalar k≈6–8 (can scalar reach the 390-bit region?).
- **A renormed joint scalar cell** would close the one gauge
  asymmetry in C4's scalar-side comparison (one 30-min slot).
- **Phase 1 store commit**: waits only on the 4 TB NVMe install.

## Provenance

Primary sources, verbatim, in [`archive/`](archive/README.md) — the
design reviews (4 rounds, ~80 findings), ten findings documents, and
the frozen design v2.4 whose decision log records every ratification
(a9: strict capture-as-written gate semantics; λ=1e-3; lr 3e-4 cosine
at 4b; site list 9–30; F7 renorm; aux-ratio-cap 1.0; the 0.9.9
charter and its purge authorizations). Compact committed evidence:
`data/phase0/` (R-D payloads, exact eval weights, single-site
placement, SAE-era provenance, the Phase −1 battery report). The
current best checkpoint is always `data/phase0/winner.json`.
