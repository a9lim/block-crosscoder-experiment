# Phase 0.9.6 tier B — the D13 4b exact-config pilot (2026-07-17)

The design-mandated ≥3M-token exact-config pilot (design v2.3.2, D13) on
gemma-3-4b, run pre-NVMe on jobe's existing `/data` disk. Store:
`/data/stores/bcc-pilot4b/gemma3_4b_8site_fineweb` — sites
(9,12,15,18,21,24,27,30), 2M whitener + 2M calib + 1M eval + 6M train
whitened bf16 + 100k raw, 348 GB, harvested in 36 min (~6k tok/s,
`stop_at_layer` 31). Four training rounds same evening (rounds 2–4 were
incident responses; scripts preserved as
`scripts/run_phase096_pilot4b_round{2,3,4}.sh`). Run logs and reports:
`/data/runs/bcc-pilot4b/`.

## Verdict

**D13 passes at lr 3e-4 — and the pilot caught exactly the class of
problem it was chartered to catch.** The a9-ratified Phase-1 optimizer
point (lr 1.2e-3 cosine) is **catastrophically unstable at 4b**; the
documented 6e-4 fallback is **marginal** (spikes at warmup peak,
recovers damaged); 3e-4 is clean, reproducible across seeds, and passes
every gate. The production store commit can proceed once a9 ratifies
the amended optimizer point (below) and the site list.

## Gate table (at the clean 3e-4 config unless noted)

| gate | result |
|---|---|
| store checksums + whitening round trip | ✓ bit-exact (rel err 0.0, bf16 match 1.0000) |
| whitener stability, 2M slice (halves rel ΔW) | ✓ shallow/mid ≤ 0.011; **L27 0.026, L30 0.031** — depth-graded drift ⇒ production harvest should use its planned 5M slice |
| held-out spectrum vs shrinkage prediction | ✓ ≤ 0.010 through L21; L27 0.043, L30 0.060 (same depth grading) |
| checkpoint/resume at production config | ✓ (run at 1.2e-3: `--max-steps 900` → `--resume`, fast-forward replay, bit-compatible continuation) |
| θ transfer (threshold avg blocks vs k=32) | ✓ 32.15 (+0.5 %); scalar 130.8 vs 128 (+2.2 %). Spiked runs degrade it (33.3–34.9) — θ transfer is also a dictionary-health signal |
| eval determinism | ✓ 8/8-style repeat plus fp32-vs-bf16 shadow agreement to 4 decimals |
| dead blocks | ✓ 0.10 % (seed 0), 0.10 % (seed 1); renorm 0.15 % |
| train throughput | ✓ 10.5–10.7k tok/s, data-wait 31 % (better than 1b's 55–70 %; prefetch carry item stands) |
| host RAM | ✗→✓ two θ-calibration OOMs (below); Phase-1 streaming quantile now **mandatory**, not just carried |

## The warmup-peak instability (the pilot's central finding)

Every arm trained at ≥6e-4 spiked near the end of the 1000-step warmup;
no arm at 3e-4 spiked at all:

| arm | lr | max rec (step) | final pooled FVU | dead |
|---|---|---|---|---|
| BSC primary | 1.2e-3 | 2.38+ (~800) | **1.134** (destroyed) | 3.4 % |
| BSC renorm | 1.2e-3 | 7.5 (~700) | killed mid-run | 6 %+ |
| BSC primary | 6e-4 | 0.45 (1050); re-spike 0.98 (1600) | 0.553 (recovered, damaged) | 3.1 % |
| BSC renorm | 6e-4 | 94.5 (1180) | 1.105 (destroyed) | 7.1 % |
| scalar | 6e-4 | 12.3 (1020) | 0.545 (recovered, damaged) | 0.5 % |
| BSC primary | 3e-4 | 0.132 (=init floor) | **0.430** | 0.10 % |
| BSC seed 1 | 3e-4 | 0.132 (=init floor) | 0.430 | 0.10 % |
| BSC renorm | 3e-4 | 0.988 (=init floor) | **0.415** | 0.15 % |
| scalar | 3e-4 | (clean) | **0.368** | 0.02 % |

Mechanism, from the full step logs (`steps.jsonl` carries per-step
`grad_norm` / `grad_norm_aux` / `dead_frac_window`):

1. **Seed**: a main-loss edge-of-stability event at peak lr —
   `grad_norm` jumps ~20× before the aux term moves. The stability edge
   *drops as training sharpens*: the 1.2e-3 run crossed lr 6e-4 at step
   490 unharmed (coarse features) but blew at ~8e-4; the 6e-4 run blew
   exactly on reaching 6e-4 at step 1000; renorm (shallow-site
   amplification up to ~4×) blew earliest and hardest.
2. **Amplifier**: the SASA AuxK loop. Churn drops blocks below the dead
   threshold → aux slams s_aux=256 "dead" units toward the residual at
   λ_aux=1.0 → revived units disrupt BatchTopK slot competition → more
   units starve → dead fraction 0.1 %→6.7 % in ~60 steps while
   `grad_norm_aux` goes 0.0003→108. There is **no gradient clipping**
   anywhere in the loop (measured `grad_norm` 107).
3. **Not BSC-specific**: the matched scalar dictionary (no Gram, no
   rank penalty, b=1) spikes identically at 6e-4 (`grad_norm_aux` 175).
   It is a property of the shared optimization stack (adamw8bit + warmup
   + SASA-C.1 AuxK at λ_aux=1.0) at 4b scale. λ_aux=1.0 is
   SASA-paper-faithful and was validated at Phase −1/0.9 — this is an
   emergent-at-scale interaction, not a config bug.
4. **There is a data-driven trigger component** (seen in the figure
   pass): both 6e-4 arms re-spike at exactly step 1600 — BSC to rec
   0.98 (its *largest* excursion, above the warmup-peak 0.45) and
   scalar to 2.84 — on the shared store order, at cosine-decayed
   lr ≈ 4.6e-4, while the 3e-4 arms sail through the same batch.
   A specific batch blows up damaged dictionaries when lr is high
   enough. This favors the batch-skip form of the spike guard over
   pure clipping, and puts a measured re-spike threshold (~4.6e-4,
   damaged-dictionary) right beside the unexplored 4.5e-4 rung.

**Phase-1 consequences (for a9 ratification):**
- **4b optimizer point: lr 3e-4 cosine** (λ=1e-3, enc-wd 0 unchanged).
  The 1b-ratified 1.2e-3 does not survive contact with 4b. Tension to
  hold openly: at 1b, ring *order* peaked at the highest stable lr —
  4b runs at 3e-4 may under-order rings relative to their own edge.
  An intermediate rung (~4.5e-4) is unexplored.
- **Loss-spike guard graduates from carry-item to requirement**, with a
  concrete trigger measured (e.g. `grad_norm` > 20× trailing median →
  skip batch). Gradient clipping is the cheaper structural fix but is a
  new optimizer-dynamics knob — a design amendment, not a default flip.
- **AuxK at scale needs a cap**: even a clip-free mitigation (α_aux < 1
  or s_aux ∝ dead-set size) would break the cascade loop; any change
  moves off the SASA-faithful point and belongs in the Phase-1 config
  decision, not silently in code.

## F7 site-renorm at 4b (matched stable-lr pair)

Renorm 0.415 pooled vs baseline 0.430 — renorm *wins* pooled FVU at 4b
(1b: wash) — and the allocation reversal replicates: renorm best at the
shallow end (L9 0.303 → L30 0.526), baseline best deep (L24/27
0.38/0.39). Two caveats: renorm's stability edge is *lower* (destroyed
at 6e-4 where baseline recovered; its shallow-site amplification is
exactly the mechanism), and the 1b tier-A ring result at renorm was
inconclusive. Net: supports the renorm lean **conditional on the 3e-4
optimizer point**.

## H3 preview (matched latent-L0, k·b = 128)

Scalar wins pooled FVU by the same margin at both scales — 1b: 0.353 vs
0.422; 4b: 0.368 vs 0.430 (~0.06, ~15 %). The block tax is
scale-stable. What the tax buys is the individuation question — see the
calendar-probe section.

## Host-RAM incidents (both recovered via `--resume`, no retraining)

1. Tier-A G=8192 (1b): θ-calibration at 128 calib batches → 62 GB
   anon-rss → OOM-killed. Fixed with `--calib-batches 64`.
2. Pilot scalar (16384 latents, 4b): OOM-killed **even at 64 batches**
   (61.7 GB); needed 16. The calibration pipeline holds several
   score-matrix-sized copies; jobe has 61 GB. The Phase-1
   streaming-quantile implementation is now blocking for any
   G ≥ 8192 or scalar-arm production run.

Also fixed en route: cusolver batched-syev batch-count limit at
S·G ≥ ~32k (`efb4f9b`, chunked eigvalsh — required for any 4b G=8192
work; that config additionally needs its own OOM-safe calibration).

## Calendar probe at 4b

(from `calendar_probe.py --tag _pilot4b`, 8M-token scan disjoint from
the store head; ring tests via `tier_a_ring_tests.py` with the pilot
store/tokenizer/acts — capitalized-only labels, class-mean top-1/top-2
consolidation, top-plane adjacency statistic, 20k-perm null)

Run history: the 8M scan + acts npz completed overnight (18,398 month /
936 weekday labeled positions, 60k background); the in-script encode
phase then CUDA-OOM'd beside the still-loaded gemma-4b (8 GB bmm
chunks), so the code-level tests ran next morning from the saved acts —
`tier_a_ring_tests.py` on CUDA (which exposed a renorm-scalars
device-placement fix; tier A never hit it on CPU) plus the new
`depth_scalar_tests.py` companion covering what that script skips:
per-site depth availability and scalar individuation (no-background
family/overall ratio screen — the interim fallback, validated against
the z-score screen at 1b). Outputs `ring_tests_pilot4b.json` +
`depth_scalar_pilot4b.json` in `/data/runs/bcc-analysis/`, mirrored to
`data/analysis/`.

### Depth availability (raw whitened stream, class-mean top plane)

Month ring 12/12 at L9/12/15/21/24 (L18: 10/12), fading to **9/12 at
L27/L30**; top-plane var only 0.33–0.54 — order-perfect but not
plane-concentrated in whitened coords. Weekday ring **7/7 at every
site** (p at its 2.75e-3 floor): the 1b "weekday null" does **not**
transfer to the 4b stream (consistent with Phase 0.5's layer-9 weekday
circularity 0.981 — that was this model). Net: both calendar rings are
available across essentially the whole site list at 4b, weakest late —
the site list sees the band, and "early-only" undersells 4b in
whitened coordinates.

### Dictionary capture (code-level ring tests, 20k-perm null)

| arm | FVU | month top1 | top2 | code ring | perm p | weekday ring |
|---|---|---|---|---|---|---|
| primary 3e-4 | 0.430 | 3/12 (b1270) | 5/12 | 6/12 | 1.2e-2 | 2/7 ns |
| seed 1, 3e-4 | 0.430 | 7/12 (b705) | 11/12 | **10/12** | **5.0e-5** | 3/7 ns |
| renorm 3e-4 | **0.415** | **10/12** (b595) | 11/12 | **10/12** | **5.0e-5** | **7/7 (floor)** |
| 6e-4 (spiked, recovered) | 0.553 | 9/12 (b1270) | 12/12 | 8/12 | 4.5e-4 | 3/7 ns |
| 6e-4 renorm (destroyed) | 1.105 | 5/12 (b595) | 11/12 | 7/12 | 2.4e-3 | 5/7 |
| 1.2e-3 (destroyed) | 1.134 | 12/12 (b3964) | 12/12 | 6/12 | 1.2e-2 | 2/7 ns |

Readings:

1. **Capture consolidation is not universal at the pilot budget** —
   unlike 1b at 16M optimizer tokens, where every run ended ≥11/12.
   The pilot sits at 12M optimizer tokens at a forced-low lr; tier A
   showed order tracks total effective optimization, and 4b-at-3e-4 is
   simply lower on that curve. Pre-registered acceptable (runbook:
   "ring order a bonus at 8M optimizer tokens; pilot success = D13
   gates + any month-selective block" — several arms clear that bar).
2. **Renorm is the only arm that captures both calendar families**:
   month 10/12 with its code ring at the permutation floor *and* the
   full 7/7 weekday ring, at the best pooled FVU. Mechanistically
   coherent with interim §D: the rings live shallow, and renorm
   reweights the loss toward shallow sites. F7 now has **ring-side**
   evidence at 4b on top of the FVU win and the allocation reversal.
3. **Consolidation-without-order is a failure signature**: the
   destroyed 1.2e-3 dictionary's b3964 claims all 12 months —
   a mega-block in an FVU>1 dictionary — with ring only 6/12.
   Top-1 consolidation must never be read without ring order and FVU
   beside it (Phase-1 standing-eval note).
4. The **lr→order gradient replicates at 4b**: seed-0 baseline goes
   6/12 → 8/12 from 3e-4 to 6e-4 (despite the 6e-4 spike damage)
   before the edge destroys everything above. Sharpens the unexplored
   ~4.5e-4 rung question.
5. **Block identity is init-determined at 4b too**: seed-0 baseline
   arms converge on b1270 at both lrs, renorm arms on b595 at both —
   and renorm *redirects* which block wins at a fixed seed.
6. **Scalar smearing replicates (qualitative H3)**: clean-scalar month
   top-1 collapses to 7 distinct features (one claims
   Jul/Aug/Sep/Dec), weekday to **one feature for all 7 days**;
   population ring over the top-24 selective features only 5/12
   (p 0.048, plane 51%). Block units individuate what the scalar
   dictionary carries diffusely — caveat: the BSC baseline has its own
   seed lottery (3/12 vs 7/12), so the clean contrast is
   renorm-vs-scalar at this budget.

### Rings inside the captured frames (block-23-style test at 4b)

Projecting the labeled acts into the captured blocks' per-site decoder
frames (`fig_pilot4b.py`, stats in `fig_pilot4b_summary.json`): b595's
rotating frames hold the month ring at **6 of 8 sites** (10–12 hits at
L9–L15/L21/L24, 12/12 at L21; weakest 7/12 at L18/L27/L30, still
p ≈ 2e-3) vs block 23 holding it at all 6 sites at 1b — partially
block-23-style at pilot budget. b862's frames carry the weekday ring
**early only** (7/7 at L9–L15, decaying to 4/7 by L21+) even though the
raw stream holds 7/7 everywhere. Depth detail from the per-site planes:
the late-site month ring collapses *asymmetrically* — the autumn arc
(Sep–Nov) crushes into a knot at L27/L30 while spring stays extended —
and the two months b595 fails to claim (Sep→b3593, Oct→b2705) are
exactly in that knot: where the stream's ring degenerates, capture
splits.

### Figures (`figures/pilot4b/`, regenerate with `scripts/analysis/fig_pilot4b.py`)

| figure | shows |
|---|---|
| `p4b_b595_ring.png` | the 4b month manifold: labeled tokens + class means in b595's code plane |
| `p4b_weekday_ring.png` | b862's weekday ring |
| `p4b_capture_maps.png` | per-arm month claim maps — the lottery, renorm's near-sweep, the mega-block, scalar smear |
| `p4b_ring_depth.png`, `p4b_depth_planes.png` | raw-stream ring availability by depth; per-site planes with the autumn-knot collapse |
| `p4b_ring_in_frames.png` | raw vs in-frame ring by depth for b595/b862 |
| `p4b_instability.png` | the warmup-peak cascade + the step-1600 data-driven re-spike |
| `p4b_allocation.png` | per-site FVU: the renorm allocation reversal at 4b |

Honesty box: weekday classes are small after the capitalization filter
(~10² tokens/class); month ring p floors at 5.0e-5 (1/20001); six arms
× two families were tested — the floor-level results survive any
multiple-comparisons correction, the 1.2e-2 ones do not.

## The manifold zoo at 4b (2026-07-18, exploratory — not gate evidence)

Generalized probe over 7 single-token families
(weekday/month/ordinal/cardinal/digit/season/compass; the zoo tranche
in `phase0/labels.py`), 8M fineweb tokens, `--per-class-cap 4000`
(first-N; added after an uncapped scan OOM-killed jobe's 61 GB host
RAM), tested against both surviving 3e-4 dictionaries with
`zoo_block_tests.py` (consolidation + order per family: adjacency ring
for cyclic, Spearman |rho| along PC1 for linear, 20k-perm nulls).
Artifacts: `zoo_block_tests_zoo4b.json`, `zoo_means_zoo4b.npz`,
`zoo_codes_{renorm,primary}_zoo4b.npz`.

**Number-lines are depth-pervasive in the raw stream and straighten
with depth** — Spearman |rho| of class order along PC1 of class means:

| family | L9 | L12 | L15 | L18 | L21 | L24 | L27 | L30 |
|---|---|---|---|---|---|---|---|---|
| ordinal (20) | 0.91 | 0.92 | 0.95 | 0.97 | 0.98 | 0.99 | 0.99 | 0.99 |
| cardinal (20) | 0.92 | 0.94 | 0.94 | 0.92 | 0.93 | 0.96 | 0.98 | 0.98 |
| digit (10) | 0.79 | 0.87 | 0.90 | 0.90 | 0.88 | 0.94 | 0.92 | 0.93 |

PC1 spacing is closer to linear than log at every depth (cardinal
Pearson 0.94–0.97 linear vs 0.85–0.94 log) — though 1–20 is too small
a range to discriminate strongly against the log-scale literature.

**Dictionary capture (best block per family; the b595/b862 calendar
rows replicate tier A on this independent 8M sample):**

| family | renorm | primary |
|---|---|---|
| weekday | b862 6/7, **ring 7/7** (p 2.75e-3, floor) | b2982 **7/7 top-1, ring 2/7** (p 0.74) |
| month | b595 10/12, **ring 10/12** (p 5e-5, floor) | b1270 3/12, ring 6/12 (p 1.2e-2) |
| ordinal | b1393 6/20, 11 distinct, rho 0.16 (n.s.) | b382 **11/20, rho 0.60** (p 6.4e-3) |
| cardinal | b3194 16/20, rho 0.58 (p 8e-3) | b2146 **17/20, rho 0.90** (p 5e-5, floor) |
| digit | 10/10 distinct (full individuation) | 9/10 distinct |
| season, compass | no capture, order nulls | no capture, order nulls |

Readings:

1. **The cardinal number-line is captured as a single block in both
   arms** — primary b2146 claims two→nineteen with code-plane order at
   the permutation floor. The strongest non-calendar manifold capture
   in the pilot.
2. **Ordinals split by frequency band.** Primary consolidates an
   ordered 3rd–12th segment into b382 while the frequency giants
   'first' (n=4000, capped) and 'second' individuate. Renorm scatters
   the family across 11 blocks — planarity-screen hit **b3227 is real
   but is the rare late-teens band** (13th/14th/16th/17th/18th, n=5–77
   per class), a segment block, not the line.
3. **Digits individuate; no dictionary forms a digit line** (10/10 and
   9/10 distinct top-1 blocks) despite stream rho ≈ 0.9 — the
   inverse of scalar smearing: one *block* per digit class.
4. **Cross-notation number binding (renorm only):** 5 of renorm's 6
   cross-family top-1 blocks pair a digit with its ordinal form —
   3/third, 4/fourth, 6/sixth, 7/seventh, plus a round-number block
   b1808 (ten/tenth/twenty). Renorm trades the ordinal *line* for
   per-number *identity* blocks. Primary's single cross-family block
   is calendar-side instead: b1270 = March/April/May + 'spring'.
5. **Consolidation-without-order occurs in healthy runs too**:
   primary's weekday block claims all 7 days top-1 with ring 2/7
   (p 0.74). The mega-block rule (top-1 capture is unreadable without
   ring/order + FVU) is not just a destroyed-run signature — it
   applies to every capture claim.
6. **Arm pattern, hypothesis-grade**: renorm wins both calendar
   *rings*, primary wins both word-number *lines*. One seed per arm
   under a known consolidation lottery — a Phase-1 eval question
   (probe both families per arm), not a conclusion.

3D stacks (`figures/pilot4b/p4b_zoo_{family}_3d.html`, regenerate with
`fig_pilot4b_3d.py` — cyclic families get first-harmonic planes,
linear families PCA planes, consecutive depths Procrustes-aligned):
ordinal/cardinal/digit show the line manifolds down the full depth
stack; season/compass show the polysemy soup.

Honesty box: exploratory, one seed per arm; ordinal/cardinal class
counts span 3 orders of magnitude ('first' capped at 4000 vs
'fourteenth' n=5 — rare-class means are noisy); the first-N cap is
not a random sample; top-1 maps are soft attribution (argmax of
class-mean selection score); season/compass carry known polysemy with
no capitalization rescue and C=4 ring stats have almost no
permutation power.

## Cross-depth geometry & cross-arm correspondence (2026-07-18, second analysis pass — exploratory, not gate evidence)

Full-dictionary weight geometry (`extract_geometry.py`, now
`--runs`/`--sites`-general with chunked batched linalg for the scalar
G=16384 stack) over seven pilot checkpoints, plus the eval-split
evalstats, the paired zoo codes, and targeted frame dumps
(`dump_block_frames.py`, new). Analysis scripts:
`crossarm_tests.py` (new), `fig_geometry4b.py` / `fig_geometry4b_3d.py`
(new). Numbers: `geometry4b_summary.json`, `crossarm_pilot4b.json`;
contexts: `identity_contexts_{renorm4b,pilot4b}.json`.

### Depth allocation, weight-level (F7 replication)

Share-argmax over sites [9,12,15,18,21,24,27,30]: **primary parks
3407/4096 blocks' energy peak at L30** (seed1 3421, lr6e-4 3537,
destroyed 3673, scalar 11033/16384) with literally zero peaks at
L12–L18; **renorm spreads peaks across all eight sites**
([818,73,398,865,518,280,896,248]). The 1b renorm allocation reversal
replicates at 4b in the decoder weights themselves
(`p4b_geo_share.png`, `p4b_geo_share_3d.html`).

### The mid-stack shear zone, and frames that track the stream

Adjacent-site frame rotation (median top-2 principal cosine) is
U-shaped in depth for every healthy arm — ~0.84 at L9-L12, trough
**0.71–0.75 at L18-L21**, relocking to 0.90–0.91 at L27-L30 — the 4b
analogue, at matching relative depth, of the 1b L13→L17 shear zone.
The **stream's own manifolds rotate through the same trough, harder**
(month first-harmonic plane cosine 0.47 at L18-L21), and each captured
block's per-site frames **track their manifold's rotation profile at
r = 0.92–0.98** (month b595, weekday b862, cardinal b3194/b2146,
ordinal b382), each frame trough exceeding the dictionary median
exactly where its manifold's rotation does (`p4b_geo_rotation.png`).
"One shared code, rotating frames" is not just available at 4b — the
learned frames follow the model's own re-embedding of the manifold.

Cross-site stacked spectra: healthy blocks are **one slowly rotating
subspace** (participation ratio median 6.6 of [4 = rigid, 32 = fresh
per site]; renorm 8.6; scalar features rotate too, PR 2.0 of 8). The
destroyed lr1.2e-3 dictionary decoheres to PR 11.8 with mid-stack
rotation cosines 0.43–0.49 — **cross-depth frame coherence is a
training-health signature**, cheap to compute from weights alone
(`p4b_geo_dimensions.png`).

### Cross-arm correspondence (open item closed)

On the paired 82k zoo tokens, the two surviving arms' family blocks
correspond token-by-token, not just class-by-class: held-out linear
code maps primary→renorm beat the **within-class** permutation null at
the floor (p 5e-4) for every captured family — cardinal
b2146↔b3194 R² 0.62 (score-corr 0.82), ordinal 0.50, weekday 0.45,
month 0.40 — and their decoder spans align per site at mean top-2
cosine **0.80 (cardinal)** / 0.64 (month) / 0.58 / 0.50 against a
0.038 random-pair scale. Digits, per the individuation story, show
nothing (R² 0.02, span 0.13). Month score-corr is *negative* (−0.23):
b1270 covers only Mar–May, b595 the full ring — same coordinates,
different class coverage. The two dictionaries are, for these
families, **different gauges of the same manifold**
(`crossarm_pilot4b.json`, `p4b_crossarm_cardinal_3d.html`).

### Packing cliques at 4b (open item closed)

J>0.9 co-activation components on the 1M-token eval split: primary a
**14-block clique + a 4-block clique**, seed1 19+2, renorm **5** —
clique mass tracks the late-heavy allocation and renorm suppresses it.
Clique blocks all fire at ~0.0018 frequency with **rank-1 codes**
(census position PR≈1, top-2≈1.0). Their frames share spans at early
sites (within-clique mean top-2 cos up to 0.90 at L15) and go
near-orthogonal late (0.09–0.14 at L27–L30; union spectrum PR 61/72)
— **one early context-detector subspace fanning into a tiling of
distinct late-site output directions**, i.e. cross-block packing of a
high-dimensional late-site event, not the Phase −1 within-block
signature. Decoded contexts: the renorm clique is a
citation/date-scaffold family — b552 fires on whitespace inside
journal-reference strings, b819/b1825 co-fire on the pre-year slot
("As long ago as ␣", "…in October ␣"). Primary's clique blocks are
hyper-late (48% mean share at L30); renorm's small clique is
early-heavy instead (`p4b_geo_packing.png`).

### Shape-space census

Over all sane-frequency blocks (freq ∈ [1e-4, 0.05]; ~4070/4096 per
arm), code PR vs top-2 eigenvalue mass separates the named structures
(`p4b_census.png`): **rings sit at PR 2.3–3.1** with top-2 mass
0.73–0.82, **lines at PR 1.9–2.2**, **numeral-identity blocks at PR
1.4–1.6** (nearly 1-dimensional codes), **clique blocks pinned at
(1.0, 1.0)**, oddballs (b510 Latin, b2324 duration, b2987 magnitude,
b1219 dollar-digits, b3227 late-teens) on the same planar shelf as the
rings. Renorm has fewer near-rank-1 codes than primary (2.7% vs 7.0%
PR<1.5) and uses its four dims more fully.

### Identity-block decode (open item closed): they are numeral blocks

Top-24 contexts for the renorm cross-notation blocks revise the
binding story: each is a **numeral-digit block** whose strongest
firings are digits inside larger numbers, with the ordinal word form
riding along — b1018 ("7/seventh") fires overwhelmingly on '7' in
"197x" years, b3234 ("6/sixth") on "196x", b2820 on "15xx"
(Copernicus/Shakespeare-era years), b1609 ("4/fourth") on round
durations (30/45/90 minutes), b2407 ("3/third") on generic '3'
(Isaiah 3, DOI .3, 193x), and b1808 on **spelled round decades**
(forty/fifty/Sixty/Thirty, cross-case). The year flavor is corpus
frequency (fineweb-edu), not necessarily mechanism: what binds is
numeral identity across notations. Related singles: b1255 = digit '5'
inside ISBNs, b1393 = ordinal-suffix 'th' in "5th edition" citations,
b2446 = sentence-initial "One", b1512 = winter-habitat, b2295 =
"Central" as region head (`identity_contexts_renorm4b.json`).

Primary-arm decode (`identity_contexts_pilot4b.json`): b2146 fires on
cardinal words (Two/two — the line block as advertised); b382's
'third' firings are heavily the "third-party" collocation; **b1270's
top firings are all spring/Spring** — primary's cross-family block is
a *season* block that swallowed Mar–May class means, inverting the
month-first reading; b2982 fires across weekday names (capture
without order, lexically real); b127 is a genuine
north/west/east compass block (the zoo's C=4 order stat just has no
power); b349 = South/Latin compound-geography prefixes. Two
same-index coincidences that aren't: **primary b1018 and renorm b1018
are both the '197x' digit-7 block, and both b636s are quantity-digit
blocks** — the arms share seed-0 init, so tier A's "block identity is
init-determined" extends across the renorm toggle. Primary's clique
blocks decode only to weak near-theta apostrophe/function-token tails
on fresh text (scores 6–16 vs 20–80 for the named blocks) — their
defining 0.0018-frequency eval event is store-idiosyncratic, in
contrast to the renorm clique's crisp citation reading.

### Honesty box (this pass)

**May contamination (caught by a9, 2026-07-18):** the zoo means npz
applies no capitalization filter, and the month label map includes
lowercase forms — May's first-N 4000-token cap is **88% modal 'may'**
(every other month 93–100% capitalized). The zoo *ring statistics*
always filtered to capitalized tokens (May count 474 ✓) and are
unaffected; the contaminated means only reached visualization and the
month stream-rotation curve. Fixed: `fig_geometry4b{,_3d}.py` now
compute cap-only month means from the calendar probe — May rejoins
the ring (L24 in-plane radius 4.4 vs others' 3.7, from 7.6
contaminated; off-plane 6.3 vs 5.4, from 9.8), the stream-rotation
trough *deepens* slightly (0.47→0.41 at L18-L21), and b595 tracking
holds at r 0.97. `p4b_zoo_month_3d.html` (zoo-means view) still
renders the contaminated May; the calendar-probe month figures are
canonical. Cap-only still can't remove sentence-initial modal "May" —
the class stays slightly heavy.

Exploratory, one seed per arm; evalstats from the 1M-token eval split
in stored order; decode contexts are the top-24 score tail on 2M fresh
fineweb tokens (not class means — both facts can coexist with the zoo
top-1 maps); the 3D flow views use a fixed joint-PCA basis holding
only 31–38% of cross-depth class-mean variance and all 3D alignment is
viz gauge; census coordinates are spectrum summaries, not manifold
proofs; cross-arm "same manifold" claims rest on the paired-token maps
and span angles, with the season pair inheriting b1270's month
structure.

### New figures (this pass, `figures/pilot4b/`)

| figure | shows |
|---|---|
| `p4b_geo_share.png` / `p4b_geo_share_3d.html` | depth-energy allocation: the L30 cliff vs renorm's plateau |
| `p4b_geo_rotation.png` | the mid-stack shear zone; block frames tracking stream-manifold rotation r ≥ 0.92 |
| `p4b_geo_dimensions.png` | one-rotating-subspace blocks; collapse decoheres; scalar rotates too |
| `p4b_geo_packing.png` | clique census, code anisotropy, clique depth profiles |
| `p4b_census.png` | shape-space census with named manifolds on the planar shelf |
| `p4b_month_flow_3d.html` / `p4b_cardinal_flow_3d.html` | manifold drift through depth in one fixed basis (the component Procrustes stacks gauge away) |
| `p4b_crossarm_cardinal_3d.html` | the cardinal line through both arms' blocks, cross-arm mismatch rungs |

## Remaining

1. ~~a9 ratification items~~ **Ratified 2026-07-18** (design decision
   log): 4b lr 3e-4 cosine, site list, F7 renorm designated, AuxK cap +
   spike guard (batch-skip form favored by the step-1600 evidence).
2. ~~Parked analysis~~ **Run 2026-07-18**: evalstats (4 runs),
   planarity screen both arms (nameable families incl. ordinal block
   b3227, digit b1219, duration b2324, magnitude b2987), 8 PNG figures
   + 4 interactive 3D stacks (`figures/pilot4b/`, scripts
   `fig_pilot4b.py` / `fig_pilot4b_3d.py`).
3. ~~Zoo probe~~ **Complete 2026-07-18** — section above; 7 zoo 3D
   stacks added to `figures/pilot4b/`.
4. ~~Cross-arm correspondence, 4b packing cliques, code anisotropy,
   oddball + number-identity decodes~~ **Closed 2026-07-18** by the
   second analysis pass (geometry section above): the arms are
   different gauges of the same manifolds; cliques are cross-block
   tilings of citation/date-scaffold events; identity blocks decode
   as numeral blocks.
5. Still open: steering/causal validation of any discovered block
   (Phase-2 territory), the b1623 astonishment plane's affect
   structure, and whether the shear zone (L18-L21) matches where
   gemma-3-4b's induction/attention regime shifts.
