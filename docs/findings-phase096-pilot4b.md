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

## Remaining

1. **a9 ratification items**: 4b lr 3e-4 (amended optimizer point;
   tension: order tracks lr, ~4.5e-4 unexplored), site list
   (9,12,15,18,21,24,27,30 — depth-availability now measured, above),
   **F7 renorm designation** (4b evidence: FVU win + allocation
   reversal + only-arm-with-both-rings; cost: lower stability edge),
   spike guard / AuxK cap for Phase 1.
2. Parked analysis: eval_activation_stats + planarity screen on the
   3e-4 checkpoints (runbook, Analysis pass).
