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
| BSC primary | 6e-4 | 0.45 (1050) | 0.553 (recovered, damaged) | 3.1 % |
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

**RESULTS PENDING — probe launched 23:45 on jobe (nohup, survives the
session), writes `/data/runs/bcc-analysis/calendar_probe_acts_pilot4b.npz`
+ `calendar_probe_codes_<name>_pilot4b.npz` for
primary_3e4 / renorm_3e4 / seed1_3e4 / scalar_3e4 / primary_6e4.**

Handoff to the next session (compaction expected here):

1. Check the probe finished: `ssh jobe 'tail /data/runs/bcc-pilot4b/probe.log'`.
2. Ring tests: `python scripts/analysis/tier_a_ring_tests.py
   --out-root /data/runs/bcc-pilot4b
   --acts /data/runs/bcc-analysis/calendar_probe_acts_pilot4b.npz
   --store /data/stores/bcc-pilot4b/gemma3_4b_8site_fineweb
   --tokenizer google/gemma-3-4b-pt
   --out /data/runs/bcc-analysis/ring_tests_pilot4b.json` (on jobe;
   bsc arms only — scalar individuation needs the top-1-latent count
   from the codes npz, per the interim Methods).
3. Fill this section: consolidation + ring per arm; scalar top-1 month
   feature count; compare depth availability (the 4b ring band should
   sit early per Phase 0.5 — sites 9/12/15 carry it).
4. a9 ratification items: **4b lr 3e-4** (the amended optimizer point),
   site list (9,12,15,18,21,24,27,30), F7 renorm designation, spike
   guard / AuxK cap for Phase 1.
5. Parked analysis: eval_activation_stats + planarity screen on the
   3e-4 checkpoints (runbook, Analysis pass).
