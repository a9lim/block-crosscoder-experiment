# Phase 0.9.9 tranche 1 — engineering validation findings

**Status: in progress (overnight campaign 2026-07-18 → 19).** E1 and the
E3 revival axis are final; E2/E3-cascade sections fill as the jobe
campaign drains. Runbook: [`runbook-phase099.md`](runbook-phase099.md).
Raw artifacts: `/data/runs/bcc-phase099/` (jobe), `data/e3_revival_report.json`
(repo, committed).

## E1 — streaming θ-quantile: ALL GATES GREEN (3/3 checkpoints)

Deterministic log-histogram quantile (`StreamingScoreQuantile`: int64
counts, 2^20 log-spaced bins over [1e-9, 1e5], batch-order independent,
~3e-5 relative resolution) validated against the exact kthvalue
estimator on the three ratified pilot checkpoints:

| checkpoint | Δ avg-blocks (gate ≤ 0.1) | full-split RSS (gate < 30 GB) | θ agreement |
|---|---|---|---|
| bsc primary | **0.0005** | 19.5 GB | 4e-6 rel |
| bsc renorm  | **0.0007** | 19.7 GB | 7e-6 rel |
| scalar (the 61 GB OOM case) | **0.0043** | **19.5 GB** | 1e-5 rel |

The scalar arm — which OOM'd host RAM at 61 GB under exact calibration
with 64 batches — now calibrates over the **full 488-batch (2M-token)
calibration split** in bounded memory. Streaming is also ~2.6× faster
than exact on identical batches (4.6 s vs 12.1 s at 32 batches).

Bonus, consistently across all three arms: the full-split θ realizes
closer to target k than the capped exact θ (primary 32.09 vs 32.76
against target 32; scalar 128.51 vs 131.33 against 128) — the
whole-split quantile is not just feasible but *better*, which is the
Phase-1 posture (13M calib tokens, mandatory streaming).

## E3 second axis — revival retention on the Phase −1 battery: ADAPTIVE CAPS PASS, STATIC FAILS

`validate_e3_revival.py`, auxk_zoo (3 common + 3 rare f=0.005 planted
blocks, G=16 oversized/dead-prone, SASA C.1), Phase −1 operating point
(10k steps × batch 1024), seeds {0, 1}. Gate: every rare block the
uncapped control keeps must survive capping; dead-count slack 2.

| arm | rare kept (s0, s1) | dead /16 (s0, s1) | cap engagement | gate |
|---|---|---|---|---|
| control (uncapped SASA) | 2/3, 3/3 | 2, 4 | — | reference |
| **frac cap 0.5** | **3/3, 3/3** | 4, 5 | s_aux_eff hit 1 (hard) | **PASS** |
| **ratio cap 1.0** | 2/3⊇, 3/3 | 3, 1 | **never engaged** (α_eff ≡ 1.0) | **PASS** |
| alpha 0.5 (static) | 3/3, 3/3 | **9, 10** | — | **FAIL** |

Reading:

- **The static attenuation candidate is eliminated.** Halving α_aux
  keeps the planted rare features but kills 9–10 of 16 blocks — it
  suppresses the revival *mechanism* wholesale, exactly the failure the
  two-axis design was built to catch. A cap that suppresses cascades by
  killing revival fails here; alpha_aux < 1 does.
- **The frac cap engages routinely and costs nothing** — the revival
  budget collapsed to s_aux_eff = 1 whenever the dead set was small,
  and rare retention *improved* over control on seed 0 (3/3 vs 2/3).
- **The ratio cap is provably inert when training is healthy**: at the
  battery operating point the aux gradient never approached the main
  gradient, so α_eff never moved off 1.0. Its retention pass is the
  trivial one — which is precisely the property wanted from a
  guard-adjacent mechanism. Whether it bites during actual pathology is
  the cascade axis (r4/r6, 4b @ 6e-4).

The pin decision (a9, at Phase-1 config freeze) therefore reduces to
the cascade axis: if the ratio cap suppresses the 4b cascade, it is the
recommended mechanism (surgical: inert when healthy, active under
pathology); if only the frac cap does, the frac cap costs nothing
either. The revival axis no longer discriminates between them — it
discriminates them from the naive alternative.

## E2 — loss-spike guard regression suite

**r2, false-positive control @ 3e-4 (the ratified point): PASS.**
skip_rate 0.0, zero guard events — not even near-misses — over 2928
steps. Pooled FVU **0.4299** (topk) vs the unguarded pilot's 0.430:
replication to the 4th digit with guard armed + streaming θ, i.e. the
guard's no-op path and the streaming quantile are jointly a true no-op
on a healthy 4b run, not just in the unit tests. Dead 0.098% (pilot
band), both eval modes deterministic, streaming θ realized 32.15 avg
blocks at target 32.

**r1, 6e-4: the guard REFUSED the run at step 1018 — and the forensics
rewrite the 6e-4 story.** The run was clean to step 1008, then:

| step | event | grad_norm | rec |
|---|---|---|---|
| 1009–1012 | near-miss ×4 (grad-only, accepted) | 0.056 → 0.392 | 0.084 → 0.281 |
| 1013–1018 | corroborated spike ×6, skipped | 3.3–3.5 | 0.85–0.92 |
| 1018 | consecutive-skip cap → RuntimeError | — | — |

Three load-bearing facts:

1. **4b training is bit-deterministic across runs.** r1's trajectory
   matches the unguarded pilot 6e-4 run to 6 decimals in rec AND
   grad_norm for 1010 steps (8-bit Adam + CUDA included). Spike sites
   are exactly reproducible; the regression suite is meaningful. The
   summary's earlier "warmup-peak spike" attribution for this arm was
   wrong — the pilot 6e-4 primary spiked at ~1010 and ~1600, mid-run.
2. **The blow-up rides in on sub-threshold accepted steps.** Each ramp
   step was individually under the 5× rec trigger (0.28 vs median
   ~0.06 = 4.7× at step 1012); by 1013 the weights were blown (rec
   ~0.9 on six *different* fresh batches). Freeze-and-skip cannot
   rescue this class — there is no poison batch to skip — and the
   consecutive-skip cap correctly disambiguates it from the poison
   case (poison: next batch normal, one skip; blown weights: every
   batch anomalous, cap reached, refuse).
3. **The unguarded run "recovered" only by riding the blown state
   downhill** (rec 0.12 by step 1100, second excursion grad 527.7 at
   1600, final FVU 0.553 damaged). The guard converts
   damaged-but-finished into refused — which is the honest verdict:
   0.553 was a fail anyway. The guard's design goal was never to make
   6e-4 usable; it is (a) silence at the ratified point (r2 ✓),
   (b) poison-batch skip-and-recover (unit-tested ✓), (c) refusal to
   censor genuine instability (r1 ✓, live).

**Cascade anatomy (unguarded pilot 6e-4 telemetry, log_every=10):** the
excursion decomposes into seed + amplifier — step 1010 is a main-loss
wobble (grad 0.135, grad_aux 0.003); within 10 steps the wobble's
block-kills (dead 0.02% → 0.9%) trigger the s_aux=256 SASA slam and
**the aux gradient becomes 97–100% of the total** (step 1050: grad
107.9, aux loss 109; dead snowballs to 13% by step 1110). The step-1600
event is separate: from a healthy state, one batch → grad 527.7,
entirely aux. The E3 caps attack exactly the amplifier stage — r4/r5/r6
ask whether removing the amplifier leaves the seed wobble sub-critical.

**r3, 1.2e-3: PASS (the crash is the pass condition).** Guard refused
at step 766 (cosine-decayed lr 9.22e-4): near-miss ramp 755–757, first
skip 758, two grad-only near-misses (the skip bought a beat), then six
consecutive corroborated spikes 761–766 → RuntimeError. By
bit-determinism the unguarded 1.2e-3 run's first excursion was also at
step ~755 — **both unstable arms blow mid-run at partially-decayed lr**,
not at the warmup peak as the earlier postmortem shorthand had it,
consistent with the interim-analysis "the lr cliff is a mid-run
instability".

**E2 verdict: the guard suite is complete and green** — silent at the
ratified point (r2), refuses genuine instability rather than censoring
it (r1, r3), skips poison batches and recovers (unit test). Skip-rate
≤0.1% is confirmed viable as a Phase-1 gate: the healthy point produced
literally zero events.

## E3 cascade axis — caps at the 4b 6e-4 reproduction point

*(pending: r4 ratio cap, r5 frac cap, r6 renorm + ratio cap; unguarded
references — 6e-4: 0.553 damaged / renorm 1.105 destroyed; clean 3e-4:
0.430 / renorm 0.415 / scalar 0.368)*

## E4/E5/E6 — offline-validated (commit `99bf1c1`)

- **E4 site-subset view**: `StoreReader(sites=[...])` slices after
  shard load, so RNG consumption is identical to the full read — the
  subset stream is provably the joint stream sliced (tested), which is
  the factorial's matched-data guarantee.
- **E5 prefetch**: order-preserving daemon-thread wrapper, exceptions
  re-raised at the consumption point (tested). Justified same night:
  the guard runs measure 30% data-wait.
- **E6 store extension**: replay the pinned corpus stream, fast-forward
  past the pilot's row-consumption upper bound (10,805 + 64 margin),
  harvest under the frozen whitener, merged `train12m` manifest
  (tested read path). Not yet run — ~245 GB, /data has 506 GB free.

## Tranche 2/3 machinery landed same night (commits `99bf1c1`, `06a2977`)

- `run_phase099_single_site.py`: both single-site factorial cells, all
  8 sites in one store pass (shared iterator, per-site slice), exact
  per-site G/b/k/λ/lr matching, lockstep checkpoint/resume
  (smoke-tested both arms + resume).
- `codec.py` + `validate_rd_codec.py`: the full preregistered R-D
  codec (orientation, clipped uniform quantizer, frozen count model +
  enumerative support bits, Bernoulli sensitivity, active-count floor,
  sequence bootstrap), 6 offline tests including gauge-rotation
  invariance (R13). First real-checkpoint numbers pending GPU.
