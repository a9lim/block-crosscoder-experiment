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

Unguarded references — 6e-4: 0.553 damaged / renorm 1.105 destroyed;
clean 3e-4: 0.430 / renorm 0.415 / scalar 0.368.

### Guarded integration arms (r4/r5/r6): guard and cap are orthogonal, by construction and now by measurement

**r4 (ratio cap 1.0 + guard, 6e-4): refused at step 1018 — the exact
registered prediction** ("the ratio cap won't engage during the
1009–1012 ramp; r4 reaches the same blown state at 1013 and the guard
refuses at ~1018 exactly as in r1"). The event ladder against r1:

| step window | r1 grad_norm | r4 grad_norm | rec (identical) | reading |
|---|---|---|---|---|
| ≤1011 (102 logged + near-misses) | — | **bit-identical to r1** | identical | cap inert when healthy, at 4b, live |
| 1012 (last accepted ramp step) | 0.3921 | **0.3798** | 0.2807 | first engagement — kills begin, cap trims only the aux component |
| 1013–1018 (skipped, frozen weights) | 3.26–3.55 | **0.95–1.03** | 0.85–0.92 | cap clamps the SASA slam ~3.4× (aux ≤ main, was 97–100% aux) |

Three conclusions:

1. **The ratio cap's inertness-when-healthy transfers from the battery
   to 4b** — bit-for-bit, through 1011 steps including the sub-threshold
   ramp. Its first engagement coincides with the first block-kills at
   step 1012, i.e. it activates exactly at the amplifier's ignition and
   not one step earlier.
2. **It suppresses the amplifier hard**: the frozen-state skipped steps
   see total grad ~1.0 instead of ~3.4 — consistent with clamping an
   aux gradient that is 97–100% of the total down to ≤1× the main
   gradient. Engagement tightens as the cascade grows, as predicted.
3. **The guard still (correctly) refuses.** The rec corroboration
   signal reads the blown *main* weights — identical to r1 at 4
   decimals on all six fresh skipped batches — and the seed wobble is
   main-loss-driven, so no aux cap can un-blow it. Cap and guard attack
   different stages (amplifier vs seed): the production stack wants
   both, and neither masks the other.

The decisive measurement — whether cap-suppressed dynamics ride the
wobble to a materially better *endpoint* — requires the guard off;
that is tranche 1b (r4b/r5b/r6b, unguarded, queued behind this
campaign).

**r5 (frac cap 0.5 + guard, 6e-4): refused at step 1050 — and the frac
cap is NOT inert when healthy.** The trajectory diverges from r1
starting at step ~250–260: the first dead block appears in the
dead-window at step 250 (1/4096, the normal early revival transient),
and from that moment the frac cap clamps s_aux_eff from 256 toward
~max(1, 0.5·n_dead), altering the aux loss on every subsequent step
(rec delta 4e-7 at step 260, growing). Everything downstream is a
different — not obviously worse — trajectory:

- The instability still arrives, ~8 steps later (near-miss ramp
  1017–1020 vs r1's 1009–1012): **6e-4 is unstable on a second,
  independent trajectory**, reinforcing that this is an
  operating-point property, not batch poison.
- **The guard's skip-and-recover path worked transiently, live**: a
  3-skip burst at 1021–1023 (rec 0.41–0.46) was followed by ~20
  accepted steps (rec back to 0.11–0.28) — the first in-vivo
  demonstration of the recovery path outside unit tests.
- A second, harder excursion then blew the weights (1045–1050, grads
  12–17.5, rec 0.62–0.84) → consecutive-skip cap → refusal. The large
  blown-state grads are consistent with the registered prediction that
  the frac cap *weakens* as the dead set grows (dead↑ → allowed
  s_aux_eff↑) — though magnitudes across different trajectories aren't
  directly comparable.

**r4 vs r5, the surgical criterion**: the ratio cap preserved
bit-determinism through 1011 healthy steps and engaged only at
amplifier ignition; the frac cap perturbs training from the first
routinely-dead block onward. Both refused under guard (correctly —
the seed wobble is main-loss-driven). On every axis measured so far
the ratio cap dominates: inert-when-healthy (bit-exact at 4b),
tightens-under-cascade (grad 3.4 → 1.0 at the blown state),
revival-retaining (battery PASS). The frac cap's remaining claim to
the pin would be a materially better *unguarded* endpoint — tranche
1b decides.

### Unguarded cap arms (tranche 1b): what the cap is actually for

**r4b (ratio cap 1.0, no guard, 6e-4): ran to completion through both
excursions.** Side-by-side with the unguarded uncapped baseline
(bit-identical seed wobble through step 1010, then):

| | uncapped baseline | rcap 1.0 |
|---|---|---|
| excursion-1 peak grad (step ~1050) | **107.9** | **0.52** |
| excursion-2 peak grad (step 1600) | **527.7** | **2.53** (≈ rcap geometry: aux ≤ main at rec-spike main-grad) |
| peak dead (kill wave, ~1110–1150) | 13.0% | 15.3% |
| dead by step 1590 (revival) | 6.1% | **0.17%** |
| final dead | 3.08% | **0.098% — the healthy band exactly** |
| final grad noise (step 2920) | 0.074 | 0.0039 |
| pooled FVU (topk) | 0.553 | **0.524** |

Four conclusions, sharper than the registered prediction:

1. **The cap does not rescue the endpoint** — 0.524 vs 0.553, both
   damaged against clean 3e-4's 0.430. The lasting damage is done by
   the main-loss wobble itself; no aux mechanism can prevent it. The
   guard's refusal remains the correct production posture for 6e-4;
   lr recovery is not what the cap buys.
2. **The cap crushes the amplifier by 2 orders of magnitude** (107.9 →
   0.52; 527.7 → 2.53) — the registered prediction (rcap tightens as
   the cascade grows) confirmed at full scale.
3. **The kill wave is NOT aux-driven — but chronic mortality is.** The
   cap did not shrink the kill wave (15.3% vs 13.0% peak: deaths come
   from the blown main weights). What it changed is what happens
   *after*: under the cap, revival works (15.3% → 0.17% before the
   second excursion; final 0.098%, indistinguishable from a healthy
   run) while the uncapped s_aux=256 slam is **self-defeating** —
   grad-100+ revival steps re-kill what they revive, and the baseline
   never gets below ~3–6% dead again. The SASA cascade's real
   production cost is chronic elevated mortality, and the ratio cap
   eliminates exactly that.
4. **The step-1600 excursion is batch-locked**: it fires at step 1600
   in both runs although their weights diverged at 1013 — a
   data-driven event (poison-batch class), precisely what the guard's
   freeze-and-skip path handles. Seed wobble → guard refuses the
   operating point; poison batch → guard skips it; cascade amplifier →
   cap defuses it. The three mechanisms partition the observed failure
   modes with no overlap.

**r5b (frac cap 0.5, no guard, 6e-4): completed; the registered
expectation was partially wrong, in an instructive way.** The
expectation — "at 13% dead the frac cap allows s_aux_eff ≈ 266 > 256,
no protection when the cascade is largest" — assumed the dead set
would grow as in the baseline. It didn't, because the cap's own
suppression is self-stabilizing: dead peaked at 11.7%, s_aux_eff
stayed well below 256, and the slams were damped ~6–13× (peak grad
24.2 at excursion 1, 41.0 at excursion 2, vs 107.9/527.7 uncapped).
Full three-way comparison at 6e-4 unguarded:

| | uncapped | fcap 0.5 | rcap 1.0 |
|---|---|---|---|
| exc-1 peak grad | 107.9 | 24.2 | **0.52** |
| exc-2 peak grad (step 1600) | 527.7 | 41.0 | **2.53** |
| dead at 1590 (revival) | 6.1% | 0.76% | **0.17%** |
| final dead | 3.08% | 0.12% | **0.098%** |
| pooled FVU (topk) | 0.553 | 0.549 | **0.524** |

- **fcap fixes mortality but not distortion**: final dead lands near
  the healthy band, yet FVU ≈ the uncapped baseline. rcap is the only
  arm that converts amplifier suppression into reconstruction gains
  (plausibly via its stronger mid-run revival — more live capacity
  through the second epoch).
- **fcap's trajectory perturbation is not free**: it altered the run
  from the first routine dead block (here even *preventing* the
  baseline's benign early deaths at steps ~1000–1010), which makes its
  healthy-run behavior a different operating point rather than the
  ratified one plus a safety net.
- **The step-1600 event fired at step 1600 on a third divergent
  trajectory** (rec spike 2.25 here, 1.19 rcap, 0.98 base) — the
  batch-locked reading is now triple-confirmed: that position in the
  fixed shuffled stream is intrinsically hostile, and any production
  run will meet its analogues. The guard's skip path is the designed
  handler; the cap bounds what the batch's kill-wave can cascade into.

**r6b (renorm + ratio cap 1.0, no guard, 6e-4): the cap does NOT save
the renorm gauge — final FVU 1.124 vs the uncapped 1.105, equally
destroyed.** The trajectory shows why, and completes the failure-mode
taxonomy. Renorm's 6e-4 failure is a *sustained* rec-led runaway: by
step 1000 both arms sit at rec ~2.65 with dead ~2.6% climbing to
~7.6%, and neither ever returns to a healthy state (rec grinds down
to ~1.1 — a destroyed dictionary, FVU > 1). Along the way the
uncapped baseline's aux monsters are colossal — grad 1,080 (step
1100), 5,424 (1200), peak **220,670** (1180) — and the cap crushes
all of it (peak 36.9, four orders of magnitude). It makes no
difference: under sustained main-loss runaway, the kill pressure is
main-driven, revival can't win at any s_aux, and the endpoint is
gauge-destruction either way. This failure class belongs entirely to
the guard, which refused it at step 977 (r6) — saving ~2,000 steps of
compute and refusing to hand downstream analysis a destroyed
dictionary that *looks* converged in FVU-vs-steps plots.

### E3 verdict and the cap-pin recommendation (for a9 to ratify at config freeze)

The full failure-mode × mechanism matrix, measured live at 4b:

| failure mode | guard | ratio cap 1.0 | frac cap 0.5 | static α 0.5 |
|---|---|---|---|---|
| healthy training (3e-4) | silent, FVU bit-replicates | **bit-inert** (1011 steps exact) | perturbs from first dead block | perturbs always |
| transient wobble + cascade (primary 6e-4) | refuses (correct: endpoint damaged regardless) | amplifier −100×, **mortality 3.08%→0.098%**, FVU 0.553→0.524 | amplifier −10×, mortality →0.12%, FVU neutral | (kills revival: 9–10/16 dead on battery) |
| poison batch (step-1600, batch-locked ×3) | skip-and-recover (demonstrated live, r5) | bounds the cascade it seeds | partially bounds | — |
| sustained rec runaway (renorm 6e-4) | **refuses (the only mechanism that matters)** | no effect on endpoint | untested (dominated) | — |
| rare-feature retention (battery) | n/a | PASS (trivially — never engages) | PASS (engages, free) | FAIL |

**Recommendation: pin `aux-ratio-cap 1.0`, alongside the guard, for
Phase 1.** The evidence chain: it is the only candidate that is
*bit-exactly* inert on healthy trajectories (the ratified operating
point is provably unperturbed — running with the cap is running the
ratified config); it engages precisely at amplifier ignition and
tightens as the cascade grows (registered prediction, confirmed); it
converts the SASA slam from self-defeating (re-killing its own
revivals, chronic 3–6% mortality) into functional revival (final dead
exactly the healthy band); and it is the only cap arm that recovers
reconstruction quality from the suppression. What it does not do —
rescue bad operating points — is the guard's job, and the pair
partition the observed failure modes exactly: guard handles seed
wobbles (refuse) and poison batches (skip), cap handles the
amplifier. The frac cap is strictly dominated at 4b (weaker
suppression, FVU-neutral, healthy-trajectory perturbation); static
attenuation is eliminated (revival-killing). Skip-rate gate for
Phase 1 stands as specified (healthy point: zero events).

**r6 (renorm + ratio cap 1.0 + guard, 6e-4): refused at step 977.**
The renorm arm's wobble is its own — earlier (near-misses from 963,
non-consecutive: 963/966/969/970, a ~14-step build) and **rec-led**:
rec climbs 1.28 → 4.34 while grads stay ~0.5–1.9 (already
rcap-clamped ≈1 at the blown state). The unguarded renorm run ended
*destroyed* (FVU 1.105) where the primary ended damaged (0.553); the
guarded ladder shows why — renorm's excursion is a runaway of the
main reconstruction itself, not a grad-spike event, and the guard's
rec corroboration catches it cleanly. All three guarded arms refused:
the guard's verdict on 6e-4 is unanimous across cap settings and both
gauges, and refusal timing/shape is trajectory-specific (1018 / 1050 /
977), not an artifact of one spike site.

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
