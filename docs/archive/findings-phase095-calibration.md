# Phase 0.9.5 findings — optimizer calibration on the 1b store

**Date:** 2026-07-17. **Hardware:** jobe (4090). **Store:** the Phase-0.9
13M-token whitened store (`/data/stores/bcc-phase09/gemma3_1b_6site_fineweb`,
gemma-3-1b, 6 sites {7,10,13,17,20,22}). **Runs:**
`/data/runs/bcc-phase095` — 31 total: 16 ladder arms, 6 lr-extension arms,
dead-dynamics arm, site-renorm arm, 7 conditional arms. All runs
bit-deterministic (8/8 modes), θ transfer clean everywhere
(threshold-mode avg blocks within 0.11 of target, thr-FVU within 3e-4 of
top-k FVU), bf16 shadow within 1e-5 of fp32 on every run.

Design authority: the 0.9.5 calibration addendum, design v2.3.2
(ratified 2026-07-17;
[`design-review-2026-07-17-fidelity.md`](design-review-2026-07-17-fidelity.md)).
Training config identical to Phase 0.9 except where a flag says otherwise:
G=1024×b=4 (BSC) / 4096×b=1 (scalar, matched latent-L0), k=16 blocks
(64 scalar), 3,906 steps × 4,096 tokens ≈ 16M tokens, 1k-step warmup,
8-bit Adam, λ=1e-3 (BSC) / 0 (scalar), seed 0 unless tagged.

## Headline

- **Phase-1 lr/schedule verdict: cosine, optimum at lr 1.2e-3, cliff at
  2.4e-3.** Both arms independently: FVU improves monotonically in lr
  through 1.2e-3, then falls off a cliff at one further doubling
  (BSC 0.4115 → 0.4584; scalar 0.3447 → 0.5004 with the campaign's only
  ladder deaths, 0.42%). The original {1,2,3,6}e-4 grid did not bracket
  the optimum — the two extension rungs (1.2e-3, 2.4e-3) were added
  mid-campaign to close it.
- **The schedule interacts with lr and the ordering flips.** linear_fifth
  (SASA B.3: hold peak, linear decay over the last fifth) beats cosine at
  every lr ≤ 3e-4 — cosine's immediate decay wastes effective lr when the
  peak is low — but turns over past 6e-4 on both arms (holding a hot lr
  for 80% of training overshoots). Cosine's best beats linear_fifth's
  best on both arms. **Phase-1 default: cosine.**
- **Encoder weight decay 1e-3 is a no-op** at the two best settings
  (Δ pooled FVU −0.0003 / +0.0002, both within seed noise). Keep 0.
- **Seed noise is ~5× smaller than the winner gap.** Four seed-1
  replications: max |Δ| 0.00054 pooled FVU. The 1.2e-3-over-6e-4 gap is
  0.0028 (BSC) / 0.0019 (scalar). The lr ranking is not a seed artifact.
- **λ=1e-3 remains ~free at the optimum**: λ=0 confirmation at the BSC
  winner scores 0.41132 vs 0.41154 — Δ within seed noise, consistent with
  the 0.9 λ-ladder. The rate–distortion protocol's largest-admissible-λ
  choice survives recalibration.
- **Dead dynamics engaged at G=4096** (k=32, preserving Phase-1
  k/G = 0.78%): 4/4096 blocks (0.098%) dead in the final window, AuxK
  active. Per the ratified conditional, the k=16 stress arm is **skipped**
  (deaths occurred). AuxK's C.1 criteria kept mortality to 0.1% at 4× the
  block count — no pathology to reject at 1b scale.
- **Site-renorm arm (F7 decision data): pooled FVU is a wash, the
  allocation reverses.** Renorm 0.4230 vs baseline 0.4220 at matched
  config — but the per-site profile flips: baseline spends capacity on
  deep sites (per-site FVU [.414 .522 .512 .392 .383 .434] for sites
  {7,10,13,17,20,22}), renorm on shallow ([.322 .435 .455 .410 .431
  .485]). Measured scalars [4.12 4.12 3.26 2.07 1.89 1.78] confirm the
  ~5× retained-power skew (F7). The renorm decision is a9's, pre-4b-store;
  the data says: it costs nothing pooled and redistributes reconstruction
  quality from deep to shallow sites.

## Full ladder (pooled FVU, top-k eval, 1M held-out tokens)

BSC (G=1024, b=4, k=16, λ=1e-3):

| lr | cosine | linear_fifth |
|------|--------|--------------|
| 1e-4 | 0.4567 | 0.4396 |
| 2e-4 | 0.4302 | 0.4242 |
| 3e-4 | 0.4220 | 0.4214 |
| 6e-4 | 0.4143 | 0.4199 |
| 1.2e-3 | **0.4115** | 0.4226 |
| 2.4e-3 | 0.4584 | — |

Scalar (G=4096, b=1, k=64, λ=0):

| lr | cosine | linear_fifth |
|------|--------|--------------|
| 1e-4 | 0.3892 | 0.3711 |
| 2e-4 | 0.3622 | 0.3564 |
| 3e-4 | 0.3535 | 0.3524 |
| 6e-4 | 0.3467 | 0.3520 |
| 1.2e-3 | **0.3447** | 0.3552 |
| 2.4e-3 | 0.5004 | — |

Conditional arms (all cosine):

| run | pooled FVU | note |
|-----|-----------|------|
| bsc 6e-4 + enc-wd 1e-3 | 0.4140 | Δ −0.0003 vs no-wd |
| bsc 1.2e-3 + enc-wd 1e-3 | 0.4118 | Δ +0.0002 vs no-wd |
| bsc 6e-4 seed 1 | 0.4138 | Δ −0.0005 |
| bsc 1.2e-3 seed 1 | 0.41150 | Δ −0.00004 |
| scalar 6e-4 seed 1 | 0.3465 | Δ −0.0002 |
| scalar 1.2e-3 seed 1 | 0.3446 | Δ −0.0001 |
| bsc 1.2e-3 λ=0 | 0.4113 | Δ −0.0002 vs λ=1e-3 |

Secondary observations:

- θ decreases monotonically with lr on both arms (BSC 2.799 → 2.697
  across the cosine ladder): better-fit models have tighter pooled score
  distributions. θ at the cliff jumps back up (2.832) — a cheap
  divergence tell for Phase-1 monitoring.
- The H3 matched-L0 gap persists under equal tuning budgets at both
  arms' optima: scalar 0.3447 vs BSC 0.4115 (0.9 saw 0.353 vs 0.422 at
  the then-default lr). Verdict still deferred to 4b (design: H3 is a
  4b question; 1b block structure may simply be absent — see 0.5's
  depth findings).
- The dead arm's FVU (0.3165 @ 32 blocks, G=4096) is not
  ladder-comparable (different k, G); it exists for mortality dynamics
  only.

## Phase-1 recommendation (for a9 to ratify)

**lr 1.2e-3, cosine schedule, encoder-wd 0, λ=1e-3** — the measured
optimum on both arms, stable across 2 seeds, zero deaths, with the λ
choice re-confirmed at the winner. Two caveats, both already covered by
the plan:

1. The optimum sits one doubling below a hard cliff. If the cliff moves
   with the 4b store's distribution, the mandatory ≥3M-token exact-config
   pilot (D12/D13, which now carries lr-point confirmation) is the guard;
   6e-4 cosine (0.7% worse FVU, 4× cliff margin) is the documented
   fallback if the pilot shows instability.
2. The lr×G interaction is unmeasured at the winner: the dead arm ran
   G=4096 at 3e-4, not 1.2e-3. The scalar cliff run is the only
   high-lr × G=4096 point and it produced deaths. The pilot (exact
   Phase-1 config: G=4096 × b=4 × 8 sites at the chosen lr) resolves
   this before the store commit.

## Incidents

- **The dead arm OOM'd in θ calibration** on first run: 524k calib
  tokens × G=4096 × fp32 ≈ 8.6 GB of pooled scores next to the model on
  a 24 GB card. Fixed in `9bb4133` — `fit_threshold_` now accumulates
  scores on CPU; estimator unchanged (exact pooled quantile). The run
  resumed from its final checkpoint (training was complete; resume went
  straight to θ fit — the 0.9 checkpoint/resume gate paying rent). The
  Phase-1 streaming-quantile carry item stands: 13M calib tokens at
  G=4096 (~213 GB) exceeds host RAM too.
- The matrix script's fail-fast abort meant the renorm arm ran after the
  fix rather than in the original sequence; its report matches the
  pre-commit smoke run bit-for-bit on FVU and θ.
