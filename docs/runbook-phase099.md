# Phase 0.9.9 — the pre-NVMe campaign (runbook)

**Chartered and ratified by a9 2026-07-18.** Everything worth running on
the pilot store (and a bounded extension of it) while Phase 1's
production store waits on the 4 TB NVMe. Provenance: fable
reconnaissance over the pilot artifacts + a Codex sol consult
(`cx-20260718-230829-9fe8`, 154k tokens of evidence in) whose main
correction — the campaign's center of gravity is factorial baselines,
honest rate–distortion, and robustness, *not* optimizer headroom — is
adopted. a9's amendments at ratification: the causal-intervention item
is **deferred** (not necessary yet); old stores may be **purged at
discretion** ("the more experiments, the better"); the factorial is
extended to the full literature 2×2 (BSF and ordinary-crosscoder
comparisons — see tranche 2).

**Progress (overnight 2026-07-19, autonomous; details in
[`findings-phase099-tranche1.md`](findings-phase099-tranche1.md)):**
tranche 1 COMPLETE — E1 green 3/3; E2 regression suite green (r1/r2/r3
+ discovery that 4b training is bit-deterministic across runs); E3
both axes done, **ratio cap 1.0 RATIFIED by a9 2026-07-19** (static α
eliminated, frac cap dominated; guard/cap/skip partition the failure
modes exactly; design pinned as v2.4); E4/E5 landed and dogfooded by
every subsequent run. Tranche 3 codec built, offline-tested (incl.
R13 gauge invariance); **λ=0 frontier trained and priced** (k∈{16,32,
64} × both arms; renorm k32; renorm k16/k64 running 07-19): renorm
strictly dominates the scalar frontier in the ~800-bit overlap
region, primary ties mid-region, support amortization ≈4× across k,
tying ≈7.8× rate cut vs per-site models. Tranche 2 COMPLETE —
**interaction term +0.011 pooled FVU**, single-site cells vindicate
F7 as the fair-allocation control. Tranche 4 FVU endpoint COMPLETE
(≥3 seeds every headline cell, spreads 0.0002–0.0009). **E6 extension
harvest launched 07-19** (`run_phase099_ext.sh`) → tranche 6 unblocks
on completion. Open: single-site R-D plane placement, tranche 6
epochs-vs-fresh, tranche 7 derisk, tranche 5 (last).

Standing epistemic rule for the whole phase: the calendar/zoo/atlas
families are **burned** as selection criteria — three analysis passes
have inspected them extensively, and further tuning on them is probe
overfitting. They remain as exploratory descriptive probes only. All
confirmatory capture claims route through the sealed panel (tranche 0)
and the predeclared aggregate endpoints (tranche 4). The mega-block
rule (top-1 capture is never read without order + FVU) applies
everywhere.

## Resources (measured 2026-07-18)

- One 12M-optimizer-token pilot run ≈ 20 min train (10.5–12.7k tok/s,
  data-wait 31–42%) + θ-calibration + eval ≈ 30–40 min wall. The 4090
  is otherwise idle. The whole campaign is bounded by code, not GPU.
- Harvest ≈ 6k tok/s; train-split cost ≈ 38 GB per 1M tokens.
- Disk after the ratified purge (below): `/data` ≈ 507 GB free with
  the pilot store intact; root NVMe untouched (809 GB free, stays
  reserved for HF cache + OS headroom per the consult's warning).
- Host RAM 61 GB — the binding constraint that makes E1 mandatory.

**Purge record (a9 authorization 2026-07-18):**
`/data/stores/bcc-phase09` (143 GB, 1b 6-site store — 0.9/0.9.5/0.9.6
tier A complete, findings committed, regenerable via
`harvest_phase09_store.py`) and `/data/stores/bcc-phase0` (22 GB, SAE
harvest acts for Phases 0/0.5 — verdicts committed, pinned 65k artifact
lives in `data/phase0/` in-repo, regenerable via the phase-0 harvest
scripts) deleted. `/data/runs/` is **kept** — checkpoints there back
committed figures and the tranche-3 codec validation.

## Tranche 0 — the sealed probe panel (define before any new training)

A fresh concept-family panel, preregistered now and opened **only after
the Phase-1 config freeze**, so at least one capture readout exists
that no tuning decision ever saw. Families (chosen blind — none has
been probed in stream or dictionary):

| family | classes | expected topology | metric (existing zoo stats) |
|---|---|---|---|
| zodiac signs | 12 | ring | adjacency ring, 20k-perm |
| US states | ~40 single-token, cap-only | geographic map | geo LOO lat/lon decode |
| military ranks | ~10 | linear | Spearman rho along PC1 |
| SI prefixes | ~10 | log-line | rho + linear-vs-log spacing |
| size adjectives | ~7 (tiny…gigantic) | ordered scale | rho |
| alphabet letters | 26 | linear | rho |

Fixture details (label maps, capitalization rules, per-class caps) are
pinned at implementation time *without* running any scan. Sealed means
sealed: no stream-side availability checks either — availability is
part of what the unsealing measures. Unsealing condition: Phase-1
config freeze (or an explicit a9 decision to unseal early for the
tranche-5 re-ratification bar, which consumes one unsealing).

## Tranche 1 — correctness-critical engineering (Phase-1-blocking)

Ordered by trust impact, per the consult: θ-calibration is part of the
declared codec and has OOM'd twice; the guard is insurance.

- **E1 — streaming θ-quantile** for `fit_threshold_`. Replaces the
  host-side exact kthvalue (currently holds several score-matrix-sized
  copies; two 61 GB OOMs on record). Validation gate: matches the
  exact quantile on the pilot 2M calib split within a pinned tolerance
  (target: |Δθ| giving |Δavg-blocks| ≤ 0.1), then a G=8192-sized
  memory test. Blocking for any G ≥ 8192 or scalar production run.
- **E2 — loss-spike guard**, batch-skip form (the step-1600 evidence
  favors it). Discipline, per the consult: trigger requires
  corroboration (grad-norm vs trailing median AND loss/nonfinite/
  update-ratio signal), main and aux norms logged separately, the
  offending batch identity snapshotted, consecutive skips capped, and
  **skip-rate is itself a gate** — an operating point needing more
  than a pinned skip frequency (proposal: ≤0.1% of steps, none outside
  a declared warmup vicinity) is not stable, and the guard must never
  make an unstable lr look stable by censoring its evidence.
  Free regression suite: clean 3e-4 (zero false positives), the 6e-4
  warmup event, the shared step-1600 batch, the 1.2e-3 catastrophe.
- **E3 — AuxK cap mechanism** (ratified 2026-07-18 as a Phase-1
  requirement; mechanism pinned here). Candidates: s_aux scaled to the
  live dead-set size (preferred first candidate), α_aux < 1, λ_aux
  ramp. Two-axis validation: (i) cascade suppression at 4b (the 6e-4
  arms reproduce it on demand); (ii) **rare-feature revival retention
  on the Phase −1 synthetic battery** — the SASA C.1 comparison rig
  exists, and a cap that quietly destroys revival for genuinely rare
  features fails even if the spike disappears. Report aux/main update
  ratios directly.
- **E4 — store-reader site-subset view** (enables tranche 2's
  single-site cells; shards already carry all 8 sites per token).
- **E5 — prefetch** (31–42% data-wait → direct wall-clock win on
  every subsequent run).
- **E6 — harvest extension patch**: `--load-whitener` (reuse the
  pinned pilot whitener, skip stage 1) + `skip_docs` past all existing
  splits so extension shards are corpus-disjoint (tranche 6).

## Tranche 2 — the full 2×2 factorial (the paper's causal-attribution spine)

The project claims the empty {block} × {cross-site} cell of the
literature's 2×2. The reviewer's first question is whether the
observed manifold results come from *blocking* or from *cross-site
shared codes* — and a9's ratification extends the answer to the full
square: compare the BSC to each piece individually.

| cell | model | status |
|---|---|---|
| {block, cross-site} | **BSC** (G=4096 × b=4, shared code, 8 sites) | have (2 seeds + renorm) |
| {scalar, cross-site} | ordinary **crosscoder** — the existing b=1 Gram arm (b=1 Gram = unit concatenated-decoder norm; per-site norms = share): the standard BatchTopK crosscoder parametrization | have (per-lr arms) |
| {block, single-site} | **BSF cell** — S=1 per site × 8 (Gram at S=1 = per-block orthonormal decoder, Fel-style in our stack) | new, config-only |
| {scalar, single-site} | ordinary per-site **SAE** — b=1, S=1 × 8 | new, config-only |

**Matching is exact and natural**: the joint model already holds
per-site encoder/decoder tensors and sums per-site encoder
contributions into the shared code — S=1 deletes exactly one degree of
freedom (code tying) with identical per-site parameter tensors, G,
b, and per-site rate (k=32 blocks / ℓ=128 latents per site per token).
Fidelity note: Fel's attribution-guided BSF variant is *not*
reproduced — the controlled in-stack comparison is the point, the
faithful reproduction is a confound. Renorm is degenerate at S=1 (a
single site's weight is a global loss scale); single-site cells run at
unit scale, and cross-cell comparisons use per-site whitened FVU
(scale-invariant) and the R-D plane.

Runs: 8 single-site models per cell per seed, each ~1/8 the data
volume of a joint run — a cell-seed costs about one joint run. Both
new cells at the production operating point (λ=1e-3, lr 3e-4 cosine —
lr sanity-checked per cell since the stability edge need not
transfer), 3 seeds each (tranche 4).

**Endpoints**: pooled + per-site FVU at matched rates; positions on
the R-D plane (tranche 3) — where cross-site support-bit amortization
is priced, the honest axis for a9's "how does it compare to either
piece" question; dead rates; burned-family capture screens as
exploratory description only. The decomposition read: main effect of
blocking, main effect of sharing, and the interaction term — **the
paper's thesis is a positive interaction**.

## Tranche 3 — the R-D codec + pilot frontiers (H3 preview)

The preregistered codec (design §Rate–distortion protocol) is
**entirely unimplemented** — canonical block orientation, calib-
quantile clipped uniform quantizer (q ∈ {4,6,8}), frozen empirical
count model + enumerative support bits, realized per-token counts,
sequence-level bootstrap, support-entropy sensitivity. Implement and
validate all of it on the pilot store *before* the production run can
need it. Standing caution from the pilot: the current "block tax"
(0.430 vs 0.368 pooled FVU) is distortion at matched activation count,
**not** a bits–distortion verdict — BSC's structural bet is cheaper
support transmission, and the honest R-D curve could move the story in
either direction.

- Codec validation: exercised end-to-end on existing checkpoints
  (θ pipeline + 2M calib split); active-count floor applied and
  excluded blocks reported (pilot calib is 2M vs production 13M — the
  floor bites harder here; worn openly).
- Frontier trainings: **λ=0 both arms per protocol**, k ∈ {16, 32, 64}
  × {BSC, scalar}, 2 seeds at the headline points; single-site cells
  from tranche 2 placed on the same plane at matched points.
- Output: the pilot H3 figure — frontier dominance or its absence
  over the shared operating region, seeds displayed separately.

## Tranche 4 — balanced seed robustness

≥3 seeds for **every** headline cell (BSC-renorm, BSC-primary, scalar
crosscoder, BSF cell) — not six seeds of the favorite. Predeclared
endpoint battery, reported for all cells: pooled/per-site FVU, R-D
position, dead rate, skip rate, shared-code correspondence (cross-site
cells), sealed-panel capture score (at unsealing). Burned-family
capture is reported descriptively, never selected on.

## Tranche 5 — targeted lr recovery (guarded, demoted to after tranches 1–3)

Renorm-first ladder {3e-4 control, 4.5e-4, 6e-4} with guard + AuxK cap
active; **no 9e-4** (well inside the demonstrated unstable regime;
running it mostly tests whether the guard can censor a bad optimizer).
One optional flat-schedule arm at reduced peak (the 1b/4b evidence
confounds peak lr with time-near-edge under a short cosine); a
warmup-length arm only if 4.5e-4 shows real headroom. The primary arm
joins only after renorm identifies a plausible point.

**Re-ratification bar** (re-opening the ratified 3e-4 point is an a9
decision; this is the proposed evidence standard): the candidate
(i) replicates on 2–3 seeds; (ii) improves a predeclared aggregate
endpoint (pooled FVU and/or R-D at fixed q) — not ring order;
(iii) dead and skip rates within pinned tolerances; (iv) revival
retention at parity on the Phase −1 battery; (v) sealed-panel score ≥
the 3e-4 control at unsealing (consumes the panel's one pre-freeze
unsealing); (vi) the step-1600 batch passes without triggering the
guard in a healthy dictionary.

## Tranche 6 — token-budget factorial (+6M extension on /data)

Post-purge, harvest **+6M unique train tokens** (~230 GB) onto `/data`
with the pinned whitener (E6), keeping ≥200 GB free throughout —
root NVMe untouched, no split-mount store. Then the factorial that
separates optimization insufficiency from data diversity at matched
24M optimizer tokens:

- 6M unique × 4 epochs vs 12M unique × 2 epochs, × {renorm, primary}.

This addresses the pilot's open consolidation question (capture was
not universal at 12M optimizer tokens; 1b tier A said order tracks
total effective optimization — epochs-vs-fresh decides whether *data*
or *optimization* is the binding budget). The planet-line
capture-threshold observation (stream |rho| 0.88, uncaptured at 12M)
rides along as exploratory description on a burned family.

## Tranche 7 — production-harvest derisk (no store required)

- Whitener stability vs slice size up to the planned 5M production
  slice (streaming covariance accumulation, nothing stored) — the
  pilot's depth-graded drift (L27 0.026 / L30 0.031 at 2M) is the
  motivating observation.
- Late-layer tail/overflow statistics on fresh tokens (bf16 headroom).
- Renorm scalars estimated on independent slices (stability check on
  the designated F7 gauge).
- Bytes/token and sustained writer-throughput forecast for the 2.17 TB
  production harvest; checksum + resume drill on the tranche-6
  extension harvest (doubles as the drill).

## Deferred (explicit)

- **Causal intervention / steering demo** — a9 2026-07-18: not
  necessary yet. Noted as the consult's top qualitative-addition pick;
  revisit at Phase 2 (whose export remains "proves the pipe").
- Full saklas seam integration (Phase 2, lands in saklas).
- b ∈ {2, 8} (design: out of scope for v1).
- 9e-4 and above; broad color-axis / attention-regime storytelling
  before core controls.

## Suggested execution order

1. Tranche 0 (panel pinned, blind) + E1–E3 with regression suites.
2. E4–E6; tranche 2 cells + tranche 3 codec in parallel (code-bound).
3. Tranche 3 frontier trainings + tranche 4 seeds (GPU-bound, ~2 GPU-days).
4. Tranche 6 harvest + factorial; tranche 7 alongside (harvest is the drill).
5. Tranche 5 last, judged against the by-then-complete endpoint battery.

**a9 decision points en route**: single-site matching-protocol sign-off
at implementation review; lr re-ratification (tranche 5 bar); panel
unsealing; any further purge beyond the two recorded stores.
