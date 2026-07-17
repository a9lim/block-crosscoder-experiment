# Phase 0.9 findings — 1b dress rehearsal

**Verdict: PASSED (2026-07-17). All plumbing gates green.** The entire
ladder ran end-to-end on gemma-3-1b at the rehearsal config: harvest →
whiten → store → train (BSC λ ∈ {0, 3e-4, 1e-3} + matched scalar
baseline) → threshold calibration → double-eval determinism → toy
manifold export. Science verdicts wait for 4b (design §Phase 0.9); a
green rehearsal does **not** clear the 4b operational risks — the
extended ≥3M-token 4b pilot (D13) remains a separate mandatory gate.

Config: gemma-3-1b-pt, sites = layers {7, 10, 13, 17, 20, 22} of 26
(25–90% band, v2.3.1), FineWeb-Edu sample-10BT, ctx 1024 with BOS +
position 1 dropped, G=1024 × b=4, k=16, batch 4096, 2 epochs
(3,906 steps ≈ 16M train tokens), 8-bit Adam. Store:
`/data/stores/bcc-phase09/gemma3_1b_6site_fineweb`; runs:
`/data/runs/bcc-phase09/`. Scripts: `harvest_phase09_store.py`,
`verify_phase09_store.py`, `run_phase09_rehearsal.py`,
`run_phase09_matrix.sh`, `export_phase09_toy_manifold.py`.

## 1. Harvest + store integrity — green

13M tokens (2M whitener slice, accumulated never stored + 2M calib +
1M eval + 8M train + 100k retained raw), 143 GB at ~13.8 KB/token,
written calib → eval → train in stream order. Harvest throughput
27.5k tok/s (stage 1) / ~17k tok/s (stage 2, io-wait ~72%).

- **Whitener stability at 2M fit tokens** (rel ΔW per site, shallow →
  deep): halves 0.0026 / 0.0027 / 0.0049 / 0.0139 / 0.0221 / 0.0246;
  quarters ≤ 0.041. The ~√33 tightening predicted from the 60k smoke
  landed exactly; deeper sites drift most, consistently.
- **Held-out transformed-covariance validation (D9)**, measured against
  the *shrinkage prediction* σ/(σ+λ) — the pinned ridge (λ = mean
  eigenvalue) makes W a shrinkage whitener, so vs-identity is the wrong
  reference: mean |eig − predicted| per site 0.0009 / 0.0010 / 0.0032 /
  0.0174 / 0.0371 / 0.0467 on 212,576 held-out tokens. Ridge-softness
  context (mean |eig − 1|): 0.94 / 0.94 / 0.91 / 0.77 / 0.74 / 0.71.
- **Every shard checksum-verified** post-hoc (54 train shards in 202 s).
- **Whitening round trip is bit-exact**: re-whitening the retained raw
  shard reproduces the stored calibration prefix with rel err 0.0,
  bf16 exact-match fraction 1.0000 (100k tokens).

## 2. Training + resume — green

Four runs, 3,906 steps each, 3.5–3.9 min/run on the 4090 at 75–130k
tok/s. Loss curves smooth (λ=0 rec 0.16 @ step 200 → 0.08 @ 1200 →
converged); `dead_frac_final_window` = 0.0 in **all four arms** (AuxK
sasa default healthy at this scale). Data-wait 55–70% — the store
reader, not the GPU, is the bottleneck; the Phase-1 prefetch flag
(pinned, 2–4 batches) is now empirically justified.

**Resume gate at production scale**: the λ=0 arm ran `--max-steps 500`,
stopped, checkpointed, and `--resume` continued via the deterministic
stream fast-forward (islice by `step_idx`); training completed and the
final eval was bit-deterministic. Checkpoint/resume plumbing holds
beyond smoke scale.

## 3. Threshold calibration + eval determinism — green

θ fit on ≈524k calibration tokens (VRAM-capped at 128 batches — the
streaming-quantile item below). θ is stable across BSC arms (2.756 /
2.757 / 2.758) and transfers: threshold-mode avg active blocks land
within 0.13% of target k on the held-out eval split, in every arm.
**Every eval pass, run twice, agreed exactly** (8/8 deterministic).

| arm | θ | FVU topk | FVU thr | blocks topk | blocks thr |
|---|---|---|---|---|---|
| BSC λ=0 | 2.758 | 0.4217 | 0.4214 | 16.0 | 16.016 |
| BSC λ=3e-4 | 2.757 | 0.4219 | 0.4215 | 16.0 | 16.021 |
| BSC λ=1e-3 | 2.756 | 0.4220 | 0.4218 | 16.0 | 16.010 |
| scalar (G·b=4096, k=64) | 1.328 | 0.3535 | 0.3533 | 64.0 | 64.074 |

Rehearsal-scale observations (not verdicts):

- **The λ ladder is ~free**: pooled FVU spread λ=0 → 1e-3 is 0.07%
  relative — the admissible set is wide open at 16M tokens on 1b,
  consistent with Phase −1's 10M-token reversal. The Phase-1 primary
  λ=1e-3 stands; carrying this narrowed set to 4b remains the
  cross-model transfer assumption worn openly (design §Phase 0.9).
- **Scalar beats BSC on FVU at matched latent-L0** (0.353 vs 0.422
  pooled; per-site gap uniform). Expected direction — 4096 freely
  selectable scalar latents vs 1024 pre-grouped blocks is a strictly
  richer selection class at equal L0. This is precisely H3's
  rate–distortion question; the science measurement happens at 4b with
  frontiers, not one point.
- Per-site FVU has a consistent shape in every arm: layers 10/13
  hardest (0.46–0.52 BSC), 17/20 easiest (0.31–0.39) — mirroring the
  whitener's depth profile.

## 4. Toy manifold export — green

`export_phase09_toy_manifold.py` exports the top contribution-energy
block of the λ=1e-3 run as a saklas-shaped folder (`manifold.json` +
`gemma-3-1b-pt.safetensors`) at
`/data/runs/bcc-phase09/toy_manifold/`, exercising the Phase-2
producer contract shape (the `discovered` source schema itself is
Phase-2 saklas work):

- **Energy accounting is exact**: the Gram-constraint invariant
  (Σ_s site energies = Σ‖z_g‖²) closes to 7.7e-8 over 1M eval tokens.
- **Block 54** (firing freq 4.5%): full rank-4 at every site, evenly
  spread σ (no rank-1 collapse), per-site truncated SVD preserving the
  coordinate map (basis + σ + b×r right-factor; ortho and recon errors
  ~1e-7), reload round trip < 1e-5 at all 6 sites.
- **Shares are contribution-energy** (never Frobenius norms; Phase −1
  findings §2.5) and rise with depth: 0.09 / 0.09 / 0.14 / 0.26 /
  0.22 / 0.21 across layers 7→22.
- **The whitener seam is worn openly** in the manifest: shares are in
  the training-side harvest-fit whitener, not re-expressed in a
  consumer-side neutral-fit whitener — that re-expression is the
  Phase-2 bridge's job.

## 5. Items carried to Phase 1

1. **Streaming quantile for `fit_threshold_`** — the pooled score
   matrix caps calibration at ~524k tokens on 24 GB; the 4b design
   wants the 13M-token calibration split (mandatory).
2. **Store reader prefetch** (pinned, 2–4 batches) — 55–70% data-wait
   at 1b scale; 4b batches are 3.4× larger.
3. The λ-transfer assumption and all D13 4b-pilot risks (late-layer
   outliers, 160 MiB-batch I/O, 671M-param optimizer throughput, full
   checkpoint mechanics, calibration tail power) are explicitly *not*
   cleared by this rehearsal.

Ops notes: two harvest smoke fixes landed before the real run (stage
boundary row margin; bf16-exact token-id test encoding). The
`gemma-scope-2-1b-pt-res` validation dictionaries (65k at layers
{7, 13, 17, 22}) remain available for eval-side comparisons if Phase-1
wants a 1b sanity anchor.

## Errata (2026-07-17, fidelity audit + sol counter-review S1–S4)

- The saved `latest.pt` checkpoints of these runs hold **θ = NaN** — θ
  was fit after the final checkpoint save; the calibrated values live
  only in each `report.json`. Driver fixed (re-save after calibration);
  reloading a 0.9 checkpoint requires re-running `fit_threshold_` (S1).
- Calibration and eval ran the **fp32 master** (now the declared codec
  primary); the training/deployment bf16 precision was not evaluated. A
  bf16 shadow eval is added to the driver for 0.9.5 onward (S2).
- The harvest recorded dataset/config/split but **no HF revision** —
  short of the design's pinned-manifest requirement; revision pinning
  added to the harvest script for all future harvests (S3).
- §1's "held-out transformed-covariance validation" is an uncentered
  **second moment** about the fit mean (difference negligible on
  fit-mean-centered whitened data); script renamed accordingly (S4).
