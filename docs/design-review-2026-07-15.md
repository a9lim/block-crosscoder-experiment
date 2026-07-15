# Adversarial design review — 2026-07-15

Pre-implementation review of design v1 (commit 8cf4f02 state), requested by
a9. Two independent passes: **Codex** (sol-tier, gaslamp thread
`bsc-design-review`, job `cx-20260715-130118-ee7d` — 35 numbered findings,
6 blockers) and **fable** (this repo's session, run in parallel before
reading Codex's reply). Codex's verdict: *"Do not freeze this design yet"* —
correct. Design v2 ([`design.md`](design.md)) is the synthesis; this file is
the finding-by-finding disposition. Finding numbers below are Codex's;
⊕ marks findings independently found in the parallel pass.

## Blockers

| # | Finding | Disposition |
|---|---|---|
| 1 ⊕ | Nuclear-norm penalty defeated by exact scale symmetry z↦cz, D↦D/c (reconstruction and selection both invariant; penalty → 0). The v1 objective's infimum was degenerate; the rank headline was meaningless. | **Fixed structurally**: per-block Gram constraint Σ_s D_g^s D_g^sᵀ = I_b with post-step retraction. Spectrum budget per block is fixed, so the penalty can only reshape, not vanish. |
| 2 | Even scale-fixed, within-block GL(b) gauge unfixed; anisotropic reparameterization contaminates spectra. | **Fixed by the same constraint**: preserved only by O(b), which spectra are invariant to. |
| 11 | log₂(G choose k) is one term of a codec, not MDL; unquantized amplitudes are infinite bits; the scale defect also made amplitude bits arbitrary. | **Fixed**: full pre-registered codec (entropy-coded per-token count + support + q-bit whitened-unit quantization, q swept); gauge-fixed units make q-bit comparable across blocks. Metric renamed *held-out activation rate–distortion*. |
| 17 | Phase 0 null is multiply ambiguous (SAE tiling, wrong layer, power) and cannot support "gemma flattens." | **Fixed**: GPT-2 positive control mandatory before the gemma verdict; gate language rescoped to "no rings recoverable from this SAE at these depths at demonstrated power"; gates reframed design-wide as decision heuristics with reported power. |
| 18 | Pretrained-SAE availability for the exact model/hooks unestablished. | **Resolved by evidence**: Gemma Scope 2 (Dec 2025, saelens-native) covers gemma-3-4b pt/it and gemma-3-1b (`google/gemma-scope-2-4b-pt` etc.). Layer-coverage check added to Phase 0.5 preamble. |
| 26 ⊕ | Full G=8k untied config + streamed gemma does not fit 24 GB (arithmetic verified). | **Fixed**: disk-backed store decouples gemma from training (a9 confirmed ~1.9 TB NVMe); primary config reduced to G=4096; G=8192 stretch re-enabled by the store; 1M-token pilot mandated before the store is written. |
| 32 | Long list of unfrozen implementation parameters. | **Fixed**: v2 specifies objective/gauge/whitening/BatchTopK domain + inference thresholding/init/optimizer/aux loss/corpus/splits/codec/rank estimators/controls/seeds/gate criteria. Remaining free parameters (lr, λ grid) are explicitly rehearsal-calibrated. |

## Majors

| # | Finding | Disposition |
|---|---|---|
| 3 ⊕ | Selection score ‖z‖·Σ‖D‖_F ≠ contribution; nullspace-aligned codes over-score; biases rank histogram. | **Fixed**: under the Gram constraint, ‖z_g‖² *is* the exact whitened contribution energy. |
| 4 ⊕ | Per-site nuclear penalties favor site concentration (Σ|a| vs Σa² geometry) — dangerously close to manufacturing the decoupling pathology. | **Instrumented**: flat-profile shared blocks in the Phase −1 synthetic gate (regularizer must not specialize them); λ=0 control; contribution-covariance primary readout. Accepted as residual risk, worn openly in design. |
| 5 | Effective rank ≠ intrinsic dimension/curvature. | **Fixed**: renamed "effective linear span dimension"; curvature/ring claims require code-distribution diagnostics, never rank alone. |
| 6 | Regularizer produces the rank it reports (circularity). | **Fixed**: λ sweep incl. 0, full spectra, rank-truncation FVU ablations, 2 seeds. |
| 7 ⊕ | Whitening inconsistently defined across selection/penalty/rank/export. | **Fixed**: single consistency rule — all model math in whitened coordinates; export un-whitens; the D vs DW ambiguity is resolved (penalty and readout on whitened D). |
| 8 | Summed encoder proves multi-view correlation, not a shared code; late→early leakage. | **Fixed**: mandatory site-dropout / leave-one-site-out / cross-site code-agreement eval battery; H5 told only over passing blocks. |
| 9 | Blocks can bundle co-active unrelated features; support-bit advantage rewards it. | **Instrumented**: bundling nulls in Phase −1; coherence diagnostics; BH correction; "block = candidate, coherence battery = claim." |
| 10 | Phase-2 QR export discards the coordinate map R_s; rank-deficient sites emit junk directions. | **Fixed**: export per-site truncated SVD (U_r, σ, right-factor); node positions embedded through the full map. |
| 12 | Honest matching: Gb latents, kb coefficients, identical everything; a block transmits b coordinates. | **Adopted verbatim** in the protocol. |
| 13 | BatchTopK variable per-token k breaks fixed-k support formula. | **Fixed**: per-token count entropy-coded in the codec. |
| 14 | Single matched point insufficient; need frontier. | **Fixed**: (k, q) frontiers, dominance over shared region. |
| 15 | Parameter bits unaccounted → don't call it total MDL. | **Fixed**: renamed; scope stated. |
| 16 | Warm-start-only-BSC would invalidate the comparison. | **Fixed**: cold-start headline runs; warm starts exploratory only. |
| 19 | PCA variance alone doesn't demonstrate a ring. | **Fixed**: held-out circular decoding, Fourier, mixture/separability, permutation + random-cluster nulls. |
| 20 ⊕ | Phase 0.5 cosine-flow gate tests decoder-direction similarity, not span+position correspondence — fails exactly when the BSC premise holds. | **Fixed**: principal angles + paired-token CCA/Procrustes + out-of-sample coordinate prediction; Laptev flow demoted to bootstrap. |
| 21 | Rank histogram not independent evidence for H4; J-lens CoM not automatically independent. | **Fixed** via the finding-6 control battery; J-lens cross-check kept but demoted from "independent validation" language. |
| 22 | Phase 2 claims more geometry than a BSC produces (σ, origin, modes are extra estimators). | **Fixed**: reframed as additional estimators with own validation burden; naturalness eval scoped to the intervention path. |
| 23 | Phase-3 site arithmetic wrong (layers×models doubles params to ~2.7B; "doubles harvest only" false). | **Fixed**: constant site budget (4 layers × 2 models); digest amended. |
| 24 | Shared tokenizer ≠ aligned inputs (chat templates). | **Fixed**: paired forwards on identical template-free token streams; template gap documented. |
| 25 ⊕ | Provenance overreach: SASA transferred from abstract; "MDL carries over unchanged"; "exists nowhere" too broad. | **Fixed**: SASA read at full text same day (see below); digest amended at four points; novelty scoped to LLM interp with the multi-view/group-sparse sweep flagged as owed. |
| 28 ⊕ | "Streaming" lacked a viable hardware topology. | **Fixed**: disk store primary (hardware confirmed), interleaved escalation path specified. |
| 29 | Dead-block death spiral: TopK gives no gradient, nuclear penalty shrinks decoder, score falls further. | **Fixed structurally** (constraint forbids decoder shrinkage) + block-level AuxK for encoder-side starvation. |
| 30 | Nuclear norm at 65k matrices/step needs a real implementation plan; mixed-precision SVD unstable. | **Fixed**: batched b×b Gram eigh in fp32, ε-smoothed; no d-dim SVD in the loop; retraction likewise b×b. |
| 31 | Token budget and wall-clock unsupported. | **Fixed**: mandatory 1M-token measured pilot before the store commit; budgets re-confirmed from measurement. |

## Minors

27 (activation-volume arithmetic correct — confirmed), 33 (b=4 is a
hypothesis: b-sweep scoped out for v1, noted in design), 34 (acausal
encoding acceptable with limited claims — the site-dropout battery is the
quantifier), 35 (nulls need demonstrated power — adopted as a design-wide
gate principle).

## Findings from the parallel pass not in Codex's list

- **Gemma Scope 2 existence/coverage** (resolves finding 18 positively) —
  verified via HF hub search.
- **fp16 overflow on gemma-3 late layers** (fact preserved in saklas
  `mahalanobis.py`) → bf16-everywhere dtype rule.
- **saklas `LayerWhitener` is a CPU one-shot Woodbury operator** — the
  convention transfers (ridge scaling), the object doesn't; training needs a
  materialized dense W_s. Also surfaced the **two-whitener seam** at Phase 2
  (training-side harvest-fit vs consumer-side neutral-fit; `share` must be
  re-expressed in the consumer's).
- **1b dress rehearsal** (Phase 0.9) — full-ladder plumbing validation at
  toy scale; a9 signed off.

## The SASA correction

Full-text read (arXiv:2606.06333, 2026-07-15): SASA's regularizer is
λ Σ_k ‖D_k E_k‖_* — the **end-to-end product**, gauge-invariant by
construction, with the variational form ‖W‖_* = min ½(‖D‖²_F + ‖E‖²_F)
(implementable as balanced weight decay). Their gating score is the
pre-activation norm ‖E_k h‖. v1's decoder-only penalty was therefore a
mis-transfer that neither SASA nor the review could save. v2's Gram
constraint reaches gauge-invariance by a different route and buys selection
exactness and a clean per-site readout on top; the SASA product form is
retained as the ablation variant. (Convergent detail: SASA's ‖E_k h‖ score
and our ‖z_g‖ score coincide in spirit — under our constraint the norm *is*
the contribution.)

## Round 2 (same day, job `cx-20260715-132049-f176`)

The v2 synthesis went back to Codex for adversarial verification.
**Verdict: "V2 repairs the fatal mathematics from v1"** — all four claimed
Gram-constraint properties VERIFIED-PASS by independent algebra (scale-gauge
death; O(b) residual gauge; ‖z_g‖² = exact individual contribution energy,
with the wording caveat that this is isolated output energy, not marginal
loss reduction; retraction correctness) — but *do not freeze unchanged*:
26 further findings (R1–R26), three of them blockers. All were accepted and
folded into design v2.1 the same day:

- **R12 (blocker)** — loss reductions unspecified, λ meaningless →
  reductions pinned; R_rank normalized to [0, √S−1] with its constant floor
  b subtracted (the floor arithmetic also sharpened *our* understanding:
  the penalty's minimum is site-exclusive direction assignment, so it is a
  site-concentration penalty first and a rank penalty by side effect).
- **R13 (blocker)** — quantizer incomplete and not O(b)-invariant →
  canonical block orientation (calibration-set second-moment
  diagonalization, frozen), quantile clipping, saturation policy, all codec
  fitting on a dedicated calibration split.
- **R16 (blocker)** — baseline must be explicitly b=1 Gram-constrained with
  the same retraction; primary H3 at λ=0 both models (at b=1 the nuclear
  term is not a rank penalty, so nonzero λ is architecture-unfair).
- **R6** — constraint forces aggregate rank b; decoder spectra renamed
  *frame capacity*; rank claims only from code-anchored readouts.
- **R7** — the nuclear penalty's unconstrained preference is *strongly*
  site-exclusive (min b vs flat b√S) → quantitative Phase −1 λ-veto with
  λ=0-primary fallback.
- **R8–R11** — Adam/retraction geometry: fp32-master retraction ordering,
  rotation-equivariance test in Phase −1, eigenvalue floor + logged Gram
  residual, decoder weight decay 0.
- **R14, R15, R17, R18** — realized per-model counts (matched only in
  training average), count model fit off-eval, declared-codec scoping +
  entropy sensitivity, sequence-level bootstrap with seeds shown separately.
- **R19–R22** — calibrated single-site evals (raw + calibrated; a summed
  encoder legitimately distributes z/S across sites), rotation-invariant
  code-agreement stats, uncentered contribution second moment added
  (centering deletes a real rank-1 mean component), minimum activation
  counts for H4 inclusion.
- **R23–R26** — store arithmetic in exact bytes (whitener slice
  accumulate-and-discard, ~260 GB true headroom), learning curves at store
  fractions + recovery-vs-frequency calibration ("conditional at
  demonstrated sample power" language), escalation retrains the baseline on
  the identical extended manifest, staged run matrix.

Codex's freeze condition was a v2.1 amendment covering exactly these; that
amendment is applied. **Design frozen for implementation at v2.1.**

## Standing debts

- Sweep older multi-view / coupled / joint group-sparse dictionary-learning
  literature before any external novelty claim.
- SASA appendices (dead-group aux loss C.1, optimizer B.3) worth a read
  before Phase-1 hyperparameter freeze.
- Phase −1 must set the numeric λ-veto tolerance and the rotation-
  equivariance divergence threshold in its config (predeclared before the
  4b runs, per R7/R8).
- Verify the fp32-master/8-bit-Adam retraction ordering against the actual
  optimizer library before Phase 0.9 (R9).
