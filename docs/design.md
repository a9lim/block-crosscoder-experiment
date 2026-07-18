# block-crosscoder-experiment — design

**A block-sparse crosscoder (BSC) — dictionary learning whose atomic unit is a
subspace, with one shared code across layers — is the unsupervised generator
of saklas's manifold artifact, and within LLM interpretability nobody has
built one.** The literature is a 2×2, {scalar, block} × {single-site,
cross-site}, with three cells occupied (SAEs; BSF/SASA; crosscoders and kin)
and the fourth empty as of 2026-07-15. This experiment fills it on gemma,
measures whether the combination earns its parameters, and lands discovered
manifolds in a real steering/probe runtime.

Full literature provenance, gap sweep, and the synergy argument live in
[`docs/research/block-sparse-crosscoders-2026-07.md`](research/block-sparse-crosscoders-2026-07.md).
One paragraph of it matters here: each parent fixes the other's failure.
Blocks absorb cross-layer within-subspace rotation that fragments scalar
crosscoder latents (SASA's splitting theorem, aggravated by depth);
crosscoders solve cross-layer block identification by construction and emit
per-layer decoder norms — the depth profile saklas bakes as `share`.

**Revision provenance.** v2, 2026-07-15: rewritten after an adversarial
design review (Codex sol-tier, thread `bsc-design-review`, 35 findings /
6 blockers) plus an independent parallel pass; the finding-by-finding
disposition is in
[`docs/design-review-2026-07-15.md`](design-review-2026-07-15.md). The v1
objective was mathematically ill-posed (gauge-defeated rank penalty); v2
fixes it structurally. SASA has now been read at full text — its regularizer
is on the product `D_k E_k`, not the decoder alone, which is how it dodges
the same gauge hole; our Gram-constraint route is different but solves
selection exactness and gauge in one move (rationale in the review doc).
**v2.1, same day**: round-2 verification PASSed the Gram-constraint algebra
(all four claimed properties) and conditioned freeze on the amendments now
folded in — objective normalization conventions, the complete quantizer
spec, the explicit b=1 Gram-constrained baseline, fp32-master retraction
ordering, calibrated shared-code evals, and exact store arithmetic.
Round-2 findings are numbered R1–R26 in the review doc.
**v2.2, 2026-07-16**: deployment re-plan + round-3 amendments
after two fresh-context adversarial passes (deployment/design D1–D14,
paper-fidelity P1–P25 over all 13 reference full texts; disposition in
[`docs/design-review-2026-07-16.md`](design-review-2026-07-16.md)). The
v2.1 store assumed a ~1.9 TB volume that does not exist (jobe measured:
2×1 TB); resolved by a dedicated 4 TB NVMe (a9, 2026-07-16), preserving
8 sites / 38M tokens / G=8192. Structural changes: whitened-bf16 store
with immutable hashed whitener; stored 13M-token calibration split;
calibration-fit inference threshold (EMA demoted to diagnostic); AuxK
re-specified from SASA C.1 with a variant comparison; Phase −1 generator
gauge-corrected (bundle null weakened — perfect co-activation is
observationally equivalent to a block under linear reconstruction); pilot
extended to actually exercise AuxK; positive control pinned to Engels'
actual artifact.
**v2.3, 2026-07-16**: post-Phase−1 consolidation. Phase −1
executed and **passed** (battery runs 1–6 + capture-campaign sweep
rounds 1–8 on jobe; gate semantics ruled by a9: strict
capture-as-written; full record in
[`docs/findings-phase-minus1-battery.md`](findings-phase-minus1-battery.md)).
Everything the harness taught is folded into the body: the λ-veto
outcome (admissible {3e-4, 1e-3}; **Phase-1 primary λ=1e-3**), the
selection-budget regime map and battery operating point, the packing
economics (block width is a packing budget; merging is a converged
optimum), the zero-slack ring-detection lesson (norm-concentration
alone cannot detect rings under budget slack), the AuxK verdict (SASA
C.1 positively separated), and the recovery-vs-frequency calibration
(clean to f=0.01 at 10M tokens). Design re-frozen from this state; the
decision log remains as provenance, and the body supersedes older log
entries wherever both speak.
**v2.3.2, 2026-07-17 (current)**: fidelity amendments after the
paper-fidelity audit + sol counter-review (F1–F11, S1–S7; disposition
in [`docs/design-review-2026-07-17-fidelity.md`](design-review-2026-07-17-fidelity.md)).
Adds the **0.9.5 calibration addendum** (lr × schedule ladder on both
arms, dead-dynamics arm at the Phase-1 k/G ratio, site-renorm arm —
greenlit, `scripts/run_phase095_matrix.sh`), the **pre-4b-store
decision item** on per-site RMS renormalization after the shrinkage
whitener, the Phase-3 freeze items (web+chat corpus mix, per-model
init pairing), and the S-series code fixes (θ serialization, bf16
shadow eval, corpus-revision pinning, checkpoint free-space floor).
Ratified by a9 2026-07-17; design frozen as v2.3.2.

## Hypotheses

- **H1 (rings exist).** Gemma carries irreducible multi-dimensional
  token-level features (Engels-style weekday/month rings) findable by post-hoc
  blockification of an existing SAE. The saklas-side centroid-level flattening
  results don't rule this out — pooling can average a ring into a blob; they
  measure different objects.
- **H2 (cross-layer coherence).** Ring/block subspaces at different depths are
  matchable — the subspace persists while its frame may rotate — **and paired
  tokens occupy corresponding positions** in the matched subspaces. Both
  halves are load-bearing (a shared code needs position correspondence, not
  just span overlap); both are testable before any training.
- **H3 (blocks earn their parameters at language).** On a held-out
  activation rate–distortion comparison (the Fel MDL argument, made honest —
  see *Rate–distortion protocol*), a BSC's frontier dominates a matched
  scalar crosscoder's. A mostly-rank-1 dictionary is a publishable
  *corroboration* of the flattening line, not a wasted run.
- **H4 (depth-resolved geometry).** Per-block per-site **effective linear
  span dimension** (see *Rank measurement* — deliberately not called
  "intrinsic dimension": a circle is intrinsically 1-D in a 2-D linear span,
  and decoder rank measures the span) localizes *where in depth* structured
  geometry lives, cross-checkable against J-lens depth center-of-mass.
- **H5 (manifold-level diffing).** With sites = layers × models, per-model
  block presence and rank answer provenance questions: is the persona fan
  present in base gemma or installed by chat-tuning?

Gates below are **decision heuristics, not falsifications**: each null is
reported with the demonstrated power of the pipeline that produced it
(positive controls, synthetic recovery), and what it does is downgrade the
prior and reroute effort, not refute the hypothesis.

## Architecture spec

### Sites, coordinates, whitening

Sites s = 1..S are residual-post taps at selected layers (Phase 3: layers ×
models). **All model mathematics — encoding, selection, loss, rank penalty,
rank readout — lives in whitened per-site coordinates**; raw space appears
only at harvest ingress and export egress. This is the single consistency
rule that keeps selection, regularization, and the H4 readout commensurable
(review finding 7), and it designs out massive-activation/rogue-dim bait.

Whitening protocol: per site, fit mean μ_s and covariance Σ_s on a dedicated
5M-token slice of the harvest corpus (disjoint from train and eval),
harvested *first*; ridge per saklas's `LayerWhitener` convention (λ_s =
mean-diagonal of Σ̂_s × `DEFAULT_RIDGE_SCALE`); eigendecompose once; freeze
W_s = (Σ_s + λ_s I)^{-1/2} and export it with the run config. Numerics
(D9): TF32 disabled for covariance GEMMs (the implementation accumulates
in fp64, which sidesteps TF32 entirely); batch-granular sufficient
statistics accumulated linearly in fp64 within quarter-accumulators and
tree-merged (sol S5 — this, not literal Welford/pairwise, is what the
code does; fp64 linear accumulation over ≤5M tokens carries relative
error ~5e-10 and is accepted as equivalent); the eight
d×d eigendecompositions run in fp64 offline; whitener stability is checked
across corpus halves/quarters and the transformed covariance spectrum
validated on held-out sequences — 5M tokens is a *candidate* size that must
pass these criteria (sequence correlation shrinks effective N; parameter
counting alone proves nothing). The store then holds **whitened bf16**
(D8): the whitener is immutable once fit — the exact μ, W, ridge, layer
set, and source manifest are hashed into every shard header and mismatches
rejected at load; a small raw shard is retained to measure whitening
round-trip error; any refit forces reharvest or an explicit store
migration. Whitened activations: x̃^s = W_s (x^s − μ_s). Honest label
(F7, fidelity audit 2026-07-17; upgraded by sol counter-review): at
`DEFAULT_RIDGE_SCALE` = 1.0 the ridge equals the mean eigenvalue, so W
is a **shrinkage** whitener — deep-site whitened spectra sit well below
identity (0.9 harvest: mean |eig−1| 0.71–0.94) and held-out validation
correctly targets the prediction σ/(σ+λ), not identity. External prose
should say "shrinkage whitening"; massive-activation suppression is
proportional, not total. The sharper consequence (sol): mean retained
variance per dimension runs ≈ 0.06 (shallow) to 0.29 (deep) across the
0.9 sites, so the equal-per-dimension L_rec weights deep sites several
times more heavily than shallow ones — the store does **not** fully
deliver the cross-site commensurability this section claims, nor
Anthropic's "each layer contributes comparably" intent. **Pre-4b-store
decision item (a9)**: whether to apply a per-site scalar RMS
renormalization *after* shrinkage whitening — preserving directional
rogue-dim suppression while restoring equal total site power — folded
into W_s and the whitener hash. Default proposal: adopt; the
read-time renorm arm in the 0.9.5 addendum (greenlit, a9 2026-07-17;
`--site-renorm`, scalars via `Whitener.site_rms_scalars` — exact on
the existing 1b store) supplies the decision data before any 4b bytes
are written.

**dtype discipline: fp16 is forbidden everywhere in the harvest and store
path** — gemma-3's late-layer channels exceed fp16 max (fact preserved in
saklas `mahalanobis.py`); harvest and store in bf16, whiten and accumulate
stats in fp32.

### Model

G blocks of width b. Per site: encoder E_g^s ∈ ℝ^{b×d}, decoder D_g^s ∈
ℝ^{b×d}, decoder bias c^s ∈ ℝ^d (init 0). Per token:

- **encode** — z_g = Σ_s E_g^s x̃^s (summed per-site maps; crosscoder
  convention; untied by default).
- **gauge constraint (load-bearing)** — per block, the concatenated decoder
  Gram is the identity:
  **Σ_s D_g^s D_g^sᵀ = I_b.**
  Enforced by retraction after every optimizer step: M_g = Σ_s D_g^s D_g^sᵀ,
  D_g^s ← M_g^{-1/2} D_g^s (batched b×b eigh, fp32). This one constraint
  simultaneously (i) kills the z↦cz, D↦D/c scale symmetry that made the v1
  rank penalty vacuous (finding 1), (ii) cuts the within-block GL(b) gauge
  down to orthogonal rotation, which spectra are invariant to (finding 2),
  (iii) makes ‖z_g‖² *exactly* the block's **individual contribution
  energy** Σ_s ‖D_g^sᵀ z_g‖² — the selection score is no longer a proxy
  (finding 3; note this is the block's isolated output energy, not its
  marginal loss reduction — blocks can overlap or cancel, same caveat as
  every SAE-family norm score), and (iv) structurally blocks the dead-block
  death spiral, since no block's decoder can shrink toward zero (finding
  29). Per-site Frobenius shares tr(D_g^s D_g^sᵀ)/b remain free — the depth
  profile survives.

  Retraction mechanics (round-2 findings R8–R11): the retraction operates on
  the **fp32 master weights** in this order — optimizer step on master →
  retract master decoders → regenerate bf16 forward copy → log post-cast
  Gram residual ‖M_g − I‖ (a training-health metric; large residual after
  init indicates trouble). λ_min(M_g) is floored (default 1e-6) before
  inversion, with floor hits logged; verify the ordering against the chosen
  8-bit-Adam implementation before Phase 0.9. **Decoder weight decay is 0**
  (uniform shrinkage is undone by retraction and only injects noise); decay
  applies to encoders only. Coordinatewise Adam is not equivariant to the
  residual O(b) rotation, so Phase −1 includes a rotation-equivariance test
  (train paired seeds from randomly rotated inits; compare recovered
  spectra/subspaces); if divergence is material, decoders move to
  projected-momentum SGD while encoders keep Adam.

  One structural consequence, worn openly (R6): the constraint forces every
  block's concatenated frame to rank exactly b — a genuinely lower-rank
  feature parks its surplus directions somewhere in site space with the
  encoder suppressing their code variance. Decoder spectra are therefore
  **frame capacity**, not used dimension; rank claims come only from the
  code-anchored readouts below.
- **select** — BatchTopK over blocks by p_g = ‖z_g‖₂ (exact contribution
  energy, comparable across blocks because of the constraint). Training:
  keep the top k·B block-activations across the batch of B tokens (per-token
  counts vary). Inference: fixed global threshold θ on p_g. The
  training-time EMA of the batch-minimum selected score is a **diagnostic
  only** — it inherits optimizer history (drift, decay constant, checkpoint
  timing), so two equivalent checkpoints could otherwise carry different
  realized count distributions straight into the codec frontier (D10). The
  final θ is fit on the calibration split to hit the preregistered average
  block count, then frozen and serialized with the codec; threshold
  sensitivity and the resulting count distribution are reported. BatchTopK,
  not L1: Minder et al. (2504.02922) show L1
  manufactures Complete Shrinkage / Latent Decoupling; the block-level
  analogues would poison exactly the H5 diffing questions.
  Statistic provenance (F6, fidelity audit 2026-07-17): the score itself
  departs from Minder's v = f_j · (Σ_s ‖d_j^s‖) — an L1-of-norms
  geometry — because under the Gram constraint ‖z_g‖² *is* the exact
  total contribution energy and the score ‖z_g‖ is its square root
  (identical ranking; the L2-of-norms member of the pair), and it is
  site-profile-neutral. Anthropic's L1-vs-L2-of-norms discussion is
  about the *penalty*, so it transfers to selection as analogy, not
  procedural equivalence (their L1-of-norms choice "surfaces
  layer-specific features"); on the penalty side our R_rank, a per-site
  sum, is the L1-of-norms member and wears its site-concentration bias
  through the R7 veto.
- **decode** — x̂̃^s = c^s + Σ_{g active} D_g^sᵀ z_g; raw-space export
  un-whitens (x̂^s = W_s^{-1} x̂̃^s + μ_s).
- **loss** — L = L_rec + λ_* R_rank + α L_aux, with reductions pinned so λ
  and α are meaningful across configs (round-2 blocker R12):
  L_rec = mean over tokens, sites, and dimensions of the squared whitened
  residual; R_rank = mean over blocks of (Σ_s ‖D_g^s‖_* − b)/b — under the
  constraint the per-block sum ranges over [b, b√S] (b = every code
  direction site-exclusive, b√S = flat across sites), so the normalized
  penalty lives in [0, √S−1] with 0 = fully site-concentrated; L_aux = the
  AuxK residual reconstruction under the same reduction as L_rec.
  Squared error is a deliberate pick between conflicting sources (F1,
  fidelity audit 2026-07-17): Anthropic, SASA, and Fel all train on
  squared L2, while Minder's written objective (their Eqs. 4/8) uses
  **unsquared** per-site L2 norms — a different gradient geometry that
  downweights high-error tokens. Sol caveat: that is Minder
  *as written* — the same paper derives Latent Scaling from a squared
  objective and later speaks in MSE terms, so the source is internally
  inconsistent and their released trainer is unchecked. We inherit
  Minder's BatchTopK/AuxK/θ machinery but follow the squared-majority
  reconstruction convention.

### Rank regularizer

R_rank = Σ_g Σ_s ‖D_g^s‖_* — per-site nuclear norm of the **whitened**
decoder, computed from batched b×b Gram eigenvalues (Σ√(eig+ε), fp32; no
d-dimensional SVD anywhere in the loop). Under the Gram constraint the total
spectrum budget per block is fixed (Σ_s Σ_i σ_i² = b), so the penalty can
only *reshape* spectra toward per-site low rank — it can no longer be gauged
away.

Known residual bias, accepted and instrumented (finding 4, sharpened by
round-2 R7): the penalty's unconstrained preference is **strongly
site-exclusive** — its minimum (normalized 0) is attained by assigning each
code direction to a single site, and the flat shared configuration sits at
the maximum (√S−1). Reconstruction must overpower that preference for
genuinely shared structure to survive nonzero λ. Mitigations are therefore
gates, not notes: (i) the Phase −1 harness plants *shared, flat-profile*
blocks and imposes a **quantitative veto** — λ is admissible only where the
planted flat blocks' depth-profile share error (contribution-energy
shares, never Frobenius — parked frame capacity poisons the latter)
stays under max(0.02, 2× the λ=0 baseline) and subspace recovery
retains ≥0.85× the λ=0 overlap; if every nonzero λ fails, λ = 0
becomes primary and the penalty is demoted to a failed ablation — the
veto cannot be passed trivially by λ=0 alone. **Veto outcome (battery
run 6, 10M-token operating point, 4 seeds): admissible = {3e-4, 1e-3},
so Phase-1 primary is λ=1e-3**; at 3e-3 the binding violation is
genuine share concentration (0.046 > 0.02). An earlier empty-set
verdict was a 3k-step measurement artifact (decision log). (ii)
headline H4 numbers are reported across the λ sweep including λ = 0;
(iii) H4's primary readouts are code-anchored (below), not the raw
decoder spectrum.

SASA's actual formulation (full-text read 2026-07-15) — penalty on the
end-to-end product ‖D_k E_k‖_*, gauge-invariant by construction, weight-decay
variational form — is the documented **ablation variant**: it needs no
retraction but leaves selection exactness unfixed and mixes encoder
conditioning into the per-site rank readout, which is why it is not primary.

### Rank measurement (H4 readout)

"Effective linear span dimension," three estimators, all reported with full
spectra (findings 5, 6; round-2 R6, R21):

1. **Decoder spectrum** = **frame capacity**, not used dimension — the
   constraint mandates aggregate rank b, so surplus directions are parked
   capacity. Reported (participation ratio (Σσ²)²/Σσ⁴, 95%-energy rank) but
   never the basis of a rank claim.
2. **Contribution second moment** (primary) — spectrum of
   D_g^sᵀ E[z_g z_gᵀ | g active] D_g^s: the span the block *actually uses*
   at that site, mean included (a large conditional mean is a real rank-1
   component that centering would delete).
3. **Contribution covariance** (centered) — spectrum of
   D_g^sᵀ Cov(z_g | g active) D_g^s: within-feature *positional* variation,
   the manifold-structure signal.

Inclusion threshold (R22): a block enters H4 histograms only above a
predeclared minimum active-token count (default 10k on the eval set);
below-threshold blocks are counted and reported separately.

Required controls: λ = 0 run, λ-sensitivity curve, seed stability (2 seeds
at primary config, judged at block level **and** at global
recovered-subspace level via principal angles — Dooms §4.3: individual
latents agree 15–50% across seeds while the global subspace agrees >90%,
so block-identity matching alone understates stability, P20), and
**rank-r truncation ablations** — held-out FVU as a
block is truncated to its top-r directions, which is the reconstruction-
anchored ground truth the spectra summarize and the final arbiter of any
rank claim. Curvature/ring claims are never made from rank alone; they
require the code-distribution diagnostics from Phase 0 (circular decoding,
Fourier structure).

### Shared-code validity (anti-leakage evals)

A summed multi-site encoder can achieve low loss by multi-view correlation
without a genuinely shared coordinate system, including late→early leakage
(finding 8). Two round-2 corrections to the eval battery (R19, R20): the
summed-encoder optimum may legitimately distribute a shared estimate across
sites (E^s x^s ≈ z/S), so *raw* single-site encoding under-reconstructs even
for a perfectly shared code — single-site evals are always reported **raw
and calibrated**, where the calibration is a per-(block, site) linear map
from site-only code to full code, fit on a calibration split, never on
eval. And per-coordinate correlation is not O(b)-invariant — cross-site
agreement uses rotation-invariant statistics (canonical correlations /
Procrustes-aligned R²), the same family Phase 0.5 uses.

Mandatory evals, run on every trained BSC:

- **Site-dropout encoding** — encode from single site s′ only, decode all
  sites; per-(s′, s) FVU matrix, raw + calibrated.
- **Leave-one-site-out** — encode from all sites but s, reconstruct s.
- **Cross-site code agreement** — CCA / Procrustes-R² between site-only
  codes, per block.

Blocks that fail these are reported as "correlated bundles," not shared
manifolds — the H5 diffing story is only told over blocks that pass.

### Sparsity hygiene

- **Aux loss (dead blocks)** — starting spec follows SASA App. C.1 (P8): a
  block is dead when its running activation frequency falls to ≤ 10⁻⁴
  (SASA's window is 1k tokens; ours is re-expressed at batch 4096 and
  calibrated); the reconstruction residual is **frozen** (no gradient
  through it) and re-encoded through dead blocks only; the s_aux dead
  blocks with largest residual energy reconstruct it. s_aux, λ_aux (SASA:
  256–512 and 1.0), window, and threshold are calibrated in Phase −1/0.9
  via a three-way comparison — SASA-style frequency-dead vs long-horizon
  dead (the former v2.1 rule) vs Fel-style runner-up AuxK — v2.1's
  top-32 / 500-batch / α=1/32 was an unsupported hybrid of three papers'
  conventions and is demoted to one arm of that comparison. **Phase −1
  outcome: the comparison separates decisively at the 10M-token
  operating point — SASA C.1 recovers 12/12 planted rare features with
  1–4 dead blocks of 16, vs 9–12 dead and lost rare features for the
  Fel and long-horizon arms — so SASA C.1 is the confirmed default**.
  Window fidelity, worn openly (F3, fidelity audit 2026-07-17): SASA's
  literal criterion (ν = 10⁻⁴ over a 1000-token window) is arithmetically
  "zero firings in the last 1k tokens" — 0.1 expected events — while our
  re-expression (≤ 10⁻⁴ over 100 batches × 4096 = 409,600 tokens = "≤ 40
  firings") is a materially stickier criterion, not a unit conversion —
  a recent-zero detector became a rare-frequency classifier. The
  batch-granular ring buffer also cannot reproduce a literal
  token-granular 1k window, so the "SASA arm" is an approximation by
  construction and is labeled as such.
  The production-batch recalibration this bullet assigned to 0.9 **did
  not happen** — the rehearsal had zero dead blocks, so no calibration
  data exists; the item moves to the ratified 0.9.5 dead-dynamics arm
  (oversized G at production batch, to observe real dead dynamics) with
  the 4b pilot's AuxK exercise + synthetic revival test (D12) as
  backstop, plus a pre-registered in-run escalation: dead-fraction
  trajectory monitored during Phase 1 against the Phase −1
  recovery-vs-frequency calibration. The Gram
  constraint removes the decoder-shrinkage spiral; the aux loss handles
  encoder-side starvation. Block resampling kept as a documented option,
  off by default.
- **Latent Scaling, blockwise** (Minder) — per-site
  reconstruction-contribution regression as the shrinkage/decoupling
  diagnostic for Phase 3 diffing.
- **Init** — D_g^s Gaussian then one retraction (≈ equal site shares at
  init); E_g^s = D_g^s (transpose-tied at init only), with encoder scale
  norm-calibrated at init (Fel App. D convention) so initial selection
  scores are comparable across blocks (P16).
- **Optimizer** — AdamW, lr 3e-4 (1k-step linear warmup, cosine decay),
  β=(0.9, 0.999), fp32 master weights, 8-bit moments, bf16 params; batch
  4096 tokens. β and batch are SASA B.3 verbatim; lr and schedule are a
  synthesis matching no parent exactly (SASA: 2e-4, linear decay "over
  one-fifth of the training" — final-fifth is our reading; Fel: cosine
  1e-4→1e-5, 2k warmup; Minder: 1e-4), and the
  fixed 1k-step warmup is a varying *fraction* of training across scales
  (25% of the 0.9 run, ~5% at 4b — F10). **The recalibration this bullet
  once promised at 0.9 was not performed there** (the rehearsal ran these
  defaults green: smooth loss, zero dead blocks — sanity, not
  calibration); it is an open item (F2, fidelity audit 2026-07-17)
  assigned to the ratified 0.9.5 calibration addendum (lr × schedule
  ladder on the existing 1b store), with the mandatory 4b pilot
  (D12/D13) as backstop. The encoder weight-decay *value* is likewise
  open (spec says decay applies to encoders only; the code default is
  0.0 — currently no decay anywhere).
  Retraction is per-step initially; lower-frequency retraction (Fel used
  QR every 20 steps) is a documented throughput ablation (P16). The SASA
  product-penalty ablation uses **symmetric** encoder/decoder weight decay
  (its variational form requires it, P9); the primary keeps decoder
  decay 0. The ablation is spec-only as of 2026-07-17 (F9): its in/out
  status in the Phase-1 run matrix is decided before the 4b freeze —
  default: **out** of both the confirmatory matrix and the 0.9.5
  addendum. Sol's sharpening stands: a faithful product-penalty run
  changes the constraint, gauge handling, decay symmetry, and the
  validity of the selection statistic at once — it is a separately
  designed method ablation, not a cheap arm, and a hasty version would
  not isolate "the SASA penalty".

### Identifiability caveat (worn openly)

Hard group sparsity rewards packing frequently co-active scalar features
into one block whether or not they form a manifold (finding 9) — and round
3 sharpened this (D11): perfectly co-active scalars with full-dimensional
joint support are *observationally equivalent* to a block under a linear
reconstruction objective, so bundling cannot be penalized away or ruled
out; what the coherence battery certifies is that *curved / manifold*
structure is never claimed where none exists.

Phase −1 sharpened it once more, empirically (findings §2.2): **block
width is a packing budget**. Independent sub-width features whose
ranks sum to ≤ b pack losslessly into one block — the union code
carries both features' coordinates and the per-site frames route
them — freeing a block plus selection budget, so merging is loss- *and*
budget-optimal and *better convergence merges more* (30k-step runs
merged strictly more than 10k). Production dictionaries will contain
packed blocks as converged optima, not as noise. The signature — full
span overlap, near-50/50 contribution-share split, degraded code-R² —
is the diagnostic; the Phase-2 `share` export treats it as a packing
flag.

Block discovery is *candidate generation*; the manifold claim for any
block rests on the coherence diagnostics (within-block code topology,
ring tests, truncation ablations, shared-code evals),
synthetic-recovery calibration, and BH multiple-comparison correction
over the unknown-block search. Pre-registered confirmatory targets:
weekday ring, month ring.

## Configurations

| | rehearsal (0.9) | primary (Phase 1) | stretch |
|---|---|---|---|
| model | gemma-3-1b (d=1152, 26L) | gemma-3-4b (d=2560, 34L) | gemma-3-4b |
| sites | 6 in 25–90% band | 8 in 25–90% band (≈ layers 9–30, resolved at harvest, frozen in config) | 8 |
| G × b | 1024 × 4 | 4096 × 4 (16,384 latents) | 8192 × 4 |
| k (blocks/token avg) | 16 | 32 (128 active coeffs) | 32 |
| params (untied) | 57M | 671M | 1.34B |
| train VRAM est. | ~2 GB | ~9 GB | ~11 GB |
| store (whitened bf16) | ~11M tok ≈ 152 GB (8M train + 1M eval + 2M calib) | 53M tok ≈ 2.17 TB (38M train + 2M eval + 13M calib) | same store |

Baseline (every config): matched scalar crosscoder = **the b=1
Gram-constrained architecture** (round-2 R16): per latent,
Σ_s ‖d_j^s‖² = 1 with the same retraction — otherwise its coefficient scale
and quantization rate are gauge-dependent and the codec comparison
collapses. G·b scalar latents, BatchTopK with matched *training-average*
L0 (E[ℓ_t] = b·E[k_t]; each model uses its own realized counts at eval),
identical sites/whitening/data/order/precision/init scheme/tuning budget,
cold start both (finding 16). Same parameter count by construction. Warm
starts (from Phase-0 clusters) are exploratory runs only, never the
headline comparison. Scope note (F6, fidelity audit 2026-07-17): this
baseline is the b=1 special case of *our* architecture — signed latents,
no ReLU, energy selection — not a literature ReLU crosscoder (a signed
latent can carry feature + anti-feature in one unit). That is exactly
what R16's internal comparability requires; external comparisons to
published crosscoder numbers should note the convention gap.

**The primary H3 comparison runs both models at λ = 0** (at b=1 the
per-site nuclear term is a pure site-concentration penalty, not a rank
penalty, so nonzero λ is not architecture-fair); nonzero-λ BSC frontiers
are secondary.

λ_* grid {0, 3e-4, 1e-3} under the pinned reductions — the Phase −1
veto passed {3e-4, 1e-3} and failed 3e-3 (share concentration), making
**λ=1e-3 the Phase-1 primary** with 0 and 3e-4 as the lower arm; the
0.9 rehearsal re-checks the battery→production transfer. k frontier
{16, 32, 64}. Staged run matrix (R26): rehearsal confirms λ →
one-seed frontier exploration at 4b → both confirmatory
seeds only at the preselected operating points. The 4b headline runs are
not a hyperparameter search.

## Data & training topology

**Disk-backed whitened activation store on a dedicated 4 TB NVMe (decision
2026-07-16; jobe's stock disks are 2×1 TB — the v2.1 "~1.9 TB free" volume
never existed and the v2.1 store was unimplementable on the real machine,
D1).** Harvest once on CUDA: gemma forward in bf16 with the LM head
skipped (D7), 8 sites, seq len 1024, FineWeb-Edu sample streamed with a
pinned manifest (shard ids + seed, shared verbatim by BSC and baseline
runs); BOS and position-0/1 activations dropped. The 5M-token whitener
slice is harvested **first** and accumulated into statistics (never
stored; only μ_s, Σ_s, W_s persist — numerics in *Sites, coordinates,
whitening*); shards are then written **whitened, in bf16**, whitener hash
in every header (D8). Exact bytes: 40,960 B/token × 38M train = 1.557 TB;
× 2M eval = 81.9 GB; × 13M calibration = 532.5 GB (sizing rationale under
*Rate–distortion protocol*). Stored total ≈ 2.17 TB on the 4 TB volume,
with a ≥15% free-space floor and byte-exact pre-write abort checks
enforced regardless of headroom (D14).

Store I/O (D6; measured sequential buffered read on jobe: ~1.94 GB/s — a
160 MiB training batch has an ~82 ms I/O floor, so **token-random mmap
access is forbidden**): 2–8 GiB atomic shards, tensor layout
[token, site, d], sequence-contiguous writes; per-epoch shard-level
shuffle; contiguous chunk reads into a 32k–128k-token RAM shuffle buffer;
batches mixed within the buffer; 2–4 pinned prefetched batches; the
permutation seed recorded and shared by BSC and baseline. Training then
runs **without gemma in VRAM** — which is what keeps the G=8192 stretch
config live — with 2-epoch default (per-epoch reshuffle), held-out FVU
gap monitored for store overfit.

Checkpoints and residency (D14): a *resumable* 8-bit-Adam checkpoint at
the primary config is ~5.4 GB (bf16 forward copy + fp32 master + two
8-bit moment tensors), roughly double at G=8192 — v2.1's "~1.3 GB" was
the inference-only bf16 weights. Keep latest-resumable + best-inference
per run only; atomic write-then-rename with a free-space check that
aborts *before* the write. The 81.9 GB eval store exceeds the box's
61 GB RAM: eval is streamed sequentially, never RAM-resident.

Sample-power discipline (R24): 38M unique tokens is the demonstrated-power
envelope, not a free pass — Phase 1 reports learning curves at store
fractions (25/50/100%), block-stability across epochs, and reads rare-block
conclusions against the Phase −1 recovery-vs-frequency calibration. A
"mostly rank-1" verdict is stated as *conditional on demonstrated sample
power*, not as unqualified corroboration of flattening.

Escalation path if rare-feature starvation shows (dead fraction high or
held-out FVU still falling at epoch end): interleaved harvest+train
streaming (gemma co-resident; primary config only, ~17–18 GB) for a
≥ 100M-token budget — **and the scalar baseline retrains on the identical
extended manifest and token order** (R25); an escalated BSC is never
compared against the un-escalated baseline. Documented, not default.

Phase-0/rehearsal harvests on the M5 Max obey the MPS async-OOM discipline
(periodic `torch.mps.synchronize()`, zero-row guards — see AGENTS.md); the
big store is CUDA-side and additionally verified by content checksums and
per-shard non-zero/finiteness audits at write time.

Mandatory pre-commitment pilot (finding 31, extended in round 3 — D7,
D12): before the full store is written, a **≥3M-token exact-config pilot**
(>500 steps at batch 4096, with margin) runs the exact production path —
same sequence batching, all 8 hooks, dtype conversion, whitening,
checksums, shard writer — and must actually **exercise** AuxK
(rationale refreshed post-AuxK-respec, sol S7: the original "cannot
cross the 500-batch dead window" argument referenced the long-horizon
arm; the SASA default window is 100 batches, which a ≥3M pilot crosses
with margin — the pilot stays mandatory on operational-coverage
grounds), checkpoint/resume,
and final-threshold calibration, including a synthetic dead-encoder
revival test. Scope limit, worn openly (sol, F2/F10): at ~732 steps the
pilot sits entirely inside the 1k-step warmup — it validates the
warmup/early-training regime and operational mechanics, and can
*confirm* a previously selected lr/schedule point only in that regime;
it cannot choose between schedules or observe decay-tail behavior.
Schedule selection belongs to the 0.9.5 addendum at full rehearsal
horizon. Logged: model tok/s, writer GB/s, GPU duty cycle,
**data-wait time** (not just aggregate throughput), VRAM high-water,
dead-block trajectory, pre/post-retraction Gram eigenvalues, floor hits,
depth-share jumps, and aux/main gradient norms. The token budget and
config are re-confirmed against measurements, not estimates. This pilot
is a **separate, mandatory operational gate** — the 1b rehearsal cannot
stand in for it (D13).

## Rate–distortion protocol (H3, made honest)

What is compared is **held-out activation rate–distortion**, not "total
MDL" — parameter bits are deliberately out of scope and the claim is scoped
accordingly (finding 15). Three data splits: train, **calibration**, eval
(untouched until scoring). The calibration split is a stored 13M-token
split (unbudgeted and underspecified in v2.1 — D4): all codec fitting
happens here — canonical orientation, clipping quantiles, the count
model, the final inference threshold θ, and the single-site calibration
maps — and its power is accounted in **active counts per block**, not
tokens: at mean activation frequency k/G ≈ 0.78%, 13M tokens give ≈100k
active samples for an average block, i.e. ~100 observations beyond a
0.1% clipping quantile. Blocks under a predeclared active-count floor
are excluded from the codec comparison and reported separately. Codec,
pre-registered:

- **Canonical block orientation** (round-2 blocker R13): after training,
  each block's code space is rotated to diagonalize its calibration-set
  active-code second moment (descending), exploiting the residual O(b)
  gauge to fix a canonical frame; orientation is then frozen. Without this,
  an arbitrary rotation changes componentwise clipping and quantization
  error while the model is unchanged.
- **Quantizer** — per-coordinate uniform quantizer in the canonical frame:
  clipping range = calibration-set quantiles (0.1%/99.9%) per coordinate,
  2^q uniform levels within range, out-of-range values saturate to the
  clip; ranges and orientation are codec metadata (fit off-eval, applied
  identically in kind to both models). q swept {4, 6, 8}; distortion
  measured through the quantized codes.
- **Support bits/token** — −log₂ P(k_t) + log₂(G choose k_t), where the
  count model P is the empirical count distribution **fit on the
  calibration split and frozen** (R15 — the eval set cannot define its own
  code). The BatchTopK variable-k is thereby priced, not idealized
  (finding 13).
- **Realized counts** (R14) — each model pays its own: the BSC token pays
  for k_t active blocks (q·b·k_t amplitude bits), the scalar token for its
  own realized ℓ_t active latents (q·ℓ_t amplitude bits against
  log₂(Gb choose ℓ_t) support bits); only the *training-average* budgets
  are matched, E[ℓ_t] = b·E[k_t]. The support-term advantage is the entire
  structural bet, priced against the block's obligation to transmit b
  coordinates (finding 12).
- **Distortion** — whitened FVU, per site and pooled.
- **Comparison** — full frontiers over (k, q) for BSC vs matched scalar at
  λ = 0 both (see *Configurations*); the H3 verdict is frontier dominance
  over the shared operating region under **this declared codec** (R17 —
  the enumerative support code is deliberately usage-agnostic; an
  empirical-support-entropy sensitivity analysis accompanies it before any
  broader compression claim), not a single matched point (finding 14).
- **Uncertainty** — bootstrap over *sequences*, not tokens (R18 —
  neighboring-token activations are correlated and token bootstrap
  understates uncertainty); the two seeds displayed separately, never
  pooled.

## Phases (gates are decision heuristics with demonstrated power)

**Phase −1 — synthetic ground-truth harness. Offline; first deliverable.**
Planted dictionaries → synthetic activations → recovery, with the
generator built **gauge-correct** (D11): intrinsic coordinates u_g ∈ ℝ^r
are embedded as block codes z_g = A_g u_g through Gram-constraint-
satisfying concatenated decoders; planted rank is defined by the spectra
of the *contribution operators*, never by decoder rows — the constraint
forces every frame to rank b, so parked capacity is unidentifiable and is
not scored as failed recovery. Recovery is identifiable only modulo one
joint O(b) per block: evaluation matches blocks by assignment, then
applies a single **global** Procrustes alignment across all sites and
codes — per-site alignment would falsely certify shared coordinates.
Must include: shared blocks with cross-site frame rotation (the thing the
BSC exists for); flat-profile shared blocks under the **quantitative
λ-veto** (see *Rank regularizer* — admissible λ must keep planted
flat-profile share error under the predeclared tolerance, with the
λ=0-primary fallback if none passes); site-specific decoys (scored as
*expected site-exclusive recoveries*, not nothing-recovered nulls);
correlated scalar bundles under the **weakened null** (D11, correcting
the round-1 finding-9 disposition: perfect co-activation with
full-dimensional joint support is observationally equivalent to a block
under linear reconstruction, so the learner may legitimately bundle — the
gate is that ring/topology/coherence tests must not hallucinate *curved
manifold* structure on them); planted manifolds in both **hollow and
radially-thickened** versions of the same geometry (Michaud's regime
split, P18 — recovery must not be confounded with the geometry regime);
planted ranks 1..b at controlled depth profiles **and controlled feature
frequencies** (the recovery-vs-frequency calibration that Phase 1's
rare-block claims are read against, R24); Bhalla-style
capture/shattering/dilution metrics — restricted R², support size,
receptive-field spread (P19); the **rotation-equivariance test** (R8):
paired seeds from randomly O(b)-rotated inits must recover matching
spectra/subspaces, else decoders move off coordinatewise Adam; and the
**AuxK variant comparison** (P8; see *Sparsity hygiene*), including a
dead-encoder revival test. Recovery metrics: subspace principal angles,
rank recovery via contribution spectra, depth-profile fidelity, code
correlation after global alignment.
*Gate:* the pipeline recovers what was planted, does not hallucinate
structure in the nulls, and yields a nonempty admissible λ set (or the
documented λ=0 fallback). Lives in `tests/` + `scripts/`; battery runs on
jobe (CUDA) with the production optimizer — the retraction-vs-8-bit-Adam
ordering check thereby lands here rather than waiting for 0.9; code stays
device-generic and unit tests run anywhere.

**Executed and PASSED, 2026-07-16** (battery runs 1–6 +
capture-campaign sweep rounds 1–8;
[`docs/findings-phase-minus1-battery.md`](findings-phase-minus1-battery.md)).
Gate semantics ruled by a9: **strict capture-as-written** — tiled or
split recoveries do not pass; the sanctioned fix space is training
conditions, never gate criteria. The operating point that passes every
hard gate on 4 seeds: 10k steps × batch 1024 (≈10M tokens), selection
budget ratio 0.8 of E[active blocks/token] (budget is the driving
factor and monotone over the tested range — matched budget junk-fills
and tiles, 0.7 starves), spare capacity G ≈ 2.5×F (removes the
init-lottery establishment deaths), site decoys at rank 3 (pairwise
rank > b, so packing is lossy and separation objective-aligned), and
the bundle scenario's budget in **block-event demand** (per-feature
ΣF double-counts co-active features relative to the packed format the
learner correctly adopts; any post-packing budget slack junk-fills
through captured rings — tolerance <1% of a block's firings — and
hides them from all-firings norm-concentration). Training is
bit-deterministic per (seed, config, device). Also verified here:
retraction ordering under 8-bit Adam; rotation-equivariance on a
4-seed control arm (spans rotation-stable; spectra are basin-variant
— decoders stay on Adam); the recovery-vs-frequency calibration
(clean to f=0.01 at 10M tokens, per-seed lottery at 0.003 and below —
the curve Phase 1's rare-block claims are read against).

**Phase 0 — post-hoc blockification. Zero training; days.**
First, **positive control**: replicate the Engels weekday/month rings
through our exact pipeline on **the artifact Engels actually used —
Bloom's 2024 GPT-2-small layer-7 residual SAE** (~25k features; SAELens
registry identity verified at phase start), spectral clustering at their
n=1000 scale — a gemma null without this is uninterpretable (finding
17/19/35), and a failed transfer to a *different* SAE would demonstrate
nothing about pipeline power (P13). If that exact artifact is
unobtainable, the nearest release runs as a labeled *transfer control*
and a faithful replication path is added. The control is **observational
only**: Engels' causal tests were on Mistral/Llama — GPT-2 scores
near-chance on the modular-arithmetic tasks (P2). Then:
`google/gemma-scope-2-4b-pt` residual SAE at the saklas-convention analysis
depth (~65%); cluster decoder directions (cosine + spectral, Engels-style)
**plus an activation-dependence branch** (P15 — Bhalla: decoder-cosine
correlation alone "need not suffice" and is weakest exactly in
shattering/dilution regimes; an Ising fit on binarized codes, or a
documented conditional-dependence approximation, runs beside it);
within-cluster PCA of code contributions over a token stream (saklas `sae`
runtime harvest). Ring criteria, quantitative — Engels' own battery in
full (P14): cluster-restricted reconstruction (out-of-cluster latents
ablated), discard of samples where no cluster element activates, PC-plane
scan (1–2 through 4–5) with separability and ε-mixture scores averaged
across planes, the PC1-as-intensity **cone check** (the ring may live in
PCs 2–3 of a cone), clustering-stability + Jaccard-overlap analysis —
plus our additions: held-out circular decoding of the known cyclic
families, Fourier structure of code angle, label-permutation and
random-cluster nulls; BH correction over unknown-cluster search.
Phase −1 instrument lesson, binding here (findings §2.3):
norm-concentration (CV of code norms) is **never a ring detector by
itself** — it misses real rings both ways. Soft phase-splits (two
blocks co-firing with phase-dependent amplitude) score CV ≈ 0.22, and
*perfectly captured* rings score 0.17–0.43 whenever selection-budget
slack junk-fills trace-magnitude firings through the ring's block
(CV ≈ 0.97·√(junk fraction); production runs always have slack). Ring
evidence is therefore span-level and *gate-conditional* — the battery
above, evaluated on the tokens where the candidate feature is active.
Expect packed blocks (see *Identifiability caveat*) as ordinary
converged structure, not anomalies.
*Gate:* rings recoverable → live targets + warm-start candidates for
exploratory runs. Null **with the control passing** → "no rings recoverable
from this SAE at these depths at demonstrated power" — downgrades H1,
reroutes to a different SAE/depth or Qwen before any training. (What a null
does *not* establish: "gemma flattens token-level geometry" — the gate
language is scoped to what it measures.)

**Phase 0.5 — cross-layer coherence pre-test. Zero training; days.**
Verify Gemma Scope 2 layer coverage first (needs SAEs at 2–3 depths). The
gate metric is **not** per-latent cosine matching — independently-trained
per-layer SAEs can span the same rotating subspace with dissimilar
individual directions, which would fail exactly when the BSC premise holds
(finding 20). Instead: (i) principal angles between Phase-0 cluster
subspaces across depths, against a random-subspace null; (ii) paired-token
CCA/Procrustes between matched subspaces; (iii) out-of-sample coordinate
prediction — fit the layer-A→layer-B code map on half the tokens, report
residuals on the other half. Laptev cosine flow (2502.03032) demotes to a
matching bootstrap. *Gate:* spans match AND positions correspond → the
shared-code assumption has legs. Span-match without position-correspondence
is itself a finding (frames persist, codes transform) and steers toward
stage-blocks.

**Phase 0.9 — 1b dress rehearsal. Days; decision 2026-07-15.**
The entire ladder end-to-end on gemma-3-1b + `gemma-scope-2-1b-pt` at the
rehearsal config: harvest → whiten → store → train BSC + scalar baseline →
full eval suite → toy export. Plumbing gates only (harvest integrity,
training stability, eval determinism, hyperparameter sanity) — science
verdicts wait for 4b. Two scope corrections from round 3: the
rehearsal-narrowed λ set carrying to 4b is a **cross-model transfer
assumption**, worn openly — the one-seed 4b frontier stage is its
backstop; and a green rehearsal does *not* clear the principal 4b
operational risks (late-layer outlier behavior, 160 MiB-batch I/O,
671M-param optimizer/retraction throughput, full checkpoint mechanics,
calibration tail power) — those belong to the extended 4b pilot, a
separate mandatory gate (D13).

**Phase 1 — train the BSC on gemma-3-4b. ≈week part-time + 4090 runs.**
Primary config from the table; store-backed; cold-start BSC + b=1
Gram-constrained baseline; staged run matrix (rehearsal-narrowed λ →
one-seed frontier exploration → 2 confirmatory seeds at preselected
operating points). Headline outputs: rate–distortion frontiers at λ=0
(H3); per-block per-site effective-span histograms + spectra with the full
control battery and learning curves at store fractions (H4, stated at
demonstrated sample power); shared-code eval matrix (raw + calibrated);
block coherence diagnostics.
*Gate:* H3 in either direction is informative; proceed to Phase 2 if any
blocks resolve multi-dimensional, coherent, AND pass the shared-code evals.

**Phase 2 — the saklas bridge. Medium feature; lands in saklas, not here.**
Export a block as a multi-layer manifold artifact. The export must preserve
the per-site coordinate map, not just the span (finding 10): per site,
truncated SVD of whitened D_g^s (keep U_r, σ, and the b×r right-factor);
node positions embedded per site through the *full* map, never re-derived
from a bare basis. share = per-site **contribution-energy** shares —
never Frobenius decoder norms, which count parked frame capacity
(Phase −1, findings §2.5) — **re-expressed in
saklas's consumer-side whitener** (the two whiteners — training-side
harvest-fit vs consumer-side neutral-fit — are not interchangeable; the
seam is explicit). Near-50/50 share splits are flagged as probable
packed blocks (see *Identifiability caveat*) before any manifold claim. Origin (neutral-mean projection), σ (per-site code-density
residual), and node labeling (clustered code-density modes + max-activating
contexts) are *additional estimators with their own validation burden*
(finding 22) — `experiment naturalness` validates the intervention path, not
the discovery. Import lands as a saklas manifold-folder source
(`manifold.json` + per-model safetensors, per the existing producer/consumer
contract).

**Phase 3 — cross-model BSC. New horizon.**
Sites = layers × models at **constant site budget** (4 layers × 2 models = 8
sites; the v1 claim that cross-model work "doubles the harvest, not the
decoder scaling" was wrong under layers×models — finding 23). Base vs
instruct gemma first, **paired forwards on identical raw token sequences**
(shared tokenizer removes vocabulary alignment; chat-template asymmetry is
the residual caveat, handled by running both models template-free on the
same stream and treating template tokens as a documented distribution gap —
finding 24). Two config-freeze items from the fidelity audit (F4,
2026-07-17), both Minder-anchored: (i) **corpus mix** — Minder trained
the diffing crosscoder on FineWeb *plus lmsys-chat* (mixture
proportions not stated in the paper — ours must be chosen and pinned
explicitly, not "matched"); chat-specific latents need
chat-distribution data to fire, so H5's chat-specific manifolds (the
persona fan) are likely underpowered or distributionally missed on
FineWeb-Edu alone — the Phase-3 harvest adopts a web+chat blend (chat
rendered template-free per the paired-forwards rule), proportions
pinned at the freeze; (ii)
**per-model init pairing** — Minder initializes base/chat encoder and
decoder pairs *identically* (a deliberate start-shared prior for
diffing); our default independent per-site Gaussian+retraction does not
— paired vs independent init is decided at the freeze, default proposal:
paired (matching the diffing parent), with the independent init kept as
the documented ablation. Manifold-level diffing + persona-fan provenance
(H5), gated on the same shared-code evals per model. H5's toolkit is **causal, not only
observational** (P12, from Minder §3.1.2): selected-block base→chat
patching through per-block decoder differences with output-KL readout,
None/All/reconstruction-error baselines, full-response *and* early-token
KL (Minder's early-token gap ran >3× the full-response gap),
sequence-level confidence intervals, template-token stratification, and
Sentence-BERT-matched nonactivating controls for any autointerp.
Blockwise Latent Scaling is spec'd before the phase (P11): leave-one-
*block*-out and reconstruction targets per model/site, held-out b×b
(ridge- or Procrustes-constrained) maps rather than a scalar coefficient,
normalized improvement ratios, all calibrated on planted shared /
site-specific / shrunk / decoupled blocks in the Phase −1 harness.
Jiralerspong's Dedicated Feature Crosscoders (scalar shared/exclusive
partitions) are the architectural comparison point, and their
model-stitching / cross-model-steering validation of exclusivity is
adopted (P25). Gemma vs Qwen cross-arch only after
token alignment is solved; document-level pooling changes the object of
study (it answers a pooled question, not the token-level ring question) and
is out of scope for the ring claim.

## Out of scope for v1

- Cross-architecture token alignment (Phase 3's second half gates on it).
- Sphere/exotic topologies (authored-only in saklas for good reason).
- Weakly-causal crosscoder variants (reach for them only if stage-splitting
  dominates).
- Steering-quality tuning of imported manifolds (Phase 2 proves the pipe).
- Quantized activation storage for the primary runs. (v2.1's RD-floor
  rationale was wrong — distortion is measured on clean bf16 eval data;
  the real reason is that store quantization acts as an
  *architecture-dependent regularizer* that can shift support, dead
  rates, spectra, and bundling differently for BSC vs scalar, and raw
  fp8 on gemma's outlier channels is unsafe outright — D5.) A whitened,
  per-channel-scaled int8 codec is the documented store-compression
  escalation, admissible only after a paired bf16-vs-quantized pilot
  agrees within preregistered tolerances on the RD frontier, block
  matching, count/dead distributions, rank spectra, and shared-code
  diagnostics, across 2 seeds.
- b-sweeps beyond {1 (baseline), 4}: b ∈ {2, 8} noted as follow-up; a
  rank-4 ceiling is a design choice, not a discovery (finding 33).

## Risks

Structurally designed out: gauge-vacuous rank penalty, selection-score
proxy error, decoder death spiral (all via the Gram constraint); rogue
dims (whitened everything); fp16 overflow (banned); pipeline-failure
nulls (Phase −1 harness — **passed 2026-07-16** — plus pinned GPT-2
positive control, 0.9 rehearsal, extended 4b pilot).

Empirically mitigated, monitored (P1 — v2.1 wrongly listed this as
designed out): L1-style shrinkage/decoupling artifacts. BatchTopK
*substantially mitigates* them (Minder §2.3.2 "may address"; their own
BatchTopK run still shows 12.0% dead validation latents), so blockwise
Latent Scaling and the causal-diffing controls stay mandatory rather
than precautionary.

Instrumented, not eliminated: per-site penalty's site-concentration
pressure (synthetic gate + λ=0 control + contribution-covariance readout);
rank-histogram circularity (λ sweep, truncation ablations, seed stability);
acausal multi-view leakage (site-dropout eval matrix); block
bundling/packing (coherence diagnostics + BH correction + the
packed-block share signature); store overfit at 38M tokens
(held-out FVU gap; interleaved escalation path).

Honestly open: the block prior may mismatch language (the Fel hedge, and
b=4 is a hypothesis, not attribution-backed — Fel's shallow-vision optima
were ≈1–3 with "trust the direction, not any single value", P5) — H3/H4
are informative under the null and a flat dictionary corroborates the
flattening line. The novelty claim is scoped to LLM interpretability and
**survived a full-text sweep of all 13 reference sources (2026-07-16)**
in its narrow form: no checked work learns sparse multidimensional blocks
with one shared vector code and distinct per-site frames across sites.
Owed reads before any external claim: the older multi-view / coupled /
joint group-sparse dictionary-learning literature (largest remaining
risk), Yun 2103.15949, Lawson 2409.04185, SMixAE, Shafran et al.,
Hindupur/SPADE, Mishra-Sharma, and the Baskaran–Sklar crosscoder work if
public (P23, round-3 novelty verdict).

## Decision log

- **2026-07-15** — v2 rewrite after adversarial review. Gram-constrained
  decoders chosen over SASA's product penalty (selection exactness + clean
  per-site rank readout; product form kept as ablation). Disk-backed store
  chosen (a9: ~1.9 TB free on the 4090 box) over interleaved streaming;
  interleaved kept as the >100M-token escalation. Phase 0.9 (1b rehearsal)
  added (a9 sign-off). Primary config reduced to G=4096 after VRAM
  arithmetic; G=8192 stretch re-enabled by the store. Rate–distortion
  framing replaces "MDL" for H3. Phase 0.5 gate metric replaced (subspace
  principal angles + position correspondence, not per-latent cosine flow).
- **2026-07-15 (v2.1)** — round-2 amendments after verification pass
  (algebra PASS; freeze conditioned on these): loss reductions pinned
  (λ, α now meaningful); canonical-orientation quantizer completing the
  codec; baseline made explicitly b=1 Gram-constrained with realized
  counts; primary H3 at λ=0 both models; fp32-master retraction order +
  eigenvalue floor + decoder weight decay 0; calibrated site-dropout and
  rotation-invariant code-agreement stats; uncentered contribution second
  moment added as primary H4 readout; frame-capacity naming; λ-veto made
  quantitative with λ=0 fallback; store arithmetic in exact bytes with
  whitener accumulate-and-discard; sequence-level bootstrap; staged run
  matrix; escalation retrains the baseline. Design frozen for
  implementation from this state.
- **2026-07-16 (v2.2)** — round-3 amendments after two fresh-context
  sol passes + a fable parallel pass
  ([`design-review-2026-07-16.md`](design-review-2026-07-16.md)).
  Hardware ground truth: jobe is 2×1 TB, not "~1.9 TB free" — the v2.1
  store was unimplementable as frozen; **a9: dedicated 4 TB NVMe
  purchased**, preserving 8 sites / 38M tokens / G=8192 (interleaved
  G=4096 streaming remains the documented no-purchase fallback; disk
  spanning rejected). Store becomes whitened bf16 with an immutable
  hashed whitener + raw validation shard; 13M-token stored calibration
  split added, powered in active counts; final threshold calibration-fit
  (EMA demoted to diagnostic); shard/I-O layout pinned (sequential
  buffered shuffle, no token-random mmap; measured 1.94 GB/s); checkpoint
  arithmetic corrected (~5.4 GB resumable); eval streamed, never
  RAM-resident. AuxK re-specified from SASA C.1 with a three-variant
  comparison; Fel norm-calibrated init; SASA-ablation symmetric decay;
  retraction-frequency ablation noted. Phase −1 generator made
  gauge-correct (global Procrustes, contribution-spectrum ranks, weakened
  bundle null — correcting the round-1 finding-9 disposition — hollow +
  thickened manifolds, Bhalla capture/shatter/dilution metrics). Phase 0:
  positive control pinned to Bloom's layer-7 GPT-2 SAE (observational
  only), Engels battery completed, activation-dependence branch added.
  Pilot extended to ≥3M tokens and made a separate mandatory gate
  (v2.1's 1M pilot could not exercise AuxK); 0.9 "science, not code"
  overclaim struck; λ 1b→4b transfer worn openly. H5 inherits Minder's
  causal diffing + DFC stitching/steering validation; blockwise Latent
  Scaling spec'd as b×b maps. Digest corrected at six points (P1–P6);
  owed-reads list expanded. Design re-frozen from this state.
- **2026-07-16 (placement)** — Phase −1 battery moved from MPS to jobe
  (CUDA); jobe is idle and the move is strictly de-risking: the harness
  trains with the production 8-bit-Adam optimizer (CUDA-only), pulling
  the retraction-ordering verification forward from Phase 0.9; native
  batched linalg; faster λ×scenario×seed battery. Code remains
  device-generic; MPS/CPU stay supported for dev runs and unit tests.
  Hardware placement only — no science change; design remains frozen.
- **2026-07-16 (λ-veto outcome)** — the Phase −1 λ-veto protocol ran
  (battery runs 1–2, `docs/findings-phase-minus1-battery.md`): the
  admissible set is empty at every tested nonzero λ
  {3e-4, 1e-3, 3e-3} — recovery overlap collapses 0.95→~0.71, far
  below the 0.85-retention floor — so the design's documented fallback
  fires: **λ=0 primary, nuclear rank reported diagnostically**.
  Mechanism note: run-1's apparent share concentration under λ>0 was
  largely a Frobenius parked-capacity measurement artifact; under
  contribution-energy shares, λ≤1e-3 passes the share tolerance and
  the veto is driven by overlap alone. Consequences: recovery/depth
  readouts (and the Phase-2 `share` export) must use
  contribution-energy shares, never Frobenius; the Phase-1 grid should
  bracket smaller λ (3e-5, 1e-4) before ruling out a benign window at
  production scale. Executes the spec'd protocol — no frozen-surface
  change.
- **2026-07-16 (λ-veto outcome, superseding amendment)** — the
  preceding entry's measurement was made at 3k steps (a CLI default
  silently shadowed the 10k operating point; battery runs 3–4 were
  also affected — see findings §6.2). Battery run 5, the first honest
  operating-point run (10k × batch 1024 ≈ 10M tokens, G16, 4 seeds),
  re-ran the veto: the overlap collapse does not occur at this scale
  (overlap ≥0.95 through λ=1e-3) and the admissible set is
  **{3e-4, 1e-3}** — the "benign window opening with more data"
  anticipated above, at values already in the grid. Per the frozen
  protocol (largest admissible λ = primary): **Phase-1 primary is
  λ=1e-3**; λ=0 and 3e-4 remain in the sweep as the lower arm; at
  λ=3e-3 the binding violation is genuine share concentration (0.046
  > 0.02 tolerance), not overlap. Numbers: findings §2.4. Executes
  the spec'd protocol — no frozen-surface change.
- **2026-07-16 (v2.3, post-Phase−1 consolidation)** — Phase −1
  executed and **passed in full** (battery runs 1–6, capture-campaign
  sweep rounds 1–8; gate semantics ruled strict capture-as-written by
  a9; complete record in the findings doc). Consolidation sweep folds
  every Phase −1 learning into the body so the design reads as one
  document again: λ-veto outcome under *Rank regularizer* and
  *Configurations* (primary λ=1e-3, grid {0, 3e-4, 1e-3}, tolerances
  as run); battery operating point and verdict under *Phases /
  Phase −1* (10k × 1024, budget ratio 0.8, G ≈ 2.5×F, rank-3 decoys,
  block-event bundle budget, bit-determinism, rotation and retraction
  checks); packing economics under *Identifiability caveat*; the
  zero-slack ring-detection lesson under *Phase 0*; AuxK verdict
  (SASA C.1 confirmed) under *Sparsity hygiene*; frequency
  calibration (clean to f=0.01 at 10M tokens); Phase-2 `share` export
  pinned to contribution-energy shares with the packed-block flag.
  Older log entries remain as provenance; where body and log
  disagree, the body is current. **Design re-frozen as v2.3.**
- **2026-07-17 (v2.3.1, site-band amendment)** — Phase 0.5 executed
  and **passed** (findings in
  `findings-phase05-cross-layer.md`): month codes correspond linearly
  across depths while raw-basis frames rotate to chance alignment —
  and the depth bracket showed activation-space calendar rings live
  *early* (layer 9 of 34 ≈ 26% depth: weekday circ 0.981 + decoder
  |r| 0.886), with early↔late frame rotation exactly where the BSC's
  advantage over single-frame methods lives. The harvest site band
  predates that evidence; amended **40–90% → 25–90%** for both the
  rehearsal and Phase-1 configs (a9 ratified 2026-07-17) so the site
  bracket includes the early stream. Single-number change; all other
  frozen surfaces untouched.
- **2026-07-17 (v2.3.2 pending, fidelity amendments)** — paper-fidelity
  audit (fable pass, F1–F11;
  [`design-review-2026-07-17-fidelity.md`](design-review-2026-07-17-fidelity.md))
  at a9's direction "fix all findings"; sol counter-review in flight
  (thread `bsc-fidelity-audit-review`). Folded now: F1 (squared
  reconstruction is a deliberate pick against Minder's unsquared form),
  F6 (selection-statistic provenance vs Minder's v; b=1 baseline scope
  note), F7 (shrinkage-whitening label), F10 (warmup-fraction note), F4
  (Phase-3 freeze items: web+chat corpus mix, per-model init pairing),
  F9 (SASA-ablation default: out of the confirmatory matrix). Code: F8
  checkpoint free-space floor (trainer + test). Reassigned open items:
  F2 (optimizer recalibration) and F3 (AuxK window calibration) move
  from the discharged-at-0.9 claim to the **0.9.5 calibration
  addendum** on the existing 1b store, 4b pilot as backstop. No
  architecture or protocol surface changed. **Sol counter-review
  returned same day** (job `cx-20260717-110930-e764`): every finding
  sustained, several sharpened — F7 upgraded to moderate (shrinkage
  whitening leaves ~5× unequal residual site power in L_rec; per-site
  RMS renorm is now a pre-4b-store decision item), F3 arithmetic
  corrected (≤40), F6 energy/√energy wording fixed, F9 hardened (no
  cheap product-penalty arm), the 4b pilot's warmup-bound scope stated
  (732 steps < 1k warmup — it cannot select schedules), and seven
  code-sweep findings S1–S7 added (θ not serialized into the
  checkpoint; fp32-master vs bf16-forward eval ambiguity; HF corpus
  revision unpinned; held-out validation is an uncentered second
  moment; Welford wording vs linear-fp64 reality; Fel-init attribution
  softened to Fel-inspired; stale pilot-AuxK rationale) — all fixed or
  worn in this amendment set. Sol endorses 0.9.5 with a corrected
  matrix (dead-dynamics arm at k=32 preserving the Phase-1 k/G ratio;
  k=16 as labeled stress only; second seed for winner + runner-up;
  λ=0 confirmation of the winner). **Ratified by a9 2026-07-17,
  including the site-renorm arm; design frozen as v2.3.2.** The
  addendum's unconditional arms are scripted in
  `scripts/run_phase095_matrix.sh` (separate out-root; 0.9 artifacts
  untouched).
- **2026-07-17 (0.9.5 executed; Phase-1 optimizer pinned)** — the
  calibration campaign ran same-day on jobe (31 runs;
  [`findings-phase095-calibration.md`](findings-phase095-calibration.md)).
  The {1,2,3,6}e-4 grid did not bracket: two extension rungs added
  mid-campaign found the cosine optimum at 1.2e-3 with a cliff at
  2.4e-3, on both arms independently; linear_fifth turns over past
  6e-4; encoder-wd is a no-op; seed noise ~5× below the winner gap;
  λ=1e-3 re-confirmed ~free at the optimum. **Phase-1 optimizer
  defaults ratified by a9 2026-07-17: lr 1.2e-3, cosine, encoder-wd 0,
  λ=1e-3** — the 4b pilot carries lr-point confirmation (the optimum
  sits one doubling under the cliff; lr×G unmeasured at the winner);
  6e-4 cosine is the documented fallback on pilot instability. The
  k=16 stress arm is skipped per its condition (dead dynamics engaged:
  0.098% mortality at G=4096/k=32). **F7 site-renorm: a9 leans renorm
  (2026-07-17)** — the arm shows wash pooled FVU with the per-site
  allocation reversed (deep→shallow); the lean is pinned (or revisited)
  at 4b store build, before the first store byte is written.
- **2026-07-18 (0.9.6 executed; 4b optimizer + F7 + AuxK ratified)** —
  tier A ([`findings-phase096-tier-a.md`](findings-phase096-tier-a.md)):
  capture consolidation universal at 1b/16M tokens, calendar *order* a
  seed lottery that epochs don't buy at the winner lr. Tier B, the D13
  4b pilot ([`findings-phase096-pilot4b.md`](findings-phase096-pilot4b.md)):
  **D13 passes at lr 3e-4** — the 1b-ratified 1.2e-3 is destroyed by a
  warmup-peak edge-of-stability spike amplified by the SASA AuxK revival
  cascade (arm-independent: scalar spikes identically; no clipping in
  the loop); 6e-4 recovers damaged. Calendar probe: both rings available
  in the 4b stream across the site list; renorm the only arm capturing
  both families (month ring at the perm floor + weekday 7/7, best FVU);
  destroyed dictionaries show consolidation-without-order (mega-block).
  **Ratified by a9 2026-07-18:** (i) 4b optimizer point **lr 3e-4
  cosine** (λ=1e-3, enc-wd 0 unchanged; ~4.5e-4 rung noted unexplored —
  order tracks lr below the edge); (ii) **site list
  (9,12,15,18,21,24,27,30)** frozen for the production harvest;
  (iii) **F7: site-renorm designated** on ring-side + FVU + allocation
  evidence, conditional on the 3e-4 point (renorm's stability edge is
  lower); (iv) **AuxK capped for Phase 1** — the concrete mechanism
  (s_aux scaled to the live dead-set size, or α_aux < 1) moves off the
  SASA-faithful point and is pinned at Phase-1 config freeze; the
  loss-spike guard remains a requirement alongside it. Standing eval
  rule from the mega-block observation: top-1 capture is never read
  without ring order and FVU beside it.
