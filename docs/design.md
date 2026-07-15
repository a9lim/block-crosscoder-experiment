# block-crosscoder-experiment — design

**A block-sparse crosscoder (BSC) — dictionary learning whose atomic unit is a
subspace, with one shared code across layers — is the unsupervised generator
of saklas's manifold artifact, and nobody has built one.** The literature is a
2×2, {scalar, block} × {single-site, cross-site}, with three cells occupied
(SAEs; BSF/SASA; crosscoders and kin) and the fourth empty as of 2026-07-15.
This experiment fills it on gemma, measures whether the combination earns its
parameters, and lands discovered manifolds in a real steering/probe runtime.

Full literature provenance, gap sweep, and the synergy argument live in
[`docs/research/block-sparse-crosscoders-2026-07.md`](research/block-sparse-crosscoders-2026-07.md)
— read it before arguing with anything here. One paragraph of it matters for
the design: each parent fixes the other's failure. Blocks absorb cross-layer
within-subspace rotation that fragments scalar crosscoder latents (SASA's
splitting theorem, aggravated by depth); crosscoders solve cross-layer block
identification by construction and emit per-layer decoder norms — the depth
profile saklas bakes as `share`.

## Hypotheses

- **H1 (rings exist).** Gemma carries irreducible multi-dimensional
  token-level features (Engels-style weekday/month rings) findable by post-hoc
  blockification of an existing SAE. The saklas-side centroid-level flattening
  results (circumplex, taxonomy, months) don't rule this out — pooling can
  average a ring into a blob; they measure different objects.
- **H2 (cross-layer coherence).** Ring/block subspaces at different depths are
  matchable — the subspace persists while its frame may rotate. This is the
  load-bearing assumption of the shared-code architecture, testable for free
  before any training.
- **H3 (blocks earn their parameters at language).** A BSC beats a
  matched-width scalar crosscoder on MDL (the Fel comparison, transferred:
  block selection costs `log₂(G choose k)` vs `log₂(Gb choose kb)` per
  token). This is the honest language-viability arbiter — the Fel hedge says
  the block prior may mismatch language, and a mostly-rank-1 dictionary is a
  publishable *corroboration* of the flattening line, not a wasted run.
- **H4 (depth-resolved geometry).** Per-block per-layer effective rank (SASA's
  nuclear norm, applied per site) localizes *where in depth* structured
  geometry lives — testing whether flattening is a late-layer phenomenon,
  cross-checkable against J-lens depth center-of-mass as an independent depth
  signature.
- **H5 (manifold-level diffing).** With sites = models, per-model block
  presence and rank answer provenance questions: is the persona fan present in
  base gemma or installed by chat-tuning? Does fine-tuning change a shared
  manifold's rank/curvature?

## Architecture sketch

Sites s = selected layers (Phase 3: layers × models). One token, activations
x^s. G blocks of width b:

- **encode** — z_g = Σ_s E_g^s x^s (per-site b×d maps, summed; crosscoder
  convention).
- **select** — BatchTopK over blocks by p_g = ‖z_g‖₂ · Σ_s ‖D_g^s‖_F, norms
  taken **whitened per site**. BatchTopK, not L1: Minder et al. (2504.02922)
  show L1 manufactures Complete Shrinkage / Latent Decoupling — the
  block-level analogues would poison exactly the H5 diffing questions.
  Whitened selection: raw-L2 block selection on an LLM residual stream is
  massive-activation bait (the vision papers never hit this; saklas's
  `LayerWhitener` is the tool).
- **decode** — x̂^s = Σ_{g active} z_g D_g^s; loss Σ_s ‖x^s − x̂^s‖²_whitened
  + λ_* Σ_g Σ_s ‖D_g^s‖_* (nuclear norm **per site**: each layer chooses its
  own effective rank per block — the H4 measurement is this regularizer's
  byproduct).
- **diagnostics** — block-level Latent Scaling as the artifact flag; MDL vs
  the matched scalar baseline (H3); per-block per-site effective-rank
  histogram (H4).

Known failure mode, accepted: a feature whose *position* transforms across
depth (computed features — Engels' modular arithmetic) can't be one shared
code and will split into stage-blocks with distinct depth profiles. Legible
(stages become visible objects), and Phase 0.5 measures how much of it to
expect.

## Phases (each gates the next)

**Phase 0 — post-hoc blockification. Zero training; days.**
Cluster decoder directions of an existing SAELens SAE on gemma (cosine +
spectral, Engels-style); fit within-cluster PCA on codes over a token stream;
look for irreducible multi-dim clusters — weekday/month rings first, then
unknowns. Harvest via saklas's `sae` runtime.
*Gate:* does gemma ring at the token level (H1)? No → sharpened flattening
finding; take the question to Qwen before training anything. Yes → the rings
are Phase 1's live targets and warm-start candidates.

**Phase 0.5 — cross-layer coherence pre-test. Zero training; days.**
Per-layer SAELens SAEs at 2–3 depths; match latents across layers with
data-free cosine flow (Laptev et al. 2502.03032); test whether Phase-0 ring
clusters persist/rotate across depth (H2).
*Gate:* coherent → shared-code assumption has legs. Incoherent → finding in
itself, and the BSC case weakens before a single GPU-hour is spent.

**Phase 1 — train the BSC. ≈week part-time + 4090 runs.**
Workspace-band BSC as sketched, on gemma-3-4b (d=2560, ~8 sites, G=8k × b=4
→ 32k latents, ~1.3B params untied — tight on the 4090; drop to G=4k / 6
sites / tied encoder / 8-bit Adam as needed). Matched-width scalar
crosscoder as the baseline. Streamed activation harvest (~40 KB/token at 8
sites; storage is a non-starter).
*Headline plots:* MDL comparison (H3); per-block per-site effective-rank
histogram (H4).
*Gate:* H3 in either direction is informative; proceed to Phase 2 if any
blocks resolve multi-dimensional and coherent.

**Phase 2 — the saklas bridge. Medium feature; lands in saklas, not here.**
Import a block as a complete multi-layer manifold artifact: per-layer basis
via QR of D_g^l, origin = neutral-mean projection (saklas convention), nodes
= clustered code-density modes auto-labeled from max-activating contexts,
share = whitened ‖D_g^l‖ normalized to mean 1, σ from per-site code-density
residual. Steerable via `%`/`subspace_inject`, probeable through existing
channels, `experiment naturalness` as the on-manifold-vs-straight-chord
eval. This is the novel deliverable: unsupervised discovery feeding a real
intervention runtime exists nowhere.

**Phase 3 — cross-model BSC. New horizon.**
Sites = models. Base vs instruct gemma first (same tokenizer — token
alignment is a non-issue): manifold-level diffing, persona-fan provenance
(H5). Then gemma vs Qwen cross-arch for "does Qwen ring?", once token
alignment is solved or side-stepped (document-level pooling is the fallback,
with the pooling caveat H1 exists to warn about). Also solves saklas's
deferred cross-model Procrustes-transfer TODO structurally: both models'
embeddings of one shared manifold, by construction.

## Out of scope for v1

- Cross-architecture token alignment (Phase 3's second half gates on it).
- Sphere/exotic topologies (authored-only in saklas for good reason).
- Weakly-causal crosscoder variants (reach for them only if stage-splitting
  turns out to dominate).
- Steering-quality tuning of imported manifolds (Phase 2 proves the pipe;
  calibration is saklas-side work).

## Risks

The Fel hedge (block prior may mismatch language) is absorbed by H3/H4 being
informative under the null. Rogue dims and L1 artifacts are designed out
(whitened selection, BatchTopK). The shared-code assumption is pre-tested
(Phase 0.5). SAE training hygiene transfers (input normalization; dead-latent
resuscitation becomes a block-level aux loss). Compute fits the 4090 at
reduced config; the harvest loop on MPS must respect the async-OOM
silent-zeros discipline (see AGENTS.md).
