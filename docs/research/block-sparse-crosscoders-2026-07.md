# Block-sparse crosscoders: unsupervised manifold discovery for saklas

*Research digest, 2026-07-15 — the canonical document for this program. Merges
the 2026-07-12 BSF digest (prompted by Fel et al. 2606.25234, "can we do this
for LLMs?") with the 2026-07-15 crosscoder extension (prompted by a9: "why
hasn't anyone combined BSF with crosscoders?"). Verification provenance: Fel
et al. read at primary source (full HTML, 07-12); SASA (2606.06333) abstract;
Engels (2405.14860) prior knowledge + search-verified; SAE-scaling
(2509.02565) [snippet]; crosscoder gap swept via web + HF paper search
(07-15); Minder 2504.02922, Gorton 2410.24184, Jiralerspong & Bricken
2602.11729 at abstract level; crosscoder formalism from the Anthropic post
(fetched, summary-level) + prior knowledge.*

**Verdict: a block-sparse crosscoder (BSC) — subspace-unit dictionary
learning with one shared code across layers (later: across models) — exists
nowhere in the literature, and the combination is better than the sum of its
parts: each parent fixes the other's characteristic failure. It is, almost
term for term, the unsupervised generator of saklas's manifold artifact.
The unclaimed territory found in the original sweep (unsupervised manifold
discovery feeding a real intervention runtime) widens rather than closes:
BSF/SASA have discovery at one layer, crosscoders have cross-layer scalar
features, saklas has the full subspace steering + probe runtime — nobody has
the triangle.**

---

## Parent 1 — Block-sparse featurizers (Fel et al., arXiv:2606.25234)

"Structuring Sparsity: Block-Sparse Featurizers Capture Visual Concept
Manifolds." Vision (DINOv3, InceptionV1, SDXL). The move: generalize SAEs by
grouping dictionary atoms into blocks of dimension b and enforcing sparsity
at the block level, so the atomic unit of interpretation is a subspace, not a
direction. A concept then has two readouts — **presence** (block norm ‖z_m‖₂,
the analogue of an SAE activation) and **position** (the within-block code,
i.e. where on the concept manifold this token sits).

Three variants, shared linear decoder `x̂ = zD`:

- **Vanilla BSF** — free untied encoder; hard block projection Π_k keeps the
  k blocks of largest ‖z_m‖₂, zeroes the rest; reconstruction-only loss.
- **Grassmannian BSF** — tied; each block D_m ∈ St(b,d) an orthonormal
  Stiefel frame; `z = Π_k(γ x Dᵀ)` with a scalar energy-compensation γ.
- **Group Lasso BSF** — free encoder + block soft-threshold
  `sh_θ(z)_m = max(1 − θ/‖z_m‖₂, 0) z_m` (prox of the ℓ₂,₁ norm).

Manifold coordinates are extracted post hoc: collect each active block's
contributions `m_m = z_m D_m` over the token stream, PCA them; the PCA frame
is the concept's natural basis (their curve-detector block recovers the full
orientation circle, with Fourier modes ω=1/2/3 at 59/18/12% of variance).

**The MDL argument** is the honest core: framing the code as compression,
block selection costs `log₂(G choose k)` bits vs an SAE's
`log₂(Gb choose kb)` — BSFs beat b=1 SAEs on total description length at
every dictionary width on DINOv3, with optimal b = 2–4. Task-derived
distortion floors (classification tolerates R²≈0.8, depth needs ≈0.9) anchor
the comparison so reconstruction alone doesn't decide it.

**Steering** (SDXL): clamp a block's code to waypoints on a Kohonen SOM fit
to the block's empirical code density — i.e. move only within the subspace's
high-density region. Cruder than saklas's kernel: no along/onto split, no
tube thickness, grid waypoints instead of a fitted surface.

Authors' own hedge on language: "applying block sparsity directly to language
or to video would mismatch the prior… I suspect the matched object should
change accordingly."

## Parent 1′ — the LLM relaxation exists (SASA, arXiv:2606.06333)

Dalili & Mahdavi, "Subspace-Aware Sparse Autoencoders," June 2026. Same move,
LLM-native: decoder subspaces + Top-s group gating + a **nuclear-norm
rank-adaptive regularizer** so each block chooses its effective rank. Theory:
reconstructing an intrinsically ≥2-D feature with 1-D decoders needs
exponentially many atoms (the formal version of feature splitting); block
size r ≥ d_i restores polynomial sample complexity. Empirics on GPT-2 +
Mistral-7B: reduced splitting and absorption, improved monosemanticity,
matches/exceeds standard SAEs on roughly half the token budget.
**No intervention/steering experiments at all.** Single layer, single model.

## Parent 2 — Crosscoders (Anthropic 2024, and kin)

One shared *scalar* latent reads from and writes to N sites at once — layers,
base-vs-chat models, checkpoints (Ge et al. 2509.17196), and, since
Jiralerspong & Bricken (2602.11729, Feb 2026), different architectures.
Reconstruction is summed over sites; the sparsity penalty weights each
latent by its summed per-site decoder norms, so the per-site norm profile is
free to vary — that profile *is* the depth profile of a cross-layer feature,
or the model attribution in diffing. Resolves cross-layer superposition (a
feature persisting layers 3–15 is represented once, not rediscovered per
layer). The unit of interpretation is still a direction.

Known pathology (Minder, Dumas, Juang, Chugtai, Nanda — 2504.02922): the L1
loss manufactures **Complete Shrinkage** and **Latent Decoupling** — latents
that read as "chat-only" when the concept exists in both models. Fixes:
**BatchTopK** selection instead of L1, plus **Latent Scaling** as the
diagnostic. Any BSC design must inherit these.

## Supporting landscape

- **Engels et al. (2405.14860)** — ground truth that irreducible multi-dim
  features exist in LLMs: weekday/month *circles* in GPT-2 and Mistral-7B,
  found by clustering SAE decoder directions, causally implicated in modular
  date arithmetic. Token-level, not centroid-level (see the science hook).
- **SAE scaling under feature manifolds (2509.02565)** [snippet] — SAEs
  reduce loss by tiling manifolds with many sparsely-activating latents at
  the expense of rarer features; block sparsity is the natural structural
  fix.
- **Hindupur, Lubana, Fel, Ba — "Projecting Assumptions"** — an SAE's
  architectural prior determines which concept geometries it can see at all.
  Single-site theory; reads as the motivation for a BSC.
- Disambiguations: **Group-SAE (2410.21508)** is layer-grouping for training
  efficiency — not this. **Gorton, "Group Crosscoders" (2410.24184)** —
  "group" = symmetry group: scalar-latent crosscoding across D₃₂-transformed
  *inputs* in InceptionV1, clustering equivariant feature families (the same
  orientation manifold Fel's curve blocks capture, reached via
  transformation-tying instead of block structure). Adjacent in spirit, not
  the thing. **CLTs** are cross-layer but scalar, built for circuit
  attribution. Single-site kin: "Do Sparse Autoencoders Capture Concept
  Manifolds?" (2604.28119), bilinear autoencoders (2605.08891).

## Gap check (2026-07-15)

The literature is a 2×2 — {scalar, block} × {single-site, cross-site} — with
three cells occupied (SAE; BSF/SASA; crosscoder + kin) and the fourth empty.
Searched: "block sparse crosscoder", "group sparse crosscoder", crosscoder ×
manifold/multi-dimensional latents, SASA follow-ups, Hindupur. Nothing.

**Why the cell is empty** (read: no blocker, just a young intersection):
blocks-for-LLMs is four weeks old; the crosscoder community's agenda is
diffing and circuits, not geometry; the geometry people (Engels, Fel,
Hindupur) haven't picked up crosscoders except Gorton in vision; and the
decoder-parameter multiplier (×N sites) discourages casual replication. The
steering-runtime half remains saklas-only territory throughout.

## Why the combination is synergistic, not a bolt-on

**Blocks fix the crosscoder's manifold failure.** A scalar crosscoder latent
forces one shared activation with one direction per site. A feature that
lives in a subspace and *rotates* within it across depth — a ring whose
frame drifts, which is what computation on a ring looks like — cannot be one
scalar latent; it fragments into families of latents with staggered depth
profiles. This is SASA's splitting theorem with depth as an extra splitting
pressure. A block absorbs within-subspace cross-site rotation into the
per-site frames D_g^s while the code stays put. [synthesis — the cross-layer
aggravation is our argument; the single-site theorem is SASA's]

**Crosscoders fix BSF's layer myopia.** A single-layer BSF gives blocks at
one layer, with no principled correspondence to blocks fit at another layer
(matching independently-trained dictionaries is notoriously unstable). The
crosscoder solves block identification across depth *by construction* — one
block, one shared code, per-layer embeddings — and hands you the depth
profile (per-site norms) for free. Saklas manifolds are multi-layer objects;
a single-layer block import was always going to be an awkward seam. This
removes it.

**The saklas convergence, exact.** Saklas independently converged on
"concept = subspace + position + thickness" from the steering side; BSF/SASA
arrived from the dictionary-learning side; and the discover-mode fit is
already a hand-authored crosscoder in miniature — one shared coordinate
layout from the cross-layer consensus Gram, per-layer embeddings, baked
per-layer Mahalanobis shares. A BSC is the unsupervised generator of exactly
that artifact:

| BSC object | saklas object (existing) |
|---|---|
| shared block code z_g | discover coords / `%` position (consensus by construction) |
| per-layer frame D_g^l | `LayerSubspace.basis` |
| per-layer decoder norm ‖D_g^l‖ | baked per-layer `share` (normalize to mean 1) |
| block presence ‖z_g‖ | probe `:fraction` / `:membership` channels |
| per-site code-density residual | σ-field tube fit (per-layer `sigma_at`) |
| SOM high-density waypoints (Fel) | σ-field tube fit (strictly richer) |
| clamp-to-waypoint steering (Fel) | `subspace_inject` along/onto |
| on-manifold vs straight chord | `experiment naturalness` (built-in eval) |
| per-site nuclear-norm rank r_g^l | *new measurement*: depth-resolved intrinsic dim |

Import path: block → standard manifold folder (per-layer basis via QR of
D_g^l, origin = neutral-mean projection per saklas convention, nodes =
clustered code-density modes auto-labeled from max-activating contexts,
share = whitened ‖D_g^l‖). Steering/probing needs ~zero change; the work is
the import path.

## Sketch (formal)

Sites s = selected layers (later: layers × models). One token, activations
x^s. G blocks of width b:

- encode: z_g = Σ_s E_g^s x^s  (per-site b×d encoder maps, summed — the
  crosscoder convention)
- select: **BatchTopK over blocks** by presence p_g = ‖z_g‖₂ · Σ_s ‖D_g^s‖_F,
  with the norms taken **whitened per site**. BatchTopK, not L1: the
  block-level analogues of Complete Shrinkage / Latent Decoupling (a
  base-model embedding driven to zero while the concept exists in both
  models) would poison exactly the manifold-diffing questions we care about.
  Whitened selection: raw-L2 block selection on an LLM residual stream is
  massive-activation bait — the vision papers never hit this (DINOv3 stats
  are tamer); the whitener is our home turf.
- decode: x̂^s = Σ_{g active} z_g D_g^s ; loss Σ_s ‖x^s − x̂^s‖²_whitened
  + λ_* Σ_g Σ_s ‖D_g^s‖_* (SASA's nuclear norm, **per site**, so each layer
  chooses its own effective rank of each block).
- diagnostics: block-level Latent Scaling (per-site
  reconstruction-contribution regression) as the artifact flag; per-block
  per-site effective-rank histogram as the headline geometry measurement;
  MDL vs a matched-width scalar crosscoder — the `log₂(G choose k)` vs
  `log₂(Gb choose kb)` argument carries over per token unchanged
  [synthesis, straightforward], and remains the honest language-viability
  arbiter it was in the single-layer plan.

Honest failure mode of the shared code: a feature whose *position* genuinely
transforms across depth (Engels' modular arithmetic — the answer-day differs
from the input-day) can't be one block with one code. It will split into
stage-blocks (input-ring, output-ring) with distinct depth profiles. That's
legible — stages become visible objects — not a dealbreaker; it's also what
the acausal-crosscoder critique predicts, and weakly-causal variants exist
if it bites. [speculation until measured]

## The science hook: centroid-level vs token-level, now depth-resolved

Our replicated "gemma flattens structured concept geometry" results
(circumplex λ2/λ1=0.31; taxonomy flat r≈0.15; months auto-fits flat) are all
**centroid-level** — one pooled point per concept corpus. Engels' rings are
**token-level**. Pooling a ring that individual tokens traverse can average
it into a blob, so the flattening findings and the ring findings don't
actually contradict — they measure different objects. A BSC tests this
directly, and adds the depth axis the single-layer plan lacked:

1. **Depth-resolved geometry** — per-site effective rank asks "*where in
   depth* does gemma keep rings?" All the flattening results are also
   single-depth-regime; a BSC measures token-level geometry as a function of
   depth in one fit, and cross-checks against J-lens depth CoM as an
   independent depth signature.
2. **Manifold-level model diffing** — the diffing question upgrades from
   "which latents are chat-only" to "which *manifolds* are chat-only, and
   does fine-tuning change a shared manifold's rank/curvature." Concrete: is
   the persona fan present in base gemma, or does chat-tuning install it?
   (Assistant Axis provenance.) Does RLHF flatten affect geometry? Base vs
   instruct gemma shares a tokenizer — the clean first pairing.
3. **Cross-model transfer without Procrustes** — sites = models gives both
   models' embeddings of one shared manifold by construction (the deferred
   `io/manifold_lifecycle.py` TODO, solved structurally). Cross-arch works
   (Jiralerspong & Bricken); "does Qwen ring?" becomes a per-model rank
   readout on a shared block. Token alignment across tokenizers is the real
   wrinkle — start same-tokenizer where it's a non-issue.

## Risks / gotchas

- **The prior may genuinely mismatch language** (the Fel hedge). Vision
  manifolds come from continuous transformation groups; language has fewer
  obvious continuous parameters, and Engels found a handful of circles, not
  hundreds. Expect most blocks to resolve effectively rank-1 — SASA's
  rank-adaptivity is the right response, and per-block (now per-site)
  effective rank is the thing to measure. A mostly-flat dictionary would
  *corroborate* the gemma-flattening line, not waste the effort — and flat
  blocks steer on the fast affine path anyway.
- **Rogue dims** — selection and reconstruction in whitened space, as above.
- **L1 sparsity artifacts** — BatchTopK + block-level Latent Scaling, as
  above.
- **Shared-code assumption** — stage-splitting on computed features, as
  above; Phase 0.5 pre-tests it for near-free.
- **SAE training hygiene transfers**: input normalization, dead-latent
  resuscitation as a block-level aux loss.
- **Compute is fine, with one step up from the single-layer plan.**
  Single-layer floor (prior digest): vanilla BSF is a ~30-line delta over a
  TopK SAE; gemma-2-2b/3-4b, one workspace-band layer, G≈8–16k × b=4, a few
  hundred million streamed tokens — a 4090 job. BSC: ~8 workspace-band sites
  on gemma-3-4b (d=2560, every other layer), G=8k × b=4 → 32k latents,
  decoder 8 × 82M ≈ 655M params, ~1.3B with untied per-site encoders. Full
  config with mixed-precision AdamW ≈ 16–21 GB — tight on the 4090;
  comfortable with 8-bit Adam, a tied (Grassmannian) encoder, G=4k, or 6
  sites. The matched-width scalar-crosscoder baseline is the same size.
  Activations stream (~40 KB/token at 8 sites; 200M tokens ≈ 8 TB raw, so
  storage was never on the table); harvest throughput is unchanged, just
  wider taps per forward. Cross-model doubles the harvest (two forwards per
  token), not the per-site decoder scaling.

## Plan (phased; each phase is a go/no-go gate for the next)

**Phase 0 — post-hoc blockification (days; zero training).** Cluster the
decoder directions of an existing SAELens SAE on gemma (Engels-style:
cosine + spectral clustering), fit within-cluster PCA on codes over a token
stream, look for irreducible multi-dim clusters (weekday/month rings first,
then unknowns). Uses the existing `sae` runtime for harvest. Deliverable:
does gemma ring at the token level? If yes → live target for Phase 1. If no
→ sharpened flattening finding; take the question to Qwen before training
anything.

**Phase 0.5 — cross-layer coherence pre-test (days; zero training).** Take
per-layer SAELens SAEs at 2–3 depths, match latents across layers with
data-free cosine flow (Laptev et al. 2502.03032), test whether Phase-0 ring
clusters persist/rotate across depth. If ring subspaces are matchable across
layers at all, the shared-code assumption has legs; if they're incoherent,
that itself is a finding (and weakens the BSC case before any training).

**Phase 1 — train the BSC (≈week part-time + 4090 runs).** Workspace-band
block-sparse crosscoder as sketched: BatchTopK block selection in whitened
space, per-site nuclear norm, matched-width scalar-crosscoder baseline. The
two headline plots: the MDL comparison vs b=1, and the per-block per-site
effective-rank histogram. Warm-start from Phase-0 clusters if they exist
(cuts token budget).

**Phase 2 — the saklas bridge (the novel bit; medium feature).** A
`discovered` manifold source: import a block as a complete multi-layer
fitted artifact (per-layer span + coords + σ from code density + share from
whitened decoder norms), steerable via `%` / `subspace_inject`, probeable
through existing channels, `experiment naturalness` as the
on-manifold-vs-straight-chord eval. Steering/probe machinery needs near-zero
change; the work is the import path + artifact format extension — and with
the BSC there is no single-layer seam left in it.

**Phase 3 — cross-model BSC (new horizon).** Base vs instruct gemma (same
tokenizer) for manifold-level diffing → persona-fan provenance; then gemma
vs Qwen cross-arch for the ring question, once token alignment is solved or
side-stepped (document-level pooling is the fallback, with the pooling
caveat the science hook warns about).

## Sources

- Fel et al., "Structuring Sparsity: Block-Sparse Featurizers Capture Visual
  Concept Manifolds" — arXiv:2606.25234 (primary, full text, 07-12)
- Dalili & Mahdavi, "Subspace-Aware Sparse Autoencoders" (SASA) —
  arXiv:2606.06333 (abstract)
- Anthropic, "Sparse Crosscoders for Cross-Layer Features and Model Diffing"
  — transformer-circuits.pub/2024/crosscoders (fetched, summary-level)
- Minder, Dumas, Juang, Chugtai, Nanda — "Overcoming Sparsity Artifacts in
  Crosscoders to Interpret Chat-Tuning" — arXiv:2504.02922 (abstract +
  search-verified)
- Jiralerspong & Bricken — "Cross-Architecture Model Diffing with
  Crosscoders" — arXiv:2602.11729 (abstract)
- Engels et al., "Not All Language Model Features Are One-Dimensionally
  Linear" — arXiv:2405.14860
- "Understanding sparse autoencoder scaling in the presence of feature
  manifolds" — arXiv:2509.02565 [snippet]
- Gorton — "Group Crosscoders for Mechanistic Analysis of Symmetry" —
  arXiv:2410.24184 (disambiguation + adjacency; abstract)
- Hindupur, Lubana, Fel, Ba — "Projecting Assumptions: The Duality Between
  Sparse Autoencoders and Concept Geometry" (motivation; Semantic Scholar
  record)
- Laptev, Balagansky, Aksenov, Gavrilov — "Analyze Feature Flow to Enhance
  Interpretation and Steering" — arXiv:2502.03032 (Phase 0.5 tool)
- Ge et al. — "Evolution of Concepts in Language Model Pre-Training" —
  arXiv:2509.17196 (checkpoint-axis crosscoding; landscape)
- Group-SAE — arXiv:2410.21508 (disambiguation only); "Do Sparse
  Autoencoders Capture Concept Manifolds?" — arXiv:2604.28119; "Bilinear
  autoencoders find interpretable manifolds" — arXiv:2605.08891 (landscape)
- Hindupur et al. 2025 was the "parallel LLM work" flagged unverified in the
  07-12 digest; resolved above.
