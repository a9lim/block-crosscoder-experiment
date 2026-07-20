# Block-sparse crosscoders in the literature

*Position and source ledger, last full-text sweep 2026-07-16. The executable
method is [`design.md`](design.md); this document establishes intellectual
lineage, adjacent work, and the scope of the novelty claim.*

## Position

The project joins two young lines of work:

- block-sparse featurizers and subspace-aware sparse autoencoders learn a
  multidimensional subspace as one sparse unit, but at one site;
- crosscoders learn one sparse scalar code across layers, models, or
  checkpoints, but keep one-dimensional units.

The checked LLM-interpretability literature therefore forms a 2×2:

| | one site | shared across sites |
|---|---|---|
| scalar unit | SAE | crosscoder and variants |
| subspace unit | BSF / SASA | **BSC (this project)** |

Narrow claim: as of the sweep, no checked work learned sparse
multidimensional blocks with one shared vector code and distinct frames
across model sites. This statement is scoped to the thirteen sources below
and LLM interpretability. It is **not yet** a claim over the older
multi-view, coupled, joint, or group-sparse dictionary-learning literature;
that is the largest owed sweep before publication.

The combination is not a mechanical tensor-product extension. Blocks absorb
within-subspace rotation that fragments scalar crosscoder latents, while a
crosscoder solves cross-layer block identity by construction. The one shared
code supplies presence and position; per-site frames supply the changing
embedding and depth profile.

## Parent 1: block-sparse features

Fel et al., *Structuring Sparsity: Block-Sparse Featurizers Capture Visual
Concept Manifolds* (arXiv:2606.25234), generalize a sparse dictionary from
scalar atoms to width-`b` blocks and impose sparsity over block norms. A unit
then exposes both presence (`||z_g||`) and within-feature position (`z_g`).
Their vision experiments recover transformation manifolds and motivate the
support-cost advantage of transmitting one block identity rather than `b`
independent scalar identities.

Important limits on attribution:

- their MDL calculation includes support, amplitudes, residual, and
  amortized dictionary bits; this project's declared activation codec omits
  parameter bits and is an activation rate–distortion comparison, not a Fel
  replication;
- their useful widths cluster around one to four with an explicit warning
  to trust the direction, not a single optimum. Width four here is a tested
  design choice, not a literature-derived natural constant;
- their authors explicitly hedge that video/language may require a
  different structured prior.

Dalili and Mahdavi, *Subspace-Aware Sparse Autoencoders* (SASA,
arXiv:2606.06333), bring grouped subspaces to GPT-2 and Mistral. Their theory
shows why reconstructing a multidimensional feature with scalar atoms can
require exponentially many directions; their nuclear penalty acts on the
end-to-end encoder/decoder product and is gauge-invariant. SASA supplies the
dead-group auxiliary-loss starting point and a rank-adaptive single-site
comparison. It does not share one vector code across layers or models and
contains no intervention experiments.

## Parent 2: cross-site scalar codes

Anthropic's 2024 sparse crosscoder learns one scalar latent across layers or
models, with per-site encoder/decoder weights and a free site-norm profile.
It resolves repeated scalar features across depth and made model diffing a
central application.

Minder et al., *Overcoming Sparsity Artifacts in Crosscoders to Interpret
Chat-Tuning* (arXiv:2504.02922), identify Complete Shrinkage and Latent
Decoupling under L1 training. BatchTopK mitigates rather than eliminates
these pathologies, and Latent Scaling remains the diagnostic. Their causal
patching protocol—including None/All/reconstruction-error controls and the
stronger early-token KL readout—anchors this project's future cross-model
phase.

Jiralerspong and Bricken, *Cross-Architecture Model Diffing with
Crosscoders* (arXiv:2602.11729), learn scalar shared/exclusive partitions
across architectures and validate them with model stitching and cross-model
steering. Their Dedicated Feature Crosscoder is the Phase-3 scalar
comparison point; it does not occupy the block/cross-site cell.

Ge et al. (arXiv:2509.17196) extend crosscoding over pretraining
checkpoints. Gorton, *Group Crosscoders for Mechanistic Analysis of
Symmetry* (arXiv:2410.24184), crosscodes scalar features across
transformation-indexed inputs in vision. Both are adjacent cross-site
geometry, but neither uses one sparse multidimensional vector unit.

## Evidence that one-dimensional units are insufficient

Engels et al., *Not All Language Model Features Are One-Dimensionally
Linear* (arXiv:2405.14860), find weekday and month rings by clustering SAE
decoder directions and causally test homologous geometry in Mistral and
Llama. GPT-2 supplies an observational positive control, not a successful
causal task result. Their restricted reconstruction, plane scan, cone,
separability, mixture, stability, and null controls informed the Phase-0
discovery battery.

Michaud et al. (arXiv:2509.02565) analyze SAE scaling in the presence of
feature manifolds. Their tiling regime is possible, not established as the
universal behavior of real SAEs; radial thickness changes the economics.
This motivated both hollow and thickened synthetic controls.

Bhalla et al., *Do Sparse Autoencoders Capture Concept Manifolds?*
(arXiv:2604.28119), motivate activation-dependence discovery and explicit
capture/shattering/dilution measures. Dooms et al., *Bilinear Autoencoders
Find Interpretable Manifolds* (arXiv:2605.08891), motivate measuring seed
stability at the global recovered-subspace level rather than by individual
unit identity alone.

Hindupur, Lubana, Fel, and Ba, *Projecting Assumptions*, argue that sparse
architecture determines the concept geometries a model can expose. It is
cited from a secondary record here and remains an owed full read once a
stable primary identifier is located.

## Why BSC and the consumer artifact line up

The discovered object has the same structure as saklas's manifold artifact:

| BSC | consumer concept |
|---|---|
| one shared block code | presence plus position coordinates |
| per-site decoder frame | per-layer subspace embedding |
| contribution-energy profile | per-layer share |
| active-code density | manifold support and thickness estimator |
| code topology | ring, line, map, or packed structure |

This is only a structural correspondence. The post-publication bridge must
translate the training whitener into the consumer whitener, retain the full
per-site coordinate map, and separately validate origins, density modes,
labels, thickness, and causal behavior. A successful file import would not
by itself establish naturalness.

## Source ledger

Primary reference metadata is machine-readable in
[`references/refs.yaml`](../references/refs.yaml). Full-text local copies are
gitignored and can be refreshed with the workspace reference fetcher.

1. Fel et al., *Structuring Sparsity: Block-Sparse Featurizers Capture
   Visual Concept Manifolds*, arXiv:2606.25234.
2. Dalili and Mahdavi, *Subspace-Aware Sparse Autoencoders*,
   arXiv:2606.06333.
3. Anthropic, *Sparse Crosscoders for Cross-Layer Features and Model
   Diffing*, 2024.
4. Minder et al., *Overcoming Sparsity Artifacts in Crosscoders to Interpret
   Chat-Tuning*, arXiv:2504.02922.
5. Jiralerspong and Bricken, *Cross-Architecture Model Diffing with
   Crosscoders*, arXiv:2602.11729.
6. Engels et al., *Not All Language Model Features Are One-Dimensionally
   Linear*, arXiv:2405.14860.
7. Michaud et al., *SAE Scaling in the Presence of Feature Manifolds*,
   arXiv:2509.02565.
8. Gorton, *Group Crosscoders for Mechanistic Analysis of Symmetry*,
   arXiv:2410.24184.
9. Laptev et al., *Analyze Feature Flow to Enhance Interpretation and
   Steering*, arXiv:2502.03032.
10. Ge et al., *Evolution of Concepts in Language Model Pre-Training*,
    arXiv:2509.17196.
11. *Group-SAE*, arXiv:2410.21508.
12. Bhalla et al., *Do Sparse Autoencoders Capture Concept Manifolds?*,
    arXiv:2604.28119.
13. Dooms et al., *Bilinear Autoencoders Find Interpretable Manifolds*,
    arXiv:2605.08891.

## Owed work before an external novelty claim

Search the older multi-view/coupled/joint/group-sparse dictionary-learning
literature and read the predecessor trail surfaced by Group-SAE, including
Yun (arXiv:2103.15949), Lawson (arXiv:2409.04185), SMixAE,
Hindupur/SPADE, Mishra-Sharma, Shafran et al., and any public
Baskaran–Sklar crosscoder work. External prose should continue to use the
narrow formulation until that sweep is complete.

The verbatim 2026-07 review record behind this condensation remains available
at Git commit `ed5816e12d20589727e1a0cc4ec7e80e36d6ea2e`.
