# Literature synthesis and scope map

*Primary-source map for the 2026-07-20 design. The committed bibliography is
`references/refs.yaml`; generated Markdown copies are local research inputs,
not committed evidence.*

## 1. The closest parents

### Signed vector blocks at one site

[Block-Sparse Featurizers](https://arxiv.org/abs/2606.25234) and
[SASA](https://arxiv.org/abs/2606.06333) are the direct parents of the BSC
latent ontology. Both select a whole signed vector by its block norm. BSF
varies the inference/decoder gauge: free affine plus Frobenius control,
decoder-tied Stiefel frames, and learned group shrinkage. SASA keeps free
encoder/decoder maps and penalizes the nuclear norm of each end-to-end block
map. These are substantively different hypotheses, not cosmetic
parameterizations.

BSF contributes the most useful truth-known generator and isolated-factor
recovery protocol. SASA contributes an explicit effective-map-dimension
regularizer and a whole-group dead-residual auxiliary. Neither paper trains a
multi-layer crosscoder. Their released implementations also differ from their
paper equations in important ways, so the experiment keeps paper and release
lineages separate.

### One scalar code across layers

[Anthropic's original sparse crosscoder](https://transformer-circuits.pub/2024/crosscoders/index.html)
is the direct parent of the multi-site topology. It sums affine site-encoder
contributions before one ReLU code and reconstructs each site with its own
decoder. Its L1 penalty weights activation by the sum of sitewise decoder
norms. The report establishes the architecture and several cross-layer
topologies, but does not disclose enough runtime detail for numerical
reproduction. It is therefore an exact architecture bridge with adapted
training values.

[BatchTopK](https://arxiv.org/abs/2412.06410) replaces a fixed per-token count
with a fixed batch-global event budget and calibrates a threshold for
single-example inference. [Minder et al.](https://arxiv.org/abs/2504.02922)
show why L1 shrinkage can corrupt decoder-norm interpretations and supply two
useful mechanisms: score crosscoder activations by summed decoder norm before
BatchTopK, and revive token-horizon-dead latents through residual
reconstruction. This project adapts those mechanics to layers of one model and
does not import the source task or its exclusivity claims.

## 2. Sparse-inference controls

- [TopK SAE scaling](https://arxiv.org/abs/2406.04093) supplies the fixed-count
  scalar baseline, dead-latent handling, and scale comparisons.
- [Gated SAEs](https://arxiv.org/abs/2404.16014) separate support selection from
  magnitude estimation, clarifying when shrinkage rather than dictionary
  geometry causes a failure.
- [SoftSAE](https://arxiv.org/abs/2605.06610) predicts an input-dependent scalar
  feature count with a dynamic-sparsity MLP, trains through differentiable Soft
  Top-K under an expected-count budget, and switches to hard TopK late in
  training and at inference. Its paper also exposes the two main adaptation
  hazards: tiny nonzero soft weights can hide information and the differentiable
  selector becomes expensive at large dictionaries. A signed-block version
  would be new and must constrain mean packet bits, not merely mean block count.
- [JumpReLU](https://arxiv.org/abs/2407.14435) supplies a direct L0-oriented
  threshold baseline.
- [Compute-optimal inference and the amortisation gap](https://arxiv.org/abs/2411.13117)
  proves that a one-pass linear/nonlinear SAE encoder can fail at sparse-code
  inference even when the learned dictionary makes the inverse problem
  solvable, and reports gains from more expressive iterative inference at
  modest extra compute. This motivates a decoder-frozen, truth-known Phase-1
  diagnostic: compare the amortized encoder with a short group-IHT/residual
  refinement path before attributing a miss to dictionary geometry. It does
  not justify placing an unrolled encoder in the live matrix without a measured
  amortization gap.
- [Projecting Assumptions / SpaDE](https://arxiv.org/abs/2503.01822) makes the
  geometry of sparse projection explicit: ReLU, TopK, and BatchTopK impose
  different receptive-field assumptions. Batch-global allocation solves
  heterogeneous count, not heterogeneous shape.
- [Cosine-scored SAEs](https://arxiv.org/abs/2606.15054) warn that raw norm can
  dominate support selection, but also find pure cosine scoring insufficient.
  Any block/multi-site extension would be a new mechanism requiring its own
  score and codec ablation.
- [Orthogonal SAEs](https://arxiv.org/abs/2509.22033) penalize high decoder-vector
  cosine similarities to reduce absorption and composition. Extending that
  scalar penalty to signed block subspaces requires a projector/principal-angle
  definition and a truth-known control where correlated factors are legitimate.
- [Matryoshka SAEs](https://arxiv.org/abs/2503.17547) train several nested
  dictionary prefixes to reconstruct independently. A block version could
  yield one broad-rate codebook, but prefix order breaks group-permutation
  symmetry and must be compared with independently trained dictionaries at
  every exact packet rate.

These papers justify strong scalar controls and selector diagnostics. They do
not establish that one selector dominates for signed vector blocks.

[Data Whitening Improves Sparse Autoencoder Learning](https://arxiv.org/abs/2511.13981)
reports improved sparse probing and disentanglement after PCA whitening despite
small reconstruction losses. This is evidence that an FVU-only normalization
choice can select the wrong representation for interpretability. The project
therefore keeps whitening in a frozen confirmation view, reports semantic
endpoints alongside raw-space distortion, and treats scalar RMS as the explicit
deployment gauge rather than claiming it is empirically optimal for every
endpoint.

[End-to-end learned compression](https://arxiv.org/abs/1611.01704) supplies the
general precedent for optimizing a differentiable rate-distortion surrogate
through quantization. It does not justify making packet-aware training part of
the identification search: a crosscoder adaptation belongs after structural
selection and must validate actual integer packets plus every entropy-model or
quantizer byte.

## 3. Manifold and local-geometry alternatives

[Not all language-model features are one-dimensionally linear](https://arxiv.org/abs/2405.14860)
and [SAE scaling in the presence of feature manifolds](https://arxiv.org/abs/2509.02565)
motivate a representation whose primitive can have intrinsic dimension above
one. The following methods test competing ontologies:

- [SMIXAE](https://arxiv.org/abs/2605.09224) uses a sparse mixture of nonlinear
  low-dimensional expert autoencoders. It is a non-additive manifold control.
- [Mixtures of factor analyzers](https://arxiv.org/abs/2602.02464) model local
  probabilistic low-rank regions rather than a globally additive factor sum.
- [Bilinear autoencoders](https://arxiv.org/abs/2605.08891) add quadratic
  interactions and can represent curved manifolds that a linear block cannot.
- [SpaDE](https://arxiv.org/abs/2503.01822) uses learned prototypes and a local
  sparse projection rather than global directions.
- [Dense low-rank scaffolds](https://arxiv.org/abs/2606.14040) place a dense
  low-rank path beside the sparse residual path, testing whether common dense
  variance is being forced into sparse events.

These are falsifiers for the BSC ontology, not switches to cross with every BSC
hyperparameter. They enter only after a common truth-known or operational
endpoint makes a meaningful comparison executable.

## 4. Evaluation literature

[SAEBench](https://arxiv.org/abs/2503.09532) evaluates more than 200 SAEs across
proxy, interpretability, disentanglement, and application metrics and finds that
proxy improvements do not reliably transfer to practical utility. The later
[benchmark reliability audit](https://arxiv.org/abs/2605.18229) further finds
material reseed noise and weak discriminability among variants of the same
architecture; two canonical benchmark metrics fail several reliability checks.
Together these results rule out selecting small design changes by a single
semantic score. Phase 2 therefore freezes semantic and behavioral endpoints as
non-selecting diagnostics with reseed uncertainty, while fixed-rate raw-space
distortion remains the common selection endpoint. Ordinary rounds retain their
parent unless a child clears a preregistered minimum effect; compressed
site-axis ranks use noninferiority plus parsimony.

[Do Sparse Autoencoders Capture Concept Manifolds?](https://arxiv.org/abs/2604.28119)
shows why reconstruction alone is insufficient. Capture, tiling, shattering,
dilution, receptive-field coverage, and isolated contribution distinguish one
coherent feature from many locally useful fragments. Phase 1 therefore binds
support, subspace, and coordinates to the same learned block and reports
split/merge pathologies. Its confirmation panel also separates shared
rank-one/rank-two site-map families from independent-map negative controls,
one/two/all-site factor spans, uniform from Zipf-alpha-one occurrence, and
independent from paired coactivation. One-site factors cannot support a
shared-feature claim. Independent maps remain eligible when support and
coordinates are shared across several sites; they test whether low-rank
site-axis factorization is necessary, not whether the cross-layer factor
exists.

[Feature Flow](https://arxiv.org/abs/2502.03032) motivates downstream
interpretation/steering endpoints for layerwise features. It is an analysis
endpoint after a code qualifies, not a training objective in the live matrix.

[Interactions Between Crosscoder Features](https://arxiv.org/abs/2606.09940)
adds an interaction metric, clustering procedure, and loss aimed at
computational sparsity. That question concerns interactions among qualified
features; it does not precede basic factor identification and is not a live
training axis.

## 5. Other multi-representation topologies

- [Universal Sparse Autoencoders](https://arxiv.org/abs/2502.03714) encode one
  source representation at a time and decode its scalar TopK code into several
  targets. This motivates the source-only control but is not joint evidence
  aggregation.
- [RouteSAE](https://arxiv.org/abs/2503.08200) routes examples through a shared
  dictionary and offers an efficiency alternative to a fully parameterized
  cross-layer encoder.
- [Group Crosscoders](https://arxiv.org/abs/2410.24184) treat transformed
  versions of an input as sites and test equivariance. Decoder slices indexed
  by a group action are not the same object as a vector-valued sparse block.
- [Group-SAE](https://arxiv.org/abs/2410.21508) groups layers for training
  efficiency; the name does not imply vector blocks.
- [Multimodal group-sparse autoencoders](https://arxiv.org/abs/2601.20028)
  encourage paired scalar supports across modalities. Shared event support
  does not establish shared within-block coordinates.
- The paper [*fmxcoders: Factorized Masked Crosscoders for Cross-Layer Feature Discovery*](https://arxiv.org/abs/2605.09438)
  factorizes layer-axis weights and uses stochastic layer masking to encourage
  one-layer-to-all-layer functional
  coherence. The live same-model adaptation first retains the exact selected
  parent, then tests Tucker-style site-axis ranks `1,2,4` against an
  unfactorized free four-layer carrier subject to frozen carrier
  noninferiority. Masking compares Bernoulli probabilities
  `0,.02,.05,.10` with exactly-one-hidden and exactly-one-retained draws after
  rank selection. It preserves the sparse
  block and coordinate axes, masks encoder evidence only, reconstructs clean
  targets at every site, and reports all-site plus every site-only-to-all-site
  endpoint. This transfers the mechanisms; it does not claim to reproduce the
  paper's tensor factorization.
- [Procrustes-conditioned joint SAEs](https://arxiv.org/abs/2607.08499)
  prealign representation spaces before learning a joint TopK code. In a
  same-model cross-layer setting this motivates a prealignment diagnostic, but
  it risks moving part of the scientific result into an unpriced dense map.

## 6. Scope boundary

Model-comparison methods that separate or track differences among distinct
models, fine-tunes, architectures, or snapshots answer a different question
from finding multidimensional factors across layers of one model. Their
partitions, difference losses, and alignment procedures therefore do not enter
the executable anchors, matrix axes, gates, gauges, or recommendations.

## 7. Synthesis

The reviewed literature supports the following decomposition:

1. BSF/SASA establish whether vector blocks recover multidimensional factors;
2. the original crosscoder establishes same-model joint layer encoding;
3. BatchTopK and decoder-weighted scoring test support allocation without L1
   shrinkage;
4. manifold papers supply truth-known failure modes and evaluation language;
5. source-only and masked-site mechanisms provide live topology/robustness
   tests, while routing and prealignment remain future candidates;
6. local/nonlinear and dense-scaffold methods remain ontology falsifiers.

The BSC is the controlled intersection of items 1 and 2. Every other mechanism
must enter as a named derived candidate with its own nearest-parent ablation,
not as an unbounded frontier-method Cartesian product.
