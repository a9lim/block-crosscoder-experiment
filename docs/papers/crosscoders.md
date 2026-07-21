# Same-model cross-layer crosscoder procedures

*Descriptive primary-source ledger, 2026-07-20. The experiment uses one model
at several layers; source mechanisms transferred from another task are labeled
`adapted`.*

## 1. What is shared

For aligned activations $x^s\in\mathbb R^{d_s}$, a scalar crosscoder infers
one code used to reconstruct every selected site. “Shared” can mean several
different things:

- **shared code:** the same latent activation contributes to several outputs;
- **joint evidence:** several sites contribute to the inferred activation;
- **operationally shared:** a code inferred from one/site subset reconstructs
  held-out sites;
- **shared coordinate:** the same multidimensional coordinate is meaningful at
  several sites;
- **causally coherent:** interventions through the code have the expected
  downstream effect.

Only the first two follow from the standard architecture. Decoder norms or
visual similarity do not establish the remaining claims.

## 2. Anthropic's original crosscoder

### 2.1 Architecture and loss

[Lindsey et al.](https://transformer-circuits.pub/2024/crosscoders/index.html)
compute one nonnegative scalar code

\[
f(x)=\operatorname{ReLU}\left(
\sum_{s=1}^{S}W^s_{\rm enc}x^s+b_{\rm enc}
\right)
\]

and reconstruct every site with

\[
\hat x^s=W^s_{\rm dec}f(x)+b^s_{\rm dec}.
\]

The displayed objective is squared reconstruction plus activation weighted by
the sum of sitewise decoder norms,

\[
L=\sum_s\lVert x^s-\hat x^s\rVert_2^2
+\lambda\sum_i f_i(x)\sum_s\lVert d_i^s\rVert_2 .
\]

The inner sum matters. Replacing it with the norm of the concatenated decoder
changes the regularizer and its incentive for distributing a feature across
sites.

### 2.2 Cross-layer procedure

The report trains a global acausal crosscoder over every residual-stream layer
of an 18-layer model, normalizing layers separately so each contributes
comparably. It sweeps width and training compute against independent per-layer
SAEs and studies several topology restrictions:

- **acausal:** every selected layer may contribute to and be reconstructed from
  the one code;
- **local:** the code spans a layer window;
- **weakly causal:** source/target ranges are bounded by feature group;
- **strictly causal:** inference reads one layer and reconstruction writes only
  downstream;
- **target variants:** residual stream and component outputs.

The exact model, corpus, token count, normalization formula, optimizer, batch,
learning rate, sparsity coefficient, initialization, dead-feature treatment,
and numeric sweep are not sufficiently disclosed. The executable
`anthropic_crosscoder_architecture_bridge` is therefore exact for the displayed
architecture/loss and adapted for its runnable dimensions and optimization.

### 2.3 Role in this project

The bridge retains dense ReLU selection during training. A calibration-only
threshold later creates an actual deployable packet so rate/distortion can be
measured. That operational diagnostic does not retroactively make the training
objective TopK or BatchTopK. The root is excluded from the main BSC-finalist
allowlist; instead, it enters only its own independently calibrated comparator
family and may occupy the frozen Anthropic comparator slot. Its value is to
answer whether the original same-model cross-layer architecture behaves as
expected under the common data/codec contract.

## 3. BatchTopK selector lineage

[Bussmann et al.](https://arxiv.org/abs/2412.06410) replace exactly $k$
features per example with exactly $Bk$ events across a batch of $B$
examples. For candidate score $v_{ti}$,

\[
\operatorname{BatchTopK}_k(v)_{ti}=
\begin{cases}
v_{ti},&(t,i)\text{ is among the largest }Bk\text{ candidates},\\
0,&\text{otherwise.}
\end{cases}
\]

Individual examples may use different support sizes while the batch mean is
fixed. Deployment cannot depend on another example in the batch, so a scalar
threshold must be fit on an independent calibration stream. A complete recipe
must still specify candidate scaling, nonlinearity, tie behavior, decoder
gauge, target support, threshold estimator, deadness unit, and Aux.

For vector blocks, replacing scalar candidates with block norms is a novel
adaptation. It is tested against token block-TopK under the same parent and at
the same achieved packet rate.

## 4. Decoder-weighted BatchTopK adaptation

[Minder et al.](https://arxiv.org/abs/2504.02922) diagnose two ways an L1
crosscoder can make decoder-norm interpretations unreliable: shrinkage can
drive one contribution toward zero, and correlated features can decouple one
concept into apparently distinct latents. Their BatchTopK carrier scores a
positive latent activation by its total decoded magnitude.

For $f_{ti}\geq0$, this project retains the disclosed score

\[
v_{ti}=f_{ti}\sum_s\lVert d_i^s\rVert_2,
\]

selects the largest batch-global event budget using $v$, and decodes the
corresponding **unscaled** $f$. The sum of sitewise norms is preserved; using
the concatenated norm would be another recipe. The live bridge uses summed
affine site encoders, ReLU, free affine site decoders, squared reconstruction,
Adam at `1e-4`, batch 2,048, 1,000-step warmup then constant LR, clip norm 1,
and a calibration target-rate threshold. Every transfer is labeled adapted.

The scientific claim is intentionally narrow: decoder-weighted BatchTopK is a
strong scalar allocation control for same-model layers. It does not establish
layer specificity, common multidimensional coordinates, or factor recovery.

The common evaluator therefore reports pre- and post-selection functional
dependence without assigning a preferred direction: omit each site in turn,
measure each block's RMS code change, max-normalize its site profile, and sum
that profile to a descriptive coherence `C`. A local feature and a broadly
cross-layer feature may both be scientifically valid.

## 5. Token-horizon residual auxiliary

The retained residual-Aux mechanism is a complete bundle:

1. count accepted token presentations since each latent last fired;
2. mark candidates dead only after the declared token horizon;
3. form the detached main residual;
4. score the eligible dead activations by the same total decoder norm;
5. select the declared Aux width;
6. decode unscaled auxiliary activations;
7. apply the declared residual normalization and coefficient.

The implemented pilot bundle uses a 10M-token horizon, 384 auxiliary latents,
coefficient `1/32`, and squared residual loss normalized by residual variance.
Those values are source-grounded adaptations, but the bundle is not a live
Phase-2 matrix arm: the live decoder-weighted anchor isolates selection, while
`auxiliary_16m` isolates BSF and SASA auxiliary families. Death is not measured
in steps or batches, and a future Minder-Aux round would have to retain the
complete bundle rather than mixing it with another method's frequency criterion
or runner-up selector.

## 6. Relevant cross-layer frontier

### fmxcoders

[Functionally coherent and scalable cross-layer dictionaries](https://arxiv.org/abs/2605.09438)
start from the same summed affine layer encoders, shared ReLU-plus-sparsifier,
and affine layer decoders as a standard scalar crosscoder (paper pp. 4--5).
They then factorize the **entire three-mode** encoder tensor and, independently,
the entire decoder tensor across activation, latent-feature, and layer modes.
The main Tensor Ring parameterization uses three ranks `(R1,R2,R3)` and a
trace of three matrix slices; the CP appendix uses one rank and a common basis
of layer patterns (pp. 5 and 15). Neither construction is merely a low-rank
matrix on the layer axis.

The other intervention is stochastic layer masking (p. 5): for every token and
layer an independent Bernoulli draw zeros that layer's **encoder input**, while
the loss reconstructs the clean activation at every layer. The paper varies
`p={0,.02,.05,.10}`. Its ablation finds that factorization rank primarily moves
reconstruction and probing quality, whereas masking primarily moves functional
coherence; treating the two as one indivisible mechanism would therefore hide
the reported separation (p. 16).

The disclosed source recipe is unusually complete (pp. 13--14): eight
post-MLP residual-stream sites in the middle of each model, 16,384 scalar
latents, ReLU then BatchTopK with `K=64`, batch 4,096, Adam at `3e-4` with
betas `(0.9,0.999)`, no warmup or decay, clip norm 1, no decoder
normalization, context 128, and 500M training tokens (300M for GPT2-Small).
The default mask probability is `.05`; rank-matched standard crosscoders and
unmasked factorizations are the controls. Reconstruction uses a held-out
roughly 410k-token panel. Functional coherence uses a separate 10M-token panel,
zeros one layer at a time, and measures the relative change in each latent with
`epsilon=1e-8` (pp. 6 and 14).

This project transfers two mechanisms while deliberately changing their
parameterization and latent ontology:

- The factorization round preserves each signed block coordinate and applies a
  transparent Tucker-style low-rank basis **only to the site axis**. It compares
  ranks `R={1,2,4}` with an unfactorized free-weight carrier and the exact
  selected parent. This is an adapted hypothesis, not a TR/CP reproduction.
- The masking round first selects a factorization parent, then compares the
  paper-exact independent Bernoulli probabilities with two declared novel
  fixed-cardinality controls: exactly one hidden site and exactly one retained
  site. All variants reconstruct clean targets and report all-site and every
  site-only-to-all-site endpoint.

The executable stages are `site_factorization_identification` and
`site_masking_identification` in Phase 1, then `site_factorization_4m` and
`site_masking_4m` in Phase 2. The distinction between source-exact Bernoulli
masking, adapted site-only factorization, and novel fixed-cardinality masks is
recorded in every emitted cell rationale.

### Group Crosscoders

[Group Crosscoders](https://arxiv.org/abs/2410.24184) treat transformed versions
of an activation as sites and analyze learned features under a group action.
They motivate equivariance tests and transformation-indexed decoder structure.
Their sparse feature remains scalar; decoder slices over transformations are
not the signed within-block coordinates studied here.

### Universal and routed SAEs

[Universal Sparse Autoencoders](https://arxiv.org/abs/2502.03714) sample a
source representation, encode only that source, and decode its scalar TopK code
into all targets. This is the source-only topology control. A joint summed
encoder is not USAE.

[RouteSAE](https://arxiv.org/abs/2503.08200) shares dictionary capacity while
routing examples/sites through a smaller active path. It is an efficiency and
conditional-computation alternative, not evidence that one joint code has
shared coordinates.

### Cross-site support and prealignment

[Multimodal group-sparse autoencoders](https://arxiv.org/abs/2601.20028) align
paired scalar supports across modalities. They justify testing common event
support separately from common coordinates.

[Procrustes-conditioned joint SAEs](https://arxiv.org/abs/2607.08499) align
spaces through a dense orthogonal map before joint sparse coding. A same-model
layer adaptation could separate easy linear rotation from nonlinear feature
change, but the prealignment map must be trained on a disjoint split, serialized,
and priced. Otherwise it moves unexplained capacity outside the sparse codec.

### Interaction structure

[Interactions Between Crosscoder Features](https://arxiv.org/abs/2606.09940)
measures feature interactions through a downstream MLP, clusters interacting
features, and introduces an interaction penalty. It addresses computational
coherence after a dictionary exists. The live campaign first establishes
factor identification and operational rate; interaction regularization is not
a universal architecture axis.

## 7. Explicit exclusion of different-model diffing methods

Designated Shared Features, Dedicated Feature Crosscoders, Delta-Crosscoder,
checkpoint-evolution methods, common-anchor comparison, and dense-versus-MoE
crosscoders are designed to separate shared from exclusive behavior across two
different models, fine-tunes, architectures, or snapshots. Their DSF/DFC
partitions, difference losses, base/chat data construction, tokenizer
alignment, and specificity gates answer a different question. They remain
useful negative context—especially the warning that decoder norms are not
causal specificity—but they provide no executable anchor, matrix row, decoder
gauge, or promotion criterion for this same-model cross-layer experiment.

## 8. Compatibility conclusions

- Sum and mean fusion are distinct because they change scale with site count.
- Dense L1, token TopK, BatchTopK, and calibrated threshold are distinct
  selectors, not interchangeable train/deploy labels.
- Activation-weighted sum-of-site-norms and concatenated decoder norm are
  distinct regularizers/scores.
- Decoder-weighted scoring chooses support; the decoder still receives the
  unscaled activation.
- Source-only encoding is a topology control, not weak joint evidence.
- Common scalar support does not imply a common vector coordinate system.
- Site masking and factorized layer weights are live derived-candidate rounds;
  prealignment, routing, and interaction penalties remain proposals until their
  full data/codec/metric contracts are implemented.
- For a free decoder, isolated decoded-energy scoring
  `sqrt(z_g^T (sum_s D_g^s D_g^{sT}) z_g)` is the priority score ablation: it
  removes reciprocal encoder/decoder gauge sensitivity and becomes ordinary
  code norm under an exact concatenated-Stiefel gauge.

The live BSC comparison therefore retains one exact same-model scalar
architecture anchor, one adapted decoder-weighted scalar carrier, source-only
controls, and the staged signed block synthesis. That is enough to test the
central hypothesis without importing a different scientific task.
