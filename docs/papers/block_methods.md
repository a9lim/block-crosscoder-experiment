# Block and manifold method procedures

*Descriptive primary-source ledger, 2026-07-20. Paper facts, inspected-release
facts, and project adaptations are kept separate.*

## 1. Why blocks are a different sparse object

For $G$ blocks of width $b$, the code is
$z=(z_1,\ldots,z_G)$ with $z_g\in\mathbb R^b$, and

\[
\operatorname{supp}_B(z)=\{g:\lVert z_g\rVert_2>0\}.
\]

Selecting one block retains a signed vector coordinate. It is not equivalent
to selecting $b$ independent ReLU latents, grouping scalar supports after
training, or partitioning a scalar decoder into several site slices. The
block's intended semantics are a low-dimensional subspace or manifold chart.

The two direct sources are:

| Source | Official record | Contribution |
|---|---|---|
| Fel et al., [*Structuring Sparsity: Block-Sparse Featurizers Capture Visual Concept Manifolds*](https://arxiv.org/abs/2606.25234) | [goodfire-ai/block-sparse-featurizer](https://github.com/goodfire-ai/block-sparse-featurizer) | signed block TopK; Frobenius, Stiefel, and group-shrinkage gauges; truth-known manifold generator |
| Dalili and Mahdavi, [*Subspace-Aware Sparse Autoencoders*](https://arxiv.org/abs/2606.06333) | [arshandalili/sasa](https://github.com/arshandalili/sasa) | signed Top-s blocks; end-to-end map nuclear norm; whole-group residual revival |

No reviewed source trains this vector-valued object jointly across layers. The
cross-layer extension is therefore a local synthesis.

## 2. Block-Sparse Featurizers

### 2.1 Shared selection rule

Let $u_g\in\mathbb R^b$ be the preactivation of block $g$. BSF's TopK
variants keep the $\ell$ largest block energies per example,

\[
z=\Pi_\ell(u),\qquad
\Pi_\ell(u)_g=
\begin{cases}
u_g,&g\text{ among the top }\ell\text{ by }\lVert u_g\rVert_2,\\
0,&\text{otherwise.}
\end{cases}
\]

The decoder reconstructs $\hat x=\sum_gD_g^\top z_g$. Within-block
coordinates are signed. The three paper variants differ in how they resolve
the encoder/decoder scaling gauge.

### 2.2 Vanilla BSF

The paper uses a free affine encoder,

\[
u=xW+a,
\]

and a free linear decoder with each block projected to
$\lVert D_g\rVert_F\leq1$. The primary objective is squared reconstruction.
The decoder is initialized first and the encoder from its transpose at a scale
intended to produce useful initial block magnitudes.

Project consequence: Frobenius-*ball* projection and exact unit-Frobenius
renormalization are different recipes. The former is the paper anchor.

### 2.3 Grassmannian BSF

Inference is tied to the decoder frame,

\[
u_g=\gamma D_gx,
\]

where $\gamma>0$ is one learned log-parameterized scalar. Each block lies on a
Stiefel manifold, $D_gD_g^\top=I_b$. The paper reports that QR retraction
roughly every 20 steps is sufficient and uses reconstruction loss without a
separate sparsity penalty.

This geometry identifies a subspace while allowing an arbitrary within-block
orthogonal basis. In a cross-layer extension, imposing Stiefel structure per
site versus on the concatenated decoder is a new decision: the two constraints
allocate block energy differently across sites.

### 2.4 Group-Lasso BSF

The group-shrinkage variant uses a free affine encoder followed by

\[
z_g=\left(1-\frac{\theta_g}{\lVert u_g\rVert_2}\right)_+u_g,
\]

with a learned positive threshold and the group penalty

\[
L=L_{\rm rec}+\lambda\sum_g\lVert z_g\rVert_2.
\]

The matched-sparsity procedure applies the group penalty while measured block
activity exceeds its target. Because encoder/decoder rescaling can otherwise
defeat the penalty, decoder scale control is part of the recipe. The paper does
not fully determine every executable scale-control choice; the pinned
unit-block-Frobenius implementation is an adapted bridge, not silently
paper-exact.

### 2.5 Synthetic manifold procedure

The controlled source experiment is the primary identification bridge:

- ambient dimension 128;
- 128 ground-truth additive factors;
- half one-dimensional atoms and half drawn from circle, disk, sphere, torus,
  Möbius-strip, Swiss-roll, and helix families;
- 50,000 calibration samples per factor, followed by centering/RMS scaling and
  random orthonormal embedding;
- exactly four distinct factors active per example;
- no observation noise in the primary toy;
- 300,000 training and 100,000 held-out examples with independent seeds;
- final scale $\mathbb E\lVert x\rVert_2^2=1$.

The paper matches each isolated ground-truth factor to the block with best
isolated-contribution R2. That is stronger than aggregate reconstruction, but
the project further requires support association, subspace overlap, and code
alignment to refer to the same learned block.

The paper describes Adam for a few hundred toy epochs but does not disclose one
complete optimizer manifest for every synthetic cell. The project's 300-pass,
batch-8,192, `1e-4` cosine bridge transfers nearby disclosed values and remains
`adapted`.

### 2.6 DINOv3 sweep

The real-activation grid uses final-layer DINOv3 ViT-B patch activations,
$d=768$, normalized to mean norm approximately $\sqrt d$. It crosses block
widths `{1,2,3,4,6,8,12,16,32}`, block counts
`{4096,8192,16384,32768}`, and active blocks `{8,16,32,64}`, pruning cells with
more than $1.6\times10^5$ latent coordinates or 400 active coordinates. The
reported optimizer is Adam, batch 8,192, 2,000 warmup steps, cosine
`1e-4 -> 1e-5`, and three epochs.

Its evaluation includes held-out reconstruction, MDL at target reconstruction,
several code-rank measures, semantic detection/probes, spatial smoothness,
probe recovery, a Fourier null, and steering. The MDL accounting prices
support, spectral code/residual rate, and amortized subspace-dictionary cost.
This motivates rate-aware BSC comparisons, but the current project uses an
actual serialized packet/codec rather than importing that analytic MDL as its
selection score.

### 2.7 Appendix runner-up auxiliary

The method descriptions give no Aux for Vanilla/Grassmannian and the group
penalty for Group Lasso. Appendix D separately describes:

1. compute the main residual $r=x-\hat x$;
2. select the next $\ell$ unselected blocks by encoder score;
3. reconstruct the detached residual from those blocks;
4. weight the auxiliary loss by $1/\ell$.

The live matrix therefore has primary and `appendix_aux` cells. Treating the
appendix term as an invisible default would confound source reproduction with
dead-feature handling.

### 2.8 Inspected-release drift

The official repository was inspected at commit
[`d183aa44c98bd5d1c67651870999506b426900da`](https://github.com/goodfire-ai/block-sparse-featurizer/tree/d183aa44c98bd5d1c67651870999506b426900da).

| Axis | Paper | Inspected release |
|---|---|---|
| target | clean reconstruction | starter defaults to denoising corruption with clean target |
| Grassmann retraction | periodic QR, about every 20 steps | QR in every forward |
| Vanilla constraint | Frobenius unit ball | exact unit-Frobenius normalization after each batch |
| Appendix Aux | runner-up residual term | absent |
| Group Lasso | smooth threshold and `L2,1` | starter defaults include a post-paper hard-gate/STE mode |
| threshold scope | paper describes per block | inspected paper mode uses one shared scalar |

The release is valuable implementation evidence but does not resolve paper
omissions. The live release-drift Group-Lasso recipe remains quarantined until
its complete optimizer, denoising, and projection runtime is representable.

## 3. Subspace-Aware Sparse Autoencoders

### 3.1 Architecture and exact structural term

SASA forms $u_g=E_gx$, keeps the `s` blocks with largest
$\lVert u_g\rVert_2$, and decodes through free $D_g$. Its defining loss is

\[
L=L_{\rm rec}+\lambda_{\rm dim}\sum_g\lVert D_gE_g\rVert_*.
\]

The nuclear norm is on the complete end-to-end map for a block. A decoder-only
nuclear norm changes the hypothesis: it measures decoder rank without the
encoder map and has a different scaling gauge.

The paper does not disclose the numerical `lambda_dim`. The live paper bridge
therefore calibrates a dimensionless initial regularizer/reconstruction ratio
after initialization on a content-bound training prefix, with targets
`0/.01/.03/.10` centered at `.03`; it does not infer an absolute coefficient
from the release. The inspected release's absolute `100` multiplies a
decoder-only nuclear norm and remains a quarantined, nonpromotable drift
diagnostic rather than evidence for the paper objective.

### 3.2 Whole-group dead-residual auxiliary

A group is dead when its firing frequency is at most `1e-4` over a 1,000-token
window. SASA then:

1. detaches the main residual;
2. re-encodes that residual through the encoders of dead groups;
3. selects the highest-energy auxiliary groups per token;
4. decodes those groups to reconstruct the residual;
5. weights this term by one.

Death, scoring, selection, and decoding all operate on whole groups. A scalar
dead-latent Aux over original preactivations is a different algorithm.

### 3.3 Language-model settings

| Setting | GPT-2 Small | Mistral-7B-v0.1 |
|---|---:|---:|
| site/data | residual-pre block 7 / OpenWebText | residual-pre block 8 / Pile |
| tokens/context | 150M / 128 | 500M / 512 |
| `(groups,width,active)` | `(2048,6,10)` | `(4096,8,10)` |
| scalar-equivalent width/L0 | `12288/60` | `32768/80` |
| auxiliary groups | 512 | 256 |

Both settings use token LayerNorm, token batch 4,096, AdamW with betas
`.9/.999`, LR `2e-4`, WD `1e-3`, 1,000 warmup steps, and linear decay over the
final fifth. Reported endpoints include KL, CE, FVE, L0, absorption, matched
scalar baselines, redundancy, probes, automated interpretation, and several
semantic subspace-geometry analyses.

### 3.4 Inspected-release drift

The official repository was inspected at commit
[`e2be58db95af1a6641bd807611b0881f28a13b69`](https://github.com/arshandalili/sasa/tree/e2be58db95af1a6641bd807611b0881f28a13b69).

| Axis | Paper | Inspected release |
|---|---|---|
| dimension term | `||D_g E_g||_*` | decoder-block nuclear norm |
| coefficient | undisclosed | default 100 |
| scale handling | not disclosed as per-step row/column normalization | encoder columns and decoder rows renormalized |
| death | whole group by firing frequency | scalar SAELens-style dead mask |
| Aux input | detached residual re-encoded by dead groups | original hidden preactivation |
| weight decay | AdamW `1e-3` | exposed argument not forwarded in inspected runner |
| bias | theory presents a linear encoder | release has encoder bias |

`sasa_paper` and `sasa_released_code_drift` are therefore different recipes.
Only the paper arm supplies the map-nuclear and whole-group-Aux hypotheses in
the main matrix.

## 4. Geometry comparators and falsifiers

The broader literature tests whether additive linear blocks are the right
ontology:

- [SpaDE](https://arxiv.org/abs/2503.01822) uses prototype distance and a local
  simplex projection, making the receptive-field assumption explicit.
- [SMIXAE](https://arxiv.org/abs/2605.09224) uses a sparse mixture of nonlinear
  low-dimensional experts.
- [Mixtures of factor analyzers](https://arxiv.org/abs/2602.02464) fit local
  probabilistic low-rank regions.
- [Bilinear autoencoders](https://arxiv.org/abs/2605.08891) add quadratic
  interactions.
- [Dense low-rank scaffolds](https://arxiv.org/abs/2606.14040) separate common
  dense variance from sparse residual events.

These are not equivalent BSC hyperparameters. A comparator enters only when it
can consume the same rows and produce the same truth-known or operational
endpoint at a declared compute/rate match.

## 5. Manifold evaluation contract

[Bhalla et al.](https://arxiv.org/abs/2604.28119) distinguish:

- **capture:** the learned representation explains the factor;
- **tiling:** several local features cover different regions of one factor;
- **shattering:** factor neighborhoods fragment in representation space;
- **dilution:** one factor's information spreads weakly across many features;
- **mixing:** one learned feature combines several factors.

The project's synthetic conjunction adds explicit same-block matching:
support association first chooses a block; subspace and aligned-code recovery
must come from that block. Split/merge/deadness guardrails prevent a low-FVU
dictionary from winning by duplication or mixture.

## 6. Consequence for the BSC matrix

The defensible synthesis is narrow:

1. reproduce or explicitly adapt the BSF/SASA block semantics;
2. test common support and common coordinates separately;
3. compare QR and polar cross-site Stiefel gauges;
4. compare an unfactorized free site axis with ranks `1/2/4`, then test
   clean-target site masking and decoded-energy support scoring;
5. compare token TopK and block BatchTopK at achieved rate;
6. test one structural penalty and one complete source-derived Aux bundle at a
   time;
7. challenge the selected recipe with one-delta geometry stresses plus
   truth-known site-map rank, factor-span, frequency, and coactivation axes;
8. defer nonlinear/local/dense alternatives until a common endpoint is
   executor-ready.

That sequence tests the block hypothesis without using the literature as a
license for an unbounded combination sweep.
