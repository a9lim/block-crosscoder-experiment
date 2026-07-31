# Block-crosscoder experimental design

*Scientific and execution contract, streamlined 2026-07-30.*

## 1. Question

For aligned activations \(x=(x^1,\ldots,x^S)\), with
\(x^s\in\mathbb R^{d_s}\), the block-sparse crosscoder computes

\[
u_g=\sum_s E_g^s x^s+a_g,\qquad
z_g=m_g u_g,\qquad
\hat x^s=c^s+\sum_g D_g^{s\top}z_g .
\]

A selected block \(g\) carries both a support event and a signed coordinate
\(z_g\in\mathbb R^b\). The support and coordinate are common across sites;
each site has its own decoder.

The experiment asks:

1. Can this representation identify a planted shared vector factor?
2. On real activations, does it improve distortion at a fixed deployable rate?
3. Does one learned code describe coherent structure across layers?
4. Does the representation capture a factor once, rather than splitting,
   duplicating, diluting, or mixing it?

Aggregate FVU is not an identification metric. Decoder norm is not proof of
site specificity. Nominal width is capacity, not used dimension.

## 2. Scope and source bridges

The experiment is same-model and cross-layer. Its main source components are:

- BSF and SASA for signed vector blocks;
- Anthropic's original crosscoder for joint inference across layers;
- BatchTopK for batch-global sparse allocation;
- FMX for site-axis factorization and observation masking.

No source publishes the complete signed vector-block crosscoder used here. The
combined representation is a project hypothesis. Exact source bridges and
adaptations are listed in [`paper_comparison.md`](paper_comparison.md).

Every scientific choice is labeled:

| Label | Meaning |
|---|---|
| `exact` | disclosed by a source and used in the same role |
| `adapted` | a sourced mechanism moved to a new setting |
| `engineering` | execution machinery intended not to change the object |
| `novel` | a project hypothesis |

Adapted and novel choices name a rationale and an ablation. Paper prose,
released code, and local adaptations remain distinct recipes.

## 3. Representation families

The executable surface contains:

- **BSC:** shared sparse block support and shared signed block coordinates;
- **BSF Grassmannian:** tied encoder, Stiefel decoder blocks, block TopK;
- **BSF Group Lasso:** affine encoder, group thresholding, group penalty;
- **SASA:** signed block code, Top-s, map-nuclear penalty, dead-group auxiliary;
- **Anthropic dense L1:** joint affine encoder, ReLU scalar code, free site
  decoders;
- **decoder-weighted BatchTopK:** scalar joint code with batch-global allocation;
- **scalar ReLU BatchTopK:** coefficient-matched scalar baseline;
- **source-only controls:** infer from one site while reconstructing all sites.

Hard selectors use deterministic cutoff ties: descending score, then lowest
declared event index. Threshold selectors use strict greater-than.

## 4. Phase 1 — truth-known identification

### Purpose

Phase 1 tests whether the implementation can recover the object it claims to
represent. It does not tune transferable numerical hyperparameters.

The carrier plants one rank-two factor in two coordinates, renders the same
coordinates through independent orthogonal site dictionaries, and fits one
width-two learner block with one active block. This removes width, capacity,
activity, and arbitrary allocation confounds.

The default blueprint has 15 cells at seeds 0, 1, and 2:

- three one-site positive instruments;
- three four-site carriers;
- nine confirmation cells covering the carrier and two negative controls.

The controls are:

- **support-only:** sites share occurrence but not vector coordinates;
- **one-site-span:** the apparent shared factor is generated from one site's
  span rather than a common cross-site coordinate.

Structure, training, factor-calibration, codec-calibration, development, and
confirmation streams use separate seeds or row ranges.

### Identification endpoint

Each planted factor is matched to a learner block on the factor-calibration
split. The frozen match is evaluated on development or confirmation.

The conjunction reports:

- support precision and recall;
- matched subspace overlap;
- aligned coordinate (R^2);
- isolated-input (R^2);
- recovered-factor fraction;
- duplicate, split, merge, dilution, and cross-factor mixing diagnostics.

Selection uses the worst normalized identification margin across native and
deployed selectors. Reconstruction FVU is reported only as a guardrail.

Phase 2 opens only after the complete Phase-1 panel has been reviewed and its
decision JSON says `go: true`.

## 5. Phase 2 — GPT-2 Small development

### Data

Phase 2 uses GPT-2 Small residual-pre activations at blocks 3, 5, 7, and 9,
with context length 128 and BOS removed. Whole sequences are assigned to one
role before token rows are packed.

| Role | Rows | Use |
|---|---:|---|
| normalization fit | 250,000 | activation transforms and encoder scale |
| calibration | 250,000 | threshold, rotations, clipping, quantizers |
| development | 1,000,000 | model and optimizer selection |
| confirmation | 1,000,000 | untouched pilot confirmation |
| train | 16,000,000 | prefix-nested optimization data |

Every derived normalization view preserves the same ordered
`(sequence, position, token_id)` rows. The primary gauge is centered scalar
RMS. `none`, `sqrt_d`, whitening, and token LayerNorm are confirmation
arms. A transform that needs source-only information for inversion is
nondeployable unless that information is serialized and priced.

### Conditional tuning

The main chain changes one or a few factors at a time:

| Round | Main question |
|---|---|
| anchors | establish BSC and comparator starting points |
| architecture | free, tied, QR, and polar geometry |
| capacity | block width, total coordinates, active coordinates |
| site factorization | full site weights versus ranks 1, 2, and 4 |
| site masking | no mask, Bernoulli masks, one hidden, one retained |
| rank revisit | revisit site rank after the selected mask |
| score × selector | code norm, decoded energy, isolated loss × token/BatchTopK |
| group threshold | complete learned-threshold method bundles |
| optimizer | learning rate, batch size, warmup, schedule, LR revisit |
| regularization | none, map nuclear, decoder nuclear |
| auxiliary | none and declared BSF/SASA residual auxiliaries |
| confirmation | frozen finalist across normalization arms |

Comparator families receive their own bounded development rounds rather than
borrowing BSC hyperparameters. A source-only model remains descriptive and
cannot promote.

Development and confirmation are disjoint. No phase-local choice is changed
after confirmation is read.

## 6. Phase 3 — frozen final panel

Phase 3 uses a reviewed Phase-2 finalist and six comparators:

1. selected BSC finalist;
2. BSF Grassmannian;
3. BSF Group Lasso;
4. SASA;
5. Anthropic dense L1;
6. decoder-weighted BatchTopK;
7. scalar ReLU BatchTopK.

Each design runs at seeds 0–4 on Gemma. A short production-shape stability
stage precedes final evaluation but does not rank or retune methods.

## 7. Rate-distortion contract

The deployable endpoint includes:

- sparse event indices;
- quantized signed coordinates;
- fixed packet fields;
- amortized serialized codec bytes;
- any required inverse-transform side information.

The measured quantizer frontier includes zero and the declared nonzero bit
depths. Selection is evaluated at exactly 256, 384, and 512 total bits/token.

If a target rate lies between measured endpoints, the lower convex envelope
defines a time-sharing mixture. The evaluator executes that mixture on paired
raw rows. It never extrapolates beyond the measured frontier.

The primary Phase-2/3 score is mean raw-space FVU across the three budgets.
Candidates must have all declared seeds and are ordered by:

1. median score;
2. worst-seed score;
3. readable candidate name.

Any minimum-effect or noninferiority threshold is a declared project policy,
not a paper value. Its sensitivity surface is reported.

## 8. Shared-code diagnostics

Every BSC and comparator cell reports:

- all-view raw-space FVU;
- each site-only input view;
- each leave-one-out input view;
- coordinate concordance;
- support intersection/IoU;
- intersection recall and energy coverage;
- functional dependence of each reconstructed site on observed sites.

These endpoints describe missing-view behavior. They do not add a promotion
hurdle beyond the common fixed-rate performance standard.

## 9. Training, calibration, and resume

Every cell resolves its model, optimizer, selector, regularizer, precision,
data roles, seed, and resource budget before training.

The ordinary lifecycle is:

```text
planned → prepared → running → trained → calibrated → evaluated → qualified
```

Training writes periodic progress and a final checkpoint. `--resume`
continues from the latest completed stage or training progress. Calibration
does not mutate the checkpoint; it writes the inference threshold and codec.
Evaluation reloads the trained model and codec, executes source-free decoding,
and writes one evaluation JSON.

The campaign stores one current `plan.json`, one `state.json` per cell, and
ordinary output files. The operator may inspect or edit these files. Git
provides project history; the runtime does not maintain a parallel provenance
ledger.

## 10. Qualification and promotion

Qualification means a cell produced a complete, finite, interpretable evidence
record. It checks the causal surface:

- the requested training completed;
- required data roles are present and disjoint;
- the codec round trip is finite and shape-correct;
- phase-specific scientific endpoints exist;
- resource limits were respected.

Qualification does not mean the hypothesis passed. The qualification JSON
separately records:

- `scientific_outcome`;
- `promotion_eligible`;
- `promotion_ineligible_reasons`;
- `selection_metrics`.

Smoke cells can exercise the workflow but never become scientific promotion
evidence. Positive and negative outcomes are both retained when they are
complete.

## 11. Runtime and resource rules

- Use shared plain Python 3.12.
- Phase 1 uses fp32.
- Real-model capture uses its declared bf16 forward precision; fp16 stores are
  unsupported.
- Only declared Adam or AdamW recipes are scientific cells.
- Training and capture run on the CUDA workstation.
- Do not keep evaluation and training models resident together on the 24 GB
  GPU.
- Storage, token, parameter, VRAM, host-RAM, and compute ceilings are refusal
  gates. Recheck live storage before a large capture or run.

## 12. Stop rules

- Stop or relabel a source branch if its bridge is not faithful.
- Stop a design delta if its named benefit fails across the complete seed set
  or crosses a scientific guardrail.
- Narrow a claim when a robustness stress fails; do not average it away.
- Do not tune on confirmation or final evidence.
- Do not open Phase 2 without a reviewed Phase-1 decision.
- Do not open Phase 3 without a reviewed Phase-2 panel.
- Do not launch when data roles or resource bounds are unresolved.
