# Findings

This is the scientific report for the BSC campaign. It is organized around
the questions asked, the methods compared, the observed data, and the design
choice that follows. Content IDs and hashes are provenance, not findings; they
are kept in the campaign artifacts and summarized only in the
footnotes.[^provenance][^artifacts]

The evidence cutoff is **2026-07-25 01:29 PDT**. Phase 1 and the Phase-2 BSC
main chain through confirmation are complete. Comparator-family calibration
is still running, so the comparator results below are within-family
development results, not a final cross-family ranking.

For Phase 2, every FVU value is the raw per-seed score averaged over the exact
256, 384, and 512 total-bit/token budgets; lower is better. Tables show seeds
0 and 1 separately, rounded to six decimals.[^metric]

## Bottom line

Phase 1 showed that the BSC method can recover the planted shared vector
factor rather than merely reconstructing its support. On real GPT-2
activations, the BSC development score improved from `0.401556957` at the
4M-token starting carrier to `0.250652800` at the selected 16M configuration,
a 37.6% relative reduction in FVU. Untouched confirmation reproduced the
result at `0.246043671`.

The current BSC configuration is:

| Surface | Adopted value | Why |
|---|---|---|
| model and sites | GPT-2 Small residual-pre blocks 3/5/7/9 | frozen real-model pilot contract |
| normalization | scalar RMS | best deployable confirmation score, though nearly tied with `sqrt_d` |
| encoder | joint untied linear, no bias, availability-rescaled site sum | tied Grassmann variants failed; removing initialization preconditioning was too small an improvement |
| decoder | free-scale-controlled, no bias, concatenated-L2 block geometry | selected parent remained strongest valid architecture |
| site factorization | rank 4 | noninferior to the full carrier and preferred by parsimony |
| code | signed, 2,048 groups × width 4, 8 active blocks | width 4 won; 32 active coordinates remains the only fully qualified BSC activity setting so far |
| score and selector | decoded energy + block BatchTopK | clear improvement over token-TopK and other score functions |
| site masking | none | every positive masking treatment was worse |
| optimizer | fused Adam, LR `3e-4`, batch 2,048 | strongest learning rate and batch-size results |
| schedule | 2% accepted-update warmup, then constant | final-fifth decay was numerically better but missed the preregistered per-seed effect floor |
| regularizer | none | every nuclear and learned-threshold treatment was worse |
| auxiliary | SASA-style frequency-dead residual, weight 1, AuxK 8, `1e-4` dead frequency, 1,000-token window | improved both seeds by more than 0.02 FVU |
| train budget | 16M unique rows and optimizer-token presentations | selected Phase-2 budget |

This is the current BSC development finalist, not yet the Phase-3 publication
winner. The independent comparator branches must finish their own calibration
and fresh 16M revisits first.

## Phase 1: does BSC identify a shared vector factor?

**Question.** Can one width-two BSC block recover a planted rank-two vector
factor shared across four independently rotated sites, rather than succeeding
only by reconstructing support?

**Baseline and controls.** The matched one-site instrument checked that the
learner could recover the easy local factor. The four-site carrier then tested
the actual shared-coordinate claim. `support_only` and `site_span_one` were
negative controls for support recovery without the full shared vector
geometry.

| Method | Seed 0 | Seed 1 | Seed 2 | Read |
|---|---:|---:|---:|---|
| one-site matched instrument | pass | pass | pass | local learner and metric are functional |
| four-site shared-coordinate carrier | 0.999600 | 1.000000 | 0.997600 | worst normalized identification margin; passed every seed |
| matched confirmation rerun | pass | pass | pass | native, deployed-codec, and conjunctive gates all passed |
| `support_only` control | fail | fail | fail | support alone did not masquerade as factor recovery |
| `site_span_one` control | conjunctive fail | conjunctive fail | conjunctive fail | recovered its narrow one-span target but not the full four-site factor |

**Interpretation.** The positive carrier and both negative controls behaved as
the truth-known contract predicted. This supports the BSC method semantics:
shared signed coordinates, all-site encoder fusion, calibrated thresholding,
and squared-L2 reconstruction.

**Adopted for Phase 2.** Carry forward the method semantics only. Phase 1 did
not export a model architecture, optimizer, width, activity, score,
regularizer, or rate winner; all of those were reopened on real data.

## Phase 2 setup

The real-model pilot captures four 768-dimensional GPT-2 Small residual
streams on OpenWebText, with disjoint normalization, codec-calibration,
development, confirmation, and training rows.[^setup]

The selection rule is:

1. compute raw-space FVU at 256, 384, and 512 total bits/token;
2. average the three fixed-rate values within each seed;
3. require both development seeds to qualify;
4. aggregate candidates by median, then worst seed;
5. apply any declared minimum-effect or noninferiority rule.

Site-only and leave-one-out endpoints are diagnostics shared by BSC and the
comparators. They do not gate promotion.

## BSC development

### Architecture and capacity

**Question.** Which BSC carrier should be tuned further?

**Baseline.** The starting carrier was the preconditioned joint-untied BSC at
width 4, 32 active coordinates, and the initial capacity.

**Architecture results.**

| Architecture | Seed 0 | Seed 1 | Outcome |
|---|---:|---:|---|
| preconditioned joint-untied parent | 0.400999 | 0.402115 | adopted |
| same model without initialization preconditioning | 0.400750 | 0.401769 | numerically better by only 0.00025/0.00035; below the 0.002 effect floor |
| tied Grassmann width 4 | failed | failed | no qualified comparison |
| tied Grassmann width 4 with polar retraction | failed | failed | no qualified comparison |

**Capacity results.**

| Capacity treatment | Seed 0 | Seed 1 | Outcome |
|---|---:|---:|---|
| parent: width 4, 32 active coordinates | 0.400999 | 0.402115 | adopted |
| half capacity | 0.410877 | 0.408604 | worse |
| width 8 | 0.438543 | 0.440345 | worse |
| scalar width 1 | 0.545655 | 0.544672 | worse |
| double capacity | qualified | failed | seed-incomplete |
| width 2 | failed | qualified | seed-incomplete |
| half activity | qualified | failed | seed-incomplete |
| double activity | qualified | failed | seed-incomplete |

A later clean family-width comparison again favored width 4 over widths 8 and
1. Width 2 again failed one seed.

**Interpretation.** The architecture alternatives did not justify replacing
the parent. Increasing width hurt reconstruction at a fixed total rate, while
several capacity/activity changes were numerically unstable enough to fail
the complete-seed contract.

**Adopted.** Keep the preconditioned joint-untied carrier at width 4 and treat
32 active coordinates as provisional pending a complete activity comparison.

### Site factorization and missing-site training

**Question.** How much site-axis rank is needed, and does masking sites during
training improve the joint code?

**Site-rank results.**

| Site factorization | Seed 0 | Seed 1 | Selection read |
|---|---:|---:|---|
| unfactorized carrier | 0.400999 | 0.402115 | numerical control |
| rank 1 | 0.995773 | 0.773756 | much worse |
| rank 2 | 0.667356 | 0.676648 | much worse |
| rank 4 | 0.405323 | 0.405399 | noninferior to carrier; simplest acceptable rank |
| full-rank factorization | 0.408113 | 0.409738 | slightly worse than rank 4 |

**Masking results.**

| Training view | Seed 0 | Seed 1 | Read |
|---|---:|---:|---|
| all sites, no masking | 0.405323 | 0.405399 | best |
| Bernoulli mask `p=0.02` | 0.405975 | 0.405622 | no improvement |
| Bernoulli mask `p=0.05` | 0.408117 | 0.407060 | worse |
| Bernoulli mask `p=0.10` | 0.412169 | 0.409951 | worse |
| exactly one hidden site | 0.418453 | 0.414067 | worse |
| exactly one retained site | 0.471264 | 0.467531 | worst |

**Interpretation.** Rank 4 gives almost all of the carrier's reconstruction
quality at much lower site-axis complexity. Training for partial views did
not improve the standard all-view crosscoder objective and degraded steadily
with stronger masking.

**Adopted.** Rank 4 with zero masking. Because zero masking won, the
factorization revisit correctly reproduced only the exact rank-4 parent.

### Score, selector, and learned thresholding

**Question.** Which hard-selection rule best allocates sparse blocks, and can
a learned Group-L21 threshold replace hard selection?

**Score × selector results.**

| Score | Selector | Seed 0 | Seed 1 | Read |
|---|---|---:|---:|---|
| decoded energy | block BatchTopK | 0.388467 | 0.387604 | best; adopted |
| isolated loss decrease | block BatchTopK | 0.399089 | 0.398394 | improvement over parent, but weaker |
| decoded energy | token-TopK | 0.405323 | 0.405399 | reproduced parent |
| code norm | block BatchTopK | 0.404275 | 0.403634 | effect too small |
| code norm | token-TopK | 0.406864 | 0.403443 | inconsistent and too small |
| isolated loss decrease | token-TopK | 0.407320 | 0.407620 | worse |

**Learned group-threshold results.**

| Method | Seed 0 | Seed 1 | Read |
|---|---:|---:|---|
| hard decoded-energy BatchTopK parent | 0.388467 | 0.387604 | retained |
| Group-L21 coefficient `3e-4` | 0.967436 | 0.967406 | failed badly |
| Group-L21 coefficient `1e-3` | 0.966761 | 0.966706 | failed badly |
| Group-L21 coefficient `3e-3` | 0.965483 | 0.965350 | failed badly |

**Interpretation.** The important gain comes from allocating blocks across
the batch using decoder-aware energy. Soft group thresholding did not merely
lose narrowly; it collapsed reconstruction across the entire coefficient
range.

**Adopted.** Decoded-energy block BatchTopK, with no learned group threshold.

### Optimizer and schedule

**Question.** Which optimizer scale and schedule convert the architectural
carrier into a strong reconstruction model?

| Trial | Option | Seed 0 | Seed 1 | Decision |
|---|---|---:|---:|---|
| learning rate | `1e-4` parent | 0.388467 | 0.387604 | baseline |
| learning rate | `3e-4` | 0.316867 | 0.316338 | adopt |
| learning rate | `3e-5` | 0.510105 | 0.511283 | reject |
| batch tokens | 4,096 parent | 0.316867 | 0.316338 | baseline |
| batch tokens | 2,048 | 0.304219 | 0.302914 | adopt |
| batch tokens | 8,192 | 0.340716 | 0.336502 | reject |
| warmup | 5% parent | 0.304219 | 0.302914 | baseline |
| warmup | 2% | 0.300071 | 0.299184 | adopt |
| warmup | 10% | 0.308331 | 0.307099 | reject |
| schedule | constant | 0.300071 | 0.299184 | adopt |
| schedule | final-fifth linear decay | 0.296505 | 0.297540 | numerically best, but one seed missed the 0.002 effect floor |
| schedule | cosine decay | 0.319408 | 0.320325 | reject |
| LR revisit | `3e-4` | 0.300071 | 0.299184 | retained |
| LR revisit | `1e-4` | 0.352483 | 0.351713 | reject |
| LR revisit | `3e-5` | 0.457663 | 0.456957 | reject |

**Interpretation.** Optimization, not architectural complexity, produced the
largest improvement in the main chain. The higher learning rate and smaller
BatchTopK comparison pool were robust in both seeds. Final-fifth decay is an
interesting near miss: it has the best aggregate number, but the declared
selection rule required a meaningful improvement in every seed.

**Adopted.** Adam at `3e-4`, batch 2,048, 2% accepted-update warmup, then a
constant rate.

### Regularization and dead-feature recovery at 16M tokens

**Question.** Does explicit rank regularization help at the full Phase-2
budget, and which dead-feature recovery mechanism should be added?

**Regularization results.**

| Treatment | Seed 0 | Seed 1 | Read |
|---|---:|---:|---|
| no regularizer | 0.273083 | 0.272379 | best valid treatment |
| decoder nuclear, coefficient 30 | 0.274271 | 0.274139 | slightly worse; diagnostic arm |
| decoder nuclear, coefficient 100 | 0.278903 | 0.279659 | worse; diagnostic arm |
| decoder nuclear, coefficient 300 | 0.288759 | 0.289405 | worse; diagnostic arm |
| end-to-end map nuclear, initial ratio 0.01 | 0.276286 | 0.276396 | worse |
| end-to-end map nuclear, initial ratio 0.03 | 0.284448 | 0.284162 | worse |
| end-to-end map nuclear, initial ratio 0.10 | 0.303541 | 0.303881 | worse |

**Auxiliary results.**

| Auxiliary | Seed 0 | Seed 1 | Read |
|---|---:|---:|---|
| SASA-source dead residual, weight 1, 1k window | 0.250241 | 0.251065 | best; adopt |
| SASA dead residual, weight 1, 100k window | 0.267686 | 0.266733 | positive but weaker |
| SASA dead residual, weight 1/32, 1k window | 0.269002 | 0.268238 | positive but weaker |
| no auxiliary | 0.273083 | 0.272379 | baseline |

**Interpretation.** Nuclear pressure consistently trades away useful
reconstruction and is not needed to obtain the selected block geometry.
Dead-residual recovery, by contrast, gives a large and repeatable gain.

**Adopted.** No nuclear regularizer. Add the SASA-source auxiliary with weight
1 and a 1,000-token deadness window.

### What the selected code does with partial views

The selected BSC reconstructs well when all four layers are available, but it
does not reconstruct the joint state from one local site.

| Diagnostic | Seed 0 | Seed 1 |
|---|---:|---:|
| all-site FVU | 0.219203 | 0.219721 |
| leave-one-out held-out-site FVU | 0.842556 | 0.855583 |
| leave-one-out coordinate concordance | 0.934882 | 0.933867 |
| leave-one-out support IoU | 0.177310 | 0.174657 |
| leave-one-out intersection energy coverage | 0.973343 | 0.972931 |
| site-only held-out-sites FVU | 47.576372 | 47.548965 |
| site-only coordinate concordance | 0.640705 | 0.637132 |
| site-only support IoU | 0.009317 | 0.009297 |
| site-only intersection energy coverage | 0.953563 | 0.953383 |

**Interpretation.** This is a joint acausal cross-layer code. A partial view
often preserves the direction and most intersection energy of the shared
coordinates, but not the exact active support or enough information to
reconstruct missing layers. That is a meaningful negative result, not a
failure of the standard crosscoder objective and not a special promotion gate
for BSC.

## Confirmation on untouched data

**Question.** Does the selected BSC reproduce on confirmation data, and which
deployable normalization should be frozen?

| Normalization | Seed 0 | Seed 1 | Mean | Outcome |
|---|---:|---:|---:|---|
| scalar RMS | 0.245329156 | 0.246758186 | **0.246043671** | adopt |
| `sqrt_d` | 0.245714309 | 0.246583468 | 0.246148888 | practical near-tie |
| none | 0.249536515 | 0.248693381 | 0.249114948 | valid, slightly worse |
| whitening | 0.263695954 | 0.264293307 | 0.263994631 | valid, worse |
| token LayerNorm | — | — | — | reconstruction checks passed, but raw decoding needs unpriced per-row oracle information |

The scalar-RMS fixed-rate frontiers were:

| Seed | 256 bits/token | 384 bits/token | 512 bits/token |
|---|---:|---:|---:|
| 0 | 0.247219379 | 0.244389791 | 0.244378297 |
| 1 | 0.248645500 | 0.245820369 | 0.245808690 |

**Interpretation.** Confirmation did not expose overfitting: scalar-RMS mean
FVU improved by about `0.00461` relative to the selected development cell.
Scalar RMS and `sqrt_d` differ by only `0.000105217`, so the scientific result
is normalization robustness rather than a large scalar-RMS advantage.

**Adopted.** Scalar RMS remains the declared parent and the numerically best
deployable normalization. Treat `sqrt_d` as effectively tied.

## Comparator-family development

These branches ask a different question from the BSC main chain: how strong
does each comparator become after receiving its own appropriate tuning?
Scores cannot yet be compared as final method results because the branches
are at different points in their calibration paths.

### 1M starting baselines

| Family | Seed 0 | Seed 1 | Mean |
|---|---:|---:|---:|
| BSC shared coordinates | 0.538981 | 0.537344 | 0.538163 |
| BSF Grassmannian | 0.813115 | 0.821469 | 0.817292 |
| BSF Group Lasso | 0.966073 | 0.966031 | 0.966052 |
| SASA | 0.550638 | 0.547401 | 0.549020 |
| Anthropic dense-L1 bridge | 0.938735 | 0.938963 | 0.938849 |
| decoder-weighted BatchTopK | 0.559838 | 0.560133 | 0.559986 |
| scalar ReLU BatchTopK | 0.769379 | 0.770003 | 0.769691 |

These are starting points only. The large changes below show why ranking
methods at the root would be misleading.

### Width calibration

| Family | Width | Seed 0 | Seed 1 | Within-family read |
|---|---:|---:|---:|---|
| BSC | 1 | 0.545655 | 0.544672 | worse |
| BSC | 2 | failed | qualified | seed-incomplete |
| BSC | 4 | 0.400999 | 0.402115 | adopt |
| BSC | 8 | 0.438543 | 0.440345 | worse |
| BSF Grassmannian | 2 | 0.565261 | 0.574003 | second |
| BSF Grassmannian | 4 | 0.554592 | 0.564405 | adopt |
| BSF Grassmannian | 8 | 0.647661 | 0.652169 | worse |
| BSF Group Lasso | 2 | 0.954693 | 0.954702 | adopt |
| BSF Group Lasso | 4 | 0.965994 | 0.965992 | worse |
| BSF Group Lasso | 8 | 0.974590 | 0.974491 | worse |
| SASA | 2 | 0.378765 | 0.381171 | adopt |
| SASA | 4 | 0.385355 | 0.387029 | second |
| SASA | 8 | 0.426618 | 0.427655 | worse |

**Interpretation.** Every block method preferred a relatively narrow block.
BSC and Grassmannian selected width 4; Group Lasso and SASA selected width 2.
The result argues against assuming that more coordinates per block buy better
fixed-rate reconstruction.

### Activity calibration

**Question.** At a fixed family architecture and width, how many active
coordinates produce the best fixed-rate reconstruction?

**Baseline and alternatives.** Each family reran its current activity setting
beside the declared lower- and higher-activity alternatives. The Grassmannian
round added here uses its selected width-4 parent at 32 active coordinates as
the baseline.

| Family | Active coordinates | Seed 0 | Seed 1 | Within-family read |
|---|---:|---:|---:|---|
| BSF Grassmannian | 16 | 0.636630 | 0.679368 | worst |
| BSF Grassmannian | 32 | 0.554592 | 0.564405 | baseline |
| BSF Grassmannian | 64 | 0.547015 | 0.550264 | adopt |
| decoder-weighted BatchTopK | 16 | 0.347461 | 0.348243 | adopt |
| decoder-weighted BatchTopK | 32 | 0.472189 | 0.473224 | worse |
| decoder-weighted BatchTopK | 64 | 0.716090 | 0.716882 | worst |
| scalar ReLU BatchTopK | 16 | 0.501047 | 0.502784 | adopt |
| scalar ReLU BatchTopK | 32 | 0.581384 | 0.581395 | worse |
| scalar ReLU BatchTopK | 64 | 0.770880 | 0.771770 | worst |
| BSC | 16 | qualified | failed | seed-incomplete |
| BSC | 32 | 0.400999 | 0.402115 | retained by eligibility |
| BSC | 64 | qualified | failed | seed-incomplete |
| BSF Group Lasso | 16 | 0.972832 | 0.972830 | worst |
| BSF Group Lasso | 32 | 0.954693 | 0.954702 | second |
| BSF Group Lasso | 64 | 0.946401 | 0.946282 | adopt |
| SASA | 16 | 0.446591 | 0.445930 | second |
| SASA | 32 | 0.378765 | 0.381171 | adopt |
| SASA | 64 | 0.608029 | 0.613059 | worst |
| Anthropic dense-L1 | 16 | 0.940578 | 0.940559 | worse |
| Anthropic dense-L1 | 32 | 0.925101 | 0.924799 | adopt |
| Anthropic dense-L1 | 64 | 0.937342 | 0.937187 | worse |

**Interpretation.** Activity is method-dependent. Grassmannian and Group Lasso
improve through 64 coordinates; the two BatchTopK scalar controls strongly
favor 16; SASA and dense-L1 favor 32. BSC does not yet have a comparative
activity result: 32 is the only seed-complete candidate. Grassmannian adopts
64, improving both seeds over its 32-coordinate baseline.

The BSC failures were narrow invariant failures, not missing jobs. Activity
16 seed 1 and activity 64 seed 1 exceeded the bound Stiefel Gram-residual
limit of `0.002` at `0.00213562` and `0.00200106`. The earlier width-2 seed 0
failed the same invariant at `0.00236131`. Therefore the current activity-32
choice must not be described as beating 16 and 64.

### Learning-rate calibration

**Question.** After each family selected its early structural settings, which
learning rate best converts those settings into reconstruction quality?

**Baseline and alternatives.** The current family parent used `1e-4`. Each
round compared `3e-5`, `1e-4`, `2e-4`, and `3e-4` at the same 4M-token budget.

| Family | Learning rate | Seed 0 | Seed 1 | Within-family read |
|---|---:|---:|---:|---|
| decoder-weighted BatchTopK | `3e-5` | 0.452081 | 0.453606 | worst |
| decoder-weighted BatchTopK | `1e-4` | 0.347461 | 0.348243 | baseline |
| decoder-weighted BatchTopK | `2e-4` | 0.318504 | 0.317965 | second |
| decoder-weighted BatchTopK | `3e-4` | 0.306748 | 0.306378 | adopt |
| scalar ReLU BatchTopK | `3e-5` | 0.708557 | 0.708556 | worst |
| scalar ReLU BatchTopK | `1e-4` | 0.501047 | 0.502784 | baseline |
| scalar ReLU BatchTopK | `2e-4` | 0.405054 | 0.406821 | second |
| scalar ReLU BatchTopK | `3e-4` | 0.366868 | 0.370329 | adopt |
| BSC shared coordinates | `3e-5` | 0.511555 | 0.509196 | worse |
| BSC shared coordinates | `1e-4` | 0.400999 | 0.402115 | retained by eligibility |
| BSC shared coordinates | `2e-4` | failed | qualified | seed-incomplete |
| BSC shared coordinates | `3e-4` | failed | failed | seed-incomplete |

**Interpretation.** Both scalar BatchTopK controls improve monotonically over
the tested range and adopt `3e-4`: decoder-weighted BatchTopK improves its mean
from about `0.34785` to `0.30656`, and scalar ReLU improves from about
`0.50192` to `0.36860`. The BSC family branch behaves differently. Its higher
rates crossed the bound Stiefel Gram-residual limit: `2e-4` seed 0 reached
`0.00203324`, while `3e-4` seeds 0/1 reached `0.00201709` and `0.002127`
against the `0.002` limit.

**Adopted.** The two scalar controls adopt `3e-4`. The BSC family branch
provisionally retains `1e-4` because it is the best seed-complete candidate;
this is not evidence that `1e-4` reconstructs better than the failed higher
rates. Its difference from the main-chain BSC's successful `3e-4` result is a
real path/configuration-sensitivity warning to revisit at 16M.

### Dense-L1 coefficient calibration

| Coefficient | Seed 0 | Seed 1 | Mean |
|---:|---:|---:|---:|
| `3e-6` | 0.925072 | 0.924810 | **0.924941** |
| `1e-5` | 0.925101 | 0.924799 | 0.924950 |
| `3e-5` | 0.925082 | 0.924833 | 0.924957 |

**Interpretation and adoption.** The panel is essentially flat: the complete
mean spread is about `1.6e-5` FVU. `3e-6` is the deterministic selected value,
not evidence for a practically meaningful coefficient effect.

### Current within-family choices

| Family | Choices supported so far |
|---|---|
| BSC shared coordinates | width 4; activity 32 provisional because flanks failed; family LR `1e-4` provisional because higher rates failed |
| BSF Grassmannian | width 4; 64 active coordinates |
| BSF Group Lasso | width 2; 64 active coordinates |
| SASA | width 2; 32 active coordinates |
| Anthropic dense-L1 | 32 active coordinates; coefficient `3e-6` by a negligible tie-break |
| decoder-weighted BatchTopK | 16 active coordinates; LR `3e-4` |
| scalar ReLU BatchTopK | 16 active coordinates; LR `3e-4` |

Group-Lasso, SASA, and dense-L1 follow-up rounds are still running. No
comparator is frozen and no cross-family winner is declared yet.

## Limitations and open reads

- Phase 2 has two seeds. It is appropriate for conditional tuning, not a
  five-seed publication claim.
- The main chain is path-dependent. Family revisits test local order
  sensitivity but cannot make the search exhaustive.
- Comparator roots use different architectures and source recipes. Only the
  completed fresh 16M revisits will support the final comparison.
- High all-site reconstruction does not prove single-site sufficiency, causal
  identifiability, semantic monosemanticity, or recovery of a global
  manifold.
- Decoder norm is not specificity, decoder capacity is not used dimension,
  and aggregate reconstruction is not manifold recovery.
- Feature-geometry plots are not yet findings. They will be generated from
  the frozen finalist and paired activation/code rows after Phase 2 reaches a
  terminal state.

[^metric]: Each cell evaluates the measured lower convex rate-distortion envelope at 256, 384, and 512 total bits/token, including fixed-width packet bits and amortized deployable-codec bytes. The schedule is executed on paired raw rows and never extrapolated. Development selection uses seeds 0 and 1, median then worst seed, and complete scientific qualification.

[^setup]: Capture uses `openai-community/gpt2` revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`, OpenWebText revision `b4325f019c648b1641a1784748667e8b74e5e064`, byte BPE, context length 128, and residual-pre hooks at blocks 3, 5, 7, and 9. Split sizes are 250k normalization-fit, 250k codec-calibration, 1M development, 1M confirmation, and 16M training rows, allocated by whole packed sequences. Forward and stored activation precision are bf16.

[^provenance]: The operational Phase-2 campaign is `/data/runs/bsc-phase2-d84627e`; the live plan at this cutoff is `study:05585e79c42292241786f9809400b669980ef75107766557f5c5e653ec4058e8`. It binds Phase-1 decision `phase1-decision:df789d6b27930bb813fcec1b9fde209e3a662d4adb9d42974850d7d05bf385c2`, transfer `phase1-transfer:a5a3dfbdaf9cc0fce9bdaacf063eaefbb53bd4e10402a80ba1d56f4b3e38f561`, Phase-2 blueprint `phase2-blueprint:f5b459552c7768341c329f43b2b7a26af9f0d9cb488fec0e263ea6c8af3ba0ae`, and common-gate amendment `phase2-gate-amendment:2801ad39b330155a1c4cf16130520b254870e389b29546c09a1449e76e71672c`.

[^artifacts]: Per-cell IDs, candidate IDs, content hashes, qualification digests, threshold-sensitivity grids, and exact environment manifests are intentionally not duplicated in the report body. The authoritative values remain in the campaign journal, plans, `selections/`, and each cell's qualification manifest. The campaign began at clean commit `d84627e`; later authenticated implementation amendments repaired orchestration and validation without changing completed scientific kernels. Authoritative execution was Python 3.12.13, PyTorch 2.8.0+cu128, CUDA 12.8, TransformerLens 3.5.1, Transformers 5.14.0, and Triton 3.4.0 on the RTX 4090 Jobe host.
