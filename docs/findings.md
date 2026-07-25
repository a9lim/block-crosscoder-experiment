# Findings

This file is the durable index of completed scientific evidence. It is empty
until a campaign cell reaches content-addressed qualification. Integrity-complete
negative results belong here just as positive results do.

Each entry must include:

- phase, blueprint, plan, stage, candidate, cell, and selection IDs;
- for Phase 1, authorization/transfer IDs, universal-method-contract and
  provisional-carrier hashes, fixed-carrier capability-panel outcomes,
  capability qualification digests, and any confirmation claim-scope narrowing;
- for Phase 2, the bound source Phase-1 decision/transfer IDs and the complete
  unpruned real-model option surface, with relevant synthetic capability
  warnings reported alongside rather than used as filters;
- source/paper/release/local lineage for every material decision;
- preregistered hypothesis, metric, thresholds, and guardrails;
- model, corpus, tokenizer, data-store, transform, code, and dependency hashes;
- unique rows, optimizer-token presentations, seeds, parameters, FLOPs, and
  storage, plus peak VRAM and peak host RAM under the bound estimator version;
- development, confirmation, or final status;
- complete seed distribution, median, worst seed, and matched controls;
- native and reloaded-codec endpoints;
- raw-space fixed-rate score and complete measured rate-distortion frontier for
  real cells;
- same-block support/subspace/code recovery and pathologies for synthetic cells;
- selected-score diagnostics for synthetic contract cells, including decoded
  energy, isolated-loss-decrease negative-gain fraction, hidden-target
  exclusion, and signed threshold calibration where applicable;
- planted site-map rank family, factor site span, frequency/coactivation law,
  and `shared_feature_claim_eligible` status for synthetic robustness cells;
- all-site, site-only-to-all-site, leave-one-site-out, and descriptive
  functional-dependence endpoints for cross-layer cells;
- qualification checks, scientific-outcome checks, promotion eligibility, and
  any ineligibility reasons;
- artifact-manifest and qualification digests;
- failures, retries, exclusions, limitations, and negative results.

Use one section per immutable decision artifact. Do not summarize an unfinished
run, mutable checkpoint, smoke profile, oracle raw inverse, incomplete seed
set, or development-only result as a finding.

## Evidence status

This ledger contains immutable evidence available at **2026-07-24 18:09 PDT**.
Phase 1 and the Phase-2 BSC main chain through `confirmation_16m` are complete.
The seven independently calibrated comparator-family branches are still in
development and are not ranked here. Their qualified 1M-token roots are
recorded as starting evidence, and the completed BSC-family width decision is
recorded, but no root score is treated as a final cross-family result.
Phase-2 paths below are relative to the operational campaign root
`/data/runs/bsc-phase2-d84627e`; its live plan at this cutoff was
`study:9311a523d5b001dac8b97cf31234f2cb26104d3f421e49a6b57b9aa4b9da5353`.

For Phase 2, every scalar score below is the mean raw-space FVU at exactly
256, 384, and 512 total bits/token. Lower is better. The serialized selection
metric is its negative. Rates include the fixed-width packet and amortized
deployable-codec bytes; interpolation uses the executed lower-convex-envelope
schedule on paired raw rows and never extrapolates. Development selection uses
seeds 0 and 1, aggregates by median then worst seed, and requires a complete
scientific qualification for both seeds. The active gate amendment does not
use partial-view diagnostics for promotion.

## Phase 1 — truth-known identification

### Scientific authorization — `phase1-decision:df789d6b27930bb813fcec1b9fde209e3a662d4adb9d42974850d7d05bf385c2`

- Outcome: `go`, authorization mode `scientific_go`; both scientific and smoke
  Phase 2 were authorized.
- Blueprint:
  `phase1-blueprint:ff7f9c41de3e412834604cba87d051c0082bc7c30d9b9462700d750b4f581e2d`;
  final plan:
  `study:92dc80c01ceed5e26e94e12503a9b5da77b76bab445350ffb0f30f35e5e1f469`.
- Campaign-manifest SHA-256:
  `sha256:33112b9c685ccff993248d735d20b98622978acacb52d9d2bedd996f809dc114`;
  blueprint-content SHA-256:
  `sha256:fc671f61b1ea113ce09c26bb97e26afec96599c058024db78efcc2e190825c54`;
  plan-content SHA-256:
  `sha256:abb942f0ef81e602d0969bf214437ec3c9c5d58e52b77bbb691efeadf9e1f566`;
  journal SHA-256:
  `sha256:2b2d4140417be422be5fca6fd8468fa19145ebae36031f5ce27683ca54d3bd2d`.
- The one-site instrument qualified in all three seeds. The fixed four-site
  carrier was selected by
  `selection:1d6180dfe8b4da0db0a920eef12cec1e7f5423f9aab4cf41f49f48766f6c4c96`
  with worst normalized truth-identification margins `0.9995999199839969`,
  `0.9999999999999867`, and `0.9975999999999998` for seeds 0/1/2.
  Its selected candidate was
  `candidate:63f7eb26f52da6c65ee8c281ba6b4f7191ae8f40b1f809fb3e1ebd987c8fbe58`.
- Selection cells were
  `cell:f5968a068b9bb7c5c5295ada56ad9f5bd214ccf56147f1fdc9285d4aea4d99ee`,
  `cell:3e5c4f9c39871e1f69b1bb707a692bfea68d4e3882e1227c21cfe584d9341574`,
  and
  `cell:328fb75f55b2550261f51318c446a6c91cdc23d75d7978f86525e90311018886`;
  their qualification digests were
  `sha256:028c6dd7584648256821c61dd9b8ca398afc9ea1c46269db566fc524a20713d9`,
  `sha256:ee9fe86c741e361a929796ce323c86e9c66af1026d1b717696f1703902c907ba`,
  and
  `sha256:099e2222f79b045b76c49e74de807e7b066ccb5601f1a94eeb14812a0326600d`.
- On untouched confirmation data, the matched baseline passed the native,
  deployed, and conjunctive identification gates in every seed. Its candidate
  was
  `candidate:bcbd4ec8b74d343a227a9e178f99543778dae7a630adb8514e3da858e52b8cf6`;
  cells were
  `cell:410c1a440e499a312f9a3ffdf6a511a756e6fd38d81ed3d7008b92163796f97d`,
  `cell:2e3a4f45313e6a5a7833bbe9ae478deee8232a0dad8799749e769e282780eae3`,
  and
  `cell:2f82e9850f529fd4679472ba044221c95a06d890451160e72d98618189b91b65`;
  qualification digests were
  `sha256:2d7d62f1ba8cc194f27fcc9d4e031ad2425d5c6d536825715d6c1d07cdbeb885`,
  `sha256:74f4b5ee8bd83e2dd2237162e9978e61be5b5c5d0c7d45c7e15a55d33ff06904`,
  and
  `sha256:fdd0bcbc7f7277efe5fbb4f651001d52af284d0ccc974f4cf39f238daebf39d1`.
- Both negative controls behaved correctly. `support_only` failed native and
  deployed identification in all three seeds.
  `site_span_one` passed its narrower native/deployed checks but failed the
  full conjunction in all three seeds. Thus reconstruction or support alone
  did not masquerade as recovery of the declared shared vector factor.
- There were no confirmation stress failures and no claim-scope narrowing.
  FVU remained only a guardrail; authorization came from truth recovery.

### Universal transfer — `phase1-transfer:a5a3dfbdaf9cc0fce9bdaacf063eaefbb53bd4e10402a80ba1d56f4b3e38f561`

- Schema: `bsc-phase1-transfer-v3`; method-contract SHA-256:
  `sha256:fd9cb96d6859437e738a819e1a302a89370c640868eb8b03601d7b89392d3f79`;
  provisional-carriers SHA-256:
  `sha256:b0cbb9e888e5dae43596dd6bb281dcd4100763a45f0c1a62b2d7c64bfe00c544`.
- Exported semantics were `shared_signed_coordinate_vector`, clean-all-site
  masked-encoder targets, calibration-only held-out target-rate thresholds,
  availability-rescaled site fusion, deterministic lowest-index cutoff ties,
  and squared-L2 reconstruction.
- `signed` activation and `decoded_energy` scoring were only provisional
  carriers. Phase 2 explicitly reopened both decisions. Capability evidence
  was diagnostic-only and pruned no real-model option.
- No synthetic architecture, optimizer, width, rate, or regularizer winner was
  exported to Phase 2.

## Phase 2 — common provenance and contract

- Blueprint:
  `phase2-blueprint:f5b459552c7768341c329f43b2b7a26af9f0d9cb488fec0e263ea6c8af3ba0ae`;
  source Phase-1 decision and transfer are the two artifacts above.
- Active common-gate amendment:
  `phase2-gate-amendment:2801ad39b330155a1c4cf16130520b254870e389b29546c09a1449e76e71672c`,
  artifact SHA-256
  `sha256:6b872d748254a54e05ce5aa4f15ecdda5ed9b33baf749626a4a43eabccc0042c`.
  It makes site-only and leave-one-out endpoints common nonpromotional
  diagnostics and requires the same complete fixed-rate raw-FVU admission for
  BSC and comparator families. Codec-excluded events remain reported but do
  not independently fail a cell because their distortion is already priced.
- Raw activation identity:
  `71a03ab0dbe746e1822bd88446889dae26ecae2392b8cbbeac929d6334329d8f`.
  Derived-view identities are `scalar_rms`
  `6ced2e5e6b5196796e1370c52b8565ba1ed8fc7aa5a9f63a86c5346551d0dbac`,
  `none`
  `95da9198a4d3d11a996ed8f4bcdf7065dccc753271ab296538ca31c0f257a1bd`,
  `sqrt_d`
  `639fc54dd5e0c2b3ca21260e34ed71a48eb98a970c89649493a3fc1ea14e40bb`,
  `whiten`
  `2bae1d3fe7ea3ea6b0865d243e0a532af8cfaf6abf8552a7a7f3982accf9a56c`,
  and `layer`
  `983f27955619dd57cd0d0474c4d2fba69d42388451bed71c88d3b8f13d2e1096`.
- Capture used `openai-community/gpt2` revision
  `607a30d783dfa663caf39e06633721c8d4cfcd7e`, GPT-2 byte BPE with tokenizer
  bundle hash
  `sha256:de7f8bdffc569820e523f5ace3f38fecba941a4da97150b2e7b4169852932c1b`,
  OpenWebText revision `b4325f019c648b1641a1784748667e8b74e5e064`,
  contexts of 128 tokens, and residual-pre hooks at blocks 3, 5, 7, and 9.
  All four sites have dimension 768. Forward and immutable store precision
  were bf16.
- Split sizes were 250,000 normalization-fit rows, 250,000 codec-calibration
  rows, 1,000,000 development rows, 1,000,000 confirmation rows, and
  16,000,000 train rows, allocated by whole packed sequences with stable
  `(sequence, position, token_id)` identity.
- The campaign began from clean commit
  `d84627eb165fcef789de12a42035e3c2e4d00b39`. Authenticated amendments bind
  subsequent orchestration and confirmation repairs through clean commits
  `e116fdc5818323ca811819d02d7f6b92716fda38`,
  `1e977bcc6436002e2b587b61ad468043e6bedff4`,
  `0ceffe6f1cb38ccfc7c8e5d393d8a78f75c4f7cb`,
  `3d13c30f9f82e11c547beabef43463c56b9190ef`,
  `3d31f704785fd4482883dd0b22393fb984802f57`, and
  `cde3388a59acb1d0ce3777021313ececefeea0ce`. These amendments repaired
  authenticated execution, saturation, frozen-policy validation, and
  retry-state replay; none changed a completed scientific cell kernel.
- Authoritative execution was Python 3.12.13, PyTorch 2.8.0+cu128, CUDA 12.8,
  TransformerLens 3.5.1, Transformers 5.14.0, and Triton 3.4.0 on an RTX 4090
  under driver 595.71.05.

## Phase-2 BSC main-chain finding

The current development finalist is a real and substantial improvement over
the 4M-token starting carrier: mean selected FVU fell from `0.401556957` to
`0.250652800`. The largest gains came from optimization (`3e-4` learning rate)
and the score-selector interaction (decoded energy with block BatchTopK);
SASA-style dead-residual recovery then supplied another robust 16M-token gain.

### Current selected configuration

| Surface | Selected value | Lineage |
|---|---|---|
| model/sites | GPT-2 Small residual-pre blocks 3/5/7/9, four 768-d sites | adapted, frozen Phase-2 pilot |
| normalization | scalar RMS, fit on 250,000 dedicated rows | adapted carrier; independently confirmed |
| encoder | joint untied linear, no bias, availability-rescaled sum | novel BSC carrier |
| decoder | free-scale-controlled, no bias, concatenated-L2 block geometry | novel plus BSF-adapted initialization |
| site factorization | rank 4 | adapted from fmxcoders; selected by preregistered noninferiority/parsimony |
| code | signed; 2,048 groups × width 4; 8 active blocks | novel signed block code; fixed 32 active coordinates |
| score/selector | decoded energy + block BatchTopK | novel, selected on real evidence |
| masking | Bernoulli probability 0 | adapted control; positive masking did not win |
| optimizer | fused Adam, LR `3e-4`, batch 2,048 tokens | Adam adapted from BSF; numeric choices tuned here |
| schedule | 2% accepted-update warmup, then constant; no weight decay | novel |
| regularizer | none | selected after nuclear-penalty panel |
| auxiliary | SASA-style frequency-dead residual, coefficient 1, AuxK 8, dead frequency `1e-4`, 1,000-token window | adapted mechanism, tuned here |
| train budget | 16,000,000 unique rows and optimizer-token presentations | novel Phase-2 budget |
| codec | q = 2/4/6/8/12/16; calibration-only threshold; priced deployable bytes | adapted/engineering |

The selected 16M cell has 50,331,680 parameters. Estimator
`dense-linear-memory-v20-e8cd28faf7b38d6e64f0426000de174679f4c01413ec6647fa6b997219978e55`
binds `4.831838208e15` FLOPs, 4,654,293,448 peak VRAM bytes,
10,737,418,240 peak host-RAM bytes, 115,916,363,912 retained storage bytes,
16 checkpoints, and 9,933,678,080 cumulative checkpoint-write bytes per seed.

### Positive and negative results

- Rank-4 site factorization was accepted despite a small FVU degradation from
  `0.401556957` to `0.405361187`: every seed stayed inside the preregistered
  `0.01` noninferiority tolerance, so the lower-rank carrier won on parsimony.
- Zero site masking won. The exact rank-4 parent was therefore the only valid
  rank-revisit materialization.
- Decoded energy with block BatchTopK improved the mean from `0.405361187` to
  `0.388035723`; both seeds improved by more than the `0.002` minimum effect.
  Isolated-loss-decrease BatchTopK reached `0.398741351`; decoded-energy
  token-TopK stayed at `0.405361187`.
- Learned group thresholding failed badly at every tested Group-L21
  coefficient: mean FVU was approximately `0.96742`, `0.96673`, and `0.96542`
  at coefficients `3e-4`, `1e-3`, and `3e-3`. The hard selected parent remained
  at `0.388035723`.
- Raising LR to `3e-4` produced the largest 4M gain, from `0.388035723` to
  `0.316602359`; the later revisit rejected `1e-4` (`0.352098007`) and `3e-5`
  (`0.457310065`).
- Batch 2,048 improved mean FVU to `0.303566257`; 2% warmup improved it again
  to `0.299627610`.
- Final-fifth linear decay had a slightly better aggregate mean
  (`0.297022670`) than constant (`0.299627610`), but one seed improved by only
  about `0.00164`, below the preregistered per-seed `0.002` floor. Constant
  therefore remained the valid selection. Cosine was worse (`0.319866572`).
- At 16M tokens, no nuclear regularizer beat the unregularized parent
  (`0.272731001`). Decoder-nuclear coefficients 30/100/300 produced means
  approximately `0.27420`, `0.27928`, and `0.28908`; the end-to-end-map
  nuclear variants ranged from approximately `0.27634` to `0.30371`.
- SASA-source dead-residual recovery improved both seeds and the mean by more
  than `0.02`, reaching `0.250652800`. The long-window and low-weight variants
  were positive but weaker (`0.267209` and `0.268620`).

### Partial-view geometry diagnostics

The selected auxiliary cells reconstruct well from all four sites but do not
support a single-site sufficiency claim:

| Diagnostic | Seed 0 | Seed 1 |
|---|---:|---:|
| all-site FVU | 0.219203 | 0.219721 |
| leave-one-out held-out-site FVU | 0.842556 | 0.855583 |
| leave-one-out coordinate concordance, mean | 0.934882 | 0.933867 |
| leave-one-out support IoU, mean | 0.177310 | 0.174657 |
| leave-one-out intersection energy coverage, mean | 0.973343 | 0.972931 |
| site-only held-out-sites FVU | 47.576372 | 47.548965 |
| site-only coordinate concordance, mean | 0.640705 | 0.637132 |
| site-only support IoU, mean | 0.009317 | 0.009297 |
| site-only intersection energy coverage, mean | 0.953563 | 0.953383 |

The evidence supports a joint acausal cross-layer code, not recovery of the
whole joint state from one local observation. High intersection energy
coverage and coordinate concordance coexist with low support overlap and poor
held-out reconstruction. These endpoints are scientifically important
negative diagnostics, but they are not a capability hurdle imposed only on
BSC.

## Phase-2 confirmation finding

`confirmation_16m` is nonselectable. All ten cells are integrity-complete and
scientifically qualified. The four deployable normalization modes remained
promotion-eligible; token LayerNorm was correctly ineligible because raw
decoding requires unpriced per-row oracle information.

| Normalization | Seed 0 score | Seed 1 score | Mean | Status |
|---|---:|---:|---:|---|
| scalar RMS | 0.245329156 | 0.246758186 | **0.246043671** | qualified, deployable |
| `sqrt_d` | 0.245714309 | 0.246583468 | 0.246148888 | qualified, deployable |
| none | 0.249536515 | 0.248693381 | 0.249114948 | qualified, deployable |
| whitening | 0.263695954 | 0.264293307 | 0.263994631 | qualified, deployable |
| token LayerNorm | — | — | — | scientific checks passed; codec-ineligible |

Scalar RMS beat `sqrt_d` by only `0.000105217` mean FVU, so the normalization
comparison is practically a near-tie. It nevertheless remained the declared
development parent and was also numerically best. Its seedwise fixed-rate
frontiers were:

- seed 0: FVU `0.247219379`, `0.244389791`, `0.244378297` at
  256/384/512 bits/token;
- seed 1: FVU `0.248645500`, `0.245820369`, `0.245808690` at
  256/384/512 bits/token.

Confirmation did not expose degradation: scalar-RMS mean FVU was about
`0.00461` better than its development parent. The exact scalar-RMS cells were
`cell:3c965dd19f6085657ecb52e16108654699ba37323150b7c201dc5bac2c837906`
and
`cell:7f29ba727322d0bcf868eb1070653c46d3e8c5ae6d54166017393874956343b1`;
qualification-file SHA-256 digests were
`sha256:c21d0f20522f449a0c8ec94ea0e2e8c6b3e1e98450d9ebee7f5ff576310f4106`
and
`sha256:feb912465fb92efdc9846cce4d04187e65dda4d6b1d141448fb089d484c61179`.

Other confirmation cell/digest pairs are:

- none:
  `cell:30d1e800134d542b109731354c52321757806017c15e5b449a83990a277de042`
  / `sha256:7362ccf0c54bcebc85c27c1340f53c5d3f84f23a5090199f192faf1ecf44f68e`,
  and
  `cell:28742c6b86dc225a5406dc6eb2652b65aba1d2c9b1f7261a775248a5b3f306ed`
  / `sha256:3f30eed5789289a8f496fdb307daaf10cd2419078720cf65bdcb296f987e1352`;
- `sqrt_d`:
  `cell:a5b8e2f7619e3b22dadeae32217c5658296407093335e5d508a6882c9cbcfcad`
  / `sha256:3e3fda0f9a22e7b1137ac071ddeeb9e2fea682cbfe9cf8a2087f8098b295b567`,
  and
  `cell:e2e5d06471595ac6932701d0922b7b4f41e667fe41a541c4c86e10ba897bff00`
  / `sha256:c1d56cee92fcb5961b8d939d2ddc684d36e785ebe189e91fd6c8b8c0bf725cfa`;
- whitening:
  `cell:4f1864fbbb64c6739b76345b6c5ceb2ace580c5cfa19d84871dec655e6018d29`
  / `sha256:4a5e14a1351e8ad2e822ea7bcaedbe26665b7d74fb11777d86aa68d1d61d3b51`,
  and
  `cell:43ecbc74b8341836a69d535b2cada20871f1057f8843defbf6d7c39aaaf2ca4f`
  / `sha256:7c00d38f1dae3109c582c9bd906fa0994bc434577e5381e1b0072c81e801f790`;
- token LayerNorm:
  `cell:27a80d62758376fa9ded86d137ff20884b6300e7f9d69c42e94cdcf5368ae36e`
  / `sha256:a0494d79dc103bcb00127906f73b73623a1253df0951e696cb1b0668c26b810e`,
  and
  `cell:b373a0f27014f601596c03370cd917aefaf0064f0ffa994b0e6ae6d075b7a238`
  / `sha256:a11eed15c902372de8e6a385b564fbb48f1fb4a9dd6a98ec5ccd550453f5278b`.

## Phase-2 main-chain immutable decision ledger

Each entry below names one immutable selection artifact. The associated
artifact itself binds the complete ranked and excluded candidate universe,
threshold-sensitivity surface, per-cell decisions, and active amendment.

### `architecture_4m` — `selection:dff1b25caef5610f37cf0a49879c3d839c110ee2de63d7d49c7cd8770411e93e`

- Artifact: `selections/architecture_4m-amended.json`; SHA-256
  `sha256:e990e63a267265757710424ebaeab21f2e501cb0d86821d77fe98d7fca450bdd`;
  source plan
  `study:0561cd4050babb6195111a36108f4ca959b289ed82933c20a35f64aef1916c04`;
  inner selection
  `selection:1f1a11f86e2f06040c757a265d8af767c9ad076588de24ac2b93135fae1c84a9`.
- Selected `candidate:d52bc40167e75cb177f23865ba1c30db1296bc32cf1fae5852091a9ae749b988`
  with seed FVU `0.400998614` / `0.402115301` (median `0.401556957`);
  cells
  `cell:d6e6fe863cf2ea0f8f745ef8be0f114cc709ed3a1126efb5c6580ead75fbbb30`
  and
  `cell:30183636bec271a7dcb6bf8e3f9cf9d825406bd2a87284f92f882104b568c0eb`.

### `capacity_4m` — `selection:adce0ae3ff0724ac409db95a2b2430518ea7345e416569718f5d2868dc67450b`

- Artifact SHA-256
  `sha256:6fef0e51d2f06f72e67713cb47efd739878b14de35cf4c7b58e18925bc812f23`;
  source plan
  `study:74b32712d3932f37b69cd2b986cba7b3a3f012f4fa33d606ce1b092f68d3d4ec`;
  inner selection
  `selection:9c1f1ddfcab27a02e8e36aa7f69d615098d67ec644ee02c7df438134ec5832e8`.
- Selected `candidate:c712ca8f10748a35becef6066106aac4dea82f864831bc9046ac9fea290594ec`
  with seed FVU `0.400998614` / `0.402115301`; cells
  `cell:a09f2df7a8caf9bcdcc0055b03525514f85169725643ab2704dbd1f76e5967ed`
  and
  `cell:bef1622da2cd9e29f9ca8a318bf00a3be470e049e7993860be8d77f5b5decf9d`.

### `site_factorization_4m` — `selection:edc7f2a60c099b38a194f6185bab62410ee835e5eaaa6b7fe0622eee4cfa2696`

- Artifact SHA-256
  `sha256:6d7e281fd7e2596bcaa3c64f5df6cc1a4ec330bcaf5fa3235b847337eb6a6c55`;
  source plan
  `study:0cd2825095e1094b9ccc339709a963290548c5700455df191fff77cc460159af`;
  inner selection
  `selection:aea44eacb57c15fbaab4b282777b1ca21abe025a39b0736934bc246cdc981482`.
- Selected rank-4
  `candidate:aba6c73e04178daba2464affa4c9b0ec1814466ba1e868f0e9a17b6f649aeb3e`
  with seed FVU `0.405322972` / `0.405399402`; cells
  `cell:ba4bc28bd59ea821bebc31fb7a77ea07d0ba37eeabe34e498779f33f286c9666`
  and
  `cell:741e6b5848a7b11280f7209940d361316e5ee037ef86ccd2181a7957deb29621`.

### `site_masking_4m` — `selection:3d7bff50ac78f5b3d1988ea5a61dd4a41bdfeb3d6d2921dc2b5afb9210d97e1b`

- Artifact SHA-256
  `sha256:fd1ca898378f4fa5d94d1758cc6459d7cca8fba8f44c428d2c1aeffe00e760f2`;
  source plan
  `study:777fdf6240ae3b3114123c3c4b46b8cbfb217a121388567f3f1d051c5500676d`;
  inner selection
  `selection:efecf671f7b09df477dee2a9b0728846056ccd0926f8c1a456c6f29420e46200`.
- Selected zero-mask
  `candidate:e39d410a230ff4ace93d33971114f1bfb9cff5525de77d7333a3de4f590a9e08`
  with seed FVU `0.405322972` / `0.405399402`; cells
  `cell:407bd2c6774c54032b22ad1238d1490a9569f98a13c034242f6246f30dd067a0`
  and
  `cell:41c2081743f0c3f60ca2e5a558de5cb88b55ac894d98d44779e62a662722a8ad`.

### `site_factorization_revisit_4m` — `selection:f300f5f5741d7b3b4c2afcdd5be567fa5a6d73bc7fe3e32cdd85e0b79d834525`

- Artifact SHA-256
  `sha256:412ee210062c8fbf53a4e2e9b2d21487c13817146f00e44b3a425d277acc8c2d`;
  source plan
  `study:2fe28b8ea8a342a1a9a906b5758b3d10cff916458c4177f476b1b4b8ad8ee8eb`;
  inner selection
  `selection:53b4585afd509f5101dd1f7c1f64ef6e47b3c7e2851c242c8ad3c557cb604e7b`.
- Zero masking deterministically emitted only its exact rank-4 parent,
  `candidate:d4bc0f544bdcb457270996f8a3fc93fe753edc55de0e6d4b002b162c8b7f4f95`,
  with seed FVU `0.405322972` / `0.405399402`; cells
  `cell:f1cc0cc62af31ead54e31ca183221972973efffc9981fc9b1406481d0d18c04c`
  and
  `cell:c10539bd9954cd5a6c3b351b07ae0a7f16125a9ab4593ae1a996847e244468a8`.

### `hard_selector_score_interaction_4m` — `selection:09cbffed72fca744a515d47a560f1823f6b4ac9fbc4a96b106ac05fba2215d7a`

- Artifact SHA-256
  `sha256:ee342fef7dc59c6b58751f6da23bbe4ba38ab5bb425b9f05dfcf1cee0921a2dc`;
  source plan
  `study:ea68dbf83bcaa908c6f6837247ee6675a40450074c6094ffc95a620b746fff77`;
  inner selection
  `selection:7252c721dcb0d95ba44c666183c951ce95a398b40f8616b8c6181642cb461dd3`.
- Selected decoded-energy BatchTopK
  `candidate:0e687f6fd5c2d2409e99be603ae28699ad74d9a181f4165a1f77d86533ffd279`
  with seed FVU `0.388467389` / `0.387604057`; cells
  `cell:f1266d19912858196cb47e92960b71d743a46aa00b78100a1410eca578a7f43f`
  and
  `cell:b6d99ed022d0bb45a0aceeff3de5babea11261c67a64edd1608c27601faf3c2a`.

### `group_threshold_method_4m` — `selection:1a63c12b2dc23dbd5b238f5d44c7f2c7337971acabf3354b4ca1554829be7a28`

- Artifact SHA-256
  `sha256:613425eb21384de2c7f1849c209c21c89664ccf1e71296d5ff49991d8fe370eb`;
  source plan
  `study:ccc666b8c1442aea77d16a37b5b39151fc5db283d8c22d91a28bc4a27d6a44bd`;
  inner selection
  `selection:b5c1a3e5db019652f844aa931a8fffcbf522d2a6de93dbc96250ecb160d9c7b2`.
- Retained hard parent
  `candidate:94021cb1988f0203b5befa76a05f6602dc70ff370dafdd27e44381026f4fb823`
  with seed FVU `0.388467389` / `0.387604057`; cells
  `cell:951a4ced54e57526a7e1cd7a5233bd6cc28f217309b9034815fe663b6d87549d`
  and
  `cell:61ba72ca8fac73cba71817236847dcb752ef0ff6f358c6581a6cf24d18ded9b0`.

### `learning_rate_4m` — `selection:8078a4e58b0ee5c5e62f42fdf4602faa34af3afe5ebad883ce11837c68165615`

- Artifact SHA-256
  `sha256:890c88292a048953ede29ac29bfbed531867645dabc14fdd5d17de470bace4db`;
  source plan
  `study:8f4201d574b56f56c5c5e40e6d982df672224da9b0fadbc522504c2d5be3a414`;
  inner selection
  `selection:3c9df42027bbb5aae049437a6aeb9ed3b05843745307a3b07121639adb761852`.
- Selected `3e-4`
  `candidate:32e0cc8e3469aa4e29aaa2515cad3e99eccec60cd4578dc5d6ea8f95d90b3a8b`
  with seed FVU `0.316867102` / `0.316337615`; cells
  `cell:febf8d496b2ce2926f75733d3d61b4a1efc210e438228282c1c54d3c7ff17c86`
  and
  `cell:6592c4169b070053134e86ab546fb6454b501b42c8330ebaec11be79e820e82f`.

### `batch_size_4m` — `selection:9a53c2d909e39d928ee8e56426a1b9e2b11cac5715fa2c85a3cefbfc8d23deb3`

- Artifact SHA-256
  `sha256:c4c26ca3c95866656a6b6a3e981f5d719c21a4b76106a91ccf9d41a7d40129ea`;
  source plan
  `study:3d686561be1a6c7d3e0b34c9839c368f96191e2ba3c4737b5ea2605748a68ddb`;
  inner selection
  `selection:cb0c8c41e712fad0da50490186d70794f4236f3e37138694c2ce7c62edc98573`.
- Selected batch 2,048
  `candidate:b3494788c252b6abe4d564beeac6059327bfcc9f4c5e0a75eb5920e197469138`
  with seed FVU `0.304219002` / `0.302913513`; cells
  `cell:c0f1cb328ec3fa894ca7a3fd0df415731fa5783a0ee7b9c6ac3b8ebb1fda49d3`
  and
  `cell:58653f053946475f1a690da04fd5591cfabf38180025e896c52e371a1eac4e23`.

### `warmup_4m` — `selection:483a67dd24fe016326f10213cd20a2a8f715100377dacf4566ee59cfb2acbe97`

- Artifact SHA-256
  `sha256:a6de96fea7c97d5e32bdc107c221ee46bba8f508a24ac1c776a1ebad837b017b`;
  source plan
  `study:f428ec912ef2f39f8310e7ddabbca1be049c5b5f16ade80a2874060fa82b180d`;
  inner selection
  `selection:268c56c4e6754d8c143ddf51646bfd63242918b95872e4a91df07ac79b16fd95`.
- Selected 2% warmup
  `candidate:ef26913905bd11f2fb56e5d84969d84b5e785197973b6c405178b15feae4441b`
  with seed FVU `0.300071230` / `0.299183990`; cells
  `cell:8e922c9c5e5d15848306ed1d5afd3cca6c241c6e20ee35493ab9ce6463ebffa0`
  and
  `cell:1ffaec030fd49cd00d6b7f0a71760c9a2a9d03b39346a15f5970675d0c1a415a`.

### `schedule_4m` — `selection:6610a9e1d2a35bf3c83625ea66cf240978e5e8cb1eb8c177766af3c8a825153e`

- Artifact SHA-256
  `sha256:4894611a05cc674687c86c117eb890a2a37e8b2847df42eef2b50b1541418f80`;
  source plan
  `study:a3495d459ce03dd7d6e0a7d1964add3a43aacc0c8513ce77c0b1baf2ca3527e1`;
  inner selection
  `selection:ee9d7b64cbb7f5528060033b53fafdce478ea238d9c5b09aaf4c221fdf47748a`.
- Retained constant schedule
  `candidate:1370e0b46885a090e816e42f3e40c85400b37692a9a04284c86d982378160e8b`
  with seed FVU `0.300071230` / `0.299183990`; cells
  `cell:34e50ced473fa4c7087a58548bcaf85886e13cf29d2b16db9aa2017394ee1480`
  and
  `cell:d8b086256efaa6be0bb6ad52379403fcf32737139224d4b4094a209f1cff0911`.

### `learning_rate_revisit_4m` — `selection:28116b81efcf830624b2b69a6b366c2c5a3c0492e8b7c5ad248762e0bd5a4f14`

- Artifact SHA-256
  `sha256:91d6f1e2176c45d435a34f2e4078e23f1eed500823f72a358a89459a493969a9`;
  source plan
  `study:fdbeea6d6b920d57b54b0da158ff20f09586d14ae28f4b5f7ac4be889c128df0`;
  inner selection
  `selection:69a031ce692817d9f4729c9cbc0309d06426fc69727c609fa3414b830e69c202`.
- Retained `3e-4`
  `candidate:9d1c8ed75fd18acaf5dba026c44e7a13b5af267aba6ddd5cebf6e50914825b37`
  with seed FVU `0.300071230` / `0.299183990`; cells
  `cell:634ca0a526b7936afdf3f5f7bbd46ed7a5eadf645db451dc96058357a210c42a`
  and
  `cell:3e85ddcbe36fafd81b76f5761ca71162d5040c82760bca9fe670f7495128f925`.

### `regularization_16m` — `selection:e48dc4752c62faa596863c119c627ef35b8c88244b447fc3b2ede2792aeef645`

- Artifact SHA-256
  `sha256:19f7d923b869a7098241e434b70ba7265fd0cdd30ebeb5415c62b09a674911e6`;
  source plan
  `study:b9f32b256e6d957d1a673784f6b027419f71a2175bea4e8a58265448d7bcfc33`;
  inner selection
  `selection:c16298381e5a7e83869f70735e4056a2d9859566445846417468b5f50a00df58`.
- Selected no regularizer
  `candidate:594ac41a672b6631cfe9953c1793b6b26d80f449926d0a59eb2a6fa79cf433f5`
  with seed FVU `0.273082867` / `0.272379135`; cells
  `cell:772300e16918f4dda72809c68c3b6d9f59efc7926b7542062c9713231b508de0`
  and
  `cell:c04387930dc7a8283e980141de63ee8da5f164b38e0a2974255224815fb8634b`.

### `auxiliary_16m` — `selection:565e82792b5919de26b58312b727946318a8ea38fd6532d81e07cef637397097`

- Artifact SHA-256
  `sha256:7cae488a3e54883a5c078f1aa863a9951e88e4113e3031bdabb1896efdbeca7a`;
  source plan
  `study:2ed08060c348f8826b19be8c610ed5a7a6219a019a87a7950bafc44308adb91d`;
  inner selection
  `selection:b6cfa36d0d73f8223c556372230697e61db02044ab8d53cb18482a8b46bfd9f2`.
- Selected SASA-source auxiliary
  `candidate:2ce8f73b7fd2a62103f5640a1a985801b903ae7b2758c776e592f9326ea84887`
  with seed FVU `0.250241042` / `0.251064558`; cells
  `cell:6180b93afda20194ecf8be8d9e43be9a44bada2140c738c72b3b541135894661`
  and
  `cell:ebe16481644019574b11bddd1585b079a39aabcd6ec9fefbb9ce510ef34e30c1`.

## Qualified comparator-family roots

These are immutable 1M-token root measurements, not final comparator results:

| Family | Candidate | Seed 0 / 1 FVU | Selection artifact | Artifact SHA-256 |
|---|---|---:|---|---|
| BSC shared coordinates | `candidate:d4e2a53798d0932aad066f759230e5f8b9b6ed39ccfb26d8fc9d297260647aaa` | 0.538981 / 0.537344 | `selection:f0a8ad1f3b531a67a3a19272d8d9a2035d74ca76bd1add91d7dfd5611759c357` | `sha256:03ac25d6a1ce21df15c7660e86391b04009fb3164ecb475be8b9b8c3ac1b156d` |
| BSF Grassmannian | `candidate:8f126cdb7cac7aeaa26d8e9bc5cc69d4897fcf48702555e46baf46163d3137a6` | 0.813115 / 0.821469 | `selection:8e955d22f7bde4567a2ea479fd4d0ea9268ce6f92c02d2637a3ffcd77a5d73e3` | `sha256:ed50c006ed5964045c1ba3eef130d582d14fc599d8faacd2e0d5e324f7325eba` |
| BSF Group Lasso | `candidate:3b230e163f3d0ac3e8730ffeaf1ec2ec6867c3e976ceeb0ec1b444d0f7a339dc` | 0.966073 / 0.966031 | `selection:f526002cc2c3aeed9dcbb0a10f802a15310ae6ff030e12e0dedadf0f3c66b411` | `sha256:3e437dabdb00059be330ad50b6f2cbf4f3775db2f161267ab4967c2542ed565d` |
| SASA | `candidate:a4c6f47e22084f4f1c8de91204170466241de3a181051ff1d1b5803cf09e5604` | 0.550638 / 0.547401 | `selection:e7d39750d677883bfeadca66cb7e938cdd413e09c7c67ba9fd14db5964534a53` | `sha256:df038e6ca6c1697399361d2f8bdad020d9276a68e377e2ca5b0b9b6421f5b547` |
| Anthropic dense-L1 bridge | `candidate:a40c17939f69651a9b4fad40aef3d487d8e31f2cb9e2109d14ebb345c8a35b78` | 0.938735 / 0.938963 | `selection:9a5b8270ae9a945a33b611dffc8a7521235d74912749dcad8471dc6d59047bbd` | `sha256:726ade858f6bfa6b5343481803af13721c4c91824dc50b44fd1b2d95ed449ea3` |
| decoder-weighted BatchTopK | `candidate:e85da4c4e9e2d647d4fa1fb01b8f154ba8c1821161ec055aaae476a114d58a8e` | 0.559838 / 0.560133 | `selection:e183c111a35249ac74e71507d644a25966b15dba45eb4b770f9a7da060ccae44` | `sha256:dbc6a23802c29ce60cf865009536c1b9986a4e884fe7fc371ee5584f79f1560f` |
| scalar ReLU BatchTopK | `candidate:f0695a29d8e31ba2af392e8b3ddf4ac08722d0473f8d2fa313624486ce6433e9` | 0.769379 / 0.770003 | `selection:da9f21629c00360ee68f2f5a27fd00bb4e2968e9f90bec34ccbb3260c06aa8f8` | `sha256:5f9e599f213b02dd917b767d1fa41cfdaf34229f75e1da1b42ed91627374f346` |

The BSC family has one additional completed width decision:
`selection:8996c3a45b82efd6892186651701d290a927170068fd2027f50b138fffae5522`
selected width 4
`candidate:de4533cfab5a4a0afe8ae8a6b7446212c1f47047eb87f38fc19bcaf762c7b931`
at seed FVU `0.400998614` / `0.402115301`, ahead of width 8
(`0.438543479` / `0.440344899`) and scalar width 1 (`0.545655121` /
`0.544671998`). Its artifact SHA-256 is
`sha256:458b5ace3c62394c5019109ffb4e2b5888fa198e559ae779b5016f37cbf131c6`.

## Current limitations

- Comparator roots differ in architecture, initialization, and 1M-token source
  recipes. Their raw scores are not a fair final ranking; every family must
  complete its own calibration chain and 16M winner/runner-up revisit first.
- Phase 2 is a two-seed pilot. It selects a Phase-3 finalist but cannot support
  five-seed publication claims.
- Main-chain choices are conditional and path-dependent. The family revisits
  test local path/order sensitivity, but they do not make the search exhaustive.
- High all-site performance does not imply single-site sufficiency, causal
  identifiability, semantic monosemanticity, or recovery of a global manifold.
  Decoder norm is not specificity, and aggregate reconstruction is not
  manifold recovery.
- No feature-geometry plot is a result yet. Those plots will be generated from
  the frozen finalist checkpoint and paired activation/code data after the
  family panel reaches a terminal state.
