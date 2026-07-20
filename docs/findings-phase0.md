# Block-sparse crosscoders: Phase 0 findings

*Pilot program, 2026-07-15 through 2026-07-19. This is the authoritative
scientific account of Phase 0; [`design.md`](design.md) is the normative
forward plan. Compact machine-readable evidence lives in
[`data/evidence/`](../data/evidence/).*

## Abstract

A block-sparse crosscoder (BSC) learns a sparse dictionary whose atomic unit
is a multidimensional subspace: one block code is shared across model depth
and decoded through a different frame at every site. It fills the previously
open {block} × {cross-site} cell of the sparse-dictionary literature's 2×2.

Phase 0 established four results. First, Gemma already contains the object
the architecture assumes: concept coordinates correspond linearly across
depth even while their decoder frames rotate to chance alignment. Second,
the month ring exists in a 65k scalar SAE but below every clustering
threshold, so post-hoc blockification cannot reach it; native block training
can, and captures rings, number lines, and a geographic map as individual
blocks. Third, the combination earns its structure on an honest
rate–distortion axis. Cross-site tying cuts rate by 7.8–7.9× at no distortion
cost, the tying × blocking interaction is positive, and the site-renormalized
BSC strictly dominates the matched scalar frontier throughout their shared
pilot range. Fourth, the production training and harvest stack is now
specified by measured failure modes rather than defaults. Phase 1 is blocked
only on installation of the dedicated 4 TB NVMe.

## 1. The experiment

The motivating literature is a 2×2:

| | one site | sites tied by one code |
|---|---|---|
| scalar unit | SAE | scalar crosscoder |
| subspace unit | block-sparse featurizer | **block-sparse crosscoder** |

The BSC has `G` blocks of width `b`. For site `s`, block `g` has encoder
`E_g^s`, decoder frame `D_g^s`, and one shared code

`z_g = Σ_s E_g^s x̃^s`.

Its concatenated decoder frame obeys

`Σ_s D_g^s D_g^{sT} = I_b`.

The constraint fixes the scale gauge, reduces the remaining within-block
gauge to an orthogonal rotation, and makes `||z_g||²` the exact isolated
contribution energy of the block. Training selects blocks by that energy;
site-specific frames decide how the same code appears at each depth.

Phase 0 combined five forms of evidence:

1. synthetic ground-truth recovery and null controls;
2. post-hoc discovery in released GPT-2 and Gemma SAEs;
3. cross-layer paired-token tests before BSC training;
4. trained 1b and 4b BSC, scalar, and single-site factorial cells;
5. a preregistered activation codec and production-harvest drills.

The known calendar, number, color, geographic, element, and planet families
were inspected repeatedly. They are therefore descriptive probes, not
confirmatory endpoints. Phase-1 confirmation uses the still-sealed panel in
the design.

## 2. The target exists, and scalar post-hoc discovery misses it

### Shared coordinates survive rotating frames

Across Gemma Scope SAEs at layers 9, 17, 22, and 29 of Gemma 3 4B, held-out
linear maps between month-family coordinates reach `R² = 0.83–0.90` across
the 9↔22↔29 triangle; CCA is at least 0.96 for 9↔22. The corresponding
raw decoder subspaces lie at their random-subspace principal-angle null
(`p ≈ 0.33–0.52`), except one 22↔29 span match (`p = 0.001`). The code
persists while the frame rotates.

Depth is not monotone. Activation-space calendar rings are strongest early
(layer-9 weekday circular score 0.981; decoder adjacency `|r| = 0.886`),
layer 22 is a visibility minimum, and layer 17 undersplits both calendar
families. A single site's dictionary is therefore not an adequate verdict on
cross-depth structure.

### The ring lies below the clustering scale

The discovery pipeline first replicated the Engels weekday, month, and year
rings on the same GPT-2 SAE artifact, all at the permutation floor
(`p = 0.005`). Its Gemma null is therefore interpretable.

At Gemma Scope 4B layer 22, no decoder-cosine or activation-dependence branch
at width 16k or 65k surfaced a multi-member ring candidate; the supervised
month skeleton consists of graph singletons at a cosine threshold of 0.5.
Yet at 65k the twelve months map to twelve distinct top-1 features whose
decoder vectors are cyclically ordered (adjacency `p = 1.5e-4`, Fisher–Lee
angle order `p = 3.5e-4`) while maximum adjacent cosine is only 0.32. The
month ring is present, but structurally below the scale at which post-hoc
clustering can bind it. Layer 9 gives the same discovery null.

This sharpens the original hypothesis: the artifact exists; the proposed
post-hoc route to it does not. Native block training is not merely a cleaner
representation of an SAE cluster. It reaches structure that clustering never
forms.

## 3. Native block training individuates manifolds

The first trained 1B BSC contained a month block that fired on 53% of month
tokens and 0.2% of background tokens. Its twelve class means formed a ring in
one code plane holding 97% of class-mean variance (`p < 5e-5`), stable under a
split-half test and re-embedded by all six site frames.

At 4B and 12M optimizer tokens, the renormalized arm was the only arm to
capture both calendar families: month block b595 claimed 10/12 months with
ring order 10/12 at the permutation floor, and weekday block b862 claimed and
ordered 7/7. The primary arm's b2146 captured a cardinal number line
(17/20 top-1, Spearman `ρ = 0.90`, permutation floor). Digits individuated
one per block, while the renormalized dictionary sometimes bound notation
variants such as `3/third` and `7/seventh`. Renormalized b1781 captured 36/48
countries; its four-dimensional code linearly decoded latitude and longitude.

The matched scalar model contains much of the same population information
without unit-level individuation: month top-1 identities collapse to four
features at 1B and seven at 4B, while weekday collapses to one.

Two caveats define the claim:

- consolidation and internal order are distinct. Sufficiently trained 1B
  seeds almost always consolidate months, but ring order ranged from 2/12 to
  12/12. Block identity is initialization-dependent;
- a mega-block can claim an entire family without representing its topology.
  One destroyed run claimed 12/12 months but ordered only 6/12; a healthy run
  claimed 7/7 weekdays but ordered 2/7.

Consequently no capture fraction is interpreted without its order statistic
and model FVU. The generated [figure catalog](../figures/README.md) applies
this rule mechanically, including for families that fail.

## 4. The combination earns its parameters

### Tying is distortion-free and rate-efficient

The full factorial trained eight independent single-site block models and
eight scalar SAEs with exactly matched per-site tensors and rates. At `q=4`,
reconstructing all eight sites independently costs 6,031 bits/token for the
block cells versus 772.7 for one joint BSC (7.8×), and 12,509 versus 1,588
for the scalar cells (7.9×). Joint support costs fall from 1,067 to 261
bits/token on the block side and from 1,051 to 246 per site in the replicated
single-site comparison.

Using exact eval-split squared-energy weights, the tied block model improves
on its single-site control in the shrinkage-whitened gauge (FVU 0.4360 versus
0.4559) and ties it exactly in the site-renormalized gauge (0.4207 in both).
Cross-site tying therefore imposes no measured distortion cost.

The causal interaction is positive:

| pooled top-k FVU | shared code | one model per site | tying gain |
|---|---:|---:|---:|
| block (`4096×b4`, `k=32`) | **0.4299** | 0.4497 | −0.0198 |
| scalar (`16384×b1`, `k=128`) | **0.3682** | 0.3768 | −0.0086 |

Tying helps both model classes and helps the block model about 2.3× more:
the interaction is `+0.011` pooled FVU and replicates across three seeds.

### Rate–distortion reverses the apparent block tax

At matched latent L0, the scalar arm leads by 0.047 FVU. That comparison
prices a block's four coefficients but ignores its cheaper support. Under the
declared codec—canonical block orientation, calibration-fit clipping,
componentwise quantization, frozen count model, enumerative support bits,
realized per-token counts, and sequence bootstrap—the verdict reverses:

| q=4 region | site-renormalized BSC | scalar | result |
|---|---:|---:|---|
| ~390 bits | 0.4869 @ 390.7 (`k=16`) | no point | blocks extend the frontier |
| ~800 bits | **0.4207 @ 770.5** (`k=32`) | 0.4306 @ 822.0 (`k=16`) | BSC dominates |
| ~1.5 kbit | **0.3660 @ 1491.6** (`k=64`) | 0.3718 @ 1588.4 (`k=32`) | BSC dominates; CIs disjoint |
| ~2.9 kbit | no `k=128` point | 0.3249 @ 2900 (`k=64`) | open frontier end |

Quantization is nearly transparent: `q=6` reproduces unquantized FVU to the
third decimal, and `q=4` costs roughly 0.004–0.005. A Bernoulli support model
lands within about 5% of the frozen empirical count model. The result is not
an entropy-model trick.

See [rate–distortion](../figures/summary/rate-distortion.png) and
[tying rate](../figures/summary/tying-rate.png).

## 5. Gauge, budget, and geometry

### Site renormalization repairs a real allocation tilt

The shrinkage whitener retains roughly 6% of per-dimension variance at
shallow sites and 29–32% deep. Under equal per-dimension reconstruction loss,
site 30 alone carries 27.5% of pooled squared energy while sites 9, 12, and
15 together carry less than 11%. A scalar RMS renormalization after shrinkage
whitening makes site weights uniform to about ±2% without giving up
directional outlier suppression.

Three independent observations designate this gauge:

1. both single-site families' per-site FVU profiles correlate `r = 0.984`
   with the joint renormalized profile, versus about 0.4 with the bare gauge;
2. the renormalized BSC wins pooled FVU and the shared R–D frontier;
3. it is the only 12M-token arm capturing both calendar families and it binds
   several number notations across form.

The two gauges still learn corresponding manifolds: cross-arm code maps and
span overlaps beat permutation controls for every captured family. The cost
is a narrower optimizer-stability margin, which the pinned learning rate and
loss-spike guard address.

### Optimizer-token budget matters more than freshness

Four cells matched at 24M optimizer tokens separate repeated epochs from
fresh examples:

| pooled top-k FVU | primary | site-renormalized |
|---|---:|---:|
| 6M unique × 2 epochs | 0.4299 | 0.4154 |
| 6M unique × 4 epochs | 0.4102 | **0.3997** |
| 12M unique × 2 epochs | 0.4089 | 0.4098† |

Doubling the optimizer budget improves FVU by 0.016–0.020. Replacing repeats
with fresh tokens adds only 0.0013 in the clean primary comparison. The
renormalized fresh cell is not a clean freshness read: it encountered a
guarded spike cluster and one skipped batch (0.017% skip rate, below the
0.1% gate). We do not infer a freshness penalty from it.

The promoted winner is the clean epoch-renormalized cell at FVU 0.3997. Its
`q=4` point is 0.4053 at 771.2 bits/token, a 0.0154 improvement at unchanged
rate over the 12M-token champion. Support cost stays at 261–265 bits/token;
the improvement is amplitude fidelity.

At this winner, the winner arm itself qualifies three descriptive families:
month b595 (8/12 top-1, ring 10/12, permutation floor), weekday b862 (7/7,
`p = 0.0028`), and country b1781 (34/48, geographic `R² = 0.354`,
permutation floor). The matched primary counterpart qualifies cardinal b2146
and ordinal b382. Thus the earlier gauge split persists: the renormalized
winner holds cyclic and geographic structure; the primary gauge holds the
ordered number lines. The uniform figure zoo intentionally uses the promoted
winner only and marks primary-only successes as winner-arm failures.

### The stream contains more manifold supply than the dictionary captures

Ordinal class means form a depth-pervasive line that straightens to
`|ρ| = 0.99` by layer 24. Country means linearly decode latitude at every
site (`R² = 0.57–0.66`), while longitude weakens by layer 30 and countries do
not cluster by continent. Planet distance is visible in-stream
(`|ρ| ≤ 0.88`) without reliable winner-block order; color is stable but
not a hue wheel; element order weakens through the middle sites.

Across these families, frame rotation has a trough around layers 18–21
(about 0.71 versus 0.84/0.91 on the flanks). Captured block frames track the
stream's rotation at `r ≥ 0.92`; destroyed runs lose this coherence. The
mid-stack shear is therefore descriptive geometry and a useful health
diagnostic, not yet a confirmatory claim.

Co-firing block cliques are also real optima, not training noise. Independent
sub-width features can pack losslessly into one width-four block and free
both a block and a support event. Near-even contribution-energy splits plus
degraded code correspondence are the packing signature. Decoder Frobenius
norms cannot diagnose use because the Gram constraint forces every frame to
retain width-four capacity.

## 6. Training and harvest are production-ready

Gemma 3 4B training is bit-deterministic across repeated runs, including
8-bit Adam. That made the three failure classes separable:

- an unstable operating point triggers more than five consecutive guarded
  skips and is refused;
- a poison batch triggers a corroborated one-batch skip (the step-1600 event
  is batch-locked across three divergent trajectories);
- a SASA-style AuxK cascade amplifies the revival gradient until it re-kills
  its own revivals; an auxiliary/main gradient-ratio cap of 1.0 suppresses
  peaks by more than 100× and restores healthy dead fractions without
  changing clean trajectories bit-for-bit.

The pinned `3e-4` cosine point had zero guard events across the campaign.
The 1B optimum at `1.2e-3` is catastrophically unstable at 4B, so optimizer
points are never transferred across model scale without calibration.

The remaining production mechanics have measured margins:

- streaming threshold fitting uses 19.5 GB on the 61 GB host and reaches
  average active-block count within 0.0043 of the exact target;
- prefetch 4 reduces measured data-wait from 30% to 12%;
- the store costs exactly 40,960 bytes/token; 53M tokens require 2.171 TB;
- a 2M-token whitener fit is already beyond the functional estimation floor;
  the site-renorm scalars are slice-stable to about 1%;
- raw late-layer channel 443 exceeds fp16 range on 43% of layer-27 tokens and
  82% of layer-30 tokens. Whitened values stay below 27.3 with zero values
  above 32 in about 13.3 billion values per site; bf16 mean relative error is
  0.14%. fp16 is therefore banned throughout harvest and storage;
- the 246 GB extension verified at 0.62 GB/s, implying about one hour for the
  production checksum pass. Atomic-shard recovery and manifest rebuild
  passed an interrupted-write drill;
- harvest sustains about 5,000 tokens/s and 205 MB/s written, forecasting
  roughly three hours for the full store plus verification.

No scientific or engineering question in the harvest path remains open. The
production store waits only on hardware.

## 7. Hypothesis status and conclusion

| hypothesis | Phase-0 status |
|---|---|
| H1: token-level rings exist and post-hoc SAE blockification can find them | **split:** the ring exists at `p≈1e-4`; post-hoc discovery is structurally null; native blocks reach it |
| H2: subspaces persist while frames rotate and positions correspond | **passed before training** (`R² = 0.83–0.90`); trained shared-code validity remains a Phase-1 endpoint |
| H3: blocks earn their parameters on activation R–D | **strong pilot preview:** strict dominance throughout the shared range; production frontier is the verdict |
| H4: effective span localizes structure over depth | instrumentation validated; production contribution spectra and truncations pending |
| H5: manifold-level model diffing | deferred to the post-publication cross-model phase |

Phase 0 changed the project from an architectural conjecture into a measured
production plan. The shared-code premise holds before training; native blocks
recover structure scalar clustering cannot bind; tying and blocking have a
positive interaction; and the honest bit axis favors the BSC throughout the
pilot overlap. Phase 1 now asks whether those effects persist at the full
store and whether the learned blocks pass strict shared-code and effective-
span tests.
