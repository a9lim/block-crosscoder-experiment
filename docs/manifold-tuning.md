# Pre-NVMe manifold-fidelity tuning campaign

*Opened 2026-07-19 by a9. This is a development campaign over the completed
Phase-0 store, not a rewrite of the pinned Phase-1 stack. The winner changes
only after the campaign finishes its seed and holdout gates.*

## Why this campaign exists

The Phase-0 winner was elected by pooled FVU at the fixed structured operating
point `G=4096`, `b=4`, `k=32`, `lambda=1e-3`, and `lr=3e-4`. That is a sound
reconstruction/rate choice, but not evidence that the same point maximizes
manifold fidelity. The missing comparisons are material:

- no causal 4B `G` sweep exists;
- the surviving `k={16,64}` off-points mostly use `lambda=0`;
- the interval between the healthy `3e-4` learning rate and the damaged
  `6e-4` point is unmeasured;
- `b=4` has never been compared with an intrinsic ring width `b=2` or a
  packing-prone `b=8`;
- most healthy 4B seeds and checkpoints were never scored for geometry;
- adjacency hits saturate at 7/7 for weekday and cannot rank a round circle
  against a thin ellipse.

The earlier tranche-5 learning-rate deferral is therefore reopened for this
campaign. The production harvest path itself remains closed and ready.

## Evidence contract

Month, weekday, and the current zoo are burned development probes. On
2026-07-19 a9 also explicitly released zodiac from the preregistered panel as
a possible cyclic tuning target. It is burned forever and can never provide
confirmatory evidence. The other five Phase-1 families remain structurally
sealed until the configuration is frozen.

Zodiac enters the objective only if it clears a raw-stream prerequisite on a
source-document-disjoint A/B capture. The initial absolute bar (`shape R2`,
chord correlation, and roundness all at least 0.5 on four sites) was falsified
by the burned positive controls: it rejected month and all but one weekday
site despite their qualified BSC rings. The replacement is calibrated only to
those controls and recorded transparently: held-out harmonic shape R2 at least
0.25, chord correlation and roundness at least 0.5, and a 20,000-permutation
cyclic-topology p-value at most 0.01 on at least three of eight sites. Month
passes at layers 9/12/21 and weekday at 9/12/15. This tests whether the base
model supplies a reproducible semantic cycle before a BSC is rewarded for
preserving it; failing the gate is a real negative result, not a reason to
search checkpoints for an accidental circle.

A representative 8M-token scan produced only 359 zodiac tokens and zero
Scorpio/Pisces examples. It is retained as an honest prevalence/background
sample, but cannot estimate twelve centroids. The geometry capture therefore
prefilters FineWeb documents by whole-word sign mentions and balances accepted
documents by class before running the model. These remain natural contexts and
are assigned to A/B/C before packing, but the enrichment changes prevalence: operational
specificity must use the separate representative background reservoir and the
paper must state the sampling design.

The corrected document-disjoint capture has 97–234 tokens per sign, with at
least 15 distinct source documents in every class/fold cell and no document
crossing A/B/C. Zodiac fails the calibrated raw gate at 0/8 sites. Layers
9/12 carry a significant but weak cycle (`shape R2=.197/.148`, chord
correlation `.384/.368`, permutation `p<5e-5`); the later sites have no
held-out harmonic. It is therefore not a primary optimization target.

There is nevertheless a specific secondary hypothesis worth freezing. Both
the 12M and 24M center checkpoints map zodiac onto month block 595. At 24M its
held-out dense code is circular (`shape R2=.658`, roundness `.984`), but the
operational code is unqualified: no zodiac class claims the block top-1 on A,
minimum B recall is .025, and chord correlation is .340 despite 162x
background lift. Every sweep cell will report zodiac under the same rule.
Only a cell that independently clears the standing response, recall, and
specificity eligibility gates may be called an exploratory zodiac capture;
an attractive relaxed candidate cannot influence the election.

The activation probe is divided A/B/C:

1. A selects a checkpoint-specific block and fits every harmonic/decoder.
2. B ranks configurations.
3. C remains unopened until the configuration and selection rule are frozen.

Representative captures split by packed sequence and retain a bf16 background
reservoir; targeted captures preassign source documents to folds before packing.
The retrospective Phase-0 artifact has neither; its results are stamped
`sample_stratified` and omit specificity.

For a cyclic family with semantic phase

`h_c = (cos(2 pi c/C), sin(2 pi c/C))`,

fit the balanced class centroids in native block code as

`mu_c = a + h_c A`.

Report on held-out data:

- harmonic shape R2;
- native roundness `2 s1 s2 / (s1^2 + s2^2)` from the singular values of
  `A`, invariant under the legitimate within-block orthogonal gauge;
- balanced token-level circular-decoding cosine and angular error;
- mean and minimum per-class operational firing recall under calibrated
  `theta`;
- background firing rate and lift when the artifact supports it;
- chord-distance correlation and per-site decoded-contribution fidelity;
- adjacency hits as a topology gate, not an optimization score;
- pooled/per-site FVU, rate, dead fraction, skip rate, and seed distribution.

Block selection on A takes the union of the top responsive blocks per class;
requires the candidate to be top-1 for at least a quarter of classes; applies
recall/selectivity eligibility; and maximizes the weaker of harmonic fit and
native roundness. The response gate prevents a label-search over thousands of
incidental linear readouts, while the maximin rule rejects both an ordered
line and a round unordered cloud without inventing scalar weights. The block
is frozen before B or C is scored.

Configuration selection is lexicographic/Pareto:

1. guard, dead, FVU, and rate constraints;
2. operational coverage and specificity;
3. held-out harmonic and token fidelity;
4. native roundness and cross-site persistence;
5. reconstruction/rate as the tie break.

For the production candidate, the hard screen is finite state, no guard
refusal, skip rate at most 0.1%, final-window dead fraction at most 1%, and
pooled top-k FVU no more than 0.02 worse than the matched-horizon center.
Rate is carried as an explicit Pareto coordinate rather than hidden in a
weighted score. A geometry showcase may sit outside the FVU non-inferiority
bar, but must be named separately and cannot replace the production result.

Seeds are the experimental units. Seed 0 screens the surface; the best one or
two cells advance to seeds 1 and 2. The decision uses the median and lower
tail, never the prettiest seed.

## Stage 1: retrospective checkpoint backfill

Before new training, score every healthy surviving 4B checkpoint covering:

- `k={16,32,64}`;
- primary and site-renormalized gauges;
- `lambda={0,1e-3}` where available;
- seeds 0/1/2;
- 12M versus 24M optimizer tokens;
- repeated epochs versus fresh tokens.

This is already enough to decide which missing cells are genuinely necessary.

The 2026-07-19 backfill covered 14 healthy checkpoints plus both missing
12M `lambda=1e-3` center arms. The table below reports operational split-B
metrics as `shape R2 / native roundness / token cosine`. These legacy captures
are sample-stratified and lack background specificity, so they determine the
new response surface but are not final estimates.

| checkpoint | FVU | month | weekday |
|---|---:|---:|---:|
| renorm, `lambda=1e-3`, 12M | 0.4154 | .781 / .978 / .919 | .775 / .784 / .770 |
| renorm, `lambda=1e-3`, 24M | 0.3997 | .735 / .978 / .928 | .747 / .496 / .768 |
| renorm, `lambda=0`, `k=16`, 12M | 0.4822 | .831 / .997 / .851 | .763 / .215 / .586 |
| renorm, `lambda=0`, `k=32`, 12M | 0.4156 | .779 / .964 / .921 | .809 / .768 / .735 |
| renorm, `lambda=0`, `k=64`, 12M | 0.3598 | .713 / .629 / .858 | .742 / .168 / .615 |
| renorm, `lambda=0`, `k=32`, seed 1 | 0.4152 | .600 / .497 / .644 | .860 / .346 / .545 |
| renorm, `lambda=0`, `k=32`, seed 2 | 0.4154 | .811 / .944 / .910 | .794 / .157 / .572 |
| renorm, fresh 24M tokens | 0.4098 | .397 / .325 / .408 | .251 / .489 / .257 |

Direct observations from this backfill:

- additional optimization improved FVU by 0.0157 while weekday roundness fell
  from .784 to .496; month stayed close to circular. Training horizon is a
  geometry hyperparameter, not merely a compute budget;
- at the 12M seed-0 center, `lambda=0` and `lambda=1e-3` are nearly tied on
  both rings. The penalty is not the first missing causal variable;
- `k=16` under-reconstructs and `k=64` produces excellent FVU but highly
  elliptical weekday codes. The intermediate `k={24,40,48}` cells are needed;
- seed and data-order effects are large enough that no single attractive ring
  can elect a production configuration;
- fresh data did not improve cyclic fidelity. Repeating the matched pilot
  split remains the clean screening design, with fresh data reserved as a
  robustness axis.

## Stage 2: bounded seed-0 response surface

All new seed-0 cells first use the site-renormalized pilot gauge,
`lambda=1e-3`, 12M optimizer tokens, cosine decay, guard, and AuxK
gradient-ratio cap 1.0 unless the cell names the varied quantity. This is a
multi-fidelity race: the complete surface is compared at one matched horizon,
then Pareto cells are retrained at 24M. The existing 12M and 24M
`(4096,4,32,3e-4)` checkpoints are the two centers.

| cell | G | b | k | lr | comparison |
|---|---:|---:|---:|---:|---|
| `lr3.5e-4` | 4096 | 4 | 32 | 3.5e-4 | safe-side LR bracket |
| `lr4e-4` | 4096 | 4 | 32 | 4e-4 | safe-side LR bracket |
| `lr4.5e-4` | 4096 | 4 | 32 | 4.5e-4 | preserved recovery rung |
| `lr5e-4` | 4096 | 4 | 32 | 5e-4 | stability-edge map |
| `lr5.5e-4` | 4096 | 4 | 32 | 5.5e-4 | last rung below known damage |
| `linear3e-4` | 4096 | 4 | 32 | 3e-4 | linear-last-fifth schedule |
| `linear_k40` | 4096 | 4 | 40 | 3e-4 | schedule × k interaction |
| `linear_k48` | 4096 | 4 | 48 | 3e-4 | schedule × k interaction |
| `linear_k64` | 4096 | 4 | 64 | 3e-4 | high-k ring-rescue test |
| `k16` | 4096 | 4 | 16 | 3e-4 | fair structured k sweep |
| `k24` | 4096 | 4 | 24 | 3e-4 | refine below the center |
| `k40` | 4096 | 4 | 40 | 3e-4 | refine above the center |
| `k48` | 4096 | 4 | 48 | 3e-4 | refine above the center |
| `k64` | 4096 | 4 | 64 | 3e-4 | fair structured k sweep |
| `G2048_k16` | 2048 | 4 | 16 | 3e-4 | density-matched G |
| `G2048_k32` | 2048 | 4 | 32 | 3e-4 | fixed-k G |
| `G8192_k32` | 8192 | 4 | 32 | 3e-4 | fixed-k G |
| `G8192_k64` | 8192 | 4 | 64 | 3e-4 | density-matched G |
| `b2_G8192_k64` | 8192 | 2 | 64 | 3e-4 | intrinsic ring width |
| `b8_G2048_k16` | 2048 | 8 | 16 | 3e-4 | packing-prone width |

The width triplet `(8192,2,64)`, `(4096,4,32)`, `(2048,8,16)` holds
`G*b=16384`, `k*b=128`, and `k/G=1/128` fixed. Declared codec rate still
differs because the number of support events differs; that is a result, not a
quantity to hide.

The live memory gate resolved the reserved `G=8192,b=4` question on the 24 GB
4090. Chunked Gram formation/retraction, dead-count reduction, and tiny-matrix
CPU eigensolves allow ordinary full-stack steps, but once the mandatory AuxK
path sees dead blocks its residual needs 160 MiB with only about 140–160 MiB
free. Both fixed-k and density-matched b4 cells are therefore hardware-
infeasible under the pinned batch/optimizer/AuxK stack and are rejected, not
scored. This is not evidence about dictionary quality. The parameter-matched
`G=8192,b=2,k=64` width cell passed its 500-step diagnostic (zero skips and
floor hits, 0.42% window-dead blocks, post-cast Gram residual `2.9e-4`) and is
running the full screen; the causal fixed-b G comparison is bounded above by
the viable `G=4096` point on this hardware.

The five elevated-LR cells and the schedule control stop first at step 1800,
beyond warmup and the known
step-1600 stressor. G8192 cells stop first at step 500 for memory and early
dead-dynamics inspection. The campaign runner resumes only surviving cells.

The first 12M screen identified a real schedule × sparsity interaction worth
isolating before any 24M finalist run. `linear3e-4` was the best balanced
calendar cell, cosine `k40` had the strongest weekday centroid geometry, and
cosine `k64` achieved the best FVU while collapsing weekday roundness. The
three follow-up cells `linear_k{40,48,64}` change only `k` under the
linear-last-fifth schedule. Existing cosine cells are the matched controls;
no LR, width, gauge, penalty, or seed axis is reopened.

## Stage 3: finalists

Retrain at most four seed-0 Pareto cells for 24M optimizer tokens under a
distinct horizon-specific campaign root; immutable per-cell fingerprints make
12M reports/checkpoints invalid for that root. Artifact-bearing legacy cells
without a fingerprint are refused rather than guessed or silently adopted.
Geometry can
improve or shear with additional optimization even when FVU improves. Advance
at most two 24M cells to seeds 1 and 2. Then add the lower-priority
`lambda=3e-4` arm at the provisional best architecture and learning rate.
Freeze the complete configuration, horizon, and selection rule before opening
split C. Only after C is recorded may the remaining five-family sealed
Phase-1 panel be used for its declared confirmation.

The production recommendation changes only if the finalist improves held-out
operational geometry across seeds without violating the stability, FVU, and
rate constraints. A showcase checkpoint may be named separately from the
headline zero-regularizer R-D model, but the distinction must remain explicit.
