# Phase-0 target: calendar rings in gemma-scope-2-4b layer 22, 16k → 65k

**Status: DISCOVERY NULL at both widths (2026-07-16/17) — but the month
ring exists in the 65k dictionary, decoder-side, below discovery's
thresholds.** No clustering branch recovers a calendar ring at either
width: every family-affine candidate is a *single* feature. Underneath,
the supervised probes trace the Engels mechanism across the widths:
ring geometry requires the dictionary to *split* a cyclic family into
per-class features, and splitting is an overcompleteness effect
arriving per-class in frequency order. At 6.4× (16k) months have only
begun to split (May first) and the few split spokes sit in calendar
order in activation space (circ 0.52, p ≈ 0.015). At 25.6× (65k) the
month split **completes** — all 12 months get distinct selective
features — and those features' decoder vectors form a measurable
cyclic structure: adjacent months more similar than non-adjacent
(p ≈ 1.5×10⁻⁴) and PC-plane angles winding in calendar order
(Fisher–Lee |r| 0.59, p ≈ 3.5×10⁻⁴). The catch that keeps discovery
null: the ring lives at decoder cosines ≤ 0.32 — far below the τ = 0.5
graph prune and drowned by the near-orthogonal bulk in every
activation-space PCA plane. Weekday, further from the splitting
frontier, has no ring at any level at either width. With the positive
control passed (`docs/findings-phase0-control.md`), both nulls are
interpretable at demonstrated power.

Run artifacts: `/data/stores/bcc-phase0/gemma3_4b_l22_pile/` and
`.../gemma3_4b_l22_65k_pile/` (4M-token CodeStores + `target_run/`
caches) on jobe; figures mirrored in `figures/phase0-gemma/`; the 65k
artifact pin + method-matched geometry in
`data/phase0/artifact_provenance.json` and
`data/phase0/split_geometry.json` (jobe). Scripts:
`scripts/harvest_phase0_gemma.py`, `scripts/run_phase0_gemma.py`,
`scripts/check_gemma_multimember.py`,
`scripts/check_gemma_supervised_ring.py`,
`scripts/pin_gemma_65k_artifact.py`,
`scripts/check_gemma_split_geometry.py`.

## Setup — 16k (provenance-gated)

- **Artifact**: `gemma-scope-2-4b-pt-res` @ `blocks.22.hook_resid_post`
  (`layer_22_width_16k_l0_medium`, d_in 2560, d_sae 16384) — pinned in
  `data/phase0/artifact_provenance.json`. Layer 22/33 ≈ 67% depth via
  saklas `select_runtime_layer`.
- **Harvest**: 4.0M tokens of `monology/pile-uncopyrighted` under the
  SAE's own conditions (ctx 1024, prepend-BOS, model bf16 — fp16 banned
  in the harvest path; BOS dropped at harvest). Mean L0 60.6, zero-code
  fraction 0, 16,204/16,384 features fired. 4.5 min on the 4090.
- **Families**: weekday (7) + month (12). Year is excluded at the
  tokenizer — gemma's SentencePiece has no single-token years (matches
  Engels' own non-GPT-2 scope).
- **Clustering, dual-branch**: gemma-16k decoder geometry is far more
  orthogonal than Bloom's GPT-2 SAE — median max-neighbor cosine 0.218
  vs 0.547, 4.8% vs 14.1% of features with a >0.7 neighbor, 6.4× vs 32×
  overcomplete — so Engels-style spectral clustering degenerates
  (median cluster size 1, one 2192-member blob, 73 clusters with ≥2
  members). The kNN-graph method (k=2, τ=0.5 — Engels' own answer for
  their 65k Mistral SAE) runs as a second candidate branch: 71
  ≥2-member components, mostly pairs, two large blocks (706, 722).
  Both branches feed the same coverage-first affinity selection and
  battery as the control.

## Results — 16k

### Labeled battery: every candidate is a singleton, and no candidate rings

Top family-affinity candidates, both branches (class-permutation null,
n_perm = 200; full JSON in `target_run/family_battery.json`):

| family | branch | cluster | members | affinity | circ | p |
|---|---|---|---|---|---|---|
| weekday | spectral | 341 | **1** (feat 7972) | 0.08 | — | — |
| weekday | graph | 890 | **1** (feat 1654) | 0.46 | 0.041 | 0.726 |
| month | spectral | 903 | **1** (feat 7107) | 0.23 | 0.024 | 0.104 |
| month | graph | 4666 | **1** (feat 5842) | 0.78 | 0.090 | 0.139 |

Every top-3 candidate in every family × branch cell is a **single
feature**. Their identities (top fired tokens, from
`check_gemma_multimember.py`):

- **feat 5842** (month/graph primary): fires near-uniformly on *all
  twelve* month tokens (" November"×143 … " August"×78) — one
  month-of-year feature.
- **feat 1654** (weekday/graph primary): all seven weekday tokens plus
  date punctuation — one day-of-week feature.
- **feat 7972** (weekday/spectral): temporal deixis (" today",
  " yesterday", " tonight") that also catches weekdays.
- **feat 7107** (month/spectral): numeric-date context (digits,
  separators, month names) — dates generally.

A single feature's cluster-restricted reconstruction is a rank-1
intensity ray, and the figures confirm exactly that: classes fully
mixed along one axis, second PC at 10⁻⁴ the scale of the first
(`figures/phase0-gemma/`). The circ scores are noise on a degenerate
plane, as the p-values say.

### Nothing was passed over: no multi-member family cluster exists

Rerunning the coverage-first affinity rule restricted to ≥2-member
clusters (`target_run/multimember_check.json`): the best multi-member
cluster in any family × branch cell has affinity ≤ 0.0054 — three
orders of magnitude below the singletons (0.46/0.78). These are diffuse
date-context clusters whose labeled firings are too rare to battery
("no labeled tokens" at min 200). The affinity selection did not miss a
ring-shaped cluster; there is none to find.

### Supervised probe: weekday null is representation-level; month has a ring skeleton

The loophole in a discovery null: per-class features (a " Monday"
feature, a " June" feature, …) could exist but sit too orthogonal to
cosine-cluster, hiding a real ring from both branches. Probed with
labels used verification-only (`check_gemma_supervised_ring.py`): rank
every feature by class-selective firing, take the top 2 per class,
battery the union through the exact production pipeline
(`target_run/supervised_ring.json`).

- **Weekday: there are no day-selective features.** The "most
  selective" features for every one of the seven classes are the same
  two generic day-of-week features (1654, 14886) at selectivity
  0.06–0.08. Union of 3 features: circ 0.153, p 0.080. Nothing to form
  a ring from — a representation-level null.
- **Month: partial splitting, and the split parts are in calendar
  order.** Ten of twelve classes share generic month/date features
  (5842, 5081, 9915, …) at selectivity 0.05–0.12, but the
  high-frequency/ambiguous months get their own: **May** (feats 5643
  sel 0.53, 2269 sel 0.32 — "may" modal / "May" name), and weaker
  class-leaning features for Mar (11177), Sep (13225), Oct (14994), and
  an Oct/Nov/Dec span (13335). The 11-feature union scores **circ
  0.524, p = 0.0149** on PCs 3/4 — and the figure
  (`figures/phase0-gemma/month_supervised_union.png`) shows exactly
  what that number can and cannot mean: not a ring but a **skeleton** —
  Jan/Feb, Mar, May, Oct, and Nov/Dec form distinct class spokes in
  correct cyclic order while Jun–Sep collapse into the center. The
  p-value is essentially the probability of ~6 distinguishable spokes
  landing in calendar order by chance ((6−1)!/2 arrangements → ≈0.017),
  which is what the class-permutation null tests here.

Two caveats bind this last result. First, member selection used the
true labels on the same tokens the battery scores (the angle fit is
5-fold held-out, and the class-permutation null protects the *cyclic
order*, which selection does not inject — but a preregistered
split-half selection would be cleaner if this ever needs to be more
than suggestive). Second, it is supervised-only: no unsupervised
pipeline surfaces this candidate, so it is a statement about what
exists in the dictionary, not about what Phase-1 discovery can find.

### Unknown-cluster scan (surfacing, BH-calibrated)

139 candidates entered the scan across both branches (spectral 73,
graph 66 — five graph components duplicated spectral member sets and
were scanned once); **121 tested, 18 gated out**
(all `insufficient_cofire_tokens`), **BH flagged: none**. The minimum
p is 0.0364 — `graph:1223`, a size-2 pair (feats 2033/14846, contrast
0.382) at its n_null=54 p-floor of ~1/55 — against a BH rank-1
threshold of 0.05/121 ≈ 4.1×10⁻⁴. Exploratory top-contrast candidates
below it (`spectral:507`, a 6-feature cluster, contrast 0.318 p 0.070;
three size-2 spectral pairs at p 0.091) are unremarkable, and none
contains any supervised-skeleton feature.

The membership cross-check explains the miss in complementary ways per
branch. On the **graph** branch all 7 skeleton features (the month
spokes and weekday family features) are **singletons** — no neighbor
survives the τ=0.5 prune, so no skeleton member ever enters the
candidate set (the same structural exclusion quantified at 65k). On
the **spectral** branch the forced 1000-way partition does the
opposite: 6 of 7 skeleton features land in giant mixed clusters of
size 1417–2155 (three of them — 5643, 13335, 13225 — share cluster
308, size 2155, tested at contrast 0.260, p 0.21), where a handful of
ring members among thousands is diluted past detection; the seventh
(11177) is a spectral singleton. One branch isolates the ring's
members; the other buries them.

**Power ceiling, recorded**: at 100 frequency-matched null draws the
empirical p-floor is 1/101 ≈ 0.0099, and at search width 139 the BH
rank-1 threshold is 0.05/139 ≈ 3.6×10⁻⁴ — a lone true ring at the
p-floor cannot clear BH; ~28 co-flagged candidates would be needed.
The scan is a surfacing instrument (verdicts belong to the labeled
battery), so top-contrast candidates are reported as exploratory
regardless of BH. A full-resolution rerun (~3000 draws) is a parameter
change if ever warranted.

### Engineering note: the scan OOM

The first scan run was OOM-killed at 62 GB RSS: member selection
densified all 4M tokens × |members| *before* gating (35 GB for the
2192-member spectral blob, repeated per null draw). Fixed by
gate-then-densify — sparse per-token member-firing counts, subsample to
max_tokens, densify only kept rows — bounding peak memory at
max_tokens × |members| everywhere (commit df25d88). A second lesson for
long jobs: launch `python -u`; the block-buffered log died empty with
the process.

## The width_65k reroute (2026-07-17, overnight)

### Setup deltas

- **Artifact**: `layer_22_width_65k_l0_medium` (d_sae 65536, same hook,
  25.6×), pinned with W_dec sha256 + repo sha by
  `scripts/pin_gemma_65k_artifact.py` (PASS). Method-matched decoder
  geometry, both widths with the same code (which independently
  reproduced the quoted 16k numbers): the 65k bulk is *more* orthogonal
  — median max-neighbor cosine 0.196 vs 0.218, >0.7-neighbor fraction
  2.1% vs 4.8% (>0.5: 4.7% vs 9.9%) — though the absolute count of
  close-neighbor features grows (~790 → ~1350). 25.6× does not create a
  denser fan bulk.
- **Harvest**: identical conditions and, because the Pile stream is
  deterministic, the *identical* 3,999,930 tokens. Mean L0 61.6,
  zero-code fraction 0, 62,495/65,536 features fired (95.4% vs 98.9% at
  16k — wider dictionaries carry more rare features at 4M tokens).
- **Branches**: kNN-graph only. The spectral branch needs a 65k² dense
  similarity + eigh (~17 GB) and was already degenerate at 16k's much
  friendlier geometry; the coact diagnostic needs the same F×F dense
  matrix. Both scoped out with printed skip lines, not silently
  truncated. `knn_graph_clusters` was row-chunked for this width
  (bit-exact at any chunk size; commit 16e9a9a).
- Graph clustering: 62,730 components, median 1, max 1215, 264
  components with ≥2 members (the scan width).

### Labeled battery — 65k: still singletons, but hotter ones

Top family-affinity candidates (all n_members = 1; full JSON in
`target_run/family_battery.json`, figures in the 65k `target_run`):

| family | cluster | affinity | circ | p |
|---|---|---|---|---|
| weekday | 25055 | 0.78 | −0.013 | 0.522 |
| month | 26946 | 0.87 | −0.040 | 0.498 |
| month | 46137 | 0.77 | 0.039 | 0.478 |
| month | 52990 | 0.75 | 0.034 | 0.289 |

Affinities jump from 16k (weekday 0.46 → 0.78; month 0.78 → 0.87, and
*three* month singletons above 0.75 where 16k had one) — more
family-selective single features exist, but they still do not cluster:
their mutual decoder cosines sit below the τ = 0.5 prune (see the split
geometry below), so the graph never joins them.

### Supervised probe — the split completes for months

Same probe, same selection rule (`check_gemma_supervised_ring.py`,
top-2 per class by selectivity; `target_run/supervised_ring.json`):

- **Month: all 12 classes now have their own top feature** (12 distinct
  top-1 features), nine at selectivity ≥ 0.31: May 0.73 (from 0.53 at
  16k), Jun 0.61, Dec 0.45, Aug/Nov 0.36, Jan/Mar/Jul 0.33, Sep 0.31;
  Feb and Oct trail at 0.11–0.12. The frequency-order splitting story
  is confirmed in vivo — the months that had "just begun" splitting at
  6.4× have all split at 25.6×.
- **Weekday: splitting has begun, as adjacent-day arcs.** Class-
  selective features now exist (16k had none) but span *pairs/runs of
  adjacent days* — one for Sun/Mon (54671, sel 0.41/0.34), one for
  Tue–Fri (27617), one for Fri/Sat (19491, 0.23/0.28). Union circ 0.41
  (up from 0.15) at p = 0.23: an ordered gradient, not yet a ring.
- **But the month union's activation-space calendar order is *gone***:
  circ 0.546, p = 0.289 (16k skeleton: circ 0.524, p = 0.0149). The
  figure (`figures/phase0-gemma/month_supervised_union_65k.png`)
  explains: the union is a **star of near-orthogonal per-class rays** —
  spokes span many angles (high circ) but their angular arrangement in
  any PCA plane no longer tracks the calendar. The 16k order was a
  property of *partially split* features still crowding a low-dim
  family subspace.

### Split geometry — the ring is in the decoders, below the thresholds

Decoder-level test (`scripts/check_gemma_split_geometry.py`,
`data/phase0/split_geometry.json`): take the top-1 feature per class,
compute pairwise decoder cosines, compare adjacent-in-cycle vs
non-adjacent pairs, and test calendar order of the features' angles in
their own PC1/2 plane (Fisher–Lee circular correlation, 20k-permutation
null).

| width | family | distinct top-1 | adj mean cos | non-adj mean cos | adjacency p | PC1/2 \|r\| | angle p |
|---|---|---|---|---|---|---|---|
| 16k | month | 8/12 | −0.038 | −0.044 | 0.39 | 0.21 | 0.10 |
| 65k | month | **12/12** | **+0.093** | **−0.048** | **1.5×10⁻⁴** | **0.59** | **3.5×10⁻⁴** |
| 65k | weekday | 3/7 | −0.117 | −0.114 | 0.50 | — | — |

**The 65k month dictionary contains a decoder-space ring**: adjacent
months are systematically more similar (max pair cosine 0.32), and the
12 decoder directions' PC1/2 angles wind in calendar order
(`figures/phase0-gemma/month_decoder_ring_l22_65k.png` — months 1–3 and
5–11 trace a clean cycle; the two off-ring points are interpretable:
May, whose dominant feature carries the "may"-modal ambiguity, and
Jan, pulled toward the center). PC1/2 hold only 29% of the 12-vector
variance — the ring is a low-cosine correction on top of near-orthogonal
per-class axes, which is exactly why it is invisible to discovery: the
kNN graph prunes at cos ≥ 0.5 against a max family cosine of 0.32, and
every activation-space PCA plane is dominated by the orthogonal part.
The 16k ↔ 65k inversion is clean — at 16k, calendar order lived in
activation space (partially split features, no decoder-side signal);
at 65k it lives in decoder space (fully split features, no
activation-plane signal).

Caveats, same discipline as the 16k skeleton: member selection is
label-derived (selection is per-class and order-blind, so cyclic order
is not injected, but split-half selection would be cleaner); the angle
test has 12 points; four family × width cells were tested (a Bonferroni
×4 leaves both month-65k statistics far under 0.05); and this is a
statement about what exists in the dictionary, not what unsupervised
discovery recovers.

### Unknown-cluster scan — 65k (surfacing, BH-calibrated)

264 multi-member graph components entered the scan; **114 tested, 150
gated out** (all `insufficient_cofire_tokens`), **BH flagged: none**.
The minimum p is 0.0396 — the size-1215 giant component (`graph:1`,
contrast 0.314), ~90× above the BH rank-1 threshold of 0.05/114 ≈
4.4e-4. The strongest small candidate is a size-2 pair (`graph:11034`,
feats 13399/24861, contrast 0.476) sitting exactly at the pair p-floor
of ~1/22. Nothing else is below p 0.11.

The membership cross-check closes the loop on *why* the scan cannot see
the supervised ring: **14 of the 15 skeleton features (12 month top-1 +
3 weekday) are singletons in the τ=0.5 kNN graph** — max family cosine
0.32 means no ring member has any neighbor above the prune, so each is
an isolated node and never enters the scan at all. The 15th (48450,
a month top-1) landed in a size-2 component that was gated out for
insufficient co-fire tokens. No tested candidate contains any skeleton
feature. The discovery null at 65k is therefore not a power shortfall
on the ring — the ring's members are structurally excluded from the
candidate set by the same low-cosine geometry the decoder-level tests
quantified.

Power note: at 65k only 21/100 size-2 null draws co-fire (vs 54/100 at
16k), so pair p-floors are correspondingly higher (~1/22); the scan
width is 264.

## Interpretation and gate consequence

The Engels mechanism, read across the control and both target widths:
**a cyclic family gets ring geometry only when the dictionary splits it
into per-class features; splitting is an overcompleteness effect that
arrives per-class in frequency order; and the ring's *geometric
strength* is itself a function of where the dictionary sits on the
splitting trajectory.** Bloom's GPT-2 SAE (32×, high-cosine geometry:
median max-neighbor 0.547) splits weekdays into per-day features whose
decoders correlate strongly enough to cluster → the ring is
recoverable unsupervised (control). Gemma-scope at 6.4× allocates one
feature per family; at 25.6× the month family fully splits and its
decoder directions demonstrably carry cyclic structure — but at
cosines (≤ 0.32) that no Engels-fidelity clustering threshold reaches,
in a bulk that is *more* orthogonal than at 6.4×, not less. The
16k → 65k prediction ("more months split, and perhaps weekdays") was
confirmed on the splitting axis and refuted on discoverability: gemma's
wider dictionary splits *more orthogonally* than GPT-2's, so the ring
never rises above the clustering noise floor.

Per the design §Phase 0 gate, the discovery verdict at both widths is
the scoped null: **"no rings recoverable from this SAE at this depth at
demonstrated power"**, control passing, so the null is a statement
about the artifact + method, not about power. But the H1 artifact
statement now splits from the discovery statement: **the month ring
exists in the 65k dictionary** (adjacency p ≈ 1.5×10⁻⁴, angle-order
p ≈ 3.5×10⁻⁴), it is simply below post-hoc blockification's reach.
Explicitly NOT established, as before: "gemma flattens token-level
geometry" — the decoder ring is direct evidence against the strong
reading.

What this buys the next phases:

1. **Phase 1 gains a sharpened hypothesis and warm-start targets.**
   Post-hoc blockification thresholds pairwise cosines; a *trained*
   block-sparse crosscoder does not — it can allocate a block to the
   month subspace directly if that subspace earns its reconstruction
   budget. The 12 top-1 month features (ids in
   `split_geometry.json`) are the natural warm-start/validation set:
   if BSC's discovered blocks capture the month-family subspace that
   clustering cannot, the {block} × {cross-site} cell earns its keep on
   exactly the structure this phase proved present-but-unreachable.
2. **The reroute chain's next stops (depths 9/17/29, Qwen) are now
   optional refinements, not the mandated next step** — the reroute
   question ("is there ring structure in *some* gemma artifact?") got a
   positive answer at layer 22/65k, decoder-side. Whether to spend
   another night bracketing depths, or proceed to Phase 0.5
   (cross-layer coherence needs no rings, and the machinery is
   control-validated), is a9's call in the morning.
3. **Weekday remains the in-vivo frontier marker**: adjacent-day arcs
   at 65k where 16k had nothing — a family caught one notch earlier on
   the same trajectory months completed. A 262k follow-up would test
   whether weekday closes its ring as months did (and whether months'
   decoder ring *strengthens* or dissolves further toward orthogonality
   with more splitting).
