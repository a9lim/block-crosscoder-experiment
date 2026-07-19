# Phase 0.5: cross-layer coherence of the calendar code (layers 9/17/22/29 @ 65k)

**Status: GATE PASSED (2026-07-17, overnight) — and in the BSC-native
form.** The month-family code corresponds linearly across depths —
held-out layer-A→layer-B code-map R² up to 0.90 across 13 layers of
gemma-3-4b, all canonical correlations ≥ 0.96 for 9↔22 — while the
per-depth decoder frames rotate to *chance-level* raw-basis alignment
(principal-angle p ≈ 0.33–0.52 against a random-feature null). Frames
transform; the code persists. That is precisely the premise the
block-sparse crosscoder architecture encodes (per-site frames
`LayerSubspace.basis`, one shared code), observed in vivo before any
training. The late pair 22→29 additionally span-matches in the raw
basis (p 0.001). En route, the depth bracket rewrote the ring story:
**activation-space calendar rings live early** — at layer 9 the
supervised weekday union scores circ 0.981 and month 0.845, both on
the *top* PCA plane at the permutation floor — and layer 22, our
Phase-0 anchor, turns out to be the ring-visibility minimum, not the
representative depth.

Run artifacts: four 65k CodeStores under `/data/stores/bcc-phase0/`
(`gemma3_4b_l{9,17,22,29}_65k_pile`, identical 3,999,930-token Pile
stream — asserted, not assumed), per-store `target_run/` probe caches,
`data/phase05/cross_layer.json`, `data/phase0/split_geometry.json`
(all jobe); figures in `figures/phase05/` and `figures/phase0-gemma/`.
Scripts: `scripts/check_phase05_cross_layer.py` (the gate metrics),
plus the Phase-0 harvest/probe/split-geometry scripts re-pointed at
the new depths.

## Setup

- **Depths**: layers 9 / 17 / 22 / 29 of 34 (26% / 50% / 65% / 85%),
  all `gemma-scope-2-4b-pt-res` `width_65k_l0_medium` (coverage
  verified per design §Phase 0.5 — these four depths are what Gemma
  Scope 2 ships for 4b). Layer-22 store from Phase 0; the other three
  harvested overnight under identical conditions (mean L0 55.1 / 62.2
  / 61.6 / 61.6 at 9/17/29/22; zero-code fraction 0 everywhere).
- **Family subspaces per depth**: supervised top-1-per-class features
  (labels verification-only, 2026-07-15 ruling) from the same probe
  used at 22. Depths with < 4 distinct members drop out of subspace
  tests (weekday@22, with 3).
- **Gate metrics per depth pair** (none are per-latent cosine
  matching — review finding 20): (i) principal angles between decoder
  subspaces vs a rank-matched random-feature null (1000 draws from
  each depth's ≥20-fire pool); (ii) paired-token CCA and
  orthogonal-Procrustes held-out R² vs shuffled-pairing nulls (200
  draws); (iii) out-of-sample affine code map fit on half the labeled
  tokens, held-out R² on the other half, both directions. Paired
  tokens: 5,121 month-labeled / 518 weekday-labeled positions of the
  shared stream.

## The depth trajectory: rings live early

Supervised-union battery per depth (class-permutation null, n_perm
200; per-class tables in each store's `supervised_ring.json`):

| layer | family | union | distinct top-1 | top-1 sel range | circ | p | plane |
|---|---|---|---|---|---|---|---|
| 9 | weekday | 9 | 6 | 0.22–0.87 | **0.981** | **0.005** | **(0,1)** |
| 17 | weekday | 5 | 4 | 0.01–0.33 | 0.474 | 0.075 | (3,4) |
| 22 | weekday | 6 | 3 | 0.09–0.41 | 0.408 | 0.23 | (1,2) |
| 29 | weekday | 8 | 6 | 0.06–0.66 | 0.451 | 0.27 | (2,3) |
| 9 | month | 18 | 11 | 0.26–1.00 | **0.845** | **0.005** | **(0,1)** |
| 17 | month | 7 | 6 | 0.06–0.62 | 0.270 | 0.005 | (1,2) |
| 22 | month | 18 | 12 | 0.11–0.73 | 0.546 | 0.289 | (0,1) |
| 29 | month | 22 | 12 | 0.14–0.97 | 0.620 | 0.010 | (0,1) |

Three observations. **(1) Layer 9 is where the rings are.** Both
families are substantially split at 9 (weekday to selectivity 0.87,
month to 1.00 — e.g. a pure " March" feature), and both unions ring
on the dominant plane at the permutation floor. **(2) The layer-17
dictionary undersplits both families** — weekday max selectivity
0.33, month 0.11 outside May — which matters below. **(3) Depth
visibility is non-monotone and our Phase-0 anchor sat at the
minimum**: month order is significant at 9, 17 (buried plane), and
29, and null only at 22. The 65k Phase-0 verdict ("order lives
decoder-side, activation plane is an orthogonal star") is a
layer-22 fact, not a gemma fact.

## Cross-layer coherence (the gate)

Month (5,121 paired tokens; member active fraction 0.91 / 0.44 /
0.81 / 0.65 at 9/17/22/29):

| pair | span mean cos² (null, p) | CCA mean (p) | Procrustes R² | code map R² A→B / B→A (p) | verdict |
|---|---|---|---|---|---|
| 9→17 | 0.044 (0.040, 0.33) | 0.299 (0.005) | 0.051 | 0.136 / 0.097 (0.005) | BSC-native (weak) |
| 9→22 | 0.020 (0.023, 0.43) | **0.905** (0.005) | 0.659 | **0.834 / 0.902** (0.005) | **BSC-native** |
| 9→29 | 0.013 (0.016, 0.43) | 0.873 (0.005) | 0.636 | 0.736 / 0.829 (0.005) | BSC-native |
| 17→22 | 0.027 (0.029, 0.36) | 0.319 (0.005) | 0.030 | 0.101 / 0.161 (0.005) | BSC-native (weak) |
| 17→29 | 0.012 (0.021, 0.52) | 0.300 (0.005) | 0.026 | 0.080 / 0.171 (0.005) | BSC-native (weak) |
| 22→29 | **0.188** (0.013, **0.001**) | 0.897 (0.005) | 0.778 | 0.883 / 0.895 (0.005) | **spans match AND correspond** |

Weekday (518 paired tokens; 22 excluded at 3 members): every tested
pair is significant-correspondence-without-span-match too — top-2
CCA 0.95/0.91 even for 9→29 — with lower overall R² (0.12–0.56),
consistent with a code that is only ~2-dimensional this early in its
splitting trajectory (arcs, not a full ring).

The reading. The design's gate language enumerated "spans match AND
positions correspond" (pass) and "frames persist, codes transform"
(stage-blocks). The dominant observed cell is the one it did not
enumerate: **positions correspond while raw-basis spans don't** — and
this cell *supports* the shared-code premise in its strongest form,
because an invertible linear map between codes is exactly what
per-site frames absorb: if z_B = M z_A, then one shared code z_A
reconstructs site B through the re-framed decoder M D_B. A shared
code with per-layer frames is not merely viable here; it is the
*only* representation in the 2×2 that captures what the model does
with this family (the raw-basis subspace rotates with depth — mean
cos² at chance — so any single-frame cross-layer dictionary would
have to duplicate the family per depth). The 22→29 pair, where spans
also match, says late-stream frames have mostly stopped rotating.

The 9↔22↔29 triangle is mutually strong while every pair through 17
is weak — and layer 17's own probe explains it: that dictionary
barely splits the families (month top-1 selectivity 0.06–0.12
outside May; member active fraction 0.44 on labeled tokens). The
correspondence instrument measures the model *through each layer's
dictionary*, so a weak dictionary bounds it from below one-sidedly:
high values prove correspondence; low values at 17 indict the 65k
SAE at 17, not the residual stream — the 9→22 R² of 0.83+ shows the
information rides *through* layer 17's stream regardless of what its
SAE captured. (A trained BSC brings its own code, which is the
point.)

Caveats, standing discipline: member selection is label-derived
(per-class and order-blind; the correspondence statistics never see
class identity, and the shuffled-pairing nulls destroy exactly the
claimed structure); month n=5,121 but weekday n=518; all statements
are about what these particular pretrained dictionaries + the stream
expose, not about all of gemma's features.

## Decoder-ring geometry by depth

Same instrument as Phase 0 (adjacency contrast + Fisher–Lee angle
order of top-1 decoder vectors; the 22/16k and 22/65k rows reproduce
the committed numbers exactly, a free regression check):

| layer | family | distinct top-1 | adj mean cos | non-adj mean cos | PC1/2 explained | \|r\| | angle p |
|---|---|---|---|---|---|---|---|
| 9 | weekday | 6/7 | **+0.124** | **−0.107** | 0.60 | **0.886** | **0.0049** |
| 17 | weekday | 4/7 | −0.041 | −0.020 | 0.80 | 0.384 | 0.20 |
| 29 | weekday | 6/7 | −0.052 | −0.055 | 0.52 | 0.097 | 0.57 |
| 9 | month | 11/12 | −0.021 | +0.005 | 0.34 | 0.287 | 0.045 |
| 17 | month | 6/12 | −0.073 | −0.022 | 0.68 | 0.100 | 0.37 |
| 22 | month | 12/12 | +0.093 | −0.048 | 0.29 | 0.592 | 0.00035 |
| 29 | month | 12/12 | **+0.129** | −0.028 | 0.25 | 0.417 | 0.012 |

The two families run the trajectory in opposite directions.
**Weekday is an early-layer object**: at layer 9 it rings in *both*
spaces (activation circ 0.981 + decoder |r| 0.886) and fades
monotonically with depth in both. **Month's decoder order arrives
late**: activation-visible at 9 with only marginal decoder order
(|r| 0.287, p 0.045), then decoder-side order strengthens through
22 (0.592) and persists at 29 (0.417) while the activation plane
loses it. With this many cells the two marginal p's (0.045, 0.012)
are exploratory; the layer-9 weekday and layer-22/29 month results
survive any reasonable correction.

The discovery-relevant constant: even the strongest ring anywhere
(layer-9 weekday) has max adjacent decoder cosine **0.27** — below
the τ = 0.5 Engels prune at every depth. Gemma's calendar rings are
never a high-cosine fan at 65k, whatever the depth.

## Is the layer-9 ring discoverable? (bonus discovery run)

**No — and for the same structural reason as at 22.** The full Phase-0
discovery battery + unknown scan, re-run on the layer-9 store where
both rings are activation-visible at the permutation floor:

- Battery top candidates are again high-affinity **singletons**
  (affinity 0.95–0.98) — no multi-feature candidate survives to the
  ring stage. The τ = 0.5 graph shatters the dictionary into 58,908
  components (largest 6,393).
- Unknown scan over 175 graph clusters: **78 tested / 97 gated out,
  BH-flagged: none.** Minimum p is 0.0099 on the giant component
  (scored contrast 0.385, n_null 100) — whose irreducibility is
  0.038, i.e. a mixed blob that decomposes, not a ring.
- Membership closure: **15/17 layer-9 skeleton features are τ = 0.5
  graph singletons** (weekday 5/6, month 10/11; the other two sit in
  size-2 components), and no two skeleton features share a component.
  The clustering front-end atomizes the family before any ring
  statistic sees it — exactly the max-adjacent-cos 0.27 < τ geometry
  predicts.

So the discovery gap is **depth-general**: even at the depth where
the ring is strongest in *both* spaces, cosine-threshold clustering
structurally excludes it from the candidate set. H1's statement —
post-hoc blockification cannot reach the ring; a method that learns
subspaces jointly with the code can — is now supported at every
probed depth, not just the Phase-0 anchor.

## Gate consequence

1. **Phase 0.5 gate: PASS.** Spans-and-positions holds outright for
   22→29; the remaining strong pairs hold in the
   frames-rotate-code-persists form that per-site frames absorb by
   construction. The shared-code assumption has legs on real
   cross-depth structure, measured with three independent instruments
   against matched nulls. Proceed to Phase 0.9 (1b dress rehearsal)
   with the design's ladder unchanged.
2. **Phase-1 site selection gains an empirical constraint.** The 8
   harvest sites should bracket depth *including the early stream*
   (layer-9-equivalent), where token-level geometry is
   activation-visible and splitting is already strong — and should
   not assume mid-depth dictionaries are representative (17 is a
   cautionary tale for judging structure through a single site).
   The 22→29 span match suggests late sites are partially redundant;
   early↔late is where the frame rotation, and hence the BSC's
   advantage over single-frame methods, actually lives.
3. **Warm-start/validation targets extended to a depth ladder**: the
   per-depth month top-1 feature ids are in `cross_layer.json`
   (members block), giving Phase 1/2 a cross-depth family to check
   discovered blocks against — including the linear maps between
   depth codes as a *quantitative* target (a discovered month block's
   shared code should reproduce ≈0.8–0.9 of the cross-depth mapped
   variance).
4. **The 262k fallback stays unspent.** The gate passed; wider-width
   ring-closure (weekday) remains an optional science add, not a
   dependency.
