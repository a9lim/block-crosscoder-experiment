# Interim artifact analysis — geometry of everything on disk (2026-07-17)

**Scope.** Analysis-only sweep over every artifact produced through Phase
0.9.5, run while Phase 1 waits on the 4 TB NVMe. No new training; one new
*measurement* (the calendar probe: a fresh 8M-token labeled harvest encoded
through existing checkpoints). Exploratory throughout — nothing here is gate
evidence, and every trained-model claim inherits the rehearsal caveats
(gemma-3-1b, 6 sites, G=1024/4096, **4M training tokens**; Phase-1 scale is
~10× that on a 4b stream). Statistics quoted at their measured values;
permutation p-values floor at 1/20001.

Figures in `figures/interim/` (regeneration: `scripts/analysis/fig_*.py`;
extraction npz on jobe under `/data/runs/bcc-analysis/`, mirrored to
`data/analysis/npz/`). Interactive 3D: `probe_block23_3d.html`,
`rings_cross_layer_3d.html`, `geo_share_3d.html`.

---

## A. The trained BSC discovered the month manifold as a single block

The strongest finding of the sweep, and an unplanned preview of H1's
artifact statement at 1b scale.

Probe: 8M fresh FineWeb-Edu tokens (stream head skipped past the store
slice), whitened with the store whitener, encoded through the 0.9.5
checkpoints; 9,550 capitalized month tokens, 923 weekday tokens, 60k
background positions (lowercase forms excluded — modal "may" etc.; see
Methods).

**Block 23 of the ratified winner (lr 1.2e-3 cosine, λ=1e-3, seed 0) is a
month block carrying a clean calendar ring in its 4-dim code:**

- fires on 53% of month tokens vs 0.2% of background (selectivity z = 14);
- the top plane of its class means holds **97%** of class-mean variance,
  and all **12/12** calendar-adjacent pairs are angularly adjacent —
  the angular order *is* calendar order (permutation p < 5e-5);
- split-half stable: 12/12 in both halves, circular correlation 1.000;
- class angles descend monotonically Jan 118° → Nov −172° (one clean
  cycle; `probe_block23_ring.png`, `probe_block23_3d.html`).

**The block's rotating frames track the ring through depth.** In the raw
whitened stream, the month ring is clean early and fades late (top-plane
class-mean test: 12/12 at L7/L10/L13, 10/12 at L17, 3/12 at L20, 4/12 at
L22) — gemma-3-1b **rings early**, echoing the 4b Phase-0.5 depth rewrite.
But projected into block 23's per-site 4-dim decoder subspaces, the ring is
**12/12 at every site** with ≥94% of class-mean variance in-plane —
including L20/L22 where the naive top plane has lost it
(`probe_ring_depth.png`). Block 23's energy share is spread across all six
sites (0.15/0.08/0.16/0.22/0.19/0.20) — unusually flat against a late-heavy
dictionary (§D). This is "frames rotate, the code persists" as a concrete
discovered object, and it is exactly the artifact the saklas seam wants —
a Phase-2 export of block 23 would be the first *real* discovered manifold
(the 0.9 `toy_manifold` export was block 54, chosen blind).

**Replication and its limits (read soft):**

| run | screen-top month block | ring, top-1 | top-2 joint |
|---|---|---|---|
| winner (lr 1.2e-3, seed 0) | 23 | **12/12** (plane 97%) | 12/12 |
| G=4096 k=32 (lr 3e-4) | 244 | 10/12 (96%) | 12/12 |
| base (lr 3e-4, seed 0) | 23 | 5/12 (95%) | **12/12** |
| renorm (lr 3e-4) | 140 | 10/12 (97%) | 10/12 (12/12 top-3) |
| winner seed 1 | 60 | 7/12 (99%) | 3/12 |

Same-seed runs concentrate month structure in the *same block index* (23 —
deterministic data order + shared init), and the ring is present in some
block combination in every arm. But **single-block consolidation varies
with optimizer strength and seed**: the ratified winner config is the one
that fully consolidates; weaker lr splits the ring across two blocks;
seed 1 shows only partial single-block order. One-seed caution stands —
and "does the known ring consolidate into one block" is a cheap,
well-defined eval probe worth carrying into Phase 1.

**Weekday: capture without ring geometry.** Block 640 fires on **100%** of
weekday tokens vs 0.3% background (z = 35) — a dedicated weekday block —
but its code does not order the week (3/7 adjacent, p ≈ 0.43;
`probe_weekday_null.png`). With 923 probe tokens and 4M training tokens,
absence of the ring is not evidence of absence in the stream; the honest
statement is family capture, no ring, at this scale.

## B. The scalar baseline carries the same information without individuating it

The matched scalar crosscoder (G=4096, b=1, k=64 — the FVU winner, 0.345
vs 0.412 pooled) represents months **coarsely at the unit level**: the
top-1 feature per month collapses to only **4 distinct features**, one of
which claims Mar–Sep entirely. The ring survives as *population* structure
(top-24 month-selective features' response space: 12/12, p < 5e-5, top-2
plane 85%) but no scalar unit owns it. Same store, same token budget: the
block unit gets the manifold as one exportable object; the scalar
dictionary smears it across merged features. This is the qualitative H3
contrast the design hoped for — while the *quantitative* H3 question
(FVU at matched latent-L0) still favors scalar at rehearsal scale and
stays deferred to 4b.

## C. Cross-site frame geometry of the trained dictionaries

First look at the learned weight-space geometry (`geo_frame_rotation.png`,
`geo_dimensions.png`, `geo_share_3d.html`):

- **Frames rotate smoothly and far above chance.** Median principal cosine
  between a block's per-site frames: 0.93 at the shortest gap (L20↔L22),
  decaying to 0.18 across the full span (L7↔L22); shuffled-block null 0.05.
- **Rotation is depth-localized, not gap-uniform**: the site×site matrix
  shows the L13→L17 crossing is the rotation hotspot (0.49, vs 0.71 for
  the *longer* L17→L22 span); deep frames (L20↔L22, 0.93) are nearly
  frozen. The scalar arm shows the same dip — this is a property of
  gemma's stream (a mid-network representational shear zone), not of the
  block architecture. It rhymes with the 4b story, where L17's dictionary
  was the cross-layer odd-one-out.
- **Blocks reuse most of their subspace across depth**: the 6-site stacked
  decoder (24 rows max) has median effective dimension **6.3** — far from
  both the one-shared-subspace floor (4) and the independent-frames
  ceiling (24). Renorm raises it to 8.1 (more depth-uniform energy engages
  more directions); G=4096 sits at 7.1.
- **Scalar features rotate too**: median effective dimension 1.6 of 6 for
  the per-site direction stack. Scalar crosscoders already do implicitly
  (and diffusely) what the BSC does explicitly with a frame per site.
- **The encoder leaves its transpose-tied init far behind**: median
  principal cosine between span(E_g^s) and span(D_g^s) is only 0.46
  (p10 0.20) — the summed-per-site encoder learns something other than
  the decoder's mirror, presumably interference cancellation across
  sites and blocks (the crosscoder analogue of SAE encoders learning
  denoising weights). Meanwhile eval-side contribution-energy shares
  match weight-space shares to ~3 decimals — the Gram constraint makes
  the baked `share` statistic functionally meaningful, validating the
  export contract.

## D. Depth allocation: the shrinkage whitener explains the late-heavy dictionary — mechanistic support for F7

Baseline dictionaries put **84%** of decoder energy on L17/L20/L22; blocks
peaking at L7/L10/L13 are nearly absent (argmax histogram 2/0/2/291/89/640
of 1024). The mechanism is visible in the whitener itself
(`whitener_spectra.png`): with the pinned ridge (λ = mean eigenvalue), the
retained-variance fraction after shrinkage averages **~6% at L7 vs ~32% at
L22** (the F7 scalars, 4.12→1.78, are exactly the inverse-rms of this), so
pooled whitened MSE cares ~5× more about deep sites. Early layers pay more
because their covariance is more top-eigenvalue-dominated (massive-activation
structure), which drags the mean-eigenvalue ridge far above their bulk
spectrum.

The renorm arm confirms the causal arrow: equalizing per-site power
flattens the energy budget to near-uniform (0.14–0.18 across all six
sites; argmax histogram 331/70/128/152/210/133) and reverses the per-site
FVU allocation deep→shallow at wash pooled FVU
(`cal_renorm_allocation.png`, `geo_share_heatmap.png`).

**Implication for the pinned F7 decision (a9 leans renorm):** this analysis
strengthens the renorm case — without it, L7–L13 of the 4b stream would be
close to unmodeled, and the Phase-0.5 depth rewrite says early sites are
where activation-space ring structure lives. One soft counterweight: at
matched lr the renorm arm's month ring consolidated slightly less cleanly
than baseline's (10/12 vs 12/12-in-two-blocks; confounded with lr, n=1).
Worth one line in the Phase-1 monitoring plan, not a lean-changer.

## E. Dictionary-scale effects and the packing signature in vivo

- **Code anisotropy** (eval second moment): G=1024 blocks use their 4 dims
  broadly (median effective dim 3.0); at G=4096 codes drift scalar-ward
  (median 2.35, 9.1% of blocks below 1.5) — with more blocks, the
  dictionary spends units on narrower structure
  (`geo_code_anisotropy.png`).
- **A packing clique exists at G=4096**: blocks {146, 1758, 2251, 3228}
  fire near-identically (pairwise Jaccard 0.90–0.93, matched frequencies
  0.0039–0.0040, similar late-heavy shares). This is the Phase-−1 packing
  signature in production form — 16 decoder dims tiling one co-active
  structure. At G=1024 the max Jaccard is 0.58 (no cliques). The Phase-2
  `share`-export packing flag is confirmed as a real production concern.
- **No dead blocks in any run** (threshold-mode eval floor > 1e-6
  everywhere); θ transfer holds (mean L0 16.00/32.00/64.03 vs targets
  16/32/64); L0 distributions are well-shaped (`geo_freq_l0.png`).

## F. Calibration landscape, re-read from the curves

- The lr **cliff at 2.4e-3 is a mid-run instability, not a smooth
  ceiling**: a ~10× loss spike near step 1.5k (and a second at ~3.8k)
  that the cosine tail never repairs (`cal_training_curves.png`). Phase-1
  monitoring should include a loss-spike guard at the chosen lr — the
  1.2e-3 optimum sits one octave below a live instability.
- λ=1e-3 is **geometrically** free, not just FVU-free: share profiles,
  stacked spectra, and frame-rotation distributions are indistinguishable
  from λ=0 at the winner lr (`geo_dimensions.png`). The nuclear penalty at
  this dose neither collapses nor shapes rehearsal-scale geometry — it
  remains purchased insurance.
- Seed noise, schedule ordering (cosine ≥ linear_fifth above 3e-4), and
  the wd no-op are all visible in `cal_lr_response.png`, matching the
  0.9.5 findings doc numbers.

## G. Figure index

| figure | shows |
|---|---|
| `probe_block23_ring.png` / `probe_block23_3d.html` | the discovered month block: token cloud, class-mean dodecagon, angle staircase |
| `probe_ring_depth.png` | raw stream loses the ring after L17; block 23's frames keep it everywhere |
| `probe_selectivity.png` | month/weekday selectivity concentrated in named blocks |
| `probe_weekday_null.png` | weekday block 640: perfect capture, no ring |
| `rings_by_depth.png` / `rings_cross_layer_3d.html` | 4b month decoder rings per depth, calendar-Fourier plane (supervised projection), Procrustes-stacked |
| `rings_codemap_heatmap.png` | phase-0.5 audited numbers: code correspondence 9-22-29, L17 odd-one-out |
| `rings_l22_threshold.png` | the ring below τ=0.5: adjacency band structure, max adjacent cos 0.32 |
| `rings_control_weekday.png` | GPT-2 Engels weekday heptagon (70% first harmonic) |
| `geo_share_heatmap.png` / `geo_share_3d.html` | per-block depth-energy allocation, baseline vs renorm |
| `geo_frame_rotation.png` | rotation decay + the L13→L17 shear zone |
| `geo_dimensions.png` | stacked spectral dimension; scalar features rotate too |
| `geo_code_anisotropy.png` | code effective dimension, G=1024 vs G=4096 |
| `geo_freq_l0.png` | firing spectra, L0 distributions, θ transfer |
| `cal_lr_response.png`, `cal_training_curves.png`, `cal_renorm_allocation.png`, `cal_dead_dynamics.png` | 0.9.5 landscape |
| `whitener_spectra.png` | covariance spectra by depth; shrinkage retention; F7 scalars |

## Methods notes (honesty box)

- **Labeling**: phase-0 single-token surface forms; analysis restricted to
  capitalized forms ("May"-class would otherwise be ~80% modal "may";
  "march"/"august" similar). Class counts after restriction: months
  551–1,234 per class, weekdays 107–168.
- **Probe stream**: FineWeb-Edu at the current dataset revision, first
  20k documents skipped to avoid the store's head slice. Same packing
  convention as the harvest (BOS + drop positions 0/1).
- **Screens**: the winner/G4096/scalar/renorm arms were screened by
  z-scored score vs 60k background positions; base_lr3e-4 and seed-1 (no
  background encode) by family/overall mean-score ratio. The two screens
  agree on the winner's top block (23). Screen choice does not affect the
  ring tests themselves.
- **Projections**: block-code ring planes are *unsupervised* (PCA of class
  means). The 4b decoder-ring planes are *supervised* (first-harmonic
  Fourier over calendar index — the component the order statistics
  detect; PCA top-2 holds only 25–34% there). The 3D cross-depth stacks
  use viz-grade Procrustes alignment between consecutive depths — gauge
  fixing only, per-depth geometry untouched. Cross-layer *claims* rest on
  the audited `cross_layer.json` numbers, not on these projections.
- **Ring statistic**: count of cyclically-adjacent class pairs that are
  adjacent in angular order in the top plane (max 12 ⇔ angular order =
  calendar order up to rotation/reflection); permutation null over class
  labels, 20k draws.
- Trained-geometry numbers come from fp32 master weights in `latest.pt`;
  Gram residuals ≤ 2.1e-6 everywhere; eval stats from the 1M-token eval
  split in stored order, threshold mode.

## What this feeds forward

1. **F7 (site renorm)**: mechanistic support for a9's renorm lean (§D);
   add one Phase-1 eval line: known-ring consolidation under renorm.
2. **Phase-1 monitoring**: loss-spike guard (§F); the ring-consolidation
   probe as a cheap standing eval; packing-clique watch at large G (§E).
3. **Phase-2 export**: block 23 (or its Phase-1 4b analogue) is the first
   real `discovered` manifold candidate for the saklas bridge; the
   share statistic is validated eval-side (§C).
4. **H1 prior**: strengthened — the BSC-native unit captures at rehearsal
   scale what post-hoc blockification structurally could not reach at 4b.
