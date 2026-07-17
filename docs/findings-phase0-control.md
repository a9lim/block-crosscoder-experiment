# Phase-0 positive control: Engels rings on GPT-2-small layer 7

**Status: PASSED (2026-07-16) — all three cyclic families recovered as
significant rings through the exact production pipeline.** Observational
only, per design (P2: GPT-2 scores near-chance on the modular-arithmetic
causal tasks — no interventions were run).

Run artifacts: `/data/stores/bcc-phase0/gpt2_l7_owt/` (4M-token CodeStore +
`control_run/` caches) on jobe; figures mirrored in
`figures/phase0-control/`. Scripts: `scripts/harvest_phase0_control.py`,
`scripts/run_phase0_control.py`.

## Setup (provenance-gated)

- **Artifact**: `gpt2-small-res-jb` @ `blocks.7.hook_resid_pre` — verified
  as the exact Bloom 2024 release Engels clustered (repo, d_sae 24576,
  hook; SHAs + decoder content hash in
  `data/phase0/artifact_provenance.json`). The true positive control, not
  a transfer control.
- **Harvest**: 4.0M OpenWebText tokens under Bloom's training conditions
  (ctx 128, prepend-BOS, `center_writing_weights`; BOS positions dropped
  at harvest). Mean L0 58.8, zero-code fraction 0, 24,570/24,576 features
  fired.
- **Clustering**: Engels App. F.1 — spectral clustering on pairwise
  angular similarity at n_clusters = 1000 over all 24,576 decoder rows
  (implementation sklearn-verified, ARI 0.958). 1000/1000 clusters
  nonempty, median size 17, max 521.
- **Family candidates**: clusters ranked by labeled-firing affinity with
  class-coverage priority (coverage ≥ n−1 classes outranks raw affinity —
  a single-day feature cluster is 100% weekday-affine but is not the
  ring). Battery run on the top 3 per family; affinity top-1 is the
  primary, the rest are recorded diagnostics.

## Results (labeled battery, class-permutation null, n_perm = 200)

| family | cluster | members | circ (held-out) | p | ring plane | Engels rank |
|---|---|---|---|---|---|---|
| weekday | 937 | 13 | **0.902** | **0.005** | PCs 3/4 | 150/981 (paper 9) |
| month | 355 | 23 | **0.904** | **0.005** | PCs 3/4 | 20/981 (paper 28) |
| year | 332 | 13 | **0.691** | **0.005** | PCs 3/4 | 502/981 (paper 15) |

p = 0.005 is the floor at 200 permutations — each family beat every
class-identity permutation. All three rings live on the same plane, PCs
3/4, one pair deeper than the intensity/context PCs — the cone regularity
Engels describes, consistent across families. Full per-plane circ scores
are in `control_run/family_battery.json` (e.g. weekday: 0.49/0.76/0.90/
0.58 across planes 1-2 … 4-5).

- **Month** is a textbook replication: twelve angular sectors in calendar
  order with the PC1-intensity cone, and Engels rank 20 vs the paper's 28.
- **Weekday** is a clean heptagonal ring on PCs 3/4 (Mon→Sun in cyclic
  order). Its irreducibility-best plane is the cone stripe (4/5) — see
  *instrument lessons* below.
- **Year** is an ordered arc, not a closed loop — 1900→1999 sweeps the
  plane in order (circ 0.69 with 100 classes; the bounded interval makes
  an arc the expected geometry, consistent with Engels' year plots).
- **Specificity**: the six non-primary candidate clusters all score at
  chance (p ≥ 0.10) except a weak secondary year cluster (767: circ 0.164,
  p = 0.005) — the battery does not hand out significance to
  family-adjacent clusters.
- **Engels ranks** (mean-plane (1−M)·S over our 1000 clusters): month
  matches the paper; weekday (150) and year (502) rank lower than the
  paper's clusters. Rank is a function of the whole cluster set and of
  plane averaging — our variance-filtered planes and stochastic k-means
  differ from theirs; the labeled battery, not the rank, is the ring
  evidence. Recorded as a caveat, not a failure.
- **Co-activation branch (P15)**: best-match Jaccard vs the geometric
  clusters — weekday 0.69, year 0.38, month 0.19. The month number is
  structural, not a bug: distinct month features fire on *disjoint*
  tokens (a date mentions one month), so co-occurrence similarity cannot
  assemble the full ring; it groups the within-token co-firing part
  (adjacent-angle interpolation + shared date context). The branch is a
  complement to decoder geometry exactly as P15 intended, but for
  class-partitioned rings the geometric branch is the primary.

## Instrument lessons (binding for the gemma run)

1. **Plane choice must come from the labeled criterion.** The
   irreducibility-best plane is often the intensity cone, not the ring
   (weekday: cone plane circ 0.584 vs ring plane 0.902). The battery
   statistic is now max-over-scanned-planes of held-out circular
   decoding, with the same max under the class-permutation null —
   selection lives inside the test, p-values stay calibrated.
2. **Class-identity permutation is the cyclic-order null.** Token-label
   shuffling rewards any consistent clump layout (planted non-ring
   control: circ 0.67, token-null p = 0.01, class-null p = 0.35).
3. **Irreducibility ≠ ring.** Sparse one-hot clusters reconstruct onto
   rays, which score high (1−M)·S (a planted ray control scored 0.889,
   above the planted ring). The label-free scan is gated on co-firing
   tokens and uses harmonic contrast (peak − median over m); the Engels
   score is a surfacing diagnostic.
4. **Null resolution must out-resolve BH at the search width** (empirical
   p floor = 1/(n_draws+1); at width 3, ≥60 draws).
5. **Engineering**: dense shard accumulation OOM'd the harvest (fixed:
   per-batch COO sparsification); the 24k² eigh needs a clean CUDA pool
   (in-place similarity normalization, cache clears between stages).

## Gate consequence

Per design §Phase 0: control passing makes the gemma null interpretable.
Next: `gemma-scope-2-4b-pt-res` layer 22 (`width_16k_l0_medium`, pinned in
provenance) — harvest a comparable token stream, same dual-branch
clustering, full battery with BH over the unknown-cluster search.
