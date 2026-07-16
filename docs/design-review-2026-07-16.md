# Adversarial design review, round 3 — 2026-07-16

Pre-implementation pass over frozen design v2.1, requested by a9 before jobe
setup. Two fresh-context Codex sol-tier consults (deliberately *not* the
round-1/2 thread, to avoid anchoring): **deployment/design** (thread
`bsc-v21-fresh-attack`, job `cx-20260716-102302-f86e`, findings D1–D14) and
**paper fidelity** (thread `bsc-papers-fidelity`, job
`cx-20260716-102304-8152`, findings P1–P25, all 13 reference texts read at
full length plus the Jiralerspong PDF). A fable parallel pass ran before
reading either reply. Both consults confirmed file access; three
quote-anchored paper claims were spot-checked verbatim against the local
texts (Minder A.10 12.0% dead, SASA C.1 ν=10⁻⁴/1k-token window, Engels F.1
Bloom layer-7 25k features / 1000 spectral clusters) — all exact.

**Headline: the core architecture survives round 3 untouched.** No paper
contradicts the Gram constraint, whitened ‖z_g‖ selection, shared vector
code, or per-site frames, and the round-1/2 algebra stands. What fails is
the *deployment plan* (the store does not fit the real machine) and a set
of seams the first two rounds never reached. Dispositions below are
**proposed** (fable synthesis); items marked ⌛ are a9's to decide.
⊕ marks findings independently found in the fable parallel pass.

## Ground truth that triggered the round

jobe measured 2026-07-16: `/data` 794 GiB available, OS drive 841 GiB
available (2×1 TB NVMe — **not** the ~1.9 TB single volume v2.1 assumed);
RTX 4090 24 GB; 61 GB RAM; live sequential buffered read ~1.94 GB/s; HF
cache currently 11 GB on `/data`. Motherboard MSI PRO Z690-A: 4 M.2 slots,
2 occupied — a disk purchase is mechanically possible.

## Blockers

| # | Finding | Proposed disposition |
|---|---|---|
| D1 ⊕ | The v2.1 store (1.638 TB + unbudgeted calibration) does not fit: `/data` alone is 794 GiB; spanning both disks leaves ~115 GB total headroom before checkpoints/cache/reserve — operationally unsafe. The single-NVMe assumption behind the R23 arithmetic and the G=8192 stretch is false. | **Accept; re-open the storage topology.** Option ranking (both passes agree): (1) 4 TB NVMe (~$300; preserves 8 sites / 38M tokens / G=8192 / harvest-once exactly) — **decided 2026-07-16: a9 buys the drive**; (2) interleaved streaming at G=4096 (sample power preserved or better; every run repays gemma forwards; stretch dies); (3) validated whitened int8 store (see D5); (4) 6 sites × 38M spanned; (5) 13–15M tokens on `/data` (weakest — see D2); spanning with the full bf16 store is rejected. Fable-only alternative recorded: two ~19M-token half-stores swapped per epoch-cycle — full 38M unique tokens on current disk, but it only beats streaming if runs share the resident half in lockstep; niche. |
| D4 ⊕ | The calibration split is unbudgeted *and* statistically underspecified: codec quantiles, canonical orientation, count model, threshold, and single-site maps all fit on it, yet it has no token count or storage line; 1–2M tokens give ~8–16 observations in a 0.1% clipping tail for an average-frequency block. | **Accept.** Define calibration power in *active counts per block*, not tokens. Adopt sol's route: a large manifest-only calibration split streamed after training (sufficient statistics + sparse codes accumulated, raw activations never persisted), plus preregistered fallbacks (better-powered quantiles or robust pooled scales; underpowered blocks excluded from the codec comparison and reported). |
| D11 | The Phase −1 generator is not yet well-posed under the model's gauge: recovery is identifiable only modulo one joint O(b) per block (per-site alignment can falsely certify shared coordinates); planted rank must be defined through contribution spectra, not decoder rows (the constraint forces frame rank b); and the "correlated scalar bundles" null is observationally impossible — perfectly co-active scalars with full-dimensional joint support are equivalent to a block under a linear reconstruction objective, so the learner cannot be required not to bundle them. **This corrects the round-1 finding-9 disposition, which overpromised.** | **Accept.** Adopt the generator spec: intrinsic coords u_g ∈ ℝ^r embedded as z_g = A_g u_g; Gram-satisfying concatenated decoders; rank controlled via contribution-operator spectra; block matching by assignment then one *global* Procrustes across all sites; site-specific decoys scored as expected site-exclusive recoveries; the correlated-scalar null weakened to "coherence/topology tests must not hallucinate *curved manifold* structure on bundles." Required before the first Phase −1 code. |
| P13 | The Phase-0 positive control is not pinned to Engels' actual setting: F.1 used Bloom's 2024 GPT-2-small **layer-7** SAE (~25k features), spectral clustering with n=1000, ~500 clusters manually inspected. "A public SAELens release" ≠ that artifact; a failed transfer would not demonstrate pipeline power. Also, GPT-2 is only the *observational* control — Engels' causal tests were Mistral/Llama (GPT-2 scores 8/49 weekday, 10/144 month on the modular tasks). | **Accept.** Pin the exact Bloom layer-7 artifact (it is a SAELens-registry release; verify identity at Phase 0). If unobtainable, the SAELens run is labeled a *transfer control* and a faithful replication path is added. Digest wording on causal scope corrected (P2). |

## Deployment/design majors (D-series)

| # | Finding | Proposed disposition |
|---|---|---|
| D2 ⊕ | A 16M-token fallback changes the claim envelope, not just duration (125k vs 297k mean active examples/block; 0.01%-frequency blocks get ~1.6k examples; epochs repeat, they don't restore unique-sample power). Realistic `/data`-only cap is 13–15M, not 16–17M. Separately: the 10k H4 inclusion threshold on 2M eval tokens already excludes blocks rarer than 0.5%. | Accept. Governs the fallback only; if taken, rerun Phase −1 recovery-frequency curves at the exact reduced N, predeclare common-feature-only claims, make streaming escalation mandatory if FVU is still falling. Never described as R24-equivalent. |
| D3 | Site cuts alone don't fit `/data` (6 sites × 38M = 1.229 TB); 6 sites is the least-damaging no-purchase store compromise but H4 becomes coarsely depth-resolved (mean spacing 2.3 → 3.2 layers) and Phase 3's 4+4 no longer follows from Phase-1 density. | Accept as fallback guidance; sites selected on Phase-0.5 evidence before config freeze; Phase-3 site budget decided explicitly, not silently. |
| D5 | The fp8-ban rationale ("noise near the RD floor") is not load-bearing — RD is measured on clean bf16 eval data. The real argument: quantization acts as an *architecture-dependent regularizer* (support, dead rates, spectra, bundling can shift differently for BSC vs scalar), and raw fp8 on gemma outliers is unsafe regardless. | Accept. Ban stays for the primary; rationale rewritten. A **whitened, per-channel-scaled int8/f8 codec** is the documented store-compression escalation, admissible only after a paired pilot (bf16 vs quantized training) agrees within preregistered tolerances on RD frontier, block matching, dead/count distributions, rank spectra, shared-code diagnostics, 2 seeds. |
| D6 | Token-random mmap reads can starve the GPU; measured 1.94 GB/s sequential means a 160 MiB batch has an 82 ms I/O floor — fine only with sequential buffered shuffling. | Accept. Store layout spec'd: 2–8 GiB atomic shards `[token, site, d]`, sequence-contiguous writes, per-epoch shard shuffle, 32k–128k-token RAM shuffle buffer, 2–4 pinned prefetch batches, no global token-random mmap access, recorded permutation seed shared by BSC and baseline. Pilot logs data-wait and GPU duty cycle, not just tok/s. |
| D7 | Harvest is hours (~1–3 h kernel-time for 45M tokens; 2–8 h end-to-end), but unmeasured; a serial pipeline could exceed 12 h. | Accept. Pilot runs the exact production path (hooks, dtype, checksum, shard writer) and reports model tok/s, writer GB/s, GPU utilization, projected wall-clock. Skip the LM head at harvest. |
| D8 | Prewhitening the store is mathematically safe (whitener is frozen pre-training anyway), cuts a 429.5 GFLOP/batch train-side matmul (~16% of encoder-forward FLOPs), and conditions any int8 codec — *if* the whitener becomes immutable. | Accept — **design change: the store holds whitened bf16.** Whitener slice harvested first; exact μ, W, ridge, layer set, and source manifest hashed into every shard header, mismatches rejected at load; a small raw validation shard retained for round-trip error; any refit ⇒ reharvest or explicit migration. |
| D9 | "5M tokens vs covariance entries" is the wrong sufficiency argument (sequence correlation shrinks effective N); TF32 GEMMs and fp32 eigh can distort small eigendirections. | Accept. TF32 off for covariance; pairwise/Welford accumulation with fp64 aggregates; the eight 2560² eigh in fp64 offline; whitener stability compared across corpus halves/quarters; transformed spectrum validated on held-out sequences. 5M is a candidate size that must pass these, not a proven one. |
| D10 | The EMA-of-batch-minimum θ inherits training history (drift, decay constant, checkpoint timing) — two equivalent checkpoints can get different realized count distributions, moving the codec frontier. | Accept. EMA demoted to training diagnostic; final θ fit on the calibration split to hit the preregistered average block count, frozen and serialized with the codec, sensitivity reported. |
| D12 | The 1M-token pilot **cannot exercise AuxK**: 244–488 steps < the 500-dead-batch eligibility window. The pilot would report a "dead-block trajectory" without ever activating the recovery mechanism. | Accept. Pilot extended to ≥3M token exposures (>500 steps with margin); explicitly exercises AuxK, checkpoint/resume, and final-threshold calibration; adds a synthetic dead-encoder revival test; logs pre/post-retraction Gram eigenvalues, floor hits, depth-share jumps, aux/main gradient norms. Retraction once, after the combined update. |
| D13 ⊕ | Phase 0.9 (1b) cannot validate the main 4b operational risks (outlier behavior, 160 MiB batch I/O, 671M-param optimizer throughput, checkpoint size, 8-site memory, G=4096 support competition, calibration tail power) — "a 4b failure after a green rehearsal is about the science, not the code" overclaims. (⊕ fable flagged the adjacent λ-transfer assumption: rehearsal-narrowed λ carrying 1b→4b is a cross-model assumption whose backstop is the one-seed 4b frontier stage — now stated openly.) | Accept. 0.9 stays plumbing-scoped; the overclaim sentence is struck; the extended exact-config 4b pilot (D12) becomes a separate, mandatory operational gate. |
| D14 ⊕ | Checkpoint/cache headroom understated ~4×: a *resumable* 8-bit-Adam checkpoint is ~5.4 GB (bf16 fwd + fp32 master + moments), not 1.3 GB (that's inference-only); atomic writes need double transiently; the 82 GB eval store exceeds 61 GB RAM; HF cache grows. (⊕ `/data` cohabitants — acot artifacts, HF cache — must be subtracted from any budget explicitly.) | Accept. Latest-resumable + best-inference checkpoints only per run; checkpoints/cache placed off the hottest volume where possible; ≥15% filesystem free floor with pre-write abort checks; eval streamed sequentially. |

## Paper-fidelity findings (P-series)

**Digest corrections (accepted; edit `docs/research/…2026-07.md`):**
P1 Minder *mitigates* shrinkage/decoupling ("may address", A.10: 12.0% dead
in their BatchTopK run) — design's "structurally designed out … (BatchTopK)"
softens to "empirically mitigated; monitored by Latent Scaling + causal
validation". P2 Engels causal scope: rings *found* in GPT-2 + Mistral;
causal tests were Mistral/Llama only. P3 Michaud's tiling-vs-rare-features
is an *unresolved possible regime* (their §4 "we do not resolve"; radial
thickening saturates basis-like near n≈2dᵢ). P4 the support-bit argument is
a Fel-*inspired* intuition, not Fel's full MDL protocol. P5 b=4 is a
hypothesis, not attribution-backed (Fel C.3 optima ≈1–3, "trust the
direction, not any single value"). P6 provenance labels updated to
full-text for all 13 (+ Jiralerspong via PDF). P7 the SASA attribution as
corrected in round 2 is verified accurate.

**Design consequences (accepted):**

| # | Finding | Proposed disposition |
|---|---|---|
| P8 | The AuxK spec is an unsupported hybrid: SASA C.1 = frequency-dead (π ≤ 10⁻⁴ over a 1k-token window), frozen-residual re-encoding through dead groups, s_aux = 256–512, λ_aux = 1; Fel D = runner-up AuxK; ours (top-32, 500 batches ≈ 2M tokens, α=1/32) has no source basis. | Adopt SASA's frozen-residual re-encoding semantics as the starting spec; Phase −1/0.9 compares SASA-style frequency-dead vs long-horizon vs Fel runner-up before the Phase-1 freeze. Merges with D12's revival test. |
| P9 | SASA B.3 supports AdamW generally but has no retraction/equivariance analysis; the SASA product-penalty *ablation* should use symmetric E/D weight decay to instantiate the variational form. | Accept; primary keeps decay-0 decoders + rotation-equivariance test; ablation gets symmetric decay. |
| P10/P11 | Latent Scaling must stay mandatory under BatchTopK (Minder's immunity is one run, and BSC changes the latent object); the blockwise analogue needs a real spec — Minder fits four coefficients per latent with leave-one-out error targets; a scalar coefficient is unfaithful for a b-dim block. | Accept: blockwise Latent Scaling = leave-one-*block*-out + reconstruction targets per model/site, held-out b×b (ridge or Procrustes-constrained) maps, normalized improvement ratios, calibrated on planted shared/site-specific/shrunk/decoupled blocks. Predeclared before Phase 3. |
| P12 | H5 should inherit Minder's *causal* diffing: selected-block base→chat patching, None/All/Error baselines, full-response + early-token KL (their early-token gap was >3× full-response), sequence-level CIs, template stratification, Sentence-BERT nonactivating controls for autointerp. | Accept for Phase 3. |
| P14 | Phase 0 omits parts of Engels' battery: cluster-restricted reconstruction (out-of-cluster latents ablated), no-active-element sample discard, PC-plane scan (1–2…4–5), averaged separability/ε-mixture across planes, PC1-as-intensity cone check, clustering-stability + Jaccard. | Accept — added to Phase 0; our circular-decoding/Fourier/permutation/BH battery is retained as the *additions* it is. |
| P15 | Bhalla: decoder-cosine clustering alone "need not suffice" — their Ising fit on binarized codes (pseudo-likelihood, EBIC, Louvain) recovers co-active *and* mutually exclusive manifold tiles; cosine-only Phase 0 is weakest exactly in shattering/dilution regimes. | Accept: an activation-dependence discovery branch (Ising if feasible, else a documented conditional-dependence approximation) runs beside cosine clustering in Phase 0. |
| P16 | Fel D details worth stealing: norm-calibrated init, log-parameterized gain, QR projection every 20 steps (vs our per-step retraction); their selection is per-sample, BatchTopK explicitly future work — our rule is a synthesis, and better-justified under the Gram constraint. | Accept: adopt init calibration; retraction-frequency ablation noted; per-step retained initially. |
| P17 | Anthropic's per-layer decoder norms are an unconstrained object; our Gram-normalized Frobenius shares are an *analogue*, not the same number. Their §5.2 isomorphic-but-wrong warning reinforces P12. | Accept — wording. |
| P18/P19 | Phase −1 nulls should plant both hollow and radially-thickened versions of the same manifold (Michaud A.1 regime split) and adopt Bhalla's capture/shattering/dilution metrics (restricted R², support size, receptive-field spread). | Accept — added to the Phase −1 harness spec alongside D11. |
| P20 | Dooms: individual-latent seed agreement is 15–50% while *global recovered subspace* agreement is >90% — judge seed stability at subspace level too. | Accept: principal-angle global-subspace stability added to the seed-stability protocol. |
| P21–P24 | Ge (norm-evolution is suggestive — no artifact diagnostics), Gorton (cross-site *geometry* with scalar codes — wording), Ghilardi (exposes Yun 2103.15949 + Lawson 2409.04185 as owed predecessor reads), Laptev (stays demoted; keep its deactivation/random-predecessor controls if used). | Accept — wording + debts. |
| P25 | Jiralerspong local copy was an abstract stub (no arXiv HTML exists; refetch confirms). Full PDF verified: Dedicated Feature Crosscoders — *scalar* shared/exclusive partitions, BatchTopK, model stitching + cross-model steering validation of exclusivity. Does not occupy the BSC cell. | Accept: refs.yaml annotated; DFC noted as a scalar architectural baseline candidate for Phase 3, and its stitching/steering validation of exclusivity adopted alongside P12. |

## Novelty verdict (P-series)

The narrow claim **survives**: within LLM interpretability, none of the
checked work learns a sparse set of multidimensional blocks with one shared
vector code and distinct per-site decoder frames across sites. Broader
prose does not survive unchanged (Dooms/SASA/Bhalla: single-site
multidimensional; Gorton: cross-site geometry, scalar code; Group-SAE/Yun/
Lawson: scalar dictionaries shared across layers; Ge/Anthropic/Minder/
Jiralerspong: scalar cross-site codes). Owed reads before any public claim:
multi-view/coupled/group-sparse dictionary learning (largest risk), Yun
2103.15949, Lawson 2409.04185, SMixAE, Shafran et al., Hindupur/SPADE,
Mishra-Sharma, the Anthropic-mentioned Baskaran–Sklar work if public.

## Verdict and freeze status

Round-3 verdict (both passes concur): **the science is implementable on
this 4090, but not under the frozen v2.1 deployment plan.** v2.1 is
re-opened in four places — storage topology (D1, with G=8192 conditioned on
actual storage), calibration/threshold specification (D4/D10), the Phase −1
generator (D11, correcting the round-1 finding-9 disposition), and the
pilot/rehearsal gates (D12/D13) — plus the accepted amendment set above.
Recommended sequence (sol's, endorsed): corrected Phase −1 harness and
Phases 0/0.5 on the Mac now; extended 4b pilot; storage in parallel.
**Both decisions resolved same day: a9 buys the 4 TB NVMe (interleaved
G=4096 streaming stays the documented fallback) and signed off on v2.2 —
all accepted amendments above are folded into `design.md`, the research
digest, and `AGENTS.md`; design re-frozen at v2.2.**

## Standing debts (carried + new)

- Multi-view / coupled / joint group-sparse dictionary-learning sweep
  (carried; still the largest novelty risk).
- Yun et al. 2103.15949 and Lawson et al. 2409.04185 (new, via Ghilardi).
- SMixAE; Shafran et al.; Hindupur/SPADE; Mishra-Sharma; Baskaran–Sklar
  (carried/new, see novelty verdict).
- Phase −1 must still set the numeric λ-veto tolerance and
  rotation-equivariance threshold in its config (carried).
- Verify fp32-master/8-bit-Adam retraction ordering against the actual
  optimizer library before Phase 0.9 (carried).
- Verify the Bloom 2024 GPT-2-small layer-7 SAE artifact's SAELens identity
  at Phase 0 start (new, P13).
