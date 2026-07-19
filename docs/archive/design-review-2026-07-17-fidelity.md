# Paper-fidelity audit — 2026-07-17

Requested by a9: a comprehensive check that the BSC training setup and
methodology are procedurally aligned with the parent papers — BSF (Fel
2606.25234), SASA (Dalili 2606.06333), Anthropic crosscoders (2024),
Minder (2504.02922) — down to hyperparameters where possible, with every
deviation justified. Fable pass + **sol counter-review same day** (job
`cx-20260717-110930-e764`, thread `bsc-fidelity-audit-review`;
disposition in the final section) — **ratified by a9 2026-07-17**
(design frozen as v2.3.2; 0.9.5 addendum greenlit including the
site-renorm arm).
Method: two independent extraction agents pulled every
procedural/hyperparameter claim from the four local full texts with
verbatim quotes; all load-bearing quotes were then re-verified against
the files by grep before use. Compared three-ways against `docs/design.md`
v2.3.1 and the code as of `2e608b8` (`gram.py`, `model.py`, `trainer.py`,
`store.py`, `harvest_phase09_store.py`, `run_phase09_rehearsal.py`).

Relation to round 3 (P1–P25): that pass audited the *design text* against
the papers. This pass adds (a) the code as actually written post-Phase-0.9,
(b) numeric hyperparameter values the papers pin, and (c) a check that the
P-series dispositions landed. Findings F1–F11 below are new; none is a
blocker.

**Headline: the architecture and training procedure are either
paper-faithful or deviate at documented, justified points. No silent
substantive deviation was found. The residue is: one undocumented
loss-form divergence from Minder (F1), three "calibration item" promises
that Phase 0.9 did not actually discharge (F2, F3), one Phase-3 spec gap
Minder makes visible (F4), and a handful of wording/code minors.**

## Paper ground truth (verified verbatim)

Numbers the audit keyed on, all re-checked against `references/*.md`:

- **Fel App. C.1/D**: Adam, cosine LR 1e-4 → 1e-5, 2000-step warmup,
  batch 8192, 3 epochs; signed codes, "no more ReLU involved"; per-sample
  TopK by raw ‖z_g‖₂ ("batch-level variants … we leave to future work");
  Grassmannian per-block per-site Stiefel constraint via QR re-projection
  "every 20 steps"; init W = Dᵀ with encoder scale calibrated; AuxK =
  next-ℓ runner-up blocks by norm, α = 1/ℓ (no stop-grad stated — sol);
  MDL grid b ∈ {1,2,3,4,6,8,12,16,32} (sol correction — not all
  integers), G ∈ {4096, 8192, 16384, 32768}, ℓ ∈ {8, 16, 32, 64}; input
  scaled to E‖x‖ ≈ √d; App. C.3
  optima b ≈ 1–3, "trust the direction … not any single value".
- **SASA §5/B.3/C.1**: linear encoder p = Eh (no ReLU, no stated bias),
  untied; no decoder-norm constraint — rank control via λ_dim Σ_k ‖D_k E_k‖_*
  (numeric λ_dim **never published**; enforced variationally as symmetric
  weight decay); per-sample Top-s by raw ‖E_k h‖₂; AdamW β = (0.9, 0.999),
  lr 2e-4, weight decay 1e-3, 1000-step warmup, linear decay over the final
  fifth, token batch 4096; GPT-2 (K,r,s) = (2048,6,10) @150M tokens ctx 128,
  Mistral (4096,8,10) @500M ctx 512; AuxK C.1: dead = running frequency
  ≤ ν = 1e-4 over a 1000-token window, top-s_aux dead groups by residual
  energy ‖E_k r‖², frozen residual, λ_aux = 1, s_aux = 512/256.
- **Anthropic crosscoders**: z = ReLU(Σ_l W_enc^l a^l + b_enc) (shared
  encoder bias), per-layer decoders + biases; loss Σ_l ‖a − â‖**²** +
  Σ_i f_i (Σ_l ‖W_dec,i^l‖) — **L1-of-norms**, with the L2-of-norms
  variant explicitly discussed (surfaces only shared features; better
  MSE/L0 frontier); per-layer activations normalized "so that each layer
  contributes comparably" — **no formula given**; optimizer/lr/batch/tokens
  all unspecified (research note).
- **Minder §2/A.10**: ReLU encoder summed over the two sites + scalar
  b_enc per latent; reconstruction = ½‖ε_base‖₂ + ½‖ε_chat‖₂ —
  **unsquared L2 norms** *as written* (sol caveat: the same paper
  derives Latent Scaling from a squared objective and later uses MSE
  language — internally inconsistent; released trainer unchecked);
  BatchTopK selects top n·k batch-wide by
  v = f_j · (‖d_base‖₂ + ‖d_chat‖₂); θ = E_batches[min selected v],
  global scalar; AuxK k_aux = 512, α = 1/32 on dead latents; lr 1e-4,
  2 epochs, expansion 32 on d = 2304 (73,728 latents), k = 100, 100M
  tokens **FineWeb + lmsys-chat**; decoder = encoder transpose at init,
  base/chat weights **paired identical at init**; Latent Scaling
  ν_ε < 0.2, ν_r < 0.5 gates; Δ_norm bands 0–0.1 / 0.4–0.6 / 0.9–1.0;
  12.0% dead (BatchTopK), dead latents excluded from analyses.

## Confirmed faithful

(Sol correction to this section's original title: rows marked "spec'd"
are paper→spec agreement only — no implementation exists yet, so
"paper → spec → code all agree" was too strong for them.)

| Item | Anchor | Status |
|---|---|---|
| Subspace unit, G blocks × width b | BSF/SASA | ✓ |
| **Signed linear codes, no ReLU** | Fel App. D explicit; SASA linear | ✓ — deviates from crosscoders' ReLU, but that is the parents' own argument (ReLU restricts a block to a cone) |
| No encoder bias | SASA-as-written / Fel-Grassmannian; principled under whitened zero-mean input | ✓ (wording note in F6) |
| Summed per-site encoder, untied | Anthropic "summing over contributions"; Minder Eq. 1 | ✓ `model.encode` |
| Per-site decoder + per-site bias c^s | Minder Eqs. 2–3 | ✓ |
| BatchTopK mechanics (top round(k·B) batch-wide, per-token counts vary) | Minder Eq. 5 / Bussmann | ✓ `batch_topk_mask`; departure from BSF/SASA per-sample selection is documented and pre-blessed by Fel's future-work note |
| AuxK default = SASA C.1 mechanism | frozen residual, dead-set re-encoding, residual-energy ranking, λ_aux = 1, s_aux = 256 (∈ SASA's 256–512; = K/16 at Phase-1 G, exactly SASA-Mistral's ratio) | ✓ `aux_loss` — window re-expression is F3 |
| AdamW β = (0.9, 0.999), token batch 4096 | SASA B.3 verbatim | ✓ |
| Decoder weight decay 0 | R11 (retraction undoes uniform shrinkage) — deliberate vs SASA's symmetric 1e-3, which *is* SASA's regularizer and therefore not copyable | ✓ |
| Init: D Gaussian + one retraction, E = Dᵀ tied at init, encoder scale calibration on data | Fel App. D; Minder A.10 transpose-init | ✓ `init_decoder_stack`, `calibrate_encoder_scale_` |
| 2-epoch default + held-out FVU-gap monitor | Minder 2 epochs; monitor is a BSC addition | ✓ |
| No gradient clipping | unspecified in all four papers | ✓ consistent absence |
| Dead-latent exclusion from readouts | Minder convention → H4 10k-active-token floor (R22) | ✓ |
| Eval FVE/FVU per site + pooled | SASA/Minder family | ✓ rehearsal driver, centered, fp64 |
| Blockwise Latent Scaling (b×b leave-one-block-out maps) + causal diffing battery for H5 | Minder §3.1.2 / Eq. 12–13, generalized per P10–P12 | ✓ spec'd (Phase 3) |

## Deliberate deviations — all documented, all justified

1. **Gram constraint Σ_s D_g^s D_g^sᵀ = I_b** — matches *neither* parent
   (Fel: per-site Stiefel; SASA: unconstrained + product penalty;
   crosscoders: unconstrained). The central design move; algebra
   independently verified round 2 (scale-gauge death, O(b) residual,
   ‖z_g‖² = exact contribution energy, dead-spiral block). Fully worn.
2. **Whitened everything** — no parent whitens (Fel √d-scaling; SASA
   layer_norm; Anthropic normalizes per-layer with no formula; Minder
   silent/library-delegated). BSC-original, extensively justified
   (massive-activation bait, cross-site commensurability, finding 7).
   There was no precise convention to violate. Shrinkage caveat: F7.
3. **Rank penalty on whitened per-site decoders** instead of SASA's
   ‖D_k E_k‖_* — the v1 mis-transfer story, fixed structurally; SASA's
   product form retained as the documented ablation with symmetric decay
   (P9). λ grid {0, 3e-4, 1e-3} set by the Phase −1 quantitative veto —
   the only principled route, since SASA never publishes λ_dim.
4. **θ fit on the calibration split** (quantile to hit preregistered mean
   count) instead of Minder Eq. 10's E[min selected v] — D10's argument
   (EMA inherits optimizer history; codec comparability) is sound, the
   paper's own EMA constant is unspecified anyway, and the rehearsal
   validated transfer to 0.13% of target k. Arguably stronger than the
   source convention.
5. **BatchTopK over per-sample TopK** — Minder's artifact argument
   (L1 → Complete Shrinkage / Latent Decoupling; block analogues would
   poison H5), plus Fel's explicit future-work blessing. Documented.
6. **Rate–distortion protocol** vs Fel's full MDL — deliberately excludes
   parameter bits, uniform quantizer + enumerative support code vs Fel's
   Gaussian water-filling; scoped as Fel-*inspired*, not a replication
   (P4). Documented.
7. **Per-step retraction** vs Fel's QR-every-20 — different manifold
   (concatenated Gram, eigh inverse-sqrt vs per-site QR); conservative
   default with `retract_every` as the documented throughput ablation
   (P16). Documented.
8. **8-bit Adam moments, fp32 master, bf16 forward** — no paper touches
   this; BSC-original, with its own Phase −1 retraction-ordering gate
   (passed). Documented.
9. **b = 4** — hypothesis, not attribution (Fel optima ≈ 1–3, P5). Worn
   in Risks.

## Findings (new this pass)

**F1 — moderate, wording.** The reconstruction loss silently sides with
Anthropic against Minder. Minder's written objective uses **unsquared**
per-site L2 norms (½‖ε_base‖₂ + ½‖ε_chat‖₂, Eqs. 4/8); Anthropic, SASA,
and Fel all use squared error; we use squared/mean (R12). The choice is
right (mainstream convention; the two crosscoder sources genuinely
conflict), but the design inherits Minder's BatchTopK/AuxK/θ while
rejecting its reconstruction form without a word. One sentence in
*Architecture spec → loss* closes it. Note the gradient geometry is not
equivalent: the unsquared form downweights high-error tokens.

**F2 — moderate, open procedural item.** The design says the optimizer
block is "all to be recalibrated in the Phase-0.9 rehearsal"; the
rehearsal ran the defaults and recorded no recalibration. Specifics: lr
3e-4 is above every parent (SASA 2e-4, Minder 1e-4, Fel 1e-4→1e-5
cosine); the schedule (1k warmup + cosine **to zero**) matches SASA's
warmup count and Fel's cosine shape but neither paper's actual schedule;
encoder weight decay is spec'd "decay applies to encoders only" while the
code default is 0.0 — currently no decay anywhere, and the value remains
an unresolved "calibration item". None of this is alarming (loss curves
smooth, SAE-lore range), but the promise should be discharged: fold an
lr/decay sanity check into the mandatory ≥3M-token 4b pilot (D12/D13),
or amend the design to accept the defaults with a stated rationale.

**F3 — moderate, open procedural item.** The AuxK dead window is a
materially different criterion from SASA's, and its promised
recalibration hasn't happened. SASA: ν = 1e-4 over a **1000-token**
window — arithmetically "zero firings in the last 1k tokens" (0.1
expected events), a twitchy criterion. Ours: ≤1e-4 over 100 batches ×
4096 = 409,600 tokens = "≤ 40 firings" (sol: 41/409,600 > 1e-4) — far
stickier: a recent-zero detector became a rare-frequency classifier,
and the batch-granular ring buffer cannot reproduce a literal
token-granular window at all. The design wears
the re-expression and promised "0.9 recalibrates window/threshold at
production batch size", but the rehearsal had zero dead blocks, so no
calibration data exists. The 4b pilot already must exercise AuxK (D12);
add window/threshold calibration to its explicit checklist.

**F4 — moderate, Phase-3 spec gap.** Minder trained the diffing
crosscoder on **FineWeb + lmsys-chat** (mixture proportions not stated —
sol; ours must be chosen and pinned, not "matched"); chat-specific
latents need chat
data to fire. Phase 3's spec pins template handling (template-free paired
forwards) but not corpus composition — on FineWeb-Edu alone, H5's
chat-specific manifolds (persona fan) may simply never activate. Also
from Minder A.10: base/chat encoder and decoder weights are **paired
identical at init** — a deliberate diffing prior (start shared, let
training diverge) that our independent per-site Gaussian init does not
replicate. Both belong in the Phase-3 config freeze.

**F5 — minor.** The "fel" AuxK arm as implemented is a hybrid: Fel uses
the next-**ℓ** runner-up blocks with α = 1/ℓ (ℓ = the *main* block
sparsity); ours uses s_aux runner-ups with α = 1/s_aux, and s_aux ≠ k in
every run made (rehearsal: 256 vs 16). The arm lost the Phase −1
comparison and is non-default, so no result changes; note it in the
findings doc so the arm isn't misread as a faithful Fel replication.

**F6 — minor, wording.** The selection statistic quietly departs from
Minder's: theirs is v = f_j · (Σ_s ‖d_j^s‖) — L1-of-norms geometry — ours
is ‖z_g‖, whose **square** is the exact contribution energy (sol
terminology fix; identical ranking) — L2-of-norms geometry (under the
constraint). Ours is better-founded (finding 3: score² = energy, exactly)
and site-profile-neutral, but the design never states the departure, and
it connects to Anthropic's own L1-vs-L2-of-norms discussion (their L1
choice "surfaces layer-specific features") — the same tension R7 wears
for the *penalty*. One sentence tying selection, penalty, and Anthropic's
discussion together would make the story complete. Same note covers the
b=1 baseline: signed latents + energy selection make it the b=1 special
case of *our* architecture (as R16 intends), not a literature crosscoder —
worth one line where the baseline is defined.

**F7 — ~~minor, wording~~ upgraded to moderate (sol).** "Whitened" is
ridge-shrunk: λ_s = mean eigenvalue (DEFAULT_RIDGE_SCALE = 1.0, saklas
convention) gives whitened spectra with mean |eig − 1| of 0.71–0.94
(0.9 harvest). Internally consistent — the harvest validates against
the shrinkage prediction σ/(σ+λ), correctly — but any external writeup
should say "shrinkage whitening", and the massive-activation protection
is proportional, not total. Sol's sharpening: mean retained variance
runs ≈ 0.06 (shallow) to 0.29 (deep) across sites, so the
equal-per-dimension L_rec weights deep sites several times more heavily
— the store does not deliver full cross-site commensurability or
Anthropic's "each layer contributes comparably" intent. A per-site
scalar RMS renormalization after shrinkage whitening (directional
suppression kept, equal total site power restored) is now a
**pre-4b-store decision item** in the design, informable by a read-time
renorm arm in 0.9.5.

**F8 — minor, code.** `Trainer.save_checkpoint` does the atomic
write-then-rename but lacks D14's free-space pre-check (`ShardWriter`
has one). Resumable 4b checkpoints are ~5.4 GB with a transient double
during atomic replace; add the same floor check before Phase 1.

**F9 — minor, decision.** The SASA product-penalty ablation (P9,
symmetric decay) exists only as spec — no implementation, and its
in/out status in the Phase-1 run matrix is undecided. Decide before the
4b freeze so it doesn't ambush the budget.

**F10 — info.** Fixed 1000-step warmup changes meaning across scales:
25% of the 3,906-step rehearsal, ~5% of the ~18.5k-step 4b run, ~3% for
SASA. Harmless, but the rehearsal's λ-ladder was more warmup-dominated
than Phase 1 will be.

**F11 — info, context.** Per-block sample power at the primary config
(~297k active examples/block at 38M tokens, mean freq 0.78%) sits ~2.4×
below SASA's GPT-2 run (~730k) and well above Minder's per-latent counts
— same order as both. R24's "conditional on demonstrated sample power"
framing already covers this; the cross-paper numbers are useful context
for the writeup.

## P-series dispositions — landed?

Checked each accepted round-3 item with a code/spec surface: P8 (AuxK
three-variant respec) ✓ implemented and compared; P16 (norm-calibrated
init, retraction-frequency knob) ✓; D10 (θ calibration-fit, EMA
demoted) ✓; D8/D9 (whitened store, fp64 accumulation — sidesteps TF32
entirely — offline fp64 eigh, stability halves/quarters, hash-guarded
immutability) ✓; R12 (pinned fp32 reductions) ✓; R16 (b=1
Gram-constrained baseline, matched training-average L0, λ=0) ✓ rehearsal
driver; R8 (rotation-equivariance) ✓ run, decoders stay on Adam; D14
(atomic checkpoints) partial — see F8; D6 (prefetch) and the streaming
quantile are acknowledged carries to Phase 1, tracked in the 0.9
findings. P9's ablation is spec-only (F9). Nothing accepted in round 3
was silently dropped.

## Verdict

The project's fidelity posture is unusually strong: where it follows a
paper it follows it to the number (SASA's optimizer family/batch/AuxK
constants; Minder's BatchTopK/2-epoch/diffing machinery; Fel's init,
signed codes, and comparison philosophy), and where it deviates, the
deviation is load-bearing, argued, and in three cases
(Gram constraint, θ calibration, whitened selection) defensibly
*stronger* than the source convention. The findings above are hygiene:
two unwritten sentences (F1, F6), two promises to discharge in the
already-mandatory 4b pilot (F2, F3), one Phase-3 config item (F4), and
minor code/decision cleanups (F8, F9). Recommended sequencing: fold F2,
F3, F8 into the D12/D13 pilot gate checklist; fold F1, F6, F7 into the
next design wording pass; pin F4 (corpus mix + paired init decision)
before the Phase-3 config freeze.

## Sol counter-review (same day) + disposition

Sol-tier fresh pass over the audit, the four full texts, the frozen
design, both findings docs, the implementation files, and the live
diffs (job `cx-20260717-110930-e764`). Headline: *"the audit is strong,
but its headline is too clean"* — every finding sustained, several
sharpened, and a code sweep found seven items the audit missed. All
quote-anchored sol claims were re-verified against the local texts
before acceptance (Minder's squared Latent-Scaling derivation; Fel's
α=1/ℓ with no stop-grad statement; 41/409,600 > 1e-4).

**Per-finding deltas (all applied in place above and in the design):**
F1 → Minder-as-written only; the paper is internally inconsistent
(squared Latent-Scaling derivation, later MSE language) and its trainer
is unchecked. F2 → strengthened: the ≥3M-token 4b pilot (~732 steps)
sits entirely inside the 1k-step warmup and structurally *cannot*
select lr or schedule — 0.9.5 is the only vehicle; add an encoder-decay
check. F3 → ≤40, not ≤41; the re-expression turned a recent-zero
detector into a rare-frequency classifier, and the batch-granular ring
buffer cannot express a literal token window — the "SASA arm" is an
approximation by construction. F4 → proportions unknown, soften
"cannot" to "likely underpowered or distributionally missed"; paired
init stays labeled a diffing prior. F5 → sustained; Fel also never
specifies stop-grad through the residual (unresolved paper detail, not
a bug). F6 → terminology: ‖z_g‖² is the energy, ‖z_g‖ its square root;
Anthropic's L1/L2 discussion is penalty-side — analogy, not
equivalence. F7 → **upgraded to moderate**: retained variance ≈
0.06–0.29 across sites means L_rec weights deep sites several times
more heavily — per-site RMS renorm after shrinkage whitening is now a
pre-4b-store decision item. F8 → resolved in the worktree; historical.
F9 → hardened: a faithful product-penalty ablation is a separately
designed method ablation (constraint, gauge, decay symmetry, selection
validity all change); no cheap arm anywhere. F10 → the 4b pilot
confirms only the warmup/early regime. F11 → separate unique-token
support (~297k/block) from optimization exposures (~594k at 2 epochs).

**Ground-truth corrections (applied):** Fel's b-grid is
{1,2,3,4,6,8,12,16,32}; SASA's "over one-fifth of the training" →
final-fifth is our labeled interpretation; the "confirmed faithful"
table title over-claimed for spec-only rows.

**Code-sweep findings S1–S7 (all fixed or worn, 2026-07-17):**

| # | Finding | Resolution |
|---|---|---|
| S1 | Calibrated θ never serialized — `latest.pt` reloads NaN; θ lived only in `report.json` | Driver re-saves the checkpoint after `fit_threshold_`; 0.9 errata note |
| S2 | Calibration/eval ran the fp32 master, not the bf16 forward copy — precision of the codec undeclared | fp32 master declared codec primary; single-pass bf16 shadow eval added per mode; 0.9 errata note |
| S3 | Corpus not pinned to an HF revision — short of the design's pinned-manifest bar | Harvest resolves and records the dataset sha, passes `revision=` to the stream |
| S4 | Held-out "covariance" is an uncentered second moment about the fit mean | Renamed + commented in the harvest script (correction negligible on fit-mean-centered data); 0.9 errata note |
| S5 | "Pairwise/Welford" design claim vs linear-fp64-within-quarters reality | Design wording amended to describe the actual algorithm (error ~5e-10, accepted) |
| S6 | Fel-init attribution too strong — per-block median equalization is BSC-specific | Docstring reworded to "Fel-inspired" |
| S7 | Pilot AuxK rationale stale post-respec (500-batch window → SASA default 100) | Design rationale refreshed; pilot justified on operational coverage + warmup-scope limit stated |

**0.9.5 verdict (sol): do it**, scoped to optimizer calibration + AuxK
stress characterization, with the corrected matrix: 8 lr×schedule arms
at the full 3,906-step horizon at λ=1e-3, winner confirmed at λ=0 (the
H3 regime); encoder-decay {0, 1e-3} at the best two settings; second
seed for winner and runner-up; dead-dynamics arm at **G=4096, k=32**
(preserves the Phase-1 k/G = 0.78% frequency ratio), with k=16 only as
a labeled stress arm; readouts: held-out FVU, train/eval gap, loss
slope, max grad norm, floor hits, Gram residual, dead occupancy /
churn / revival latency / relapse, AuxK gradient share, threshold
transfer. 1b rejects pathological criteria; the 4b pilot confirms the
surviving point. No product-penalty arm. θ persistence (S1) and eval
precision (S2) fixed before any addendum run — done.
