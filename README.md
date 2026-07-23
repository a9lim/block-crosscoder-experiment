# Block-crosscoder experiment

Research code for testing whether one sparse block support with signed vector
coordinates can recover coherent multidimensional factors across layers of a
single language model.

The scientific program has three phases:

1. **Phase 1 — synthetic identification.** Establish factor recovery when the
   true support, subspace, and coordinates are known using one tiny, exactly
   matched orthogonal contract. Export only universal method semantics—not
   synthetic numeric hyperparameter winners.
2. **Phase 2 — small-model pilot.** Run a conditional GPT-2 Small campaign and
   own all model-, hook-, scale-, optimization-, and rate-dependent tuning,
   choosing among qualified candidates by raw-space FVU at exact total-bit
   budgets.
3. **Phase 3 — publishable artifact.** Freeze the Phase-2 decision, require
   every frozen design to pass a short production-shape fp32/bf16 stability
   cell, then train the five-seed Gemma panel without reopening architecture
   search.

This is a staged experiment, not a general-purpose sparse-autoencoder library.
The normative contract is [docs/design.md](docs/design.md); the live
paper-to-code matrix is [docs/paper_comparison.md](docs/paper_comparison.md).
The source procedure ledgers are
[docs/papers/block_methods.md](docs/papers/block_methods.md) and
[docs/papers/crosscoders.md](docs/papers/crosscoders.md).

## Decision rule

Phase 1 does not select on reconstruction alone. A factor must be associated
with a learned block, recover its subspace and aligned coordinates through that
same block, and pass split/merge/deadness guardrails. Phase 2 selects on the
negative mean raw-space FVU at the frozen 256, 384, and 512-bit/token budgets.
Packet bits and every byte of the deployable codec are priced. Adjacent points
on the measured lower convex rate–distortion envelope may be time-shared, but
their distortion is measured by reloading an exact serialized 32-byte schedule
record and executing it on the paired raw evaluation rows; extrapolation is
forbidden.
Candidates are aggregated across a complete seed set by median, then by the
worst seed, with content ID as the deterministic final tie-break.
Before panel freeze, the untouched scalar-RMS confirmation rerun must re-pass
qualification and sharing on every seed and remain within `0.02` fixed-rate
score of its exact development parent. This novel tolerance carries an
executed `.01/.02/.05` sensitivity report and an ungated descriptive result.

Phase 1 fixes one rank-two factor in four dimensions, exactly one width-two
learner block, and one active factor/block. Three one-site
instrument cells must pass before the same coordinates are rendered through
four aligned orthogonal site dictionaries. Confirmation reruns that
carrier beside support-only and one-site-span negative controls. There is no
synthetic capacity, optimizer, score, or selector tournament.

Phase 2 inherits that content-addressed method contract but independently
tunes real-model architecture, width/capacity/activity, site-axis rank,
masking, score, selector, optimizer, regularization, and Aux choices. It revisits
site-axis rank after the selected masking intervention, then measures the full
three-score by two-hard-selector interaction. Learned group thresholding is a
separate complete-method comparison at three coefficients, not another value
of the hard selector factor. The pilot retains a narrow QR/polar
tied-architecture comparison because retraction behavior can depend on real
activation conditioning, dimension, and numeric regime.
SASA's undisclosed absolute map-penalty coefficient is never guessed: its
paper bridge and main map arm fit a declared initial penalty/reconstruction
ratio on the content-bound first training batch, with independent targets
`0/.01/.03/.10`. The inspected decoder-only nuclear penalty remains a
nonpromotable release diagnostic.

Main-chain Phase-2 admission requires, for both site-only and leave-one-out inference, a
worst-site decoded-coordinate Lin concordance of at least `.80` in the
all-site decoder-Gram geometry, including the mean-offset penalty, plus
worst-site support-intersection recall of at least `.75` and decoded-energy
coverage of at least `.90`. Parent- and root-relative partial-view FVU,
support-IoU, and absolute-FVU safety guards remain conjunctive. All-view FVU
advantage is reported only as a descriptive quantity; redundancy can be a
legitimate shared feature.
Comparator-family calibration reports these same endpoints but does not gate
on them, preserving deliberately non-sharing baselines for the frozen Phase-3
comparison. Its top-two union deduplicates resolved non-replicate execution
signatures while preserving every stage/candidate alias.
Every comparator family independently calibrates the same four peak learning
rates: `3e-5`, `1e-4`, `2e-4`, and `3e-4`.

Phase 2 cells use their declared bf16 forward precision. The executable
fp32/bf16 parity-and-short-run stability gate is a Phase-3-only preflight; the
pilot does not claim to have passed that later production-shape gate.

At the default seeds, Phase 1 declares and executes **15 cells**. Phase 2 has
a **410-cell pre-elision ceiling**: 176 main-chain cells and 234 independently
calibrated comparator-family cells. At materialization, execution-equivalent
parent/center variants are deterministically elided and recorded. The rank
revisit keeps only its parent when zero Bernoulli masking wins, so the realized
Phase-2 count is lower and evidence-dependent. The Appendix-D runner-up arm is
also elided when the selected parent is not fixed per-token TopK. These values
are derived from the serialized blueprints rather than maintained as parallel
constants.
Phase 3 projects eight short production-stability cells plus the frozen 40-cell
final panel. Its rate budgets are `1024/1536/2048` bits/token: the explicit
fourfold transfer of the Phase-2 frontier matching `128/32` nominal active
coordinates.

The Phase-3 envelope is portable evidence, not merely a collection of hashes:
it reruns the live selection reducer over embedded qualifications, including
all exclusions, gates, threshold-sensitivity surfaces, and comparator
nomination deduplication. Only a replay-verified next-ranked nonduplicate may
replace a colliding comparator winner.

Every resolved choice is labeled `exact`, `adapted`, `engineering`, or `novel`.
Adapted and novel scientific choices carry a rationale and a falsifying
ablation. Paper prose, inspected release behavior, and local adaptations are
never silently merged. Winner-changing practical-effect, noninferiority, and
sharing thresholds are novel preregistered project policies, not paper values;
their rationale and complete sensitivity grid are content-bound with each
applicable selection policy.

## Command surface

```bash
bsc --help
bsc matrix --help
bsc data --help
bsc cell --help

# Inspect the currently materialized prefix and the full conditional projection.
bsc matrix estimate --phase phase1 --seeds 0 1 2
bsc matrix estimate --phase phase2 --seeds 0 1

# Register, execute, select, and append one declared round at a time.
bsc matrix plan --root /data/runs/bsc-phase1 --phase phase1 --seeds 0 1 2
bsc matrix run --root /data/runs/bsc-phase1
bsc matrix status --root /data/runs/bsc-phase1
bsc matrix select --root /data/runs/bsc-phase1 \
  --stage multisite_learnability --out /data/runs/bsc-phase1/truth-selection.json
bsc matrix advance --root /data/runs/bsc-phase1 \
  --selection /data/runs/bsc-phase1/truth-selection.json
# Qualify the truth-contract confirmation stage, then freeze the authenticated go/no-go and
# universal-method and provisional-carrier transfer contract.
bsc matrix freeze-phase1 --root /data/runs/bsc-phase1 \
  --out /data/runs/bsc-phase1/decisions/phase2-authorization.json

# Phase-2 comparator families branch independently from their own anchor.
bsc matrix select-family-root --root /data/runs/bsc-phase2 \
  --family bsf_grassmannian --out /data/runs/bsc-phase2/bsf-root.json
bsc matrix advance-family --root /data/runs/bsc-phase2 \
  --family bsf_grassmannian --selection /data/runs/bsc-phase2/bsf-root.json
# Qualify the emitted family round, use `matrix select --stage <emitted-stage>`,
# and repeat `advance-family` until its declared 4M rounds are complete.
bsc matrix nominate-family-revisit --root /data/runs/bsc-phase2 \
  --family bsf_grassmannian --out /data/runs/bsc-phase2/bsf-top2.json
bsc matrix revisit-family --root /data/runs/bsc-phase2 \
  --family bsf_grassmannian --selection /data/runs/bsc-phase2/bsf-top2.json
# After the 16M revisit qualifies, this freezes that family finalist:
# bsc matrix select --root /data/runs/bsc-phase2 \
#   --stage family_bsf_grassmannian_top2_revisit_16m

# After every Phase-2 round and confirmation cell is complete, freeze the
# evidence-bound panel and consume it in Phase 3.
bsc matrix freeze-panel --root /data/runs/bsc-phase2 \
  --out /data/runs/bsc-phase2/decisions/phase3-panel.json
bsc matrix plan --root /data/runs/bsc-phase3 --phase phase3 \
  --panel-decision /data/runs/bsc-phase2/decisions/phase3-panel.json

# Tiny schema-complete CPU execution profile.
bsc matrix plan --root /tmp/bsc-smoke --phase phase1 --seeds 0 --smoke
bsc matrix run --root /tmp/bsc-smoke --limit 1
```

Plans and cells are canonical-JSON content addressed. The campaign journal is
append-only, and its legal lifecycle is:

```text
planned -> prepared -> running -> trained -> calibrated -> evaluated
                                                          -> qualified
                                                          -> promoted
```

Qualification means that the evidence bundle is complete and internally
consistent. Scientific outcome and promotion eligibility are separate fields;
a well-recorded negative result is still valid evidence.

A smoke campaign may freeze and consume protocol-only selections and a
protocol-only panel through a smoke Phase 3. Those artifacts remain
non-scientific and are rejected if supplied to a non-smoke Phase-3 campaign.

The journal is append-only through the campaign API, but before a freeze it is
not an externally authenticated origin ledger. A writer with filesystem access
could replace the journal and evidence together. Freeze replays and
content-addresses the current internally consistent evidence; it does not prove
who produced the pre-freeze bytes. Protect the campaign directory operationally
and preserve the frozen decision outside that trust boundary when provenance
against a malicious local writer matters.

## Activation-store runbook

Capture every requested hook from one pinned model and one packed-token pass.
Capture preflight resolves the model and corpus revisions, loads the reviewed
slow tokenizer at that exact revision, checks its class/BOS/vocabulary hash,
and passes that tokenizer explicitly into TransformerLens. Store format v3
binds `int64` row identities and writes an incomplete, hash-bound manifest after
every durable shard. If capture stops, repeat the identical command with
`--resume`. The phase profile fixes the complete role set and canonical
allocation order, independent of `--split` argument order; changed code,
dependencies, sources, profile, role sizes, resulting binding, or shard
geometry are refused.
The completed `capture.json` embeds the full binding. Verification recomputes
its canonical digest and cross-checks implementation, allocation, geometry,
and the complete profile role set; an empty directory or a digest-shaped string
without matching content is not a successful verification.

Phase 2 capture and materialized views:

```bash
bsc data estimate \
  --split normalization_fit=250000 --split calibration=250000 \
  --split development=1000000 --split confirmation=1000000 \
  --split train=16000000 \
  --site-dim 768 --site-dim 768 --site-dim 768 --site-dim 768

bsc data capture \
  --profile phase2 \
  --source 'openai-community/gpt2|607a30d783dfa663caf39e06633721c8d4cfcd7e|blocks.3.hook_resid_pre' \
  --source 'openai-community/gpt2|607a30d783dfa663caf39e06633721c8d4cfcd7e|blocks.5.hook_resid_pre' \
  --source 'openai-community/gpt2|607a30d783dfa663caf39e06633721c8d4cfcd7e|blocks.7.hook_resid_pre' \
  --source 'openai-community/gpt2|607a30d783dfa663caf39e06633721c8d4cfcd7e|blocks.9.hook_resid_pre' \
  --corpus Skylion007/openwebtext \
  --corpus-revision b4325f019c648b1641a1784748667e8b74e5e064 \
  --corpus-config plain_text --context 128 \
  --tokenizer-contract gpt2-byte-bpe-files-v1 \
  --store-contract-version activation-store-v3-derived-views \
  --split normalization_fit=250000 --split calibration=250000 \
  --split development=1000000 --split confirmation=1000000 \
  --split train=16000000 --out /data/stores/bsc-gpt2-raw

# Only after an interrupted invocation; every other argument must be identical.
# bsc data capture --profile phase2 ... \
#   --out /data/stores/bsc-gpt2-raw --resume

bsc data derive --raw /data/stores/bsc-gpt2-raw \
  --out /data/stores/bsc-gpt2-views \
  --mode none --mode scalar_rms --mode sqrt_d --mode whiten --mode layer
# After interruption, repeat the identical derive command with --resume.
bsc data verify --store /data/stores/bsc-gpt2-raw

export BSC_RAW_STORE_ROOT=/data/stores/bsc-gpt2-raw
export BSC_VIEW_ROOT=/data/stores/bsc-gpt2-views
bsc matrix plan --root /data/runs/bsc-phase2 --phase phase2 --seeds 0 1 \
  --phase1-decision \
    /data/runs/bsc-phase1/decisions/phase2-authorization.json
bsc matrix run --root /data/runs/bsc-phase2 \
  --view-root "$BSC_VIEW_ROOT"
```

Capture, derived-view materialization, and transform fitting each hold one
nonblocking producer lock for their output root; a second writer is refused
with the recorded host and PID. Derived-view resume reuses only a fully
hash-verified, source/transform-bound prefix of complete split manifests. A
partial split or foreign output fails closed and names the exact split that
must be reviewed before any external cleanup. Store verification also checks
the raw-source transform binding, canonical sequence/position allocation, and
structural site dimensions rather than treating payload checksums alone as a
complete capture. Each derived mode has one exact-key `view.json` verification
surface: it embeds and reauthenticates the complete capture evidence, binds the
Whitener and every split/content/row digest, and refuses missing or foreign
members. Derive and transform fitting refuse a source whose complete
`capture.json` binding and raw role set do not verify.

Every data producer performs a prewrite check on the filesystem that will
actually receive the bytes. Capture prices the exact whole-sequence row
allocation and, on resume, only the verified remaining rows; derive prices each
missing physical padded split; transform fitting prices its content-addressed
tensor artifact. These checks preserve the writer's 15% free-space floor and
report raw free bytes separately from bytes available above that floor.

`--view-root` is a dispatcher, not a scientific override. Before any campaign
transition it checks each `<mode>/whitener.pt`, every required self-hashed split
manifest, and cross-mode row-stream identity, then passes the exact view chosen
by that cell's immutable `data.normalization` decision. The cell executor still
checks the complete source contract and either rehashes payloads or reuses a
stat-bound verification receipt whose file identity and timestamps are exact.
Phase 2 currently
materializes the `none` view as well as the raw store because every derived-view
cell binds a frozen transform artifact uniformly; a raw-as-identity alias is
not silently substituted or credited by the estimator.

Phase 3 captures one raw view and stores only content-addressed transforms:

```bash
bsc data estimate \
  --split normalization_fit=250000 --split calibration=250000 \
  --split stability=250000 --split final=2000000 --split train=25000000 \
  --site-dim 2560 --site-dim 2560 --site-dim 2560 --site-dim 2560

bsc data capture \
  --profile phase3 \
  --source 'google/gemma-3-4b-pt|cc012e0a6d0787b4adcc0fa2c4da74402494554d|blocks.8.hook_resid_pre' \
  --source 'google/gemma-3-4b-pt|cc012e0a6d0787b4adcc0fa2c4da74402494554d|blocks.14.hook_resid_pre' \
  --source 'google/gemma-3-4b-pt|cc012e0a6d0787b4adcc0fa2c4da74402494554d|blocks.20.hook_resid_pre' \
  --source 'google/gemma-3-4b-pt|cc012e0a6d0787b4adcc0fa2c4da74402494554d|blocks.26.hook_resid_pre' \
  --corpus HuggingFaceFW/fineweb-edu --corpus-config sample-10BT \
  --corpus-revision 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9 \
  --context 512 --tokenizer-contract gemma3-tokenizer-files-v1 \
  --store-contract-version activation-store-v3-single-view \
  --split normalization_fit=250000 --split calibration=250000 \
  --split stability=250000 --split final=2000000 --split train=25000000 \
  --out /data/stores/bsc-gemma-raw

bsc data fit-transform --raw /data/stores/bsc-gemma-raw \
  --out /data/stores/bsc-gemma-raw/transforms --mode scalar_rms
bsc data verify --store /data/stores/bsc-gemma-raw

export BSC_RAW_STORE_ROOT=/data/stores/bsc-gemma-raw
export BSC_TRANSFORM_ROOT=/data/stores/bsc-gemma-raw/transforms
bsc matrix plan --root /data/runs/bsc-phase3 --phase phase3 \
  --panel-decision /data/runs/bsc-phase2/decisions/phase3-panel.json
bsc matrix run --root /data/runs/bsc-phase3
```

Matrix estimates label whether they price the current conditional prefix or a
complete frozen panel. Planning computes an incremental storage requirement:
configured `BSC_VIEW_ROOT`, `BSC_ACTIVATION_STORE`, `BSC_RAW_STORE_ROOT`, and
`BSC_TRANSFORM_ROOT` inputs receive credit only after their manifests and
content hashes verify **and** their source/allocation/view contract matches the
materialized plan. Phase 1 receives no activation-store credit. Already
materialized recorded campaign artifacts receive only their verified
physical-byte credit. `matrix run` repeats this live check immediately before
execution; the planning-only insufficient-storage override cannot bypass it.
If any selected cell fails, the command exits nonzero even though it still
prints the complete run summary. The full scientific estimate remains
unchanged.

The incremental check is device-local: campaign artifacts are charged to the
campaign filesystem, activation inputs are charged only against authenticated
input credit, same-device roles are summed, and free space on one device never
authorizes another. An explicit `matrix run --view-root` participates in this
binding even when no environment variable names it. Because future raw/view
output placement is not a scientific plan field, any unmaterialized activation
remainder is conservatively required on every declared input-destination
filesystem. The planning-only override may acknowledge that conservative
preparation state; a scientific launch is expected to have complete verified
inputs, so this remainder is normally zero.

Canonical cell execution persists store-verification receipts only under the
authenticated campaign root; `BSC_VERIFICATION_CACHE_ROOT` is rejected. Receipt
reuse still reads a deterministic bounded content probe from every shard in
addition to matching its exact stat fingerprint. The probe narrows the
silent-media-corruption gap between full hashes; it is not a replacement for
the initial complete checksum and is not claimed as whole-store proof after
that point. CUDA cells also take one
host-global per-device advisory lock for the persistent worker lifetime, so
separate campaigns cannot load training/evaluation work concurrently on the
same GPU. `matrix run --limit` must be positive and cannot be combined with
explicit `--cell`; `--resume --view-root` includes both failed and interrupted
cells. Every launch also replays the current phase resource ceilings against
the loaded plan before its storage check. During dispatch, SIGTERM is converted
to stack unwinding so the runner closes the worker process group before the CLI
exits with the conventional signal status.

The 1 TB `/data` volume should not be assumed to hold both campaigns. After the
Phase-2 panel is frozen and its checkpoints, codecs, evaluations, decisions,
logs, and evidence envelope are independently backed up, verify the raw and
derived stores one last time and move reproducible Phase-2 activation stores to
another volume before Phase-3 capture. The campaign never performs this move or
deletes recorded final checkpoints/stores itself; external deletion is an
operator decision, and any later verifier will reject a recorded artifact that
is missing. Re-run planning after the handoff so the incremental preflight
reflects the live filesystem.

## References and environment

`references/refs.yaml` is the committed bibliography. Regenerate the local,
gitignored Markdown corpus through the workspace utility:

```bash
python -m transformer_experiments.references
```

Use the workspace's shared plain Python 3.12 environment. GPU capture and
training run on `jobe`; do not create a project virtual environment.

```bash
python -m pytest -q
python -m compileall -q block_crosscoder_experiment
git diff --check
```
