# Reviewer setup

The repository has two honest execution paths:

1. a portable CPU walkthrough that exercises installation, campaign state,
   training, calibration, evaluation, qualification, selection, and all three
   phase definitions with tiny data;
2. the full real-model experiment, which additionally needs Linux, an NVIDIA
   GPU, model downloads, and substantial storage.

Native macOS and Linux are supported for the CPU walkthrough. On Windows, use
WSL because the campaign uses POSIX process and file-lock semantics.

## Fresh CPU review

Install Python 3.12, then clone into a disposable virtual environment:

```bash
git clone https://github.com/a9lim/block-crosscoder-experiment.git
cd block-crosscoder-experiment
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[review]"
```

Run the complete download-free walkthrough:

```bash
bsc review
```

The command retains its temporary run directory and prints its path. It runs
five real Phase-1 smoke cells through the complete lifecycle, selects the
multisite carrier, runs the confirmation variants, writes the Phase-1
decision, and loads and estimates the Phase-2 and Phase-3 definitions. A
scientific `go` is not required: the tiny walkthrough tests execution, not the
research conclusion.

Useful manual checks are:

```bash
bsc --help
bsc matrix estimate --phase phase1 --smoke
python -m pytest -q
```

The test suite is optional for using the package. It is included in the
`review` extra so a reviewer can inspect or run the focused checks.

## Real-model and CUDA setup

The publication-scale path is Linux/NVIDIA only. Install the Python 3.12
environment above, then:

1. Install the appropriate stable CUDA build of PyTorch using the official
   [PyTorch selector](https://pytorch.org/get-started/locally/).
2. Install the model/data and CUDA extras:

   ```bash
   python -m pip install -e ".[full,cuda,review]"
   ```

3. Confirm the actual runtime:

   ```bash
   python - <<'PY'
   import torch
   print("torch:", torch.__version__)
   print("CUDA build:", torch.version.cuda)
   print("CUDA available:", torch.cuda.is_available())
   if torch.cuda.is_available():
       print("GPU:", torch.cuda.get_device_name(0))
   PY
   ```

4. Authenticate with Hugging Face:

   ```bash
   hf auth login
   hf auth whoami
   ```

   Phase 3 uses
   [`google/gemma-3-4b-pt`](https://huggingface.co/google/gemma-3-4b-pt),
   whose files require accepting Google's Gemma usage license while logged in.

5. Put the Hugging Face cache and run roots on storage with enough free space:

   ```bash
   export HF_HOME=/path/with/space/hf-cache
   export BSC_RUNS=/path/with/space/bsc-runs
   mkdir -p "$HF_HOME" "$BSC_RUNS"
   ```

Always inspect the estimate before materializing a scientific plan:

```bash
bsc matrix estimate --phase phase1
bsc data estimate \
  --split normalization_fit=250000 \
  --split calibration=250000 \
  --split development=1000000 \
  --split confirmation=1000000 \
  --split train=16000000 \
  --site-dim 768 --site-dim 768 --site-dim 768 --site-dim 768 \
  --views 1
```

The program refuses a plan that exceeds its declared resource ceilings.

## Manual Phase-1 smoke workflow

`bsc review` automates this sequence. The explicit commands are useful when
reviewing each handoff:

```bash
ROOT=tmp/manual-phase1-review

bsc matrix plan --phase phase1 --smoke --seeds 0 --root "$ROOT"
bsc matrix run --root "$ROOT"
bsc matrix run --root "$ROOT"
bsc matrix select \
  --root "$ROOT" \
  --stage multisite_learnability \
  --out "$ROOT/selections/multisite_learnability.json"
bsc matrix advance \
  --root "$ROOT" \
  --selection "$ROOT/selections/multisite_learnability.json"
bsc matrix run --root "$ROOT"
bsc matrix freeze-phase1 --root "$ROOT"
bsc matrix status --root "$ROOT"
```

The scientific path uses the same commands without `--smoke` and with the
declared seeds.

## Phase-2 capture and launch shape

Phase 2 captures GPT-2 Small residual-pre activations at blocks 3, 5, 7, and
9. The smoke-sized form below downloads the real model but keeps every split
tiny:

```bash
RAW="$BSC_RUNS/phase2-smoke-raw"
VIEWS="$BSC_RUNS/phase2-smoke-views"
GPT2_REV=607a30d783dfa663caf39e06633721c8d4cfcd7e
OWT_REV=b4325f019c648b1641a1784748667e8b74e5e064

bsc data capture \
  --source "openai-community/gpt2|$GPT2_REV|blocks.3.hook_resid_pre" \
  --source "openai-community/gpt2|$GPT2_REV|blocks.5.hook_resid_pre" \
  --source "openai-community/gpt2|$GPT2_REV|blocks.7.hook_resid_pre" \
  --source "openai-community/gpt2|$GPT2_REV|blocks.9.hook_resid_pre" \
  --corpus Skylion007/openwebtext \
  --corpus-revision "$OWT_REV" \
  --corpus-config plain_text \
  --tokenizer-contract gpt2-byte-bpe-files-v1 \
  --profile phase2 \
  --split normalization_fit=64 \
  --split calibration=64 \
  --split development=64 \
  --split confirmation=64 \
  --split train=64 \
  --device cuda \
  --out "$RAW"

bsc data derive \
  --raw "$RAW" \
  --out "$VIEWS" \
  --mode scalar_rms

bsc matrix plan \
  --phase phase2 \
  --smoke \
  --seeds 0 \
  --phase1-decision tmp/manual-phase1-review/decisions/phase2-authorization.json \
  --root "$BSC_RUNS/phase2-smoke"

bsc matrix run \
  --root "$BSC_RUNS/phase2-smoke" \
  --view-root "$VIEWS"
```

Phase 2 is conditional: after each completed development stage, use
`bsc matrix select` and `bsc matrix advance`. Comparator families use the
corresponding `select-family-root` and `advance-family` commands. Run
`bsc matrix --help` for the deliberately small command surface.

After confirmation and comparator calibration are complete:

```bash
bsc matrix freeze-panel --root "$BSC_RUNS/phase2-smoke"
```

That panel decision opens the Phase-3 smoke plan:

```bash
RAW3="$BSC_RUNS/phase3-smoke-raw"
GEMMA_REV=cc012e0a6d0787b4adcc0fa2c4da74402494554d
FINEWEB_REV=87f09149ef4734204d70ed1d046ddc9ca3f2b8f9

bsc data capture \
  --source "google/gemma-3-4b-pt|$GEMMA_REV|blocks.8.hook_resid_pre" \
  --source "google/gemma-3-4b-pt|$GEMMA_REV|blocks.14.hook_resid_pre" \
  --source "google/gemma-3-4b-pt|$GEMMA_REV|blocks.20.hook_resid_pre" \
  --source "google/gemma-3-4b-pt|$GEMMA_REV|blocks.26.hook_resid_pre" \
  --corpus HuggingFaceFW/fineweb-edu \
  --corpus-revision "$FINEWEB_REV" \
  --corpus-config sample-10BT \
  --tokenizer-contract gemma3-tokenizer-files-v1 \
  --store-contract-version activation-store-v3-single-view \
  --profile phase3 \
  --split normalization_fit=64 \
  --split calibration=64 \
  --split stability=64 \
  --split final=64 \
  --split train=64 \
  --context 512 \
  --device cuda \
  --out "$RAW3"

bsc data fit-transform \
  --raw "$RAW3" \
  --out "$RAW3/transforms" \
  --mode scalar_rms

bsc matrix plan \
  --phase phase3 \
  --smoke \
  --seeds 0 \
  --panel-decision "$BSC_RUNS/phase2-smoke/decisions/phase3-panel.json" \
  --root "$BSC_RUNS/phase3-smoke"

BSC_RAW_STORE_ROOT="$RAW3" \
BSC_TRANSFORM_ROOT="$RAW3/transforms" \
bsc matrix run --root "$BSC_RUNS/phase3-smoke"
```

Run the Phase-3 command again after the stability cells qualify to execute the
already-frozen final panel. The full split sizes and model bindings are
authoritative in `block_crosscoder_experiment/studies.py`; use
`bsc data estimate` before capture and `bsc matrix estimate` before planning.

## What the portable walkthrough does not claim

The CPU walkthrough proves that a fresh installation can execute the package
and its staged lifecycle. It does not reproduce the GPT-2/Gemma results.
Those require the real-model inputs, GPU kernels, and declared scientific
budgets above.
