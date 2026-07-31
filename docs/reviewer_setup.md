# Reviewer replication

The reviewer-facing path directly trains the selected BSC formula and emits a
single usable artifact. It does not run a CPU tour or replay the staged search.

## One copyable command

On a Linux machine with Python 3.12 and an NVIDIA driver:

```bash
python3.12 -m pip install "block-crosscoder-experiment @ git+https://github.com/a9lim/block-crosscoder-experiment.git" && bsc replicate
```

The package has one runtime dependency set; model capture, training, CUDA
kernels, calibration, and evaluation are all installed together.

The command uses public pinned inputs, so Hugging Face authentication is not
needed. It checks that PyTorch can see a bf16-capable CUDA device before
downloading the corpus.

## What it runs

The command resolves one seed of this fixed formula:

| Surface | Value |
|---|---|
| model and sites | GPT-2 Small residual-pre blocks 3/5/7/9 |
| normalization | scalar RMS |
| encoder | joint untied linear, no bias, availability-rescaled site sum |
| decoder | free-scale-controlled, no bias, concatenated-L2 geometry |
| site factorization | rank 4 |
| sparse code | signed, 2,048 groups × width 4, 8 active blocks |
| score and selector | decoded energy + block BatchTopK |
| site masking | none |
| optimizer | fused Adam, batch 512, learning rate `6e-4` |
| schedule | zero warmup, final-fifth linear decay to zero |
| regularizer | none |
| auxiliary | frequency-dead residual, weight 1, AuxK 8, dead frequency `1e-4`, 1,000-token window |
| train budget | 16M tokens |

This is an explicit extrapolation: the `6e-4` formula won the complete 4M
development audit, but this exact formula has not previously been trained or
confirmed at 16M. The generated artifact and result preserve that claim
boundary in plain text.

## Requirements and output

Allow roughly 215 GiB for the raw and scalar-RMS activation stores plus the
program's free-space margin. The resolved model estimate is about 3.8 GiB peak
training VRAM, while capture also holds GPT-2 Small. Any compatible
bf16-capable CUDA device may be selected with `--device cuda:N`.

The default output is:

```text
bsc-16m/
├── bsc-16m.pt
└── result.json
```

`bsc-16m.pt` contains the trained BSC weights, calibrated codec, scalar-RMS
normalization, and raw calibration mean. `result.json` contains the resolved
formula, runtime, training summary, evaluation, and qualification result.

Raw activations and campaign intermediates are removed after the artifact is
written. Add `--keep-work` to retain them. If capture or training is
interrupted, the partial work remains and the same run continues with:

```bash
bsc replicate --resume
```

Use another directory or seed when desired:

```bash
bsc replicate --out reviewer-seed-1 --seed 1
```

## Loading the artifact

```python
import torch
from block_crosscoder_experiment import load_artifact

artifact = load_artifact("bsc-16m/bsc-16m.pt", device="cuda")

# raw_activations has shape [tokens, 4, 768].
raw_activations = torch.randn(32, 4, 768, device="cuda")
packet = artifact.encode(raw_activations, q=8)
reconstructed = artifact.decode(packet)
```

For a no-download inspection of the exact plan and estimate:

```bash
bsc replicate --dry-run
```
