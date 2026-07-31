"""Train the selected BSC formula for 16M tokens on one CUDA device."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import torch

from block_crosscoder_experiment.artifact import (
    ARTIFACT_SCHEMA,
    DEFAULT_ARTIFACT_NAME,
)
from block_crosscoder_experiment.campaign import Campaign, CampaignRunner, RunState
from block_crosscoder_experiment.studies import (
    GPT2_CORPUS,
    GPT2_CORPUS_REVISION,
    GPT2_MODEL,
    GPT2_MODEL_REVISION,
    WINNING_FORMULA_TRAIN_TOKENS,
    build_winning_formula_plan,
    estimate_plan,
)

RESULT_SCHEMA = "bsc-winning-formula-result-v1"
CLAIM = (
    "Direct 16M-token extrapolation of the audited 4M development winner; "
    "this changed 6e-4 recipe was not previously validated at 16M."
)
HOOKS = tuple(f"blocks.{layer}.hook_resid_pre" for layer in (3, 5, 7, 9))
SPLITS = (
    ("normalization_fit", 250_000),
    ("calibration", 250_000),
    ("development", 1_000_000),
    ("confirmation", 1_000_000),
    ("train", WINNING_FORMULA_TRAIN_TOKENS),
)
FORMULA: dict[str, Any] = {
    "model": "GPT-2 Small",
    "sites": list(HOOKS),
    "normalization": "scalar_rms",
    "encoder": "joint_untied_linear_no_bias_availability_rescaled_sum",
    "decoder": "free_scale_controlled_no_bias_concat_l2",
    "site_rank": 4,
    "code": {
        "activation": "signed",
        "groups": 2_048,
        "block_width": 4,
        "active_blocks": 8,
        "score": "decoded_energy",
        "selector": "block_batchtopk",
    },
    "site_masking": "none",
    "optimizer": {
        "name": "adam",
        "fused": True,
        "batch_tokens": 512,
        "learning_rate": 6e-4,
        "warmup_fraction": 0.0,
        "schedule": "final_fifth_linear_to_zero",
    },
    "regularizer": "none",
    "auxiliary": {
        "name": "frequency_dead_residual",
        "weight": 1.0,
        "aux_k": 8,
        "dead_frequency": 1e-4,
        "dead_window_tokens": 1_000,
    },
    "train_tokens": WINNING_FORMULA_TRAIN_TOKENS,
}


def _human_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    for unit in units:
        if amount < 1024.0 or unit == units[-1]:
            return f"{amount:.1f} {unit}"
        amount /= 1024.0
    raise AssertionError("unreachable")


def _run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _capture(raw: Path, *, device: str, resume: bool) -> None:
    if (raw / "capture.json").is_file():
        print(f"Using completed activation capture at {raw}", flush=True)
        return
    command = [
        sys.executable,
        "-m",
        "block_crosscoder_experiment.cli.data",
        "capture",
    ]
    for hook in HOOKS:
        command.extend(
            ("--source", f"{GPT2_MODEL}|{GPT2_MODEL_REVISION}|{hook}")
        )
    command.extend(
        (
            "--corpus",
            GPT2_CORPUS,
            "--corpus-config",
            "plain_text",
            "--corpus-revision",
            GPT2_CORPUS_REVISION,
            "--tokenizer-contract",
            "gpt2-byte-bpe-files-v1",
            "--profile",
            "phase2",
        )
    )
    for name, tokens in SPLITS:
        command.extend(("--split", f"{name}={tokens}"))
    command.extend(("--device", device, "--out", str(raw)))
    if raw.exists():
        if not resume:
            raise RuntimeError(
                f"incomplete capture exists at {raw}; rerun with --resume"
            )
        command.append("--resume")
    _run(command)


def _derive(raw: Path, views: Path, *, resume: bool) -> Path:
    scalar = views / "scalar_rms"
    if (scalar / "view.json").is_file():
        print(f"Using completed scalar-RMS view at {scalar}", flush=True)
        return scalar
    command = [
        sys.executable,
        "-m",
        "block_crosscoder_experiment.cli.data",
        "derive",
        "--raw",
        str(raw),
        "--out",
        str(views),
        "--mode",
        "scalar_rms",
    ]
    if scalar.exists():
        if not resume:
            raise RuntimeError(
                f"incomplete derived view exists at {scalar}; rerun with --resume"
            )
        command.append("--resume")
    _run(command)
    return scalar


def _cuda_preflight(device: str) -> dict[str, Any]:
    parsed = torch.device(device)
    if parsed.type != "cuda":
        raise RuntimeError("the replication command requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "PyTorch cannot see CUDA; run this command on a Linux/NVIDIA machine"
        )
    index = torch.cuda.current_device() if parsed.index is None else parsed.index
    with torch.cuda.device(index):
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("the selected CUDA device does not support bf16")
    properties = torch.cuda.get_device_properties(index)
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": f"cuda:{index}",
        "gpu": properties.name,
        "vram_bytes": properties.total_memory,
    }


def _publish_artifact(
    deployment_path: Path,
    artifact_path: Path,
) -> dict[str, Any]:
    deployed = torch.load(deployment_path, map_location="cpu", weights_only=True)
    if not isinstance(deployed, dict):
        raise RuntimeError("trained deployment artifact is not a mapping")
    clean = {
        "schema": ARTIFACT_SCHEMA,
        "claim": CLAIM,
        "formula": FORMULA,
        "model_cfg": deployed["model_cfg"],
        "model_state": deployed["model_state"],
        "codec": deployed["codec_payload"],
        "normalization": deployed["normalization"],
        "raw_calibration_mean": deployed["raw_calibration_mean"],
        "training": deployed["training_summary"],
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = artifact_path.with_suffix(".pt.tmp")
    torch.save(clean, temporary)
    os.replace(temporary, artifact_path)
    return dict(clean["training"])


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected a JSON object at {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    plan = build_winning_formula_plan(args.seed)
    estimate = estimate_plan(plan)
    preview = {
        "claim": CLAIM,
        "formula": FORMULA,
        "seed": args.seed,
        "resources": estimate.to_dict(),
        "working_storage": _human_bytes(estimate.storage_bytes),
    }
    if args.dry_run:
        return preview

    output = args.out.expanduser().resolve()
    artifact_path = output / DEFAULT_ARTIFACT_NAME
    result_path = output / "result.json"
    if artifact_path.is_file() and result_path.is_file():
        return _read_json(result_path)
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise RuntimeError(f"{output} is incomplete; rerun with --resume")

    runtime = _cuda_preflight(args.device)
    output.mkdir(parents=True, exist_ok=True)
    print(
        f"Replicating the 16M formula on {runtime['gpu']}; "
        f"allow about {_human_bytes(estimate.storage_bytes)} of working storage.",
        flush=True,
    )
    work = output / "work"
    raw = work / "raw"
    views = work / "views"
    campaign_root = work / "campaign"
    _capture(raw, device=args.device, resume=args.resume)
    scalar_view = _derive(raw, views, resume=args.resume)

    campaign = Campaign(campaign_root)
    if not campaign.plan_path.is_file():
        campaign.register(
            plan,
            blueprint_manifest={
                "schema": "bsc-direct-replication-v1",
                "claim": CLAIM,
                "formula": FORMULA,
            },
        )
    summary = CampaignRunner(
        campaign,
        env={"BSC_ACTIVATION_STORE": str(scalar_view)},
    ).run(resume=args.resume)
    record = campaign.record(plan.cells[0].cell_id)
    if summary.failed_cells or record.state is not RunState.QUALIFIED:
        raise RuntimeError(
            f"training stopped in state {record.state.value}: {record.message}"
        )

    artifacts = record.artifact_map
    training = _publish_artifact(
        artifacts["deployment_codec"].resolve(campaign.root),
        artifact_path,
    )
    result = {
        "schema": RESULT_SCHEMA,
        "status": "complete",
        "claim": CLAIM,
        "artifact": str(artifact_path),
        "formula": FORMULA,
        "seed": args.seed,
        "runtime": runtime,
        "training": training,
        "evaluation": _read_json(
            artifacts["evaluation"].resolve(campaign.root)
        ),
        "qualification": _read_json(
            artifacts["qualification"].resolve(campaign.root)
        ),
    }
    _write_json(result_path, result)
    if not args.keep_work:
        shutil.rmtree(work)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("bsc-16m"),
        help="output directory (default: ./bsc-16m)",
    )
    parser.add_argument("--device", default="cuda", help="CUDA device (default: cuda)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue an interrupted capture, view derivation, or training run",
    )
    parser.add_argument(
        "--keep-work",
        action="store_true",
        help="retain raw activations, the derived view, and campaign files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the exact formula and resource estimate without requiring CUDA",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if sys.version_info[:2] != (3, 12):
        raise SystemExit(
            f"bsc replicate requires Python 3.12; found {platform.python_version()}"
        )
    try:
        result = run(args)
    except (
        KeyError,
        OSError,
        RuntimeError,
        subprocess.SubprocessError,
        TypeError,
        ValueError,
    ) as exc:
        raise SystemExit(f"replication stopped: {exc}") from exc
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
