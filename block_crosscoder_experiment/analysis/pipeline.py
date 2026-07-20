"""Refresh all winner-scoped analysis artifacts and canonical figures.

Each checkpoint-heavy stage runs in a fresh subprocess. This is deliberate:
loading a checkpoint restores fp32 masters and optimizer state, so process
boundaries are the reliable way to release GPU memory before the next model.
The descriptive zoo is never a selection endpoint and this pipeline never
opens or inspects the sealed panel.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .artifacts import EVIDENCE_DIR, REPO_ROOT, analysis_dir, load_winner
from .catalog import ZOO_FAMILIES


def _run(command: str, *args: object, env: dict[str, str]) -> None:
    argv = [
        sys.executable,
        "-m",
        "block_crosscoder_experiment.cli",
        command,
        *(str(arg) for arg in args),
    ]
    print("\n+", " ".join(argv), flush=True)
    subprocess.run(argv, check=True, env=env)


def _blocks(tests_path: Path, arm: str) -> list[int]:
    tests = json.loads(tests_path.read_text())
    return sorted(
        {
            int(entry["best_block"])
            for entry in tests[arm].values()
            if isinstance(entry, dict) and "best_block" in entry
        }
    )


def main() -> None:
    winner = load_winner()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("all", "capture", "artifacts", "figures"),
        default="all",
        help="run the full pipeline or resume at one stage",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--analysis-dir", type=Path, default=None)
    parser.add_argument("--scan-tokens", type=int, default=8_000_000)
    parser.add_argument("--skip-docs", type=int, default=20_000)
    parser.add_argument("--per-class-cap", type=int, default=600)
    parser.add_argument(
        "--summary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also regenerate the compact cross-family PNG summaries",
    )
    args = parser.parse_args()

    artifact_dir = args.analysis_dir or analysis_dir(winner)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "BCC_ANALYSIS_DIR": str(artifact_dir)}
    winner_run = Path(winner["ckpt"]).parent
    primary_run = Path(winner["counterpart_primary"])
    runs = (f"winner={winner_run}", f"primary={primary_run}")
    acts = artifact_dir / "zoo_activations.npz"
    tests = artifact_dir / "zoo_block_tests.json"

    if args.stage in {"all", "capture"}:
        _run(
            "capture-zoo",
            "--device",
            args.device,
            "--out",
            artifact_dir,
            "--model",
            winner["model"],
            "--store",
            winner["store"],
            "--scan-tokens",
            args.scan_tokens,
            "--skip-docs",
            args.skip_docs,
            "--per-class-cap",
            args.per_class_cap,
            "--families",
            *ZOO_FAMILIES,
            env=env,
        )

    if args.stage in {"all", "artifacts"}:
        if not acts.exists():
            raise SystemExit(f"missing {acts}; run `bsc refresh-analysis --stage capture`")
        _run(
            "probe-families",
            "--acts",
            acts,
            "--store",
            winner["store"],
            "--tokenizer",
            winner["model"],
            "--device",
            args.device,
            "--out-dir",
            artifact_dir,
            "--runs",
            *runs,
            env=env,
        )
        _run("derive-showcase", env=env)
        shutil.copyfile(artifact_dir / "showcase_blocks.json", REPO_ROOT / "data" / "showcase.json")

        _run(
            "extract-geometry",
            "--device",
            args.device,
            "--out",
            artifact_dir,
            "--runs",
            *runs,
            "--sites",
            *winner["sites"],
            env=env,
        )
        _run(
            "activation-stats",
            "--device",
            args.device,
            "--out",
            artifact_dir,
            "--store",
            winner["store"],
            "--runs",
            *runs,
            env=env,
        )
        for arm, run in (("winner", winner_run), ("primary", primary_run)):
            _run(
                "dump-frames",
                "--run",
                run,
                "--blocks",
                *_blocks(tests, arm),
                "--out",
                artifact_dir / f"frames_{arm}.npz",
                env=env,
            )
        _run("probe-crossarm", env=env)

    if args.stage in {"all", "figures"}:
        _run("figures", "--analysis-dir", artifact_dir, env=env)
        if args.summary:
            _run("fig-capture", env=env)
            _run("fig-geometry", env=env)
            _run(
                "fig-rd-frontier",
                "--inputs",
                EVIDENCE_DIR / "rd_*.json",
                EVIDENCE_DIR / "f_*.json",
                EVIDENCE_DIR / "s_*.json",
                "--out",
                REPO_ROOT / "figures" / "summary" / "rate-distortion.png",
                env=env,
            )
            _run("fig-rd-tying", env=env)

    print(f"\nanalysis current: {artifact_dir}")
    print(f"figure catalog:   {REPO_ROOT / 'figures' / 'index.html'}")


if __name__ == "__main__":
    main()
