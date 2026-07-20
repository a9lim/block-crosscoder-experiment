"""Evaluate every completed cell in a manifold sweep with one frozen probe."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--acts", type=Path, required=True)
    parser.add_argument("--background-acts", type=Path, default=None)
    parser.add_argument("--families", nargs="+", required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    reports = sorted(args.campaign_root.glob("*/seed*/*/report.json"))
    if not reports:
        raise SystemExit(f"no completed cells under {args.campaign_root}")
    runs = []
    for report in reports:
        relative = report.parent.relative_to(args.campaign_root)
        cell, seed = relative.parts[:2]
        runs.append(f"{cell}_{seed}={report.parent}")

    command = [
        sys.executable,
        "-m",
        "block_crosscoder_experiment.cli",
        "eval-manifolds",
        "--acts", str(args.acts),
        "--store", str(args.store),
        "--tokenizer", args.tokenizer,
        "--device", args.device,
        "--out", str(args.out),
        "--families", *args.families,
        "--runs", *runs,
    ]
    if args.background_acts is not None:
        command.extend(("--background-acts", str(args.background_acts)))
    print(f"evaluating {len(runs)} completed cells", flush=True)
    raise SystemExit(subprocess.run(command, check=False).returncode)


if __name__ == "__main__":
    main()
