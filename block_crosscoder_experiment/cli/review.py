"""Run the portable, download-free reviewer walkthrough."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Sequence

import torch

from block_crosscoder_experiment.campaign import Campaign, CampaignRunner
from block_crosscoder_experiment.studies import (
    FrozenSelection,
    build_phase1_blueprint,
    build_phase1_plan,
    build_phase2_blueprint,
    build_phase2_plan,
    build_phase3_blueprint,
    estimate_plan,
    materialize_child_plan,
)


def _package_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for name in ("block-crosscoder-experiment", "numpy", "safetensors", "torch"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "editable source"
    return result


def _run_open_cells(campaign: Campaign) -> dict[str, int]:
    totals = {
        "selected_cells": 0,
        "completed_cells": 0,
        "failed_cells": 0,
        "skipped_cells": 0,
    }
    while int(campaign.status()["runnable"]) > 0:
        summary = CampaignRunner(campaign).run()
        for key, value in summary.to_dict().items():
            totals[key] += int(value)
        if summary.failed_cells:
            raise RuntimeError(
                f"portable review failed; inspect the cell logs under {campaign.root}"
            )
        if not summary.completed_cells:
            raise RuntimeError("portable review made no progress")
    return totals


def run_review(root: Path) -> dict[str, Any]:
    """Exercise the real Phase-1 lifecycle and load every phase definition."""

    phase1_blueprint = build_phase1_blueprint((0,), smoke=True)
    campaign = Campaign(root)
    campaign.register(
        build_phase1_plan((0,), smoke=True),
        blueprint_manifest=phase1_blueprint.to_manifest(),
    )
    first_pass = _run_open_cells(campaign)

    source_stage = phase1_blueprint.rounds[0].source_stage
    selection_path = root / "selections" / f"{source_stage}.json"
    selection_payload = campaign.select_stage(source_stage, out=selection_path)
    selection = FrozenSelection.from_dict(selection_payload["selected"][0])
    campaign.extend(
        materialize_child_plan(campaign.plan, phase1_blueprint, selection),
        selection=selection,
        selection_path=selection_path,
    )
    confirmation_pass = _run_open_cells(campaign)
    decision = campaign.freeze_phase1_decision()

    phase2_blueprint = build_phase2_blueprint((0,), smoke=True)
    phase2_plan = build_phase2_plan((0,), smoke=True)
    phase3_blueprint = build_phase3_blueprint((0,), smoke=True)

    return {
        "status": "passed",
        "root": str(root.resolve()),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch_device": (
                "cuda"
                if torch.cuda.is_available()
                else (
                    "mps"
                    if torch.backends.mps.is_available()
                    else "cpu"
                )
            ),
            "packages": _package_versions(),
        },
        "phase1": {
            "qualified_cells": campaign.status()["counts"].get("qualified", 0),
            "first_pass": first_pass,
            "confirmation_pass": confirmation_pass,
            "protocol_complete": True,
            "smoke_handoff_open": bool(decision["go"]),
            "scientific_go": bool(decision["scientific_go"]),
            "decision": str(
                (root / "decisions" / "phase2-authorization.json").resolve()
            ),
        },
        "phase2": {
            "definition_loaded": True,
            "initial_cells": len(phase2_plan.cells),
            "declared_cell_ceiling": phase2_blueprint.to_manifest()[
                "declared_cell_ceiling"
            ],
            "initial_estimate": estimate_plan(phase2_plan).to_dict(),
        },
        "phase3": {
            "definition_loaded": True,
            "panel_slots": len(phase3_blueprint.panel_slots),
            "projected_smoke_cells": phase3_blueprint.to_manifest()[
                "projected_cells"
            ],
        },
        "note": (
            "The portable walkthrough uses tiny synthetic CPU data. "
            "Real-model capture and publication-scale training require the "
            "full/CUDA setup described in docs/reviewer_setup.md."
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        help=(
            "artifact directory; the default is a new temporary directory that "
            "is retained for inspection"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if sys.version_info[:2] != (3, 12):
        raise SystemExit(
            f"bsc review requires Python 3.12; found {platform.python_version()}"
        )
    root = args.root or Path(tempfile.mkdtemp(prefix="bsc-review-"))
    try:
        payload = run_review(root)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise SystemExit(f"review failed: {exc}") from exc
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
