"""The dynamic winner pointer: which BSC the showcase figures are drawn from.

`data/phase0/winner.json` names the current best checkpoint (the ratified
operating point with the lowest pooled FVU at the largest budget) plus its
matched-budget primary-gauge counterpart (two-arm figures need both).
Figure scripts call `load_winner()` instead of hardcoding a run, so
promoting a better run means editing one JSON and re-running the
regeneration pass — no script changes.

Layout helpers keep every regenerated artifact scoped to the winner:

  analysis_dir()   /data/runs/bcc-analysis/<run_name> (override with
                   $BCC_ANALYSIS_DIR) — probe outputs, never clobbering
                   an earlier winner's artifacts
  figures_dir()    figures/phase0 — the canonical committed set
  load_showcase()  showcase_blocks.json in analysis_dir: the per-family
                   {arm, block, ring stats, qualified} map derived by
                   derive_showcase.py. Block indices are checkpoint-
                   specific; figures must look them up here, never
                   hardcode (mega-block rule: capture is only read with
                   ring order and FVU beside it).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

WINNER_PATH = Path(__file__).resolve().parents[2] / "data" / "phase0" / "winner.json"


def load_winner() -> dict:
    w = json.loads(WINNER_PATH.read_text())
    w.pop("_comment", None)
    return w


def analysis_dir(w: dict | None = None) -> Path:
    env = os.environ.get("BCC_ANALYSIS_DIR")
    if env:
        return Path(env)
    w = w or load_winner()
    return Path("/data/runs/bcc-analysis") / w["run_name"]


def figures_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "figures" / "phase0"


def load_showcase(w: dict | None = None) -> dict:
    """family -> {arm, block, capture stats, qualified} for the current winner."""
    p = analysis_dir(w) / "showcase_blocks.json"
    if not p.exists():
        raise SystemExit(
            f"missing {p} — run derive_showcase.py after the capture probes"
        )
    return json.loads(p.read_text())


def block_codes_path(arm: str = "winner", w: dict | None = None) -> Path:
    """Calendar-probe code export for one arm ('winner' | 'primary')."""
    w = w or load_winner()
    name = w["run_name"] if arm == "winner" else Path(w["counterpart_primary"]).name
    return analysis_dir(w) / f"block_codes_{name}.npz"


if __name__ == "__main__":
    for k, v in load_winner().items():
        print(f"{k}: {v}")
