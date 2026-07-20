"""Paths and provenance for winner-scoped analysis artifacts.

``data/winner.json`` is the only model pointer used by current analysis.
Promoting a checkpoint changes that file; probes and figures then follow it
without hard-coded run names or block identities.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WINNER_PATH = REPO_ROOT / "data" / "winner.json"
EVIDENCE_DIR = REPO_ROOT / "data" / "evidence"
FIGURES_DIR = REPO_ROOT / "figures"


def load_winner() -> dict:
    w = json.loads(WINNER_PATH.read_text())
    w.pop("_comment", None)
    return w


def analysis_dir(w: dict | None = None) -> Path:
    """Return the artifacts for the promoted winner.

    ``BCC_ANALYSIS_DIR`` always wins. A local ignored cache is preferred when
    present (useful on the Mac); otherwise the canonical jobe run directory is
    returned.
    """
    env = os.environ.get("BCC_ANALYSIS_DIR")
    if env:
        return Path(env)
    w = w or load_winner()
    local = REPO_ROOT / "data" / "analysis" / "current"
    if local.exists():
        return local
    return Path("/data/runs/bcc-analysis") / w["run_name"]


def figures_dir() -> Path:
    return FIGURES_DIR


def summary_dir() -> Path:
    return FIGURES_DIR / "summary"


def family_dir(family: str) -> Path:
    return FIGURES_DIR / family


def load_showcase(w: dict | None = None) -> dict:
    """family -> {arm, block, capture stats, qualified} for the current winner."""
    p = analysis_dir(w) / "showcase_blocks.json"
    if not p.exists():
        p = REPO_ROOT / "data" / "showcase.json"
    if not p.exists():
        raise SystemExit(
            f"missing {p} — run `bsc derive-showcase` after the capture probes"
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
