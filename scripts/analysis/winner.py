"""The dynamic winner pointer: which BSC the showcase figures are drawn from.

`data/phase0/winner.json` names the current best checkpoint (the ratified
operating point with the lowest pooled FVU at the largest budget). Figure
scripts call `load_winner()` instead of hardcoding a run, so promoting a
better run means editing one JSON and re-running the regeneration pass —
no script changes.
"""
from __future__ import annotations

import json
from pathlib import Path

WINNER_PATH = Path(__file__).resolve().parents[2] / "data" / "phase0" / "winner.json"


def load_winner() -> dict:
    w = json.loads(WINNER_PATH.read_text())
    w.pop("_comment", None)
    return w


if __name__ == "__main__":
    for k, v in load_winner().items():
        print(f"{k}: {v}")
