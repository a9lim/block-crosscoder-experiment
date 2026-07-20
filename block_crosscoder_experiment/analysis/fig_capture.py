"""Compact winner-level capture and site-allocation summary figures.

The interactive per-family catalog owns all class geometry.  This module
retains only two non-duplicative global summaries: which winner block claims
each family class and how the winner's per-site distortion compares with its
matched primary-gauge counterpart.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import style as st
from .artifacts import analysis_dir, load_winner, summary_dir
from .catalog import ZOO

W = None
DATA = None
OUT = None
SITES = []
ARMS = {}


def _configure() -> None:
    global W, DATA, OUT, SITES, ARMS
    W = load_winner()
    DATA = analysis_dir(W)
    OUT = summary_dir()
    SITES = W["sites"]
    ARMS = {
        "site-renorm winner": (W["run_name"], Path(W["ckpt"]).parent),
        "primary gauge": (
            Path(W["counterpart_primary"]).name,
            Path(W["counterpart_primary"]),
        ),
    }


def _tests() -> dict:
    path = DATA / "zoo_block_tests.json"
    if not path.exists():
        raise SystemExit(f"missing {path}; run `bsc refresh-analysis --stage artifacts`")
    return json.loads(path.read_text())


def fig_capture(tests: dict) -> None:
    """Draw the top-1 block assignment for every class in every family."""

    winner_key = W["run_name"]
    source = tests.get(winner_key, tests.get("winner"))
    if source is None:
        raise SystemExit(f"winner {winner_key!r} absent from zoo_block_tests.json")

    rows = []
    for family in ZOO:
        entry = source[family]
        assignment = entry.get("top1_map") or entry.get("top1_blocks")
        if isinstance(assignment, dict):
            assignment = list(assignment.values())
        if assignment is None:
            # Older compact artifacts retain only the winning block and count.
            assignment = [entry["best_block"]] * len(ZOO[family].labels)
        rows.append((family, [int(block) for block in assignment]))

    counts = Counter(block for _, assignments in rows for block in assignments)
    notable = [block for block, count in counts.most_common(len(st.CAT)) if count >= 2]
    color_of = {block: st.CAT[i] for i, block in enumerate(notable)}
    width = max(len(assignments) for _, assignments in rows)
    fig, ax = plt.subplots(figsize=(12.0, 0.48 * len(rows) + 1.8))
    for row, (family, assignments) in enumerate(rows):
        y = len(rows) - 1 - row
        for col, block in enumerate(assignments):
            face = color_of.get(block, st.BASELINE)
            ax.add_patch(
                plt.Rectangle(
                    (col, y), 0.96, 0.88, facecolor=face, edgecolor=st.SURFACE
                )
            )
            ax.text(
                col + 0.48,
                y + 0.44,
                str(block),
                ha="center",
                va="center",
                fontsize=6.5,
                color="white" if block in color_of else st.INK2,
            )
    ax.set_xlim(0, width)
    ax.set_ylim(0, len(rows))
    ax.set_yticks(np.arange(len(rows)) + 0.44, [name for name, _ in reversed(rows)])
    ax.set_xticks([])
    ax.tick_params(length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(
        "Winner top-1 class assignments across the descriptive zoo\n"
        "repeated colors expose consolidation; every family remains non-selective"
    )
    fig.tight_layout()
    fig.savefig(OUT / "capture.png", dpi=160)
    plt.close(fig)


def fig_allocation() -> None:
    """Compare per-site FVU in the winner and matched primary gauge."""

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for (label, (_, root)), color in zip(ARMS.items(), (st.CAT[5], st.CAT[0])):
        report = json.loads((root / "report.json").read_text())
        ax.plot(
            np.arange(len(SITES)),
            report["eval"]["topk"]["fvu_per_site"],
            "o-",
            color=color,
            label=label,
        )
    ax.set_xticks(np.arange(len(SITES)), [f"L{site}" for site in SITES])
    ax.set_xlabel("site (Gemma 3 4B layer)")
    ax.set_ylabel("eval FVU (top-k mode)")
    ax.set_title("The coordinate gauge changes where the dictionary spends capacity")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "site-allocation.png", dpi=160)
    plt.close(fig)


def main() -> None:
    _configure()
    st.apply()
    OUT.mkdir(parents=True, exist_ok=True)
    fig_capture(_tests())
    fig_allocation()
    print(f"-> {OUT / 'capture.png'}")
    print(f"-> {OUT / 'site-allocation.png'}")


if __name__ == "__main__":
    main()
