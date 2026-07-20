"""The committed figure catalog and unified CLI are structural contracts."""

from __future__ import annotations

import importlib
import json

from block_crosscoder_experiment.analysis.artifacts import FIGURES_DIR
from block_crosscoder_experiment.analysis.catalog import (
    CAP_ONLY,
    FAMILIES,
    TUNING,
    ZOO_FAMILIES,
)
from block_crosscoder_experiment.cli import COMMANDS


def test_every_cli_command_resolves_to_a_module():
    for module_name, _ in COMMANDS.values():
        assert importlib.import_module(module_name)


def test_zodiac_is_a_burned_tuning_ring_not_a_figure_family():
    assert "zodiac" not in ZOO_FAMILIES
    assert TUNING["zodiac"].topology == "ring"
    assert len(FAMILIES["zodiac"]) == 12
    assert "zodiac" in CAP_ONLY


def test_committed_figure_catalog_is_complete_and_shared_runtime():
    manifest = json.loads((FIGURES_DIR / "manifest.json").read_text())
    assert tuple(sorted(manifest["families"])) == ZOO_FAMILIES
    assert (FIGURES_DIR / "assets" / "plotly.min.js").stat().st_size > 1_000_000

    for family in ZOO_FAMILIES:
        entry = manifest["families"][family]
        assert entry["files"] == [
            f"{family}/frames.html",
            f"{family}/flow.html",
            f"{family}/stream.html",
            f"{family}/code.html",
        ]
        for relative in entry["files"]:
            page = (FIGURES_DIR / relative).read_text()
            assert 'src="../assets/plotly.min.js"' in page
            assert "pooled FVU" in page
            assert "selection endpoint" in page
        code_page = (FIGURES_DIR / family / "code.html").read_text()
        assert "shared code" in code_page
        assert "One class mean per point" in code_page
        assert "between-class code-mean variance" in code_page
