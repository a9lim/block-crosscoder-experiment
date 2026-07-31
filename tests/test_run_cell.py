"""Focused tests for the cell executor and scientific handoff."""

from __future__ import annotations

import json
from pathlib import Path

from block_crosscoder_experiment.campaign import Campaign, CampaignRunner, RunState
from block_crosscoder_experiment.cli.run_cell import (
    _balanced_schedule_uses_upper,
    _first_nonfinite_json_path,
    _lower_convex_rate_envelope,
    _phase1_identification_outcome,
    validate_cell_config,
)
from block_crosscoder_experiment.studies import (
    Phase,
    build_phase1_blueprint,
    build_phase1_plan,
)


def test_nonfinite_json_diagnostic_reports_the_first_path() -> None:
    assert (
        _first_nonfinite_json_path(
            {"valid": [1, None], "invalid": {"values": (0.0, float("nan"))}}
        )
        == "$.invalid.values[1]"
    )
    assert _first_nonfinite_json_path({"finite": [True, -3.5]}) is None


def test_lower_envelope_drops_dominated_and_concave_points() -> None:
    points = [
        {"name": "zero", "total_bits_per_token": 0.0, "raw_space_fvu": 1.0},
        {"name": "q2", "total_bits_per_token": 2.0, "raw_space_fvu": 0.8},
        {"name": "q4", "total_bits_per_token": 4.0, "raw_space_fvu": 0.7},
        {"name": "dominated", "total_bits_per_token": 6.0, "raw_space_fvu": 0.75},
        {"name": "q8", "total_bits_per_token": 8.0, "raw_space_fvu": 0.4},
    ]
    envelope = _lower_convex_rate_envelope(points)
    assert [point["name"] for point in envelope] == ["zero", "q2", "q8"]


def test_balanced_schedule_uses_the_exact_requested_number_of_upper_packets() -> None:
    decisions = [
        _balanced_schedule_uses_upper(
            index,
            upper_tokens=3,
            horizon_tokens=10,
        )
        for index in range(10)
    ]
    assert sum(decisions) == 3


def test_phase1_inapplicable_identification_is_reported_neutrally() -> None:
    reason = "token_layer_normalization_is_not_a_fixed_linear_factor_map"
    passed, inapplicable = _phase1_identification_outcome(
        Phase.PHASE1,
        {
            "native": {"applicable": False, "ineligible_reason": reason},
            "deployed": {"applicable": False, "ineligible_reason": reason},
        },
        {},
    )
    assert passed
    assert inapplicable == {"phase1_identification": reason}


def test_smoke_cell_runs_prepare_through_qualification(tmp_path: Path) -> None:
    plan = build_phase1_plan((0,), smoke=True)
    blueprint = build_phase1_blueprint((0,), smoke=True)
    campaign = Campaign(tmp_path)
    campaign.register(plan, blueprint_manifest=blueprint.to_manifest())
    cell = plan.cells[0]

    model_cfg, train_cfg = validate_cell_config(cell)
    assert model_cfg.block_dim == 2
    assert train_cfg.total_steps == 4

    summary = CampaignRunner(campaign).run(limit=1)
    assert summary.completed_cells == 1
    assert summary.failed_cells == 0

    record = campaign.record(cell.cell_id)
    assert record.state is RunState.QUALIFIED
    qualification = json.loads(
        record.artifact_map["qualification"].resolve(tmp_path).read_text()
    )
    assert qualification["qualified"] is True
    assert qualification["checks"]["training_complete"] is True
    assert qualification["scientific_outcome"]["passed"] is True
