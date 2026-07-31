"""Focused tests for the trusted-operator campaign."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from block_crosscoder_experiment.campaign import (
    ArtifactRef,
    Campaign,
    CampaignError,
    InvalidTransition,
    RunState,
)
from block_crosscoder_experiment.studies import (
    build_phase1_blueprint,
    build_phase1_plan,
)


def _campaign(tmp_path: Path) -> Campaign:
    plan = build_phase1_plan((0,), smoke=True)
    blueprint = build_phase1_blueprint((0,), smoke=True)
    campaign = Campaign(tmp_path)
    campaign.register(plan, blueprint_manifest=blueprint.to_manifest())
    return campaign


def _file(root: Path, name: str, body: str = "ok") -> Path:
    path = root / "artifacts" / name.replace(":", "_")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return path


def _advance_to_qualified(
    campaign: Campaign,
    cell_id: str,
    *,
    metric: float = 0.5,
) -> None:
    root = campaign.root
    prep = ArtifactRef.from_path("preparation", _file(root, f"{cell_id}-prep"), root)
    campaign.transition(cell_id, RunState.PREPARED, artifacts=(prep,))
    campaign.transition(cell_id, RunState.RUNNING)

    train = (
        ArtifactRef.from_path("checkpoint", _file(root, f"{cell_id}-checkpoint"), root),
        ArtifactRef.from_path(
            "training_report", _file(root, f"{cell_id}-training"), root
        ),
    )
    campaign.transition(cell_id, RunState.TRAINED, artifacts=train)

    calibration = (
        ArtifactRef.from_path("calibration", _file(root, f"{cell_id}-codec"), root),
        ArtifactRef.from_path(
            "deployment_codec", _file(root, f"{cell_id}-deployment"), root
        ),
        ArtifactRef.from_path(
            "calibration_record", _file(root, f"{cell_id}-calibration"), root
        ),
    )
    campaign.transition(cell_id, RunState.CALIBRATED, artifacts=calibration)

    evaluation = (
        ArtifactRef.from_path(
            "deployment_schedules", _file(root, f"{cell_id}-schedules"), root
        ),
        ArtifactRef.from_path("evaluation", _file(root, f"{cell_id}-evaluation"), root),
    )
    campaign.transition(cell_id, RunState.EVALUATED, artifacts=evaluation)

    qualification_path = _file(root, f"{cell_id}-qualification")
    qualification_path.write_text(
        json.dumps(
            {
                "cell_id": cell_id,
                "qualified": True,
                "scientific_outcome": {"passed": True},
                "promotion_eligible": False,
                "selection_eligible_for_protocol_test": True,
                "validation": {"phase1_identification_margin": metric},
            }
        )
    )
    qualification = ArtifactRef.from_path(
        "qualification", qualification_path, root
    )
    campaign.transition(cell_id, RunState.QUALIFIED, artifacts=(qualification,))


def test_registration_writes_one_current_plan_and_state_per_cell(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)

    assert campaign.plan.plan_id.startswith("study:")
    assert campaign.blueprint_path.is_file()
    assert not (tmp_path / "journal.jsonl").exists()
    assert {record.state for record in campaign.records()} == {RunState.PLANNED}
    for cell in campaign.plan.cells:
        assert campaign.cell_manifest_path(cell.cell_id).is_file()
        assert campaign.state_path(cell.cell_id).is_file()


def test_artifacts_are_ordinary_paths_and_operator_edits_are_allowed(
    tmp_path: Path,
) -> None:
    campaign = _campaign(tmp_path)
    path = _file(tmp_path, "notes.json", '{"choice": 1}')
    ref = ArtifactRef.from_path("notes", path, tmp_path)

    path.write_text('{"choice": 2}')
    assert ref.verify(tmp_path).resolve(tmp_path).read_text() == '{"choice": 2}'


def test_lifecycle_and_retry_resume_from_latest_completed_stage(
    tmp_path: Path,
) -> None:
    campaign = _campaign(tmp_path)
    cell_id = campaign.plan.cells[0].cell_id
    prep = ArtifactRef.from_path("preparation", _file(tmp_path, "prep"), tmp_path)

    campaign.transition(cell_id, RunState.PREPARED, artifacts=(prep,))
    campaign.transition(cell_id, RunState.RUNNING)
    failed = campaign.transition(cell_id, RunState.FAILED, message="interrupted")
    assert failed.resume_state is RunState.PREPARED

    retried = campaign.retry(cell_id)
    assert retried.state is RunState.PREPARED
    with pytest.raises(InvalidTransition):
        campaign.transition(cell_id, RunState.EVALUATED)


def test_stage_dependency_opens_after_source_qualification(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)
    first, second = campaign.plan.stages

    assert campaign.stage_open(first.name)
    assert not campaign.stage_open(second.name)
    _advance_to_qualified(campaign, first.cells[0].cell_id)
    assert campaign.stage_open(second.name)


def test_selection_reads_declared_metric_and_writes_reviewable_json(
    tmp_path: Path,
) -> None:
    campaign = _campaign(tmp_path)
    stage = campaign.plan.stages[-1]
    _advance_to_qualified(campaign, stage.cells[0].cell_id, metric=0.42)

    payload = campaign.select_stage(stage.name)
    selected = payload["selected"][0]
    assert selected["selection_id"].startswith("selection:")
    assert selected["cell_ids"] == [stage.cells[0].cell_id]
    assert selected["metric_values"] == [0.42]
    assert (tmp_path / "selections" / f"{stage.name}.json").is_file()


def test_register_refuses_overwriting_an_existing_campaign(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)
    with pytest.raises(CampaignError, match="already exists"):
        campaign.register(
            campaign.plan,
            blueprint_manifest=build_phase1_blueprint((0,), smoke=True).to_manifest(),
        )
