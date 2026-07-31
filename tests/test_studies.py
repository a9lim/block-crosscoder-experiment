"""Focused tests for declarative plans and scientific policies."""

from __future__ import annotations

from block_crosscoder_experiment.studies import (
    FrozenSelection,
    Phase,
    build_phase1_blueprint,
    build_phase1_plan,
    build_phase2_blueprint,
    build_phase2_plan,
    build_phase3_blueprint,
    estimate_plan,
)


def test_phase1_scientific_plan_has_declared_cell_count_and_seeds() -> None:
    blueprint = build_phase1_blueprint()
    plan = build_phase1_plan()
    assert plan.phase is Phase.PHASE1
    assert len(plan.cells) == 6
    assert blueprint.projected_cells == 15
    assert {cell.seed for cell in plan.cells} == {0, 1, 2}


def test_smoke_plan_is_a_small_complete_prefix() -> None:
    blueprint = build_phase1_blueprint((0,), smoke=True)
    plan = build_phase1_plan((0,), smoke=True)
    assert blueprint.name.endswith("_smoke")
    assert len(plan.cells) == 2
    assert [stage.name for stage in plan.stages] == [
        "single_site_learnability",
        "multisite_learnability",
    ]


def test_identifiers_are_readable_names_not_digests() -> None:
    blueprint = build_phase1_blueprint((0,), smoke=True)
    plan = build_phase1_plan((0,), smoke=True)
    cell = plan.cells[0]

    assert plan.plan_id == "study:phase1_synthetic_prefix_2_smoke"
    assert blueprint.blueprint_id.startswith("phase1-blueprint:")
    assert cell.cell_id.startswith("cell:phase1.")
    assert cell.recipe_id.startswith("recipe:")
    assert cell.candidate_id.startswith("candidate:")


def test_every_decision_keeps_scientific_lineage_and_rationale() -> None:
    plan = build_phase1_plan((0,), smoke=True)
    allowed = {"exact", "adapted", "engineering", "novel"}
    for cell in plan.cells:
        for decision in cell.decisions:
            assert decision.lineage in allowed
            assert decision.name
            if decision.lineage in {"adapted", "novel"}:
                assert decision.rationale
                assert decision.ablation


def test_phase1_selection_uses_identification_margin() -> None:
    policy = build_phase1_plan((0,), smoke=True).stages[-1].selection_policy
    assert policy is not None
    assert policy.metric_path == "validation.phase1_identification_margin"
    assert policy.direction == "max"
    assert policy.require_all_seeds is True
    assert policy.require_scientific_outcome_pass is True


def test_frozen_selection_records_reviewable_files() -> None:
    stage = build_phase1_plan((0,), smoke=True).stages[-1]
    policy = stage.selection_policy
    assert policy is not None
    selection = FrozenSelection.from_cells(
        policy,
        stage.cells,
        (0.37,),
        ("cells/example/outputs/qualification.json",),
    )
    payload = selection.to_dict()

    assert payload["selection_id"].startswith("selection:")
    assert payload["qualification_files"] == [
        "cells/example/outputs/qualification.json"
    ]
    assert not any("sha" in key for key in payload)


def test_phase2_preview_keeps_real_model_tuning_separate_from_phase1() -> None:
    blueprint = build_phase2_blueprint((0,), smoke=True)
    plan = build_phase2_plan((0,), smoke=True)
    assert plan.phase is Phase.PHASE2
    assert blueprint.name.endswith("_smoke")
    assert all(cell.decision_map["runtime.smoke"] is True for cell in plan.cells)
    assert all(cell.decision_map["data.kind"] != "synthetic" for cell in plan.cells)


def test_phase3_panel_has_one_finalist_and_six_comparators() -> None:
    blueprint = build_phase3_blueprint((0,), smoke=True)
    assert len(blueprint.panel_slots) == 7
    assert sum(slot.role == "selected_finalist" for slot in blueprint.panel_slots) == 1
    assert {
        slot.name
        for slot in blueprint.panel_slots
        if slot.role != "selected_finalist"
    } == {
        "bsf_grassmannian",
        "bsf_group_lasso",
        "sasa",
        "anthropic_dense_l1",
        "decoder_weighted_batchtopk",
        "scalar_relu_batchtopk",
    }


def test_resource_estimate_is_positive_and_tracks_cells() -> None:
    plan = build_phase1_plan((0,), smoke=True)
    estimate = estimate_plan(plan)
    assert estimate.training_tokens > 0
    assert estimate.parameters > 0
    assert estimate.compute_flops > 0
    assert estimate.storage_bytes > 0
