"""End-to-end and fail-closed tests for the generic cell executor."""

from __future__ import annotations

import copy
import json
import hashlib
import inspect
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from block_crosscoder_experiment.campaign import (
    Campaign,
    CampaignRunner,
    QUALIFICATION_SCHEMA,
    RunState,
)
from block_crosscoder_experiment.cli.run_cell import (
    CellExecutionError,
    _Context,
    _RawEndpointErrorCache,
    _VERIFIED_STORE_BINDINGS,
    _apply_encoder_scale_calibration,
    _encoder_scale_fit_batches,
    _evaluate_cached_time_sharing,
    _balanced_schedule_uses_upper,
    _expected_capture_allocation,
    _expected_real_source_contract,
    _fixed_rate_raw_score,
    _lower_convex_rate_envelope,
    _load_deployable_codec,
    _load_deployment_schedule_bundle,
    _load_capture_contract,
    _matching_pathologies,
    _model_config,
    _normalization_record,
    _production_precision_preflight,
    _resolve_real_store,
    _selection_validation_metrics,
    _synthetic_batches,
    _synthetic_dataset,
    _synthetic_source_contract,
    _support_confusion,
    _support_matched_subspace_overlap,
    _tensor_payload_digest,
    _time_sharing_plan_key,
    _transform_on_cuda,
    _training_batches,
    _train_config,
    _validate_final_checkpoint,
    _verify_real_source_contract,
    _verify_store_reader_once,
    _write_deployment_schedule_bundle,
    validate_cell_config,
)
from block_crosscoder_experiment.cli.data import fit_transform_artifacts
from block_crosscoder_experiment.codec import Codec
from block_crosscoder_experiment.model import BlockCrosscoder
from block_crosscoder_experiment.runtime_limits import (
    DECODED_ENERGY_EXACT_IMPLEMENTATION,
    DECODED_ENERGY_STIEFEL_CODE_NORM_IMPLEMENTATION,
    ISOLATED_LOSS_EXACT_IMPLEMENTATION,
    ISOLATED_LOSS_MAPPED_IMPLEMENTATION,
)
from block_crosscoder_experiment.store import ShardWriter, StoreReader
from block_crosscoder_experiment.studies import (
    BSC_FACTOR_CONTESTS,
    CellSpec,
    FrozenPanelDecision,
    FrozenPanelEntry,
    FrozenSelection,
    Phase,
    Phase1Blueprint,
    PHASE2_INELIGIBLE_SELECTION_SCORE,
    PHASE2_SELECTION_METRIC_KEY,
    PHASE2_SELECTION_METRIC_PATH,
    RELEASE_DIAGNOSTIC_RECIPES,
    StageSpec,
    StudyPlan,
    build_phase1_plan,
    build_phase2_plan,
    build_phase2_blueprint,
    build_phase3_blueprint,
    build_phase3_plan,
    canonical_json,
    engineering,
    materialize_child_plan,
    merge_decisions,
)

import pytest
import torch

import block_crosscoder_experiment.cli.run_cell as run_cell_module


REPO = Path(__file__).resolve().parents[1]


def test_evaluate_uses_one_common_selector_and_shared_stream() -> None:
    source = inspect.getsource(run_cell_module._evaluate)
    assert source.count("_prefetched_evaluation_batches(") == 1
    assert "evaluate_selector_and_shared_code_modes(" in source
    assert "_evaluate_native_selector(" not in source


def test_subspace_overlap_is_bound_to_the_support_selected_group() -> None:
    # Each factor's geometrically best group is deliberately the *other*
    # group.  Recovery must report the overlap at the support assignment,
    # rather than splicing together evidence from unrelated learned blocks.
    overlaps = torch.tensor([[0.1, 0.95], [0.9, 0.2]])
    support_assignment = torch.tensor([0, 1], dtype=torch.long)
    matched = _support_matched_subspace_overlap(overlaps, support_assignment)
    assert matched.tolist() == pytest.approx([0.1, 0.2])
    assert matched.tolist() != pytest.approx(overlaps.max(dim=1).values.tolist())


def _cell(
    *,
    phase: Phase = Phase.PHASE1,
    recipe_index: int = 0,
    seed: int = 0,
) -> CellSpec:
    if phase is Phase.PHASE1:
        recipe_name = BSC_FACTOR_CONTESTS[recipe_index].name
        base = next(
            cell
            for cell in build_phase1_plan(seeds=(seed,), smoke=True).cells
            if cell.recipe_name == recipe_name
        )
    else:
        base = build_phase2_plan(seeds=(seed,), smoke=True).cells[0]
    return replace(
        base,
        name=f"{phase.value}.test.executor{recipe_index}.s{seed}",
        stage="test",
    )


def _stiefel_decoded_energy_cell(*, seed: int = 0) -> CellSpec:
    base = _cell(recipe_index=0, seed=seed)
    overrides = {
        "model.selection_score": "decoded_energy",
        "implementation.decoded_energy_implementation": (
            DECODED_ENERGY_STIEFEL_CODE_NORM_IMPLEMENTATION
        ),
        "optimizer.retract_every_steps": 1,
    }
    return replace(
        base,
        name=f"phase1.test.stiefel_decoded_energy.s{seed}",
        decisions=tuple(
            replace(decision, value=overrides[decision.name])
            if decision.name in overrides
            else decision
            for decision in base.decisions
        ),
    )


def _mapped_isolated_loss_cell(*, seed: int = 0) -> CellSpec:
    base = _cell(recipe_index=0, seed=seed)
    overrides = {
        "model.selection_score": "isolated_loss_decrease",
        "model.decoder": "free_scale_controlled",
        "model.decoder_bias": False,
        "objective.reconstruction": "squared_l2",
        "implementation.isolated_loss_decrease_implementation": (
            ISOLATED_LOSS_MAPPED_IMPLEMENTATION
        ),
    }
    return replace(
        base,
        name=f"phase1.test.mapped_isolated_loss.s{seed}",
        decisions=tuple(
            replace(decision, value=overrides[decision.name])
            if decision.name in overrides
            else decision
            for decision in base.decisions
        ),
    )


def _campaign(tmp_path: Path, cell: CellSpec) -> Campaign:
    source_stage = build_phase1_plan(
        seeds=(cell.seed,), smoke=bool(cell.decision_map["runtime.smoke"])
    ).stages[-1]
    assert source_stage.selection_policy is not None
    selection_policy = replace(
        source_stage.selection_policy,
        eligible_recipe_names=(cell.recipe_name,),
    )
    plan = StudyPlan(
        f"test_executor_{cell.phase.value}_{cell.seed}_{cell.recipe_name}",
        cell.phase,
        (StageSpec("test", (cell,), selection_policy=selection_policy),),
    )
    blueprint = Phase1Blueprint(
        name="run_cell_test_blueprint",
        seeds=(cell.seed,),
        initial_stages=plan.stages,
        rounds=(),
    )
    campaign = Campaign(tmp_path / "campaign")
    # Executor integration tests intentionally use one tiny smoke cell. Patch
    # registration's canonical builders only for this non-scientific fixture;
    # _Context still authenticates the persisted plan, blueprint, history,
    # journal, and exact cell path in the subprocess.
    with (
        patch(
            "block_crosscoder_experiment.campaign.build_phase1_blueprint",
            return_value=blueprint,
        ),
        patch(
            "block_crosscoder_experiment.campaign.build_phase1_plan",
            return_value=plan,
        ),
    ):
        campaign.register(plan, blueprint_manifest=blueprint.to_manifest())
    return campaign


def _runner(campaign: Campaign, **extra_env: str) -> CampaignRunner:
    pythonpath = str(REPO)
    if os.environ.get("PYTHONPATH"):
        pythonpath += os.pathsep + os.environ["PYTHONPATH"]
    return CampaignRunner(
        campaign,
        python=sys.executable,
        env={"PYTHONPATH": pythonpath, **extra_env},
    )


@pytest.mark.parametrize("cell_index", (0, 7))
def test_executor_rejects_unbound_phase2_preview_cell(
    tmp_path, monkeypatch, cell_index
):
    blueprint = build_phase2_blueprint(seeds=(0,), smoke=True)
    plan = build_phase2_plan(seeds=(0,), smoke=True)
    cell = plan.cells[cell_index]
    campaign = Campaign(tmp_path)
    campaign.plan_path.write_text(
        json.dumps(plan.to_manifest(), indent=2, sort_keys=True) + "\n"
    )
    campaign.blueprint_path.write_text(
        json.dumps(blueprint.to_manifest(), indent=2, sort_keys=True) + "\n"
    )
    campaign.plans_dir.mkdir(parents=True)
    (campaign.plans_dir / "preview.json").write_text(
        json.dumps(plan.to_manifest(), indent=2, sort_keys=True) + "\n"
    )
    cell_path = campaign.cell_manifest_path(cell.cell_id)
    cell_dir = cell_path.parent
    cell_dir.mkdir(parents=True)
    cell_path.write_text(
        json.dumps(cell.to_manifest(), indent=2, sort_keys=True) + "\n"
    )
    monkeypatch.setenv("BSC_CAMPAIGN_ROOT", str(tmp_path))

    with pytest.raises(
        CellExecutionError,
        match="planning estimates only|authenticated Phase-1 decision",
    ):
        _Context(cell_path, cell_dir / "artifacts.json", "prepare")


def _phase3_cell() -> CellSpec:
    blueprint = build_phase3_blueprint(seeds=(0,), smoke=True)
    phase2_blueprint = build_phase2_blueprint(seeds=(0,), smoke=True)
    phase2 = build_phase2_plan(seeds=(0,), smoke=True)
    while phase2.stages[-1].selection_policy is not None:
        stage = phase2.stages[-1]
        groups: dict[str, list[CellSpec]] = {}
        for cell in stage.cells:
            groups.setdefault(cell.candidate_id, []).append(cell)
        eligible = [
            group
            for group in groups.values()
            if not stage.selection_policy.eligible_recipe_names
            or group[0].recipe_name in stage.selection_policy.eligible_recipe_names
        ]
        eligible = [
            group
            for group in eligible
            if all(
                cell.decision_map["qualification.promotable"] is True for cell in group
            )
        ]
        selected = tuple(
            sorted(
                sorted(eligible, key=lambda group: group[0].candidate_id)[0],
                key=lambda item: item.seed,
            )
        )
        frozen = FrozenSelection.from_cells(
            stage.selection_policy,
            selected,
            tuple(0.5 for _ in selected),
            tuple(
                "sha256:" + hashlib.sha256(cell.cell_id.encode()).hexdigest()
                for cell in selected
            ),
            "sha256:" + hashlib.sha256(stage.name.encode()).hexdigest(),
        )
        phase2 = materialize_child_plan(phase2, phase2_blueprint, frozen)
    anchors: dict[str, tuple[CellSpec, ...]] = {}
    for cell in phase2.stages[0].cells:
        anchors.setdefault(cell.recipe_name, tuple())
        anchors[cell.recipe_name] += (cell,)
    finalists: dict[str, tuple[CellSpec, ...]] = {}
    for cell in phase2.stages[-1].cells:
        finalists.setdefault(cell.candidate_id, tuple())
        finalists[cell.candidate_id] += (cell,)
    finalist = next(
        cells
        for cells in finalists.values()
        if cells[0].decision_map["data.normalization"] == "scalar_rms"
    )
    entries = []
    for slot in blueprint.panel_slots:
        if slot.role == "selected_finalist":
            entries.append(
                FrozenPanelEntry.from_cells(
                    panel_slot=slot.name,
                    role=slot.role,
                    source_cells=finalist,
                    selection_ids=finalist[0].decision_map[
                        "selection.upstream_selection_ids"
                    ],
                    qualification_sha256s=(
                        "sha256:" + hashlib.sha256(b"final-qualification").hexdigest(),
                    ),
                    confirmation_sha256s=(
                        "sha256:" + hashlib.sha256(b"final-confirmation").hexdigest(),
                    ),
                )
            )
        else:
            root_cells = anchors[str(slot.comparator_recipe_name)]
            derived_recipe_id = (
                "derived-recipe:"
                + hashlib.sha256(f"family:{slot.name}".encode()).hexdigest()
            )
            source_cells = tuple(
                replace(
                    cell,
                    stage=f"family_{slot.name}_top2_revisit_16m",
                    recipe_name=f"derived_family_{slot.name}_revisit_winner",
                    recipe_id=derived_recipe_id,
                    decisions=merge_decisions(
                        cell.decisions,
                        (
                            engineering(
                                "selection.comparator_family_name",
                                slot.comparator_family_name,
                                rationale="test fixture binds the calibrated family",
                            ),
                            engineering(
                                "selection.comparator_family_blueprint_id",
                                slot.comparator_family_id,
                                rationale="test fixture binds the family blueprint",
                            ),
                            engineering(
                                "selection.family_root_recipe_id",
                                slot.comparator_recipe_id,
                                rationale="test fixture binds the family root lineage",
                            ),
                        ),
                    ),
                )
                for cell in root_cells
            )
            entries.append(
                FrozenPanelEntry.from_cells(
                    panel_slot=slot.name,
                    role=slot.role,
                    source_cells=source_cells,
                    selection_ids=(
                        "selection:"
                        + hashlib.sha256(
                            f"family-selection:{slot.name}".encode()
                        ).hexdigest(),
                    ),
                    qualification_sha256s=(
                        "sha256:" + hashlib.sha256(slot.name.encode()).hexdigest(),
                    ),
                )
            )
    panel = FrozenPanelDecision(
        source_phase2_plan_id=phase2.plan_id,
        source_phase2_blueprint_id=phase2_blueprint.blueprint_id,
        phase2_campaign_manifest_sha256=(
            "sha256:" + hashlib.sha256(b"phase2-campaign").hexdigest()
        ),
        selection_universe_sha256=(
            "sha256:" + hashlib.sha256(b"phase2-universe").hexdigest()
        ),
        entries=tuple(entries),
    )
    return build_phase3_plan(seeds=(0,), smoke=True, panel_decision=panel).cells[0]


@pytest.mark.parametrize(
    ("label", "cell_factory"),
    (
        ("Phase-1", lambda: build_phase1_plan(seeds=(0,), smoke=True).cells[0]),
        ("Phase-3", _phase3_cell),
    ),
)
def test_executor_rejects_orphan_phase1_and_phase3_cells(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    cell_factory,
) -> None:
    cell = cell_factory()
    cell_dir = tmp_path / "cells" / label.lower().replace("-", "")
    cell_dir.mkdir(parents=True)
    cell_path = cell_dir / "cell.json"
    cell_path.write_text(
        json.dumps(cell.to_manifest(), indent=2, sort_keys=True) + "\n"
    )
    monkeypatch.setenv("BSC_CAMPAIGN_ROOT", str(tmp_path))

    with pytest.raises(
        CellExecutionError,
        match="exact active campaign binding",
    ):
        _Context(cell_path, cell_dir / "artifacts.json", "prepare")


def test_tiny_phase1_cell_runs_all_five_stages_and_binds_inputs(tmp_path: Path) -> None:
    cell = _cell()
    campaign = _campaign(tmp_path, cell)
    summary = _runner(campaign).run(limit=1)
    assert summary.to_dict() == {
        "selected_cells": 1,
        "completed_cells": 1,
        "failed_cells": 0,
        "skipped_cells": 0,
    }
    record = campaign.record(cell.cell_id)
    assert record.state is RunState.QUALIFIED
    refs = record.artifact_map
    for kind in (
        "preparation",
        "checkpoint",
        "training_report",
        "calibration",
        "deployment_codec",
        "deployment_schedules",
        "calibration_record",
        "evaluation",
        "qualification",
    ):
        refs[kind].verify(campaign.root)

    qualification = json.loads(refs["qualification"].resolve(campaign.root).read_text())
    assert qualification["schema"] == QUALIFICATION_SCHEMA
    assert qualification["qualified"] is True
    assert all(qualification["checks"].values())
    assert set(qualification["checks"]) == {
        "deployment_schedule_integrity",
        "finite",
        "method_endpoints",
        "precision_preflight_integrity",
        "provenance",
        "regularizer_calibration_integrity",
        "resource_compliance",
        "selection_score_diagnostics_integrity",
        "scientific_endpoint_complete",
        "split_integrity",
    }
    assert isinstance(qualification["scientific_outcome"]["passed"], bool)
    assert qualification["promotion_eligible"] is False
    assert "runtime_smoke" in qualification["promotion_ineligible_reasons"]
    assert qualification["selection_eligible_for_protocol_test"] is True
    assert qualification["selection_eligibility_mode"] == "smoke_protocol_only"
    assert qualification["inputs"] == {
        kind: refs[kind].sha256
        for kind in (
            "checkpoint",
            "calibration",
            "deployment_codec",
            "deployment_schedules",
            "evaluation",
        )
    }
    evaluation = json.loads(refs["evaluation"].resolve(campaign.root).read_text())
    assert evaluation["synthetic_recovery"]["native"]["n_truth_factors"] > 0
    assert evaluation["synthetic_recovery"]["deployed"]["n_truth_factors"] > 0
    for endpoint in ("native", "deployed"):
        assert (
            evaluation["synthetic_recovery"][endpoint]["n_factor_calibration_examples"]
            == cell.decision_map["data.synthetic_factor_calibration_examples"]
        )
        assert (
            evaluation["synthetic_recovery"][endpoint]["n_examples"]
            == (cell.decision_map["data.synthetic_development_examples"])
        )
    assert evaluation["native_selector"]["mode"] == "topk"
    assert evaluation["deployed_selector"]["mode"] == "threshold"
    assert evaluation["codec_roundtrip"]["source_free_decode"] is True
    assert evaluation["rate_distortion"]["points"]["8"]["rate_bits_per_token"] >= 0


def test_factorized_masked_decoded_energy_cell_runs_through_saved_codec(
    tmp_path: Path,
) -> None:
    base = _cell(recipe_index=2, seed=41)
    overrides = {
        "model.decoder": "free_scale_controlled",
        "model.site_rank": 2,
        "model.selection_score": "decoded_energy",
        "objective.encoder_site_mask_probability": 0.10,
    }
    cell = replace(
        base,
        name="phase1.test.factorized_masked_decoded_energy.s41",
        decisions=tuple(
            replace(decision, value=overrides[decision.name])
            if decision.name in overrides
            else decision
            for decision in base.decisions
        ),
    )
    model_cfg, train_cfg = validate_cell_config(cell)
    assert model_cfg.site_rank == 2
    assert model_cfg.selection_score == "decoded_energy"
    assert train_cfg.encoder_site_mask_probability == 0.10

    campaign = _campaign(tmp_path, cell)
    summary = _runner(campaign).run(limit=1)
    assert summary.failed_cells == 0
    record = campaign.record(cell.cell_id)
    assert record.state is RunState.QUALIFIED
    evaluation = json.loads(
        record.artifact_map["evaluation"].resolve(campaign.root).read_text()
    )
    for endpoint in ("native", "deployed"):
        dependence = evaluation["shared_code"][endpoint]["functional_dependence"]
        assert {"pre_selection", "post_selection"}.issubset(dependence)
        assert dependence["pre_selection"]["delta_by_site_block"]


def test_deployable_codec_is_the_complete_validated_consumer_artifact(
    tmp_path: Path,
) -> None:
    cell = _stiefel_decoded_energy_cell(seed=37)
    campaign = _campaign(tmp_path, cell)
    assert _runner(campaign).run(limit=1).failed_cells == 0
    refs = campaign.record(cell.cell_id).artifact_map
    deployment_path = refs["deployment_codec"].resolve(campaign.root)
    payload = torch.load(deployment_path, map_location="cpu", weights_only=True)
    preparation_hash = refs["preparation"].sha256

    loaded, model, codec, summary = _load_deployable_codec(
        deployment_path,
        cell_id=cell.cell_id,
        checkpoint_hash=refs["checkpoint"].sha256,
        calibration_hash=refs["calibration"].sha256,
        preparation_hash=preparation_hash,
        device=torch.device("cpu"),
    )
    assert loaded["schema"] == "bsc-deployable-codec-v2"
    assert model.cfg.n_blocks == codec.included.numel()
    assert (
        model.cfg.decoded_energy_implementation
        == DECODED_ENERGY_STIEFEL_CODE_NORM_IMPLEMENTATION
    )
    assert summary["accepted_tokens"] > 0

    checkpoint_payload = torch.load(
        refs["checkpoint"].resolve(campaign.root),
        map_location="cpu",
        weights_only=True,
    )
    mismatched_checkpoint = copy.deepcopy(checkpoint_payload)
    mismatched_checkpoint["model_cfg"]["decoded_energy_implementation"] = (
        DECODED_ENERGY_EXACT_IMPLEMENTATION
    )
    checkpoint_path = tmp_path / "mismatched-checkpoint-config.pt"
    torch.save(mismatched_checkpoint, checkpoint_path)
    with pytest.raises(CellExecutionError, match="top-level configuration"):
        _validate_final_checkpoint(
            checkpoint_path,
            checkpoint_payload["run_binding"],
        )

    nested_identity_mismatch = copy.deepcopy(payload)
    nested_codec = Codec.from_payload(
        nested_identity_mismatch["codec_payload"],
        source="test nested codec",
    )
    nested_codec.meta["model_cfg"]["decoded_energy_implementation"] = (
        DECODED_ENERGY_EXACT_IMPLEMENTATION
    )
    nested_identity_mismatch["codec_payload"] = nested_codec.to_payload()
    unsigned = {
        key: value
        for key, value in nested_identity_mismatch.items()
        if key != "artifact_sha256"
    }
    nested_identity_mismatch["artifact_sha256"] = _tensor_payload_digest(unsigned)
    nested_identity_path = tmp_path / "nested-model-identity-mismatch.pt"
    torch.save(nested_identity_mismatch, nested_identity_path)
    with pytest.raises(CellExecutionError, match="model config|model-config"):
        _load_deployable_codec(
            nested_identity_path,
            cell_id=cell.cell_id,
            checkpoint_hash=refs["checkpoint"].sha256,
            calibration_hash=refs["calibration"].sha256,
            preparation_hash=preparation_hash,
            device=torch.device("cpu"),
        )

    finite_off_manifold = copy.deepcopy(payload)
    finite_off_manifold["model_state"]["D"].mul_(1.01)
    assert torch.isfinite(finite_off_manifold["model_state"]["D"]).all()
    unsigned = {
        key: value
        for key, value in finite_off_manifold.items()
        if key != "artifact_sha256"
    }
    finite_off_manifold["artifact_sha256"] = _tensor_payload_digest(unsigned)
    finite_gram_path = tmp_path / "finite-off-manifold-model.pt"
    torch.save(finite_off_manifold, finite_gram_path)
    with pytest.raises(CellExecutionError, match="decoded-energy invariant"):
        _load_deployable_codec(
            finite_gram_path,
            cell_id=cell.cell_id,
            checkpoint_hash=refs["checkpoint"].sha256,
            calibration_hash=refs["calibration"].sha256,
            preparation_hash=preparation_hash,
            device=torch.device("cpu"),
        )

    corrupt_tensor = copy.deepcopy(payload)
    state_name = next(iter(corrupt_tensor["model_state"]))
    corrupt_tensor["model_state"][state_name].view(-1)[0] += 1
    tensor_path = tmp_path / "corrupt-model-tensor.pt"
    torch.save(corrupt_tensor, tensor_path)
    with pytest.raises(CellExecutionError, match="internal content hash"):
        _load_deployable_codec(
            tensor_path,
            cell_id=cell.cell_id,
            checkpoint_hash=refs["checkpoint"].sha256,
            calibration_hash=refs["calibration"].sha256,
            preparation_hash=preparation_hash,
            device=torch.device("cpu"),
        )

    missing_state = copy.deepcopy(payload)
    missing_state["model_state"].pop(next(iter(missing_state["model_state"])))
    unsigned = {
        key: value for key, value in missing_state.items() if key != "artifact_sha256"
    }
    missing_state["artifact_sha256"] = _tensor_payload_digest(unsigned)
    state_path = tmp_path / "missing-model-state.pt"
    torch.save(missing_state, state_path)
    with pytest.raises(CellExecutionError, match="reconstruct consumer"):
        _load_deployable_codec(
            state_path,
            cell_id=cell.cell_id,
            checkpoint_hash=refs["checkpoint"].sha256,
            calibration_hash=refs["calibration"].sha256,
            preparation_hash=preparation_hash,
            device=torch.device("cpu"),
        )

    corrupt_codec = copy.deepcopy(payload)
    corrupt_codec["codec_payload"]["lo"].view(-1)[0] += 1
    unsigned = {
        key: value for key, value in corrupt_codec.items() if key != "artifact_sha256"
    }
    corrupt_codec["artifact_sha256"] = _tensor_payload_digest(unsigned)
    codec_path = tmp_path / "corrupt-nested-codec.pt"
    torch.save(corrupt_codec, codec_path)
    with pytest.raises(CellExecutionError, match="codec artifact hash mismatch"):
        _load_deployable_codec(
            codec_path,
            cell_id=cell.cell_id,
            checkpoint_hash=refs["checkpoint"].sha256,
            calibration_hash=refs["calibration"].sha256,
            preparation_hash=preparation_hash,
            device=torch.device("cpu"),
        )

    missing_normalization = copy.deepcopy(payload)
    missing_normalization.pop("normalization")
    unsigned = {
        key: value
        for key, value in missing_normalization.items()
        if key != "artifact_sha256"
    }
    missing_normalization["artifact_sha256"] = _tensor_payload_digest(unsigned)
    normalization_path = tmp_path / "missing-normalization.pt"
    torch.save(missing_normalization, normalization_path)
    with pytest.raises(CellExecutionError, match="field set mismatch"):
        _load_deployable_codec(
            normalization_path,
            cell_id=cell.cell_id,
            checkpoint_hash=refs["checkpoint"].sha256,
            calibration_hash=refs["calibration"].sha256,
            preparation_hash=preparation_hash,
            device=torch.device("cpu"),
        )

    malformed_normalization = copy.deepcopy(payload)
    malformed_normalization["normalization"]["record"]["scale"] = []
    unsigned = {
        key: value
        for key, value in malformed_normalization.items()
        if key != "artifact_sha256"
    }
    malformed_normalization["artifact_sha256"] = _tensor_payload_digest(unsigned)
    malformed_path = tmp_path / "malformed-normalization.pt"
    torch.save(malformed_normalization, malformed_path)
    with pytest.raises(CellExecutionError, match="normalization scale shape"):
        _load_deployable_codec(
            malformed_path,
            cell_id=cell.cell_id,
            checkpoint_hash=refs["checkpoint"].sha256,
            calibration_hash=refs["calibration"].sha256,
            preparation_hash=preparation_hash,
            device=torch.device("cpu"),
        )


def test_mapped_isolated_loss_identity_round_trips_and_refuses_forgery(
    tmp_path: Path,
) -> None:
    cell = _mapped_isolated_loss_cell(seed=41)
    campaign = _campaign(tmp_path, cell)
    assert _runner(campaign).run(limit=1).failed_cells == 0
    refs = campaign.record(cell.cell_id).artifact_map
    deployment_path = refs["deployment_codec"].resolve(campaign.root)
    payload, model, codec, _summary = _load_deployable_codec(
        deployment_path,
        cell_id=cell.cell_id,
        checkpoint_hash=refs["checkpoint"].sha256,
        calibration_hash=refs["calibration"].sha256,
        preparation_hash=refs["preparation"].sha256,
        device=torch.device("cpu"),
    )
    assert model.cfg.isolated_loss_decrease_implementation == (
        ISOLATED_LOSS_MAPPED_IMPLEMENTATION
    )
    assert payload["model_cfg"]["isolated_loss_decrease_implementation"] == (
        ISOLATED_LOSS_MAPPED_IMPLEMENTATION
    )
    assert codec.meta["model_cfg"][
        "isolated_loss_decrease_implementation"
    ] == ISOLATED_LOSS_MAPPED_IMPLEMENTATION

    forged = copy.deepcopy(payload)
    forged["model_cfg"]["isolated_loss_decrease_implementation"] = (
        ISOLATED_LOSS_EXACT_IMPLEMENTATION
    )
    unsigned = {key: value for key, value in forged.items() if key != "artifact_sha256"}
    forged["artifact_sha256"] = _tensor_payload_digest(unsigned)
    forged_path = tmp_path / "forged-mapped-identity.pt"
    torch.save(forged, forged_path)
    with pytest.raises(CellExecutionError, match="model config|model-config"):
        _load_deployable_codec(
            forged_path,
            cell_id=cell.cell_id,
            checkpoint_hash=refs["checkpoint"].sha256,
            calibration_hash=refs["calibration"].sha256,
            preparation_hash=refs["preparation"].sha256,
            device=torch.device("cpu"),
        )

    checkpoint_payload = torch.load(
        refs["checkpoint"].resolve(campaign.root),
        map_location="cpu",
        weights_only=True,
    )
    checkpoint_payload["model_cfg"].pop(
        "isolated_loss_decrease_implementation"
    )
    checkpoint_payload["run_binding"]["model_cfg"].pop(
        "isolated_loss_decrease_implementation"
    )
    missing_path = tmp_path / "missing-mapped-identity.pt"
    torch.save(checkpoint_payload, missing_path)
    with pytest.raises(
        CellExecutionError,
        match="lacks isolated_loss_decrease_implementation",
    ):
        _validate_final_checkpoint(
            missing_path,
            checkpoint_payload["run_binding"],
        )


@pytest.mark.parametrize(
    ("decision_name", "decision_value", "message"),
    (
        (
            "implementation.decoded_energy_implementation",
            "ambient_cuda_default",
            "unknown decoded-energy implementation identity",
        ),
        (
            "optimizer.retract_every_steps",
            20,
            "violates its carrier predicate",
        ),
    ),
)
def test_runner_refuses_unknown_or_ineligible_decoded_energy_implementation(
    decision_name: str,
    decision_value,
    message: str,
) -> None:
    cell = _stiefel_decoded_energy_cell()
    changed = replace(
        cell,
        decisions=tuple(
            replace(decision, value=decision_value)
            if decision.name == decision_name
            else decision
            for decision in cell.decisions
        ),
    )
    with pytest.raises(CellExecutionError, match=message):
        _model_config(changed)


def test_runner_resolves_and_refuses_mapped_isolated_loss_implementation() -> None:
    cell = _mapped_isolated_loss_cell()
    cfg = _model_config(cell)
    assert cfg.isolated_loss_decrease_implementation == (
        ISOLATED_LOSS_MAPPED_IMPLEMENTATION
    )

    for decision_name, decision_value, message in (
        (
            "implementation.isolated_loss_decrease_implementation",
            "ambient_cuda_default",
            "unknown isolated-loss implementation identity",
        ),
        (
            "model.decoder",
            "concatenated_stiefel",
            "violates its carrier predicate",
        ),
    ):
        changed = replace(
            cell,
            decisions=tuple(
                replace(decision, value=decision_value)
                if decision.name == decision_name
                else decision
                for decision in cell.decisions
            ),
        )
        with pytest.raises(CellExecutionError, match=message):
            _model_config(changed)

    mean_squared = replace(
        cell,
        decisions=tuple(
            replace(decision, value="mean_squared")
            if decision.name == "objective.reconstruction"
            else decision
            for decision in cell.decisions
        ),
    )
    assert _model_config(mean_squared).reconstruction_loss == "mean_squared"


def test_decoded_energy_preflight_precedes_every_score_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cell = _stiefel_decoded_energy_cell(seed=39)
    campaign = _campaign(tmp_path, cell)
    runner = _runner(campaign)
    assert runner.run(stop_after="prepare").completed_cells == 1
    campaign.transition(cell.cell_id, RunState.RUNNING)
    monkeypatch.setenv("BSC_CAMPAIGN_ROOT", str(campaign.root))
    ctx = _Context(
        campaign.cell_manifest_path(cell.cell_id),
        campaign.cell_dir(cell.cell_id) / "preflight-order-artifacts.json",
        "train",
    )
    events: list[str] = []
    original_validate = BlockCrosscoder.validate_decoded_energy_implementation

    def track_validation(model):
        record = original_validate(model)
        if record["applicable"]:
            events.append("validated")
        return record

    class FirstScoreSeen(RuntimeError):
        pass

    def stop_at_first_score(model, *args, **kwargs):
        events.append("score")
        raise FirstScoreSeen

    with monkeypatch.context() as order:
        order.setattr(
            BlockCrosscoder,
            "validate_decoded_energy_implementation",
            track_validation,
        )
        order.setattr(BlockCrosscoder, "scores", stop_at_first_score)
        with pytest.raises(FirstScoreSeen):
            run_cell_module._train(ctx, ctx.prerequisites(), resume=False)
    assert events
    assert events[0] == "validated"
    assert events[-1] == "score"
    assert "validated" in events[: events.index("score")]


def test_failed_scientific_outcome_still_yields_admissible_terminal_report(
    tmp_path: Path,
) -> None:
    base = _cell(seed=31)
    thresholds = tuple(
        (name, 1.01 if name == "aggregate.recovered_factor_fraction_min" else value)
        for name, value in base.decision_map[
            "qualification.phase1_identification_thresholds"
        ]
    )
    cell = replace(
        base,
        name="phase1.test.expected_negative.s31",
        decisions=tuple(
            replace(decision, value=thresholds)
            if decision.name == "qualification.phase1_identification_thresholds"
            else decision
            for decision in base.decisions
        ),
    )
    campaign = _campaign(tmp_path, cell)
    summary = _runner(campaign).run(limit=1)
    assert summary.completed_cells == 1
    assert summary.failed_cells == 0
    record = campaign.record(cell.cell_id)
    assert record.state is RunState.QUALIFIED
    qualification = json.loads(
        record.artifact_map["qualification"].resolve(campaign.root).read_text()
    )
    assert all(qualification["checks"].values())
    assert qualification["scientific_outcome"]["passed"] is False
    assert (
        qualification["scientific_outcome"]["checks"]["phase1_identification"] is False
    )
    assert qualification["promotion_eligible"] is False
    assert "scientific_outcome_failed" in qualification["promotion_ineligible_reasons"]


def test_final_training_artifacts_are_reused_not_mutated_on_resume(
    tmp_path: Path,
) -> None:
    cell = _cell(seed=1)
    campaign = _campaign(tmp_path, cell)
    runner = _runner(campaign)
    assert runner.run(stop_after="prepare").completed_cells == 1
    campaign.transition(cell.cell_id, RunState.RUNNING)

    first = runner._invoke(cell.cell_id, "train", resume=False)
    first_refs = {item.kind: item for item in first}
    second = runner._invoke(cell.cell_id, "train", resume=True)
    second_refs = {item.kind: item for item in second}
    assert first_refs["checkpoint"].sha256 == second_refs["checkpoint"].sha256
    assert first_refs["training_report"].sha256 == second_refs["training_report"].sha256
    first_refs["checkpoint"].verify(campaign.root)
    second_refs["checkpoint"].verify(campaign.root)
    assert not (
        campaign.cell_dir(cell.cell_id) / "outputs" / "training-progress.pt"
    ).exists()


def test_isolated_loss_decrease_manifest_lifecycle_and_resume_are_exact(
    tmp_path: Path,
) -> None:
    base = _cell(recipe_index=2, seed=44)
    cell = replace(
        base,
        name="phase1.test.isolated_loss_decrease.s44",
        decisions=merge_decisions(
            base.decisions,
            (
                engineering(
                    "model.selection_score",
                    "isolated_loss_decrease",
                    rationale="exercise the algebraic observed-loss score end to end",
                ),
            ),
        ),
    )
    assert CellSpec.from_manifest(json.loads(json.dumps(cell.to_manifest()))) == cell
    resume_campaign = _campaign(tmp_path / "resume", cell)
    resume_runner = _runner(resume_campaign)
    assert resume_runner.run(stop_after="prepare").completed_cells == 1
    resume_campaign.transition(cell.cell_id, RunState.RUNNING)
    first = {
        item.kind: item
        for item in resume_runner._invoke(cell.cell_id, "train", resume=False)
    }
    resumed = {
        item.kind: item
        for item in resume_runner._invoke(cell.cell_id, "train", resume=True)
    }
    assert resumed["checkpoint"].sha256 == first["checkpoint"].sha256
    assert resumed["training_report"].sha256 == first["training_report"].sha256

    campaign = _campaign(tmp_path / "lifecycle", cell)
    runner = _runner(campaign)
    assert runner.run(limit=1).completed_cells == 1
    record = campaign.record(cell.cell_id)
    assert record.state is RunState.QUALIFIED
    checkpoint_ref = record.artifact_map["checkpoint"]
    report_ref = record.artifact_map["training_report"]
    checkpoint = torch.load(
        checkpoint_ref.resolve(campaign.root),
        map_location="cpu",
        weights_only=True,
    )
    assert checkpoint["model_cfg"]["selection_score"] == ("isolated_loss_decrease")
    assert report_ref.sha256
    evaluation = json.loads(
        record.artifact_map["evaluation"].resolve(campaign.root).read_text()
    )
    for endpoint_name in ("native_selector", "deployed_selector"):
        endpoint = evaluation[endpoint_name]
        diagnostics = endpoint["isolated_loss_gain_diagnostics"]
        assert diagnostics["schema"] == "bsc-isolated-loss-gain-diagnostics-v1"
        assert diagnostics["applicable"] is True
        assert diagnostics["candidate_event_count"] == (
            endpoint["n_tokens"] * cell.decision_map["model.groups"]
        )
        assert (
            sum(
                diagnostics[f"candidate_{sign}_gain_count"]
                for sign in ("negative", "zero", "positive")
            )
            == diagnostics["candidate_event_count"]
        )
        assert diagnostics["candidate_negative_gain_fraction"] >= 0.0
    qualification = json.loads(
        record.artifact_map["qualification"].resolve(campaign.root).read_text()
    )
    assert qualification["checks"]["selection_score_diagnostics_integrity"] is True


@pytest.mark.parametrize("target_ratio", (0.0, 0.03))
def test_initial_loss_ratio_regularizer_is_resolved_once_and_resume_exact(
    tmp_path: Path,
    target_ratio: float,
) -> None:
    base = _cell(recipe_index=2, seed=43)
    cell = replace(
        base,
        name=(
            "phase1.test.regularizer_ratio_"
            + str(target_ratio).replace(".", "p")
            + ".s43"
        ),
        decisions=merge_decisions(
            base.decisions,
            (
                engineering(
                    "objective.regularizer",
                    "end_to_end_map_nuclear",
                    rationale="exercise ratio-calibration provenance",
                ),
                engineering(
                    "objective.regularizer_reduction",
                    "sum_blocks",
                    rationale="exercise the SASA map penalty path",
                ),
                engineering(
                    "objective.regularizer_coefficient",
                    0.0,
                    rationale="reserve the coefficient for calibration",
                ),
                engineering(
                    "objective.regularizer_coefficient_mode",
                    "initial_loss_ratio",
                    rationale="exercise the dimensionless fit contract",
                ),
                engineering(
                    "objective.regularizer_target_initial_ratio",
                    target_ratio,
                    rationale="exercise a declared ratio-ladder point",
                ),
                engineering(
                    "objective.regularizer_calibration_contract",
                    "post_init_train_prefix_true_observation_fp32_v1",
                    rationale="bind the first training batch and fp32 fit",
                ),
            ),
        ),
    )
    campaign = _campaign(tmp_path / f"ratio-{target_ratio}", cell)
    runner = _runner(campaign)
    assert runner.run(stop_after="prepare").completed_cells == 1
    campaign.transition(cell.cell_id, RunState.RUNNING)
    first = {
        item.kind: item for item in runner._invoke(cell.cell_id, "train", resume=False)
    }
    checkpoint = torch.load(
        first["checkpoint"].resolve(campaign.root),
        map_location="cpu",
        weights_only=True,
    )
    report = json.loads(first["training_report"].resolve(campaign.root).read_text())
    calibration = checkpoint["run_binding"]["initialization"]["regularizer_calibration"]
    assert report["regularizer_calibration"] == calibration
    assert calibration["mode"] == "initial_loss_ratio"
    assert calibration["target_initial_ratio"] == target_ratio
    assert calibration["achieved_initial_ratio"] == pytest.approx(target_ratio)
    assert calibration["initial_reconstruction_loss"] > 0
    assert calibration["initial_regularizer_unweighted"] > 0
    assert (
        calibration["resolved_coefficient"] == report["model_cfg"]["lambda_regularizer"]
    )
    assert (calibration["resolved_coefficient"] == 0.0) is (target_ratio == 0.0)
    second = {
        item.kind: item for item in runner._invoke(cell.cell_id, "train", resume=True)
    }
    assert second["checkpoint"].sha256 == first["checkpoint"].sha256
    assert second["training_report"].sha256 == first["training_report"].sha256


def test_nonzero_ratio_regularizer_resumes_exactly_from_recovery_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_ratio = 0.03
    base = _cell(recipe_index=2, seed=45)
    cell = replace(
        base,
        name="phase1.test.regularizer_ratio_interrupted.s45",
        decisions=merge_decisions(
            base.decisions,
            (
                engineering(
                    "objective.regularizer",
                    "end_to_end_map_nuclear",
                    rationale="exercise ratio-calibration recovery",
                ),
                engineering(
                    "objective.regularizer_reduction",
                    "sum_blocks",
                    rationale="exercise the SASA map penalty path",
                ),
                engineering(
                    "objective.regularizer_coefficient",
                    0.0,
                    rationale="reserve the coefficient for calibration",
                ),
                engineering(
                    "objective.regularizer_coefficient_mode",
                    "initial_loss_ratio",
                    rationale="exercise recovery with a resolved coefficient",
                ),
                engineering(
                    "objective.regularizer_target_initial_ratio",
                    target_ratio,
                    rationale="require a nonzero calibrated coefficient",
                ),
                engineering(
                    "objective.regularizer_calibration_contract",
                    "post_init_train_prefix_true_observation_fp32_v1",
                    rationale="bind the first training batch and fp32 fit",
                ),
            ),
        ),
    )

    baseline_campaign = _campaign(tmp_path / "baseline", cell)
    baseline_runner = _runner(baseline_campaign)
    assert baseline_runner.run(stop_after="prepare").completed_cells == 1
    baseline_campaign.transition(cell.cell_id, RunState.RUNNING)
    baseline = {
        item.kind: item
        for item in baseline_runner._invoke(cell.cell_id, "train", resume=False)
    }
    baseline_checkpoint = torch.load(
        baseline["checkpoint"].resolve(baseline_campaign.root),
        map_location="cpu",
        weights_only=True,
    )

    resumed_campaign = _campaign(tmp_path / "resumed", cell)
    resumed_runner = _runner(resumed_campaign)
    assert resumed_runner.run(stop_after="prepare").completed_cells == 1
    resumed_campaign.transition(cell.cell_id, RunState.RUNNING)
    monkeypatch.setenv("BSC_CAMPAIGN_ROOT", str(resumed_campaign.root))
    ctx = _Context(
        resumed_campaign.cell_manifest_path(cell.cell_id),
        resumed_campaign.cell_dir(cell.cell_id) / "train-test-artifacts.json",
        "train",
    )
    prerequisites = ctx.prerequisites()
    original_training_batches = run_cell_module._training_batches

    def interrupt_after_recovery_checkpoint(*args, **kwargs):
        for index, batch in enumerate(original_training_batches(*args, **kwargs)):
            if index == 2:
                raise RuntimeError("intentional interruption after recovery checkpoint")
            yield batch

    with monkeypatch.context() as interruption:
        interruption.setattr(
            run_cell_module,
            "_training_batches",
            interrupt_after_recovery_checkpoint,
        )
        with pytest.raises(
            RuntimeError,
            match="intentional interruption after recovery checkpoint",
        ):
            run_cell_module._train(ctx, prerequisites, resume=False)

    assert ctx.progress.is_file()
    assert not ctx.checkpoint.exists()
    recovery = torch.load(ctx.progress, map_location="cpu", weights_only=True)
    assert recovery["step_idx"] == 2
    assert recovery["accepted_tokens"] == 32
    assert recovery["data_cursor"] == {"next_token": 32, "stream": "train"}
    resolved = recovery["model_cfg"]["lambda_regularizer"]
    assert resolved > 0.0
    assert recovery["run_binding"]["model_cfg"]["lambda_regularizer"] == resolved

    original_load_checkpoint = run_cell_module.Trainer.load_checkpoint
    load_calls: list[Path] = []

    def tracking_load_checkpoint(
        cls,
        path,
        *,
        device="cpu",
        expected_binding=None,
    ):
        load_calls.append(Path(path))
        return original_load_checkpoint(
            path,
            device=device,
            expected_binding=expected_binding,
        )

    with monkeypatch.context() as resume_patch:
        resume_patch.setattr(
            run_cell_module.Trainer,
            "load_checkpoint",
            classmethod(tracking_load_checkpoint),
        )
        resumed = dict(run_cell_module._train(ctx, prerequisites, resume=True))

    assert load_calls == [ctx.progress]
    assert not ctx.progress.exists()
    resumed_checkpoint = torch.load(
        resumed["checkpoint"], map_location="cpu", weights_only=True
    )

    def assert_nested_exact(actual, expected) -> None:
        if torch.is_tensor(expected):
            assert actual.dtype == expected.dtype
            assert actual.shape == expected.shape
            assert torch.equal(
                actual.contiguous().reshape(-1).view(torch.uint8),
                expected.contiguous().reshape(-1).view(torch.uint8),
            )
        elif isinstance(expected, dict):
            assert actual.keys() == expected.keys()
            for key in expected:
                assert_nested_exact(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            assert type(actual) is type(expected)
            assert len(actual) == len(expected)
            for actual_item, expected_item in zip(actual, expected):
                assert_nested_exact(actual_item, expected_item)
        else:
            assert actual == expected

    assert_nested_exact(resumed_checkpoint, baseline_checkpoint)


def test_orphan_final_checkpoint_rebuilds_an_exact_cursor_report(
    tmp_path: Path,
) -> None:
    cell = _cell(seed=29)
    campaign = _campaign(tmp_path, cell)
    runner = _runner(campaign)
    assert runner.run(stop_after="prepare").completed_cells == 1
    campaign.transition(cell.cell_id, RunState.RUNNING)
    first = {
        item.kind: item for item in runner._invoke(cell.cell_id, "train", resume=False)
    }
    first["training_report"].resolve(campaign.root).unlink()

    resumed = {
        item.kind: item for item in runner._invoke(cell.cell_id, "train", resume=True)
    }
    assert resumed["checkpoint"].sha256 == first["checkpoint"].sha256
    report = json.loads(resumed["training_report"].resolve(campaign.root).read_text())
    assert report["attempted_tokens"] == cell.decision_map["data.train_tokens"]
    assert report["data_cursor"] == {
        "next_token": cell.decision_map["data.train_tokens"],
        "stream": "train",
    }
    assert (
        report["step_idx"]
        == (
            cell.decision_map["data.train_tokens"]
            + cell.decision_map["optimizer.batch_tokens"]
            - 1
        )
        // cell.decision_map["optimizer.batch_tokens"]
    )
    checkpoint = torch.load(
        resumed["checkpoint"].resolve(campaign.root),
        map_location="cpu",
        weights_only=True,
    )
    assert report["terminal_log"] == checkpoint["history"][-1]


def test_synthetic_truth_is_independent_of_learner_width_and_support() -> None:
    base = _cell(seed=17)
    learner_changes = {
        "model.block_width": 1,
        "model.active_blocks": 1,
    }
    changed = replace(
        base,
        name="phase1.test.learner_only_delta.s17",
        decisions=tuple(
            replace(decision, value=learner_changes[decision.name])
            if decision.name in learner_changes
            else decision
            for decision in base.decisions
        ),
    )
    left = _synthetic_dataset(base, "train")
    right = _synthetic_dataset(changed, "train")
    assert (
        left.protocol_dict()["stream_digest"] == right.protocol_dict()["stream_digest"]
    )
    assert torch.equal(left.sample(32, start=0).x, right.sample(32, start=0).x)
    assert _synthetic_source_contract(base.decision_map) == _synthetic_source_contract(
        changed.decision_map
    )


def test_every_materialized_prefix_cell_resolves_without_hidden_defaults() -> None:
    phase1 = build_phase1_plan(seeds=(0,), smoke=True)
    phase2 = build_phase2_plan(seeds=(0,), smoke=True)
    for cell in (*phase1.cells, *phase2.cells):
        model, train = validate_cell_config(cell)
        values = cell.decision_map
        assert (
            model.decoder_init_distribution == values["model.decoder_init_distribution"]
        )
        assert (
            model.decoder_init_preconditioning
            == values["model.decoder_init_preconditioning"]
        )
        assert (
            model.decoder_init_operation_order
            == values["model.decoder_init_operation_order"]
        )
        assert train.total_steps == (
            int(values["data.train_tokens"]) + int(values["optimizer.batch_tokens"]) - 1
        ) // int(values["optimizer.batch_tokens"])
    for cell in phase1.cells:
        dataset = _synthetic_dataset(cell, "train")
        _normalization_record(dataset, cell.decision_map)
        _synthetic_source_contract(cell.decision_map)


def test_bsf_encoder_scale_fit_has_an_independent_declared_prefix() -> None:
    cell = next(
        cell
        for cell in build_phase1_plan(seeds=(0,), smoke=True).cells
        if cell.recipe_name == "bsf_vanilla_primary"
    )
    values = cell.decision_map
    assert values["data.normalization"] == "none"
    assert values["data.normalization_fit_count"] == 0
    assert values["model.encoder_scale_fit_split"] == "train_unique_prefix"
    assert values["model.encoder_scale_fit_count"] == 64
    train = _synthetic_dataset(cell, "train")
    preparation = {
        "data": {
            "kind": "synthetic",
            "normalization": _normalization_record(train, values),
        }
    }
    ctx = SimpleNamespace(cell=cell, values=values)
    batches = list(_encoder_scale_fit_batches(ctx, preparation))
    assert sum(len(batch) for batch in batches) == 64
    assert all(torch.isfinite(batch).all() for batch in batches)


def test_phase1_confirmation_uses_a_distinct_bound_stream() -> None:
    cell = _cell(seed=19)
    development = _synthetic_dataset(cell, "eval")
    confirmation = _synthetic_dataset(cell, "confirmation")
    assert (
        development.protocol_dict()["stream_digest"]
        != confirmation.protocol_dict()["stream_digest"]
    )
    assert (
        development.protocol_dict()["split_seed"]
        == cell.decision_map["random.eval_data_seed"]
    )
    assert (
        confirmation.protocol_dict()["split_seed"]
        == cell.decision_map["random.confirmation_data_seed"]
    )
    assert not torch.equal(
        development.sample(16, start=0).x,
        confirmation.sample(16, start=0).x,
    )
    train = _synthetic_dataset(cell, "train")
    preparation = {
        "data": {
            "kind": "synthetic",
            "evaluation_stream": "confirmation",
            "normalization": _normalization_record(train, cell.decision_map),
            "ranges": {
                "factor_calibration": [0, 16],
                "calibration": [16, 32],
                "evaluation": [64, 128],
            },
        }
    }
    factor_calibration_batches = list(
        _synthetic_batches(cell, preparation, "factor_calibration", 8)
    )
    calibration_batches = list(_synthetic_batches(cell, preparation, "calibration", 8))
    assert torch.equal(
        torch.cat(factor_calibration_batches), development.sample(16, start=0).x
    )
    assert torch.equal(
        torch.cat(calibration_batches), development.sample(16, start=16).x
    )
    batches = list(_synthetic_batches(cell, preparation, "evaluation", 16))
    expected = confirmation.sample(64, start=64).x
    assert torch.equal(torch.cat(batches), expected)


def test_store_verification_receipt_skips_rehash_until_file_stat_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "store"
    writer = ShardWriter(
        root,
        "train",
        whitener_hash="raw:test",
        sites=(0,),
        d_model=3,
        meta={"site_dims": [3]},
        tokens_per_shard=8,
    )
    writer.add(
        torch.randn(8, 1, 3),
        torch.stack((torch.arange(8), torch.arange(8)), dim=1),
    )
    writer.close()
    monkeypatch.setenv("BSC_VERIFICATION_CACHE_ROOT", str(tmp_path / "receipts"))
    calls = 0
    original_verify = StoreReader.verify

    def counted(reader, **kwargs):
        nonlocal calls
        calls += 1
        return original_verify(reader, **kwargs)

    monkeypatch.setattr(StoreReader, "verify", counted)
    _VERIFIED_STORE_BINDINGS.clear()
    _verify_store_reader_once(StoreReader(root, "train"), root, "train")
    assert calls == 1
    _VERIFIED_STORE_BINDINGS.clear()  # simulate a fresh stage subprocess
    _verify_store_reader_once(StoreReader(root, "train"), root, "train")
    assert calls == 1

    shard = root / "train" / "shard_00000.safetensors"
    os.utime(shard, None)
    _VERIFIED_STORE_BINDINGS.clear()
    _verify_store_reader_once(StoreReader(root, "train"), root, "train")
    assert calls == 2


def test_ad_hoc_store_verification_uses_process_cache_without_shared_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "store"
    writer = ShardWriter(
        root,
        "train",
        whitener_hash="raw:test",
        sites=(0,),
        d_model=3,
        meta={"site_dims": [3]},
        tokens_per_shard=8,
    )
    writer.add(
        torch.randn(8, 1, 3),
        torch.stack((torch.arange(8), torch.arange(8)), dim=1),
    )
    writer.close()
    monkeypatch.delenv("BSC_VERIFICATION_CACHE_ROOT", raising=False)
    monkeypatch.delenv("BSC_CAMPAIGN_ROOT", raising=False)
    calls = 0
    original_verify = StoreReader.verify

    def counted(reader, **kwargs):
        nonlocal calls
        calls += 1
        return original_verify(reader, **kwargs)

    monkeypatch.setattr(StoreReader, "verify", counted)
    _VERIFIED_STORE_BINDINGS.clear()
    reader = StoreReader(root, "train")
    _verify_store_reader_once(reader, root, "train")
    _verify_store_reader_once(reader, root, "train")
    assert calls == 1

    # A fresh process cannot inherit an unauthenticated receipt from shared
    # temporary storage, so clearing the in-memory cache forces a full verify.
    _VERIFIED_STORE_BINDINGS.clear()
    _verify_store_reader_once(StoreReader(root, "train"), root, "train")
    assert calls == 2


def test_tampered_checkpoint_fails_before_calibration(tmp_path: Path) -> None:
    cell = _cell(seed=2)
    campaign = _campaign(tmp_path, cell)
    runner = _runner(campaign)
    assert runner.run(stop_after="train").completed_cells == 1
    checkpoint = campaign.record(cell.cell_id).artifact_map["checkpoint"]
    checkpoint_path = checkpoint.resolve(campaign.root)
    body = bytearray(checkpoint_path.read_bytes())
    body[len(body) // 2] ^= 0x01
    checkpoint_path.write_bytes(body)

    summary = runner.run(limit=1)
    assert summary.failed_cells == 1
    assert campaign.record(cell.cell_id).state is RunState.FAILED
    messages = [event["message"] for event in campaign.events(cell.cell_id)]
    assert any("calibrate stage failed" in message for message in messages)
    assert not (
        campaign.cell_dir(cell.cell_id) / "outputs" / "calibration-codec.pt"
    ).exists()


def test_codec_event_memory_ceiling_fails_before_materialization(
    tmp_path: Path,
) -> None:
    base = _cell(seed=23)
    cell = replace(
        base,
        name="phase1.test.codec_memory_preflight.s23",
        decisions=tuple(
            replace(decision, value=1)
            if decision.name == "codec.max_calibration_event_bytes"
            else decision
            for decision in base.decisions
        ),
    )
    campaign = _campaign(tmp_path, cell)
    summary = _runner(campaign).run(limit=1, stop_after="calibrate")
    assert summary.failed_cells == 1
    messages = [event["message"] for event in campaign.events(cell.cell_id)]
    assert any(
        "exceeds its resolved event-memory ceiling before event materialization"
        in message
        for message in messages
    )
    assert not (
        campaign.cell_dir(cell.cell_id) / "outputs" / "calibration-codec.pt"
    ).exists()


def test_phase2_without_configured_store_fails_clearly_in_prepare(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cell = _cell(phase=Phase.PHASE2)
    missing = tmp_path / "does-not-exist"
    monkeypatch.setenv("BSC_ACTIVATION_STORE", str(missing))
    monkeypatch.setenv("BSC_STORE_ROOT", str(missing))
    with pytest.raises(CellExecutionError, match="has no train/split.json"):
        _resolve_real_store(cell.decision_map)


def test_declared_but_unimplemented_semantics_fail_closed(tmp_path: Path) -> None:
    base = _cell()
    cell = replace(
        base,
        decisions=tuple(
            replace(decision, value="unimplemented_decoder")
            if decision.name == "model.decoder"
            else decision
            for decision in base.decisions
        ),
    )
    campaign = _campaign(tmp_path, cell)
    summary = _runner(campaign).run(limit=1)
    assert summary.failed_cells == 1
    assert campaign.record(cell.cell_id).state is RunState.FAILED
    messages = [event["message"] for event in campaign.events(cell.cell_id)]
    assert any("unknown resolved decoder" in message for message in messages)


def test_support_confusion_distinguishes_fdr_from_false_positive_rate() -> None:
    truth = torch.tensor([[True, False, False], [False, True, False]])
    predicted = torch.tensor([[True, True, False], [False, True, False]])
    metrics = _support_confusion(predicted, truth)
    assert metrics["precision"] == 2 / 3
    assert metrics["false_discovery_rate"] == 1 / 3
    assert metrics["false_positive_rate"] == 1 / 4


def test_matching_pathologies_keep_split_and_merge_directions_distinct() -> None:
    perfect = _matching_pathologies(torch.eye(3))
    assert perfect["split_factor_fraction"] == 0
    assert perfect["merge_group_fraction"] == 0

    split = _matching_pathologies(torch.tensor([[0.9, 0.8, 0.0], [0.0, 0.0, 0.9]]))
    assert split["split_factor_fraction"] == 0.5
    assert split["merge_group_fraction"] == 0

    merge = _matching_pathologies(torch.tensor([[0.9, 0.0], [0.8, 0.0], [0.0, 0.9]]))
    assert merge["split_factor_fraction"] == 0
    assert merge["merge_group_fraction"] == 0.5


def test_real_source_contract_is_exact_not_shape_only() -> None:
    cell = _cell(phase=Phase.PHASE2)
    expected = _expected_real_source_contract(cell.decision_map)
    assert _verify_real_source_contract(expected, cell.decision_map) == expected

    wrong = {**expected, "corpus_revision": "wrong-but-same-shape"}
    with pytest.raises(CellExecutionError, match="corpus_revision"):
        _verify_real_source_contract(wrong, cell.decision_map)

    wrong_capture = {**expected, "text_field": "body"}
    with pytest.raises(CellExecutionError, match="text_field"):
        _verify_real_source_contract(wrong_capture, cell.decision_map)

    unexpected = {**expected, "undeclared_capture_behavior": True}
    with pytest.raises(CellExecutionError, match="undeclared_capture_behavior"):
        _verify_real_source_contract(unexpected, cell.decision_map)


def test_synthetic_capture_contract_is_exact() -> None:
    cell = _cell()
    source = _synthetic_source_contract(cell.decision_map)["contract"]
    assert source["coordinate_amplitude_law"] == "gaussian"
    assert source["factor_subspace_overlap"] == "uncontrolled"
    changed = replace(
        cell,
        decisions=tuple(
            replace(
                decision,
                value=(("version", "wrong"),),
            )
            if decision.name == "data.capture_contract"
            else decision
            for decision in cell.decisions
        ),
    )
    with pytest.raises(CellExecutionError, match="data.capture_contract"):
        _synthetic_source_contract(changed.decision_map)


@pytest.mark.parametrize(
    ("decision_name", "decision_value", "protocol_key"),
    (
        (
            "data.coordinate_amplitude_law",
            "student_t_df3",
            "coordinate_amplitude_law",
        ),
        (
            "data.factor_subspace_overlap",
            "paired_30deg",
            "factor_subspace_overlap",
        ),
    ),
)
def test_executor_maps_phase1_robustness_geometry_into_the_generator(
    decision_name: str,
    decision_value: str,
    protocol_key: str,
) -> None:
    base = _cell(recipe_index=2)
    cell = replace(
        base,
        decisions=tuple(
            replace(decision, value=decision_value)
            if decision.name == decision_name
            else decision
            for decision in base.decisions
        ),
    )
    dataset = _synthetic_dataset(cell, "confirmation")
    assert dataset.protocol_dict()["sampling"][protocol_key] == decision_value
    assert (
        _synthetic_source_contract(cell.decision_map)["contract"][
            decision_name.removeprefix("data.")
        ]
        == decision_value
    )


def test_rate_envelope_prunes_higher_rate_worse_distortion_points() -> None:
    two = _lower_convex_rate_envelope(
        (
            {"name": "low", "total_bits_per_token": 4.0, "raw_space_fvu": 0.8},
            {"name": "high", "total_bits_per_token": 8.0, "raw_space_fvu": 0.9},
        )
    )
    assert [point["name"] for point in two] == ["low"]

    three = _lower_convex_rate_envelope(
        (
            {"name": "zero", "total_bits_per_token": 0.0, "raw_space_fvu": 1.0},
            {"name": "q4", "total_bits_per_token": 4.0, "raw_space_fvu": 0.8},
            {"name": "q8", "total_bits_per_token": 8.0, "raw_space_fvu": 0.9},
        )
    )
    assert [point["name"] for point in three] == ["zero", "q4"]


def test_balanced_time_sharing_schedule_has_exact_count_without_mode_bits() -> None:
    decisions = [
        _balanced_schedule_uses_upper(index, upper_tokens=3, horizon_tokens=11)
        for index in range(11)
    ]
    assert sum(decisions) == 3
    assert decisions == [
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        True,
        False,
        False,
        True,
    ]
    with pytest.raises(ValueError, match="invalid balanced"):
        _balanced_schedule_uses_upper(11, upper_tokens=3, horizon_tokens=11)


def test_cached_time_sharing_matches_paired_batch_execution() -> None:
    chunks = (
        torch.tensor(
            [[10.0, 11.0, 12.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=torch.float64,
        ),
        torch.tensor(
            [
                [13.0, 14.0, 15.0, 16.0],
                [7.0, 8.0, 9.0, 10.0],
                [11.0, 12.0, 13.0, 14.0],
            ],
            dtype=torch.float64,
        ),
        torch.tensor(
            [
                [17.0, 18.0, 19.0, 20.0],
                [15.0, 16.0, 17.0, 18.0],
                [19.0, 20.0, 21.0, 22.0],
            ],
            dtype=torch.float64,
        ),
    )
    cache = _RawEndpointErrorCache(
        endpoint_names=("zero_event_calibration_mean", "q4", "q8"),
        chunks=chunks,
        tokens=11,
        pooled_denominator=200.0,
    )
    plan = {
        "schedule": {
            "lower_name": "q4",
            "upper_name": "q8",
            "upper_tokens": 3,
            "horizon_tokens": 11,
        }
    }
    result = _evaluate_cached_time_sharing(cache, plan, device=torch.device("cpu"))
    lower = torch.cat([chunk[1] for chunk in chunks])
    upper = torch.cat([chunk[2] for chunk in chunks])
    mask = torch.tensor(
        [
            _balanced_schedule_uses_upper(
                index,
                upper_tokens=3,
                horizon_tokens=11,
            )
            for index in range(11)
        ]
    )
    expected = float(torch.where(mask, upper, lower).sum() / 200.0)
    assert result["schedule"]["raw_space_fvu"] == expected
    assert result["schedule"]["evaluation_upper_tokens"] == 3
    prefix_plan = {
        "schedule": {
            **plan["schedule"],
            "upper_tokens": 6,
            "horizon_tokens": 22,
        }
    }
    prefix = _evaluate_cached_time_sharing(
        cache,
        prefix_plan,
        device=torch.device("cpu"),
    )
    assert prefix["schedule"]["raw_space_fvu"] == expected
    assert prefix["schedule"]["evaluation_tokens"] == 11
    with pytest.raises(CellExecutionError, match="horizon"):
        _evaluate_cached_time_sharing(
            cache,
            {"schedule": {**plan["schedule"], "horizon_tokens": 10}},
            device=torch.device("cpu"),
        )


def _schedule_bundle(
    tmp_path: Path,
    *,
    cell: CellSpec,
    deployment_hash: str,
    values: dict,
    plans: dict,
) -> tuple[Path, dict, dict]:
    path = tmp_path / "deployment-schedules.bin"
    manifest = _write_deployment_schedule_bundle(
        path,
        cell_id=cell.cell_id,
        deployment_codec_sha256=deployment_hash,
        schedule_contract=values["codec.time_sharing_schedule_contract"],
        plans=plans,
    )
    loaded = _load_deployment_schedule_bundle(
        path,
        manifest,
        cell_id=cell.cell_id,
        deployment_codec_sha256=deployment_hash,
        schedule_contract=values["codec.time_sharing_schedule_contract"],
    )
    return path, manifest, loaded


def test_deployment_schedule_bundle_roundtrip_and_tamper_refusal(
    tmp_path: Path,
) -> None:
    cell = _cell(phase=Phase.PHASE2)
    deployment_hash = "ab" * 32
    budget = 384.0
    horizon = 100_000_000
    upper_tokens = 12_345_678
    schedule_key = _time_sharing_plan_key(
        budget=budget,
        lower_name="q4",
        upper_name="q8",
        upper_tokens=upper_tokens,
        horizon_tokens=horizon,
    )
    path, manifest, loaded = _schedule_bundle(
        tmp_path,
        cell=cell,
        deployment_hash=deployment_hash,
        values=cell.decision_map,
        plans={
            schedule_key: {
                "budget_bits_per_token": budget,
                "lower_name": "q4",
                "lower_q": 4,
                "upper_name": "q8",
                "upper_q": 8,
                "upper_tokens": upper_tokens,
                "horizon_tokens": horizon,
                "upper_mixture_weight": upper_tokens / horizon,
                "achieved_total_bits_per_token": 383.5,
            }
        },
    )
    assert path.stat().st_size == 32
    assert manifest["record_count"] == 1
    assert loaded[schedule_key]["deployment_schedule"]["size_bytes"] == 32

    damaged = bytearray(path.read_bytes())
    damaged[9] ^= 1
    path.write_bytes(damaged)
    with pytest.raises(CellExecutionError, match="bytes/manifest|record hash"):
        _load_deployment_schedule_bundle(
            path,
            manifest,
            cell_id=cell.cell_id,
            deployment_codec_sha256=deployment_hash,
            schedule_contract=cell.decision_map["codec.time_sharing_schedule_contract"],
        )
    forged_manifest = copy.deepcopy(manifest)
    forged_manifest["artifact_sha256"] = hashlib.sha256(damaged).hexdigest()
    forged_manifest["records"][0]["sha256"] = hashlib.sha256(damaged).hexdigest()
    with pytest.raises(CellExecutionError, match="header binding"):
        _load_deployment_schedule_bundle(
            path,
            forged_manifest,
            cell_id=cell.cell_id,
            deployment_codec_sha256=deployment_hash,
            schedule_contract=cell.decision_map["codec.time_sharing_schedule_contract"],
        )


def test_fixed_rate_budget_uses_better_lower_rate_packet(
    tmp_path: Path,
) -> None:
    cell = _cell(phase=Phase.PHASE2)
    values = {
        **cell.decision_map,
        "evaluation.fixed_rate_budgets_bits_per_token": (8.0,),
        "evaluation.side_information_amortization_tokens": 1_000_000,
    }
    deployment = tmp_path / "deployable-codec.pt"
    deployment.write_bytes(b"x")
    deployment_hash = hashlib.sha256(deployment.read_bytes()).hexdigest()
    _, schedule_manifest, _ = _schedule_bundle(
        tmp_path,
        cell=cell,
        deployment_hash=deployment_hash,
        values=values,
        plans={},
    )
    ctx = SimpleNamespace(cell=cell, values=values)
    result = _fixed_rate_raw_score(
        ctx,
        rd={
            "rate_model": "fixed_width_count_plus_block_ids_plus_amplitudes_v1",
            "support_count_width_bits": 5,
            "support_id_width_bits": 8,
            "codec_meta": {
                "packet_contract": values["codec.packet_contract"],
                "side_information_contract": values["codec.side_information_contract"],
                "count_alphabet_max": 16,
            },
            "points": {
                "4": {"rate_bits_per_token": 4.0},
                "8": {"rate_bits_per_token": 8.0},
            },
        },
        raw_space={
            "eligible": True,
            "points": {
                "4": {"fvu_pooled": 0.8},
                "8": {"fvu_pooled": 0.9},
            },
        },
        deployment_path=deployment,
        deployment_hash=deployment_hash,
        calibration_hash="0" * 64,
        deployment_schedule_manifest=schedule_manifest,
    )
    assert result["eligible"] is True
    assert result["fixed_budgets"][0]["raw_space_fvu"] == pytest.approx(0.8)
    assert result["fixed_budgets"][0]["bracket"] == ["q4", "q4"]


def test_ineligible_fixed_rate_endpoint_produces_durable_worst_score(
    tmp_path: Path,
) -> None:
    cell = _cell(phase=Phase.PHASE2)
    values = {
        **cell.decision_map,
        "evaluation.fixed_rate_budgets_bits_per_token": (0.001,),
        "evaluation.side_information_amortization_tokens": 1_000,
    }
    deployment = tmp_path / "deployable-codec.pt"
    deployment.write_bytes(b"x")
    deployment_hash = hashlib.sha256(deployment.read_bytes()).hexdigest()
    _, schedule_manifest, _ = _schedule_bundle(
        tmp_path,
        cell=cell,
        deployment_hash=deployment_hash,
        values=values,
        plans={},
    )
    fixed_rate = _fixed_rate_raw_score(
        SimpleNamespace(cell=cell, values=values),
        rd={
            "rate_model": "fixed_width_decodable_payload_bits_v1",
            "support_count_width_bits": 5,
            "support_id_width_bits": 8,
            "codec_meta": {
                "packet_contract": values["codec.packet_contract"],
                "side_information_contract": values["codec.side_information_contract"],
                "count_alphabet_max": 16,
            },
            "points": {
                "4": {"rate_bits_per_token": 4.0},
                "8": {"rate_bits_per_token": 8.0},
            },
        },
        raw_space={
            "eligible": True,
            "points": {
                "4": {"fvu_pooled": 0.8},
                "8": {"fvu_pooled": 0.7},
            },
        },
        deployment_path=deployment,
        deployment_hash=deployment_hash,
        calibration_hash="0" * 64,
        deployment_schedule_manifest=schedule_manifest,
    )
    assert fixed_rate["eligible"] is False
    assert fixed_rate["selection_score"] is None
    validation = _selection_validation_metrics(
        Phase.PHASE2,
        identification=None,
        fixed_rate=fixed_rate,
    )
    assert validation == {
        PHASE2_SELECTION_METRIC_KEY: PHASE2_INELIGIBLE_SELECTION_SCORE
    }
    policy = build_phase2_plan(seeds=(0,), smoke=True).stages[0].selection_policy
    assert policy is not None
    assert (
        Campaign._policy_metric({"validation": validation}, policy)
        == PHASE2_INELIGIBLE_SELECTION_SCORE
    )


def test_fixed_rate_mixture_prices_reproducible_schedule_header(
    tmp_path: Path,
) -> None:
    cell = _cell(phase=Phase.PHASE2)
    values = {
        **cell.decision_map,
        "evaluation.fixed_rate_budgets_bits_per_token": (6.0,),
        "evaluation.side_information_amortization_tokens": 1_000,
    }
    deployment = tmp_path / "deployable-codec.pt"
    deployment.write_bytes(b"x")
    side_rate = 8.0 * deployment.stat().st_size / 1_000
    schedule_rate = 8.0 * 32 / 1_000
    lower_rate = 4.0 + side_rate
    upper_rate = 8.0 + side_rate
    upper_tokens = int(
        ((6.0 - schedule_rate - lower_rate) / (upper_rate - lower_rate)) * 1_000
    )
    schedule_key = _time_sharing_plan_key(
        budget=6.0,
        lower_name="q4",
        upper_name="q8",
        upper_tokens=upper_tokens,
        horizon_tokens=1_000,
    )
    deployment_hash = hashlib.sha256(deployment.read_bytes()).hexdigest()
    achieved_rate = (
        (1.0 - upper_tokens / 1_000) * lower_rate
        + (upper_tokens / 1_000) * upper_rate
        + schedule_rate
    )
    _, schedule_manifest, loaded_schedules = _schedule_bundle(
        tmp_path,
        cell=cell,
        deployment_hash=deployment_hash,
        values=values,
        plans={
            schedule_key: {
                "budget_bits_per_token": 6.0,
                "lower_name": "q4",
                "lower_q": 4,
                "upper_name": "q8",
                "upper_q": 8,
                "upper_tokens": upper_tokens,
                "horizon_tokens": 1_000,
                "upper_mixture_weight": upper_tokens / 1_000,
                "achieved_total_bits_per_token": achieved_rate,
            }
        },
    )
    rd_payload = {
        "rate_model": "fixed_width_decodable_payload_bits_v1",
        "support_count_width_bits": 5,
        "support_id_width_bits": 8,
        "codec_meta": {
            "packet_contract": values["codec.packet_contract"],
            "side_information_contract": values["codec.side_information_contract"],
            "count_alphabet_max": 16,
        },
        "points": {
            "4": {"rate_bits_per_token": 4.0},
            "8": {"rate_bits_per_token": 8.0},
        },
    }
    raw_payload = {
        "eligible": True,
        "points": {
            "4": {"fvu_pooled": 0.5},
            "8": {"fvu_pooled": 0.4},
        },
        "operational_time_sharing": {
            schedule_key: {
                **loaded_schedules[schedule_key],
                "evaluation_tokens": 17,
                "evaluation_upper_tokens": 7,
                "raw_space_fvu": 0.37,
                "distortion_measurement": (
                    "executed_balanced_schedule_on_paired_raw_evaluation_rows"
                ),
            }
        },
    }
    result = _fixed_rate_raw_score(
        SimpleNamespace(cell=cell, values=values),
        rd=rd_payload,
        raw_space=raw_payload,
        deployment_path=deployment,
        deployment_hash=deployment_hash,
        calibration_hash="0" * 64,
        deployment_schedule_manifest=schedule_manifest,
    )
    point = result["fixed_budgets"][0]
    assert point["eligible"] is True
    # This is the executed schedule result, not the 0.4566... analytic blend
    # of the two aggregate endpoint FVUs.
    assert point["raw_space_fvu"] == pytest.approx(0.37)
    assert point["bracket"] == ["q4", "q8"]
    assert point["achieved_total_bits_per_token"] <= 6.0
    schedule = point["mixing_schedule"]
    assert schedule["contract"] == "balanced_global_token_counter_u64_v1"
    assert schedule["header_bytes"] == 32
    assert schedule["per_token_mode_bits"] == 0
    assert schedule["upper_tokens"] == round(
        point["upper_mixture_weight"] * schedule["horizon_tokens"]
    )
    assert schedule["evaluation_upper_tokens"] == 7
    assert schedule["distortion_measurement"].startswith("executed_")

    worse_schedule = copy.deepcopy(raw_payload)
    worse_schedule["operational_time_sharing"][schedule_key]["raw_space_fvu"] = 0.6
    fallback = _fixed_rate_raw_score(
        SimpleNamespace(cell=cell, values=values),
        rd=rd_payload,
        raw_space=worse_schedule,
        deployment_path=deployment,
        deployment_hash=deployment_hash,
        calibration_hash="0" * 64,
        deployment_schedule_manifest=schedule_manifest,
    )["fixed_budgets"][0]
    assert fallback["bracket"] == ["q4", "q4"]
    assert fallback["raw_space_fvu"] == pytest.approx(0.5)
    assert fallback["reason"] == "lower_endpoint_outperformed_executed_time_sharing"


def test_phase2_evaluator_metric_schema_resolves_through_live_policy() -> None:
    validation = _selection_validation_metrics(
        Phase.PHASE2,
        identification=None,
        fixed_rate={"selection_score": -0.375},
    )
    assert validation == {PHASE2_SELECTION_METRIC_KEY: -0.375}
    policy = build_phase2_plan(seeds=(0,), smoke=True).stages[0].selection_policy
    assert policy is not None
    assert policy.metric_path == PHASE2_SELECTION_METRIC_PATH
    assert policy.direction == "max"
    selection_metrics = {"validation": validation}
    assert Campaign._policy_metric(selection_metrics, policy) == -0.375
    better = {
        "validation": _selection_validation_metrics(
            Phase.PHASE2,
            identification=None,
            fixed_rate={"selection_score": -0.25},
        )
    }
    assert Campaign._policy_metric(better, policy) > Campaign._policy_metric(
        selection_metrics, policy
    )


def test_bsf_released_bridge_wires_shared_group_threshold() -> None:
    base = _cell()
    release = RELEASE_DIAGNOSTIC_RECIPES["bsf_group_lasso_released_paper_mode_drift"]
    threshold_names = {
        "model.threshold_scope",
        "model.threshold_parameterization",
        "model.threshold_raw_init",
        "model.threshold_effective_init",
    }
    cell = replace(
        base,
        decisions=merge_decisions(
            base.decisions,
            tuple(
                decision
                for decision in release.decisions
                if decision.name in threshold_names
            ),
        ),
    )
    config = _model_config(cell)
    assert config.group_threshold_scope == "shared_scalar"
    assert config.group_threshold_parameterization == "exp"
    assert config.group_threshold_raw_init == 0
    assert config.group_threshold_effective_init == 1


def test_anthropic_anchor_maps_only_the_minimal_dense_l1_method() -> None:
    cell = next(
        cell
        for cell in build_phase1_plan(seeds=(0,), smoke=True).cells
        if cell.recipe_name == "anthropic_crosscoder_architecture_bridge"
    )
    config = _model_config(cell)
    assert config.selection == "dense"
    assert config.code_activation == "relu"
    assert config.regularizer == "crosscoder_l1"
    assert config.lambda_regularizer > 0
    # The dense-L1 root is excluded from the sparse-finalist allowlist, but it
    # must remain eligible for its own independently calibrated comparator
    # family chain.
    assert cell.decision_map["qualification.promotable"] is True


def test_capture_contract_refuses_reordered_or_reassigned_split_allocation(
    tmp_path: Path,
) -> None:
    values = _phase3_cell().decision_map
    source = _expected_real_source_contract(values)
    source_hash = hashlib.sha256(canonical_json(source).encode("utf-8")).hexdigest()
    split_order, split_plan = _expected_capture_allocation(values)
    payload = {
        "schema": "bsc-capture-manifest-v1",
        "source": source,
        "source_hash": source_hash,
        "split_order": list(split_order),
        "split_plan": split_plan,
        "splits": split_plan,
        "capture_implementation": {"test_runtime": "exact"},
        "capture_binding_sha256": hashlib.sha256(b"binding").hexdigest(),
    }
    tmp_path.mkdir(exist_ok=True)
    capture_path = tmp_path / "capture.json"
    capture_path.write_text(json.dumps(payload, indent=2) + "\n")
    loaded = _load_capture_contract(tmp_path, values)
    assert loaded["split_order"] == split_order
    assert loaded["split_plan"] == split_plan

    reordered = copy.deepcopy(payload)
    reordered["split_order"][0], reordered["split_order"][1] = (
        reordered["split_order"][1],
        reordered["split_order"][0],
    )
    capture_path.write_text(json.dumps(reordered, indent=2) + "\n")
    with pytest.raises(CellExecutionError, match="split order"):
        _load_capture_contract(tmp_path, values)

    reassigned = copy.deepcopy(payload)
    first, second = split_order[:2]
    (
        reassigned["split_plan"][first]["sequence_start"],
        reassigned["split_plan"][second]["sequence_start"],
    ) = (
        reassigned["split_plan"][second]["sequence_start"],
        reassigned["split_plan"][first]["sequence_start"],
    )
    reassigned["splits"] = copy.deepcopy(reassigned["split_plan"])
    capture_path.write_text(json.dumps(reassigned, indent=2) + "\n")
    with pytest.raises(CellExecutionError, match="split allocation"):
        _load_capture_contract(tmp_path, values)


def test_phase3_single_raw_store_resolves_bound_transform_only_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cell = _phase3_cell()
    values = cell.decision_map
    raw_root = tmp_path / "raw"
    source = _expected_real_source_contract(values)
    source.update(
        {
            "format_version": 2,
            "row_identity_columns": ["sequence", "position", "token_id"],
            "capture_mode": "raw_once",
        }
    )
    source_hash = hashlib.sha256(canonical_json(source).encode("utf-8")).hexdigest()
    split_order, split_plan = _expected_capture_allocation(values)
    capture_binding_sha256 = hashlib.sha256(b"phase3-test-capture-binding").hexdigest()
    raw_root.mkdir()
    (raw_root / "capture.json").write_text(
        json.dumps(
            {
                "schema": "bsc-capture-manifest-v1",
                "source": source,
                "source_hash": source_hash,
                "split_order": list(split_order),
                "split_plan": split_plan,
                "splits": split_plan,
                "capture_implementation": {"test_runtime": "exact"},
                "capture_binding_sha256": capture_binding_sha256,
            },
            indent=2,
        )
        + "\n"
    )
    store_site_count = len(values["data.store_sites"])
    captured_width = max(int(item) for item in values["data.site_dims"])
    for split in split_order:
        allocation = split_plan[split]
        sequence_start = allocation["sequence_start"]
        sequence_stop = allocation["sequence_stop_exclusive"]
        tokens_per_sequence = allocation["tokens_per_sequence"]
        n_tokens = allocation["actual_tokens"]
        generator = torch.Generator().manual_seed(sequence_start + 1)
        x = torch.randn(n_tokens, store_site_count, captured_width, generator=generator)
        row_ids = torch.stack(
            (
                torch.arange(sequence_start, sequence_stop).repeat_interleave(
                    tokens_per_sequence
                ),
                torch.arange(1, tokens_per_sequence + 1).repeat(
                    sequence_stop - sequence_start
                ),
                torch.arange(n_tokens),
            ),
            dim=1,
        )
        writer = ShardWriter(
            raw_root,
            split,
            whitener_hash=f"raw:{source_hash}",
            sites=range(store_site_count),
            d_model=captured_width,
            meta={
                **source,
                "site_dims": [captured_width] * store_site_count,
                "split_requested_tokens": allocation["requested_tokens"],
                "split_actual_tokens": n_tokens,
                "sequence_start": sequence_start,
                "sequence_stop_exclusive": sequence_stop,
                "tokens_per_sequence": tokens_per_sequence,
                "sequence_allocation": "whole_packed_contexts_v1",
                "capture_binding_sha256": capture_binding_sha256,
                "ordered_split_allocation": list(split_order),
            },
            tokens_per_shard=n_tokens,
        )
        writer.add(x, row_ids)
        writer.close()
    fit_transform_artifacts(
        raw_root,
        raw_root / "transforms",
        ("scalar_rms",),
        batch_size=16,
    )
    monkeypatch.setenv("BSC_ACTIVATION_STORE", str(raw_root))
    monkeypatch.setenv("BSC_TRANSFORM_ROOT", str(raw_root / "transforms"))
    resolved = _resolve_real_store(values)
    assert resolved["store_view_policy"].startswith("single_bf16_raw_view")
    assert (
        resolved["bindings"]["train"]["n_tokens"]
        == split_plan["train"]["actual_tokens"]
    )
    assert resolved["normalization"]["application"] == "on_the_fly"
    assert (
        resolved["normalization"]["source_capture_sha256"]
        == resolved["source_contract"]["sha256"]
    )

    # Campaign registration deliberately refuses an isolated Phase-3 stage:
    # the real launch must carry the exact frozen-panel and blueprint
    # manifests.  Exercise the numerical refusal gate directly here while
    # the campaign tests cover manifest binding and forgery rejection.
    preparation = {"data": {"kind": "activation_store", **resolved}}
    assert not _transform_on_cuda(preparation, torch.device("cpu"))
    assert _transform_on_cuda(preparation, torch.device("cuda"))
    preflight_ctx = SimpleNamespace(values=values)
    init_x = next(_training_batches(preflight_ctx, preparation, start_token=0))
    model = BlockCrosscoder(_model_config(cell))
    model.initialize_decoder_bias_(init_x.float())
    model.project_decoder_()
    _apply_encoder_scale_calibration(preflight_ctx, preparation, model)
    precision = _production_precision_preflight(
        preflight_ctx,
        model,
        init_x,
        None,
    )
    assert precision["applicable"] is True
    assert precision["contract"] == "fp32_bf16_initial_forward_v1"
    assert precision["passed"] is True
    assert precision["reconstruction_relative_error"] <= 0.05
    assert precision["support_iou"] >= 0.90

    replay_values = {
        **values,
        "data.unique_tokens": 10,
        "data.train_tokens": 23,
        "optimizer.batch_tokens": 8,
    }
    replay_resolved = _resolve_real_store(replay_values)
    preparation = {"data": {"kind": "activation_store", **replay_resolved}}
    replay_ctx = SimpleNamespace(values=replay_values)
    full_batches = list(_training_batches(replay_ctx, preparation, start_token=0))
    assert [len(batch) for batch in full_batches] == [8, 8, 7]
    resumed_batches = list(_training_batches(replay_ctx, preparation, start_token=16))
    assert [len(batch) for batch in resumed_batches] == [7]
    assert torch.equal(torch.cat(full_batches)[16:], torch.cat(resumed_batches))

    final_reader = StoreReader(raw_root, "final")
    final_batches = list(final_reader.sequential_batches_with_ids(128))
    final_meta = dict(final_reader.manifest["meta"])
    (raw_root / "final").rename(tmp_path / "valid-final")
    bad_writer = ShardWriter(
        raw_root,
        "final",
        whitener_hash=final_reader.whitener_hash,
        sites=final_reader.sites,
        d_model=final_reader.d_model,
        meta=final_meta,
        tokens_per_shard=final_reader.manifest["tokens_per_shard"],
    )
    for acts, row_ids in final_batches:
        shifted_ids = row_ids.clone()
        shifted_ids[:, 0] += 100
        bad_writer.add(acts, shifted_ids)
    bad_writer.close()
    _VERIFIED_STORE_BINDINGS.clear()
    with pytest.raises(CellExecutionError, match="row identity sequence"):
        _resolve_real_store(values)


def test_training_batches_do_not_copy_aligned_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = [
        torch.full((8, 2, 3), float(index))
        for index in range(4)
    ]

    class Reader:
        def shuffled_batches(self, *args, **kwargs):
            yield from chunks

    monkeypatch.setattr(run_cell_module, "_store_reader", lambda *args: Reader())
    values = {
        "optimizer.batch_tokens": 8,
        "data.train_tokens": 32,
        "data.unique_tokens": 32,
        "random.train_data_seed": 7,
    }
    preparation = {"data": {"kind": "activation_store"}}
    batches = list(
        _training_batches(
            SimpleNamespace(values=values),
            preparation,
            start_token=0,
            apply_transform=False,
        )
    )
    assert len(batches) == len(chunks)
    assert all(
        batch.untyped_storage().data_ptr() == chunk.untyped_storage().data_ptr()
        for batch, chunk in zip(batches, chunks)
    )

    tails = [torch.arange(60).reshape(10, 2, 3).float() + 60 * i for i in range(3)]
    monkeypatch.setattr(
        run_cell_module,
        "_store_reader",
        lambda *args: type(
            "TailReader",
            (),
            {"shuffled_batches": lambda self, *a, **kw: iter(tails)},
        )(),
    )
    values.update(
        {
            "data.train_tokens": 23,
            "data.unique_tokens": 10,
        }
    )
    full = list(
        _training_batches(
            SimpleNamespace(values=values),
            preparation,
            start_token=0,
            apply_transform=False,
        )
    )
    resumed = list(
        _training_batches(
            SimpleNamespace(values=values),
            preparation,
            start_token=16,
            apply_transform=False,
        )
    )
    assert [len(batch) for batch in full] == [8, 8, 7]
    assert [len(batch) for batch in resumed] == [7]
    assert torch.equal(torch.cat(full)[16:], torch.cat(resumed))


def test_released_and_adapted_mechanics_reach_declared_config_branches() -> None:
    base = _cell()
    sasa = RELEASE_DIAGNOSTIC_RECIPES["sasa_released_code_drift"]
    sasa_cell = replace(
        base,
        decisions=merge_decisions(
            base.decisions,
            tuple(
                decision
                for decision in sasa.decisions
                if decision.name in {"model.encoder", "model.encoder_constraint"}
            ),
        ),
    )
    assert _model_config(sasa_cell).encoder_mode == "untied"
    assert _model_config(sasa_cell).encoder_constraint == "unit_latent"

    adapted_values = {
        "model.block_width": 1,
        "model.selector": "decoder_weighted_batchtopk",
        "objective.auxiliary": "decoder_weighted_token_horizon_residual",
        "objective.auxiliary_reconstruction": ("squared_l2_over_residual_variance"),
        "auxiliary.apply_b_dec_to_input": False,
    }
    adapted_cell = replace(
        base,
        name="phase1.test.decoder_weighted_token_horizon.s0",
        decisions=tuple(
            replace(decision, value=adapted_values[decision.name])
            if decision.name in adapted_values
            else decision
            for decision in base.decisions
        ),
    )
    train = _train_config(adapted_cell)
    assert train.aux_variant == "decoder_weighted_token_horizon"
    assert train.aux_reconstruction == "squared_l2_over_residual_variance"
