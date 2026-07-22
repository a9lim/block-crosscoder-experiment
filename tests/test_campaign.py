import copy
import concurrent.futures
import hashlib
import json
import os
import signal
import shutil
import socket
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from statistics import median
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import block_crosscoder_experiment.campaign as campaign_module
from block_crosscoder_experiment.campaign import (
    ArtifactError,
    ArtifactRef,
    Campaign,
    CampaignError,
    CampaignLocked,
    CampaignRecord,
    CampaignRunner,
    EVALUATION_SCHEMA,
    InvalidTransition,
    PREPARATION_SCHEMA,
    PROMOTION_SCHEMA,
    QUALIFICATION_SCHEMA,
    RunState,
)
from block_crosscoder_experiment.cli.matrix import main as matrix_main
from block_crosscoder_experiment.studies import (
    CellSpec,
    FrozenPanelDecision,
    FrozenSelection,
    GateCondition,
    Phase,
    Phase1Blueprint,
    SELECTION_THRESHOLD_BASIS,
    SELECTION_THRESHOLD_SENSITIVITY,
    SelectionPolicy,
    StageSpec,
    StudyPlan,
    build_phase1_blueprint,
    build_phase1_plan,
    build_phase1_transfer,
    build_phase2_blueprint,
    build_phase2_plan,
    build_phase3_blueprint,
    build_phase3_plan,
    canonical_json,
    content_id,
    materialize_child_plan,
    materialize_family_child_plan,
    materialize_family_revisit_plan,
)


TEST_IMPLEMENTATION_IDENTITY = {
    "executor_schema": "bsc-cell-executor-v12",
    "executor_process_model": "persistent_exact_snapshot_lineage_v5",
    "python_source_sha256": "1" * 64,
    "python_source_files": 1,
    "git_commit": "1" * 40,
    "git_dirty": False,
    "python": "3.12.test",
    "torch": "2.8.0",
    "torch_cuda_build": "12.8",
    "dependencies": {
        "datasets": "test",
        "huggingface-hub": "test",
        "numpy": "test",
        "sae-lens": "test",
        "safetensors": "test",
        "torch": "test",
        "transformers": "test",
    },
}
TEST_IMPLEMENTATION_IDENTITY_SHA256 = hashlib.sha256(
    canonical_json(TEST_IMPLEMENTATION_IDENTITY).encode("utf-8")
).hexdigest()


def phase1_selection_template(
    seed: int, *, smoke: bool = True
) -> tuple[CellSpec, object]:
    source_stage = (
        build_phase1_plan((seed,), smoke=True).stages[-1]
        if smoke
        else build_phase1_plan().stages[-1]
    )
    resolved = next(
        cell
        for cell in source_stage.cells
        if cell.seed == seed
        if cell.recipe_name in source_stage.selection_policy.eligible_recipe_names
    )
    return resolved, source_stage.selection_policy


def one_cell_plan(*, seed: int = 0, smoke: bool = True) -> StudyPlan:
    resolved, selection_policy = phase1_selection_template(seed, smoke=smoke)
    cell = CellSpec(
        name=f"phase1.test.cell.s{seed}",
        phase=Phase.PHASE1,
        stage="test",
        recipe_name=resolved.recipe_name,
        recipe_id=resolved.recipe_id,
        seed=seed,
        decisions=resolved.decisions,
    )
    return StudyPlan(
        "test_campaign",
        Phase.PHASE1,
        (
            StageSpec(
                "test",
                (cell,),
                selection_policy=selection_policy,
            ),
        ),
    )


def focused_blueprint(plan: StudyPlan) -> Phase1Blueprint:
    """Bind a focused Phase-1 unit-test plan to a canonical blueprint."""

    return Phase1Blueprint(
        name=f"{plan.name}_blueprint",
        seeds=tuple(sorted({cell.seed for cell in plan.cells})),
        initial_stages=plan.stages,
        rounds=(),
    )


def register_test_plan(campaign: Campaign, plan: StudyPlan) -> None:
    blueprint = focused_blueprint(plan)
    # Generic journal/state-machine unit tests use a deliberately focused
    # fixture.  Patch only the canonical builders during registration so the
    # shipped registration path remains fail-closed to reduced Phase-1 plans.
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


_PHASE1_DECISION_CACHE: dict[bool, dict[str, object]] = {}


def phase1_decision_for_phase2(
    *,
    smoke: bool,
    forge_nonpromotable_stage: str | None = None,
    capability_failure: tuple[str, str, int] | None = None,
    implementation_identity: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build a complete self-contained decision without file I/O."""

    if (
        forge_nonpromotable_stage is None
        and capability_failure is None
        and implementation_identity is None
        and smoke in _PHASE1_DECISION_CACHE
    ):
        return json.loads(json.dumps(_PHASE1_DECISION_CACHE[smoke]))
    resolved_implementation_identity = (
        TEST_IMPLEMENTATION_IDENTITY
        if implementation_identity is None
        else implementation_identity
    )
    resolved_implementation_identity_sha256 = hashlib.sha256(
        canonical_json(resolved_implementation_identity).encode("utf-8")
    ).hexdigest()
    seeds = (0,) if smoke else (0, 1, 2)
    blueprint = build_phase1_blueprint(seeds=seeds, smoke=smoke)
    plan = build_phase1_plan(seeds=seeds, smoke=smoke)
    stage_order = (
        *(stage.name for stage in blueprint.initial_stages),
        *(round_spec.name for round_spec in blueprint.rounds),
    )
    forge_index = (
        stage_order.index(forge_nonpromotable_stage)
        if forge_nonpromotable_stage is not None
        else None
    )

    def forged_lineage(stage_name: str) -> bool:
        return bool(
            forge_index is not None and stage_order.index(stage_name) >= forge_index
        )

    def qualification_for(cell: CellSpec) -> dict[str, object]:
        intent = cell.decision_map.get("qualification.promotable") is True
        forged_intent = bool(forged_lineage(cell.stage) and not intent)
        negative = cell.decision_map.get("factor.robustness") in {
            "support_only",
            "site_span_one",
        }
        scientific_passed = not negative
        capability_conjunction_failed = bool(
            capability_failure is not None
            and (cell.stage, cell.recipe_name, cell.seed)
            == (
                capability_failure[0],
                f"derived_{capability_failure[0]}_{capability_failure[1]}",
                capability_failure[2],
            )
        )
        identification_conjunction = bool(
            scientific_passed and not capability_conjunction_failed
        )
        identification_margin = (
            -0.25
            if capability_conjunction_failed
            else (1.0 if scientific_passed else -1.0)
        )
        selection_metrics = {
            "validation": {
                "phase1_identification_conjunction": identification_conjunction,
                "phase1_identification_margin": identification_margin,
            }
        }
        inputs = {
            kind: hashlib.sha256(f"{cell.cell_id}:{kind}".encode()).hexdigest()
            for kind in (
                "preparation",
                "checkpoint",
                "calibration",
                "deployment_codec",
                "deployment_schedules",
                "evaluation",
            )
        }
        promotion_eligible = bool(
            not smoke and (intent or forged_intent) and scientific_passed
        )
        protocol_eligible = bool(smoke and (intent or forged_intent))
        identification_inapplicable = (
            cell.decision_map.get("data.normalization") == "layer"
        )
        scientific_identification_passed = bool(
            identification_inapplicable or scientific_passed
        )
        scientific_checks = {
            "support_target_calibration": True,
            "codec_calibration_exclusion": True,
            "codec_evaluation_exclusion": True,
            "phase1_identification": scientific_identification_passed,
            "production_precision_finite": True,
            "production_precision_reconstruction": True,
            "production_precision_support": True,
            "production_fixed_rate_frontier": True,
        }
        reasons = []
        if smoke:
            reasons.append("runtime_smoke")
        if not (intent or forged_intent):
            reasons.append("resolved_nonpromotable_cell")
        if not scientific_identification_passed:
            reasons.append("scientific_outcome_failed")
        return {
            "schema": QUALIFICATION_SCHEMA,
            "cell_id": cell.cell_id,
            "qualified": True,
            "checks": {
                "finite": True,
                "method_endpoints": True,
                "provenance": True,
                "resource_compliance": True,
                "deployment_schedule_integrity": True,
                "encoder_scale_calibration_integrity": True,
                "regularizer_calibration_integrity": True,
                "precision_preflight_integrity": True,
                "selection_score_diagnostics_integrity": True,
                "scientific_endpoint_complete": True,
                "split_integrity": True,
            },
            "scientific_outcome": {
                "passed": scientific_identification_passed,
                "checks": scientific_checks,
                "inapplicable_checks": (
                    {
                        "phase1_identification": (
                            "token_layer_normalization_is_not_a_fixed_linear_factor_map"
                        )
                    }
                    if identification_inapplicable
                    else {}
                ),
                "margins": {
                    "support_target_abs_error": 0.1,
                    "codec_calibration_excluded_fraction": 0.01,
                    "codec_evaluation_excluded_fraction": 0.01,
                    "phase1_native_identification": (
                        -0.25
                        if capability_conjunction_failed
                        else (1.0 if scientific_passed else -1.0)
                    ),
                    "phase1_deployed_identification": (
                        0.5
                        if capability_conjunction_failed
                        else (1.0 if scientific_passed else -1.0)
                    ),
                    "production_precision_reconstruction": None,
                    "production_precision_support_iou": None,
                    "production_fixed_rate_nonzero_endpoints": None,
                },
            },
            "inputs": inputs,
            "validation": selection_metrics["validation"],
            "selection_metrics": selection_metrics,
            "selection_metrics_sha256": hashlib.sha256(
                canonical_json(selection_metrics).encode()
            ).hexdigest(),
            "selection_metrics_evaluation_sha256": inputs["evaluation"],
            "implementation_identity": resolved_implementation_identity,
            "implementation_identity_sha256": (
                resolved_implementation_identity_sha256
            ),
            "qualification_profile": cell.decision_map["qualification.profile"],
            "thresholds_version": cell.decision_map[
                "qualification.thresholds_version"
            ],
            "thresholds": campaign_module._qualification_thresholds(cell),
            "promotion_eligible": promotion_eligible,
            "promotion_ineligible_reasons": (
                [] if promotion_eligible else reasons
            ),
            "selection_eligible_for_protocol_test": protocol_eligible,
            "selection_eligibility_mode": (
                "scientific_promotion"
                if promotion_eligible
                else "smoke_protocol_only"
                if protocol_eligible
                else "none"
            ),
        }

    def qualification_sha256(cell: CellSpec) -> str:
        body = (
            json.dumps(
                qualification_for(cell),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode()
        return "sha256:" + hashlib.sha256(body).hexdigest()

    selection_chain: list[dict[str, object]] = []
    plan_history = [plan]
    while len(plan.stages) < len(blueprint.initial_stages) + len(blueprint.rounds):
        stage = plan.stages[-1]
        policy = stage.selection_policy
        assert policy is not None
        round_spec = next(
            (item for item in blueprint.rounds if item.name == stage.name),
            None,
        )
        by_candidate: dict[str, list[CellSpec]] = {}
        for cell in stage.cells:
            if (
                stage.name != forge_nonpromotable_stage
                and policy.eligible_recipe_names
                and cell.recipe_name not in set(policy.eligible_recipe_names)
            ):
                continue
            if (
                round_spec is not None
                and round_spec.role == "capability_panel"
                and not forged_lineage(stage.name)
                and cell.decision_map.get("qualification.promotable") is not True
            ):
                continue
            by_candidate.setdefault(cell.candidate_id, []).append(cell)
        assert by_candidate
        if stage.name == forge_nonpromotable_stage:
            candidate_id = sorted(
                candidate_id
                for candidate_id, cells in by_candidate.items()
                if all(
                    cell.decision_map.get("qualification.promotable") is not True
                    for cell in cells
                )
            )[0]
        elif round_spec is not None and round_spec.role == "capability_panel":
            carrier_recipe = (
                f"derived_{round_spec.name}_{round_spec.fixed_carrier_variant}"
            )
            candidate_id = next(
                candidate_id
                for candidate_id, cells in by_candidate.items()
                if {cell.recipe_name for cell in cells} == {carrier_recipe}
            )
        else:
            candidate_id = sorted(by_candidate)[0]
        selected_cells = tuple(
            sorted(by_candidate[candidate_id], key=lambda cell: cell.seed)
        )
        if (
            round_spec is not None
            and round_spec.role == "capability_panel"
            and round_spec.name != forge_nonpromotable_stage
        ):
            assert round_spec.fixed_carrier_variant is not None
            assert {cell.recipe_name for cell in selected_cells} == {
                f"derived_{round_spec.name}_{round_spec.fixed_carrier_variant}"
            }
        qualification_hashes = tuple(
            qualification_sha256(cell) for cell in selected_cells
        )
        normalized_evidence = {
            cell.cell_id: {
                "state": "qualified",
                "qualification": qualification_for(cell),
                "qualification_sha256": qualification_sha256(cell),
            }
            for cell in stage.cells
        }
        ranked_candidates, excluded_candidates, _, _ = (
            Campaign._selection_universe_from_evidence(
                stage.name,
                stage.cells,
                policy,
                normalized_evidence,
                sharing_guard_for_cell=lambda *_args: pytest.fail(
                    "Phase-1 fixture unexpectedly requested a sharing guard"
                ),
            )
        )
        universe_sha256 = (
            "sha256:"
            + hashlib.sha256(
                canonical_json(
                    {
                        "plan_id": plan.plan_id,
                        "source_stage": stage.name,
                        "policy_id": policy.policy_id,
                        "ranked_candidates": ranked_candidates,
                        "excluded_candidates": excluded_candidates,
                    }
                ).encode()
            ).hexdigest()
        )
        ranked_candidate = next(
            (
                item
                for item in ranked_candidates
                if item["candidate_id"] == candidate_id
            ),
            None,
        )
        metric_values = (
            [float(item["metric"]) for item in ranked_candidate["observations"]]
            if ranked_candidate is not None
            else [0.0 if smoke else 1.0 for _ in selected_cells]
        )
        selection = FrozenSelection.from_cells(
            policy,
            selected_cells,
            metric_values,
            qualification_hashes,
            universe_sha256,
        )
        extended = materialize_child_plan(plan, blueprint, selection)
        selection_chain.append(
            {
                "source_plan_id": plan.plan_id,
                "source_stage": stage.name,
                "target_plan_id": extended.plan_id,
                "target_stage": extended.stages[-1].name,
                "policy_id": policy.policy_id,
                "selection_id": selection.selection_id,
                "selection_universe_sha256": universe_sha256,
                "selection_artifact_sha256": "sha256:" + "1" * 64,
                "selection_artifact_sha256_semantics": (
                    "opaque_historical_commitment_requires_trusted_origin"
                ),
                "selection": selection.to_dict(),
            }
        )
        plan = extended
        plan_history.append(plan)

    cells: list[dict[str, object]] = []
    by_variant: dict[str, list[CellSpec]] = {}
    for cell in plan.cells:
        qualification = qualification_for(cell)
        evidence = {
            "cell_id": cell.cell_id,
            "candidate_id": cell.candidate_id,
            "stage": cell.stage,
            "seed": cell.seed,
            "recipe_name": cell.recipe_name,
            "recipe_id": cell.recipe_id,
            "state": "qualified",
            "qualification_sha256": qualification_sha256(cell),
            "qualification": qualification,
        }
        cells.append(evidence)
        if cell.stage == plan.stages[-1].name:
            variant = str(cell.decision_map["factor.robustness"])
            by_variant.setdefault(variant, []).append(cell)
    evidence_by_id = {str(item["cell_id"]): item for item in cells}
    results = []
    for variant, variant_cells in sorted(by_variant.items()):
        per_seed = []
        for cell in sorted(variant_cells, key=lambda item: item.seed):
            evidence = evidence_by_id[cell.cell_id]
            per_seed.append(
                {
                    "seed": cell.seed,
                    "cell_id": cell.cell_id,
                    "qualification_sha256": evidence["qualification_sha256"],
                    **Campaign._phase1_claim_evidence(
                        evidence["qualification"], smoke=smoke
                    ),
                }
            )
        results.append(
            {
                "variant": variant,
                "candidate_id": variant_cells[0].candidate_id,
                "required_baseline": variant == "baseline",
                "negative_control": variant in {"support_only", "site_span_one"},
                "negative_control_passed": (
                    None
                    if smoke or variant not in {"support_only", "site_span_one"}
                    else all(item["conjunction_passed"] is False for item in per_seed)
                ),
                "passed_all_seeds": all(
                    item["conjunction_passed"] is True for item in per_seed
                ),
                "per_seed": per_seed,
            }
        )
    manifest = {
        "schema": campaign_module.PHASE1_CAMPAIGN_MANIFEST_SCHEMA,
        "source_phase1_plan_id": plan.plan_id,
        "source_phase1_blueprint_id": blueprint.blueprint_id,
        "plan_content_sha256": "sha256:"
        + hashlib.sha256(canonical_json(plan.to_manifest()).encode()).hexdigest(),
        "blueprint_content_sha256": "sha256:"
        + hashlib.sha256(canonical_json(blueprint.to_manifest()).encode()).hexdigest(),
        "plan_sha256": campaign_module._run_cell_json_sha256(plan.to_manifest()),
        "blueprint_sha256": campaign_module._run_cell_json_sha256(
            blueprint.to_manifest()
        ),
        "journal_sha256": "sha256:" + "4" * 64,
        "journal_sha256_semantics": (
            "opaque_historical_commitment_requires_trusted_origin"
        ),
        "smoke": smoke,
        "plan": plan.to_manifest(),
        "blueprint": blueprint.to_manifest(),
        "plan_history": [
            {
                "plan_id": historical.plan_id,
                "sha256": campaign_module._run_cell_json_sha256(
                    historical.to_manifest()
                ),
            }
            for historical in plan_history
        ],
        "selection_chain": selection_chain,
        "cells": cells,
        "confirmation": {
            "results": results,
            "stress_failures": [],
            "scope_narrowing": {},
        },
    }
    body = {
        "schema": campaign_module.PHASE1_DECISION_SCHEMA,
        "source_phase1_plan_id": plan.plan_id,
        "source_phase1_blueprint_id": blueprint.blueprint_id,
        "authorization_mode": ("smoke_protocol_only" if smoke else "scientific_go"),
        "decision": "protocol_complete" if smoke else "go",
        "authorizes_phase2_scientific": not smoke,
        "authorizes_phase2_smoke": True,
        "phase1_campaign_manifest_sha256": "sha256:"
        + hashlib.sha256(canonical_json(manifest).encode()).hexdigest(),
        "phase1_campaign_manifest": manifest,
        "phase1_transfer": build_phase1_transfer(manifest),
    }
    payload = {**body, "decision_id": content_id(body, prefix="phase1-decision")}
    if forge_nonpromotable_stage is None:
        Campaign.phase1_decision_from_manifest(payload)
    if (
        forge_nonpromotable_stage is None
        and capability_failure is None
        and implementation_identity is None
    ):
        _PHASE1_DECISION_CACHE[smoke] = payload
    return json.loads(json.dumps(payload))


def phase2_test_inputs(
    *, seeds: tuple[int, ...] | None = None, smoke: bool
) -> tuple[StudyPlan, object, dict[str, object]]:
    if seeds is None:
        seeds = (0,) if smoke else (0, 1)
    phase1_decision = phase1_decision_for_phase2(smoke=smoke)
    blueprint = build_phase2_blueprint(
        seeds=seeds,
        smoke=smoke,
        phase1_decision=phase1_decision,
    )
    plan = build_phase2_plan(
        seeds=seeds,
        smoke=smoke,
        phase1_decision=phase1_decision,
    )
    return plan, blueprint, phase1_decision


def register_phase2_test_plan(
    campaign: Campaign,
    plan: StudyPlan,
    blueprint: object,
    phase1_decision: dict[str, object],
) -> None:
    campaign.register(
        plan,
        blueprint_manifest=blueprint.to_manifest(),
        phase1_decision_manifest=phase1_decision,
    )


def qualification_file_sha256(payload: dict[str, object]) -> str:
    body = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    return "sha256:" + hashlib.sha256(body).hexdigest()


def rehash_phase1_decision(
    payload: dict[str, object], *, rebuild_transfer: bool = True
) -> dict[str, object]:
    manifest = payload["phase1_campaign_manifest"]
    payload["phase1_campaign_manifest_sha256"] = (
        "sha256:" + hashlib.sha256(canonical_json(manifest).encode()).hexdigest()
    )
    if rebuild_transfer:
        payload["phase1_transfer"] = build_phase1_transfer(manifest)
    body = dict(payload)
    body.pop("decision_id", None)
    payload["decision_id"] = content_id(body, prefix="phase1-decision")
    return payload


def rehash_panel_decision(payload: dict[str, object]) -> dict[str, object]:
    payload["phase2_campaign_manifest_sha256"] = (
        "sha256:"
        + hashlib.sha256(
            canonical_json(payload["phase2_campaign_manifest"]).encode()
        ).hexdigest()
    )
    payload["selection_universe_sha256"] = (
        "sha256:"
        + hashlib.sha256(
            canonical_json(payload["selection_universe"]).encode()
        ).hexdigest()
    )
    body = {
        key: payload[key]
        for key in (
            "schema",
            "source_phase2_plan_id",
            "source_phase2_blueprint_id",
            "phase2_campaign_manifest_sha256",
            "selection_universe_sha256",
            "entries",
        )
    }
    payload["panel_id"] = content_id(body, prefix="panel-selection")
    return payload


def unreferenced_phase1_cell_evidence(
    payload: dict[str, object],
) -> dict[str, object]:
    manifest = payload["phase1_campaign_manifest"]
    referenced_cell_ids = {
        cell_id
        for item in manifest["selection_chain"]
        for cell_id in item["selection"]["cell_ids"]
    }
    referenced_cell_ids.update(
        row["cell_id"]
        for result in manifest["confirmation"]["results"]
        for row in result["per_seed"]
    )
    return next(
        item
        for item in manifest["cells"]
        if item["cell_id"] not in referenced_cell_ids
    )


def set_phase1_variant_outcome(
    payload: dict[str, object], variant: str, *, passed: bool
) -> None:
    manifest = payload["phase1_campaign_manifest"]
    plan = StudyPlan.from_manifest(manifest["plan"])
    variant_cells = {
        cell.cell_id
        for cell in plan.stages[-1].cells
        if cell.decision_map.get("factor.robustness") == variant
    }
    evidence_by_id = {item["cell_id"]: item for item in manifest["cells"]}
    for cell_id in variant_cells:
        evidence = evidence_by_id[cell_id]
        qualification = evidence["qualification"]
        qualification["scientific_outcome"]["passed"] = passed
        qualification["scientific_outcome"]["checks"]["phase1_identification"] = passed
        qualification["scientific_outcome"]["margins"].update(
            {
                "phase1_native_identification": 1.0 if passed else -1.0,
                "phase1_deployed_identification": 1.0 if passed else -1.0,
            }
        )
        qualification["selection_metrics"]["validation"] = {
            "phase1_identification_conjunction": passed,
            "phase1_identification_margin": 1.0 if passed else -1.0,
        }
        qualification["validation"] = qualification["selection_metrics"]["validation"]
        qualification["selection_metrics_sha256"] = hashlib.sha256(
            canonical_json(qualification["selection_metrics"]).encode()
        ).hexdigest()
        qualification["promotion_eligible"] = passed
        qualification["promotion_ineligible_reasons"] = (
            [] if passed else ["scientific_outcome_failed"]
        )
        qualification["selection_eligibility_mode"] = (
            "scientific_promotion" if passed else "none"
        )
        evidence["qualification_sha256"] = qualification_file_sha256(qualification)
    result = next(
        item
        for item in manifest["confirmation"]["results"]
        if item["variant"] == variant
    )
    for row in result["per_seed"]:
        evidence = evidence_by_id[row["cell_id"]]
        row["qualification_sha256"] = evidence["qualification_sha256"]
        row.update(
            Campaign._phase1_claim_evidence(evidence["qualification"], smoke=False)
        )
    result["passed_all_seeds"] = all(
        row["conjunction_passed"] is True for row in result["per_seed"]
    )
    if result["negative_control"]:
        result["negative_control_passed"] = all(
            row["conjunction_passed"] is False for row in result["per_seed"]
        )


def write_artifact(root: Path, kind: str, payload: object) -> ArtifactRef:
    path = root / "outputs" / f"{kind}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return ArtifactRef.from_path(kind, path, root=root)


def evaluation_payload(
    cell_id: str,
    inputs: dict[str, str],
    validation: dict[str, float],
    *,
    sharing_guard: dict[str, float] | None = None,
) -> dict[str, object]:
    selection_metrics = {
        "validation": validation,
        "sharing_guard": sharing_guard
        or {
            "all_site_fvu_mean": 0.10,
            "site_only_heldout_fvu_mean": 0.30,
            "leave_one_out_heldout_fvu_mean": 0.25,
            "site_only_support_iou_mean": 0.80,
            "leave_one_out_support_iou_mean": 0.75,
            "site_only_coordinate_concordance_mean": 0.90,
            "leave_one_out_coordinate_concordance_mean": 0.85,
            "site_only_coordinate_concordance_min": 0.90,
            "leave_one_out_coordinate_concordance_min": 0.85,
            "site_only_intersection_recall_mean": 0.80,
            "leave_one_out_intersection_recall_mean": 0.75,
            "site_only_intersection_recall_min": 0.80,
            "leave_one_out_intersection_recall_min": 0.75,
            "site_only_intersection_energy_coverage_mean": 0.95,
            "leave_one_out_intersection_energy_coverage_mean": 0.92,
            "site_only_intersection_energy_coverage_min": 0.95,
            "leave_one_out_intersection_energy_coverage_min": 0.92,
        },
    }
    return {
        "schema": EVALUATION_SCHEMA,
        "evaluation_execution_implementation": (
            "fused_deployable_full_view_packet_v2"
        ),
        "cell_id": cell_id,
        "inputs": inputs,
        "validation": validation,
        "selection_metrics": selection_metrics,
        "selection_metrics_sha256": hashlib.sha256(
            canonical_json(selection_metrics).encode("utf-8")
        ).hexdigest(),
        "raw_space": {"eligible": True},
        "fixed_rate_raw_selection": {"eligible": True},
        "synthetic_recovery": {
            "deployed": {"shared_feature_claim_eligible": True}
        },
    }


def advance_to_evaluated(campaign: Campaign, cell_id: str) -> dict[str, ArtifactRef]:
    preparation = write_artifact(
        campaign.root,
        "preparation",
        {
            "schema": PREPARATION_SCHEMA,
            "cell_id": cell_id,
            "ready": True,
            "implementation": TEST_IMPLEMENTATION_IDENTITY,
            "implementation_sha256": TEST_IMPLEMENTATION_IDENTITY_SHA256,
        },
    )
    prepare_manifest = write_artifact(
        campaign.root, "prepare_manifest", {"stage": "prepare"}
    )
    campaign.transition(
        cell_id,
        RunState.PREPARED,
        artifacts=(preparation, prepare_manifest),
    )
    campaign.transition(cell_id, RunState.RUNNING)
    checkpoint = write_artifact(campaign.root, "checkpoint", {"weights": [1, 2]})
    training_report = write_artifact(
        campaign.root, "training_report", {"attempted_tokens": 64}
    )
    train_manifest = write_artifact(campaign.root, "train_manifest", {"stage": "train"})
    campaign.transition(
        cell_id,
        RunState.TRAINED,
        artifacts=(checkpoint, training_report, train_manifest),
    )
    calibration = write_artifact(campaign.root, "calibration", {"threshold": 0.2})
    deployment_codec = write_artifact(
        campaign.root, "deployment_codec", {"model": "deployable"}
    )
    calibration_record = write_artifact(
        campaign.root, "calibration_record", {"threshold": 0.2}
    )
    calibrate_manifest = write_artifact(
        campaign.root, "calibrate_manifest", {"stage": "calibrate"}
    )
    campaign.transition(
        cell_id,
        RunState.CALIBRATED,
        artifacts=(
            calibration,
            deployment_codec,
            calibration_record,
            calibrate_manifest,
        ),
    )
    deployment_schedules = write_artifact(
        campaign.root,
        "deployment_schedules",
        {"schema": "test-deployment-schedules-v1"},
    )
    validation = {"fvu": 0.1, "rate_distortion": 0.8}
    evaluation = write_artifact(
        campaign.root,
        "evaluation",
        evaluation_payload(
            cell_id,
            {
                "checkpoint": checkpoint.sha256,
                "calibration": calibration.sha256,
                "deployment_codec": deployment_codec.sha256,
                "deployment_schedules": deployment_schedules.sha256,
            },
            validation,
        ),
    )
    evaluate_manifest = write_artifact(
        campaign.root, "evaluate_manifest", {"stage": "evaluate"}
    )
    campaign.transition(
        cell_id,
        RunState.EVALUATED,
        artifacts=(deployment_schedules, evaluation, evaluate_manifest),
    )
    return {
        "preparation": preparation,
        "checkpoint": checkpoint,
        "calibration": calibration,
        "deployment_codec": deployment_codec,
        "deployment_schedules": deployment_schedules,
        "evaluation": evaluation,
    }


def qualification_payload(
    cell: CellSpec,
    inputs: dict[str, str],
    *,
    scientific_passed: bool = True,
    validation: dict[str, float] | None = None,
    sharing_guard: dict[str, float] | None = None,
    promotion_eligible: bool | None = None,
    protocol_eligible: bool = False,
) -> dict[str, object]:
    cell_id = cell.cell_id
    validation = validation or {"fvu": 0.1, "rate_distortion": 0.8}
    resolved_promotion = (
        scientific_passed
        if promotion_eligible is None
        else scientific_passed and promotion_eligible
    )
    selection_metrics = {
        "validation": validation,
        "sharing_guard": sharing_guard
        or {
            "all_site_fvu_mean": 0.10,
            "site_only_heldout_fvu_mean": 0.30,
            "leave_one_out_heldout_fvu_mean": 0.25,
            "site_only_support_iou_mean": 0.80,
            "leave_one_out_support_iou_mean": 0.75,
            "site_only_coordinate_concordance_mean": 0.90,
            "leave_one_out_coordinate_concordance_mean": 0.85,
            "site_only_coordinate_concordance_min": 0.90,
            "leave_one_out_coordinate_concordance_min": 0.85,
            "site_only_intersection_recall_mean": 0.80,
            "leave_one_out_intersection_recall_mean": 0.75,
            "site_only_intersection_recall_min": 0.80,
            "leave_one_out_intersection_recall_min": 0.75,
            "site_only_intersection_energy_coverage_mean": 0.95,
            "leave_one_out_intersection_energy_coverage_mean": 0.92,
            "site_only_intersection_energy_coverage_min": 0.95,
            "leave_one_out_intersection_energy_coverage_min": 0.92,
        },
    }
    identification_inapplicable = (
        cell.phase is Phase.PHASE1
        and cell.decision_map.get("data.normalization") == "layer"
    )
    scientific_checks = {
        "support_target_calibration": scientific_passed,
        "codec_calibration_exclusion": scientific_passed,
        "codec_evaluation_exclusion": scientific_passed,
        "phase1_identification": (
            True if identification_inapplicable else scientific_passed
        ),
        "production_precision_finite": scientific_passed,
        "production_precision_reconstruction": scientific_passed,
        "production_precision_support": scientific_passed,
        "production_fixed_rate_frontier": scientific_passed,
    }
    reasons = []
    if cell.decision_map["runtime.smoke"] is True:
        reasons.append("runtime_smoke")
    if cell.decision_map["qualification.promotable"] is not True:
        reasons.append("resolved_nonpromotable_cell")
    if not scientific_passed:
        reasons.append("scientific_outcome_failed")
    return {
        "schema": QUALIFICATION_SCHEMA,
        "cell_id": cell_id,
        "qualified": True,
        "checks": {
            "finite": True,
            "method_endpoints": True,
            "provenance": True,
            "resource_compliance": True,
            "deployment_schedule_integrity": True,
            "encoder_scale_calibration_integrity": True,
            "regularizer_calibration_integrity": True,
            "precision_preflight_integrity": True,
            "selection_score_diagnostics_integrity": True,
            "scientific_endpoint_complete": True,
            "split_integrity": True,
        },
        "scientific_outcome": {
            "passed": scientific_passed,
            "checks": scientific_checks,
            "inapplicable_checks": (
                {
                    "phase1_identification": (
                        "token_layer_normalization_is_not_a_fixed_linear_factor_map"
                    )
                }
                if identification_inapplicable
                else {}
            ),
            "margins": {
                "support_target_abs_error": 0.1,
                "codec_calibration_excluded_fraction": 0.01,
                "codec_evaluation_excluded_fraction": 0.01,
                "phase1_native_identification": (
                    0.1 if scientific_passed else -0.1
                ),
                "phase1_deployed_identification": (
                    0.1 if scientific_passed else -0.1
                ),
                "production_precision_reconstruction": None,
                "production_precision_support_iou": None,
                "production_fixed_rate_nonzero_endpoints": None,
            },
        },
        "inputs": inputs,
        "validation": validation,
        "selection_metrics": selection_metrics,
        "selection_metrics_sha256": hashlib.sha256(
            canonical_json(selection_metrics).encode("utf-8")
        ).hexdigest(),
        "selection_metrics_evaluation_sha256": inputs["evaluation"],
        "implementation_identity": TEST_IMPLEMENTATION_IDENTITY,
        "implementation_identity_sha256": TEST_IMPLEMENTATION_IDENTITY_SHA256,
        "qualification_profile": cell.decision_map["qualification.profile"],
        "thresholds_version": cell.decision_map["qualification.thresholds_version"],
        "thresholds": campaign_module._qualification_thresholds(cell),
        "promotion_eligible": resolved_promotion,
        "promotion_ineligible_reasons": (
            [] if resolved_promotion else reasons
        ),
        "selection_eligible_for_protocol_test": protocol_eligible,
        "selection_eligibility_mode": (
            "scientific_promotion"
            if resolved_promotion
            else "smoke_protocol_only"
            if protocol_eligible
            else "none"
        ),
    }


def good_qualification(
    campaign: Campaign,
    cell_id: str,
    inputs: dict[str, ArtifactRef],
) -> ArtifactRef:
    cell = campaign._require_cell(cell_id)
    smoke = cell.decision_map.get("runtime.smoke") is True
    return write_artifact(
        campaign.root,
        "qualification",
        qualification_payload(
            cell,
            {kind: ref.sha256 for kind, ref in inputs.items()},
            promotion_eligible=not smoke,
            protocol_eligible=(
                smoke and cell.decision_map.get("qualification.promotable") is True
            ),
        ),
    )


def write_cell_artifact(
    campaign: Campaign,
    cell_id: str,
    kind: str,
    payload: object,
) -> ArtifactRef:
    path = campaign.cell_dir(cell_id) / "unit-artifacts" / f"{kind}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return ArtifactRef.from_path(kind, path, root=campaign.root)


def qualify_cell(
    campaign: Campaign,
    cell_id: str,
    *,
    metric: float,
    scientific_passed: bool = True,
    validation: dict[str, float] | None = None,
    sharing_guard: dict[str, float] | None = None,
) -> ArtifactRef:
    preparation = write_cell_artifact(
        campaign,
        cell_id,
        "preparation",
        {
            "schema": PREPARATION_SCHEMA,
            "cell_id": cell_id,
            "ready": True,
            "implementation": TEST_IMPLEMENTATION_IDENTITY,
            "implementation_sha256": TEST_IMPLEMENTATION_IDENTITY_SHA256,
        },
    )
    prepare_manifest = write_cell_artifact(
        campaign, cell_id, "prepare_manifest", {"stage": "prepare"}
    )
    campaign.transition(
        cell_id, RunState.PREPARED, artifacts=(preparation, prepare_manifest)
    )
    campaign.transition(cell_id, RunState.RUNNING)
    checkpoint = write_cell_artifact(campaign, cell_id, "checkpoint", {"weights": []})
    training_report = write_cell_artifact(
        campaign, cell_id, "training_report", {"tokens": 64}
    )
    train_manifest = write_cell_artifact(
        campaign, cell_id, "train_manifest", {"stage": "train"}
    )
    campaign.transition(
        cell_id,
        RunState.TRAINED,
        artifacts=(checkpoint, training_report, train_manifest),
    )
    calibration = write_cell_artifact(campaign, cell_id, "calibration", {"k": 1})
    deployment_codec = write_cell_artifact(
        campaign, cell_id, "deployment_codec", {"model": "deployable"}
    )
    calibration_record = write_cell_artifact(
        campaign, cell_id, "calibration_record", {"k": 1}
    )
    calibrate_manifest = write_cell_artifact(
        campaign, cell_id, "calibrate_manifest", {"stage": "calibrate"}
    )
    campaign.transition(
        cell_id,
        RunState.CALIBRATED,
        artifacts=(
            calibration,
            deployment_codec,
            calibration_record,
            calibrate_manifest,
        ),
    )
    deployment_schedules = write_cell_artifact(
        campaign,
        cell_id,
        "deployment_schedules",
        {"schema": "test-deployment-schedules-v1"},
    )
    resolved_validation = (
        {"phase1_identification_margin": metric} if validation is None else validation
    )
    evaluation = write_cell_artifact(
        campaign,
        cell_id,
        "evaluation",
        evaluation_payload(
            cell_id,
            {
                "checkpoint": checkpoint.sha256,
                "calibration": calibration.sha256,
                "deployment_codec": deployment_codec.sha256,
                "deployment_schedules": deployment_schedules.sha256,
            },
            resolved_validation,
            sharing_guard=sharing_guard,
        ),
    )
    evaluate_manifest = write_cell_artifact(
        campaign, cell_id, "evaluate_manifest", {"stage": "evaluate"}
    )
    campaign.transition(
        cell_id,
        RunState.EVALUATED,
        artifacts=(deployment_schedules, evaluation, evaluate_manifest),
    )
    qualification = write_cell_artifact(
        campaign,
        cell_id,
        "qualification",
        qualification_payload(
            campaign._require_cell(cell_id),
            {
                "preparation": preparation.sha256,
                "checkpoint": checkpoint.sha256,
                "calibration": calibration.sha256,
                "deployment_codec": deployment_codec.sha256,
                "deployment_schedules": deployment_schedules.sha256,
                "evaluation": evaluation.sha256,
            },
            scientific_passed=scientific_passed,
            validation=resolved_validation,
            sharing_guard=sharing_guard,
            promotion_eligible=(
                campaign._require_cell(cell_id).decision_map.get("runtime.smoke")
                is not True
                and campaign._require_cell(cell_id).decision_map.get(
                    "qualification.promotable"
                )
                is True
            ),
            protocol_eligible=(
                campaign._require_cell(cell_id).decision_map.get("runtime.smoke")
                is True
                and campaign._require_cell(cell_id).decision_map.get(
                    "qualification.promotable"
                )
                is True
            ),
        ),
    )
    qualify_manifest = write_cell_artifact(
        campaign, cell_id, "qualify_manifest", {"stage": "qualify"}
    )
    campaign.transition(
        cell_id,
        RunState.QUALIFIED,
        artifacts=(qualification, qualify_manifest),
    )
    return qualification


def qualified_phase2_campaign(
    root: Path,
    *,
    seeds: tuple[int, ...] = (0,),
) -> tuple[Campaign, object]:
    """Materialize every smoke round and qualify its complete declared panel."""

    plan, blueprint, phase1_decision = phase2_test_inputs(
        seeds=seeds,
        smoke=True,
    )
    campaign = Campaign(root)
    register_phase2_test_plan(campaign, plan, blueprint, phase1_decision)
    while True:
        stage = campaign.plan.stages[-1]
        for index, cell in enumerate(stage.cells):
            qualify_cell(
                campaign,
                cell.cell_id,
                metric=float(index + 1),
                validation={"negative_mean_raw_fvu": float(index + 1)},
            )
        if stage.selection_policy is None:
            break
        selection_path = root / "selections" / f"{stage.name}.json"
        payload = campaign.select_stage(stage.name, out=selection_path)
        selection = FrozenSelection.from_dict(payload["selected"][0])
        extended = materialize_child_plan(campaign.plan, blueprint, selection)
        campaign.extend(
            extended,
            selection=selection,
            selection_path=selection_path,
        )
    for family in blueprint.comparator_families:
        root_selection_path = root / "selections" / f"family_{family.name}_root.json"
        root_payload = campaign.select_family_root(family.name, out=root_selection_path)
        root_selection = FrozenSelection.from_dict(root_payload["selected"][0])
        extended = materialize_family_child_plan(
            campaign.plan, blueprint, family.name, root_selection
        )
        campaign.extend_family(
            extended,
            family_name=family.name,
            selection=root_selection,
            selection_path=root_selection_path,
        )
        while True:
            stage = campaign.plan.stages[-1]
            for index, cell in enumerate(stage.cells):
                qualify_cell(
                    campaign,
                    cell.cell_id,
                    metric=float(index + 1),
                    validation={"negative_mean_raw_fvu": float(index + 1)},
                )
            if stage.name == family.rounds[-1].name:
                nomination_path = (
                    root
                    / "selections"
                    / f"family_{family.name}_revisit_nomination.json"
                )
                nomination = campaign.select_family_revisit_inputs(
                    family.name, out=nomination_path
                )
                selections = tuple(
                    FrozenSelection.from_dict(item) for item in nomination["selected"]
                )
                extended = materialize_family_revisit_plan(
                    campaign.plan, blueprint, family.name, selections
                )
                campaign.extend_family_revisit(
                    extended,
                    family_name=family.name,
                    selection_path=nomination_path,
                )
                revisit_stage = campaign.plan.stages[-1]
                for index, cell in enumerate(revisit_stage.cells):
                    qualify_cell(
                        campaign,
                        cell.cell_id,
                        metric=float(index + 1),
                        validation={"negative_mean_raw_fvu": float(index + 1)},
                    )
                campaign.select_stage(
                    revisit_stage.name,
                    out=(root / "selections" / f"family_{family.name}_final.json"),
                )
                break
            selection_path = root / "selections" / f"{stage.name}.json"
            payload = campaign.select_stage(stage.name, out=selection_path)
            selection = FrozenSelection.from_dict(payload["selected"][0])
            extended = materialize_family_child_plan(
                campaign.plan, blueprint, family.name, selection
            )
            campaign.extend_family(
                extended,
                family_name=family.name,
                selection=selection,
                selection_path=selection_path,
            )
    return campaign, blueprint


def test_phase1_registration_rejects_a_reduced_custom_blueprint(tmp_path):
    plan = one_cell_plan()
    with pytest.raises(CampaignError, match="exact canonical plan/blueprint"):
        Campaign(tmp_path).register(
            plan,
            blueprint_manifest=focused_blueprint(plan).to_manifest(),
        )


def test_registration_is_idempotent_and_journal_is_authoritative(tmp_path):
    plan = one_cell_plan()
    campaign = Campaign(tmp_path)
    with pytest.raises(CampaignError, match="requires an exact frozen blueprint"):
        campaign.register(plan)
    register_test_plan(campaign, plan)
    first_lines = campaign.journal_path.read_text().splitlines()
    register_test_plan(campaign, plan)
    assert campaign.journal_path.read_text().splitlines() == first_lines
    assert campaign.status()["counts"] == {"planned": 1}

    # Snapshots are caches.  Removing one and reconciling does not alter the
    # append-only transition history.
    campaign.state_path(plan.cells[0].cell_id).unlink()
    result = campaign.reconcile()
    assert result == {"stale_locks": [], "snapshots_rebuilt": 1}
    assert campaign.state_path(plan.cells[0].cell_id).is_file()
    assert campaign.record(plan.cells[0].cell_id).state is RunState.PLANNED

    other = one_cell_plan(seed=1)
    with pytest.raises(CampaignError, match="different frozen blueprint"):
        register_test_plan(campaign, other)


def test_phase2_registration_requires_authenticated_phase1_go(tmp_path):
    preview_blueprint = build_phase2_blueprint(smoke=False)
    preview_plan = build_phase2_plan(smoke=False)
    with pytest.raises(CampaignError, match="Phase-1 go/no-go"):
        Campaign(tmp_path / "missing").register(
            preview_plan,
            blueprint_manifest=preview_blueprint.to_manifest(),
        )
    smoke_decision = phase1_decision_for_phase2(smoke=True)
    escalation_blueprint = build_phase2_blueprint(
        seeds=(0, 1),
        smoke=False,
        phase1_decision=smoke_decision,
    )
    escalation_plan = build_phase2_plan(
        seeds=(0, 1),
        smoke=False,
        phase1_decision=smoke_decision,
    )
    with pytest.raises(CampaignError, match="protocol-only"):
        Campaign(tmp_path / "escalation").register(
            escalation_plan,
            blueprint_manifest=escalation_blueprint.to_manifest(),
            phase1_decision_manifest=smoke_decision,
        )
    plan, blueprint, phase1_decision = phase2_test_inputs(smoke=False)
    Campaign(tmp_path / "authorized").register(
        plan,
        blueprint_manifest=blueprint.to_manifest(),
        phase1_decision_manifest=phase1_decision,
    )


def test_phase2_registration_rejects_unbound_preview(tmp_path):
    blueprint = build_phase2_blueprint(smoke=False)
    plan = build_phase2_plan(smoke=False)
    with pytest.raises(CampaignError, match="exactly inherit"):
        Campaign(tmp_path).register(
            plan,
            blueprint_manifest=blueprint.to_manifest(),
            phase1_decision_manifest=phase1_decision_for_phase2(smoke=False),
        )


def test_phase2_bound_plan_carries_exact_phase1_ids(tmp_path):
    plan, blueprint, phase1_decision = phase2_test_inputs(smoke=False)
    decision_id = phase1_decision["decision_id"]
    transfer_id = phase1_decision["phase1_transfer"]["transfer_id"]
    assert blueprint.source_phase1_decision_id == decision_id
    assert blueprint.phase1_transfer_id == transfer_id
    inherited = [
        cell
        for cell in plan.cells
        if "provenance.phase1_decision_id" in cell.decision_map
    ]
    assert {cell.recipe_name for cell in inherited} == {
        "phase1_contract_bsc",
        "phase1_contract_source_only_control",
    }
    assert {
        cell.decision_map["provenance.phase1_decision_id"] for cell in inherited
    } == {decision_id}
    assert {
        cell.decision_map["provenance.phase1_transfer_id"] for cell in inherited
    } == {transfer_id}
    register_phase2_test_plan(
        Campaign(tmp_path),
        plan,
        blueprint,
        phase1_decision,
    )


def test_phase1_transfer_reopens_activation_operator_but_keeps_signed_ontology():
    transfer = phase1_decision_for_phase2(smoke=False)["phase1_transfer"]
    contract_names = {item["name"] for item in transfer["method_contract"]}
    carriers = {
        item["name"]: item["value"] for item in transfer["provisional_carriers"]
    }
    assert "model.activation" not in contract_names
    assert carriers == {
        "model.activation": "signed",
        "model.selection_score": "decoded_energy",
    }
    assert transfer["phase2_reopened_decisions"] == [
        "model.activation",
        "model.selection_score",
    ]
    assert transfer["method_semantics"]["feature_ontology"] == (
        "shared_signed_coordinate_vector"
    )


def test_phase1_decision_rejects_transfer_and_decision_id_tampering():
    transfer_tamper = phase1_decision_for_phase2(smoke=False)
    transfer_tamper["phase1_transfer"]["method_semantics"]["feature_ontology"] = (
        "forged_scalar_feature"
    )
    rehash_phase1_decision(transfer_tamper, rebuild_transfer=False)
    with pytest.raises(CampaignError, match="transfer contract differs"):
        Campaign.phase1_decision_from_manifest(transfer_tamper)

    decision_tamper = phase1_decision_for_phase2(smoke=False)
    decision_tamper["decision_id"] = "phase1-decision:" + "f" * 64
    with pytest.raises(CampaignError, match="decision content ID mismatch"):
        Campaign.phase1_decision_from_manifest(decision_tamper)


def test_phase1_decision_rejects_self_consistent_reduced_blueprint():
    payload = phase1_decision_for_phase2(smoke=True)
    manifest = payload["phase1_campaign_manifest"]
    blueprint = Phase1Blueprint.from_manifest(manifest["blueprint"])
    reduced = replace(blueprint, rounds=blueprint.rounds[:-1])
    manifest["blueprint"] = reduced.to_manifest()
    manifest["source_phase1_blueprint_id"] = reduced.blueprint_id
    manifest["blueprint_content_sha256"] = (
        "sha256:"
        + hashlib.sha256(canonical_json(reduced.to_manifest()).encode()).hexdigest()
    )
    manifest["blueprint_sha256"] = campaign_module._run_cell_json_sha256(
        reduced.to_manifest()
    )
    payload["source_phase1_blueprint_id"] = reduced.blueprint_id
    rehash_phase1_decision(payload, rebuild_transfer=False)
    with pytest.raises(CampaignError, match="exact canonical Phase-1 blueprint"):
        Campaign.phase1_decision_from_manifest(payload)


def test_phase2_registration_rejects_tampered_transferred_root(tmp_path):
    plan, blueprint, phase1_decision = phase2_test_inputs(smoke=False)
    forged_decision_id = "phase1-decision:" + "f" * 64
    root = next(
        cell for cell in plan.cells if cell.recipe_name == "phase1_contract_bsc"
    )
    tampered_root = replace(
        root,
        decisions=tuple(
            replace(decision, value=forged_decision_id)
            if decision.name == "provenance.phase1_decision_id"
            else decision
            for decision in root.decisions
        ),
    )
    tampered_stage = replace(
        plan.stages[0],
        cells=tuple(
            tampered_root if cell.cell_id == root.cell_id else cell
            for cell in plan.stages[0].cells
        ),
    )
    tampered_plan = StudyPlan(plan.name, plan.phase, (tampered_stage,))
    with pytest.raises(CampaignError, match="exactly inherit"):
        Campaign(tmp_path).register(
            tampered_plan,
            blueprint_manifest=blueprint.to_manifest(),
            phase1_decision_manifest=phase1_decision,
        )


def test_phase1_decision_rejects_embedded_qualification_tampering():
    payload = phase1_decision_for_phase2(smoke=False)
    baseline = next(
        item
        for item in payload["phase1_campaign_manifest"]["confirmation"]["results"]
        if item["variant"] == "baseline"
    )
    cell_id = baseline["per_seed"][0]["cell_id"]
    evidence = next(
        item
        for item in payload["phase1_campaign_manifest"]["cells"]
        if item["cell_id"] == cell_id
    )
    evidence["qualification"]["promotion_eligible"] = False
    rehash_phase1_decision(payload)
    with pytest.raises(CampaignError, match="bound qualification"):
        Campaign.phase1_decision_from_manifest(payload)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda qualification: qualification.update(qualified=False),
        lambda qualification: qualification["checks"].update(provenance=False),
        lambda qualification: qualification.update(inputs={}),
        lambda qualification: qualification["scientific_outcome"].update(
            passed=not qualification["scientific_outcome"]["passed"]
        ),
        lambda qualification: qualification["scientific_outcome"].update(
            inapplicable_checks={"phase1_identification": "forged"}
        ),
        lambda qualification: qualification.update(
            selection_eligibility_mode="smoke_protocol_only"
        ),
        lambda qualification: qualification["checks"].pop(
            "deployment_schedule_integrity"
        ),
        lambda qualification: qualification["scientific_outcome"]["checks"].update(
            invented_endpoint=True
        ),
        lambda qualification: qualification["thresholds"].update(
            support_target_abs_error_max=999.0
        ),
        lambda qualification: qualification["promotion_ineligible_reasons"].append(
            "invented_reason"
        ),
    ),
)
def test_phase1_decision_rejects_consistently_rehashed_contradictory_qualification(
    mutate,
):
    payload = phase1_decision_for_phase2(smoke=False)
    evidence = unreferenced_phase1_cell_evidence(payload)
    mutate(evidence["qualification"])
    evidence["qualification_sha256"] = qualification_file_sha256(
        evidence["qualification"]
    )
    rehash_phase1_decision(payload)

    with pytest.raises(CampaignError, match="qualification semantic replay"):
        Campaign.phase1_decision_from_manifest(payload)


def test_phase1_decision_rejects_mixed_implementation_identities_after_rehash():
    payload = phase1_decision_for_phase2(smoke=False)
    evidence = unreferenced_phase1_cell_evidence(payload)
    qualification = evidence["qualification"]
    qualification["implementation_identity"] = {
        **qualification["implementation_identity"],
        "python_source_sha256": "f" * 64,
    }
    qualification["implementation_identity_sha256"] = hashlib.sha256(
        canonical_json(qualification["implementation_identity"]).encode("utf-8")
    ).hexdigest()
    evidence["qualification_sha256"] = qualification_file_sha256(qualification)
    rehash_phase1_decision(payload)

    with pytest.raises(CampaignError, match="mixes qualification implementation"):
        Campaign.phase1_decision_from_manifest(payload)


def test_phase1_scientific_decision_rejects_dirty_identity():
    identity = {**TEST_IMPLEMENTATION_IDENTITY, "git_dirty": True}
    with pytest.raises(CampaignError, match="clean committed identity"):
        phase1_decision_for_phase2(
            smoke=False,
            implementation_identity=identity,
        )


def test_phase1_decision_recomputes_plan_history_hashes():
    payload = phase1_decision_for_phase2(smoke=True)
    payload["phase1_campaign_manifest"]["plan_history"][0]["sha256"] = (
        "sha256:" + "f" * 64
    )
    rehash_phase1_decision(payload)

    with pytest.raises(CampaignError, match="plan-history evidence"):
        Campaign.phase1_decision_from_manifest(payload)


def test_phase1_decision_labels_opaque_journal_commitment():
    payload = phase1_decision_for_phase2(smoke=True)
    payload["phase1_campaign_manifest"]["journal_sha256_semantics"] = "recomputed"
    rehash_phase1_decision(payload)

    with pytest.raises(CampaignError, match="journal commitment"):
        Campaign.phase1_decision_from_manifest(payload)


@pytest.mark.parametrize("smoke", (False, True))
def test_phase1_decision_rejects_selected_nonpromotable_capability_arm(smoke):
    payload = phase1_decision_for_phase2(
        smoke=smoke,
        forge_nonpromotable_stage="capacity_identification",
    )
    selected = payload["phase1_campaign_manifest"]["selection_chain"][1]["selection"]
    plan = StudyPlan.from_manifest(payload["phase1_campaign_manifest"]["plan"])
    selected_cell = next(
        cell for cell in plan.cells if cell.cell_id == selected["cell_ids"][0]
    )
    assert selected_cell.decision_map["qualification.promotable"] is False

    with pytest.raises(CampaignError, match="qualification semantic replay"):
        Campaign.phase1_decision_from_manifest(payload)


def test_phase1_transfer_records_full_capability_identification_conjunction():
    failed_stage = "capacity_identification"
    failed_variant = "width_1"
    failed_seed = 0
    payload = phase1_decision_for_phase2(
        smoke=False,
        capability_failure=(failed_stage, failed_variant, failed_seed),
    )
    manifest = payload["phase1_campaign_manifest"]
    plan = StudyPlan.from_manifest(manifest["plan"])
    selected_ids = {
        cell_id
        for chain_item in manifest["selection_chain"]
        for cell_id in chain_item["selection"]["cell_ids"]
    }
    diagnostic = next(
        cell
        for cell in plan.cells
        if cell.stage == failed_stage
        and cell.recipe_name == f"derived_{failed_stage}_{failed_variant}"
        and cell.seed == failed_seed
    )
    assert diagnostic.cell_id not in selected_ids

    verified = Campaign.phase1_decision_from_manifest(payload)
    capacity = next(
        panel
        for panel in verified["phase1_transfer"]["capability_evidence"]
        if panel["stage"] == "capacity_identification"
    )
    variant_name = diagnostic.recipe_name.removeprefix(
        "derived_capacity_identification_"
    )
    variant = next(
        item for item in capacity["variants"] if item["variant"] == variant_name
    )
    row = variant["per_seed"][0]
    assert row["scientific_outcome_passed"] is True
    assert row["native_passed"] is False
    assert row["deployed_passed"] is True
    assert row["validation_conjunction_passed"] is False
    assert row["conjunction_passed"] is False
    assert row["passed"] is False
    assert variant["passed_all_seeds"] is False


def test_phase1_false_positive_negative_control_forces_no_go(tmp_path):
    payload = phase1_decision_for_phase2(smoke=False)
    set_phase1_variant_outcome(payload, "support_only", passed=True)
    payload.update(
        {
            "authorization_mode": "scientific_no_go",
            "decision": "no_go",
            "authorizes_phase2_scientific": False,
            "authorizes_phase2_smoke": False,
        }
    )
    rehash_phase1_decision(payload)
    verified = Campaign.phase1_decision_from_manifest(payload)
    assert verified["decision"] == "no_go"
    blueprint = build_phase2_blueprint(
        smoke=False,
        phase1_decision=payload,
    )
    with pytest.raises(CampaignError, match="protocol-only"):
        Campaign(tmp_path).register(
            build_phase2_plan(
                smoke=False,
                phase1_decision=payload,
            ),
            blueprint_manifest=blueprint.to_manifest(),
            phase1_decision_manifest=payload,
        )
    payload["authorization_mode"] = "scientific_go"
    payload["decision"] = "go"
    payload["authorizes_phase2_scientific"] = True
    payload["authorizes_phase2_smoke"] = True
    rehash_phase1_decision(payload)
    with pytest.raises(CampaignError, match="authorization differs"):
        Campaign.phase1_decision_from_manifest(payload)


def test_phase1_failed_stress_requires_explicit_scope_narrowing():
    payload = phase1_decision_for_phase2(smoke=False)
    set_phase1_variant_outcome(payload, "noise", passed=False)
    confirmation = payload["phase1_campaign_manifest"]["confirmation"]
    confirmation["stress_failures"] = ["noise"]
    rehash_phase1_decision(payload)
    with pytest.raises(CampaignError, match="explicit scope narrowing"):
        Campaign.phase1_decision_from_manifest(payload)
    confirmation["scope_narrowing"] = {
        "noise": "claims exclude observation noise at standard deviation 0.1"
    }
    rehash_phase1_decision(payload)
    assert Campaign.phase1_decision_from_manifest(payload)["decision"] == "go"


def test_only_legal_transitions_and_required_artifacts_are_accepted(tmp_path):
    plan = one_cell_plan()
    campaign = Campaign(tmp_path)
    register_test_plan(campaign, plan)
    cell_id = plan.cells[0].cell_id
    with pytest.raises(InvalidTransition, match="planned -> trained"):
        campaign.transition(cell_id, RunState.TRAINED)
    preparation = write_artifact(tmp_path, "preparation", {"ready": True})
    prepare_manifest = write_artifact(
        tmp_path, "prepare_manifest", {"stage": "prepare"}
    )
    campaign.transition(
        cell_id, RunState.PREPARED, artifacts=(preparation, prepare_manifest)
    )
    campaign.transition(cell_id, RunState.RUNNING)
    with pytest.raises(ArtifactError, match="missing content-addressed"):
        campaign.transition(cell_id, RunState.TRAINED)
    checkpoint = write_artifact(tmp_path, "checkpoint", {"weights": []})
    training_report = write_artifact(tmp_path, "training_report", {"tokens": 64})
    train_manifest = write_artifact(tmp_path, "train_manifest", {"stage": "train"})
    assert (
        campaign.transition(
            cell_id,
            RunState.TRAINED,
            artifacts=(checkpoint, training_report, train_manifest),
        ).state
        is RunState.TRAINED
    )
    with pytest.raises(InvalidTransition):
        campaign.transition(cell_id, RunState.EVALUATED)


def test_transition_rejects_future_artifacts_but_allows_identical_reemission(tmp_path):
    plan = one_cell_plan()
    campaign = Campaign(tmp_path)
    register_test_plan(campaign, plan)
    cell_id = plan.cells[0].cell_id
    preparation = write_artifact(tmp_path, "preparation", {"ready": True})
    prepare_manifest = write_artifact(
        tmp_path, "prepare_manifest", {"stage": "prepare"}
    )
    future_checkpoint = write_artifact(tmp_path, "checkpoint", {"future": True})
    with pytest.raises(ArtifactError, match="belonging to another stage"):
        campaign.transition(
            cell_id,
            RunState.PREPARED,
            artifacts=(preparation, prepare_manifest, future_checkpoint),
        )
    campaign.transition(
        cell_id,
        RunState.PREPARED,
        artifacts=(preparation, prepare_manifest),
    )
    assert (
        campaign.transition(
            cell_id,
            RunState.RUNNING,
            artifacts=(preparation,),
        ).state
        is RunState.RUNNING
    )


def test_journal_replay_rejects_illegal_transitions_and_artifact_replacement(tmp_path):
    illegal = Campaign(tmp_path / "illegal")
    illegal_plan = one_cell_plan()
    register_test_plan(illegal, illegal_plan)
    illegal_cell = illegal_plan.cells[0].cell_id
    illegal._append_event(
        illegal._event(
            "transition",
            illegal_cell,
            previous=RunState.PLANNED,
            target=RunState.EVALUATED,
        )
    )
    with pytest.raises(CampaignError, match="illegal journal transition"):
        illegal.record(illegal_cell)

    replacement = Campaign(tmp_path / "replacement")
    replacement_plan = one_cell_plan()
    register_test_plan(replacement, replacement_plan)
    replacement_cell = replacement_plan.cells[0].cell_id
    original = write_artifact(replacement.root, "preparation", {"version": 1})
    prepare_manifest = write_artifact(
        replacement.root, "prepare_manifest", {"stage": "prepare"}
    )
    replacement.transition(
        replacement_cell,
        RunState.PREPARED,
        artifacts=(original, prepare_manifest),
    )
    replacement_path = replacement.root / "outputs" / "replacement.json"
    replacement_path.write_text('{"version": 2}\n')
    forged = ArtifactRef.from_path(
        "preparation", replacement_path, root=replacement.root
    )
    replacement._append_event(
        replacement._event(
            "transition",
            replacement_cell,
            previous=RunState.PREPARED,
            target=RunState.RUNNING,
            artifacts=(forged,),
        )
    )
    with pytest.raises(CampaignError, match="artifact replacement"):
        replacement.record(replacement_cell)


def test_evaluation_report_is_not_qualification_and_inputs_are_hash_bound(tmp_path):
    plan = one_cell_plan(smoke=False)
    campaign = Campaign(tmp_path)
    register_test_plan(campaign, plan)
    cell_id = plan.cells[0].cell_id
    inputs = advance_to_evaluated(campaign, cell_id)
    qualify_manifest = write_artifact(
        tmp_path, "qualify_manifest", {"stage": "qualify"}
    )
    assert campaign.eligible_for_qualification(cell_id)
    assert not campaign.eligible_for_promotion(cell_id)

    report_only = write_artifact(
        tmp_path,
        "qualification",
        {
            "schema": QUALIFICATION_SCHEMA,
            "cell_id": cell_id,
            "qualified": True,
            "metrics": {"fvu": 0.1},
        },
    )
    with pytest.raises(ArtifactError, match="noncanonical field set"):
        campaign.transition(
            cell_id,
            RunState.QUALIFIED,
            artifacts=(report_only, qualify_manifest),
        )

    wrong_binding = write_artifact(
        tmp_path,
        "qualification",
        qualification_payload(
            campaign._require_cell(cell_id),
            {kind: "0" * 64 for kind in inputs},
        ),
    )
    with pytest.raises(ArtifactError, match="binding mismatch"):
        campaign.transition(
            cell_id,
            RunState.QUALIFIED,
            artifacts=(wrong_binding, qualify_manifest),
        )

    qualification = good_qualification(campaign, cell_id, inputs)
    record = campaign.transition(
        cell_id,
        RunState.QUALIFIED,
        artifacts=(qualification, qualify_manifest),
    )
    assert record.state is RunState.QUALIFIED
    assert campaign.eligible_for_promotion(cell_id)


def test_preparation_rejects_campaign_implementation_drift(tmp_path):
    first_plan = one_cell_plan(seed=0)
    first = first_plan.cells[0]
    second = one_cell_plan(seed=1).cells[0]
    plan = StudyPlan(
        "implementation_drift",
        Phase.PHASE1,
        (
            StageSpec(
                "test",
                (first, second),
                selection_policy=first_plan.stages[0].selection_policy,
            ),
        ),
    )
    campaign = Campaign(tmp_path)
    register_test_plan(campaign, plan)

    first_preparation = write_cell_artifact(
        campaign,
        first.cell_id,
        "preparation",
        {
            "schema": PREPARATION_SCHEMA,
            "cell_id": first.cell_id,
            "implementation": TEST_IMPLEMENTATION_IDENTITY,
            "implementation_sha256": TEST_IMPLEMENTATION_IDENTITY_SHA256,
        },
    )
    first_manifest = write_cell_artifact(
        campaign, first.cell_id, "prepare_manifest", {"stage": "prepare"}
    )
    campaign.transition(
        first.cell_id,
        RunState.PREPARED,
        artifacts=(first_preparation, first_manifest),
    )

    different_identity = {
        **TEST_IMPLEMENTATION_IDENTITY,
        "python_source_sha256": "f" * 64,
    }
    second_preparation = write_cell_artifact(
        campaign,
        second.cell_id,
        "preparation",
        {
            "schema": PREPARATION_SCHEMA,
            "cell_id": second.cell_id,
            "implementation": different_identity,
            "implementation_sha256": hashlib.sha256(
                canonical_json(different_identity).encode("utf-8")
            ).hexdigest(),
        },
    )
    second_manifest = write_cell_artifact(
        campaign, second.cell_id, "prepare_manifest", {"stage": "prepare"}
    )
    with pytest.raises(ArtifactError, match="already prepared campaign cell"):
        campaign.transition(
            second.cell_id,
            RunState.PREPARED,
            artifacts=(second_preparation, second_manifest),
        )
    assert campaign.record(second.cell_id).state is RunState.PLANNED


def test_concurrent_first_preparations_atomically_pin_one_identity(tmp_path):
    first_plan = one_cell_plan(seed=0, smoke=False)
    first = first_plan.cells[0]
    second = one_cell_plan(seed=1, smoke=False).cells[0]
    plan = StudyPlan(
        "atomic_implementation_pin",
        Phase.PHASE1,
        (
            StageSpec(
                "test",
                (first, second),
                selection_policy=first_plan.stages[0].selection_policy,
            ),
        ),
    )
    campaign = Campaign(tmp_path)
    register_test_plan(campaign, plan)
    identities = {
        first.cell_id: TEST_IMPLEMENTATION_IDENTITY,
        second.cell_id: {
            **TEST_IMPLEMENTATION_IDENTITY,
            "python_source_sha256": "f" * 64,
        },
    }
    transitions = {}
    for cell in (first, second):
        identity = identities[cell.cell_id]
        preparation = write_cell_artifact(
            campaign,
            cell.cell_id,
            "preparation",
            {
                "schema": PREPARATION_SCHEMA,
                "cell_id": cell.cell_id,
                "implementation": identity,
                "implementation_sha256": hashlib.sha256(
                    canonical_json(identity).encode("utf-8")
                ).hexdigest(),
            },
        )
        manifest = write_cell_artifact(
            campaign,
            cell.cell_id,
            "prepare_manifest",
            {"stage": "prepare"},
        )
        transitions[cell.cell_id] = (preparation, manifest)

    def prepare(cell_id: str) -> str:
        try:
            campaign.transition(
                cell_id,
                RunState.PREPARED,
                artifacts=transitions[cell_id],
            )
        except ArtifactError:
            return "refused"
        return "prepared"

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(prepare, (first.cell_id, second.cell_id)))

    assert sorted(outcomes) == ["prepared", "refused"]
    prepared = next(
        cell
        for cell in (first, second)
        if campaign.record(cell.cell_id).state is RunState.PREPARED
    )
    refused = second if prepared is first else first
    assert campaign.record(refused.cell_id).state is RunState.PLANNED
    pin = json.loads(campaign.implementation_identity_path.read_text())
    assert pin["implementation_identity"] == identities[prepared.cell_id]


def test_running_and_failed_cells_are_explicitly_not_default_runnable(tmp_path):
    plan = one_cell_plan()
    campaign = Campaign(tmp_path)
    register_test_plan(campaign, plan)
    cell_id = plan.cells[0].cell_id
    preparation = write_artifact(tmp_path, "preparation", {"ready": True})
    prepare_manifest = write_artifact(
        tmp_path, "prepare_manifest", {"stage": "prepare"}
    )
    campaign.transition(
        cell_id,
        RunState.PREPARED,
        artifacts=(preparation, prepare_manifest),
    )
    campaign.transition(cell_id, RunState.RUNNING)

    assert campaign.runnable_cell_ids() == ()
    assert campaign.runnable_cell_ids(include_resume_required=True) == (cell_id,)
    assert campaign.status()["resume_required"] == 1
    campaign.transition(cell_id, RunState.FAILED)
    assert campaign.runnable_cell_ids() == ()
    assert campaign.runnable_cell_ids(include_failed=True) == (cell_id,)
    assert campaign.status()["failed_retry_required"] == 1


def test_qualification_cannot_forge_selection_metrics_while_naming_real_evaluation(
    tmp_path,
):
    plan = one_cell_plan()
    campaign = Campaign(tmp_path)
    register_test_plan(campaign, plan)
    cell_id = plan.cells[0].cell_id
    inputs = advance_to_evaluated(campaign, cell_id)
    forged = write_artifact(
        tmp_path,
        "qualification",
        qualification_payload(
            campaign._require_cell(cell_id),
            {kind: ref.sha256 for kind, ref in inputs.items()},
            validation={"fvu": -1_000.0, "rate_distortion": 1_000.0},
            promotion_eligible=False,
            protocol_eligible=True,
        ),
    )
    qualify_manifest = write_artifact(
        tmp_path, "qualify_manifest", {"stage": "qualify"}
    )
    with pytest.raises(ArtifactError, match="differ from the bound evaluation"):
        campaign.transition(
            cell_id,
            RunState.QUALIFIED,
            artifacts=(forged, qualify_manifest),
        )
    assert campaign.record(cell_id).state is RunState.EVALUATED


@pytest.mark.parametrize(
    ("smoke", "error"),
    (
        (False, "immutable nonpromotable recipe"),
        (True, "smoke protocol eligibility is inconsistent"),
    ),
)
def test_qualification_cannot_override_resolved_nonpromotable_recipe(
    tmp_path,
    smoke,
    error,
):
    resolved, selection_policy = phase1_selection_template(0, smoke=smoke)
    diagnostic = replace(
        resolved,
        name=f"phase1.test.nonpromotable.{str(smoke).lower()}",
        stage="test",
        decisions=tuple(
            replace(decision, value=False)
            if decision.name == "qualification.promotable"
            else decision
            for decision in resolved.decisions
        ),
    )
    assert diagnostic.decision_map["qualification.promotable"] is False
    plan = StudyPlan(
        f"test_nonpromotable_qualification_{str(smoke).lower()}",
        Phase.PHASE1,
        (
            StageSpec(
                "test",
                (diagnostic,),
                selection_policy=replace(
                    selection_policy,
                    eligible_recipe_names=(diagnostic.recipe_name,),
                ),
            ),
        ),
    )
    campaign = Campaign(tmp_path)
    register_test_plan(campaign, plan)
    inputs = advance_to_evaluated(campaign, diagnostic.cell_id)
    forged = write_artifact(
        tmp_path,
        "qualification",
        qualification_payload(
            diagnostic,
            {kind: ref.sha256 for kind, ref in inputs.items()},
            promotion_eligible=not smoke,
            protocol_eligible=smoke,
        ),
    )
    qualify_manifest = write_artifact(
        tmp_path, "qualify_manifest", {"stage": "qualify"}
    )

    with pytest.raises(ArtifactError, match=error):
        campaign.transition(
            diagnostic.cell_id,
            RunState.QUALIFIED,
            artifacts=(forged, qualify_manifest),
        )
    assert campaign.record(diagnostic.cell_id).state is RunState.EVALUATED


def test_promotion_requires_an_explicit_qualification_bound_decision(tmp_path):
    plan = one_cell_plan(smoke=False)
    campaign = Campaign(tmp_path)
    register_test_plan(campaign, plan)
    cell_id = plan.cells[0].cell_id
    inputs = advance_to_evaluated(campaign, cell_id)
    qualification = good_qualification(campaign, cell_id, inputs)
    qualify_manifest = write_artifact(
        tmp_path, "qualify_manifest", {"stage": "qualify"}
    )
    campaign.transition(
        cell_id,
        RunState.QUALIFIED,
        artifacts=(qualification, qualify_manifest),
    )

    bad = tmp_path / "bad-promotion.json"
    bad.write_text(
        json.dumps(
            {
                "schema": PROMOTION_SCHEMA,
                "cell_id": cell_id,
                "approved": True,
                "qualification_sha256": "0" * 64,
            }
        )
    )
    with pytest.raises(ArtifactError, match="not bound"):
        campaign.promote(cell_id, bad)

    good = tmp_path / "promotion.json"
    good.write_text(
        json.dumps(
            {
                "schema": PROMOTION_SCHEMA,
                "cell_id": cell_id,
                "approved": True,
                "qualification_sha256": qualification.sha256,
            }
        )
    )
    assert campaign.promote(cell_id, good).state is RunState.PROMOTED


def test_artifact_mutation_fails_closed_on_the_next_gate(tmp_path):
    plan = one_cell_plan()
    campaign = Campaign(tmp_path)
    register_test_plan(campaign, plan)
    cell_id = plan.cells[0].cell_id
    preparation = write_artifact(tmp_path, "preparation", {"ready": True})
    prepare_manifest = write_artifact(
        tmp_path, "prepare_manifest", {"stage": "prepare"}
    )
    campaign.transition(
        cell_id, RunState.PREPARED, artifacts=(preparation, prepare_manifest)
    )
    campaign.transition(cell_id, RunState.RUNNING)
    checkpoint = write_artifact(tmp_path, "checkpoint", {"weights": [1]})
    training_report = write_artifact(tmp_path, "training_report", {"tokens": 64})
    train_manifest = write_artifact(tmp_path, "train_manifest", {"stage": "train"})
    campaign.transition(
        cell_id,
        RunState.TRAINED,
        artifacts=(checkpoint, training_report, train_manifest),
    )
    checkpoint.resolve(tmp_path).write_text("mutated\n")
    calibration = write_artifact(tmp_path, "calibration", {"threshold": 1})
    deployment_codec = write_artifact(
        tmp_path, "deployment_codec", {"model": "deployable"}
    )
    calibration_record = write_artifact(
        tmp_path, "calibration_record", {"threshold": 1}
    )
    calibrate_manifest = write_artifact(
        tmp_path, "calibrate_manifest", {"stage": "calibrate"}
    )
    with pytest.raises(ArtifactError, match="mismatch"):
        campaign.transition(
            cell_id,
            RunState.CALIBRATED,
            artifacts=(
                calibration,
                deployment_codec,
                calibration_record,
                calibrate_manifest,
            ),
        )


def test_failure_is_append_only_and_retry_returns_to_durable_predecessor(tmp_path):
    plan = one_cell_plan()
    campaign = Campaign(tmp_path)
    register_test_plan(campaign, plan)
    cell_id = plan.cells[0].cell_id
    preparation = write_artifact(tmp_path, "preparation", {"ready": True})
    prepare_manifest = write_artifact(
        tmp_path, "prepare_manifest", {"stage": "prepare"}
    )
    campaign.transition(
        cell_id, RunState.PREPARED, artifacts=(preparation, prepare_manifest)
    )
    campaign.transition(cell_id, RunState.RUNNING)
    failed = campaign.transition(
        cell_id,
        RunState.FAILED,
        message="injected crash",
        metadata={"stage": "train"},
    )
    assert failed.resume_state is RunState.PREPARED
    before = len(campaign.events(cell_id))
    retried = campaign.retry(cell_id)
    assert retried.state is RunState.PREPARED
    assert len(campaign.events(cell_id)) == before + 1
    assert [
        event["target"]
        for event in campaign.events(cell_id)
        if event["event"] == "transition"
    ] == [
        "planned",
        "prepared",
        "running",
        "failed",
        "prepared",
    ]


def test_stale_lock_requires_explicit_reconciliation(tmp_path):
    now = [100.0]
    plan = one_cell_plan()
    campaign = Campaign(tmp_path, clock=lambda: now[0])
    register_test_plan(campaign, plan)
    cell_id = plan.cells[0].cell_id
    lock = campaign.lock(cell_id)
    lock.__enter__()
    lock_payload = json.loads(campaign.lock_path(cell_id).read_text())
    try:
        with pytest.raises(CampaignLocked):
            campaign.transition(cell_id, RunState.PREPARED)
    finally:
        lock.__exit__(None, None, None)
    lock_payload["host"] = "retired-worker.invalid"
    campaign.lock_path(cell_id).write_text(json.dumps(lock_payload) + "\n")
    try:
        now[0] = 109.0
        assert campaign.reconcile_stale_locks(10.0) == ()
        now[0] = 111.0
        assert campaign.reconcile_stale_locks(10.0) == (cell_id,)
        assert not campaign.lock_path(cell_id).exists()
        preparation = write_artifact(tmp_path, "preparation", {"ready": True})
        prepare_manifest = write_artifact(
            tmp_path, "prepare_manifest", {"stage": "prepare"}
        )
        assert (
            campaign.transition(
                cell_id,
                RunState.PREPARED,
                artifacts=(preparation, prepare_manifest),
            ).state
            is RunState.PREPARED
        )
    finally:
        campaign.lock_path(cell_id).unlink(missing_ok=True)


def test_cell_lock_release_waits_out_inflight_heartbeat(tmp_path, monkeypatch):
    plan = one_cell_plan()
    campaign = Campaign(tmp_path, lock_heartbeat_seconds=0.01)
    register_test_plan(campaign, plan)
    lock = campaign.lock(plan.cells[0].cell_id)
    lock.__enter__()
    original_atomic_json = campaign_module._atomic_json
    heartbeat_started = threading.Event()
    release_heartbeat = threading.Event()

    def delayed_atomic_json(path, payload):
        if path == lock.path and threading.current_thread() is lock._thread:
            heartbeat_started.set()
            assert release_heartbeat.wait(timeout=5)
        original_atomic_json(path, payload)

    monkeypatch.setattr(campaign_module, "_atomic_json", delayed_atomic_json)
    assert heartbeat_started.wait(timeout=5)
    released = threading.Event()

    def release_lock():
        lock.__exit__(None, None, None)
        released.set()

    thread = threading.Thread(target=release_lock)
    thread.start()
    assert not released.wait(timeout=0.05)
    release_heartbeat.set()
    thread.join(timeout=5)
    assert released.is_set()
    assert not lock.path.exists()
    assert lock.guard_path.exists()


def test_stale_reconcile_terminates_owned_orphan_worker_group(tmp_path, monkeypatch):
    now = 100.0
    plan = one_cell_plan()
    campaign = Campaign(tmp_path, clock=lambda: now)
    register_test_plan(campaign, plan)
    cell_id = plan.cells[0].cell_id
    lease = {
        "schema": campaign_module.CAMPAIGN_SCHEMA,
        "cell_id": cell_id,
        "attempt_id": "orphan-attempt",
        "pid": 111,
        "owner_process_identity": "owner-birth",
        "host": socket.gethostname(),
        "acquired_at": 0.0,
        "heartbeat_at": 0.0,
        "worker_pid": 222,
        "worker_pgid": 333,
        "worker_process_identity": "worker-birth",
    }
    campaign_module._atomic_json(campaign.lock_path(cell_id), lease)
    worker_checks = iter((True, False))

    def process_matches(pid, identity):
        if pid == 111:
            return False
        assert (pid, identity) == (222, "worker-birth")
        return next(worker_checks)

    signals = []
    monkeypatch.setattr(campaign_module, "_process_matches", process_matches)
    monkeypatch.setattr(os, "getpgid", lambda pid: 333)
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: signals.append((pgid, sig)))

    assert campaign.reconcile_stale_locks(10.0) == (cell_id,)
    assert signals == [(333, signal.SIGTERM)]
    assert not campaign.lock_path(cell_id).exists()
    event = campaign.events()[-1]
    assert event["metadata"]["worker_process_group_terminated"] is True


def test_closed_stage_cannot_be_bypassed_with_an_explicit_cell_id(tmp_path):
    resolved_first, _ = phase1_selection_template(0, smoke=False)
    resolved_second, selection_policy = phase1_selection_template(1, smoke=False)
    first = CellSpec(
        "phase1.alpha.cell.s0",
        Phase.PHASE1,
        "alpha",
        resolved_first.recipe_name,
        resolved_first.recipe_id,
        0,
        resolved_first.decisions,
    )
    second = CellSpec(
        "phase1.beta.cell.s0",
        Phase.PHASE1,
        "beta",
        resolved_second.recipe_name,
        resolved_second.recipe_id,
        1,
        resolved_second.decisions,
    )
    plan = StudyPlan(
        "gated_campaign",
        Phase.PHASE1,
        (
            StageSpec("alpha", (first,)),
            StageSpec(
                "beta",
                (second,),
                depends_on=("alpha",),
                gate=GateCondition("alpha", minimum_count=1),
                selection_policy=selection_policy,
            ),
        ),
    )
    campaign = Campaign(tmp_path)
    register_test_plan(campaign, plan)
    assert campaign.runnable_cell_ids() == (first.cell_id,)
    with pytest.raises(CampaignError, match="unopened stage"):
        CampaignRunner(campaign).run(cell_ids=(second.cell_id,))

    inputs = advance_to_evaluated(campaign, first.cell_id)
    qualification = good_qualification(campaign, first.cell_id, inputs)
    qualify_manifest = write_artifact(
        tmp_path, "qualify_manifest", {"stage": "qualify"}
    )
    campaign.transition(
        first.cell_id,
        RunState.QUALIFIED,
        artifacts=(qualification, qualify_manifest),
    )
    assert campaign.stage_open("beta")
    assert campaign.runnable_cell_ids() == (second.cell_id,)
    qualification.resolve(tmp_path).write_text("tampered\n")
    assert not campaign.stage_open("beta")
    assert campaign.runnable_cell_ids() == ()


def test_frozen_selection_excludes_negative_outcomes_and_extends_exact_blueprint(
    tmp_path,
):
    blueprint = build_phase1_blueprint(smoke=False)
    plan = build_phase1_plan(smoke=False)
    campaign = Campaign(tmp_path)
    campaign.register(plan, blueprint_manifest=blueprint.to_manifest())
    stage = plan.stages[-1]
    candidate_ids = sorted({cell.candidate_id for cell in stage.cells})
    assert len(candidate_ids) >= 2
    positive_candidate = next(
        cell.candidate_id
        for cell in stage.cells
        if cell.recipe_name in stage.selection_policy.eligible_recipe_names
    )
    for cell in stage.cells:
        negative = cell.candidate_id != positive_candidate
        qualify_cell(
            campaign,
            cell.cell_id,
            metric=10.0 if negative else 1.0,
            scientific_passed=not negative,
        )
    selection_path = tmp_path / "selections" / f"{stage.name}.json"
    payload = campaign.select_stage(stage.name, out=selection_path)
    assert len(payload["selected"]) == 1
    assert payload["selected"][0]["candidate_id"] == positive_candidate
    assert {item["reason"] for item in payload["excluded_candidates"]} <= {
        "scientific_outcome_failed",
        "recipe_not_eligible_under_frozen_policy",
    }

    selection = FrozenSelection.from_dict(payload["selected"][0])
    extended = materialize_child_plan(plan, blueprint, selection)
    child_stage = extended.stages[-1]
    first_child = child_stage.cells[0]
    changed_decisions = tuple(
        replace(decision, rationale=decision.rationale + " rogue")
        if decision.name == "data.train_tokens"
        else decision
        for decision in first_child.decisions
    )
    rogue_child = replace(first_child, decisions=changed_decisions)
    rogue_stage = replace(child_stage, cells=(rogue_child, *child_stage.cells[1:]))
    rogue_plan = StudyPlan("rogue_extension", plan.phase, (*plan.stages, rogue_stage))
    with pytest.raises(CampaignError, match="differs from the next frozen blueprint"):
        campaign.extend(
            rogue_plan,
            selection=selection,
            selection_path=selection_path,
        )
    assert campaign.plan == plan
    campaign.extend(
        extended,
        selection=selection,
        selection_path=selection_path,
    )
    assert campaign.plan == extended
    assert campaign.plan.stages[-1].name == blueprint.rounds[0].name
    assert campaign.status()["counts"]["planned"] == (
        len(plan.cells) - len(stage.cells) + len(extended.stages[-1].cells)
    )


def test_main_extension_reconcile_recovers_post_commit_pointer_crash(
    tmp_path, monkeypatch
):
    import block_crosscoder_experiment.campaign as campaign_module

    blueprint = build_phase1_blueprint(seeds=(0,), smoke=True)
    plan = build_phase1_plan(seeds=(0,), smoke=True)
    campaign = Campaign(tmp_path)
    campaign.register(plan, blueprint_manifest=blueprint.to_manifest())
    stage = plan.stages[-1]
    for index, cell in enumerate(stage.cells):
        qualify_cell(campaign, cell.cell_id, metric=float(index + 1))
    selection_path = tmp_path / "selections" / f"{stage.name}.json"
    payload = campaign.select_stage(stage.name, out=selection_path)
    selection = FrozenSelection.from_dict(payload["selected"][0])
    extended = materialize_child_plan(plan, blueprint, selection)

    original_atomic_json = campaign_module._atomic_json

    def crash_before_plan_projection(path, body):
        if Path(path) == campaign.plan_path:
            raise OSError("injected post-journal crash")
        return original_atomic_json(path, body)

    monkeypatch.setattr(campaign_module, "_atomic_json", crash_before_plan_projection)
    with pytest.raises(OSError, match="post-journal crash"):
        campaign.extend(
            extended,
            selection=selection,
            selection_path=selection_path,
        )
    assert campaign.plan == plan
    extensions = [
        event for event in campaign.events() if event.get("event") == "plan_extension"
    ]
    assert len(extensions) == 1
    assert extensions[0]["metadata"]["plan_id"] == extended.plan_id

    monkeypatch.setattr(campaign_module, "_atomic_json", original_atomic_json)
    result = campaign.reconcile()
    assert result["plan_republished"] == extended.plan_id
    assert campaign.plan == extended
    assert (
        len(
            [
                event
                for event in campaign.events()
                if event.get("event") == "plan_extension"
            ]
        )
        == 1
    )


def test_family_revisit_reconcile_recovers_post_commit_pointer_crash(
    tmp_path, monkeypatch
):
    import block_crosscoder_experiment.campaign as campaign_module

    plan, blueprint, phase1_decision = phase2_test_inputs(smoke=True)
    campaign = Campaign(tmp_path)
    register_phase2_test_plan(campaign, plan, blueprint, phase1_decision)

    for round_spec in blueprint.rounds:
        stage = campaign.plan.stages[-1]
        assert stage.name == round_spec.source_stage
        for index, cell in enumerate(stage.cells):
            qualify_cell(
                campaign,
                cell.cell_id,
                metric=float(index + 1),
                validation={"negative_mean_raw_fvu": float(index + 1)},
            )
        selection_path = tmp_path / "selections" / f"{stage.name}.json"
        payload = campaign.select_stage(stage.name, out=selection_path)
        selection = FrozenSelection.from_dict(payload["selected"][0])
        extended = materialize_child_plan(campaign.plan, blueprint, selection)
        campaign.extend(
            extended,
            selection=selection,
            selection_path=selection_path,
        )

    family = blueprint.comparator_families[0]
    root_selection_path = tmp_path / "selections" / f"{family.name}_root.json"
    root_payload = campaign.select_family_root(family.name, out=root_selection_path)
    root_selection = FrozenSelection.from_dict(root_payload["selected"][0])
    campaign.extend_family(
        materialize_family_child_plan(
            campaign.plan, blueprint, family.name, root_selection
        ),
        family_name=family.name,
        selection=root_selection,
        selection_path=root_selection_path,
    )
    for index, round_spec in enumerate(family.rounds):
        stage = campaign.plan.stages[-1]
        assert stage.name == round_spec.name
        for cell_index, cell in enumerate(stage.cells):
            qualify_cell(
                campaign,
                cell.cell_id,
                metric=float(cell_index + 1),
                validation={"negative_mean_raw_fvu": float(cell_index + 1)},
            )
        if index == len(family.rounds) - 1:
            break
        selection_path = tmp_path / "selections" / f"{stage.name}.json"
        payload = campaign.select_stage(stage.name, out=selection_path)
        selection = FrozenSelection.from_dict(payload["selected"][0])
        campaign.extend_family(
            materialize_family_child_plan(
                campaign.plan, blueprint, family.name, selection
            ),
            family_name=family.name,
            selection=selection,
            selection_path=selection_path,
        )

    nomination_path = tmp_path / "selections" / f"{family.name}_nomination.json"
    nomination = campaign.select_family_revisit_inputs(family.name, out=nomination_path)
    nominations = tuple(
        FrozenSelection.from_dict(item) for item in nomination["selected"]
    )
    before_revisit = campaign.plan
    revisit_plan = materialize_family_revisit_plan(
        before_revisit, blueprint, family.name, nominations
    )

    original_atomic_json = campaign_module._atomic_json

    def crash_before_plan_projection(path, body):
        if Path(path) == campaign.plan_path:
            raise OSError("injected revisit post-journal crash")
        return original_atomic_json(path, body)

    monkeypatch.setattr(campaign_module, "_atomic_json", crash_before_plan_projection)
    with pytest.raises(OSError, match="revisit post-journal crash"):
        campaign.extend_family_revisit(
            revisit_plan,
            family_name=family.name,
            selection_path=nomination_path,
        )
    assert campaign.plan == before_revisit
    assert campaign.events()[-1]["metadata"]["plan_id"] == revisit_plan.plan_id

    monkeypatch.setattr(campaign_module, "_atomic_json", original_atomic_json)
    result = campaign.reconcile()
    assert result["plan_republished"] == revisit_plan.plan_id
    assert campaign.plan == revisit_plan


def test_selection_rejects_candidates_missing_a_declared_seed(tmp_path):
    blueprint = build_phase1_blueprint(seeds=(0, 1), smoke=True)
    source = blueprint.initial_stages[-1]
    by_candidate: dict[str, list[CellSpec]] = {}
    for cell in source.cells:
        by_candidate.setdefault(cell.candidate_id, []).append(cell)
    candidates = [
        sorted(cells, key=lambda item: item.seed) for cells in by_candidate.values()
    ]
    incomplete_cells = (candidates[0][0], candidates[1][1])
    stage = StageSpec(
        source.name,
        incomplete_cells,
        selection_policy=source.selection_policy,
    )
    plan = StudyPlan("incomplete_seed_stage", Phase.PHASE1, (stage,))
    campaign = Campaign(tmp_path)
    register_test_plan(campaign, plan)
    for cell in incomplete_cells:
        qualify_cell(campaign, cell.cell_id, metric=1.0)
    with pytest.raises(CampaignError, match="no seed-complete"):
        campaign.select_stage(stage.name)


def test_phase2_selection_enforces_authenticated_per_seed_sharing_guards(tmp_path):
    plan, blueprint, phase1_decision = phase2_test_inputs(smoke=False)
    campaign = Campaign(tmp_path)
    register_phase2_test_plan(campaign, plan, blueprint, phase1_decision)
    for index, cell in enumerate(plan.stages[0].cells):
        qualify_cell(
            campaign,
            cell.cell_id,
            metric=float(index + 1),
            validation={"negative_mean_raw_fvu": float(index + 1)},
        )
    selection_path = tmp_path / "selections" / "anchors.json"
    payload = campaign.select_stage("anchors_1m", out=selection_path)
    selection = FrozenSelection.from_dict(payload["selected"][0])
    campaign.extend(
        materialize_child_plan(campaign.plan, blueprint, selection),
        selection=selection,
        selection_path=selection_path,
    )
    topology = campaign.plan.stages[-1]
    for cell in topology.cells:
        guard = {
            "all_site_fvu_mean": 0.10,
            "site_only_heldout_fvu_mean": 0.33,
            "leave_one_out_heldout_fvu_mean": 0.25,
            "site_only_support_iou_mean": 0.80,
            "leave_one_out_support_iou_mean": 0.75,
            "site_only_coordinate_concordance_mean": 0.90,
            "leave_one_out_coordinate_concordance_mean": 0.85,
            "site_only_coordinate_concordance_min": 0.90,
            "leave_one_out_coordinate_concordance_min": 0.85,
            "site_only_intersection_recall_mean": 0.80,
            "leave_one_out_intersection_recall_mean": 0.75,
            "site_only_intersection_recall_min": 0.80,
            "leave_one_out_intersection_recall_min": 0.75,
            "site_only_intersection_energy_coverage_mean": 0.95,
            "leave_one_out_intersection_energy_coverage_mean": 0.92,
            "site_only_intersection_energy_coverage_min": 0.95,
            "leave_one_out_intersection_energy_coverage_min": 0.92,
        }
        qualify_cell(
            campaign,
            cell.cell_id,
            metric=1.0,
            validation={"negative_mean_raw_fvu": 1.0},
            sharing_guard=guard,
        )
    with pytest.raises(CampaignError, match="no seed-complete"):
        campaign.select_stage(topology.name)


def test_sharing_guard_enforces_absolute_and_root_relative_fvu(tmp_path):
    def guard(site_only: float, leave_one_out: float) -> dict[str, float]:
        return {
            "all_site_fvu_mean": 0.10,
            "site_only_heldout_fvu_mean": site_only,
            "leave_one_out_heldout_fvu_mean": leave_one_out,
            "site_only_support_iou_mean": 0.80,
            "leave_one_out_support_iou_mean": 0.75,
            "site_only_coordinate_concordance_mean": 0.90,
            "leave_one_out_coordinate_concordance_mean": 0.85,
            "site_only_coordinate_concordance_min": 0.90,
            "leave_one_out_coordinate_concordance_min": 0.85,
            "site_only_intersection_recall_mean": 0.80,
            "leave_one_out_intersection_recall_mean": 0.75,
            "site_only_intersection_recall_min": 0.80,
            "leave_one_out_intersection_recall_min": 0.75,
            "site_only_intersection_energy_coverage_mean": 0.95,
            "leave_one_out_intersection_energy_coverage_mean": 0.92,
            "site_only_intersection_energy_coverage_min": 0.95,
            "leave_one_out_intersection_energy_coverage_min": 0.92,
        }

    plan, blueprint, phase1_decision = phase2_test_inputs(smoke=False)
    absolute = Campaign(tmp_path / "absolute")
    register_phase2_test_plan(
        absolute,
        plan,
        blueprint,
        phase1_decision,
    )
    absolute_root = absolute.plan.stages[-1]
    for cell in absolute_root.cells:
        qualify_cell(
            absolute,
            cell.cell_id,
            metric=1.0,
            validation={"negative_mean_raw_fvu": 1.0},
            sharing_guard=guard(0.995, 0.995),
        )
    absolute_path = tmp_path / "absolute" / "selections" / "root.json"
    absolute_payload = absolute.select_stage(absolute_root.name, out=absolute_path)
    absolute_selection = FrozenSelection.from_dict(absolute_payload["selected"][0])
    absolute.extend(
        materialize_child_plan(absolute.plan, blueprint, absolute_selection),
        selection=absolute_selection,
        selection_path=absolute_path,
    )
    for cell in absolute.plan.stages[-1].cells:
        qualify_cell(
            absolute,
            cell.cell_id,
            metric=1.0,
            validation={"negative_mean_raw_fvu": 1.0},
            sharing_guard=guard(1.001, 1.001),
        )
    with pytest.raises(
        CampaignError,
        match="lacks eligible default-parent|no seed-complete",
    ):
        absolute.select_stage(absolute.plan.stages[-1].name)

    cumulative = Campaign(tmp_path / "cumulative")
    register_phase2_test_plan(
        cumulative,
        plan,
        blueprint,
        phase1_decision,
    )
    root_stage = cumulative.plan.stages[-1]
    for cell in root_stage.cells:
        qualify_cell(
            cumulative,
            cell.cell_id,
            metric=1.0,
            validation={"negative_mean_raw_fvu": 1.0},
            sharing_guard=guard(0.30, 0.25),
        )
    selection_path = tmp_path / "cumulative" / "selections" / "root.json"
    payload = cumulative.select_stage(root_stage.name, out=selection_path)
    selection = FrozenSelection.from_dict(payload["selected"][0])
    cumulative.extend(
        materialize_child_plan(cumulative.plan, blueprint, selection),
        selection=selection,
        selection_path=selection_path,
    )
    first_child = cumulative.plan.stages[-1]
    for cell in first_child.cells:
        qualify_cell(
            cumulative,
            cell.cell_id,
            metric=1.0,
            validation={"negative_mean_raw_fvu": 1.0},
            sharing_guard=guard(0.319, 0.269),
        )
    first_path = tmp_path / "cumulative" / "selections" / "first.json"
    first_payload = cumulative.select_stage(first_child.name, out=first_path)
    assert len(first_payload["ranked_candidates"]) == 1
    assert first_payload["ranked_candidates"][0]["recipe_name"].endswith(
        "selected_parent"
    )
    assert (
        first_payload["ranked_candidates"][0]["minimum_effect_gate"]["role"]
        == "default_parent"
    )
    assert any(
        item.get("reason") == "minimum_effect_not_met_against_default_parent"
        for item in first_payload["excluded_candidates"]
    )
    sensitivity = first_payload["threshold_sensitivity"]
    assert sensitivity["applicable"] is True
    assert sensitivity["mode"] == ("marginal_counterfactuals_center_policy_not_retuned")
    assert {surface["name"] for surface in sensitivity["surfaces"]} == {
        name for name, _ in SELECTION_THRESHOLD_SENSITIVITY
    }
    minimum_effect_surface = next(
        surface
        for surface in sensitivity["surfaces"]
        if surface["name"] == "minimum_effect_absolute"
    )
    assert [row["threshold"] for row in minimum_effect_surface["rows"]] == [
        0.0,
        0.001,
        0.002,
        0.005,
    ]
    concordance_surface = next(
        surface
        for surface in sensitivity["surfaces"]
        if surface["name"] == "sharing_coordinate_concordance_min"
    )
    assert all(row["evaluated_candidate_ids"] for row in concordance_surface["rows"])
    selected_guard = first_payload["ranked_candidates"][0]["observations"][0][
        "sharing_guard"
    ]
    assert (
        selected_guard["root_cell_id"]
        == selected_guard["authenticated_lineage"][-1]["cell_id"]
    )
    assert len(selected_guard["authenticated_lineage"]) == 2
    selection = FrozenSelection.from_dict(first_payload["selected"][0])
    cumulative.extend(
        materialize_child_plan(cumulative.plan, blueprint, selection),
        selection=selection,
        selection_path=first_path,
    )
    second_child = cumulative.plan.stages[-1]
    for cell in second_child.cells:
        qualify_cell(
            cumulative,
            cell.cell_id,
            metric=1.0,
            validation={"negative_mean_raw_fvu": 1.0},
            sharing_guard=guard(0.329, 0.279),
        )
    with pytest.raises(CampaignError, match="no seed-complete"):
        cumulative.select_stage(second_child.name)


def test_sharing_guard_uses_coordinate_agreement_not_all_view_synergy(tmp_path):
    def guard(*, concordance: float, coverage: float) -> dict[str, float]:
        return {
            # Exact redundancy has no all-view FVU advantage and must remain
            # eligible when its support and within-block coordinate agree.
            "all_site_fvu_mean": 0.30,
            "site_only_heldout_fvu_mean": 0.30,
            "leave_one_out_heldout_fvu_mean": 0.30,
            "site_only_support_iou_mean": 1.0,
            "leave_one_out_support_iou_mean": 1.0,
            "site_only_coordinate_concordance_mean": concordance,
            "leave_one_out_coordinate_concordance_mean": concordance,
            "site_only_coordinate_concordance_min": concordance,
            "leave_one_out_coordinate_concordance_min": concordance,
            "site_only_intersection_recall_mean": 1.0,
            "leave_one_out_intersection_recall_mean": 1.0,
            "site_only_intersection_recall_min": 1.0,
            "leave_one_out_intersection_recall_min": 1.0,
            "site_only_intersection_energy_coverage_mean": coverage,
            "leave_one_out_intersection_energy_coverage_mean": coverage,
            "site_only_intersection_energy_coverage_min": coverage,
            "leave_one_out_intersection_energy_coverage_min": coverage,
        }

    plan, blueprint, phase1_decision = phase2_test_inputs(smoke=False)

    def first_child(root: Path, child_guard: dict[str, float]) -> Campaign:
        campaign = Campaign(root)
        register_phase2_test_plan(campaign, plan, blueprint, phase1_decision)
        source = campaign.plan.stages[-1]
        for cell in source.cells:
            qualify_cell(
                campaign,
                cell.cell_id,
                metric=1.0,
                validation={"negative_mean_raw_fvu": 1.0},
                sharing_guard=guard(concordance=1.0, coverage=1.0),
            )
        selection_path = root / "selections" / "root.json"
        payload = campaign.select_stage(source.name, out=selection_path)
        selection = FrozenSelection.from_dict(payload["selected"][0])
        campaign.extend(
            materialize_child_plan(campaign.plan, blueprint, selection),
            selection=selection,
            selection_path=selection_path,
        )
        for cell in campaign.plan.stages[-1].cells:
            qualify_cell(
                campaign,
                cell.cell_id,
                metric=1.0,
                validation={"negative_mean_raw_fvu": 1.0},
                sharing_guard=child_guard,
            )
        return campaign

    redundant = first_child(
        tmp_path / "redundant",
        guard(concordance=1.0, coverage=1.0),
    )
    payload = redundant.select_stage(redundant.plan.stages[-1].name)
    assert payload["selected"]
    measurements = payload["ranked_candidates"][0]["observations"][0]["sharing_guard"][
        "measurements"
    ]
    assert measurements["all_view_fvu_advantage_descriptive"] == 0.0

    rotated = first_child(
        tmp_path / "rotated",
        guard(concordance=0.49, coverage=1.0),
    )
    with pytest.raises(CampaignError, match="no seed-complete"):
        rotated.select_stage(rotated.plan.stages[-1].name)

    empty = first_child(
        tmp_path / "empty-intersection",
        guard(concordance=1.0, coverage=0.0),
    )
    with pytest.raises(CampaignError, match="no seed-complete"):
        empty.select_stage(empty.plan.stages[-1].name)


def test_recomputed_sharing_guard_rejects_forged_measurements() -> None:
    policy = next(
        round_spec.selection_policy
        for round_spec in build_phase2_blueprint().rounds
        if round_spec.selection_policy is not None
        and round_spec.selection_policy.require_sharing_guard
    )
    assert policy is not None
    metrics = {
        "site_only_heldout_fvu_mean": 0.30,
        "leave_one_out_heldout_fvu_mean": 0.25,
        "site_only_support_iou_mean": 0.90,
        "leave_one_out_support_iou_mean": 0.90,
        "all_site_fvu_mean": 0.30,
        "site_only_coordinate_concordance_mean": 0.90,
        "leave_one_out_coordinate_concordance_mean": 0.85,
        "site_only_coordinate_concordance_min": 0.90,
        "leave_one_out_coordinate_concordance_min": 0.85,
        "site_only_intersection_recall_mean": 0.90,
        "leave_one_out_intersection_recall_mean": 0.90,
        "site_only_intersection_recall_min": 0.90,
        "leave_one_out_intersection_recall_min": 0.90,
        "site_only_intersection_energy_coverage_mean": 0.90,
        "leave_one_out_intersection_energy_coverage_mean": 0.90,
        "site_only_intersection_energy_coverage_min": 0.90,
        "leave_one_out_intersection_energy_coverage_min": 0.90,
    }
    guard = Campaign._sharing_guard_payload(metrics, metrics, metrics, policy)
    forged = copy.deepcopy(guard)
    forged["measurements"]["site_only_coordinate_concordance"] = 0.99
    with pytest.raises(
        CampaignError,
        match="not recomputed from authenticated metrics",
    ):
        Campaign._validate_recomputed_sharing_guard(
            forged, metrics, metrics, metrics, policy
        )


def test_confirmation_score_noninferiority_boundary_is_seedwise(tmp_path, monkeypatch):
    blueprint = build_phase2_blueprint(smoke=False)
    extended = build_phase2_plan(smoke=False)
    while extended.stages[-1].name != "confirmation_16m":
        source = extended.stages[-1]
        assert source.selection_policy is not None
        groups: dict[str, list[CellSpec]] = {}
        for cell in source.cells:
            if (
                source.selection_policy.eligible_recipe_names
                and cell.recipe_name
                not in source.selection_policy.eligible_recipe_names
            ):
                continue
            groups.setdefault(cell.candidate_id, []).append(cell)
        selected = tuple(sorted(groups[sorted(groups)[0]], key=lambda cell: cell.seed))
        assert tuple(cell.seed for cell in selected) == (0, 1)
        selection = FrozenSelection.from_cells(
            source.selection_policy,
            selected,
            tuple(1.0 for _ in selected),
            tuple("sha256:" + "1" * 64 for _ in selected),
            "sha256:" + "2" * 64,
        )
        extended = materialize_child_plan(extended, blueprint, selection)

    confirmation_stage = extended.stages[-1]
    source_cells = tuple(
        sorted(
            (
                cell
                for cell in confirmation_stage.cells
                if cell.decision_map["data.normalization"] == "scalar_rms"
            ),
            key=lambda cell: cell.seed,
        )
    )
    assert tuple(cell.seed for cell in source_cells) == (0, 1)
    development_stage = extended.stages[-2]
    assert development_stage.selection_policy is not None
    plan_cells = {cell.cell_id: cell for cell in extended.cells}
    parents = tuple(
        next(
            plan_cells[str(parent_id)]
            for parent_id in cell.decision_map["selection.parent_cell_ids"]
            if plan_cells[str(parent_id)].seed == cell.seed
        )
        for cell in source_cells
    )
    campaign = Campaign(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    campaign.plan_path.write_text(
        json.dumps(extended.to_manifest(), indent=2, sort_keys=True) + "\n"
    )

    def record_for(cell: CellSpec, score: float) -> CampaignRecord:
        metrics = {
            "validation": {"negative_mean_raw_fvu": score},
            "sharing_guard": {
                "all_site_fvu_mean": 0.10,
                "site_only_heldout_fvu_mean": 0.30,
                "leave_one_out_heldout_fvu_mean": 0.25,
                "site_only_support_iou_mean": 0.80,
                "leave_one_out_support_iou_mean": 0.75,
                "site_only_coordinate_concordance_mean": 0.90,
                "leave_one_out_coordinate_concordance_mean": 0.85,
                "site_only_coordinate_concordance_min": 0.90,
                "leave_one_out_coordinate_concordance_min": 0.85,
                "site_only_intersection_recall_mean": 0.80,
                "leave_one_out_intersection_recall_mean": 0.75,
                "site_only_intersection_recall_min": 0.80,
                "leave_one_out_intersection_recall_min": 0.75,
                "site_only_intersection_energy_coverage_mean": 0.95,
                "leave_one_out_intersection_energy_coverage_mean": 0.92,
                "site_only_intersection_energy_coverage_min": 0.95,
                "leave_one_out_intersection_energy_coverage_min": 0.92,
            },
        }
        payload = {
            "schema": QUALIFICATION_SCHEMA,
            "cell_id": cell.cell_id,
            "scientific_outcome": {"passed": True},
            "selection_metrics": metrics,
            "selection_metrics_sha256": hashlib.sha256(
                canonical_json(metrics).encode()
            ).hexdigest(),
            "promotion_eligible": True,
            "selection_eligibility_mode": "scientific_promotion",
        }
        ref = write_cell_artifact(campaign, cell.cell_id, "qualification", payload)
        return CampaignRecord(
            cell_id=cell.cell_id,
            state=RunState.QUALIFIED,
            artifacts=(ref,),
        )

    records = {parent.cell_id: record_for(parent, 1.0) for parent in parents}
    records.update({cell.cell_id: record_for(cell, 0.98) for cell in source_cells})
    monkeypatch.setattr(
        campaign,
        "_sharing_guard_result",
        lambda *_args, **_kwargs: {"passed": True},
    )
    evidence = campaign._confirmation_noninferiority_evidence(
        source_cells,
        records,
        development_stage.selection_policy,
        smoke=False,
    )
    assert [row["score_degradation"] for row in evidence["per_seed"]] == (
        pytest.approx([0.02, 0.02])
    )
    assert evidence["passed"] is True
    sensitivity = evidence["score_degradation_sensitivity"]
    assert [row["threshold"] for row in sensitivity["rows"]] == [0.01, 0.02, 0.05]
    assert [row["passed_all_seeds"] for row in sensitivity["rows"]] == [
        False,
        True,
        True,
    ]
    assert sensitivity["ungated_passed_all_seeds"] is True

    failing_cell = source_cells[-1]
    records[failing_cell.cell_id] = record_for(failing_cell, 0.979)
    evidence = campaign._confirmation_noninferiority_evidence(
        source_cells,
        records,
        development_stage.selection_policy,
        smoke=False,
    )
    assert evidence["per_seed"][-1]["score_noninferiority_passed"] is False
    assert evidence["passed"] is False


def _phase2_campaign_at_main_stage(
    root: Path,
    target_stage: str,
) -> Campaign:
    plan, blueprint, phase1_decision = phase2_test_inputs(smoke=False)
    campaign = Campaign(root)
    register_phase2_test_plan(
        campaign,
        plan,
        blueprint,
        phase1_decision,
    )
    while campaign.plan.stages[-1].name != target_stage:
        stage = campaign.plan.stages[-1]
        for index, cell in enumerate(stage.cells):
            qualify_cell(
                campaign,
                cell.cell_id,
                metric=float(index + 1),
                validation={"negative_mean_raw_fvu": float(index + 1)},
            )
        selection_path = root / "selections" / f"{stage.name}.json"
        payload = campaign.select_stage(stage.name, out=selection_path)
        selection = FrozenSelection.from_dict(payload["selected"][0])
        campaign.extend(
            materialize_child_plan(campaign.plan, blueprint, selection),
            selection=selection,
            selection_path=selection_path,
        )
    return campaign


def test_factorization_carrier_failure_retains_only_exact_parent_control(tmp_path):
    campaign = _phase2_campaign_at_main_stage(
        tmp_path,
        "site_factorization_4m",
    )
    factorization = campaign.plan.stages[-1]
    for cell in factorization.cells:
        metric = (
            1.0
            if cell.recipe_name.endswith("selected_parent_carrier")
            else 0.0
            if cell.recipe_name.endswith("site_rank_full")
            else 10.0
        )
        qualify_cell(
            campaign,
            cell.cell_id,
            metric=metric,
            validation={"negative_mean_raw_fvu": metric},
        )
    payload = campaign.select_stage(factorization.name)
    assert len(payload["selected"]) == 1
    selected = FrozenSelection.from_dict(payload["selected"][0])
    selected_cell = next(
        cell for cell in factorization.cells if cell.cell_id in selected.cell_ids
    )
    assert selected_cell.recipe_name.endswith("selected_parent_carrier")
    assert all(
        item.get("reason") == "required_carrier_noninferiority_failed"
        for item in payload["excluded_candidates"]
        if item.get("recipe_name", "").startswith("derived_site_factorization_4m_")
    )
    gate = payload["ranked_candidates"][0]["noninferiority_gate"]
    assert gate["passed"] is False
    assert gate["per_seed"][0]["degradation"] == 1.0


def test_factorization_selection_uses_lowest_seedwise_noninferior_rank(tmp_path):
    campaign = _phase2_campaign_at_main_stage(
        tmp_path,
        "site_factorization_4m",
    )
    factorization = campaign.plan.stages[-1]
    metrics = {
        "selected_parent_carrier": 1.0,
        "site_rank_1": 0.989,
        "site_rank_2": 0.99,
        "site_rank_4": 1.10,
        "site_rank_full": 1.0,
    }
    for cell in factorization.cells:
        variant = cell.recipe_name.removeprefix("derived_site_factorization_4m_")
        qualify_cell(
            campaign,
            cell.cell_id,
            metric=metrics[variant],
            validation={"negative_mean_raw_fvu": metrics[variant]},
        )

    payload = campaign.select_stage(factorization.name)

    assert len(payload["selected"]) == 1
    selected = FrozenSelection.from_dict(payload["selected"][0])
    selected_cell = next(
        cell for cell in factorization.cells if cell.cell_id in selected.cell_ids
    )
    assert selected_cell.recipe_name.endswith("site_rank_2")
    gate = payload["ranked_candidates"][0]["parsimony_gate"]
    assert gate["selected_variant"] == "site_rank_2"
    assert gate["comparisons"][0]["passed"] is False
    assert gate["comparisons"][1]["worst_seed_degradation"] == pytest.approx(0.01)
    assert gate["comparisons"][2]["passed"] is True


def _gate_candidate(
    stage: str,
    variant: str,
    metrics: tuple[float, ...],
) -> dict[str, object]:
    return {
        "candidate_id": f"candidate:{variant}",
        "recipe_name": f"derived_{stage}_{variant}",
        "recipe_id": f"recipe:{variant}",
        "median": float(median(metrics)),
        "worst_seed": min(metrics),
        "observations": [
            {
                "cell_id": f"cell:{variant}:{seed}",
                "seed": seed,
                "metric": metric,
            }
            for seed, metric in enumerate(metrics)
        ],
    }


def test_minimum_effect_gate_retains_parent_until_seedwise_and_aggregate_boundary(
    tmp_path,
):
    base = build_phase2_plan(smoke=False).stages[-1].selection_policy
    assert isinstance(base, SelectionPolicy)
    policy = replace(
        base,
        default_parent_variant="selected_parent",
        minimum_effect_absolute=0.002,
        minimum_effect_reduction="per_seed_and_median_and_worst",
        threshold_basis=SELECTION_THRESHOLD_BASIS,
        threshold_sensitivity=SELECTION_THRESHOLD_SENSITIVITY,
    )
    stage = "minimum_effect_test"
    parent = _gate_candidate(stage, "selected_parent", (1.0, 1.0))
    boundary = _gate_candidate(stage, "boundary", (1.002, 1.002))
    fragile = _gate_candidate(stage, "fragile", (1.003, 1.001))
    excluded: list[dict[str, object]] = []

    retained = Campaign(tmp_path)._apply_minimum_effect_gate(
        stage,
        policy,
        [parent, boundary, fragile],
        excluded,
    )

    assert [item["candidate_id"] for item in retained] == [
        "candidate:selected_parent",
        "candidate:boundary",
    ]
    gate = boundary["minimum_effect_gate"]
    assert gate["median_improvement"] == pytest.approx(0.002)
    assert gate["worst_seed_improvement"] == pytest.approx(0.002)
    assert gate["passed"] is True
    assert fragile["minimum_effect_gate"]["median_passed"] is True
    assert fragile["minimum_effect_gate"]["per_seed"][1]["passed"] is False
    assert excluded[0]["reason"] == ("minimum_effect_not_met_against_default_parent")


def test_family_nomination_deduplicates_before_outcome_ranking():
    earlier = {
        "candidate_id": "candidate:" + "1" * 64,
        "source_stage": "round_a",
        "execution_signature": "a" * 64,
        "median": 0.1,
        "worst_seed": 0.1,
        "observations": [{"seed": 0, "metric": 0.1}],
    }
    later_better_repeat = {
        "candidate_id": "candidate:" + "2" * 64,
        "source_stage": "round_b",
        "execution_signature": "a" * 64,
        "median": 0.9,
        "worst_seed": 0.9,
        "observations": [{"seed": 0, "metric": 0.9}],
    }
    distinct, duplicates = Campaign._deduplicate_family_nomination_candidates(
        [later_better_repeat, earlier],
        ("round_a", "round_b"),
    )
    assert [item["candidate_id"] for item in distinct] == [earlier["candidate_id"]]
    assert duplicates[0]["candidate_id"] == later_better_repeat["candidate_id"]
    spread = distinct[0]["execution_alias_metric_spread"]
    assert spread["median_max_minus_min"] == pytest.approx(0.8)
    assert spread["worst_seed_max_minus_min"] == pytest.approx(0.8)
    assert spread["maximum_per_seed_max_minus_min"] == pytest.approx(0.8)


def test_policy_retained_cutoff_ties_are_identical_at_selection_and_replay():
    base = build_phase1_plan(smoke=False).stages[-1].selection_policy
    assert isinstance(base, SelectionPolicy)
    policy = replace(base, tie_policy="retain_all_at_cutoff", retain_count=1)
    candidates = [
        {"candidate_id": "candidate:a", "median": 1.0, "worst_seed": 0.5},
        {"candidate_id": "candidate:b", "median": 1.0, "worst_seed": 0.5},
        {"candidate_id": "candidate:c", "median": 0.9, "worst_seed": 0.4},
    ]
    retained = campaign_module._policy_retained_candidates(
        candidates,
        policy,
        smoke_protocol_only=False,
    )
    assert [item["candidate_id"] for item in retained] == [
        "candidate:a",
        "candidate:b",
    ]
    assert campaign_module._policy_retained_candidates(
        candidates,
        policy,
        smoke_protocol_only=True,
    ) == [candidates[0]]


def test_panel_entry_seed_coverage_is_exact():
    complete = SimpleNamespace(
        source_cells=(SimpleNamespace(seed=0), SimpleNamespace(seed=1))
    )
    campaign_module._validate_panel_entry_seed_coverage((complete,), (0, 1))
    missing = SimpleNamespace(source_cells=(SimpleNamespace(seed=0),))
    with pytest.raises(CampaignError, match="exactly cover"):
        campaign_module._validate_panel_entry_seed_coverage((missing,), (0, 1))


def test_confirmation_guard_reuse_requires_exact_structure():
    expected = {
        "cell_id": "cell:current",
        "parent_cell_id": "cell:parent",
        "checks": {"sharing": True},
    }
    campaign_module._validate_exact_confirmation_guard(copy.deepcopy(expected), expected)
    with pytest.raises(CampaignError, match="exact authenticated sharing guard"):
        campaign_module._validate_exact_confirmation_guard(
            {**expected, "unbound_summary": True},
            expected,
        )


def test_rank_parsimony_chooses_lowest_seedwise_noninferior_rank(tmp_path):
    base = build_phase2_plan(smoke=False).stages[-1].selection_policy
    assert isinstance(base, SelectionPolicy)
    policy = replace(
        base,
        required_control_variant="selected_parent_carrier",
        noninferiority_candidate_variant="site_rank_full",
        control_noninferiority_absolute_tolerance=0.01,
        parsimony_order_variants=(
            "site_rank_1",
            "site_rank_2",
            "site_rank_4",
            "site_rank_full",
        ),
        parsimony_noninferiority_absolute_tolerance=0.01,
        parsimony_reduction="per_seed_and_median_and_worst",
        threshold_basis=SELECTION_THRESHOLD_BASIS,
        threshold_sensitivity=SELECTION_THRESHOLD_SENSITIVITY,
    )
    stage = "site_factorization_test"
    candidates = [
        _gate_candidate(stage, "selected_parent_carrier", (1.0, 1.0)),
        _gate_candidate(stage, "site_rank_1", (0.989, 1.0)),
        _gate_candidate(stage, "site_rank_2", (0.99, 0.99)),
        _gate_candidate(stage, "site_rank_4", (1.01, 1.01)),
        _gate_candidate(stage, "site_rank_full", (1.0, 1.0)),
    ]
    for candidate in candidates:
        candidate["noninferiority_gate"] = {"passed": True}
    excluded: list[dict[str, object]] = []

    retained = Campaign(tmp_path)._apply_rank_parsimony_gate(
        stage,
        policy,
        candidates,
        excluded,
    )

    assert [item["candidate_id"] for item in retained] == ["candidate:site_rank_2"]
    gate = retained[0]["parsimony_gate"]
    assert gate["selected_variant"] == "site_rank_2"
    assert gate["comparisons"][0]["passed"] is False
    assert gate["comparisons"][1]["worst_seed_degradation"] == pytest.approx(0.01)
    assert gate["comparisons"][1]["passed"] is True
    assert {item["reason"] for item in excluded} == {
        "carrier_control_only_after_noninferiority_pass",
        "site_rank_parsimony_not_selected",
    }


def test_advance_rejects_a_selection_after_any_universe_member_changes(tmp_path):
    blueprint = build_phase1_blueprint(seeds=(0, 1), smoke=True)
    plan = build_phase1_plan(seeds=(0, 1), smoke=True)
    campaign = Campaign(tmp_path)
    campaign.register(plan, blueprint_manifest=blueprint.to_manifest())
    stage = plan.stages[-1]
    for index, cell in enumerate(stage.cells):
        qualify_cell(campaign, cell.cell_id, metric=float(index + 1))
    selection_path = tmp_path / "selections" / f"{stage.name}.json"
    payload = campaign.select_stage(stage.name, out=selection_path)
    selection = FrozenSelection.from_dict(payload["selected"][0])
    extended = materialize_child_plan(plan, blueprint, selection)
    selected_cell = next(
        cell for cell in stage.cells if cell.cell_id in set(selection.cell_ids)
    )
    campaign.transition(
        selected_cell.cell_id, RunState.FAILED, message="late invalidation"
    )
    with pytest.raises(CampaignError):
        campaign.extend(
            extended,
            selection=selection,
            selection_path=selection_path,
        )


def test_advance_rechecks_cumulative_storage_before_appending(
    tmp_path, monkeypatch, capsys
):
    blueprint = build_phase1_blueprint(seeds=(0,), smoke=True)
    plan = build_phase1_plan(seeds=(0,), smoke=True)
    campaign = Campaign(tmp_path)
    campaign.register(plan, blueprint_manifest=blueprint.to_manifest())
    stage = plan.stages[-1]
    for index, cell in enumerate(stage.cells):
        qualify_cell(campaign, cell.cell_id, metric=float(index + 1))
    selection_path = tmp_path / "selections" / f"{stage.name}.json"
    campaign.select_stage(stage.name, out=selection_path)
    monkeypatch.setattr(
        "block_crosscoder_experiment.cli.matrix.shutil.disk_usage",
        lambda _path: SimpleNamespace(free=0, total=1, used=1),
    )
    with pytest.raises(SystemExit) as exc_info:
        matrix_main(
            [
                "advance",
                "--root",
                str(tmp_path),
                "--selection",
                str(selection_path),
            ]
        )
    assert exc_info.value.code == 2
    assert "conservative cumulative estimate" in capsys.readouterr().err
    assert campaign.plan == plan
    matrix_main(
        [
            "advance",
            "--root",
            str(tmp_path),
            "--selection",
            str(selection_path),
            "--allow-insufficient-local-storage",
        ]
    )
    assert Campaign(tmp_path).plan.stages[-1].name == blueprint.rounds[0].name


def test_phase2_freeze_builds_a_verified_seed_complete_phase3_panel(
    tmp_path, monkeypatch
):
    project_configurations = Campaign._projected_scientific_configurations
    forced_group_lasso_candidate_id = None

    def force_one_real_duplicate(decision, *, smoke):
        nonlocal forced_group_lasso_candidate_id
        projected = project_configurations(decision, smoke=smoke)
        entries = {entry.panel_slot: entry for entry in decision.entries}
        group_lasso = entries.get("bsf_group_lasso")
        if group_lasso is not None:
            candidate_id = group_lasso.source_cells[0].candidate_id
            if forced_group_lasso_candidate_id is None:
                forced_group_lasso_candidate_id = candidate_id
            if candidate_id == forced_group_lasso_candidate_id:
                projected["bsf_group_lasso"] = projected["bsf_grassmannian"]
        return projected

    monkeypatch.setattr(
        Campaign,
        "_projected_scientific_configurations",
        staticmethod(force_one_real_duplicate),
    )
    campaign, blueprint = qualified_phase2_campaign(tmp_path / "phase2")
    payload = campaign.freeze_panel()
    decision = Campaign.panel_decision_from_manifest(payload)
    assert isinstance(decision, FrozenPanelDecision)
    assert decision.source_phase2_plan_id == campaign.plan.plan_id
    assert decision.source_phase2_blueprint_id == blueprint.blueprint_id
    assert len(decision.entries) == 8
    finalist = next(
        entry for entry in decision.entries if entry.role == "selected_finalist"
    )
    assert tuple(cell.seed for cell in finalist.source_cells) == blueprint.seeds
    assert all(
        cell.decision_map["data.normalization"] == "scalar_rms"
        for cell in finalist.source_cells
    )
    assert len(finalist.selection_ids) == len(blueprint.rounds)
    assert len(finalist.confirmation_sha256s) == len(blueprint.seeds)
    comparators = [
        entry for entry in decision.entries if entry.role != "selected_finalist"
    ]
    assert len(comparators) == 7
    assert all(entry.selection_ids for entry in comparators)
    assert all(entry.recipe_id.startswith("derived-recipe:") for entry in comparators)
    assert all(not entry.confirmation_sha256s for entry in comparators)
    substitutions = payload["phase2_campaign_manifest"]["duplicate_substitutions"]
    assert len(substitutions) == 1
    assert substitutions[0]["panel_slot"] == "bsf_group_lasso"
    assert substitutions[0]["substitute_rank"] == 2
    assert substitutions[0]["policy"] == "next_ranked_nonduplicate"

    for field, value, error in (
        ("qualified", False, "does not approve this cell"),
        ("inputs", {}, "exact input-hash set"),
    ):
        qualification_forgery = copy.deepcopy(payload)
        evidence = qualification_forgery["phase2_campaign_manifest"]["cells"][0]
        evidence["qualification"][field] = value
        qualification_ref = next(
            item for item in evidence["artifacts"] if item["kind"] == "qualification"
        )
        qualification_ref["sha256"] = qualification_file_sha256(
            evidence["qualification"]
        ).removeprefix("sha256:")
        rehash_panel_decision(qualification_forgery)
        with pytest.raises(
            CampaignError,
            match=f"qualification semantic replay.*{error}",
        ):
            Campaign.panel_decision_from_manifest(qualification_forgery)

    runner_up_forgery = copy.deepcopy(payload)
    forged_universe = runner_up_forgery["selection_universe"]
    forged_manifest = runner_up_forgery["phase2_campaign_manifest"]
    forged_index = next(
        index
        for index, row in enumerate(forged_universe["selection_chain"])
        if row["target_plan_id"] is None and row["family_name"] is not None
    )
    forged_row = forged_universe["selection_chain"][forged_index]
    ranked_universe = forged_universe["ranked_stage_universes"][forged_index]
    runner_up = ranked_universe["ranked_candidates"][1]
    cells_by_id = {
        item["cell_id"]: CellSpec.from_manifest(item["cell"])
        for item in forged_manifest["cells"]
    }
    runner_cells = [cells_by_id[item["cell_id"]] for item in runner_up["observations"]]
    runner_selection = FrozenSelection.from_cells(
        SelectionPolicy.from_dict(ranked_universe["policy"]),
        runner_cells,
        [float(item["metric"]) for item in runner_up["observations"]],
        [str(item["qualification_sha256"]) for item in runner_up["observations"]],
        str(ranked_universe["selection_universe_sha256"]),
    )
    replacement_row = {
        **forged_row,
        "selection_id": runner_selection.selection_id,
        "candidate_id": runner_selection.candidate_id,
        "cell_ids": list(runner_selection.cell_ids),
        "qualification_sha256s": list(runner_selection.qualification_sha256s),
    }
    family_name = str(forged_row["family_name"])
    for manifest in (forged_universe, forged_manifest):
        manifest["selection_chain"][forged_index] = replacement_row
        family_rows = manifest["family_selection_chains"][family_name]
        family_index = next(
            index
            for index, row in enumerate(family_rows)
            if row["source_stage"] == forged_row["source_stage"]
        )
        family_rows[family_index] = replacement_row
    rehash_panel_decision(runner_up_forgery)
    with pytest.raises(CampaignError, match="policy-retained candidate"):
        Campaign.panel_decision_from_manifest(runner_up_forgery)

    excluded_winner_forgery = copy.deepcopy(payload)
    forged_universe = excluded_winner_forgery["selection_universe"]
    forged_manifest = excluded_winner_forgery["phase2_campaign_manifest"]
    ranked_universe = next(
        item
        for item in forged_universe["ranked_stage_universes"]
        if len(item["ranked_candidates"]) > 1
        and not any(
            row["source_plan_id"] == item["source_plan_id"]
            and row["source_stage"] == item["source_stage"]
            and row["branch"] == "comparator_family_duplicate_substitute"
            for row in forged_universe["selection_chain"]
        )
    )
    old_universe_sha256 = ranked_universe["selection_universe_sha256"]
    removed_winner = ranked_universe["ranked_candidates"].pop(0)
    ranked_universe["excluded_candidates"].append(
        {**removed_winner, "reason": "forged_winner_exclusion"}
    )
    stage_universe_body = {
        "plan_id": ranked_universe["source_plan_id"],
        "source_stage": ranked_universe["source_stage"],
        "policy_id": ranked_universe["policy"]["policy_id"],
        "ranked_candidates": ranked_universe["ranked_candidates"],
        "excluded_candidates": ranked_universe["excluded_candidates"],
    }
    forged_universe_sha256 = (
        "sha256:"
        + hashlib.sha256(canonical_json(stage_universe_body).encode()).hexdigest()
    )
    ranked_universe["selection_universe_sha256"] = forged_universe_sha256
    for manifest in (forged_universe, forged_manifest):
        for row in manifest["selection_chain"]:
            if row["selection_universe_sha256"] == old_universe_sha256:
                row["selection_universe_sha256"] = forged_universe_sha256
        for row in manifest["main_selection_chain"]:
            if row["selection_universe_sha256"] == old_universe_sha256:
                row["selection_universe_sha256"] = forged_universe_sha256
        for family_rows in manifest["family_selection_chains"].values():
            for row in family_rows:
                if row["selection_universe_sha256"] == old_universe_sha256:
                    row["selection_universe_sha256"] = forged_universe_sha256
    rehash_panel_decision(excluded_winner_forgery)
    with pytest.raises(CampaignError, match="authenticated eligibility replay"):
        Campaign.panel_decision_from_manifest(excluded_winner_forgery)

    nomination_metric_forgery = copy.deepcopy(payload)
    forged_universe = nomination_metric_forgery["selection_universe"]
    forged_nomination = forged_universe["family_nominations"][0]
    nomination_payload = forged_nomination["nomination_payload"]
    nomination_payload["ranked_candidates"][0]["median"] = 123.0
    forged_nomination["ranked_candidates"] = copy.deepcopy(
        nomination_payload["ranked_candidates"]
    )
    nomination_body = dict(nomination_payload)
    nomination_body.pop("nomination_id")
    forged_nomination_id = content_id(nomination_body, prefix="family-nomination")
    nomination_payload["nomination_id"] = forged_nomination_id
    forged_nomination["nomination_id"] = forged_nomination_id
    nomination_metric_forgery["phase2_campaign_manifest"]["family_nominations"] = (
        copy.deepcopy(forged_universe["family_nominations"])
    )
    rehash_panel_decision(nomination_metric_forgery)
    with pytest.raises(CampaignError, match="cross-round replay"):
        Campaign.panel_decision_from_manifest(nomination_metric_forgery)

    reordered = copy.deepcopy(payload)
    reordered_entry = next(
        entry for entry in reordered["entries"] if entry["role"] == "selected_finalist"
    )
    reordered_entry["selection_ids"] = list(reversed(reordered_entry["selection_ids"]))
    reordered_manifest_entry = next(
        entry
        for entry in reordered["phase2_campaign_manifest"]["panel_entries"]
        if entry["role"] == "selected_finalist"
    )
    reordered_manifest_entry["selection_ids"] = list(reordered_entry["selection_ids"])
    rehash_panel_decision(reordered)
    with pytest.raises(CampaignError, match="finalist selection chain"):
        Campaign.panel_decision_from_manifest(reordered)

    forged_target = copy.deepcopy(payload)
    first_selection_id = forged_target["selection_universe"]["main_selection_chain"][0][
        "selection_id"
    ]
    for manifest_name in ("phase2_campaign_manifest", "selection_universe"):
        manifest = forged_target[manifest_name]
        for chain_name in ("selection_chain", "main_selection_chain"):
            for chain_item in manifest[chain_name]:
                if chain_item["selection_id"] == first_selection_id:
                    chain_item["target_plan_id"] = "study:" + "f" * 64
    rehash_panel_decision(forged_target)
    with pytest.raises(CampaignError, match="target differs from blueprint replay"):
        Campaign.panel_decision_from_manifest(forged_target)

    stale_phase1 = copy.deepcopy(payload)
    stale_phase1["phase2_campaign_manifest"]["phase1_decision_sha256"] = (
        "sha256:" + "f" * 64
    )
    rehash_panel_decision(stale_phase1)
    with pytest.raises(CampaignError, match="Phase-1 decision-file hash"):
        Campaign.panel_decision_from_manifest(stale_phase1)

    for manifest_name, error in (
        ("phase2_campaign_manifest", "campaign manifest.*noncanonical field"),
        ("selection_universe", "selection universe.*noncanonical field"),
    ):
        extra_field = copy.deepcopy(payload)
        extra_field[manifest_name]["future_unbound_field"] = True
        rehash_panel_decision(extra_field)
        with pytest.raises(CampaignError, match=error):
            Campaign.panel_decision_from_manifest(extra_field)

    forged_parent = copy.deepcopy(payload)
    confirmation = forged_parent["phase2_campaign_manifest"][
        "confirmation_noninferiority"
    ]
    row = confirmation["per_seed"][0]
    wrong_parent = next(
        evidence
        for evidence in forged_parent["phase2_campaign_manifest"]["cells"]
        if evidence["seed"] == row["seed"]
        and evidence["cell_id"] not in {row["cell_id"], row["parent_cell_id"]}
    )
    wrong_parent_qualification = next(
        artifact
        for artifact in wrong_parent["artifacts"]
        if artifact["kind"] == "qualification"
    )
    row["parent_cell_id"] = wrong_parent["cell_id"]
    row["parent_qualification_sha256"] = (
        "sha256:" + wrong_parent_qualification["sha256"]
    )
    forged_parent["selection_universe"]["confirmation_noninferiority"] = (
        copy.deepcopy(confirmation)
    )
    rehash_panel_decision(forged_parent)
    with pytest.raises(CampaignError, match="parent binding mismatch"):
        Campaign.panel_decision_from_manifest(forged_parent)

    phase3_blueprint = build_phase3_blueprint(
        seeds=(0,), smoke=True, panel_decision=decision
    )
    phase3_plan = build_phase3_plan(
        seeds=phase3_blueprint.seeds,
        smoke=True,
        panel_decision=decision,
    )
    phase3 = Campaign(tmp_path / "phase3")
    phase3.register(
        phase3_plan,
        blueprint_manifest=phase3_blueprint.to_manifest(),
        panel_decision_manifest=payload,
    )
    assert phase3.plan == phase3_plan
    assert phase3.panel_decision_path.is_file()
    preview_blueprint = build_phase3_blueprint(seeds=(0,), smoke=True)
    with pytest.raises(CampaignError, match="plan/blueprint differs"):
        Campaign(tmp_path / "preview-blueprint").register(
            phase3_plan,
            blueprint_manifest=preview_blueprint.to_manifest(),
            panel_decision_manifest=payload,
        )
    with pytest.raises(CampaignError, match="campaign-evidence-bound panel decision"):
        Campaign(tmp_path / "unverified").register(
            phase3_plan,
            blueprint_manifest=phase3_blueprint.to_manifest(),
        )
    with pytest.raises(CampaignError, match="freeze evidence envelope"):
        Campaign.panel_decision_from_manifest(decision.to_dict())
    production_blueprint = build_phase3_blueprint(smoke=False)
    production_plan = build_phase3_plan(smoke=False, panel_decision=decision)
    with pytest.raises(CampaignError, match="smoke Phase-2 panel"):
        Campaign(tmp_path / "smoke-escalation").register(
            production_plan,
            blueprint_manifest=production_blueprint.to_manifest(),
            panel_decision_manifest=payload,
        )


def test_phase2_freeze_rejects_negative_missing_and_stale_evidence(tmp_path):
    source, _ = qualified_phase2_campaign(tmp_path / "source", seeds=(0,))

    negative_root = tmp_path / "negative"
    shutil.copytree(source.root, negative_root)
    negative = Campaign(negative_root)
    finalist_cell = next(
        cell
        for cell in negative.plan.cells
        if cell.decision_map["data.normalization"] == "scalar_rms"
        and cell.decision_map["evaluation.split"] == "confirmation"
    )
    qualification_ref = negative.record(finalist_cell.cell_id).artifact_map[
        "qualification"
    ]
    qualification_path = qualification_ref.resolve(negative.root)
    qualification_payload = json.loads(qualification_path.read_text())
    qualification_payload["scientific_outcome"]["passed"] = False
    qualification_payload["scientific_outcome"]["checks"][
        "support_target_calibration"
    ] = False
    qualification_payload["scientific_outcome"]["margins"][
        "support_target_abs_error"
    ] = -0.1
    qualification_payload["promotion_eligible"] = False
    evaluation_path = negative.record(finalist_cell.cell_id).artifact_map[
        "evaluation"
    ].resolve(negative.root)
    qualification_payload["promotion_ineligible_reasons"] = (
        campaign_module._promotion_reasons_from_evidence(
            finalist_cell,
            outcome_passed=False,
            evaluation=json.loads(evaluation_path.read_text()),
        )
    )
    qualification_path.write_text(
        json.dumps(
            qualification_payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    qualification_sha256 = hashlib.sha256(qualification_path.read_bytes()).hexdigest()
    qualification_size = qualification_path.stat().st_size
    rewritten_events = []
    for event in negative.events():
        if (
            event.get("event") == "transition"
            and event.get("cell_id") == finalist_cell.cell_id
            and event.get("target") == RunState.QUALIFIED.value
        ):
            for artifact in event["artifacts"]:
                if artifact["kind"] == "qualification":
                    artifact["sha256"] = qualification_sha256
                    artifact["size_bytes"] = qualification_size
        rewritten_events.append(event)
    negative.journal_path.write_text(
        "".join(canonical_json(event) + "\n" for event in rewritten_events)
    )
    protocol_panel = negative.freeze_panel()
    assert protocol_panel["phase2_campaign_manifest"]["smoke"] is True

    incomplete_root = tmp_path / "incomplete"
    shutil.copytree(source.root, incomplete_root)
    incomplete = Campaign(incomplete_root)
    omitted = incomplete.plan.stages[-1].cells[-1]
    incomplete_events = [
        event
        for event in incomplete.events()
        if not (
            event.get("event") == "transition"
            and event.get("cell_id") == omitted.cell_id
            and event.get("target") == RunState.QUALIFIED.value
        )
    ]
    incomplete.journal_path.write_text(
        "".join(canonical_json(event) + "\n" for event in incomplete_events)
    )
    with pytest.raises(CampaignError, match="every materialized cell"):
        incomplete.freeze_panel()

    stale_root = tmp_path / "stale"
    shutil.copytree(source.root, stale_root)
    stale = Campaign(stale_root)
    comparator = next(
        cell
        for cell in stale.plan.stages[0].cells
        if cell.recipe_name == "control_scalar_relu_batchtopk"
    )
    qualification = stale.record(comparator.cell_id).artifact_map["qualification"]
    qualification.resolve(stale.root).write_text("tampered\n")
    with pytest.raises(ArtifactError, match="mismatch"):
        stale.freeze_panel(out=stale.root / "decisions" / "tampered.json")

    altered_root = tmp_path / "altered"
    shutil.copytree(source.root, altered_root)
    campaign = Campaign(altered_root)
    campaign.freeze_panel()

    original_plan = campaign.plan_path.read_text()
    plan_payload = json.loads(original_plan)
    plan_payload["stages"][-1]["cells"][0]["seed"] += 1
    campaign.plan_path.write_text(json.dumps(plan_payload) + "\n")
    with pytest.raises(CampaignError, match="invalid Phase-2 plan"):
        campaign.freeze_panel(out=tmp_path / "decisions" / "altered-plan.json")
    campaign.plan_path.write_text(original_plan)

    original_blueprint = campaign.blueprint_path.read_text()
    blueprint_payload = json.loads(original_blueprint)
    blueprint_payload["rounds"][0]["train_tokens"] += 1
    campaign.blueprint_path.write_text(json.dumps(blueprint_payload) + "\n")
    with pytest.raises(CampaignError, match="invalid Phase-2 plan or blueprint"):
        campaign.freeze_panel(out=tmp_path / "decisions" / "altered-blueprint.json")
    campaign.blueprint_path.write_text(original_blueprint)

    campaign._append_event(
        campaign._event(
            "lock_reconciled",
            campaign.plan.cells[0].cell_id,
            message="post-freeze journal mutation",
        )
    )
    with pytest.raises(CampaignError, match="immutable decision"):
        campaign.freeze_panel()


def test_gate_basis_separates_integrity_completion_from_scientific_pass(tmp_path):
    assert not hasattr(GateCondition("source"), "metric")
    resolved_first, _ = phase1_selection_template(0, smoke=False)
    resolved_second, selection_policy = phase1_selection_template(1, smoke=False)
    first = CellSpec(
        "phase1.integrity.source.s0",
        Phase.PHASE1,
        "source",
        resolved_first.recipe_name,
        resolved_first.recipe_id,
        0,
        resolved_first.decisions,
    )
    child = CellSpec(
        "phase1.integrity.child.s1",
        Phase.PHASE1,
        "child",
        resolved_second.recipe_name,
        resolved_second.recipe_id,
        1,
        resolved_second.decisions,
    )
    integrity_plan = StudyPlan(
        "integrity_gate",
        Phase.PHASE1,
        (
            StageSpec("source", (first,)),
            StageSpec(
                "child",
                (child,),
                depends_on=("source",),
                gate=GateCondition(
                    "source", minimum_count=1, basis="integrity_complete"
                ),
                selection_policy=selection_policy,
            ),
        ),
    )
    campaign = Campaign(tmp_path / "integrity")
    register_test_plan(campaign, integrity_plan)
    qualify_cell(campaign, first.cell_id, metric=-1.0, scientific_passed=False)
    assert campaign.stage_open("child")

    scientific_plan = StudyPlan(
        "scientific_gate",
        Phase.PHASE1,
        (
            StageSpec("source", (first,)),
            StageSpec(
                "child",
                (child,),
                depends_on=("source",),
                gate=GateCondition("source", minimum_count=1, basis="qualified"),
                selection_policy=selection_policy,
            ),
        ),
    )
    campaign = Campaign(tmp_path / "scientific")
    register_test_plan(campaign, scientific_plan)
    qualify_cell(campaign, first.cell_id, metric=-1.0, scientific_passed=False)
    assert not campaign.stage_open("child")


FAKE_RUN_CELL = r"""
import argparse
import hashlib
import json
import os
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--cell", type=Path, required=True)
p.add_argument("--stage", required=True)
p.add_argument("--artifacts-out", type=Path, required=True)
p.add_argument("--resume", action="store_true")
a = p.parse_args()
cell = json.loads(a.cell.read_text())
values = {item["name"]: item["value"] for item in cell["decisions"]}
implementation = {
    "executor_schema": "test-executor-v1",
    "executor_process_model": "test-process-model-v1",
    "python_source_sha256": "1" * 64,
    "python_source_files": 1,
    "git_commit": "1" * 40,
    "git_dirty": False,
    "python": "3.12.test",
    "torch": "2.8.0",
    "torch_cuda_build": None,
    "dependencies": {
        "datasets": "test",
        "huggingface-hub": "test",
        "numpy": "test",
        "sae-lens": "test",
        "safetensors": "test",
        "torch": "test",
        "transformers": "test",
    },
}
implementation_sha256 = hashlib.sha256(json.dumps(
    implementation, sort_keys=True, separators=(",", ":")
).encode("utf-8")).hexdigest()
root = Path(os.environ["BSC_CAMPAIGN_ROOT"])
outdir = a.cell.parent / "fake-outputs"
outdir.mkdir(parents=True, exist_ok=True)
artifacts = []
outputs = []
if a.stage != "prepare":
    state = json.loads((a.cell.parent / "state.json").read_text())
    artifacts.extend(state["artifacts"])

stage_kinds = {
    "prepare": ("preparation",),
    "train": ("checkpoint", "training_report"),
    "calibrate": ("calibration", "deployment_codec", "calibration_record"),
    "evaluate": ("deployment_schedules", "evaluation"),
    "qualify": ("qualification",),
}
for kind in stage_kinds[a.stage]:
    target = outdir / f"{kind}.json"
    if kind == "qualification":
        state = json.loads((a.cell.parent / "state.json").read_text())
        refs = {item["kind"]: item for item in state["artifacts"]}
        selection_metrics = {
            "validation": {"fvu": 0.1, "rate_distortion": 0.8}
        }
        evaluation_sha256 = refs["evaluation"]["sha256"]
        scientific_checks = {
            "support_target_calibration": True,
            "codec_calibration_exclusion": True,
            "codec_evaluation_exclusion": True,
            "phase1_identification": True,
            "production_precision_finite": True,
            "production_precision_reconstruction": True,
            "production_precision_support": True,
            "production_fixed_rate_frontier": True,
        }
        payload = {
            "schema": "bsc-qualification-v3",
            "cell_id": cell["cell_id"],
            "qualified": True,
            "checks": {
                "deployment_schedule_integrity": True,
                "encoder_scale_calibration_integrity": True,
                "finite": True,
                "method_endpoints": True,
                "precision_preflight_integrity": True,
                "provenance": True,
                "regularizer_calibration_integrity": True,
                "resource_compliance": True,
                "selection_score_diagnostics_integrity": True,
                "scientific_endpoint_complete": True,
                "split_integrity": True,
            },
            "scientific_outcome": {
                "passed": True,
                "checks": scientific_checks,
                "inapplicable_checks": (
                    {"phase1_identification": (
                        "token_layer_normalization_is_not_a_fixed_linear_factor_map"
                    )}
                    if values["data.normalization"] == "layer"
                    else {}
                ),
                "margins": {
                    "support_target_abs_error": 0.1,
                    "codec_calibration_excluded_fraction": 0.01,
                    "codec_evaluation_excluded_fraction": 0.01,
                    "phase1_native_identification": 0.1,
                    "phase1_deployed_identification": 0.1,
                    "production_precision_reconstruction": None,
                    "production_precision_support_iou": None,
                    "production_fixed_rate_nonzero_endpoints": None,
                },
            },
            "inputs": {name: refs[name]["sha256"] for name in (
                "preparation", "checkpoint", "calibration", "deployment_codec",
                "deployment_schedules", "evaluation"
            )},
            "implementation_identity": implementation,
            "implementation_identity_sha256": implementation_sha256,
            "validation": {"fvu": 0.1, "rate_distortion": 0.8},
            "qualification_profile": values["qualification.profile"],
            "thresholds_version": values["qualification.thresholds_version"],
            "thresholds": {
                "schema": "bsc-integrity-thresholds-2026-07-22.v2",
                "support_target_abs_error_max": 0.1,
                "codec_excluded_calibration_event_fraction_max": 0.01,
                "codec_excluded_evaluation_event_fraction_max": 0.01,
                "probability_metric_range": [0.0, 1.0],
                "required_quantizer_bits": values["codec.quantizer_bits"],
                "phase1_identification_thresholds": values[
                    "qualification.phase1_identification_thresholds"
                ],
                "phase1_identification_enforced": False,
                "phase1_margin_normalization_contract": values[
                    "evaluation.phase1_margin_normalization"
                ],
                "phase1_rank_mismatch_contract": values[
                    "evaluation.rank_mismatch_contract"
                ],
                "phase1_pathology_association_contract": values[
                    "evaluation.pathology_association_contract"
                ],
                "phase1_pathology_strong_association_cutoff": values[
                    "evaluation.pathology_strong_association_cutoff"
                ],
                "phase1_pathology_weak_association_cutoff": values[
                    "evaluation.pathology_weak_association_cutoff"
                ],
                "phase1_pathology_association_cutoff_sensitivity": values[
                    "evaluation.pathology_association_cutoff_sensitivity"
                ],
                "encoder_scale_fit_statistic": values[
                    "model.encoder_scale_fit_statistic"
                ],
                "encoder_scale_fit_solver": values["model.encoder_scale_fit_solver"],
                "encoder_scale_fit_target": values["model.encoder_scale_fit_target"],
                "encoder_scale_fit_tolerance": values[
                    "model.encoder_scale_fit_tolerance"
                ],
                "encoder_scale_fit_max_iterations": values[
                    "model.encoder_scale_fit_max_iterations"
                ],
                "fixed_rate_budget_scale_factor": values[
                    "evaluation.fixed_rate_budget_scale_factor"
                ],
                "fixed_rate_budget_scale_contract": values[
                    "evaluation.fixed_rate_budget_scale_contract"
                ],
                "production_min_nonzero_rate_endpoints": values[
                    "precision.preflight_min_nonzero_rate_endpoints"
                ],
            },
            "selection_metrics": selection_metrics,
            "selection_metrics_sha256": hashlib.sha256(json.dumps(
                selection_metrics, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")).hexdigest(),
            "selection_metrics_evaluation_sha256": evaluation_sha256,
            "promotion_eligible": False,
            "promotion_ineligible_reasons": ["runtime_smoke"],
            "selection_eligible_for_protocol_test": True,
            "selection_eligibility_mode": "smoke_protocol_only",
        }
    elif kind == "evaluation":
        refs = {item["kind"]: item for item in artifacts}
        validation = {"fvu": 0.1, "rate_distortion": 0.8}
        selection_metrics = {"validation": validation}
        payload = {
            "schema": "bsc-evaluation-v2",
            "evaluation_execution_implementation": (
                "fused_deployable_full_view_packet_v2"
            ),
            "cell_id": cell["cell_id"],
            "inputs": {name: refs[name]["sha256"] for name in (
                "checkpoint", "calibration", "deployment_codec",
                "deployment_schedules"
            )},
            "validation": validation,
            "selection_metrics": selection_metrics,
            "selection_metrics_sha256": hashlib.sha256(json.dumps(
                selection_metrics, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")).hexdigest(),
            "raw_space": {"eligible": True},
            "fixed_rate_raw_selection": {"eligible": True},
            "synthetic_recovery": {
                "deployed": {"shared_feature_claim_eligible": True}
            },
        }
    elif kind == "preparation":
        payload = {
            "schema": "bsc-preparation-v3",
            "cell_id": cell["cell_id"],
            "implementation": implementation,
            "implementation_sha256": implementation_sha256,
        }
    else:
        payload = {"cell_id": cell["cell_id"], "stage": a.stage, "resume": a.resume}
    target.write_text(json.dumps(payload, sort_keys=True) + "\n")
    body = target.read_bytes()
    entry = {
        "kind": kind,
        "path": str(target),
        "sha256": hashlib.sha256(body).hexdigest(),
        "size_bytes": len(body),
    }
    artifacts.append(entry)
    outputs.append(entry)

a.artifacts_out.parent.mkdir(parents=True, exist_ok=True)
a.artifacts_out.write_text(json.dumps({
    "schema": "bsc-stage-artifacts-v2",
    "cell_id": cell["cell_id"],
    "stage": a.stage,
    "artifacts": outputs,
}, sort_keys=True) + "\n")
"""


def test_generic_runner_stops_at_evaluation_and_never_conflates_it_with_qualification(
    tmp_path,
):
    module_root = tmp_path / "module"
    module_root.mkdir()
    (module_root / "fake_run_cell.py").write_text(FAKE_RUN_CELL)
    root = tmp_path / "campaign"
    plan = one_cell_plan()
    campaign = Campaign(root)
    register_test_plan(campaign, plan)
    pythonpath = str(module_root)
    if os.environ.get("PYTHONPATH"):
        pythonpath += os.pathsep + os.environ["PYTHONPATH"]
    runner = CampaignRunner(
        campaign,
        module="fake_run_cell",
        env={"PYTHONPATH": pythonpath},
    )
    first = runner.run(limit=1, stop_after="evaluate")
    cell_id = plan.cells[0].cell_id
    assert first.completed_cells == 1
    assert campaign.record(cell_id).state is RunState.EVALUATED
    assert campaign.eligible_for_qualification(cell_id)
    assert not campaign.eligible_for_promotion(cell_id)

    journal_before = campaign.journal_path.read_bytes()
    repeated = runner.run(cell_ids=(cell_id,), stop_after="evaluate")
    assert repeated.completed_cells == 1
    assert campaign.record(cell_id).state is RunState.EVALUATED
    assert campaign.journal_path.read_bytes() == journal_before

    second = runner.run(limit=1)
    assert second.completed_cells == 1
    assert campaign.record(cell_id).state is RunState.QUALIFIED
    assert not campaign.eligible_for_promotion(cell_id)
    commands = [event["message"] for event in campaign.events(cell_id)]
    assert "evaluate stage completed" in commands
    assert "qualify stage completed" in commands


def test_runner_rejects_ambiguous_or_empty_limits(tmp_path):
    plan = one_cell_plan()
    campaign = Campaign(tmp_path)
    register_test_plan(campaign, plan)
    runner = CampaignRunner(campaign)
    with pytest.raises(CampaignError, match="limit must be positive"):
        runner.run(limit=0)
    with pytest.raises(CampaignError, match="combined with explicit cell IDs"):
        runner.run(limit=1, cell_ids=(plan.cells[0].cell_id,))


def test_persistent_worker_owns_and_terminates_its_process_group(tmp_path):
    script = (
        "import subprocess,sys; "
        "child=subprocess.Popen([sys.executable,'-c',"
        "'import time; time.sleep(60)']); "
        "print(child.pid, flush=True); sys.stdin.readline()"
    )
    worker = campaign_module._PersistentCellWorker(
        command=(sys.executable, "-c", script),
        cwd=tmp_path,
        environment=os.environ,
    )
    assert worker._process.stdout is not None
    child_pid = int(worker._process.stdout.readline())
    assert os.getpgid(child_pid) == worker.pgid
    try:
        worker.close()
        deadline = time.time() + 5
        while time.time() < deadline:
            state = subprocess.run(
                ["ps", "-o", "stat=", "-p", str(child_pid)],
                check=False,
                capture_output=True,
                text=True,
            ).stdout.strip()
            if not state or state.startswith("Z"):
                break
            time.sleep(0.05)
        assert not state or state.startswith("Z")
    finally:
        try:
            os.kill(child_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_custom_executor_modules_are_smoke_only(tmp_path):
    smoke_plan = one_cell_plan(smoke=True)
    smoke_campaign = Campaign(tmp_path / "smoke")
    register_test_plan(smoke_campaign, smoke_plan)
    CampaignRunner(smoke_campaign, module="custom.executor")._validate_executor_module(
        (smoke_plan.cells[0].cell_id,)
    )

    scientific_plan = one_cell_plan(smoke=False)
    scientific_campaign = Campaign(tmp_path / "scientific")
    register_test_plan(scientific_campaign, scientific_plan)
    with pytest.raises(CampaignError, match="custom modules are smoke-only"):
        CampaignRunner(
            scientific_campaign,
            module="custom.executor",
        )._validate_executor_module((scientific_plan.cells[0].cell_id,))


def test_artifact_ref_detects_bad_declared_hash_and_size(tmp_path):
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"abc")
    digest = hashlib.sha256(b"abc").hexdigest()
    ArtifactRef("x", str(path), digest, 3).verify(tmp_path)
    with pytest.raises(ArtifactError, match="size mismatch"):
        ArtifactRef("x", str(path), digest, 4).verify(tmp_path)


def test_campaign_publications_fsync_replacements_and_journal_directory(
    tmp_path,
    monkeypatch,
):
    replacements = []
    directory_syncs = []
    original_replace = campaign_module.durable_replace

    def observed_replace(source, destination, *, file_already_synced=False):
        replacements.append((Path(destination), file_already_synced))
        return original_replace(
            source,
            destination,
            file_already_synced=file_already_synced,
        )

    original_directory_sync = campaign_module.fsync_directory

    def observed_directory_sync(path):
        directory_syncs.append(Path(path))
        return original_directory_sync(path)

    monkeypatch.setattr(campaign_module, "durable_replace", observed_replace)
    monkeypatch.setattr(campaign_module, "fsync_directory", observed_directory_sync)
    target = tmp_path / "atomic.json"
    campaign_module._atomic_json(target, {"ready": True})
    campaign = Campaign(tmp_path / "campaign")
    campaign._append_event(campaign._event("audit", "__campaign__"))
    campaign._append_event(campaign._event("audit", "__campaign__"))

    assert replacements == [(target, True)]
    assert directory_syncs.count(campaign.root) == 1


def test_stage_manifest_rejects_extra_kinds_and_malformed_sizes_cleanly(tmp_path):
    plan = one_cell_plan()
    campaign = Campaign(tmp_path)
    register_test_plan(campaign, plan)
    cell_id = plan.cells[0].cell_id
    runner = CampaignRunner(campaign)
    preparation = tmp_path / "preparation.json"
    preparation.write_text('{"ready": true}\n')
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text('{"future": true}\n')

    def entry(kind: str, path: Path, size: object | None = None) -> dict[str, object]:
        body = path.read_bytes()
        return {
            "kind": kind,
            "path": str(path),
            "sha256": hashlib.sha256(body).hexdigest(),
            "size_bytes": len(body) if size is None else size,
        }

    manifest = tmp_path / "stage.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "bsc-stage-artifacts-v2",
                "cell_id": cell_id,
                "stage": "prepare",
                "artifacts": [
                    entry("preparation", preparation),
                    entry("checkpoint", checkpoint),
                ],
            }
        )
        + "\n"
    )
    with pytest.raises(ArtifactError, match="must be exactly"):
        runner._load_artifact_manifest(cell_id, "prepare", manifest)

    manifest.write_text(
        json.dumps(
            {
                "schema": "bsc-stage-artifacts-v2",
                "cell_id": cell_id,
                "stage": "prepare",
                "artifacts": [entry("preparation", preparation, "not-an-int")],
            }
        )
        + "\n"
    )
    with pytest.raises(ArtifactError, match="not an integer"):
        runner._load_artifact_manifest(cell_id, "prepare", manifest)


def _write_stage_artifact_manifest(
    path: Path,
    *,
    cell_id: str,
    stage: str,
    artifacts: tuple[tuple[str, Path], ...],
) -> None:
    entries = []
    for kind, artifact_path in artifacts:
        body = artifact_path.read_bytes()
        entries.append(
            {
                "kind": kind,
                "path": str(artifact_path),
                "sha256": hashlib.sha256(body).hexdigest(),
                "size_bytes": len(body),
            }
        )
    path.write_text(
        json.dumps(
            {
                "schema": "bsc-stage-artifacts-v2",
                "cell_id": cell_id,
                "stage": stage,
                "artifacts": entries,
            },
            sort_keys=True,
        )
        + "\n"
    )


def test_runner_hashes_each_new_artifact_once_without_gate_rescans(
    tmp_path,
    monkeypatch,
):
    plan = one_cell_plan()
    campaign = Campaign(tmp_path)
    register_test_plan(campaign, plan)
    cell_id = plan.cells[0].cell_id
    runner = CampaignRunner(campaign)

    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    preparation = output_dir / "preparation.json"
    preparation.write_text('{"ready": true}\n')
    prepare_manifest = tmp_path / "prepare-stage.json"
    _write_stage_artifact_manifest(
        prepare_manifest,
        cell_id=cell_id,
        stage="prepare",
        artifacts=(("preparation", preparation),),
    )

    calls: list[Path] = []
    real_sha256 = campaign_module._sha256

    def counted_sha256(path: Path, chunk_bytes: int = 1 << 20) -> str:
        calls.append(path.resolve())
        return real_sha256(path, chunk_bytes)

    monkeypatch.setattr(campaign_module, "_sha256", counted_sha256)
    prepare_refs = list(
        runner._load_artifact_manifest(cell_id, "prepare", prepare_manifest)
    )
    prepare_refs.append(
        campaign._verified_artifact_from_path("prepare_manifest", prepare_manifest)
    )
    campaign.transition(cell_id, RunState.PREPARED, artifacts=prepare_refs)
    assert calls == [preparation.resolve(), prepare_manifest.resolve()]

    campaign.transition(cell_id, RunState.RUNNING)
    checkpoint = output_dir / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    training_report = output_dir / "training-report.json"
    training_report.write_text('{"trained": true}\n')
    train_manifest = tmp_path / "train-stage.json"
    _write_stage_artifact_manifest(
        train_manifest,
        cell_id=cell_id,
        stage="train",
        artifacts=(
            ("checkpoint", checkpoint),
            ("training_report", training_report),
        ),
    )
    train_refs = list(runner._load_artifact_manifest(cell_id, "train", train_manifest))
    train_refs.append(
        campaign._verified_artifact_from_path("train_manifest", train_manifest)
    )
    campaign.transition(cell_id, RunState.TRAINED, artifacts=train_refs)
    assert calls == [
        preparation.resolve(),
        prepare_manifest.resolve(),
        checkpoint.resolve(),
        training_report.resolve(),
        train_manifest.resolve(),
    ]


def test_new_campaign_process_rehashes_unchanged_ancestors_once(
    tmp_path,
    monkeypatch,
):
    plan = one_cell_plan()
    first = Campaign(tmp_path)
    register_test_plan(first, plan)
    cell_id = plan.cells[0].cell_id
    preparation = write_artifact(tmp_path, "preparation", {"ready": True})
    prepare_manifest = write_artifact(
        tmp_path,
        "prepare_manifest",
        {"stage": "prepare"},
    )
    first.transition(
        cell_id,
        RunState.PREPARED,
        artifacts=(preparation, prepare_manifest),
    )
    first.transition(cell_id, RunState.RUNNING)

    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    training_report_path = tmp_path / "training-report.json"
    training_report_path.write_text('{"trained": true}\n')
    train_manifest_path = tmp_path / "train-stage.json"
    _write_stage_artifact_manifest(
        train_manifest_path,
        cell_id=cell_id,
        stage="train",
        artifacts=(
            ("checkpoint", checkpoint_path),
            ("training_report", training_report_path),
        ),
    )

    restarted = Campaign(tmp_path)
    runner = CampaignRunner(restarted)
    calls: list[Path] = []
    real_sha256 = campaign_module._sha256

    def counted_sha256(path: Path, chunk_bytes: int = 1 << 20) -> str:
        calls.append(path.resolve())
        return real_sha256(path, chunk_bytes)

    monkeypatch.setattr(campaign_module, "_sha256", counted_sha256)
    refs = list(
        runner._load_artifact_manifest(cell_id, "train", train_manifest_path)
    )
    refs.append(
        restarted._verified_artifact_from_path(
            "train_manifest",
            train_manifest_path,
        )
    )
    restarted.transition(cell_id, RunState.TRAINED, artifacts=refs)
    assert sorted(calls) == sorted(
        [
            checkpoint_path.resolve(),
            training_report_path.resolve(),
            train_manifest_path.resolve(),
            preparation.resolve(tmp_path),
            prepare_manifest.resolve(tmp_path),
        ]
    )


def test_verified_token_refuses_same_size_mutation_with_restored_mtime(
    tmp_path,
    monkeypatch,
):
    campaign = Campaign(tmp_path)
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"original")
    token = campaign._verified_artifact_from_path("artifact", path)
    original_stat = path.stat()
    time.sleep(0.002)
    path.write_bytes(b"mutation")
    os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

    calls = 0
    real_sha256 = campaign_module._sha256

    def counted_sha256(target: Path, chunk_bytes: int = 1 << 20) -> str:
        nonlocal calls
        calls += 1
        return real_sha256(target, chunk_bytes)

    monkeypatch.setattr(campaign_module, "_sha256", counted_sha256)
    with pytest.raises(ArtifactError, match="hash mismatch"):
        campaign._verify_artifact(token)
    assert calls == 1


def test_verified_token_refuses_inode_replacement(tmp_path, monkeypatch):
    campaign = Campaign(tmp_path)
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"original")
    token = campaign._verified_artifact_from_path("artifact", path)
    original_stat = path.stat()
    replacement = tmp_path / "replacement.bin"
    replacement.write_bytes(b"mutation")
    os.utime(replacement, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    os.replace(replacement, path)
    assert path.stat().st_ino != original_stat.st_ino

    calls = 0
    real_sha256 = campaign_module._sha256

    def counted_sha256(target: Path, chunk_bytes: int = 1 << 20) -> str:
        nonlocal calls
        calls += 1
        return real_sha256(target, chunk_bytes)

    monkeypatch.setattr(campaign_module, "_sha256", counted_sha256)
    with pytest.raises(ArtifactError, match="hash mismatch"):
        campaign._verify_artifact(token)
    assert calls == 1


def test_verified_token_refuses_truncation_before_hash(tmp_path, monkeypatch):
    campaign = Campaign(tmp_path)
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"original")
    token = campaign._verified_artifact_from_path("artifact", path)
    path.write_bytes(b"short")

    calls = 0

    def counted_sha256(target: Path, chunk_bytes: int = 1 << 20) -> str:
        nonlocal calls
        calls += 1
        return hashlib.sha256(target.read_bytes()).hexdigest()

    monkeypatch.setattr(campaign_module, "_sha256", counted_sha256)
    with pytest.raises(ArtifactError, match="size mismatch"):
        campaign._verify_artifact(token)
    assert calls == 0


def test_stale_and_foreign_tokens_force_content_reverification(tmp_path, monkeypatch):
    first = Campaign(tmp_path)
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"original")
    token = first._verified_artifact_from_path("artifact", path)
    original_stat = path.stat()
    time.sleep(0.002)
    path.write_bytes(b"original")
    os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

    calls = 0
    real_sha256 = campaign_module._sha256

    def counted_sha256(target: Path, chunk_bytes: int = 1 << 20) -> str:
        nonlocal calls
        calls += 1
        return real_sha256(target, chunk_bytes)

    monkeypatch.setattr(campaign_module, "_sha256", counted_sha256)
    refreshed = first._verify_artifact(token)
    assert calls == 1
    assert refreshed._verification != token._verification

    forged_digest = replace(refreshed, sha256="0" * 64)
    with pytest.raises(ArtifactError, match="hash mismatch"):
        first._verify_artifact(forged_digest)
    assert calls == 2

    restarted = Campaign(tmp_path)
    restarted_ref = restarted._verify_artifact(refreshed)
    assert calls == 3
    assert restarted_ref._verification != refreshed._verification
