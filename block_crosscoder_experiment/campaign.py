"""Content-addressed campaign journal, state machine, and generic cell runner.

The append-only journal is authoritative.  ``state.json`` files are disposable
atomic snapshots rebuilt from that journal by :meth:`Campaign.reconcile`.
Training success is only the ``trained`` state: qualification is a distinct
stage whose signed inputs must name the hashes of the checkpoint, calibration,
deployable codec, and evaluation artifacts.

``run_cell`` contract
---------------------

The runner invokes::

    python -m block_crosscoder_experiment.cli.run_cell \
        --cell CELL_MANIFEST --stage STAGE --artifacts-out OUTPUT

and adds ``--resume`` when requested.  Production runners keep one isolated
child alive for all remaining stages of a cell and exchange the same requests
over a line-delimited control channel.  A custom implementation module retains
the one-shot command contract above.  The child atomically writes only the new
outputs from each stage to ``OUTPUT`` as::

    {"schema": "bsc-stage-artifacts-v2", "cell_id": "...", "stage": "train",
     "artifacts": [{"kind": "checkpoint", "path": "...", "sha256": "..."}]}

Relative artifact paths are relative to the campaign root.  Hashes are always
recomputed by this module once.  A process-local stat fingerprint permits
later gates in the same runner to reuse that verification; a changed file or
new process forces another content hash.  The qualification artifact has the
stricter contract documented in :meth:`Campaign._validate_qualification`.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import copy
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import fcntl
from statistics import median
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .activation_identity import (
    ACTIVATION_CONTENT_IDENTITY_SCHEMA,
    activation_content_identity,
)
from .durability import (
    durable_create,
    durable_mkdir,
    durable_replace,
    fsync_directory,
)
from .implementation import (
    implementation_identity,
    validate_implementation_identity,
)
from .studies import (
    CellSpec,
    FrozenPanelDecision,
    FrozenPanelEntry,
    FrozenSelection,
    Phase,
    Phase1Blueprint,
    Phase2Blueprint,
    Phase3Blueprint,
    PHASE2_SHARING_COORDINATE_CONCORDANCE_MIN,
    PHASE2_CONFIRMATION_SCORE_DEGRADATION_MAX,
    PHASE2_CONFIRMATION_SCORE_DEGRADATION_SENSITIVITY,
    PHASE2_CONFIRMATION_THRESHOLD_BASIS,
    PHASE2_INELIGIBLE_SELECTION_SCORE,
    PHASE2_SELECTION_METRIC_KEY,
    PHASE2_SHARING_FVU_ABSOLUTE_MAX,
    PHASE2_SHARING_INTERSECTION_ENERGY_COVERAGE_MIN,
    PHASE2_SHARING_INTERSECTION_RECALL_MIN,
    PHASE2_SHARING_ROOT_FVU_DEGRADATION_MAX,
    SelectionPolicy,
    StudyError,
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
    resolved_candidate_execution_signature,
)


CAMPAIGN_SCHEMA = "bsc-campaign-v1"
ARTIFACT_SCHEMA = "bsc-stage-artifacts-v2"
QUALIFICATION_SCHEMA = "bsc-qualification-v4"
PREPARATION_SCHEMA = "bsc-preparation-v4"
EVALUATION_SCHEMA = "bsc-evaluation-v3"
EVALUATION_EXECUTION_IMPLEMENTATION = "fused_deployable_full_view_packet_v2"
CANONICAL_CELL_MODULE = "block_crosscoder_experiment.cli.run_cell"
CAMPAIGN_IMPLEMENTATION_SCHEMA = "bsc-campaign-implementation-v2"
PROMOTION_SCHEMA = "bsc-promotion-v1"
SELECTION_SCHEMA = "bsc-stage-selection-v2"
FAMILY_NOMINATION_SCHEMA = "bsc-family-revisit-nomination-v3"
PHASE1_DECISION_SCHEMA = "bsc-phase1-go-no-go-decision-v3"
PHASE1_CAMPAIGN_MANIFEST_SCHEMA = "bsc-phase1-campaign-manifest-v3"
PANEL_DECISION_PRODUCER_SCHEMA = "bsc-phase3-panel-decision-v2"
PHASE2_CAMPAIGN_MANIFEST_SCHEMA = "bsc-phase2-campaign-manifest-v3"
SELECTION_UNIVERSE_SCHEMA = "bsc-phase2-selection-universe-v3"
PHASE2_GATE_AMENDMENT_SCHEMA = "bsc-phase2-common-gate-amendment-v1"
PHASE2_GATE_AMENDMENT_ARTIFACT_KIND = "phase2_gate_amendment"
PHASE2_IMPLEMENTATION_AMENDMENT_SCHEMA = "bsc-phase2-implementation-amendment-v1"
PHASE2_IMPLEMENTATION_AMENDMENT_ARTIFACT_KIND = "phase2_implementation_amendment"
PHASE2_CODEC_DIAGNOSTIC_CHECKS = frozenset(
    {"codec_calibration_exclusion", "codec_evaluation_exclusion"}
)
_AUTO_PHASE2_GATE_AMENDMENT = object()
PHASE2_CAMPAIGN_MANIFEST_KEYS = frozenset(
    {
        "schema",
        "source_phase2_plan_id",
        "source_phase2_blueprint_id",
        "plan_sha256",
        "blueprint_sha256",
        "journal_sha256",
        "journal_sha256_semantics",
        "smoke",
        "phase1_decision_sha256",
        "phase1_decision",
        "phase1_transfer_id",
        "plan_history",
        "selection_chain",
        "main_selection_chain",
        "family_selection_chains",
        "family_nominations",
        "confirmation_noninferiority",
        "duplicate_substitutions",
        "cells",
        "panel_entries",
    }
)
PHASE2_SELECTION_UNIVERSE_KEYS = frozenset(
    {
        "schema",
        "source_phase2_plan_id",
        "source_phase2_blueprint_id",
        "selection_chain",
        "main_selection_chain",
        "family_selection_chains",
        "family_nominations",
        "ranked_stage_universes",
        "panel_source_candidate_ids",
        "phase1_decision_id",
        "phase1_transfer_id",
        "confirmation_noninferiority",
        "duplicate_substitutions",
    }
)

REQUIRED_QUALIFICATION_CHECKS = frozenset(
    {
        "deployment_schedule_integrity",
        "encoder_scale_calibration_integrity",
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
)
REQUIRED_SCIENTIFIC_OUTCOME_CHECKS = frozenset(
    {
        "support_target_calibration",
        "codec_calibration_exclusion",
        "codec_evaluation_exclusion",
        "phase1_identification",
        "production_precision_finite",
        "production_precision_reconstruction",
        "production_precision_support",
        "production_fixed_rate_frontier",
    }
)
REQUIRED_SCIENTIFIC_MARGIN_KEYS = frozenset(
    {
        "support_target_abs_error",
        "codec_calibration_excluded_fraction",
        "codec_evaluation_excluded_fraction",
        "phase1_native_identification",
        "phase1_deployed_identification",
        "production_precision_reconstruction",
        "production_precision_support_iou",
        "production_fixed_rate_nonzero_endpoints",
    }
)
QUALIFICATION_KEYS = frozenset(
    {
        "schema",
        "cell_id",
        "qualified",
        "checks",
        "scientific_outcome",
        "inputs",
        "implementation_identity",
        "implementation_identity_sha256",
        "validation",
        "qualification_profile",
        "thresholds_version",
        "thresholds",
        "selection_metrics",
        "selection_metrics_sha256",
        "selection_metrics_evaluation_sha256",
        "fixed_rate_operating_policy",
        "promotion_eligible",
        "promotion_ineligible_reasons",
        "selection_eligible_for_protocol_test",
        "selection_eligibility_mode",
    }
)
QUALIFICATION_INPUT_KINDS = (
    "preparation",
    "checkpoint",
    "calibration",
    "deployment_codec",
    "deployment_schedules",
    "evaluation",
)


class CampaignError(RuntimeError):
    pass


class InvalidTransition(CampaignError):
    pass


class ArtifactError(CampaignError):
    pass


class CampaignLocked(CampaignError):
    pass


class RunState(str, Enum):
    PLANNED = "planned"
    PREPARED = "prepared"
    RUNNING = "running"
    TRAINED = "trained"
    CALIBRATED = "calibrated"
    EVALUATED = "evaluated"
    QUALIFIED = "qualified"
    FAILED = "failed"
    PROMOTED = "promoted"


LEGAL_TRANSITIONS: Mapping[RunState, frozenset[RunState]] = {
    RunState.PLANNED: frozenset({RunState.PREPARED, RunState.FAILED}),
    RunState.PREPARED: frozenset({RunState.RUNNING, RunState.FAILED}),
    RunState.RUNNING: frozenset({RunState.TRAINED, RunState.FAILED}),
    RunState.TRAINED: frozenset({RunState.CALIBRATED, RunState.FAILED}),
    RunState.CALIBRATED: frozenset({RunState.EVALUATED, RunState.FAILED}),
    RunState.EVALUATED: frozenset({RunState.QUALIFIED, RunState.FAILED}),
    RunState.QUALIFIED: frozenset({RunState.PROMOTED, RunState.FAILED}),
    RunState.FAILED: frozenset(),  # retry() is the only recovery surface
    RunState.PROMOTED: frozenset(),
}


REQUIRED_ARTIFACTS: Mapping[RunState, frozenset[str]] = {
    RunState.PREPARED: frozenset({"preparation", "prepare_manifest"}),
    RunState.TRAINED: frozenset(
        {
            "preparation",
            "prepare_manifest",
            "checkpoint",
            "training_report",
            "train_manifest",
        }
    ),
    RunState.CALIBRATED: frozenset(
        {
            "preparation",
            "prepare_manifest",
            "checkpoint",
            "training_report",
            "train_manifest",
            "calibration",
            "deployment_codec",
            "calibration_record",
            "calibrate_manifest",
        }
    ),
    RunState.EVALUATED: frozenset(
        {
            "preparation",
            "prepare_manifest",
            "checkpoint",
            "training_report",
            "train_manifest",
            "calibration",
            "deployment_codec",
            "calibration_record",
            "calibrate_manifest",
            "deployment_schedules",
            "evaluation",
            "evaluate_manifest",
        }
    ),
    RunState.QUALIFIED: frozenset(
        {
            "preparation",
            "prepare_manifest",
            "checkpoint",
            "training_report",
            "train_manifest",
            "calibration",
            "deployment_codec",
            "calibration_record",
            "calibrate_manifest",
            "deployment_schedules",
            "evaluation",
            "evaluate_manifest",
            "qualification",
            "qualify_manifest",
        }
    ),
    RunState.PROMOTED: frozenset(
        {
            "preparation",
            "prepare_manifest",
            "checkpoint",
            "training_report",
            "train_manifest",
            "calibration",
            "deployment_codec",
            "calibration_record",
            "calibrate_manifest",
            "deployment_schedules",
            "evaluation",
            "evaluate_manifest",
            "qualification",
            "qualify_manifest",
            "promotion",
        }
    ),
}


STAGE_TARGETS: Mapping[str, RunState] = {
    "prepare": RunState.PREPARED,
    "train": RunState.TRAINED,
    "calibrate": RunState.CALIBRATED,
    "evaluate": RunState.EVALUATED,
    "qualify": RunState.QUALIFIED,
}


EXPECTED_STAGE_ARTIFACTS: Mapping[str, frozenset[str]] = {
    "prepare": frozenset({"preparation"}),
    "train": frozenset({"checkpoint", "training_report"}),
    "calibrate": frozenset(
        {
            "calibration",
            "deployment_codec",
            "calibration_record",
        }
    ),
    "evaluate": frozenset(
        {
            "deployment_schedules",
            "evaluation",
        }
    ),
    "qualify": frozenset({"qualification"}),
}
TARGET_STAGES: Mapping[RunState, str] = {
    target: stage for stage, target in STAGE_TARGETS.items()
}


TRANSITION_ARTIFACT_KINDS: Mapping[RunState, frozenset[str]] = {
    RunState.PREPARED: frozenset({"preparation", "prepare_manifest"}),
    RunState.RUNNING: frozenset(),
    RunState.TRAINED: frozenset({"checkpoint", "training_report", "train_manifest"}),
    RunState.CALIBRATED: frozenset(
        {
            "calibration",
            "deployment_codec",
            "calibration_record",
            "calibrate_manifest",
        }
    ),
    RunState.EVALUATED: frozenset(
        {"deployment_schedules", "evaluation", "evaluate_manifest"}
    ),
    RunState.QUALIFIED: frozenset({"qualification", "qualify_manifest"}),
    RunState.FAILED: frozenset(),
    RunState.PROMOTED: frozenset({"promotion"}),
}


def _slug(identifier: str) -> str:
    return identifier.replace(":", "_").replace("/", "_")


def _sha256(path: Path, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return (
        "sha256:" + hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    )


def _run_cell_json_sha256(payload: Mapping[str, Any]) -> str:
    """Hash the exact reviewable JSON bytes emitted by ``run_cell``."""

    body = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(body).hexdigest()


def _sha256_canonical_payload(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _is_sha256_hex(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _process_identity(pid: int) -> str | None:
    """Return a PID-reuse-resistant process birth identity when available."""

    proc_stat = Path(f"/proc/{pid}/stat")
    try:
        fields = proc_stat.read_text(encoding="utf-8").split()
        if len(fields) > 21:
            return "proc-start:" + fields[21]
    except (OSError, UnicodeError):
        pass
    try:
        completed = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(pid)],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    started = completed.stdout.strip()
    return None if completed.returncode != 0 or not started else "ps-start:" + started


def _process_matches(pid: int, identity: Any) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    if identity is None:
        return True
    return isinstance(identity, str) and _process_identity(pid) == identity


def _process_matches_for_termination(pid: int, identity: Any) -> bool:
    """Return true only for a live process with an exact birth identity."""

    if not isinstance(identity, str) or not identity:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    observed = _process_identity(pid)
    return observed is not None and observed == identity


def _qualification_thresholds(cell: CellSpec) -> dict[str, Any]:
    values = cell.decision_map
    return {
        "schema": "bsc-integrity-thresholds-2026-07-22.v2",
        "support_target_abs_error_max": 0.1,
        "codec_excluded_calibration_event_fraction_max": 0.01,
        "codec_excluded_evaluation_event_fraction_max": 0.01,
        "probability_metric_range": [0.0, 1.0],
        "required_quantizer_bits": list(values["codec.quantizer_bits"]),
        "phase1_identification_thresholds": [
            list(item)
            for item in values["qualification.phase1_identification_thresholds"]
        ],
        "phase1_identification_enforced": values["runtime.smoke"] is False,
        "phase1_margin_normalization_contract": values[
            "evaluation.phase1_margin_normalization"
        ],
        "phase1_rank_mismatch_contract": values["evaluation.rank_mismatch_contract"],
        "phase1_pathology_association_contract": values[
            "evaluation.pathology_association_contract"
        ],
        "phase1_pathology_strong_association_cutoff": values[
            "evaluation.pathology_strong_association_cutoff"
        ],
        "phase1_pathology_weak_association_cutoff": values[
            "evaluation.pathology_weak_association_cutoff"
        ],
        "phase1_pathology_association_cutoff_sensitivity": [
            list(item)
            for item in values["evaluation.pathology_association_cutoff_sensitivity"]
        ],
        "encoder_scale_fit_statistic": values["model.encoder_scale_fit_statistic"],
        "encoder_scale_fit_solver": values["model.encoder_scale_fit_solver"],
        "encoder_scale_fit_target": values["model.encoder_scale_fit_target"],
        "encoder_scale_fit_tolerance": values["model.encoder_scale_fit_tolerance"],
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
    }


def _validate_implementation_identity(
    identity: Mapping[str, Any],
    *,
    scientific: bool,
) -> str:
    try:
        return validate_implementation_identity(identity, scientific=scientific)
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactError(str(exc)) from exc


def _promotion_reasons_from_evidence(
    cell: CellSpec,
    *,
    outcome_passed: bool,
    evaluation: Mapping[str, Any],
) -> list[str]:
    reasons: list[str] = []
    if cell.decision_map["runtime.smoke"] is not False:
        reasons.append("runtime_smoke")
    if evaluation.get("raw_space", {}).get("eligible") is not True:
        reasons.append("raw_codec_requires_unpriced_side_information")
    if (
        cell.phase is not Phase.PHASE1
        and evaluation.get("fixed_rate_raw_selection", {}).get("eligible") is not True
    ):
        reasons.append("fixed_rate_budget_ineligible")
    if (
        cell.phase is Phase.PHASE1
        and evaluation.get("synthetic_recovery", {})
        .get("deployed", {})
        .get("shared_feature_claim_eligible")
        is not True
    ):
        reasons.append("synthetic_shared_feature_claim_ineligible")
    if cell.decision_map["qualification.promotable"] is not True:
        reasons.append("resolved_nonpromotable_cell")
    if outcome_passed is not True:
        reasons.append("scientific_outcome_failed")
    if cell.phase is Phase.PHASE3 or "confirmation" in cell.stage:
        if not cell.decision_map["selection.parent_cell_ids"]:
            reasons.append("missing_frozen_phase2_selection_decision")
    return reasons


def _policy_retained_candidates(
    candidates: Sequence[Mapping[str, Any]],
    policy: SelectionPolicy,
    *,
    smoke_protocol_only: bool,
) -> list[Mapping[str, Any]]:
    """Apply the frozen cutoff and tie policy identically at select and replay."""

    if policy.retain_count is not None:
        keep = min(len(candidates), policy.retain_count)
    else:
        assert policy.retain_fraction is not None
        keep = max(1, math.ceil(len(candidates) * policy.retain_fraction))
    retained = list(candidates[:keep])
    if (
        retained
        and not smoke_protocol_only
        and policy.tie_policy == "retain_all_at_cutoff"
        and keep < len(candidates)
    ):
        cutoff = (
            float(candidates[keep - 1]["median"]),
            float(candidates[keep - 1]["worst_seed"]),
        )
        retained.extend(
            candidate
            for candidate in candidates[keep:]
            if (
                float(candidate["median"]),
                float(candidate["worst_seed"]),
            )
            == cutoff
        )
    return retained


def _validate_panel_entry_seed_coverage(
    entries: Sequence[FrozenPanelEntry],
    expected_seeds: tuple[int, ...],
) -> None:
    for entry in entries:
        if tuple(cell.seed for cell in entry.source_cells) != expected_seeds:
            raise CampaignError(
                "panel entry does not exactly cover the blueprint seeds"
            )


def _validate_exact_confirmation_guard(
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    if observed != expected:
        raise CampaignError(
            "confirmation does not reuse the exact authenticated sharing guard"
        )


def _confirmation_score_sensitivity_payload(
    rows: Sequence[Mapping[str, Any]],
    thresholds: Sequence[float],
    *,
    smoke: bool,
    require_sharing_guard: bool,
) -> dict[str, Any]:
    """Vary only the score cutoff while holding applicable center gates fixed."""

    sensitivity_rows = []
    for threshold in thresholds:
        passing_seeds = (
            None
            if smoke
            else [
                row["seed"]
                for row in rows
                if row.get("qualification_passed") is True
                and (
                    not require_sharing_guard
                    or row.get("sharing_guard_passed") is True
                )
                and Campaign._meets_upper_bound(
                    float(row["score_degradation"]), float(threshold)
                )
            ]
        )
        sensitivity_rows.append(
            {
                "threshold": threshold,
                "passing_seeds": passing_seeds,
                "passed_all_seeds": (
                    None if smoke else len(passing_seeds) == len(rows)
                ),
            }
        )
    return {
        "mode": "marginal_counterfactuals_center_policy_not_retuned",
        "rows": sensitivity_rows,
        "ungated_passed_all_seeds": (
            None
            if smoke
            else all(
                row.get("qualification_passed") is True
                and (
                    not require_sharing_guard
                    or row.get("sharing_guard_passed") is True
                )
                for row in rows
            )
        ),
    }


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _phase1_min_margin(value: float, threshold: float) -> float:
    denominator = (
        max(1.0 - threshold, 1.0e-12)
        if value >= threshold
        else max(abs(threshold), 1.0e-12)
    )
    return (value - threshold) / denominator


def _phase1_max_margin(value: float, threshold: float) -> float:
    denominator = (
        max(abs(threshold), 1.0e-12)
        if value <= threshold
        else max(1.0 - threshold, 1.0e-12)
    )
    return (threshold - value) / denominator


def _float_mapping_matches(
    observed: Mapping[str, Any], expected: Mapping[str, float]
) -> bool:
    return set(observed) == set(expected) and all(
        _finite_number(observed.get(name))
        and math.isclose(float(observed[name]), value, rel_tol=0.0, abs_tol=1e-12)
        for name, value in expected.items()
    )


def _validated_phase1_identification(
    selection_metrics: Mapping[str, Any], *, cell: CellSpec
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """Replay the Phase-1 identification reducer consumed by authorization.

    In particular, an endpoint passes its declared aggregate conjunction.  Its
    headline margin may still be negative when the preregistered recovered-
    factor fraction permits a minority of factors to fail; authorization must
    not silently strengthen that policy to a 100% per-factor requirement.
    """

    identification = selection_metrics.get("identification")
    validation = selection_metrics.get("validation")
    if not isinstance(identification, Mapping) or set(identification) != {
        "native",
        "deployed",
    }:
        raise ArtifactError(
            "Phase-1 selection metrics require exact native/deployed identification"
        )
    if not isinstance(validation, Mapping) or set(validation) != {
        "phase1_identification_applicable",
        "phase1_identification_conjunction",
        "phase1_identification_margin",
    }:
        raise ArtifactError("Phase-1 validation uses a noncanonical field set")

    threshold_items = cell.decision_map.get(
        "qualification.phase1_identification_thresholds"
    )
    try:
        expected_thresholds = {str(item[0]): float(item[1]) for item in threshold_items}
    except (TypeError, ValueError, IndexError) as exc:
        raise ArtifactError(
            "resolved Phase-1 cell has malformed identification thresholds"
        ) from exc
    required_thresholds = {
        "per_factor.support_association_min",
        "per_factor.subspace_overlap_min_when_eligible",
        "per_factor.global_isolated_input_r2_min",
        "per_factor.aligned_code_r2_min",
        "aggregate.recovered_factor_fraction_min",
        "aggregate.support_precision_min",
        "aggregate.support_recall_min",
        "pathology.duplicate_block_fraction_max",
        "pathology.cross_factor_mixing_fraction_max",
        "pathology.nonfinite_count_max",
    }
    if set(expected_thresholds) != required_thresholds:
        raise ArtifactError("resolved Phase-1 identification thresholds are incomplete")

    applicable_keys = {
        "applicable",
        "ineligible_reason",
        "thresholds",
        "per_factor",
        "aggregate",
        "checks",
        "normalized_margins",
        "margin_normalization_contract",
        "margin",
        "passed",
    }
    inapplicable_keys = applicable_keys
    factor_keys = {
        "factor",
        "support_association",
        "subspace_overlap",
        "global_isolated_input_r2",
        "aligned_code_r2",
        "rank_mismatch",
        "component_margins",
        "margin",
        "identified",
        "same_block_gate_has_positive_headroom",
    }
    aggregate_keys = {
        "recovered_factor_fraction",
        "support_precision",
        "support_recall",
        "inactive_dictionary_fraction_diagnostic",
        "duplicate_block_fraction",
        "cross_factor_mixing_fraction",
        "nonfinite_count",
    }
    check_keys = {
        "recovered_factor_fraction",
        "support_precision",
        "support_recall",
        "duplicate_block_fraction",
        "cross_factor_mixing_fraction",
        "nonfinite_count",
    }
    normalized_margin_keys = {
        "worst_eligible_per_factor",
        "recovered_factor_fraction",
        "support_precision",
        "support_recall",
        "duplicate_block_fraction",
        "cross_factor_mixing_fraction",
    }

    parsed: list[Mapping[str, Any]] = []
    for endpoint_name in ("native", "deployed"):
        endpoint = identification.get(endpoint_name)
        if not isinstance(endpoint, Mapping) or endpoint.get("applicable") not in {
            True,
            False,
        }:
            raise ArtifactError(
                f"Phase-1 {endpoint_name} identification evidence is malformed"
            )
        applicable = endpoint["applicable"] is True
        if set(endpoint) != (applicable_keys if applicable else inapplicable_keys):
            raise ArtifactError(
                f"Phase-1 {endpoint_name} identification uses a noncanonical field set"
            )
        if endpoint.get("thresholds") != expected_thresholds:
            raise ArtifactError(
                f"Phase-1 {endpoint_name} identification thresholds differ from the cell"
            )
        if endpoint.get("margin_normalization_contract") != cell.decision_map.get(
            "evaluation.phase1_margin_normalization"
        ):
            raise ArtifactError(
                f"Phase-1 {endpoint_name} margin contract differs from the cell"
            )
        if not applicable:
            if (
                not isinstance(endpoint.get("ineligible_reason"), str)
                or not endpoint["ineligible_reason"]
                or endpoint.get("per_factor") != []
                or not isinstance(endpoint.get("aggregate"), Mapping)
                or set(endpoint["aggregate"])
                != {"support_precision_diagnostic", "support_recall_diagnostic"}
                or endpoint.get("checks") != {}
                or endpoint.get("normalized_margins") != {}
                or endpoint.get("margin") is not None
                or endpoint.get("passed") is not None
            ):
                raise ArtifactError(
                    f"Phase-1 {endpoint_name} inapplicable evidence is malformed"
                )
            parsed.append(endpoint)
            continue

        if endpoint.get("ineligible_reason") is not None:
            raise ArtifactError(
                f"Phase-1 {endpoint_name} applicable evidence names an ineligible reason"
            )
        per_factor = endpoint.get("per_factor")
        expected_factor_count = cell.decision_map.get("data.n_factors")
        if (
            not isinstance(per_factor, list)
            or not isinstance(expected_factor_count, int)
            or isinstance(expected_factor_count, bool)
            or len(per_factor) != expected_factor_count
            or not per_factor
        ):
            raise ArtifactError(
                f"Phase-1 {endpoint_name} factor evidence is incomplete"
            )
        identified_count = 0
        factor_margins: list[float] = []
        for factor_index, factor in enumerate(per_factor):
            if not isinstance(factor, Mapping) or set(factor) != factor_keys:
                raise ArtifactError(
                    f"Phase-1 {endpoint_name} factor evidence is noncanonical"
                )
            subspace = factor.get("subspace_overlap")
            components = factor.get("component_margins")
            expected_component_keys = {
                "support_association",
                "global_isolated_input_r2",
                "aligned_code_r2",
                *(() if subspace is None else ("subspace_overlap",)),
            }
            if (
                factor.get("factor") != factor_index
                or not all(
                    _finite_number(factor.get(name))
                    for name in (
                        "support_association",
                        "global_isolated_input_r2",
                        "aligned_code_r2",
                        "margin",
                    )
                )
                or (subspace is not None and not _finite_number(subspace))
                or not isinstance(factor.get("rank_mismatch"), Mapping)
                or not isinstance(components, Mapping)
                or set(components) != expected_component_keys
                or not all(_finite_number(value) for value in components.values())
                or not isinstance(factor.get("identified"), bool)
                or factor["identified"] is not (float(factor["margin"]) >= 0.0)
                or not isinstance(
                    factor.get("same_block_gate_has_positive_headroom"), bool
                )
                or not 0.0 <= float(factor["support_association"]) <= 1.0
                or (subspace is not None and not 0.0 <= float(subspace) <= 1.0 + 1e-9)
                or float(factor["global_isolated_input_r2"]) > 1.0 + 1e-9
                or float(factor["aligned_code_r2"]) > 1.0 + 1e-9
            ):
                raise ArtifactError(
                    f"Phase-1 {endpoint_name} factor evidence has invalid types or values"
                )
            expected_components = {
                "support_association": _phase1_min_margin(
                    float(factor["support_association"]),
                    expected_thresholds["per_factor.support_association_min"],
                ),
                "global_isolated_input_r2": _phase1_min_margin(
                    float(factor["global_isolated_input_r2"]),
                    expected_thresholds["per_factor.global_isolated_input_r2_min"],
                ),
                "aligned_code_r2": _phase1_min_margin(
                    float(factor["aligned_code_r2"]),
                    expected_thresholds["per_factor.aligned_code_r2_min"],
                ),
            }
            if subspace is not None:
                expected_components["subspace_overlap"] = _phase1_min_margin(
                    float(subspace),
                    expected_thresholds[
                        "per_factor.subspace_overlap_min_when_eligible"
                    ],
                )
            if not _float_mapping_matches(components, expected_components):
                raise ArtifactError(
                    f"Phase-1 {endpoint_name} component margins are inconsistent"
                )
            factor_margin = min(expected_components.values())
            if not math.isclose(
                float(factor["margin"]), factor_margin, rel_tol=0.0, abs_tol=1e-12
            ) or factor["identified"] is not (factor_margin >= 0.0):
                raise ArtifactError(
                    f"Phase-1 {endpoint_name} factor margin is inconsistent"
                )
            factor_margins.append(factor_margin)
            identified_count += int(factor["identified"])

        aggregate = endpoint.get("aggregate")
        checks = endpoint.get("checks")
        normalized = endpoint.get("normalized_margins")
        if (
            not isinstance(aggregate, Mapping)
            or set(aggregate) != aggregate_keys
            or not all(
                _finite_number(aggregate.get(name))
                for name in aggregate_keys - {"nonfinite_count"}
            )
            or not isinstance(aggregate.get("nonfinite_count"), int)
            or isinstance(aggregate.get("nonfinite_count"), bool)
            or aggregate["nonfinite_count"] < 0
            or not isinstance(checks, Mapping)
            or set(checks) != check_keys
            or not all(isinstance(value, bool) for value in checks.values())
            or not isinstance(normalized, Mapping)
            or set(normalized) != normalized_margin_keys
            or not all(_finite_number(value) for value in normalized.values())
            or not _finite_number(endpoint.get("margin"))
            or not isinstance(endpoint.get("passed"), bool)
            or any(
                not 0.0 <= float(aggregate[name]) <= 1.0
                for name in (
                    "recovered_factor_fraction",
                    "support_precision",
                    "support_recall",
                    "inactive_dictionary_fraction_diagnostic",
                    "duplicate_block_fraction",
                    "cross_factor_mixing_fraction",
                )
            )
        ):
            raise ArtifactError(
                f"Phase-1 {endpoint_name} aggregate identification evidence is malformed"
            )
        recovered_fraction = identified_count / len(per_factor)
        if not math.isclose(
            float(aggregate["recovered_factor_fraction"]),
            recovered_fraction,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ArtifactError(
                f"Phase-1 {endpoint_name} recovered-factor fraction is inconsistent"
            )
        expected_checks = {
            "recovered_factor_fraction": recovered_fraction
            >= expected_thresholds["aggregate.recovered_factor_fraction_min"],
            "support_precision": float(aggregate["support_precision"])
            >= expected_thresholds["aggregate.support_precision_min"],
            "support_recall": float(aggregate["support_recall"])
            >= expected_thresholds["aggregate.support_recall_min"],
            "duplicate_block_fraction": float(aggregate["duplicate_block_fraction"])
            <= expected_thresholds["pathology.duplicate_block_fraction_max"],
            "cross_factor_mixing_fraction": float(
                aggregate["cross_factor_mixing_fraction"]
            )
            <= expected_thresholds["pathology.cross_factor_mixing_fraction_max"],
            "nonfinite_count": aggregate["nonfinite_count"]
            <= expected_thresholds["pathology.nonfinite_count_max"],
        }
        expected_normalized = {
            "worst_eligible_per_factor": min(factor_margins),
            "recovered_factor_fraction": _phase1_min_margin(
                recovered_fraction,
                expected_thresholds["aggregate.recovered_factor_fraction_min"],
            ),
            "support_precision": _phase1_min_margin(
                float(aggregate["support_precision"]),
                expected_thresholds["aggregate.support_precision_min"],
            ),
            "support_recall": _phase1_min_margin(
                float(aggregate["support_recall"]),
                expected_thresholds["aggregate.support_recall_min"],
            ),
            "duplicate_block_fraction": _phase1_max_margin(
                float(aggregate["duplicate_block_fraction"]),
                expected_thresholds["pathology.duplicate_block_fraction_max"],
            ),
            "cross_factor_mixing_fraction": _phase1_max_margin(
                float(aggregate["cross_factor_mixing_fraction"]),
                expected_thresholds["pathology.cross_factor_mixing_fraction_max"],
            ),
        }
        if dict(checks) != expected_checks or endpoint["passed"] is not all(
            expected_checks.values()
        ):
            raise ArtifactError(
                f"Phase-1 {endpoint_name} identification conjunction is inconsistent"
            )
        if not _float_mapping_matches(
            normalized, expected_normalized
        ) or not math.isclose(
            float(endpoint["margin"]),
            min(expected_normalized.values()),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ArtifactError(
                f"Phase-1 {endpoint_name} normalized margin is inconsistent"
            )
        parsed.append(endpoint)

    native, deployed = parsed
    applicable = native["applicable"] is True and deployed["applicable"] is True
    both_inapplicable = (
        native["applicable"] is False and deployed["applicable"] is False
    )
    if not applicable and not both_inapplicable:
        raise ArtifactError(
            "Phase-1 identification endpoints disagree on applicability"
        )
    if both_inapplicable:
        reasons = {native.get("ineligible_reason"), deployed.get("ineligible_reason")}
        if cell.decision_map.get("data.normalization") != "layer" or reasons != {
            "token_layer_normalization_is_not_a_fixed_linear_factor_map"
        }:
            raise ArtifactError(
                "Phase-1 identification inapplicability is unauthorized"
            )
        expected_validation = {
            "phase1_identification_applicable": False,
            "phase1_identification_conjunction": False,
            "phase1_identification_margin": None,
        }
    else:
        if cell.decision_map.get("data.normalization") == "layer":
            raise ArtifactError(
                "Layer-normalized Phase-1 identification must be inapplicable"
            )
        for endpoint_name, endpoint in (
            ("native", native),
            ("deployed", deployed),
        ):
            center_replay = _phase1_counterfactual_endpoint(
                endpoint, expected_thresholds
            )
            if center_replay["passed"] is not endpoint["passed"] or not math.isclose(
                float(center_replay["recovered_factor_fraction"]),
                float(endpoint["aggregate"]["recovered_factor_fraction"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ArtifactError(
                    f"Phase-1 {endpoint_name} center sensitivity contradicts "
                    "its identification gate"
                )
        expected_validation = {
            "phase1_identification_applicable": True,
            "phase1_identification_conjunction": bool(
                native["passed"] and deployed["passed"]
            ),
            "phase1_identification_margin": min(
                float(native["margin"]), float(deployed["margin"])
            ),
        }
    expected_margin = expected_validation["phase1_identification_margin"]
    observed_margin = validation.get("phase1_identification_margin")
    if (
        validation.get("phase1_identification_applicable")
        is not expected_validation["phase1_identification_applicable"]
        or validation.get("phase1_identification_conjunction")
        is not expected_validation["phase1_identification_conjunction"]
        or (
            (expected_margin is None) != (observed_margin is None)
            or expected_margin is not None
            and (
                not _finite_number(observed_margin)
                or not math.isclose(
                    float(observed_margin),
                    float(expected_margin),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            )
        )
    ):
        raise ArtifactError("Phase-1 validation disagrees with identification evidence")
    return native, deployed, validation


def _phase1_counterfactual_endpoint(
    endpoint: Mapping[str, Any],
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    if endpoint.get("applicable") is not True:
        return {"passed": None, "recovered_factor_fraction": None}
    factors = endpoint["per_factor"]
    identified = 0
    for factor in factors:
        component_passes = [
            float(factor["support_association"])
            >= thresholds["per_factor.support_association_min"],
            float(factor["global_isolated_input_r2"])
            >= thresholds["per_factor.global_isolated_input_r2_min"],
            float(factor["aligned_code_r2"])
            >= thresholds["per_factor.aligned_code_r2_min"],
        ]
        if factor["subspace_overlap"] is not None:
            component_passes.append(
                float(factor["subspace_overlap"])
                >= thresholds["per_factor.subspace_overlap_min_when_eligible"]
            )
        identified += int(all(component_passes))
    recovered_fraction = identified / len(factors)
    aggregate = endpoint["aggregate"]
    passed = all(
        (
            recovered_fraction >= thresholds["aggregate.recovered_factor_fraction_min"],
            float(aggregate["support_precision"])
            >= thresholds["aggregate.support_precision_min"],
            float(aggregate["support_recall"])
            >= thresholds["aggregate.support_recall_min"],
            float(aggregate["duplicate_block_fraction"])
            <= thresholds["pathology.duplicate_block_fraction_max"],
            float(aggregate["cross_factor_mixing_fraction"])
            <= thresholds["pathology.cross_factor_mixing_fraction_max"],
            int(aggregate["nonfinite_count"])
            <= thresholds["pathology.nonfinite_count_max"],
        )
    )
    return {"passed": passed, "recovered_factor_fraction": recovered_fraction}


def _expected_phase1_threshold_sensitivity(
    identification: Mapping[str, Any],
    cell: CellSpec,
) -> dict[str, Any]:
    center = {
        str(name): float(value)
        for name, value in identification["native"]["thresholds"].items()
    }
    sensitivity = cell.decision_map.get("qualification.phase1_threshold_sensitivity")
    if not isinstance(sensitivity, (list, tuple)) or {
        str(item[0]) for item in sensitivity
    } != set(center):
        raise ArtifactError("Phase-1 threshold sensitivity grid is incomplete")
    rows: list[dict[str, Any]] = []
    for name, values in sensitivity:
        name = str(name)
        value_rows: list[dict[str, Any]] = []
        for value in values:
            varied = dict(center)
            varied[name] = float(value)
            native = _phase1_counterfactual_endpoint(identification["native"], varied)
            deployed = _phase1_counterfactual_endpoint(
                identification["deployed"], varied
            )
            value_rows.append(
                {
                    "value": float(value),
                    "native": native,
                    "deployed": deployed,
                    "conjunction_passed": (
                        None
                        if native["passed"] is None and deployed["passed"] is None
                        else bool(native["passed"] and deployed["passed"])
                    ),
                }
            )
        rows.append({"threshold": name, "values": value_rows})
    return {
        "schema": "bsc-phase1-identification-threshold-sensitivity-v1",
        "mode": "marginal_counterfactuals_center_policy_not_retuned",
        "center_thresholds": center,
        "rows": rows,
        "changes_primary_gate": False,
    }


def _validate_fixed_rate_operating_policy(
    policy: Any,
    *,
    cell: CellSpec,
) -> None:
    if policy is None:
        return
    expected_keys = {
        "schema",
        "source_cell_id",
        "source_evidence_sha256",
        "aggregation",
        "rows",
        "content_sha256",
    }
    if not isinstance(policy, Mapping) or set(policy) != expected_keys:
        raise ArtifactError("fixed-rate operating policy is noncanonical")
    content = dict(policy)
    content_digest = content.pop("content_sha256")
    if (
        policy.get("schema") != "bsc-fixed-rate-operating-policy-v1"
        or not isinstance(policy.get("source_cell_id"), str)
        or not policy["source_cell_id"]
        or not _is_sha256_hex(policy.get("source_evidence_sha256"))
        or content_digest != _sha256_canonical_payload(content)
    ):
        raise ArtifactError("fixed-rate operating policy content binding is invalid")
    expected_aggregation = (
        "worst_seed_frozen_parent"
        if cell.phase is Phase.PHASE3
        or cell.decision_map.get("evaluation.split") == "confirmation"
        else "development_cell"
    )
    if policy.get("aggregation") != expected_aggregation:
        raise ArtifactError("fixed-rate operating policy has the wrong evidence role")
    if (
        expected_aggregation == "development_cell"
        and policy.get("source_cell_id") != cell.cell_id
    ):
        raise ArtifactError("development operating policy names another cell")
    if expected_aggregation == "worst_seed_frozen_parent" and policy.get(
        "source_cell_id"
    ) not in set(cell.decision_map.get("selection.parent_cell_ids", ())):
        raise ArtifactError("frozen operating policy names a non-parent cell")
    rows = policy.get("rows")
    expected_budgets = tuple(
        float(value)
        for value in cell.decision_map.get(
            "evaluation.fixed_rate_budgets_bits_per_token", ()
        )
    )
    if not isinstance(rows, list) or len(rows) != len(expected_budgets):
        raise ArtifactError("fixed-rate operating policy does not cover exact budgets")
    observed_budgets: list[float] = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "budget_bits_per_token",
            "lower_name",
            "lower_q",
            "upper_name",
            "upper_q",
        }:
            raise ArtifactError("fixed-rate operating-policy row is noncanonical")
        budget = row.get("budget_bits_per_token")
        if not _finite_number(budget):
            raise ArtifactError("fixed-rate operating-policy budget is invalid")
        observed_budgets.append(float(budget))
        for side in ("lower", "upper"):
            name = row.get(f"{side}_name")
            q = row.get(f"{side}_q")
            if name == "zero_event_calibration_mean":
                valid = q is None
            else:
                valid = (
                    isinstance(name, str)
                    and name.startswith("q")
                    and name[1:].isdigit()
                    and type(q) is int
                    and q == int(name[1:])
                )
            if not valid:
                raise ArtifactError(
                    "fixed-rate operating-policy endpoint identity is invalid"
                )
    if tuple(observed_budgets) != expected_budgets:
        raise ArtifactError(
            "fixed-rate operating-policy budget order differs from cell"
        )


def _fixed_rate_lower_envelope(
    points: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Replay the executor's operational lower convex envelope exactly."""

    best_at_rate: dict[float, dict[str, Any]] = {}
    for raw in points:
        rate = float(raw["total_bits_per_token"])
        fvu = float(raw["raw_space_fvu"])
        candidate = dict(raw)
        current = best_at_rate.get(rate)
        if current is None or fvu < float(current["raw_space_fvu"]):
            best_at_rate[rate] = candidate
    ordered = [best_at_rate[rate] for rate in sorted(best_at_rate)]
    pareto: list[dict[str, Any]] = []
    best_fvu = math.inf
    for point in ordered:
        fvu = float(point["raw_space_fvu"])
        if fvu < best_fvu:
            pareto.append(point)
            best_fvu = fvu
    hull: list[dict[str, Any]] = []
    for point in pareto:
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            slope_ab = (float(b["raw_space_fvu"]) - float(a["raw_space_fvu"])) / (
                float(b["total_bits_per_token"]) - float(a["total_bits_per_token"])
            )
            slope_bc = (float(point["raw_space_fvu"]) - float(b["raw_space_fvu"])) / (
                float(point["total_bits_per_token"]) - float(b["total_bits_per_token"])
            )
            if slope_ab < slope_bc:
                break
            hull.pop()
        hull.append(point)
    return hull


def _validate_fixed_rate_evidence(
    fixed_rate: Any,
    *,
    cell: CellSpec,
    validation: Any,
    inputs: Mapping[str, str],
) -> None:
    """Replay the complete fixed-rate score consumed by Phase-2/3 selection."""

    if cell.phase is Phase.PHASE1:
        expected = {
            "schema": "bsc-fixed-rate-raw-selection-v2",
            "applicable": False,
            "eligible": False,
            "operating_policy": None,
            "reason": "phase1_uses_truth_known_identification",
        }
        if fixed_rate != expected:
            raise ArtifactError("Phase-1 fixed-rate evidence is noncanonical")
        return

    expected_keys = {
        "schema",
        "applicable",
        "cell_id",
        "deployment_codec_sha256",
        "calibration_sha256",
        "side_information",
        "packet_formula",
        "rate_axis",
        "interpolation",
        "out_of_range",
        "zero_rate_reconstruction",
        "measured_points",
        "lower_convex_envelope",
        "fixed_budgets",
        "selection_score_name",
        "selection_score",
        "operating_policy",
        "eligible",
        "reason",
        "content_sha256",
    }
    if (
        not isinstance(fixed_rate, Mapping)
        or set(fixed_rate) != expected_keys
        or fixed_rate.get("schema") != "bsc-fixed-rate-raw-selection-v2"
        or fixed_rate.get("applicable") is not True
        or fixed_rate.get("cell_id") != cell.cell_id
    ):
        raise ArtifactError("fixed-rate evidence is missing or noncanonical")
    content = dict(fixed_rate)
    content_sha256 = content.pop("content_sha256")
    if content_sha256 != _sha256_canonical_payload(content):
        raise ArtifactError("fixed-rate evidence content hash mismatch")
    if (
        fixed_rate.get("deployment_codec_sha256") != inputs["deployment_codec"]
        or fixed_rate.get("calibration_sha256") != inputs["calibration"]
    ):
        raise ArtifactError("fixed-rate evidence input binding mismatch")

    values = cell.decision_map
    expected_contract = {
        "rate_axis": values["evaluation.rate_axis"],
        "interpolation": values["evaluation.rate_interpolation"],
        "out_of_range": values["evaluation.rate_out_of_range"],
        "zero_rate_reconstruction": values["evaluation.zero_rate_reconstruction"],
        "selection_score_name": values["evaluation.selection_score"],
    }
    if any(fixed_rate.get(name) != value for name, value in expected_contract.items()):
        raise ArtifactError("fixed-rate evidence differs from the resolved cell policy")

    side = fixed_rate.get("side_information")
    side_keys = {
        "artifact",
        "artifact_size_bytes",
        "amortization_tokens",
        "bits_per_token",
        "includes_optimizer_checkpoint",
        "time_sharing_schedule_contract",
        "operating_record_bytes_per_budget",
        "deployment_schedule_bundle_sha256",
        "deployment_schedule_bundle_size_bytes",
        "deployment_schedule_record_count",
    }
    horizon = values["evaluation.side_information_amortization_tokens"]
    if (
        not isinstance(side, Mapping)
        or set(side) != side_keys
        or side.get("artifact") != "deployable_codec"
        or type(side.get("artifact_size_bytes")) is not int
        or side["artifact_size_bytes"] <= 0
        or side.get("amortization_tokens") != horizon
        or side.get("includes_optimizer_checkpoint") is not False
        or side.get("time_sharing_schedule_contract")
        != values["codec.time_sharing_schedule_contract"]
        or side.get("operating_record_bytes_per_budget") != 32
        or not _is_sha256_hex(side.get("deployment_schedule_bundle_sha256"))
        or side.get("deployment_schedule_bundle_sha256")
        != inputs["deployment_schedules"]
        or type(side.get("deployment_schedule_bundle_size_bytes")) is not int
        or type(side.get("deployment_schedule_record_count")) is not int
        or not _finite_number(side.get("bits_per_token"))
        or not math.isclose(
            float(side["bits_per_token"]),
            8.0 * side["artifact_size_bytes"] / horizon,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ArtifactError("fixed-rate side-information accounting is invalid")
    record_count = side["deployment_schedule_record_count"]
    if (
        record_count < 0
        or side["deployment_schedule_bundle_size_bytes"] != 32 * record_count
    ):
        raise ArtifactError("fixed-rate operating-record bundle accounting is invalid")

    packet = fixed_rate.get("packet_formula")
    if (
        not isinstance(packet, Mapping)
        or set(packet)
        != {
            "rate_model",
            "count_width_bits",
            "block_id_width_bits",
            "included_block_alphabet",
            "sparse_amplitude_bits",
        }
        or packet.get("rate_model") != "fixed_width_decodable_payload_bits_v1"
        or any(
            type(packet.get(name)) is not int or packet[name] <= 0
            for name in (
                "count_width_bits",
                "block_id_width_bits",
                "included_block_alphabet",
            )
        )
        or packet["count_width_bits"]
        != max(1, math.ceil(math.log2(packet["included_block_alphabet"] + 1)))
        or packet["block_id_width_bits"]
        != max(1, math.ceil(math.log2(packet["included_block_alphabet"])))
        or packet.get("sparse_amplitude_bits") != ("q * block_width * selected_events")
    ):
        raise ArtifactError("fixed-rate packet formula is invalid")

    point_keys = {
        "name",
        "q",
        "packet_bits_per_token",
        "side_information_bits_per_token",
        "total_bits_per_token",
        "raw_space_fvu",
    }
    measured = fixed_rate.get("measured_points")
    expected_qs = tuple(sorted(int(item) for item in values["codec.quantizer_bits"]))
    expected_names = ("zero_event_calibration_mean", *(f"q{q}" for q in expected_qs))
    if (
        not isinstance(measured, list)
        or len(measured) != len(expected_names)
        or tuple(
            point.get("name") if isinstance(point, Mapping) else None
            for point in measured
        )
        != expected_names
    ):
        raise ArtifactError("fixed-rate measured endpoint set is incomplete")
    point_by_name: dict[str, Mapping[str, Any]] = {}
    for index, point in enumerate(measured):
        expected_q = None if index == 0 else expected_qs[index - 1]
        if (
            not isinstance(point, Mapping)
            or set(point) != point_keys
            or point.get("q") != expected_q
            or not all(
                _finite_number(point.get(name)) and float(point[name]) >= 0.0
                for name in (
                    "packet_bits_per_token",
                    "side_information_bits_per_token",
                    "total_bits_per_token",
                    "raw_space_fvu",
                )
            )
            or not math.isclose(
                float(point["side_information_bits_per_token"]),
                float(side["bits_per_token"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                float(point["total_bits_per_token"]),
                float(point["packet_bits_per_token"])
                + float(point["side_information_bits_per_token"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or (index == 0 and point["packet_bits_per_token"] != 0.0)
            or (index == 0 and point["raw_space_fvu"] != 1.0)
            or (index > 0 and point["packet_bits_per_token"] <= 0.0)
            or (
                index > 1
                and float(point["packet_bits_per_token"])
                < float(measured[index - 1]["packet_bits_per_token"])
            )
        ):
            raise ArtifactError("fixed-rate measured endpoint is invalid")
        point_by_name[str(point["name"])] = point
    expected_hull = _fixed_rate_lower_envelope(measured)
    if fixed_rate.get("lower_convex_envelope") != expected_hull:
        raise ArtifactError("fixed-rate lower convex envelope is inconsistent")

    budgets = tuple(
        float(item) for item in values["evaluation.fixed_rate_budgets_bits_per_token"]
    )
    rows = fixed_rate.get("fixed_budgets")
    row_keys = {
        "budget_bits_per_token",
        "eligible",
        "raw_space_fvu",
        "bracket",
        "upper_mixture_weight",
        "achieved_total_bits_per_token",
        "operating_record",
        "reason",
    }
    if (
        not isinstance(rows, list)
        or len(rows) != len(budgets)
        or any(
            not isinstance(row, Mapping)
            or set(row) != row_keys
            or row.get("budget_bits_per_token") != budget
            or not isinstance(row.get("eligible"), bool)
            for row, budget in zip(rows, budgets, strict=True)
        )
    ):
        raise ArtifactError("fixed-rate budget evidence is noncanonical")

    policy = fixed_rate.get("operating_policy")
    _validate_fixed_rate_operating_policy(policy, cell=cell)
    aggregation = None if policy is None else policy["aggregation"]
    policy_rows = (
        {}
        if policy is None
        else {float(row["budget_bits_per_token"]): row for row in policy["rows"]}
    )
    eligible_rows = 0
    record_indices: set[int] = set()
    for row, budget in zip(rows, budgets, strict=True):
        if row["eligible"] is not True:
            if any(
                row.get(name) is not None
                for name in (
                    "raw_space_fvu",
                    "bracket",
                    "upper_mixture_weight",
                    "achieved_total_bits_per_token",
                    "operating_record",
                )
            ) or row.get("reason") not in {
                "raw_space_decoder_requires_unpriced_oracle_information",
                "budget_outside_frozen_operating_policy",
            }:
                raise ArtifactError("ineligible fixed-rate budget row is malformed")
            continue
        eligible_rows += 1
        bracket = row.get("bracket")
        record = row.get("operating_record")
        if (
            not isinstance(bracket, list)
            or len(bracket) != 2
            or any(name not in point_by_name for name in bracket)
            or not _finite_number(row.get("raw_space_fvu"))
            or float(row["raw_space_fvu"]) < 0.0
            or not _finite_number(row.get("upper_mixture_weight"))
            or not _finite_number(row.get("achieved_total_bits_per_token"))
            or not isinstance(record, Mapping)
        ):
            raise ArtifactError("eligible fixed-rate budget row is malformed")
        record_keys = {
            "contract",
            "header_bytes",
            "header_layout",
            "artifact_sha256",
            "record_index",
            "record_offset_bytes",
            "record_sha256",
            "binding_magic_u64",
            "horizon_tokens",
            "upper_tokens",
            "rule",
            "per_token_mode_bits",
            "header_bits_per_token",
            "evaluation_tokens",
            "evaluation_upper_tokens",
            "distortion_measurement",
        }
        if set(record) != record_keys:
            raise ArtifactError("fixed-rate operating record is noncanonical")
        upper_tokens = record.get("upper_tokens")
        record_index = record.get("record_index")
        evaluation_tokens = record.get("evaluation_tokens")
        if (
            record.get("contract") != values["codec.time_sharing_schedule_contract"]
            or record.get("header_bytes") != 32
            or record.get("header_layout")
            != "binding_magic_u64,horizon_u64,upper_count_u64,lower_mode_u32,upper_mode_u32"
            or record.get("artifact_sha256")
            != side["deployment_schedule_bundle_sha256"]
            or not _is_sha256_hex(record.get("record_sha256"))
            or type(record_index) is not int
            or record_index < 0
            or record_index in record_indices
            or record.get("record_offset_bytes") != 32 * record_index
            or type(record.get("binding_magic_u64")) is not int
            or not 0 <= record["binding_magic_u64"] < 2**64
            or record.get("horizon_tokens") != horizon
            or type(upper_tokens) is not int
            or not 0 <= upper_tokens < horizon
            or record.get("per_token_mode_bits") != 0
            or not math.isclose(
                float(record.get("header_bits_per_token", math.nan)),
                256.0 / horizon,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or type(evaluation_tokens) is not int
            or not 0 < evaluation_tokens <= horizon
            or record.get("evaluation_upper_tokens")
            != evaluation_tokens * upper_tokens // horizon
            or record.get("rule")
            != (
                "upper iff floor((i+1)*upper_tokens/horizon) > "
                "floor(i*upper_tokens/horizon)"
            )
        ):
            raise ArtifactError("fixed-rate operating record binding is invalid")
        record_indices.add(record_index)
        weight = upper_tokens / horizon
        lower = point_by_name[str(bracket[0])]
        upper = point_by_name[str(bracket[1])]
        distortion_measurement = record.get("distortion_measurement")
        expected_executed_measurement = (
            "executed_balanced_schedule_on_paired_raw_evaluation_rows"
        )
        retained_lower_measurement = (
            "retained_lower_endpoint_from_paired_raw_evaluation_rows"
        )
        achieved = (
            (1.0 - weight) * float(lower["total_bits_per_token"])
            + weight * float(upper["total_bits_per_token"])
            + 256.0 / horizon
        )
        if (
            not math.isclose(
                float(row["upper_mixture_weight"]),
                weight,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                float(row["achieved_total_bits_per_token"]),
                achieved,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or achieved > budget + 1e-12
            or ((bracket[0] == bracket[1]) != (upper_tokens == 0))
            or row.get("reason")
            != (
                "frozen_parent_operating_policy_replayed"
                if aggregation == "worst_seed_frozen_parent"
                else "development_envelope_operating_policy_selected"
            )
        ):
            raise ArtifactError("fixed-rate executed schedule arithmetic is invalid")

        schedule_rate = 256.0 / horizon
        if aggregation == "worst_seed_frozen_parent":
            if distortion_measurement != expected_executed_measurement:
                raise ArtifactError(
                    "frozen fixed-rate evidence has an invalid measurement contract"
                )
            if bracket[0] != bracket[1]:
                lower_rate = float(lower["total_bits_per_token"])
                upper_rate = float(upper["total_bits_per_token"])
                if upper_rate <= lower_rate:
                    raise ArtifactError(
                        "frozen fixed-rate mixture has nonincreasing endpoint rates"
                    )
                expected_upper_tokens = math.floor(
                    ((budget - schedule_rate - lower_rate) / (upper_rate - lower_rate))
                    * horizon
                )
                if upper_tokens != expected_upper_tokens or not (
                    0 < expected_upper_tokens < horizon
                ):
                    raise ArtifactError(
                        "frozen fixed-rate mixture is not the maximal fitting schedule"
                    )
        else:
            available_rate = budget - schedule_rate
            if available_rate < float(expected_hull[0]["total_bits_per_token"]):
                raise ArtifactError(
                    "development fixed-rate row is eligible below its envelope"
                )
            if available_rate >= float(expected_hull[-1]["total_bits_per_token"]):
                planned_lower = planned_upper = expected_hull[-1]
                planned_upper_tokens = 0
            else:
                upper_index = next(
                    index
                    for index, point in enumerate(expected_hull)
                    if float(point["total_bits_per_token"]) >= available_rate
                )
                exact_endpoint = expected_hull[upper_index]
                if math.isclose(
                    float(exact_endpoint["total_bits_per_token"]),
                    available_rate,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    planned_lower = planned_upper = exact_endpoint
                    planned_upper_tokens = 0
                else:
                    planned_lower = expected_hull[max(0, upper_index - 1)]
                    planned_upper = exact_endpoint
                    planned_upper_tokens = math.floor(
                        (
                            (
                                available_rate
                                - float(planned_lower["total_bits_per_token"])
                            )
                            / (
                                float(planned_upper["total_bits_per_token"])
                                - float(planned_lower["total_bits_per_token"])
                            )
                        )
                        * horizon
                    )
                    if planned_upper_tokens <= 0:
                        planned_upper = planned_lower
                        planned_upper_tokens = 0
            planned_names = [planned_lower["name"], planned_upper["name"]]
            if bracket == planned_names and upper_tokens == planned_upper_tokens:
                if distortion_measurement != expected_executed_measurement:
                    raise ArtifactError(
                        "development fixed-rate schedule has an invalid measurement contract"
                    )
                if bracket[0] != bracket[1] and not (
                    float(row["raw_space_fvu"]) < float(lower["raw_space_fvu"])
                ):
                    raise ArtifactError(
                        "development mixture failed to improve on its lower endpoint"
                    )
            elif (
                planned_lower["name"] != planned_upper["name"]
                and bracket == [planned_lower["name"], planned_lower["name"]]
                and upper_tokens == 0
            ):
                if (
                    distortion_measurement != retained_lower_measurement
                    or not math.isclose(
                        float(row["raw_space_fvu"]),
                        float(planned_lower["raw_space_fvu"]),
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                ):
                    raise ArtifactError(
                        "development lower-endpoint fallback is inconsistent"
                    )
            else:
                raise ArtifactError(
                    "development fixed-rate schedule was not selected from the envelope"
                )
        if bracket[0] == bracket[1] and not math.isclose(
            float(row["raw_space_fvu"]),
            float(lower["raw_space_fvu"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ArtifactError(
                "pure fixed-rate schedule differs from its measured endpoint"
            )
        policy_row = policy_rows.get(budget)
        expected_policy_row = {
            "budget_bits_per_token": budget,
            "lower_name": bracket[0],
            "lower_q": lower["q"],
            "upper_name": bracket[1],
            "upper_q": upper["q"],
        }
        if policy is not None and policy_row != expected_policy_row:
            raise ArtifactError("fixed-rate policy row differs from executed evidence")

    eligible = bool(rows) and eligible_rows == len(rows)
    if fixed_rate.get("eligible") is not eligible:
        raise ArtifactError(
            "fixed-rate eligibility disagrees with exact budget coverage"
        )
    expects_frozen_policy = (
        cell.phase is Phase.PHASE3
        or cell.decision_map.get("evaluation.split") == "confirmation"
    )
    if expects_frozen_policy:
        if policy is None or aggregation != "worst_seed_frozen_parent":
            raise ArtifactError("frozen fixed-rate evidence lacks its parent policy")
    elif (policy is not None) is not eligible:
        raise ArtifactError("development policy presence disagrees with eligibility")
    if record_count != eligible_rows or record_indices != set(range(record_count)):
        raise ArtifactError("fixed-rate record count/index coverage is inconsistent")
    expected_reason = (
        "eligible"
        if eligible
        else (
            "raw_space_decoder_requires_unpriced_oracle_information"
            if rows
            and all(
                row.get("reason")
                == "raw_space_decoder_requires_unpriced_oracle_information"
                for row in rows
            )
            else "one_or_more_budgets_outside_measured_envelope"
        )
    )
    if fixed_rate.get("reason") != expected_reason:
        raise ArtifactError("fixed-rate aggregate reason is inconsistent")
    expected_score = (
        -sum(float(row["raw_space_fvu"]) for row in rows) / len(rows)
        if eligible
        else None
    )
    if (expected_score is None and fixed_rate.get("selection_score") is not None) or (
        expected_score is not None
        and (
            not _finite_number(fixed_rate.get("selection_score"))
            or not math.isclose(
                float(fixed_rate["selection_score"]),
                expected_score,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )
    ):
        raise ArtifactError("fixed-rate selection score is inconsistent")
    expected_validation_score = (
        PHASE2_INELIGIBLE_SELECTION_SCORE if expected_score is None else expected_score
    )
    if (
        not isinstance(validation, Mapping)
        or set(validation) != {PHASE2_SELECTION_METRIC_KEY}
        or not _finite_number(validation.get(PHASE2_SELECTION_METRIC_KEY))
        or not math.isclose(
            float(validation[PHASE2_SELECTION_METRIC_KEY]),
            expected_validation_score,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ArtifactError("selection validation differs from fixed-rate score")


def _validate_qualification_payload(
    payload: Mapping[str, Any],
    *,
    cell: CellSpec,
    expected_artifact_hashes: Mapping[str, str] | None = None,
    expected_implementation_identity: Mapping[str, Any] | None = None,
    evaluation: Mapping[str, Any] | None = None,
) -> str:
    """Replay the complete qualification contract without filesystem access.

    The returned value is the authenticated implementation-identity digest.
    Live campaign gates supply exact artifact hashes, the preparation identity,
    and the evaluation payload.  Detached decision parsers supply every piece
    embedded in their evidence envelope and therefore exercise the same
    semantic checks rather than treating a self-consistent rehash as approval.
    """

    cell_id = cell.cell_id
    if payload.get("schema") != QUALIFICATION_SCHEMA:
        raise ArtifactError("qualification artifact has the wrong schema")
    if set(payload) != set(QUALIFICATION_KEYS):
        raise ArtifactError("qualification artifact has a noncanonical field set")
    if payload.get("cell_id") != cell_id or payload.get("qualified") is not True:
        raise ArtifactError("qualification artifact does not approve this cell")
    checks = payload.get("checks")
    if not isinstance(checks, Mapping) or set(checks) != set(
        REQUIRED_QUALIFICATION_CHECKS
    ):
        raise ArtifactError("qualification checks must use the exact v3 check set")
    if not all(value is True for value in checks.values()):
        raise ArtifactError("qualification checks must all be true")

    if (
        payload.get("qualification_profile")
        != cell.decision_map["qualification.profile"]
        or payload.get("thresholds_version")
        != cell.decision_map["qualification.thresholds_version"]
        or payload.get("thresholds") != _qualification_thresholds(cell)
    ):
        raise ArtifactError(
            "qualification profile/version/thresholds disagree with the resolved cell"
        )

    scientific_outcome = payload.get("scientific_outcome")
    if not isinstance(scientific_outcome, Mapping) or set(scientific_outcome) != {
        "passed",
        "checks",
        "inapplicable_checks",
        "margins",
    }:
        raise ArtifactError("qualification must report scientific_outcome")
    outcome_checks = scientific_outcome.get("checks")
    if not isinstance(outcome_checks, Mapping) or set(outcome_checks) != set(
        REQUIRED_SCIENTIFIC_OUTCOME_CHECKS
    ):
        raise ArtifactError("scientific_outcome checks must use the exact v3 check set")
    if not all(isinstance(value, bool) for value in outcome_checks.values()):
        raise ArtifactError("scientific_outcome checks must be boolean")
    outcome_passed = scientific_outcome.get("passed")
    if not isinstance(outcome_passed, bool):
        raise ArtifactError("scientific_outcome must decide passed")
    if outcome_passed is not all(outcome_checks.values()):
        raise ArtifactError("scientific_outcome passed disagrees with its checks")
    inapplicable_checks = scientific_outcome.get("inapplicable_checks", {})
    if not isinstance(inapplicable_checks, Mapping) or any(
        check not in outcome_checks
        or outcome_checks[check] is not True
        or not isinstance(reason, str)
        or not reason
        for check, reason in inapplicable_checks.items()
    ):
        raise ArtifactError("scientific_outcome inapplicable checks are malformed")
    expected_inapplicable_checks = (
        {
            "phase1_identification": (
                "token_layer_normalization_is_not_a_fixed_linear_factor_map"
            )
        }
        if cell.phase is Phase.PHASE1
        and cell.decision_map.get("data.normalization") == "layer"
        else (
            {
                "codec_calibration_exclusion": (
                    "phase2_excluded_events_are_priced_in_fixed_rate_distortion"
                ),
                "codec_evaluation_exclusion": (
                    "phase2_excluded_events_are_priced_in_fixed_rate_distortion"
                ),
            }
            if cell.phase is Phase.PHASE2
            else {}
        )
    )
    # Pre-amendment Phase-2 qualifications used the same numeric thresholds
    # but enforced the two codec-exclusion checks. They remain authenticated
    # evidence; the campaign amendment reinterprets only those negative
    # outcomes during selection.
    legacy_phase2_strict = cell.phase is Phase.PHASE2 and not inapplicable_checks
    if (
        dict(inapplicable_checks) != expected_inapplicable_checks
        and not legacy_phase2_strict
    ):
        raise ArtifactError(
            "scientific_outcome inapplicability disagrees with the resolved cell"
        )
    margins = scientific_outcome.get("margins")
    if (
        not isinstance(margins, Mapping)
        or set(margins) != set(REQUIRED_SCIENTIFIC_MARGIN_KEYS)
        or any(
            value is not None
            and (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math_isfinite(value)
            )
            for value in margins.values()
        )
    ):
        raise ArtifactError("scientific_outcome margins must be finite numbers or null")

    promotion_eligible = payload.get("promotion_eligible")
    if not isinstance(promotion_eligible, bool):
        raise ArtifactError("qualification must decide promotion_eligible")
    reasons = payload.get("promotion_ineligible_reasons")
    if promotion_eligible:
        if reasons != []:
            raise ArtifactError(
                "a promotion-eligible qualification cannot name ineligible reasons"
            )
    elif (
        not isinstance(reasons, list)
        or not reasons
        or not all(isinstance(item, str) and item for item in reasons)
    ):
        raise ArtifactError(
            "a diagnostic qualification must name promotion-ineligible reasons"
        )
    protocol_eligible = payload.get("selection_eligible_for_protocol_test")
    eligibility_mode = payload.get("selection_eligibility_mode")
    if not isinstance(protocol_eligible, bool) or eligibility_mode not in {
        "scientific_promotion",
        "smoke_protocol_only",
        "none",
    }:
        raise ArtifactError(
            "qualification must bind its scientific/protocol selection mode"
        )
    is_smoke = cell.decision_map.get("runtime.smoke") is True
    resolved_promotable = cell.decision_map.get("qualification.promotable") is True
    expected_mode = (
        "scientific_promotion"
        if promotion_eligible
        else "smoke_protocol_only"
        if protocol_eligible
        else "none"
    )
    if eligibility_mode != expected_mode:
        raise ArtifactError("qualification selection mode disagrees with eligibility")
    if promotion_eligible and not resolved_promotable:
        raise ArtifactError(
            "qualification cannot override the cell's immutable nonpromotable recipe"
        )
    if promotion_eligible and not outcome_passed:
        raise ArtifactError(
            "a scientifically failed qualification cannot be promotion eligible"
        )
    if promotion_eligible and is_smoke:
        raise ArtifactError("a smoke qualification cannot be promotion eligible")
    if protocol_eligible and (
        promotion_eligible or not is_smoke or not resolved_promotable
    ):
        raise ArtifactError(
            "smoke protocol eligibility is inconsistent with the resolved cell"
        )
    expected_protocol_eligible = bool(is_smoke and resolved_promotable)
    if protocol_eligible is not expected_protocol_eligible:
        raise ArtifactError(
            "qualification protocol eligibility disagrees with the resolved cell"
        )

    inputs = payload.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != set(QUALIFICATION_INPUT_KINDS):
        raise ArtifactError("qualification artifact must bind its exact input-hash set")
    if not all(_is_sha256_hex(value) for value in inputs.values()):
        raise ArtifactError(
            "qualification input hashes must be 64 lowercase hex characters"
        )
    if expected_artifact_hashes is not None:
        expected = {
            kind: expected_artifact_hashes[kind] for kind in QUALIFICATION_INPUT_KINDS
        }
        if dict(inputs) != expected:
            raise ArtifactError(
                "qualification input binding mismatch: "
                + canonical_json({"expected": expected, "actual": inputs})
            )

    implementation_identity = payload.get("implementation_identity")
    implementation_identity_sha256 = payload.get("implementation_identity_sha256")
    if not isinstance(implementation_identity, Mapping) or not _is_sha256_hex(
        implementation_identity_sha256
    ):
        raise ArtifactError(
            "qualification must bind its complete implementation identity"
        )
    observed_implementation_sha256 = _validate_implementation_identity(
        implementation_identity,
        scientific=not is_smoke,
    )
    if implementation_identity_sha256 != observed_implementation_sha256:
        raise ArtifactError("qualification implementation-identity hash mismatch")
    if expected_implementation_identity is not None and dict(
        implementation_identity
    ) != dict(expected_implementation_identity):
        raise ArtifactError(
            "qualification implementation identity differs from preparation"
        )

    selection_metrics = payload.get("selection_metrics")
    selection_metrics_sha256 = payload.get("selection_metrics_sha256")
    if not isinstance(selection_metrics, Mapping) or not _is_sha256_hex(
        selection_metrics_sha256
    ):
        raise ArtifactError("qualification must bind its selection metrics")
    if selection_metrics_sha256 != _sha256_canonical_payload(selection_metrics):
        raise ArtifactError("qualification selection-metrics hash mismatch")
    if payload.get("selection_metrics_evaluation_sha256") != inputs["evaluation"]:
        raise ArtifactError(
            "selection metrics are not bound to the evaluation artifact"
        )
    if payload.get("validation") != selection_metrics.get("validation"):
        raise ArtifactError(
            "qualification validation differs from bound selection metrics"
        )
    fixed_rate_policy = payload.get("fixed_rate_operating_policy")
    _validate_fixed_rate_operating_policy(fixed_rate_policy, cell=cell)
    bound_fixed_rate = selection_metrics.get("fixed_rate_raw_selection")
    if not isinstance(bound_fixed_rate, Mapping):
        raise ArtifactError("selection metrics lack fixed-rate evidence")
    _validate_fixed_rate_evidence(
        bound_fixed_rate,
        cell=cell,
        validation=selection_metrics.get("validation"),
        inputs=inputs,
    )
    expected_policy = bound_fixed_rate.get("operating_policy")
    if fixed_rate_policy != expected_policy:
        raise ArtifactError(
            "qualification operating policy differs from bound selection metrics"
        )
    if cell.phase is Phase.PHASE1 and fixed_rate_policy is not None:
        raise ArtifactError("Phase-1 qualification cannot carry an operating policy")
    if (
        cell.phase is not Phase.PHASE1
        and bound_fixed_rate.get("eligible") is True
        and fixed_rate_policy is None
    ):
        raise ArtifactError("eligible real-data evidence lacks an operating policy")
    if cell.phase is Phase.PHASE1:
        native_identification, deployed_identification, phase1_validation = (
            _validated_phase1_identification(selection_metrics, cell=cell)
        )
        recovery = selection_metrics.get("recovery")
        site_dims = cell.decision_map.get("data.site_dims")
        shared_feature_claim_required = bool(
            isinstance(site_dims, (list, tuple)) and len(site_dims) > 1
        )
        shared_feature_claim_eligible = bool(
            shared_feature_claim_required
            and cell.decision_map.get("data.dgp_step")
            not in {"shared_support", "support_only"}
            and cell.decision_map.get("data.site_presence_span") != "one"
        )
        if not isinstance(recovery, Mapping) or set(recovery) != {
            "native",
            "deployed",
        }:
            raise ArtifactError(
                "Phase-1 selection metrics require exact native/deployed recovery"
            )
        for endpoint_name in ("native", "deployed"):
            endpoint_recovery = recovery.get(endpoint_name)
            if (
                not isinstance(endpoint_recovery, Mapping)
                or endpoint_recovery.get("shared_feature_claim_required")
                is not shared_feature_claim_required
                or endpoint_recovery.get("shared_feature_claim_eligible")
                is not shared_feature_claim_eligible
            ):
                raise ArtifactError(
                    f"Phase-1 {endpoint_name} recovery disagrees with the "
                    "resolved shared-feature claim"
                )
        expected_sensitivity = _expected_phase1_threshold_sensitivity(
            {
                "native": native_identification,
                "deployed": deployed_identification,
            },
            cell,
        )
        if (
            selection_metrics.get("phase1_threshold_sensitivity")
            != expected_sensitivity
        ):
            raise ArtifactError(
                "Phase-1 threshold sensitivity does not execute the declared grid"
            )
        both_inapplicable = (
            native_identification["applicable"] is False
            and deployed_identification["applicable"] is False
        )
        expected_identification_outcome = bool(
            (not shared_feature_claim_required or shared_feature_claim_eligible)
            and (
                both_inapplicable
                or (
                    native_identification["passed"] is True
                    and deployed_identification["passed"] is True
                    and phase1_validation["phase1_identification_conjunction"] is True
                )
            )
        )
        if (
            outcome_checks.get("phase1_identification")
            is not expected_identification_outcome
            or margins.get("phase1_native_identification")
            != native_identification.get("margin")
            or margins.get("phase1_deployed_identification")
            != deployed_identification.get("margin")
        ):
            raise ArtifactError(
                "Phase-1 scientific outcome disagrees with identification evidence"
            )
    elif selection_metrics.get("phase1_threshold_sensitivity") is not None:
        raise ArtifactError(
            "real-data selection metrics cannot carry Phase-1 sensitivity"
        )

    if evaluation is not None:
        expected_evaluation_inputs = {
            kind: inputs[kind]
            for kind in (
                "checkpoint",
                "calibration",
                "deployment_codec",
                "deployment_schedules",
            )
        }
        if (
            evaluation.get("schema") != EVALUATION_SCHEMA
            or evaluation.get("evaluation_execution_implementation")
            != EVALUATION_EXECUTION_IMPLEMENTATION
            or evaluation.get("cell_id") != cell_id
            or evaluation.get("inputs") != expected_evaluation_inputs
        ):
            raise ArtifactError(
                "evaluation artifact schema/cell/input binding mismatch"
            )
        evaluation_metrics = evaluation.get("selection_metrics")
        evaluation_metrics_sha256 = evaluation.get("selection_metrics_sha256")
        if not isinstance(evaluation_metrics, Mapping) or not _is_sha256_hex(
            evaluation_metrics_sha256
        ):
            raise ArtifactError("evaluation lacks authenticated selection metrics")
        if evaluation_metrics_sha256 != _sha256_canonical_payload(evaluation_metrics):
            raise ArtifactError("evaluation selection-metrics hash mismatch")
        if (
            selection_metrics != evaluation_metrics
            or selection_metrics_sha256 != evaluation_metrics_sha256
        ):
            raise ArtifactError(
                "qualification selection metrics differ from the bound evaluation"
            )
        if evaluation.get("validation") != evaluation_metrics.get("validation"):
            raise ArtifactError(
                "evaluation validation differs from bound selection metrics"
            )
        if evaluation.get("phase1_threshold_sensitivity") != evaluation_metrics.get(
            "phase1_threshold_sensitivity"
        ):
            raise ArtifactError(
                "evaluation threshold sensitivity differs from bound selection metrics"
            )
        if evaluation.get("fixed_rate_raw_selection") != bound_fixed_rate:
            raise ArtifactError(
                "evaluation fixed-rate evidence differs from bound selection metrics"
            )
        expected_reasons = _promotion_reasons_from_evidence(
            cell,
            outcome_passed=outcome_passed,
            evaluation=evaluation,
        )
        if reasons != expected_reasons or promotion_eligible is not bool(
            not expected_reasons
        ):
            raise ArtifactError(
                "qualification promotion eligibility/reasons differ from bound evidence"
            )
    else:
        allowed_reasons = {
            "runtime_smoke",
            "raw_codec_requires_unpriced_side_information",
            "fixed_rate_budget_ineligible",
            "synthetic_shared_feature_claim_ineligible",
            "resolved_nonpromotable_cell",
            "scientific_outcome_failed",
            "missing_frozen_phase2_selection_decision",
        }
        if any(reason not in allowed_reasons for reason in reasons):
            raise ArtifactError("qualification names an unknown promotion reason")
        mandatory_reasons = {
            *(("runtime_smoke",) if is_smoke else ()),
            *(("resolved_nonpromotable_cell",) if not resolved_promotable else ()),
            *(("scientific_outcome_failed",) if not outcome_passed else ()),
            *(
                ("missing_frozen_phase2_selection_decision",)
                if (cell.phase is Phase.PHASE3 or "confirmation" in cell.stage)
                and not cell.decision_map["selection.parent_cell_ids"]
                else ()
            ),
        }
        if not mandatory_reasons.issubset(reasons):
            raise ArtifactError(
                "qualification omits a cell-derived promotion-ineligibility reason"
            )
    return observed_implementation_sha256


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    durable_mkdir(path.parent, parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        tmp = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        durable_replace(tmp, path, file_already_synced=True)
    finally:
        if tmp.exists():
            tmp.unlink()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CampaignError(f"expected a JSON object at {path}")
    return payload


def _write_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    durable_mkdir(path.parent, parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            durable_create(temporary, path, file_already_synced=True)
            temporary = None
        except FileExistsError:
            existing = _read_json(path)
            if canonical_json(existing) != canonical_json(payload):
                raise CampaignError(
                    f"immutable decision already exists with different content: {path}"
                )
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _campaign_output_path(
    root: Path,
    requested: str | Path | None,
    default: Path,
) -> Path:
    if requested is None:
        destination = default
    else:
        destination = Path(requested)
        if not destination.is_absolute():
            destination = root / destination
    root_resolved = root.resolve()
    try:
        destination.parent.resolve().relative_to(root_resolved)
    except ValueError as exc:
        raise CampaignError(
            "campaign decisions must be written inside campaign root"
        ) from exc
    if destination.exists():
        try:
            destination.resolve().relative_to(root_resolved)
        except ValueError as exc:
            raise CampaignError(
                "campaign decision path resolves outside campaign root"
            ) from exc
    return destination


@dataclass(frozen=True, slots=True)
class _ArtifactFingerprint:
    device: int
    inode: int
    size_bytes: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def from_path(cls, path: Path) -> "_ArtifactFingerprint":
        stat = path.stat()
        return cls(
            device=stat.st_dev,
            inode=stat.st_ino,
            size_bytes=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
            ctime_ns=stat.st_ctime_ns,
        )


@dataclass(frozen=True, slots=True)
class _ArtifactVerification:
    """Unserialized proof that this process hashed one exact file instance."""

    issuer: object
    key: tuple[str, str, int]
    fingerprint: _ArtifactFingerprint
    executor_cell_id: str | None = None
    executor_stage: str | None = None
    canonical_executor: bool = False


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    kind: str
    path: str
    sha256: str
    size_bytes: int
    _verification: _ArtifactVerification | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not self.kind or "/" in self.kind:
            raise ArtifactError(f"invalid artifact kind {self.kind!r}")
        if len(self.sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.sha256
        ):
            raise ArtifactError("artifact sha256 must be 64 lowercase hex characters")
        if self.size_bytes < 0:
            raise ArtifactError("artifact size must be non-negative")

    @classmethod
    def from_path(cls, kind: str, path: Path, *, root: Path) -> "ArtifactRef":
        resolved = path.resolve()
        if not resolved.is_file():
            raise ArtifactError(f"artifact does not exist or is not a file: {path}")
        try:
            stored_path = str(resolved.relative_to(root.resolve()))
        except ValueError as exc:
            raise ArtifactError(
                "campaign artifact is outside the campaign root"
            ) from exc
        return cls(kind, stored_path, _sha256(resolved), resolved.stat().st_size)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArtifactRef":
        return cls(
            kind=str(payload["kind"]),
            path=str(payload["path"]),
            sha256=str(payload["sha256"]),
            size_bytes=int(payload["size_bytes"]),
        )

    def resolve(self, root: Path) -> Path:
        path = Path(self.path)
        if path.is_absolute():
            raise ArtifactError("campaign artifact paths must be root-relative")
        resolved = (root / path).resolve()
        try:
            resolved.relative_to(root.resolve())
        except ValueError as exc:
            raise ArtifactError(
                "campaign artifact path escapes the campaign root"
            ) from exc
        return resolved

    def verify(self, root: Path) -> None:
        path = self.resolve(root)
        if not path.is_file():
            raise ArtifactError(f"artifact disappeared: {path}")
        actual_size = path.stat().st_size
        if actual_size != self.size_bytes:
            raise ArtifactError(
                f"artifact size mismatch for {path}: {actual_size} != {self.size_bytes}"
            )
        actual_hash = _sha256(path)
        if actual_hash != self.sha256:
            raise ArtifactError(
                f"artifact hash mismatch for {path}: {actual_hash} != {self.sha256}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class CampaignRecord:
    cell_id: str
    state: RunState
    artifacts: tuple[ArtifactRef, ...] = ()
    resume_state: RunState | None = None
    event_count: int = 0
    updated_at: float | None = None

    @property
    def artifact_map(self) -> dict[str, ArtifactRef]:
        return {artifact.kind: artifact for artifact in self.artifacts}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CAMPAIGN_SCHEMA,
            "cell_id": self.cell_id,
            "state": self.state.value,
            "resume_state": None
            if self.resume_state is None
            else self.resume_state.value,
            "event_count": self.event_count,
            "updated_at": self.updated_at,
            "artifacts": [item.to_dict() for item in self.artifacts],
        }


class CellLock(AbstractContextManager["CellLock"]):
    def __init__(self, campaign: "Campaign", cell_id: str):
        self.campaign = campaign
        self.cell_id = cell_id
        self.path = campaign.lock_path(cell_id)
        self.guard_path = campaign.lock_guard_path(cell_id)
        self.held = False
        self.attempt_id = uuid.uuid4().hex
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._guard_handle: Any | None = None
        self._metadata_lock = threading.Lock()
        self._worker_pid: int | None = None
        self._worker_pgid: int | None = None
        self._worker_process_identity: str | None = None

    def _payload(self, acquired_at: float) -> dict[str, Any]:
        return {
            "schema": CAMPAIGN_SCHEMA,
            "cell_id": self.cell_id,
            "attempt_id": self.attempt_id,
            "pid": os.getpid(),
            "owner_process_identity": _process_identity(os.getpid()),
            "host": socket.gethostname(),
            "acquired_at": acquired_at,
            "heartbeat_at": float(self.campaign.clock()),
            "worker_pid": self._worker_pid,
            "worker_pgid": self._worker_pgid,
            "worker_process_identity": self._worker_process_identity,
        }

    def _publish(self, acquired_at: float) -> None:
        with self._metadata_lock:
            _atomic_json(self.path, self._payload(acquired_at))

    def _heartbeat(self, acquired_at: float) -> None:
        interval = self.campaign.lock_heartbeat_seconds
        while not self._stop.wait(interval):
            try:
                self._publish(acquired_at)
            except (CampaignError, FileNotFoundError, OSError):
                return

    def bind_worker(self, *, pid: int, pgid: int) -> None:
        if not self.held:
            raise CampaignError("cannot bind a worker to an unheld cell lock")
        self._worker_pid = int(pid)
        self._worker_pgid = int(pgid)
        self._worker_process_identity = _process_identity(pid)
        self._publish(self._acquired_at)

    def __enter__(self) -> "CellLock":
        durable_mkdir(self.path.parent, parents=True, exist_ok=True)
        acquired_at = float(self.campaign.clock())
        self._acquired_at = acquired_at
        guard_handle = self.guard_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(
                guard_handle.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError as exc:
            guard_handle.close()
            raise CampaignLocked(f"cell is locked: {self.cell_id}") from exc
        if self.path.exists():
            fcntl.flock(guard_handle.fileno(), fcntl.LOCK_UN)
            guard_handle.close()
            raise CampaignLocked(f"cell has an unreconciled lock lease: {self.cell_id}")
        self._guard_handle = guard_handle
        try:
            self._publish(acquired_at)
        except Exception:
            fcntl.flock(guard_handle.fileno(), fcntl.LOCK_UN)
            guard_handle.close()
            self._guard_handle = None
            raise
        self.held = True
        self._thread = threading.Thread(
            target=self._heartbeat,
            args=(acquired_at,),
            name=f"bsc-lock-{_slug(self.cell_id)[:24]}",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self.held:
            self._stop.set()
            if self._thread is not None:
                self._thread.join()
            with self._metadata_lock:
                try:
                    current = _read_json(self.path)
                    if current.get("attempt_id") == self.attempt_id:
                        self.path.unlink()
                except (CampaignError, FileNotFoundError):
                    pass
            if self._guard_handle is not None:
                fcntl.flock(self._guard_handle.fileno(), fcntl.LOCK_UN)
                self._guard_handle.close()
                self._guard_handle = None
            self.held = False


class Campaign:
    """A registered study and its authoritative append-only event journal."""

    def __init__(
        self,
        root: str | Path,
        *,
        clock: Callable[[], float] = time.time,
        lock_heartbeat_seconds: float = 30.0,
    ):
        if lock_heartbeat_seconds <= 0:
            raise CampaignError("lock heartbeat interval must be positive")
        self.root = Path(root)
        self.clock = clock
        self.lock_heartbeat_seconds = float(lock_heartbeat_seconds)
        self.journal_path = self.root / "journal.jsonl"
        self.plan_path = self.root / "plan.json"
        self.plans_dir = self.root / "plans"
        self.blueprint_path = self.root / "blueprint.json"
        self.phase1_decision_path = self.root / "phase1-decision.json"
        self.panel_decision_path = self.root / "panel-decision.json"
        self.amendments_dir = self.root / "amendments"
        self.implementation_identity_path = self.root / "implementation-identity.json"
        self.implementation_identity_lock_path = self.root / ".implementation.lock"
        self.mutation_lock_path = self.root / ".mutation.lock"
        self.activation_identity_path = self.root / "activation-identity.json"
        self._events_cache: tuple[dict[str, Any], ...] | None = None
        self._events_by_cell_cache: dict[str, tuple[dict[str, Any], ...]] = {}
        self._events_cache_signature: tuple[int, int, int, int] | None = None
        self._events_cache_lock = threading.RLock()
        self._mutation_thread_lock = threading.RLock()
        self._mutation_local = threading.local()
        self._authority_lock = threading.RLock()
        self._activation_projection_lock = threading.RLock()
        self._history_plan_cache: dict[
            str, tuple[_ArtifactFingerprint, StudyPlan]
        ] = {}
        self._authority_cursor = 0
        self._authority_last_event_id: str | None = None
        self._authority_active_plan_id: str | None = None
        self._authority_registration_plan_ids: dict[str, str] = {}
        self._authority_extension_evidence: list[ArtifactRef] = []
        self._cell_projection_cache: dict[
            str, tuple[_ArtifactFingerprint, CellSpec]
        ] = {}
        self._activation_projection_cursor = 0
        self._activation_projection_last_event_id: str | None = None
        self._activation_projection_raw_digest: str | None = None
        self._activation_projection_views: dict[str, str] = {}
        # Verification receipts are intentionally process-local.  They are
        # never journaled, and their opaque issuer prevents a new Campaign
        # instance from accepting a token inherited from an older runner.
        self._artifact_verification_issuer = object()
        self._artifact_verification_cache: dict[
            tuple[str, str, int], _ArtifactFingerprint
        ] = {}

    @staticmethod
    def _artifact_cache_key(
        artifact: ArtifactRef,
        path: Path,
    ) -> tuple[str, str, int]:
        return (str(path), artifact.sha256, artifact.size_bytes)

    def _verify_artifact(self, artifact: ArtifactRef) -> ArtifactRef:
        """Authenticate an artifact once per unchanged file instance.

        Size, device, inode, mtime, and ctime form the reuse guard.  Any
        difference discards the receipt and performs a full content hash.
        A before/after fingerprint also refuses files changed during hashing.
        """

        campaign_root = self.root.resolve()
        path = artifact.resolve(self.root).resolve()
        try:
            path.relative_to(campaign_root)
        except ValueError as exc:
            raise ArtifactError(
                f"artifact resolves outside campaign root: {path}"
            ) from exc
        if not path.is_file():
            raise ArtifactError(f"artifact disappeared: {path}")
        try:
            before = _ArtifactFingerprint.from_path(path)
        except OSError as exc:
            raise ArtifactError(f"cannot stat artifact {path}: {exc}") from exc
        if before.size_bytes != artifact.size_bytes:
            raise ArtifactError(
                f"artifact size mismatch for {path}: "
                f"{before.size_bytes} != {artifact.size_bytes}"
            )
        key = self._artifact_cache_key(artifact, path)
        token = artifact._verification
        if (
            token is not None
            and token.issuer is self._artifact_verification_issuer
            and token.key == key
            and token.fingerprint == before
        ):
            self._artifact_verification_cache[key] = before
            return artifact
        if self._artifact_verification_cache.get(key) == before:
            return replace(
                artifact,
                _verification=_ArtifactVerification(
                    self._artifact_verification_issuer,
                    key,
                    before,
                ),
            )
        try:
            actual_hash = _sha256(path)
            after = _ArtifactFingerprint.from_path(path)
        except OSError as exc:
            raise ArtifactError(f"cannot hash artifact {path}: {exc}") from exc
        if after != before:
            raise ArtifactError(f"artifact changed while hashing: {path}")
        if actual_hash != artifact.sha256:
            raise ArtifactError(
                f"artifact hash mismatch for {path}: {actual_hash} != {artifact.sha256}"
            )
        self._artifact_verification_cache[key] = after
        return replace(
            artifact,
            _verification=_ArtifactVerification(
                self._artifact_verification_issuer,
                key,
                after,
            ),
        )

    def _verified_artifact_from_path(self, kind: str, path: Path) -> ArtifactRef:
        resolved = path.resolve()
        if not resolved.is_file():
            raise ArtifactError(f"artifact does not exist or is not a file: {path}")
        try:
            before = _ArtifactFingerprint.from_path(resolved)
            digest = _sha256(resolved)
            after = _ArtifactFingerprint.from_path(resolved)
        except OSError as exc:
            raise ArtifactError(f"cannot hash artifact {resolved}: {exc}") from exc
        if after != before:
            raise ArtifactError(f"artifact changed while hashing: {resolved}")
        try:
            stored_path = str(resolved.relative_to(self.root.resolve()))
        except ValueError as exc:
            raise ArtifactError(
                "campaign artifact is outside the campaign root"
            ) from exc
        ref = ArtifactRef(kind, stored_path, digest, after.size_bytes)
        key = self._artifact_cache_key(ref, resolved)
        self._artifact_verification_cache[key] = after
        return replace(
            ref,
            _verification=_ArtifactVerification(
                self._artifact_verification_issuer,
                key,
                after,
            ),
        )

    def _authorize_executor_stage_artifacts(
        self,
        cell_id: str,
        stage: str,
        artifacts: Iterable[ArtifactRef],
        *,
        canonical_executor: bool,
    ) -> tuple[ArtifactRef, ...]:
        """Attach an unjournaled receipt for one completed executor stage.

        The canonical child validates its own durable outputs before emitting
        the stage manifest.  This receipt lets Campaign admission reuse that
        validation without loading a second checkpoint/model concurrently.
        """

        expected = EXPECTED_STAGE_ARTIFACTS.get(stage)
        if expected is None:
            raise ArtifactError(f"unknown executor stage {stage!r}")
        verified = tuple(self._verify_artifact(item) for item in artifacts)
        observed = {item.kind for item in verified}
        if observed != {*expected, f"{stage}_manifest"}:
            raise ArtifactError(
                f"executor receipt for {stage} has wrong artifact kinds: "
                f"{sorted(observed)}"
            )
        authorized: list[ArtifactRef] = []
        for artifact in verified:
            path = artifact.resolve(self.root).resolve()
            fingerprint = _ArtifactFingerprint.from_path(path)
            key = self._artifact_cache_key(artifact, path)
            token = artifact._verification
            if (
                token is None
                or token.issuer is not self._artifact_verification_issuer
                or token.key != key
                or token.fingerprint != fingerprint
            ):
                raise ArtifactError(
                    f"executor artifact changed before admission: {path}"
                )
            authorized.append(
                replace(
                    artifact,
                    _verification=_ArtifactVerification(
                        self._artifact_verification_issuer,
                        key,
                        fingerprint,
                        executor_cell_id=cell_id,
                        executor_stage=stage,
                        canonical_executor=canonical_executor,
                    ),
                )
            )
        return tuple(authorized)

    @contextmanager
    def _campaign_mutation(self):
        """Serialize journal-dependent validation and its commit across processes."""

        with self._mutation_thread_lock:
            depth = int(getattr(self._mutation_local, "depth", 0))
            if depth:
                self._mutation_local.depth = depth + 1
                try:
                    yield
                finally:
                    self._mutation_local.depth = depth
                return
            durable_mkdir(self.root, parents=True, exist_ok=True)
            with self.mutation_lock_path.open("a+", encoding="utf-8") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                self._mutation_local.depth = 1
                try:
                    yield
                finally:
                    self._mutation_local.depth = 0
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @contextmanager
    def _registration_mutation(self):
        """Acquire campaign-wide locks in their single canonical order."""

        durable_mkdir(self.root, parents=True, exist_ok=True)
        registration_lock = self.root / ".registration.lock"
        with registration_lock.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            with self._campaign_mutation():
                yield

    def _history_plan(self, plan_id: str) -> StudyPlan:
        if not plan_id:
            raise CampaignError("journal plan identity is empty")
        path = self.plans_dir / f"{_slug(plan_id)}.json"
        if not path.is_file():
            raise CampaignError(
                f"journal-bound plan lacks immutable history {plan_id}"
            )
        try:
            before = _ArtifactFingerprint.from_path(path)
            cached = self._history_plan_cache.get(plan_id)
            if cached is not None and cached[0] == before:
                return cached[1]
            payload = _read_json(path)
            plan = StudyPlan.from_manifest(payload)
            after = _ArtifactFingerprint.from_path(path)
        except (KeyError, OSError, TypeError, ValueError, StudyError) as exc:
            raise CampaignError(
                f"invalid immutable plan history for {plan_id}: {exc}"
            ) from exc
        if before != after:
            raise CampaignError(
                f"immutable plan history changed while reading {plan_id}"
            )
        if plan.plan_id != plan_id or canonical_json(payload) != canonical_json(
            plan.to_manifest()
        ):
            raise CampaignError(
                f"immutable plan history content mismatch for {plan_id}"
            )
        self._history_plan_cache[plan_id] = (after, plan)
        return plan

    def _validate_plan_extension_event(
        self,
        current: StudyPlan,
        event: Mapping[str, Any],
    ) -> tuple[StudyPlan, ArtifactRef]:
        metadata = event.get("metadata")
        if not isinstance(metadata, Mapping):
            raise CampaignError("plan-extension metadata must be an object")
        if metadata.get("previous_plan_id") != current.plan_id:
            raise CampaignError(
                "plan-extension journal does not form one ordered plan chain"
            )
        target_plan_id = metadata.get("plan_id")
        if not isinstance(target_plan_id, str) or not target_plan_id:
            raise CampaignError("plan-extension event lacks its target plan ID")
        target = self._history_plan(target_plan_id)
        if (
            target.phase is not current.phase
            or len(target.stages) != len(current.stages) + 1
            or target.stages[:-1] != current.stages
            or metadata.get("stage") != target.stages[-1].name
        ):
            raise CampaignError(
                "committed plan extension differs from immutable plan history"
            )
        raw_artifacts = event.get("artifacts")
        if not isinstance(raw_artifacts, list) or len(raw_artifacts) != 1:
            raise CampaignError(
                "committed plan extension must bind exactly one selection artifact"
            )
        try:
            evidence = self._verify_artifact(ArtifactRef.from_dict(raw_artifacts[0]))
        except (ArtifactError, KeyError, TypeError, ValueError) as exc:
            raise CampaignError(
                f"committed plan extension has invalid artifact evidence: {exc}"
            ) from exc
        return target, evidence

    def _journal_authority(self) -> tuple[StudyPlan, Mapping[str, str]]:
        """Return journal-derived active plan and registered-cell provenance.

        The cursor processes normal appends once.  If the observed prefix ever
        changes, authority is rebuilt from the complete journal.
        """

        with self._authority_lock:
            return self._journal_authority_locked()

    def _journal_authority_locked(self) -> tuple[StudyPlan, Mapping[str, str]]:

        events = self._events_cached()
        prefix_valid = (
            self._authority_cursor == 0
            or (
                len(events) >= self._authority_cursor
                and events[self._authority_cursor - 1].get("event_id")
                == self._authority_last_event_id
            )
        )
        if not prefix_valid:
            self._authority_cursor = 0
            self._authority_last_event_id = None
            self._authority_active_plan_id = None
            self._authority_registration_plan_ids = {}
            self._authority_extension_evidence = []
        active = (
            None
            if self._authority_active_plan_id is None
            else self._history_plan(self._authority_active_plan_id)
        )
        for evidence in self._authority_extension_evidence:
            self._verify_artifact(evidence)
        for event in events[self._authority_cursor :]:
            event_type = event.get("event")
            if event_type == "transition" and event.get("previous") is None:
                if event.get("target") != RunState.PLANNED.value:
                    raise CampaignError(
                        "cell registration must enter the planned state"
                    )
                cell_id = event.get("cell_id")
                metadata = event.get("metadata")
                plan_id = (
                    metadata.get("plan_id")
                    if isinstance(metadata, Mapping)
                    else None
                )
                if (
                    not isinstance(cell_id, str)
                    or not cell_id
                    or not isinstance(plan_id, str)
                    or not plan_id
                ):
                    raise CampaignError(
                        "cell registration lacks journal-bound plan provenance"
                    )
                prior = self._authority_registration_plan_ids.get(cell_id)
                if prior is not None and prior != plan_id:
                    raise CampaignError(
                        f"cell {cell_id} is registered to conflicting plans"
                    )
                history = self._history_plan(plan_id)
                if not any(cell.cell_id == cell_id for cell in history.cells):
                    raise CampaignError(
                        f"registered cell {cell_id} is absent from plan {plan_id}"
                    )
                self._authority_registration_plan_ids[cell_id] = plan_id
                if active is None:
                    active = history
            elif event_type == "plan_extension":
                if active is None:
                    raise CampaignError("plan extension precedes initial registration")
                active, evidence = self._validate_plan_extension_event(active, event)
                self._authority_extension_evidence.append(evidence)
        self._authority_cursor = len(events)
        self._authority_last_event_id = (
            None if not events else str(events[-1].get("event_id"))
        )
        self._authority_active_plan_id = None if active is None else active.plan_id
        if active is None:
            raise CampaignError("campaign journal has no registered plan")
        return active, dict(self._authority_registration_plan_ids)

    @property
    def plan(self) -> StudyPlan:
        if self._events_cached():
            return self._journal_authority()[0]
        if not self.plan_path.exists():
            raise CampaignError(f"campaign has no plan: {self.plan_path}")
        try:
            payload = _read_json(self.plan_path)
            plan = StudyPlan.from_manifest(payload)
        except (KeyError, OSError, TypeError, ValueError, StudyError) as exc:
            raise CampaignError(f"active plan projection is invalid: {exc}") from exc
        if canonical_json(payload) != canonical_json(plan.to_manifest()):
            raise CampaignError("active plan projection is noncanonical")
        return plan

    def cell_dir(self, cell_id: str) -> Path:
        return self.root / "cells" / _slug(cell_id)

    def cell_manifest_path(self, cell_id: str) -> Path:
        return self.cell_dir(cell_id) / "cell.json"

    def state_path(self, cell_id: str) -> Path:
        return self.cell_dir(cell_id) / "state.json"

    def lock_path(self, cell_id: str) -> Path:
        return self.root / ".locks" / f"{_slug(cell_id)}.lock"

    def lock_guard_path(self, cell_id: str) -> Path:
        return self.root / ".locks" / f"{_slug(cell_id)}.guard"

    def lock(self, cell_id: str) -> CellLock:
        self._require_active_cell(cell_id)
        return CellLock(self, cell_id)

    def register(
        self,
        plan: StudyPlan,
        *,
        blueprint_manifest: Mapping[str, Any] | None = None,
        phase1_decision_manifest: Mapping[str, Any] | None = None,
        panel_decision_manifest: Mapping[str, Any] | None = None,
    ) -> None:
        """Register a plan idempotently; a different plan fails closed."""

        durable_mkdir(self.root, parents=True, exist_ok=True)
        registration_lock = self.root / ".registration.lock"
        with registration_lock.open("a+", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            with self._campaign_mutation():
                self._register_locked(
                    plan,
                    blueprint_manifest=blueprint_manifest,
                    phase1_decision_manifest=phase1_decision_manifest,
                    panel_decision_manifest=panel_decision_manifest,
                )

    def _register_locked(
        self,
        plan: StudyPlan,
        *,
        blueprint_manifest: Mapping[str, Any] | None = None,
        phase1_decision_manifest: Mapping[str, Any] | None = None,
        panel_decision_manifest: Mapping[str, Any] | None = None,
    ) -> None:
        if blueprint_manifest is None:
            raise CampaignError(
                "campaign registration requires an exact frozen blueprint manifest"
            )
        if panel_decision_manifest is not None and (plan.phase.value != "phase3"):
            raise CampaignError(
                "a panel decision may only register an exact Phase 3 blueprint"
            )
        if phase1_decision_manifest is not None and plan.phase.value != "phase2":
            raise CampaignError(
                "a Phase-1 decision may only authorize an exact Phase-2 blueprint"
            )
        if plan.phase.value == "phase1":
            blueprint = Phase1Blueprint.from_manifest(blueprint_manifest)
            smoke_values = {
                cell.decision_map.get("runtime.smoke") for cell in plan.cells
            }
            if smoke_values not in ({False}, {True}):
                raise CampaignError(
                    "Phase-1 registration has inconsistent smoke provenance"
                )
            phase1_smoke = smoke_values == {True}
            try:
                expected_blueprint = build_phase1_blueprint(
                    blueprint.seeds,
                    smoke=phase1_smoke,
                )
                expected_plan = build_phase1_plan(
                    blueprint.seeds,
                    smoke=phase1_smoke,
                )
            except StudyError as exc:
                raise CampaignError(
                    f"cannot derive canonical Phase-1 campaign: {exc}"
                ) from exc
            if blueprint != expected_blueprint or plan != expected_plan:
                raise CampaignError(
                    "Phase-1 registration requires the exact canonical plan/blueprint"
                )
            expected_prefix = expected_plan.stages
        elif plan.phase.value == "phase2":
            blueprint = Phase2Blueprint.from_manifest(blueprint_manifest)
            expected_prefix = (blueprint.initial_stage,)
            if phase1_decision_manifest is None:
                raise CampaignError(
                    "Phase 2 registration requires an authenticated Phase-1 go/no-go decision"
                )
            phase1_decision = self.phase1_decision_from_manifest(
                phase1_decision_manifest
            )
            authorization_mode = phase1_decision["authorization_mode"]
            phase2_smoke = (
                blueprint.initial_stage.cells[0].decision_map.get("runtime.smoke")
                is True
            )
            if not phase2_smoke:
                if (
                    authorization_mode != "scientific_go"
                    or phase1_decision.get("authorizes_phase2_scientific") is not True
                ):
                    raise CampaignError(
                        "a non-go or protocol-only Phase-1 decision cannot authorize scientific Phase 2"
                    )
            elif phase1_decision.get("authorizes_phase2_smoke") is not True:
                raise CampaignError(
                    "the Phase-1 decision does not authorize smoke Phase 2"
                )
            try:
                expected_blueprint = build_phase2_blueprint(
                    blueprint.seeds,
                    smoke=phase2_smoke,
                    phase1_decision=phase1_decision,
                )
                expected_plan = build_phase2_plan(
                    blueprint.seeds,
                    smoke=phase2_smoke,
                    phase1_decision=phase1_decision,
                )
            except StudyError as exc:
                raise CampaignError(f"cannot derive Phase-2 transfer: {exc}") from exc
            if blueprint != expected_blueprint or plan != expected_plan:
                raise CampaignError(
                    "Phase-2 plan/blueprint does not exactly inherit its Phase-1 transfer"
                )
        else:
            blueprint = Phase3Blueprint.from_manifest(blueprint_manifest)
            expected_prefix = plan.stages
            if panel_decision_manifest is None:
                raise CampaignError(
                    "Phase 3 registration requires a campaign-evidence-bound panel decision"
                )
            panel = self.panel_decision_from_manifest(panel_decision_manifest)
            source_manifest = panel_decision_manifest["phase2_campaign_manifest"]
            if source_manifest.get("smoke") is True and not blueprint.smoke:
                raise CampaignError(
                    "a smoke Phase-2 panel may only register a smoke Phase-3 plan"
                )
            expected_blueprint = build_phase3_blueprint(
                blueprint.seeds,
                smoke=blueprint.smoke,
                panel_decision=panel,
            )
            expected_plan = build_phase3_plan(
                blueprint.seeds,
                smoke=blueprint.smoke,
                panel_decision=panel,
            )
            if blueprint != expected_blueprint or plan != expected_plan:
                raise CampaignError(
                    "Phase 3 plan/blueprint differs from its verified frozen panel decision"
                )
        canonical_blueprint_manifest = blueprint.to_manifest()
        if canonical_json(blueprint_manifest) != canonical_json(
            canonical_blueprint_manifest
        ):
            raise CampaignError("frozen blueprint manifest is noncanonical")
        if plan.stages != expected_prefix:
            raise CampaignError(
                "registered plan is not the exact initial prefix of its blueprint"
            )
        if self.blueprint_path.exists() and canonical_json(
            _read_json(self.blueprint_path)
        ) != canonical_json(canonical_blueprint_manifest):
            raise CampaignError(
                "campaign already contains a different frozen blueprint"
            )
        current_identity = implementation_identity()
        current_digest = _validate_implementation_identity(
            current_identity,
            scientific=False,
        )
        if self.implementation_identity_path.exists():
            pin = _read_json(self.implementation_identity_path)
            if (
                set(pin)
                != {
                    "schema",
                    "implementation_identity",
                    "implementation_identity_sha256",
                }
                or pin.get("schema") != CAMPAIGN_IMPLEMENTATION_SCHEMA
            ):
                raise CampaignError("campaign implementation pin is noncanonical")
            pinned_identity = pin.get("implementation_identity")
            if not isinstance(pinned_identity, Mapping):
                raise CampaignError("campaign implementation pin lacks its identity")
            pinned_digest = _validate_implementation_identity(
                pinned_identity,
                scientific=False,
            )
            if (
                pin.get("implementation_identity_sha256") != pinned_digest
                or pinned_digest != current_digest
            ):
                raise CampaignError(
                    "registered campaign implementation differs from current execution"
                )
        else:
            _write_immutable_json(
                self.implementation_identity_path,
                {
                    "schema": CAMPAIGN_IMPLEMENTATION_SCHEMA,
                    "implementation_identity": current_identity,
                    "implementation_identity_sha256": current_digest,
                },
            )
        if self._events_cached():
            existing = self.plan
            if existing.plan_id != plan.plan_id:
                raise CampaignError(
                    f"campaign already contains plan {existing.plan_id}, not {plan.plan_id}"
                )
            try:
                projected_payload = _read_json(self.plan_path)
                projected = StudyPlan.from_manifest(projected_payload)
            except (
                CampaignError,
                FileNotFoundError,
                KeyError,
                OSError,
                TypeError,
                ValueError,
                StudyError,
            ):
                projected = None
                projected_payload = None
            if (
                projected != existing
                or canonical_json(projected_payload)
                != canonical_json(existing.to_manifest())
            ):
                _atomic_json(self.plan_path, existing.to_manifest())
        elif self.plan_path.exists():
            existing = StudyPlan.from_manifest(_read_json(self.plan_path))
            if existing.plan_id != plan.plan_id:
                raise CampaignError(
                    f"campaign already contains plan {existing.plan_id}, not {plan.plan_id}"
                )
        else:
            _atomic_json(self.plan_path, plan.to_manifest())
        _write_immutable_json(
            self.plans_dir / f"{_slug(plan.plan_id)}.json", plan.to_manifest()
        )
        _write_immutable_json(self.blueprint_path, canonical_blueprint_manifest)
        if phase1_decision_manifest is not None:
            _write_immutable_json(self.phase1_decision_path, phase1_decision_manifest)
        if panel_decision_manifest is not None:
            _write_immutable_json(self.panel_decision_path, panel_decision_manifest)
        self._register_cells(plan.cells, plan_id=plan.plan_id)

    @staticmethod
    def _phase1_claim_evidence(
        qualification: Mapping[str, Any], *, cell: CellSpec, smoke: bool
    ) -> dict[str, Any]:
        """Extract the exact Phase-1 conjunction used by the go/no-go gate."""

        outcome = qualification.get("scientific_outcome")
        metrics = qualification.get("selection_metrics")
        if not isinstance(metrics, Mapping):
            raise CampaignError("Phase-1 qualification lacks selection metrics")
        try:
            native, deployed, validation = _validated_phase1_identification(
                metrics, cell=cell
            )
        except ArtifactError as exc:
            raise CampaignError(str(exc)) from exc
        if smoke:
            protocol_passed = bool(
                qualification.get("selection_eligibility_mode") == "smoke_protocol_only"
                and qualification.get("selection_eligible_for_protocol_test") is True
            )
            return {
                "scientific_outcome_passed": None,
                "native_passed": None,
                "deployed_passed": None,
                "conjunction_passed": protocol_passed,
            }
        checks = outcome.get("checks") if isinstance(outcome, Mapping) else None
        native_passed = native["passed"] is True
        deployed_passed = deployed["passed"] is True
        scientific_passed = bool(
            isinstance(outcome, Mapping) and outcome.get("passed") is True
        )
        conjunction_passed = bool(
            isinstance(checks, Mapping)
            and checks.get("phase1_identification") is True
            and validation.get("phase1_identification_conjunction") is True
            and native_passed
            and deployed_passed
        )
        return {
            "scientific_outcome_passed": scientific_passed,
            "native_passed": native_passed,
            "deployed_passed": deployed_passed,
            "conjunction_passed": conjunction_passed,
        }

    @classmethod
    def phase1_decision_from_manifest(
        cls, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Validate a canonical, content-addressed Phase-1 decision envelope.

        As with the Phase-2 panel envelope, this establishes complete internal
        consistency and content binding.  Origin authentication still requires
        the live-file checks performed by :meth:`freeze_phase1_decision` or a
        separately trusted signature/root.
        """

        expected_keys = {
            "schema",
            "decision_id",
            "source_phase1_plan_id",
            "source_phase1_blueprint_id",
            "authorization_mode",
            "decision",
            "authorizes_phase2_scientific",
            "authorizes_phase2_smoke",
            "phase1_campaign_manifest_sha256",
            "phase1_campaign_manifest",
            "phase1_transfer",
        }
        if (
            payload.get("schema") != PHASE1_DECISION_SCHEMA
            or set(payload) != expected_keys
        ):
            raise CampaignError("Phase-1 decision envelope is missing or noncanonical")
        body = dict(payload)
        decision_id = body.pop("decision_id", None)
        if decision_id != content_id(body, prefix="phase1-decision"):
            raise CampaignError("Phase-1 decision content ID mismatch")
        manifest = payload.get("phase1_campaign_manifest")
        if not isinstance(manifest, Mapping):
            raise CampaignError("Phase-1 decision lacks its campaign manifest")
        if payload.get("phase1_campaign_manifest_sha256") != _canonical_sha256(
            manifest
        ):
            raise CampaignError("Phase-1 campaign-manifest hash mismatch")
        transfer = payload.get("phase1_transfer")
        if not isinstance(transfer, Mapping):
            raise CampaignError("Phase-1 decision lacks its transfer contract")
        manifest_keys = {
            "schema",
            "source_phase1_plan_id",
            "source_phase1_blueprint_id",
            "plan_content_sha256",
            "blueprint_content_sha256",
            "plan_sha256",
            "blueprint_sha256",
            "journal_sha256",
            "journal_sha256_semantics",
            "smoke",
            "plan",
            "blueprint",
            "plan_history",
            "selection_chain",
            "cells",
            "confirmation",
        }
        if (
            manifest.get("schema") != PHASE1_CAMPAIGN_MANIFEST_SCHEMA
            or set(manifest) != manifest_keys
        ):
            raise CampaignError("Phase-1 campaign manifest is missing or noncanonical")
        try:
            plan = StudyPlan.from_manifest(manifest["plan"])
            blueprint = Phase1Blueprint.from_manifest(manifest["blueprint"])
        except (KeyError, TypeError, ValueError, StudyError) as exc:
            raise CampaignError(
                f"invalid Phase-1 plan/blueprint evidence: {exc}"
            ) from exc
        if plan.phase.value != "phase1":
            raise CampaignError("Phase-1 decision embeds a non-Phase-1 plan")
        if canonical_json(manifest["plan"]) != canonical_json(
            plan.to_manifest()
        ) or canonical_json(manifest["blueprint"]) != canonical_json(
            blueprint.to_manifest()
        ):
            raise CampaignError("Phase-1 decision embeds noncanonical plan evidence")
        if manifest.get("plan_sha256") != _run_cell_json_sha256(
            plan.to_manifest()
        ) or manifest.get("blueprint_sha256") != _run_cell_json_sha256(
            blueprint.to_manifest()
        ):
            raise CampaignError("Phase-1 plan/blueprint file hash is stale")
        if (
            not isinstance(manifest.get("journal_sha256"), str)
            or not str(manifest["journal_sha256"]).startswith("sha256:")
            or not _is_sha256_hex(
                str(manifest["journal_sha256"]).removeprefix("sha256:")
            )
            or manifest.get("journal_sha256_semantics")
            != "opaque_historical_commitment_requires_trusted_origin"
        ):
            raise CampaignError("Phase-1 journal commitment is noncanonical")
        if (
            manifest.get("source_phase1_plan_id") != plan.plan_id
            or payload.get("source_phase1_plan_id") != plan.plan_id
            or manifest.get("source_phase1_blueprint_id") != blueprint.blueprint_id
            or payload.get("source_phase1_blueprint_id") != blueprint.blueprint_id
            or manifest.get("plan_content_sha256")
            != _canonical_sha256(plan.to_manifest())
            or manifest.get("blueprint_content_sha256")
            != _canonical_sha256(blueprint.to_manifest())
        ):
            raise CampaignError("Phase-1 decision source binding mismatch")
        smoke_values = {cell.decision_map.get("runtime.smoke") for cell in plan.cells}
        if smoke_values not in ({False}, {True}) or manifest.get("smoke") != (
            smoke_values == {True}
        ):
            raise CampaignError("Phase-1 decision has inconsistent smoke provenance")
        smoke = smoke_values == {True}
        try:
            canonical_blueprint = build_phase1_blueprint(
                blueprint.seeds,
                smoke=smoke,
            )
            canonical_prefix = build_phase1_plan(
                blueprint.seeds,
                smoke=smoke,
            )
        except StudyError as exc:
            raise CampaignError(
                f"cannot derive canonical Phase-1 evidence contract: {exc}"
            ) from exc
        if blueprint != canonical_blueprint:
            raise CampaignError(
                "Phase-1 decision does not bind the exact canonical Phase-1 blueprint"
            )
        if plan.stages[: len(canonical_prefix.stages)] != canonical_prefix.stages:
            raise CampaignError(
                "Phase-1 decision does not begin with the exact canonical prefix"
            )
        try:
            expected_transfer = build_phase1_transfer(manifest)
        except StudyError as exc:
            raise CampaignError(f"invalid Phase-1 transfer contract: {exc}") from exc
        if canonical_json(transfer) != canonical_json(expected_transfer):
            raise CampaignError("Phase-1 transfer contract differs from bound evidence")
        expected_stage_names = (
            *(stage.name for stage in blueprint.initial_stages),
            *(round_spec.name for round_spec in blueprint.rounds),
        )
        if tuple(stage.name for stage in plan.stages) != expected_stage_names:
            raise CampaignError("Phase-1 decision plan is not the complete blueprint")

        cells = manifest.get("cells")
        if not isinstance(cells, list) or len(cells) != len(plan.cells):
            raise CampaignError("Phase-1 decision has incomplete cell evidence")
        cells_by_id: dict[str, Mapping[str, Any]] = {}
        implementation_sha256s: set[str] = set()
        plan_cells_by_id = {cell.cell_id: cell for cell in plan.cells}
        phase1_cell_evidence_keys = {
            "cell_id",
            "candidate_id",
            "stage",
            "seed",
            "recipe_name",
            "recipe_id",
            "cell",
            "state",
            "qualification_sha256",
            "qualification",
        }
        for evidence in cells:
            if (
                not isinstance(evidence, Mapping)
                or set(evidence) != phase1_cell_evidence_keys
            ):
                raise CampaignError("Phase-1 cell evidence must be an object")
            cell_id = str(evidence.get("cell_id", ""))
            if cell_id in cells_by_id or cell_id not in plan_cells_by_id:
                raise CampaignError("Phase-1 cell evidence is repeated or unknown")
            cell = plan_cells_by_id[cell_id]
            if evidence.get("cell") != cell.to_manifest():
                raise CampaignError("Phase-1 embedded cell differs from its plan")
            if evidence.get("state") not in {
                RunState.QUALIFIED.value,
                RunState.PROMOTED.value,
            }:
                raise CampaignError("Phase-1 decision includes an unqualified cell")
            for field_name, observed in (
                ("candidate_id", cell.candidate_id),
                ("stage", cell.stage),
                ("seed", cell.seed),
                ("recipe_name", cell.recipe_name),
                ("recipe_id", cell.recipe_id),
            ):
                if evidence.get(field_name) != observed:
                    raise CampaignError("Phase-1 cell evidence identity mismatch")
            qualification_sha256 = evidence.get("qualification_sha256")
            qualification = evidence.get("qualification")
            if (
                not isinstance(qualification_sha256, str)
                or not qualification_sha256.startswith("sha256:")
                or not isinstance(qualification, Mapping)
                or qualification.get("schema") != QUALIFICATION_SCHEMA
                or qualification.get("cell_id") != cell_id
                or _run_cell_json_sha256(qualification) != qualification_sha256
            ):
                raise CampaignError("Phase-1 cell lacks bound qualification evidence")
            try:
                implementation_sha256s.add(
                    _validate_qualification_payload(qualification, cell=cell)
                )
            except ArtifactError as exc:
                raise CampaignError(
                    f"Phase-1 qualification semantic replay failed for {cell_id}: {exc}"
                ) from exc
            cells_by_id[cell_id] = evidence
        if len(implementation_sha256s) != 1:
            raise CampaignError(
                "Phase-1 campaign mixes qualification implementation identities"
            )

        expected = canonical_prefix
        replayed_plans = [expected]
        chain = manifest.get("selection_chain")
        if not isinstance(chain, list) or len(chain) != len(blueprint.rounds):
            raise CampaignError("Phase-1 decision has an incomplete selection chain")
        for index, chain_item in enumerate(chain):
            if not isinstance(chain_item, Mapping):
                raise CampaignError("Phase-1 selection-chain item must be an object")
            if set(chain_item) != {
                "source_plan_id",
                "source_stage",
                "target_plan_id",
                "target_stage",
                "policy_id",
                "selection_id",
                "selection_universe_sha256",
                "selection_artifact_sha256",
                "selection_artifact_sha256_semantics",
                "selection",
            }:
                raise CampaignError("Phase-1 selection-chain item is noncanonical")
            selection_artifact_sha256 = chain_item.get("selection_artifact_sha256")
            if (
                not isinstance(selection_artifact_sha256, str)
                or not selection_artifact_sha256.startswith("sha256:")
                or not _is_sha256_hex(selection_artifact_sha256.removeprefix("sha256:"))
                or chain_item.get("selection_artifact_sha256_semantics")
                != "opaque_historical_commitment_requires_trusted_origin"
            ):
                raise CampaignError(
                    "Phase-1 selection-artifact commitment is noncanonical"
                )
            source_stage = expected.stages[-1]
            if source_stage.selection_policy is None:
                raise CampaignError(
                    "Phase-1 selection chain names a nonselectable stage"
                )
            try:
                selection = FrozenSelection.from_dict(chain_item["selection"])
            except (KeyError, TypeError, ValueError, StudyError) as exc:
                raise CampaignError(f"invalid Phase-1 frozen selection: {exc}") from exc
            policy = source_stage.selection_policy
            normalized_source_evidence = {
                cell.cell_id: {
                    "state": cells_by_id[cell.cell_id]["state"],
                    "qualification": cells_by_id[cell.cell_id]["qualification"],
                    "qualification_sha256": cells_by_id[cell.cell_id][
                        "qualification_sha256"
                    ],
                }
                for cell in source_stage.cells
            }

            def phase1_no_sharing_guard(
                _cell: CellSpec,
                _selection_metrics: Mapping[str, Any],
                _policy: SelectionPolicy,
            ) -> Mapping[str, Any]:
                raise CampaignError(
                    "Phase-1 selection unexpectedly requested a sharing guard"
                )

            (
                replayed_candidates,
                replayed_excluded,
                _,
                replayed_smoke,
            ) = cls._selection_universe_from_evidence(
                source_stage.name,
                source_stage.cells,
                policy,
                normalized_source_evidence,
                sharing_guard_for_cell=phase1_no_sharing_guard,
            )
            universe_payload = {
                "plan_id": expected.plan_id,
                "source_stage": source_stage.name,
                "policy_id": policy.policy_id,
                "ranked_candidates": replayed_candidates,
                "excluded_candidates": replayed_excluded,
            }
            replayed_universe_sha256 = _canonical_sha256(universe_payload)
            replayed_selected = _policy_retained_candidates(
                replayed_candidates,
                policy,
                smoke_protocol_only=replayed_smoke,
            )
            source_cells_by_id = {cell.cell_id: cell for cell in source_stage.cells}
            try:
                selected_cells = tuple(
                    source_cells_by_id[cell_id] for cell_id in selection.cell_ids
                )
            except KeyError as exc:
                raise CampaignError(
                    "Phase-1 selection names a cell outside its source"
                ) from exc
            if any(
                cell.decision_map.get("qualification.promotable") is not True
                for cell in selected_cells
            ):
                raise CampaignError(
                    "Phase-1 selection includes a nonpromotable candidate"
                )
            try:
                replayed_selections = tuple(
                    FrozenSelection.from_cells(
                        policy,
                        tuple(
                            source_cells_by_id[str(observation["cell_id"])]
                            for observation in candidate["observations"]
                        ),
                        tuple(
                            float(observation["metric"])
                            for observation in candidate["observations"]
                        ),
                        tuple(
                            str(observation["qualification_sha256"])
                            for observation in candidate["observations"]
                        ),
                        replayed_universe_sha256,
                    )
                    for candidate in replayed_selected
                )
            except (KeyError, TypeError, ValueError, StudyError) as exc:
                raise CampaignError(
                    f"cannot reconstruct Phase-1 selected universe: {exc}"
                ) from exc
            if selection not in replayed_selections:
                raise CampaignError(
                    "Phase-1 selection is not among the exact policy-retained winners"
                )
            if (
                chain_item.get("source_plan_id") != expected.plan_id
                or chain_item.get("source_stage") != source_stage.name
                or chain_item.get("policy_id")
                != source_stage.selection_policy.policy_id
                or selection.source_stage != source_stage.name
                or selection.policy_id != source_stage.selection_policy.policy_id
                or selection.seeds != blueprint.seeds
                or any(
                    cell.candidate_id != selection.candidate_id
                    for cell in selected_cells
                )
                or tuple(
                    cells_by_id[cell.cell_id]["qualification_sha256"]
                    for cell in selected_cells
                )
                != selection.qualification_sha256s
            ):
                raise CampaignError("Phase-1 selection-chain binding mismatch")
            for cell, expected_metric in zip(
                selected_cells, selection.metric_values, strict=True
            ):
                qualification = cells_by_id[cell.cell_id]["qualification"]
                if cell.decision_map.get("qualification.promotable") is not True:
                    raise CampaignError(
                        "Phase-1 selection includes a nonpromotable candidate"
                    )
                if smoke:
                    if (
                        qualification.get("selection_eligibility_mode")
                        != "smoke_protocol_only"
                        or qualification.get("selection_eligible_for_protocol_test")
                        is not True
                        or expected_metric != 0.0
                    ):
                        raise CampaignError(
                            "Phase-1 smoke selection lacks protocol eligibility"
                        )
                else:
                    if (
                        qualification.get("promotion_eligible") is not True
                        or qualification.get("selection_eligibility_mode")
                        != "scientific_promotion"
                    ):
                        raise CampaignError(
                            "Phase-1 selection includes a nonpromotable candidate"
                        )
                    if (
                        cls._phase1_claim_evidence(
                            qualification, cell=cell, smoke=False
                        ).get("conjunction_passed")
                        is not True
                    ):
                        raise CampaignError(
                            "Phase-1 selection includes a failed scientific candidate"
                        )
                    actual_metric = cls._policy_metric(
                        qualification["selection_metrics"],
                        source_stage.selection_policy,
                    )
                    if actual_metric != expected_metric:
                        raise CampaignError(
                            "Phase-1 selected metric differs from qualification"
                        )
            extended = materialize_child_plan(expected, blueprint, selection)
            if (
                chain_item.get("target_plan_id") != extended.plan_id
                or chain_item.get("target_stage") != extended.stages[-1].name
                or chain_item.get("selection_id") != selection.selection_id
                or chain_item.get("selection_universe_sha256")
                != selection.selection_universe_sha256
                or plan.stages[: len(extended.stages)] != extended.stages
            ):
                raise CampaignError(
                    "Phase-1 selection chain does not replay its blueprint"
                )
            expected = extended
            replayed_plans.append(expected)
        if expected != plan:
            raise CampaignError("Phase-1 decision does not replay to its final plan")
        plan_history = manifest.get("plan_history")
        if (
            not isinstance(plan_history, list)
            or len(plan_history) != len(replayed_plans)
            or any(
                not isinstance(item, Mapping)
                or set(item) != {"plan_id", "sha256"}
                or item.get("plan_id") != replayed.plan_id
                or item.get("sha256") != _run_cell_json_sha256(replayed.to_manifest())
                for item, replayed in zip(plan_history, replayed_plans, strict=True)
            )
        ):
            raise CampaignError("Phase-1 plan-history evidence is incomplete")

        final_stage = plan.stages[-1]
        evidence_by_variant: dict[str, list[tuple[CellSpec, Mapping[str, Any]]]] = {}
        for cell in final_stage.cells:
            variant = cell.decision_map.get("factor.robustness")
            if not isinstance(variant, str):
                raise CampaignError(
                    "Phase-1 confirmation cell lacks a robustness variant"
                )
            evidence_by_variant.setdefault(variant, []).append(
                (cell, cells_by_id[cell.cell_id])
            )
        results: list[dict[str, Any]] = []
        for variant, entries in sorted(evidence_by_variant.items()):
            ordered = sorted(entries, key=lambda item: item[0].seed)
            if tuple(cell.seed for cell, _ in ordered) != blueprint.seeds:
                raise CampaignError("Phase-1 confirmation variant is not seed-complete")
            roles = {
                cell.decision_map.get("qualification.phase1_confirmation_role")
                for cell, _ in ordered
            }
            if len(roles) != 1 or next(iter(roles)) not in {
                "required_baseline_pass",
                "required_negative_control_failure",
                "claim_scope_stress",
            }:
                raise CampaignError(
                    "Phase-1 confirmation variant lacks one frozen decision role"
                )
            role = next(iter(roles))
            per_seed = []
            for cell, evidence in ordered:
                claim = cls._phase1_claim_evidence(
                    evidence["qualification"], cell=cell, smoke=smoke
                )
                per_seed.append(
                    {
                        "seed": cell.seed,
                        "cell_id": cell.cell_id,
                        "qualification_sha256": evidence["qualification_sha256"],
                        **claim,
                    }
                )
            results.append(
                {
                    "variant": variant,
                    "candidate_id": ordered[0][0].candidate_id,
                    "required_baseline": role == "required_baseline_pass",
                    "negative_control": role == "required_negative_control_failure",
                    "negative_control_passed": (
                        None
                        if smoke or role != "required_negative_control_failure"
                        else all(
                            item["conjunction_passed"] is False for item in per_seed
                        )
                    ),
                    "passed_all_seeds": all(
                        item["conjunction_passed"] is True for item in per_seed
                    ),
                    "per_seed": per_seed,
                }
            )
        confirmation = manifest.get("confirmation")
        if (
            not isinstance(confirmation, Mapping)
            or set(confirmation)
            != {
                "results",
                "stress_failures",
                "scope_narrowing",
            }
            or confirmation.get("results") != results
        ):
            raise CampaignError(
                "Phase-1 confirmation summary differs from cell evidence"
            )
        result_by_variant = {item["variant"]: item for item in results}
        baseline_results = [item for item in results if item["required_baseline"]]
        negative_control_results = [
            item for item in results if item["negative_control"]
        ]
        if len(baseline_results) != 1 or len(negative_control_results) != 2:
            raise CampaignError(
                "Phase-1 confirmation lacks its frozen baseline/negative-control roles"
            )
        baseline_variant = str(baseline_results[0]["variant"])
        negative_control_variants = {
            str(item["variant"]) for item in negative_control_results
        }
        stress_failures = sorted(
            variant
            for variant, result in result_by_variant.items()
            if variant != baseline_variant
            and variant not in negative_control_variants
            and result["passed_all_seeds"] is not True
        )
        scope_narrowing = confirmation.get("scope_narrowing")
        if (
            confirmation.get("stress_failures") != stress_failures
            or not isinstance(scope_narrowing, Mapping)
            or set(scope_narrowing) != set(stress_failures)
            or any(
                not isinstance(value, str) or not value.strip()
                for value in scope_narrowing.values()
            )
        ):
            raise CampaignError(
                "every failed Phase-1 stress requires an explicit scope narrowing"
            )
        baseline_passed = (
            result_by_variant[baseline_variant]["passed_all_seeds"] is True
        )
        negative_controls_passed = smoke or all(
            result_by_variant[variant].get("negative_control_passed") is True
            for variant in negative_control_variants
        )
        if smoke:
            expected_mode = "smoke_protocol_only"
            expected_decision = "protocol_complete"
            expected_scientific = False
            expected_smoke = baseline_passed
        else:
            scientific_go = baseline_passed and negative_controls_passed
            expected_mode = "scientific_go" if scientific_go else "scientific_no_go"
            expected_decision = "go" if scientific_go else "no_go"
            expected_scientific = scientific_go
            expected_smoke = scientific_go
        if (
            payload.get("authorization_mode") != expected_mode
            or payload.get("decision") != expected_decision
            or payload.get("authorizes_phase2_scientific") is not expected_scientific
            or payload.get("authorizes_phase2_smoke") is not expected_smoke
        ):
            raise CampaignError(
                "Phase-1 authorization differs from confirmation evidence"
            )
        return dict(payload)

    @staticmethod
    def panel_decision_from_manifest(
        payload: Mapping[str, Any],
    ) -> FrozenPanelDecision:
        """Validate a self-contained campaign-evidence decision envelope.

        ``FrozenPanelDecision`` deliberately models the scientific decision,
        not an external attestation.  This parser proves internal canonical
        consistency of the embedded, content-addressed Phase-2 campaign and
        ranked-universe manifests.  It does *not* prove that a self-consistent
        envelope originated from a particular machine or historical run;
        :meth:`freeze_panel` supplies the stronger live-file and journal checks
        before it emits an envelope.  Cryptographic origin would require a
        signature or separately trusted root, neither of which this format
        claims.
        """

        if payload.get("producer_schema") != PANEL_DECISION_PRODUCER_SCHEMA:
            raise CampaignError(
                "panel decision lacks the campaign freeze evidence envelope"
            )
        try:
            decision = FrozenPanelDecision.from_dict(payload)
        except (KeyError, TypeError, ValueError, StudyError) as exc:
            raise CampaignError(f"invalid frozen panel decision: {exc}") from exc
        expected_keys = {
            *decision.to_dict().keys(),
            "producer_schema",
            "phase2_campaign_manifest",
            "selection_universe",
        }
        if set(payload) != expected_keys or any(
            payload.get(key) != value for key, value in decision.to_dict().items()
        ):
            raise CampaignError("panel decision envelope is noncanonical")
        campaign_manifest = payload.get("phase2_campaign_manifest")
        universe = payload.get("selection_universe")
        if not isinstance(campaign_manifest, Mapping) or not isinstance(
            universe, Mapping
        ):
            raise CampaignError(
                "panel decision lacks its embedded Phase-2 evidence manifests"
            )
        if campaign_manifest.get("schema") != PHASE2_CAMPAIGN_MANIFEST_SCHEMA:
            raise CampaignError("panel decision has the wrong campaign-manifest schema")
        if universe.get("schema") != SELECTION_UNIVERSE_SCHEMA:
            raise CampaignError(
                "panel decision has the wrong selection-universe schema"
            )
        if set(campaign_manifest) != set(PHASE2_CAMPAIGN_MANIFEST_KEYS):
            raise CampaignError("panel campaign manifest has a noncanonical field set")
        if set(universe) != set(PHASE2_SELECTION_UNIVERSE_KEYS):
            raise CampaignError("panel selection universe has a noncanonical field set")
        if not isinstance(campaign_manifest.get("smoke"), bool):
            raise CampaignError("campaign manifest must declare its smoke status")
        journal_sha256 = campaign_manifest.get("journal_sha256")
        if (
            not isinstance(journal_sha256, str)
            or not journal_sha256.startswith("sha256:")
            or not _is_sha256_hex(journal_sha256.removeprefix("sha256:"))
            or campaign_manifest.get("journal_sha256_semantics")
            != "opaque_historical_commitment_requires_trusted_origin"
        ):
            raise CampaignError("panel journal commitment is noncanonical")
        phase1_decision = campaign_manifest.get("phase1_decision")
        if not isinstance(phase1_decision, Mapping):
            raise CampaignError("panel decision lacks its Phase-1 authorization")
        verified_phase1 = Campaign.phase1_decision_from_manifest(phase1_decision)
        if universe.get("phase1_decision_id") != verified_phase1.get("decision_id"):
            raise CampaignError("panel universe has the wrong Phase-1 decision")
        expected_transfer_id = verified_phase1["phase1_transfer"]["transfer_id"]
        if (
            universe.get("phase1_transfer_id") != expected_transfer_id
            or campaign_manifest.get("phase1_transfer_id") != expected_transfer_id
        ):
            raise CampaignError("panel evidence has the wrong Phase-1 transfer")
        if campaign_manifest.get("smoke") is True:
            if verified_phase1.get("authorizes_phase2_smoke") is not True:
                raise CampaignError("Phase-1 decision did not authorize smoke Phase 2")
        elif (
            verified_phase1.get("authorization_mode") != "scientific_go"
            or verified_phase1.get("authorizes_phase2_scientific") is not True
        ):
            raise CampaignError("scientific panel lacks a Phase-1 go decision")
        phase1_decision_sha256 = campaign_manifest.get("phase1_decision_sha256")
        if (
            not isinstance(phase1_decision_sha256, str)
            or not phase1_decision_sha256.startswith("sha256:")
            or phase1_decision_sha256 != _run_cell_json_sha256(verified_phase1)
        ):
            raise CampaignError("panel lacks the bound Phase-1 decision-file hash")
        if _canonical_sha256(campaign_manifest) != (
            decision.phase2_campaign_manifest_sha256
        ):
            raise CampaignError("Phase-2 campaign manifest hash mismatch")
        if _canonical_sha256(universe) != decision.selection_universe_sha256:
            raise CampaignError("Phase-2 selection-universe hash mismatch")
        for manifest in (campaign_manifest, universe):
            if (
                manifest.get("source_phase2_plan_id") != decision.source_phase2_plan_id
                or manifest.get("source_phase2_blueprint_id")
                != decision.source_phase2_blueprint_id
            ):
                raise CampaignError("panel evidence source binding mismatch")
        if campaign_manifest.get("panel_entries") != [
            entry.to_dict() for entry in decision.entries
        ]:
            raise CampaignError(
                "panel entries differ from the campaign evidence manifest"
            )
        if universe.get("selection_chain") != campaign_manifest.get("selection_chain"):
            raise CampaignError("campaign and universe selection chains differ")
        for field_name in (
            "main_selection_chain",
            "family_selection_chains",
            "family_nominations",
            "confirmation_noninferiority",
            "duplicate_substitutions",
        ):
            if universe.get(field_name) != campaign_manifest.get(field_name):
                raise CampaignError(
                    f"campaign and universe {field_name} evidence differs"
                )
        main_chain = universe.get("main_selection_chain")
        family_chains = universe.get("family_selection_chains")
        family_nominations = universe.get("family_nominations")
        if (
            not isinstance(main_chain, list)
            or not isinstance(family_chains, Mapping)
            or not isinstance(family_nominations, list)
            or not all(isinstance(items, list) for items in family_chains.values())
        ):
            raise CampaignError(
                "panel decision has malformed family selection evidence"
            )
        partitioned_chain = [
            *main_chain,
            *(
                item
                for family_name in sorted(family_chains)
                for item in family_chains[family_name]
            ),
        ]
        flat_chain = universe.get("selection_chain")
        partitioned_rows = [canonical_json(item) for item in partitioned_chain]
        flat_rows = (
            []
            if not isinstance(flat_chain, list)
            else [canonical_json(item) for item in flat_chain]
        )
        if (
            not isinstance(flat_chain, list)
            or len(partitioned_rows) != len(flat_rows)
            or len(set(partitioned_rows)) != len(partitioned_rows)
            or set(partitioned_rows) != set(flat_rows)
        ):
            raise CampaignError(
                "main/family chain partition differs from the complete selection chain"
            )
        if universe.get("panel_source_candidate_ids") != {
            entry.panel_slot: entry.source_candidate_id for entry in decision.entries
        }:
            raise CampaignError(
                "selection universe does not bind every panel candidate"
            )
        cells = campaign_manifest.get("cells")
        if not isinstance(cells, list) or len(
            {item.get("cell_id") for item in cells if isinstance(item, Mapping)}
        ) != len(cells):
            raise CampaignError(
                "campaign manifest cell evidence is missing or repeated"
            )
        cells_by_id = {
            str(item["cell_id"]): item
            for item in cells
            if isinstance(item, Mapping) and "cell_id" in item
        }
        if len(cells_by_id) != len(cells):
            raise CampaignError("campaign manifest has malformed cell evidence")
        embedded_cells_by_id: dict[str, CellSpec] = {}
        implementation_sha256s: set[str] = set()
        phase2_cell_evidence_keys = {
            "cell_id",
            "candidate_id",
            "stage",
            "seed",
            "recipe_name",
            "recipe_id",
            "cell",
            "state",
            "qualification",
            "artifacts",
        }
        for cell_id, evidence in cells_by_id.items():
            if set(evidence) != phase2_cell_evidence_keys:
                raise CampaignError(
                    "Phase-2 cell evidence uses a noncanonical field set"
                )
            try:
                embedded_cell = CellSpec.from_manifest(evidence["cell"])
            except (KeyError, TypeError, ValueError, StudyError) as exc:
                raise CampaignError(f"invalid embedded Phase-2 cell: {exc}") from exc
            if embedded_cell.cell_id != cell_id:
                raise CampaignError("embedded Phase-2 cell identity mismatch")
            if evidence.get("cell") != embedded_cell.to_manifest():
                raise CampaignError("embedded Phase-2 cell manifest is noncanonical")
            for field_name, observed in (
                ("candidate_id", embedded_cell.candidate_id),
                ("stage", embedded_cell.stage),
                ("seed", embedded_cell.seed),
                ("recipe_name", embedded_cell.recipe_name),
                ("recipe_id", embedded_cell.recipe_id),
            ):
                if evidence.get(field_name) != observed:
                    raise CampaignError("Phase-2 cell evidence identity mismatch")
            if embedded_cell.decision_map.get(
                "runtime.smoke"
            ) is not campaign_manifest.get("smoke"):
                raise CampaignError("embedded Phase-2 cell smoke provenance mismatch")
            embedded_cells_by_id[cell_id] = embedded_cell
            qualification = evidence.get("qualification")
            artifacts = evidence.get("artifacts")
            try:
                artifact_refs = (
                    [ArtifactRef.from_dict(item) for item in artifacts]
                    if isinstance(artifacts, list)
                    else []
                )
            except (ArtifactError, KeyError, TypeError, ValueError) as exc:
                raise CampaignError(
                    f"campaign cell has malformed artifact evidence: {cell_id}: {exc}"
                ) from exc
            artifact_hashes = {ref.kind: ref.sha256 for ref in artifact_refs}
            if (
                evidence.get("state")
                not in {RunState.QUALIFIED.value, RunState.PROMOTED.value}
                or not isinstance(qualification, Mapping)
                or qualification.get("schema") != QUALIFICATION_SCHEMA
                or qualification.get("cell_id") != cell_id
                or len(artifact_hashes) != len(artifact_refs)
                or "qualification" not in artifact_hashes
                or _run_cell_json_sha256(qualification)
                != "sha256:" + artifact_hashes["qualification"]
            ):
                raise CampaignError("campaign cell lacks its qualification payload")
            try:
                implementation_sha256s.add(
                    _validate_qualification_payload(
                        qualification,
                        cell=embedded_cell,
                        expected_artifact_hashes=artifact_hashes,
                    )
                )
            except (ArtifactError, KeyError) as exc:
                raise CampaignError(
                    f"Phase-2 qualification semantic replay failed for {cell_id}: {exc}"
                ) from exc
        if len(implementation_sha256s) != 1:
            raise CampaignError(
                "Phase-2 campaign mixes qualification implementation identities"
            )
        phase2_seeds = tuple(
            sorted({cell.seed for cell in embedded_cells_by_id.values()})
        )
        try:
            replay_blueprint = build_phase2_blueprint(
                phase2_seeds,
                smoke=campaign_manifest.get("smoke") is True,
                phase1_decision=verified_phase1,
            )
            replay_plan = build_phase2_plan(
                phase2_seeds,
                smoke=campaign_manifest.get("smoke") is True,
                phase1_decision=verified_phase1,
            )
        except StudyError as exc:
            raise CampaignError(
                f"cannot reconstruct the panel's Phase-2 source: {exc}"
            ) from exc
        if replay_blueprint.blueprint_id != decision.source_phase2_blueprint_id:
            raise CampaignError(
                "panel source blueprint does not replay its Phase-1 transfer"
            )
        _validate_panel_entry_seed_coverage(decision.entries, replay_blueprint.seeds)
        for entry in decision.entries:
            for index, cell in enumerate(entry.source_cells):
                evidence = cells_by_id.get(cell.cell_id)
                if evidence is None or evidence.get("state") not in {
                    RunState.QUALIFIED.value,
                    RunState.PROMOTED.value,
                }:
                    raise CampaignError(
                        "panel source cell lacks qualified campaign evidence"
                    )
                for field_name, observed in (
                    ("candidate_id", cell.candidate_id),
                    ("stage", cell.stage),
                    ("seed", cell.seed),
                    ("recipe_name", cell.recipe_name),
                    ("recipe_id", cell.recipe_id),
                ):
                    if evidence.get(field_name) != observed:
                        raise CampaignError(
                            "panel source cell differs from campaign evidence"
                        )
                artifacts = evidence.get("artifacts")
                if not isinstance(artifacts, list):
                    raise CampaignError("panel source cell has no artifact evidence")
                artifact_hashes = {
                    item.get("kind"): "sha256:" + str(item.get("sha256"))
                    for item in artifacts
                    if isinstance(item, Mapping)
                }
                if (
                    artifact_hashes.get("qualification")
                    != (entry.qualification_sha256s[index])
                ):
                    raise CampaignError(
                        "panel qualification hash lacks campaign evidence"
                    )
                if (
                    entry.role == "selected_finalist"
                    and artifact_hashes.get("evaluation")
                    != entry.confirmation_sha256s[index]
                ):
                    raise CampaignError(
                        "panel confirmation hash lacks campaign evidence"
                    )

        confirmation = campaign_manifest.get("confirmation_noninferiority")
        confirmation_keys = {
            "mode",
            "metric_path",
            "direction",
            "score_degradation_max",
            "score_degradation_threshold_basis",
            "score_degradation_sensitivity",
            "policy",
            "per_seed",
            "passed",
        }
        if (
            not isinstance(confirmation, Mapping)
            or set(confirmation) != confirmation_keys
        ):
            raise CampaignError("panel lacks confirmation noninferiority evidence")
        try:
            confirmation_policy = SelectionPolicy.from_dict(confirmation["policy"])
        except (KeyError, TypeError, ValueError, StudyError) as exc:
            raise CampaignError(f"invalid confirmation policy evidence: {exc}") from exc
        smoke = campaign_manifest.get("smoke") is True
        if (
            confirmation.get("metric_path") != confirmation_policy.metric_path
            or confirmation.get("direction") != confirmation_policy.direction
            or confirmation_policy.direction != "max"
            or confirmation.get("mode")
            != ("smoke_protocol_only" if smoke else "scientific_confirmation")
        ):
            raise CampaignError("confirmation noninferiority contract mismatch")
        finalist_entries = [
            entry for entry in decision.entries if entry.role == "selected_finalist"
        ]
        if len(finalist_entries) != 1:
            raise CampaignError("panel must contain one confirmation finalist")
        finalist = finalist_entries[0]
        score_degradation_values = {
            cell.decision_map.get("qualification.confirmation_score_degradation_max")
            for cell in finalist.source_cells
        }
        score_sensitivity_values = {
            cell.decision_map.get(
                "qualification.confirmation_score_degradation_sensitivity"
            )
            for cell in finalist.source_cells
        }
        threshold_basis_values = {
            cell.decision_map.get("qualification.confirmation_threshold_basis")
            for cell in finalist.source_cells
        }
        if (
            score_degradation_values != {PHASE2_CONFIRMATION_SCORE_DEGRADATION_MAX}
            or score_sensitivity_values
            != {PHASE2_CONFIRMATION_SCORE_DEGRADATION_SENSITIVITY}
            or threshold_basis_values != {PHASE2_CONFIRMATION_THRESHOLD_BASIS}
            or confirmation.get("score_degradation_max")
            != PHASE2_CONFIRMATION_SCORE_DEGRADATION_MAX
            or confirmation.get("score_degradation_threshold_basis")
            != PHASE2_CONFIRMATION_THRESHOLD_BASIS
        ):
            raise CampaignError(
                "confirmation score guard differs from its frozen cell contract"
            )
        score_degradation_max = PHASE2_CONFIRMATION_SCORE_DEGRADATION_MAX
        rows = confirmation.get("per_seed")
        if not isinstance(rows, list) or len(rows) != len(finalist.source_cells):
            raise CampaignError("confirmation evidence is not seed-complete")
        confirmation_row_keys = {
            "seed",
            "cell_id",
            "parent_cell_id",
            "qualification_sha256",
            "parent_qualification_sha256",
            "confirmation_score",
            "parent_score",
            "score_degradation",
            "sharing_guard",
            "qualification_passed",
            "score_noninferiority_passed",
            "sharing_guard_passed",
            "passed",
        }
        if any(
            not isinstance(row, Mapping) or set(row) != confirmation_row_keys
            for row in rows
        ):
            raise CampaignError("confirmation rows use a noncanonical field set")
        rows_by_seed = {
            row.get("seed"): row for row in rows if isinstance(row, Mapping)
        }
        if set(rows_by_seed) != {cell.seed for cell in finalist.source_cells}:
            raise CampaignError("confirmation rows repeat or omit a finalist seed")
        if len(rows_by_seed) != len(rows):
            raise CampaignError("confirmation evidence repeats or malforms a seed")

        def qualification_hash(evidence: Mapping[str, Any]) -> str | None:
            artifacts = evidence.get("artifacts")
            if not isinstance(artifacts, list):
                return None
            matches = [
                "sha256:" + str(item.get("sha256"))
                for item in artifacts
                if isinstance(item, Mapping) and item.get("kind") == "qualification"
            ]
            return matches[0] if len(matches) == 1 else None

        def embedded_sharing_guard(
            cell: CellSpec,
            selection_metrics: Mapping[str, Any],
            policy: SelectionPolicy,
        ) -> dict[str, Any]:
            """Rebuild one guard solely from embedded authenticated evidence."""

            current_cell = cell
            seen: set[str] = set()
            trace: list[dict[str, Any]] = []
            parent_cell_id: str | None = None
            parent_metrics: dict[str, float] | None = None
            while True:
                if current_cell.cell_id in seen:
                    raise CampaignError("embedded sharing lineage contains a cycle")
                seen.add(current_cell.cell_id)
                evidence = cells_by_id.get(current_cell.cell_id)
                if evidence is None:
                    raise CampaignError("embedded sharing lineage lacks cell evidence")
                qualification = evidence.get("qualification")
                metrics_payload = (
                    qualification.get("selection_metrics")
                    if isinstance(qualification, Mapping)
                    else None
                )
                if not isinstance(metrics_payload, Mapping):
                    raise CampaignError(
                        "embedded sharing lineage lacks authenticated metrics"
                    )
                if current_cell.cell_id == cell.cell_id and canonical_json(
                    metrics_payload
                ) != canonical_json(selection_metrics):
                    raise CampaignError(
                        "embedded sharing callback received different cell metrics"
                    )
                try:
                    current_metrics = Campaign._sharing_metrics(
                        metrics_payload,
                        context=f"embedded cell {current_cell.cell_id}",
                    )
                except ArtifactError as exc:
                    raise CampaignError(str(exc)) from exc
                trace.append(
                    {
                        "cell_id": current_cell.cell_id,
                        "seed": current_cell.seed,
                        "qualification_sha256": qualification_hash(evidence),
                        "metrics": current_metrics,
                    }
                )
                parent_ids = current_cell.decision_map.get(
                    "selection.parent_cell_ids", ()
                )
                if not isinstance(parent_ids, (tuple, list)):
                    raise CampaignError(
                        "embedded sharing lineage has malformed parent IDs"
                    )
                if not parent_ids:
                    immediate = (
                        current_metrics if parent_metrics is None else parent_metrics
                    )
                    return {
                        "cell_id": cell.cell_id,
                        "seed": cell.seed,
                        "parent_cell_id": parent_cell_id,
                        "root_cell_id": current_cell.cell_id,
                        "authenticated_lineage": trace,
                        **Campaign._sharing_guard_payload(
                            trace[0]["metrics"],
                            immediate,
                            current_metrics,
                            policy,
                        ),
                    }
                matching = [
                    embedded_cells_by_id.get(str(parent_id)) for parent_id in parent_ids
                ]
                matching = [
                    parent
                    for parent in matching
                    if parent is not None and parent.seed == cell.seed
                ]
                if len(matching) != 1:
                    raise CampaignError(
                        "embedded sharing lineage lacks one same-seed parent"
                    )
                current_cell = matching[0]
                if parent_cell_id is None:
                    parent_cell_id = current_cell.cell_id
                    parent_evidence = cells_by_id.get(current_cell.cell_id)
                    qualification = (
                        parent_evidence.get("qualification")
                        if isinstance(parent_evidence, Mapping)
                        else None
                    )
                    metrics_payload = (
                        qualification.get("selection_metrics")
                        if isinstance(qualification, Mapping)
                        else None
                    )
                    if not isinstance(metrics_payload, Mapping):
                        raise CampaignError(
                            "embedded sharing parent lacks authenticated metrics"
                        )
                    try:
                        parent_metrics = Campaign._sharing_metrics(
                            metrics_payload,
                            context=f"embedded parent {current_cell.cell_id}",
                        )
                    except ArtifactError as exc:
                        raise CampaignError(str(exc)) from exc

        for cell, expected_hash in zip(
            finalist.source_cells,
            finalist.qualification_sha256s,
            strict=True,
        ):
            row = rows_by_seed.get(cell.seed)
            current_evidence = cells_by_id.get(cell.cell_id)
            if not isinstance(row, Mapping) or current_evidence is None:
                raise CampaignError("confirmation finalist seed lacks evidence")
            parent_cell_id = row.get("parent_cell_id")
            parent_evidence = cells_by_id.get(str(parent_cell_id))
            if parent_evidence is None:
                raise CampaignError("confirmation parent lacks campaign evidence")
            declared_parent_ids = cell.decision_map.get("selection.parent_cell_ids", ())
            if not isinstance(declared_parent_ids, (tuple, list)):
                raise CampaignError("confirmation cell has malformed parent binding")
            same_seed_declared_parents = [
                embedded_cells_by_id.get(str(parent_id))
                for parent_id in declared_parent_ids
            ]
            same_seed_declared_parents = [
                parent
                for parent in same_seed_declared_parents
                if parent is not None and parent.seed == cell.seed
            ]
            if (
                row.get("cell_id") != cell.cell_id
                or row.get("qualification_sha256") != expected_hash
                or qualification_hash(current_evidence) != expected_hash
                or row.get("parent_qualification_sha256")
                != qualification_hash(parent_evidence)
                or parent_evidence.get("seed") != cell.seed
                or len(same_seed_declared_parents) != 1
                or same_seed_declared_parents[0].cell_id != parent_cell_id
            ):
                raise CampaignError("confirmation seed/hash/parent binding mismatch")
            current_qualification = current_evidence["qualification"]
            parent_qualification = parent_evidence["qualification"]
            if smoke:
                protocol_passed = bool(
                    current_qualification.get("selection_eligibility_mode")
                    == "smoke_protocol_only"
                    and current_qualification.get(
                        "selection_eligible_for_protocol_test"
                    )
                    is True
                )
                if (
                    row.get("confirmation_score") is not None
                    or row.get("parent_score") is not None
                    or row.get("score_degradation") is not None
                    or row.get("sharing_guard") is not None
                    or row.get("qualification_passed") is not protocol_passed
                    or row.get("score_noninferiority_passed") is not None
                    or row.get("sharing_guard_passed") is not None
                    or row.get("passed") is not protocol_passed
                ):
                    raise CampaignError("smoke confirmation evidence is inconsistent")
                continue
            current_metrics = current_qualification.get("selection_metrics")
            parent_metrics = parent_qualification.get("selection_metrics")
            if not isinstance(current_metrics, Mapping) or not isinstance(
                parent_metrics, Mapping
            ):
                raise CampaignError("confirmation metrics are missing")
            current_score = Campaign._policy_metric(
                current_metrics, confirmation_policy
            )
            parent_score = Campaign._policy_metric(parent_metrics, confirmation_policy)
            degradation = parent_score - current_score
            qualification_passed = bool(
                current_qualification.get("scientific_outcome", {}).get("passed")
                is True
                and current_qualification.get("promotion_eligible") is True
                and current_qualification.get("selection_eligibility_mode")
                == "scientific_promotion"
            )
            score_passed = bool(
                degradation <= score_degradation_max
                or math.isclose(
                    degradation,
                    score_degradation_max,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            )
            if not confirmation_policy.require_sharing_guard:
                expected_passed = qualification_passed and score_passed
                if (
                    row.get("sharing_guard") is not None
                    or row.get("sharing_guard_passed") is not None
                    or row.get("confirmation_score") != current_score
                    or row.get("parent_score") != parent_score
                    or row.get("score_degradation") != degradation
                    or row.get("qualification_passed") is not qualification_passed
                    or row.get("score_noninferiority_passed") is not score_passed
                    or row.get("passed") is not expected_passed
                ):
                    raise CampaignError(
                        "confirmation score-only noninferiority evidence is forged"
                    )
                continue
            guard = row.get("sharing_guard")
            if not isinstance(guard, Mapping):
                raise CampaignError("confirmation lacks its sharing guard")
            expected_guard = embedded_sharing_guard(
                cell,
                current_metrics,
                confirmation_policy,
            )
            _validate_exact_confirmation_guard(guard, expected_guard)
            trace = guard.get("authenticated_lineage")
            checks = guard.get("checks")
            measurements = guard.get("measurements")
            thresholds = guard.get("thresholds")
            if (
                guard.get("cell_id") != cell.cell_id
                or guard.get("seed") != cell.seed
                or guard.get("parent_cell_id") != parent_cell_id
                or not isinstance(trace, list)
                or len(trace) < 2
                or not isinstance(checks, Mapping)
                or not isinstance(measurements, Mapping)
                or not isinstance(thresholds, Mapping)
                or thresholds.get("fvu_absolute_max") != PHASE2_SHARING_FVU_ABSOLUTE_MAX
                or thresholds.get("root_site_only_fvu_degradation_max")
                != PHASE2_SHARING_ROOT_FVU_DEGRADATION_MAX
                or thresholds.get("root_leave_one_out_fvu_degradation_max")
                != PHASE2_SHARING_ROOT_FVU_DEGRADATION_MAX
                or thresholds.get("coordinate_concordance_min")
                != PHASE2_SHARING_COORDINATE_CONCORDANCE_MIN
                or thresholds.get("intersection_energy_coverage_min")
                != PHASE2_SHARING_INTERSECTION_ENERGY_COVERAGE_MIN
                or thresholds.get("intersection_recall_min")
                != PHASE2_SHARING_INTERSECTION_RECALL_MIN
            ):
                raise CampaignError("confirmation sharing-guard contract mismatch")
            try:
                current_sharing = Campaign._sharing_metrics(
                    current_metrics, context="confirmation current"
                )
                parent_sharing = Campaign._sharing_metrics(
                    parent_metrics, context="confirmation parent"
                )
            except ArtifactError as exc:
                raise CampaignError(str(exc)) from exc
            if (
                trace[0].get("cell_id") != cell.cell_id
                or trace[0].get("qualification_sha256") != expected_hash
                or trace[0].get("metrics") != current_sharing
                or trace[1].get("cell_id") != parent_cell_id
                or trace[1].get("qualification_sha256")
                != qualification_hash(parent_evidence)
                or trace[1].get("metrics") != parent_sharing
                or guard.get("root_cell_id") != trace[-1].get("cell_id")
            ):
                raise CampaignError("confirmation sharing lineage is unauthenticated")
            authenticated_trace_metrics: list[dict[str, float]] = []
            for trace_item in trace:
                if not isinstance(trace_item, Mapping):
                    raise CampaignError("sharing lineage item must be an object")
                evidence = cells_by_id.get(str(trace_item.get("cell_id")))
                if (
                    evidence is None
                    or evidence.get("seed") != cell.seed
                    or trace_item.get("qualification_sha256")
                    != qualification_hash(evidence)
                ):
                    raise CampaignError("sharing root trace lacks campaign evidence")
                qualification = evidence.get("qualification")
                selection_metrics = (
                    qualification.get("selection_metrics")
                    if isinstance(qualification, Mapping)
                    else None
                )
                if not isinstance(selection_metrics, Mapping):
                    raise CampaignError("sharing trace lacks qualification metrics")
                try:
                    authenticated_metrics = Campaign._sharing_metrics(
                        selection_metrics,
                        context=f"sharing trace {trace_item.get('cell_id')}",
                    )
                except ArtifactError as exc:
                    raise CampaignError(str(exc)) from exc
                if trace_item.get("metrics") != authenticated_metrics:
                    raise CampaignError("sharing trace metrics lack campaign evidence")
                authenticated_trace_metrics.append(authenticated_metrics)
            Campaign._validate_recomputed_sharing_guard(
                guard,
                current_sharing,
                parent_sharing,
                authenticated_trace_metrics[-1],
                confirmation_policy,
            )
            required_measurements = (
                "site_only_fvu_degradation",
                "leave_one_out_fvu_degradation",
                "site_only_support_iou_drop",
                "leave_one_out_support_iou_drop",
                "all_view_fvu_advantage_descriptive",
                "site_only_coordinate_concordance",
                "leave_one_out_coordinate_concordance",
                "site_only_intersection_recall",
                "leave_one_out_intersection_recall",
                "site_only_intersection_energy_coverage",
                "leave_one_out_intersection_energy_coverage",
                "root_site_only_fvu_degradation",
                "root_leave_one_out_fvu_degradation",
                "site_only_fvu_absolute",
                "leave_one_out_fvu_absolute",
            )
            required_thresholds = (
                "site_only_fvu_degradation_max",
                "leave_one_out_fvu_degradation_max",
                "support_iou_drop_max",
                "coordinate_concordance_min",
                "intersection_recall_min",
                "intersection_energy_coverage_min",
                "root_site_only_fvu_degradation_max",
                "root_leave_one_out_fvu_degradation_max",
                "fvu_absolute_max",
            )
            if any(
                not isinstance(mapping.get(name), (int, float))
                or isinstance(mapping.get(name), bool)
                or not math_isfinite(mapping[name])
                for mapping, names in (
                    (measurements, required_measurements),
                    (thresholds, required_thresholds),
                )
                for name in names
            ):
                raise CampaignError("confirmation sharing metrics are malformed")
            expected_checks = {
                "site_only_fvu_degradation": measurements.get(
                    "site_only_fvu_degradation"
                )
                <= thresholds.get("site_only_fvu_degradation_max"),
                "leave_one_out_fvu_degradation": measurements.get(
                    "leave_one_out_fvu_degradation"
                )
                <= thresholds.get("leave_one_out_fvu_degradation_max"),
                "site_only_support_iou_drop": measurements.get(
                    "site_only_support_iou_drop"
                )
                <= thresholds.get("support_iou_drop_max"),
                "leave_one_out_support_iou_drop": measurements.get(
                    "leave_one_out_support_iou_drop"
                )
                <= thresholds.get("support_iou_drop_max"),
                "site_only_coordinate_concordance": measurements.get(
                    "site_only_coordinate_concordance"
                )
                >= thresholds.get("coordinate_concordance_min"),
                "leave_one_out_coordinate_concordance": measurements.get(
                    "leave_one_out_coordinate_concordance"
                )
                >= thresholds.get("coordinate_concordance_min"),
                "site_only_intersection_recall": measurements.get(
                    "site_only_intersection_recall"
                )
                >= thresholds.get("intersection_recall_min"),
                "leave_one_out_intersection_recall": measurements.get(
                    "leave_one_out_intersection_recall"
                )
                >= thresholds.get("intersection_recall_min"),
                "site_only_intersection_energy_coverage": measurements.get(
                    "site_only_intersection_energy_coverage"
                )
                >= thresholds.get("intersection_energy_coverage_min"),
                "leave_one_out_intersection_energy_coverage": measurements.get(
                    "leave_one_out_intersection_energy_coverage"
                )
                >= thresholds.get("intersection_energy_coverage_min"),
                "root_site_only_fvu_degradation": measurements.get(
                    "root_site_only_fvu_degradation"
                )
                <= thresholds.get("root_site_only_fvu_degradation_max"),
                "root_leave_one_out_fvu_degradation": measurements.get(
                    "root_leave_one_out_fvu_degradation"
                )
                <= thresholds.get("root_leave_one_out_fvu_degradation_max"),
                "site_only_fvu_absolute": measurements.get("site_only_fvu_absolute")
                <= thresholds.get("fvu_absolute_max"),
                "leave_one_out_fvu_absolute": measurements.get(
                    "leave_one_out_fvu_absolute"
                )
                <= thresholds.get("fvu_absolute_max"),
            }
            sharing_passed = all(expected_checks.values())
            if (
                row.get("confirmation_score") != current_score
                or row.get("parent_score") != parent_score
                or row.get("score_degradation") != degradation
                or row.get("qualification_passed") is not qualification_passed
                or row.get("score_noninferiority_passed") is not score_passed
                or checks != expected_checks
                or guard.get("passed") is not sharing_passed
                or row.get("sharing_guard_passed") is not sharing_passed
                or row.get("passed")
                is not (qualification_passed and score_passed and sharing_passed)
            ):
                raise CampaignError("confirmation noninferiority evidence is forged")
        expected_sensitivity = _confirmation_score_sensitivity_payload(
            rows,
            PHASE2_CONFIRMATION_SCORE_DEGRADATION_SENSITIVITY,
            smoke=smoke,
            require_sharing_guard=confirmation_policy.require_sharing_guard,
        )
        if confirmation.get("score_degradation_sensitivity") != expected_sensitivity:
            raise CampaignError("confirmation score sensitivity evidence is forged")
        if confirmation.get("passed") is not True or any(
            not isinstance(row, Mapping) or row.get("passed") is not True
            for row in rows
        ):
            raise CampaignError("panel confirmation gate did not pass every seed")

        substitutions = campaign_manifest.get("duplicate_substitutions")
        if not isinstance(substitutions, list) or len(
            {
                item.get("panel_slot")
                for item in substitutions
                if isinstance(item, Mapping)
            }
        ) != len(substitutions):
            raise CampaignError("duplicate-substitution evidence is malformed")
        decision_entries = {entry.panel_slot: entry for entry in decision.entries}
        all_ranked_universes = universe.get("ranked_stage_universes")
        if not isinstance(all_ranked_universes, list):
            raise CampaignError("panel lacks ranked universes for substitutions")

        def provisional_with_entries(
            replacement_entries: Mapping[str, FrozenPanelEntry],
        ) -> FrozenPanelDecision:
            return FrozenPanelDecision(
                source_phase2_plan_id=decision.source_phase2_plan_id,
                source_phase2_blueprint_id=decision.source_phase2_blueprint_id,
                phase2_campaign_manifest_sha256="sha256:" + "0" * 64,
                selection_universe_sha256="sha256:" + "0" * 64,
                entries=tuple(
                    replacement_entries.get(entry.panel_slot, entry)
                    for entry in decision.entries
                ),
            )

        validated_duplicate_substitution_chains: set[str] = set()
        for substitution in substitutions:
            if not isinstance(substitution, Mapping):
                raise CampaignError("duplicate substitution must be an object")
            slot = str(substitution.get("panel_slot", ""))
            entry = decision_entries.get(slot)
            chain = family_chains.get(slot)
            if (
                entry is None
                or entry.role == "selected_finalist"
                or not isinstance(chain, list)
                or not chain
            ):
                raise CampaignError("duplicate substitute names a non-comparator slot")
            final_chain_item = chain[-1]
            source_stage = final_chain_item.get("source_stage")
            ranked_matches = [
                item
                for item in all_ranked_universes
                if isinstance(item, Mapping)
                and item.get("source_stage") == source_stage
                and item.get("source_plan_id") == final_chain_item.get("source_plan_id")
            ]
            if len(ranked_matches) != 1:
                raise CampaignError("duplicate substitute lacks one ranked source")
            ranked = ranked_matches[0]
            try:
                policy = SelectionPolicy.from_dict(ranked["policy"])
            except (KeyError, TypeError, ValueError, StudyError) as exc:
                raise CampaignError(f"invalid substitute policy: {exc}") from exc
            candidates = ranked.get("ranked_candidates")
            rank = substitution.get("substitute_rank")
            if (
                substitution.get("policy") != "next_ranked_nonduplicate"
                or substitution.get("reason")
                != "projected_scientific_configuration_duplicate"
                or not isinstance(candidates, list)
                or not isinstance(rank, int)
                or isinstance(rank, bool)
                or rank < 2
                or rank > len(candidates)
            ):
                raise CampaignError("duplicate substitution contract mismatch")

            def frozen_candidate(candidate: Mapping[str, Any]) -> FrozenSelection:
                observations = candidate.get("observations")
                if not isinstance(observations, list):
                    raise CampaignError("substitute candidate lacks observations")
                try:
                    candidate_cells = [
                        CellSpec.from_manifest(
                            cells_by_id[str(item["cell_id"])]["cell"]
                        )
                        for item in observations
                    ]
                    return FrozenSelection.from_cells(
                        policy,
                        candidate_cells,
                        [float(item["metric"]) for item in observations],
                        [str(item["qualification_sha256"]) for item in observations],
                        str(ranked["selection_universe_sha256"]),
                    )
                except (KeyError, TypeError, ValueError, StudyError) as exc:
                    raise CampaignError(
                        f"invalid substitute candidate evidence: {exc}"
                    ) from exc

            if not isinstance(candidates[0], Mapping) or not isinstance(
                candidates[rank - 1], Mapping
            ):
                raise CampaignError("substitute ranked candidates must be objects")
            original_selection = frozen_candidate(candidates[0])
            substitute_selection = frozen_candidate(candidates[rank - 1])
            if (
                substitution.get("original_candidate_id")
                != original_selection.candidate_id
                or substitution.get("original_selection_id")
                != original_selection.selection_id
                or substitution.get("substitute_candidate_id")
                != substitute_selection.candidate_id
                or substitution.get("substitute_selection_id")
                != substitute_selection.selection_id
                or entry.source_candidate_id != substitute_selection.candidate_id
                or final_chain_item.get("branch")
                != "comparator_family_duplicate_substitute"
                or final_chain_item.get("selection_id")
                != substitute_selection.selection_id
            ):
                raise CampaignError("duplicate substitution selection binding mismatch")

            def entry_for_selection(
                selection: FrozenSelection,
            ) -> FrozenPanelEntry:
                source_cells = [
                    CellSpec.from_manifest(cells_by_id[cell_id]["cell"])
                    for cell_id in selection.cell_ids
                ]
                selection_ids = tuple(
                    selection.selection_id
                    if item == substitute_selection.selection_id
                    else item
                    for item in entry.selection_ids
                )
                return FrozenPanelEntry.from_cells(
                    panel_slot=entry.panel_slot,
                    role=entry.role,
                    source_cells=source_cells,
                    selection_ids=selection_ids,
                    qualification_sha256s=selection.qualification_sha256s,
                )

            original_entry = entry_for_selection(original_selection)
            original_decision = provisional_with_entries({slot: original_entry})
            original_fingerprints = Campaign._projected_scientific_configurations(
                original_decision, smoke=smoke
            )
            original_groups = Campaign._duplicate_projected_configurations(
                original_decision, smoke=smoke
            )
            collided_group = next(
                (item for item in original_groups if slot in item["panel_slots"]),
                None,
            )
            final_fingerprints = Campaign._projected_scientific_configurations(
                decision, smoke=smoke
            )
            if (
                collided_group is None
                or set(substitution.get("collided_panel_slots", ()))
                != set(collided_group["panel_slots"])
                or substitution.get("original_scientific_configuration_id")
                != original_fingerprints[slot]
                or substitution.get("substitute_scientific_configuration_id")
                != final_fingerprints[slot]
            ):
                raise CampaignError(
                    "duplicate substitution fingerprint evidence mismatch"
                )
            for earlier_candidate in candidates[1 : rank - 1]:
                if not isinstance(earlier_candidate, Mapping):
                    raise CampaignError("ranked substitute candidate must be an object")
                earlier_selection = frozen_candidate(earlier_candidate)
                earlier_entry = entry_for_selection(earlier_selection)
                earlier_decision = provisional_with_entries({slot: earlier_entry})
                if not any(
                    slot in item["panel_slots"]
                    for item in Campaign._duplicate_projected_configurations(
                        earlier_decision, smoke=smoke
                    )
                ):
                    raise CampaignError(
                        "declared duplicate substitute is not the next ranked nonduplicate"
                    )
            validated_duplicate_substitution_chains.add(
                canonical_json(final_chain_item)
            )
        family_blueprints = {
            family.name: family for family in replay_blueprint.comparator_families
        }
        nomination_selections_by_family: dict[str, tuple[FrozenSelection, ...]] = {}
        nominations_by_source_plan: dict[str, Mapping[str, Any]] = {}
        for nomination in family_nominations:
            if not isinstance(nomination, Mapping):
                raise CampaignError("family nomination evidence must be an object")
            nomination_payload = nomination.get("nomination_payload")
            if not isinstance(nomination_payload, Mapping):
                raise CampaignError("family nomination lacks its complete payload")
            if nomination_payload.get("schema") != FAMILY_NOMINATION_SCHEMA:
                raise CampaignError("family nomination has the wrong schema")
            if nomination_payload.get("plan_id") != nomination.get(
                "source_plan_id"
            ) or nomination_payload.get("source_rounds") != nomination.get(
                "source_rounds"
            ):
                raise CampaignError("family nomination source binding mismatch")
            body = dict(nomination_payload)
            nomination_id = body.pop("nomination_id", None)
            if nomination_id != content_id(body, prefix="family-nomination"):
                raise CampaignError("family nomination content ID mismatch")
            for field_name in (
                "family_name",
                "family_id",
                "nomination_id",
                "selection_universe_sha256",
                "ranked_candidates",
                "excluded_candidates",
                "source_threshold_sensitivity",
                "selected",
            ):
                if nomination.get(field_name) != nomination_payload.get(field_name):
                    raise CampaignError(
                        "family nomination summary differs from its complete payload"
                    )
            policy_payload = nomination_payload.get("policy")
            if not isinstance(policy_payload, Mapping):
                raise CampaignError("family nomination lacks its frozen policy")
            family_name = str(nomination.get("family_name", ""))
            family = family_blueprints.get(family_name)
            if family is None:
                raise CampaignError(
                    "family nomination names an unknown blueprint family"
                )
            try:
                nomination_policy = SelectionPolicy.from_dict(policy_payload)
            except (KeyError, TypeError, ValueError, StudyError) as exc:
                raise CampaignError(f"invalid family nomination policy: {exc}") from exc
            if (
                nomination_policy != family.revisit.nomination_policy
                or canonical_json(policy_payload)
                != canonical_json(nomination_policy.to_dict())
                or nomination_payload.get("blueprint_id")
                != replay_blueprint.blueprint_id
                or nomination_payload.get("family_id") != family.family_id
                or nomination_payload.get("revisit_id") != family.revisit.revisit_id
                or tuple(nomination_payload.get("source_rounds", ()))
                != family.revisit.source_rounds
                or nomination_payload.get("phase") != "phase2"
                or nomination_payload.get("smoke") is not smoke
                or nomination.get("source_stage") != family.revisit.name
            ):
                raise CampaignError(
                    "family nomination differs from its canonical blueprint contract"
                )
            replayed_nomination_ranked: list[dict[str, Any]] = []
            replayed_nomination_excluded: list[dict[str, Any]] = []
            replayed_source_sensitivity: dict[str, Any] = {}
            for source_round in family.revisit.source_rounds:
                source_cells = tuple(
                    cell
                    for cell in embedded_cells_by_id.values()
                    if cell.stage == source_round
                )
                normalized_source_evidence: dict[str, dict[str, Any]] = {}
                for cell in source_cells:
                    evidence = cells_by_id[cell.cell_id]
                    normalized: dict[str, Any] = {
                        "state": evidence.get("state"),
                    }
                    if evidence.get("state") in {
                        RunState.QUALIFIED.value,
                        RunState.PROMOTED.value,
                    }:
                        normalized.update(
                            {
                                "qualification": evidence.get("qualification"),
                                "qualification_sha256": qualification_hash(evidence),
                            }
                        )
                    normalized_source_evidence[cell.cell_id] = normalized
                (
                    source_ranked,
                    source_excluded,
                    source_sensitivity,
                    source_smoke,
                ) = Campaign._selection_universe_from_evidence(
                    source_round,
                    source_cells,
                    nomination_policy,
                    normalized_source_evidence,
                    sharing_guard_for_cell=embedded_sharing_guard,
                )
                if source_smoke is not smoke:
                    raise CampaignError(
                        "family nomination source has inconsistent smoke provenance"
                    )
                replayed_nomination_ranked.extend(
                    {**candidate, "source_stage": source_round}
                    for candidate in source_ranked
                )
                replayed_nomination_excluded.extend(
                    {**candidate, "source_stage": source_round}
                    for candidate in source_excluded
                )
                replayed_source_sensitivity[source_round] = source_sensitivity
            replayed_candidate_ids = [
                str(candidate["candidate_id"])
                for candidate in replayed_nomination_ranked
            ]
            if len(replayed_candidate_ids) != len(set(replayed_candidate_ids)):
                raise CampaignError(
                    "family nomination replay repeats a candidate identity"
                )
            for candidate in replayed_nomination_ranked:
                observations = candidate.get("observations")
                if not isinstance(observations, list) or not observations:
                    raise CampaignError(
                        "replayed family nomination candidate lacks observations"
                    )
                try:
                    candidate_cells = tuple(
                        embedded_cells_by_id[str(item["cell_id"])]
                        for item in observations
                    )
                    candidate["execution_signature"] = (
                        resolved_candidate_execution_signature(candidate_cells)
                    )
                except (KeyError, TypeError, StudyError) as exc:
                    raise CampaignError(
                        f"invalid replayed nomination execution signature: {exc}"
                    ) from exc
            (
                replayed_nomination_ranked,
                duplicate_aliases,
            ) = Campaign._deduplicate_family_nomination_candidates(
                replayed_nomination_ranked,
                family.revisit.source_rounds,
            )
            replayed_nomination_excluded.extend(duplicate_aliases)
            nomination_sign = 1.0 if nomination_policy.direction == "min" else -1.0
            replayed_nomination_ranked.sort(
                key=lambda item: (
                    nomination_sign * float(item["median"]),
                    nomination_sign * float(item["worst_seed"]),
                    str(item["candidate_id"]),
                )
            )
            if len(replayed_nomination_ranked) < family.revisit.top_k:
                raise CampaignError(
                    "family nomination replay has too few distinct configurations"
                )
            if (
                canonical_json(replayed_nomination_ranked)
                != canonical_json(nomination_payload.get("ranked_candidates"))
                or canonical_json(replayed_nomination_excluded)
                != canonical_json(nomination_payload.get("excluded_candidates"))
                or canonical_json(replayed_source_sensitivity)
                != canonical_json(
                    nomination_payload.get("source_threshold_sensitivity")
                )
            ):
                raise CampaignError(
                    "family nomination differs from authenticated cross-round replay"
                )
            nomination_universe = {
                "plan_id": nomination_payload.get("plan_id"),
                "source_stage": nomination.get("source_stage"),
                "policy_id": nomination_policy.policy_id,
                "family_name": nomination.get("family_name"),
                "family_id": nomination.get("family_id"),
                "source_rounds": nomination.get("source_rounds"),
                "ranked_candidates": nomination.get("ranked_candidates"),
                "excluded_candidates": nomination.get("excluded_candidates"),
                "source_threshold_sensitivity": nomination.get(
                    "source_threshold_sensitivity"
                ),
            }
            if _canonical_sha256(nomination_universe) != nomination.get(
                "selection_universe_sha256"
            ):
                raise CampaignError("family nomination universe hash mismatch")
            try:
                selected_nominations = tuple(
                    FrozenSelection.from_dict(item)
                    for item in nomination_payload.get("selected", ())
                )
            except (KeyError, TypeError, ValueError, StudyError) as exc:
                raise CampaignError(
                    f"invalid nominated family selection: {exc}"
                ) from exc
            if not selected_nominations or any(
                selection.selection_universe_sha256
                != nomination.get("selection_universe_sha256")
                for selection in selected_nominations
            ):
                raise CampaignError("family nomination selections lack union binding")
            ranked_candidates = nomination_payload.get("ranked_candidates")
            if (
                not isinstance(ranked_candidates, list)
                or len(ranked_candidates) < family.revisit.top_k
            ):
                raise CampaignError("family nomination ranked universe is incomplete")
            reconstructed_nominations: list[FrozenSelection] = []
            for candidate in ranked_candidates[: family.revisit.top_k]:
                if not isinstance(candidate, Mapping):
                    raise CampaignError("family nomination candidate must be an object")
                source_stage = str(candidate.get("source_stage", ""))
                if source_stage not in family.revisit.source_rounds:
                    raise CampaignError(
                        "family nomination candidate escaped its declared source rounds"
                    )
                observations = candidate.get("observations")
                if not isinstance(observations, list) or not observations:
                    raise CampaignError(
                        "family nomination candidate lacks observations"
                    )
                candidate_cells: list[CellSpec] = []
                metric_values: list[float] = []
                qualification_sha256s: list[str] = []
                for observation in observations:
                    if not isinstance(observation, Mapping):
                        raise CampaignError(
                            "family nomination observation must be an object"
                        )
                    cell = embedded_cells_by_id.get(str(observation.get("cell_id", "")))
                    evidence = cells_by_id.get(str(observation.get("cell_id", "")))
                    if (
                        cell is None
                        or evidence is None
                        or cell.stage != source_stage
                        or cell.candidate_id != candidate.get("candidate_id")
                        or observation.get("seed") != cell.seed
                        or observation.get("qualification_sha256")
                        != qualification_hash(evidence)
                    ):
                        raise CampaignError(
                            "family nomination observation lacks exact cell evidence"
                        )
                    candidate_cells.append(cell)
                    metric_values.append(float(observation["metric"]))
                    qualification_sha256s.append(
                        str(observation["qualification_sha256"])
                    )
                try:
                    execution_signature = resolved_candidate_execution_signature(
                        candidate_cells
                    )
                except StudyError as exc:
                    raise CampaignError(
                        f"invalid nominated execution signature: {exc}"
                    ) from exc
                if candidate.get("execution_signature") != execution_signature:
                    raise CampaignError(
                        "family nomination execution signature differs from its cells"
                    )
                aliases = candidate.get("execution_aliases")
                if not isinstance(aliases, list) or not aliases:
                    raise CampaignError(
                        "family nomination representative lacks execution aliases"
                    )
                alias_keys: set[tuple[str, str]] = set()
                for alias in aliases:
                    if not isinstance(alias, Mapping):
                        raise CampaignError(
                            "family nomination execution alias must be an object"
                        )
                    alias_key = (
                        str(alias.get("source_stage", "")),
                        str(alias.get("candidate_id", "")),
                    )
                    if alias_key in alias_keys:
                        raise CampaignError(
                            "family nomination repeats an execution alias"
                        )
                    alias_keys.add(alias_key)
                    alias_cells = tuple(
                        cell
                        for cell in embedded_cells_by_id.values()
                        if cell.stage == alias_key[0]
                        and cell.candidate_id == alias_key[1]
                    )
                    try:
                        alias_signature = resolved_candidate_execution_signature(
                            alias_cells
                        )
                    except StudyError as exc:
                        raise CampaignError(
                            f"invalid family nomination alias: {exc}"
                        ) from exc
                    if alias_signature != execution_signature:
                        raise CampaignError(
                            "family nomination alias changes the resolved configuration"
                        )
                if (
                    str(candidate.get("source_stage", "")),
                    str(candidate.get("candidate_id", "")),
                ) not in alias_keys:
                    raise CampaignError(
                        "family nomination aliases omit their representative"
                    )
                try:
                    reconstructed_nominations.append(
                        FrozenSelection.from_cells(
                            nomination_policy,
                            candidate_cells,
                            metric_values,
                            qualification_sha256s,
                            str(nomination["selection_universe_sha256"]),
                        )
                    )
                except (KeyError, TypeError, ValueError, StudyError) as exc:
                    raise CampaignError(
                        f"invalid family nomination candidate binding: {exc}"
                    ) from exc
            if tuple(reconstructed_nominations) != selected_nominations:
                raise CampaignError(
                    "family nomination selections differ from their ranked evidence"
                )
            if family_name in nomination_selections_by_family:
                raise CampaignError("panel decision repeats a family nomination")
            source_plan_id = str(nomination.get("source_plan_id", ""))
            if source_plan_id in nominations_by_source_plan:
                raise CampaignError("multiple family nominations share one source plan")
            nomination_selections_by_family[family_name] = selected_nominations
            nominations_by_source_plan[source_plan_id] = nomination
        if set(nomination_selections_by_family) != set(family_chains):
            raise CampaignError(
                "panel nominations do not cover every comparator family"
            )
        ranked_universes = universe.get("ranked_stage_universes")
        selection_chain = universe.get("selection_chain")
        if (
            not isinstance(ranked_universes, list)
            or not isinstance(selection_chain, list)
            or len(ranked_universes) != len(selection_chain)
        ):
            raise CampaignError("panel decision has an incomplete ranked universe")
        chain_bindings: list[
            tuple[Mapping[str, Any], FrozenSelection, SelectionPolicy]
        ] = []
        for ranked, chain_item in zip(ranked_universes, selection_chain, strict=True):
            if not isinstance(ranked, Mapping) or not isinstance(chain_item, Mapping):
                raise CampaignError("panel selection evidence must be objects")
            policy_payload = ranked.get("policy")
            if not isinstance(policy_payload, Mapping):
                raise CampaignError("ranked universe lacks its frozen policy")
            expected_ranked_keys = {
                "schema",
                "source_plan_id",
                "source_stage",
                "phase",
                "policy",
                "selection_universe_sha256",
                "ranked_candidates",
                "excluded_candidates",
                "threshold_sensitivity",
                "smoke",
                "smoke_protocol_only",
                "selection_mode",
            }
            if set(ranked) != expected_ranked_keys:
                raise CampaignError("ranked universe evidence is noncanonical")
            try:
                policy = SelectionPolicy.from_dict(policy_payload)
            except (KeyError, TypeError, ValueError, StudyError) as exc:
                raise CampaignError(f"invalid ranked-universe policy: {exc}") from exc
            if canonical_json(policy_payload) != canonical_json(policy.to_dict()):
                raise CampaignError("ranked universe policy is noncanonical")
            universe_body = {
                "plan_id": ranked.get("source_plan_id"),
                "source_stage": ranked.get("source_stage"),
                "policy_id": policy.policy_id,
                "ranked_candidates": ranked.get("ranked_candidates"),
                "excluded_candidates": ranked.get("excluded_candidates"),
            }
            if _canonical_sha256(universe_body) != ranked.get(
                "selection_universe_sha256"
            ):
                raise CampaignError("ranked stage universe hash mismatch")
            for field_name in (
                "source_plan_id",
                "source_stage",
                "selection_universe_sha256",
            ):
                if ranked.get(field_name) != chain_item.get(field_name):
                    raise CampaignError(
                        "ranked universe differs from its selection chain"
                    )
            expected_chain_keys = {
                "source_plan_id",
                "source_stage",
                "target_plan_id",
                "target_stage",
                "branch",
                "family_name",
                "selection_id",
                "selection_artifact_sha256",
                "selection_artifact_sha256_semantics",
                "selection_universe_sha256",
                "policy_id",
                "candidate_id",
                "cell_ids",
                "qualification_sha256s",
            }
            if set(chain_item) != expected_chain_keys:
                raise CampaignError("panel selection-chain item is noncanonical")
            selection_artifact_sha256 = chain_item.get("selection_artifact_sha256")
            if selection_artifact_sha256 is not None and (
                not isinstance(selection_artifact_sha256, str)
                or not selection_artifact_sha256.startswith("sha256:")
                or not _is_sha256_hex(selection_artifact_sha256.removeprefix("sha256:"))
            ):
                raise CampaignError("panel selection commitment is malformed")
            if chain_item.get("selection_artifact_sha256_semantics") != (
                "not_applicable"
                if selection_artifact_sha256 is None
                else "opaque_historical_commitment_requires_trusted_origin"
            ):
                raise CampaignError("panel selection commitment is mislabeled")
            candidates = ranked.get("ranked_candidates")
            excluded_candidates = ranked.get("excluded_candidates")
            if (
                not isinstance(candidates, list)
                or not candidates
                or not all(isinstance(candidate, Mapping) for candidate in candidates)
                or not isinstance(excluded_candidates, list)
                or not all(
                    isinstance(candidate, Mapping) for candidate in excluded_candidates
                )
            ):
                raise CampaignError("ranked universe lacks candidate evidence")
            source_stage = str(ranked.get("source_stage", ""))
            stage_cells = tuple(
                cell
                for cell in embedded_cells_by_id.values()
                if cell.stage == source_stage
            )
            normalized_stage_evidence: dict[str, dict[str, Any]] = {}
            for cell in stage_cells:
                evidence = cells_by_id[cell.cell_id]
                normalized: dict[str, Any] = {
                    "state": evidence.get("state"),
                }
                if evidence.get("state") in {
                    RunState.QUALIFIED.value,
                    RunState.PROMOTED.value,
                }:
                    normalized.update(
                        {
                            "qualification": evidence.get("qualification"),
                            "qualification_sha256": qualification_hash(evidence),
                        }
                    )
                normalized_stage_evidence[cell.cell_id] = normalized
            (
                replayed_candidates,
                replayed_excluded,
                replayed_sensitivity,
                replayed_smoke,
            ) = Campaign._selection_universe_from_evidence(
                source_stage,
                stage_cells,
                policy,
                normalized_stage_evidence,
                sharing_guard_for_cell=embedded_sharing_guard,
            )
            if (
                ranked.get("schema") != SELECTION_SCHEMA
                or ranked.get("phase") != "phase2"
                or ranked.get("smoke") is not smoke
                or ranked.get("smoke_protocol_only") is not smoke
                or ranked.get("selection_mode")
                != ("smoke_protocol_only" if smoke else "scientific_promotion")
                or replayed_smoke is not smoke
                or canonical_json(replayed_candidates) != canonical_json(candidates)
                or canonical_json(replayed_excluded)
                != canonical_json(excluded_candidates)
                or canonical_json(replayed_sensitivity)
                != canonical_json(ranked.get("threshold_sensitivity"))
            ):
                raise CampaignError(
                    "ranked/excluded universe differs from authenticated eligibility replay"
                )
            stage_candidate_ids = {
                cell.candidate_id
                for cell in embedded_cells_by_id.values()
                if cell.stage == source_stage
            }
            universe_candidate_ids = {
                str(candidate.get("candidate_id", ""))
                for candidate in (*candidates, *excluded_candidates)
                if isinstance(candidate, Mapping)
            }
            if universe_candidate_ids != stage_candidate_ids:
                raise CampaignError(
                    "ranked/excluded universe does not cover its complete source stage"
                )
            sign = 1.0 if policy.direction == "min" else -1.0
            expected_order = sorted(
                candidates,
                key=lambda item: (
                    sign * float(item["median"]),
                    sign * float(item["worst_seed"]),
                    str(item["candidate_id"]),
                ),
            )
            if [item["candidate_id"] for item in candidates] != [
                item["candidate_id"] for item in expected_order
            ]:
                raise CampaignError("ranked candidate order contradicts its metrics")
            for candidate in candidates:
                observations = candidate.get("observations")
                if not isinstance(observations, list) or not observations:
                    raise CampaignError("ranked candidate lacks observations")
                observed_seeds: list[int] = []
                recomputed_metrics: list[float] = []
                for observation in observations:
                    if not isinstance(observation, Mapping):
                        raise CampaignError(
                            "ranked candidate observation must be an object"
                        )
                    cell_id = str(observation.get("cell_id", ""))
                    cell = embedded_cells_by_id.get(cell_id)
                    evidence = cells_by_id.get(cell_id)
                    qualification = (
                        evidence.get("qualification")
                        if isinstance(evidence, Mapping)
                        else None
                    )
                    selection_metrics = (
                        qualification.get("selection_metrics")
                        if isinstance(qualification, Mapping)
                        else None
                    )
                    if (
                        cell is None
                        or evidence is None
                        or not isinstance(qualification, Mapping)
                        or not isinstance(selection_metrics, Mapping)
                        or cell.stage != source_stage
                        or cell.candidate_id != candidate.get("candidate_id")
                        or cell.recipe_name != candidate.get("recipe_name")
                        or cell.recipe_id != candidate.get("recipe_id")
                        or observation.get("seed") != cell.seed
                        or observation.get("qualification_sha256")
                        != qualification_hash(evidence)
                    ):
                        raise CampaignError(
                            "ranked candidate observation lacks exact authenticated evidence"
                        )
                    if smoke:
                        if (
                            qualification.get("selection_eligibility_mode")
                            != "smoke_protocol_only"
                            or qualification.get("selection_eligible_for_protocol_test")
                            is not True
                        ):
                            raise CampaignError(
                                "smoke ranked candidate is not protocol eligible"
                            )
                        expected_metric = 0.0
                    else:
                        if (
                            qualification.get("scientific_outcome", {}).get("passed")
                            is not True
                            or qualification.get("promotion_eligible") is not True
                        ):
                            raise CampaignError(
                                "scientific ranked candidate is not promotion eligible"
                            )
                        expected_metric = Campaign._policy_metric(
                            selection_metrics,
                            policy,
                        )
                    if not math.isclose(
                        float(observation.get("metric")),
                        expected_metric,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    ):
                        raise CampaignError(
                            "ranked candidate metric differs from authenticated qualification"
                        )
                    expected_guard = (
                        embedded_sharing_guard(cell, selection_metrics, policy)
                        if policy.require_sharing_guard and not smoke
                        else None
                    )
                    if observation.get("sharing_guard") != expected_guard:
                        raise CampaignError(
                            "ranked candidate sharing guard differs from embedded evidence"
                        )
                    observed_seeds.append(cell.seed)
                    recomputed_metrics.append(expected_metric)
                if tuple(sorted(observed_seeds)) != replay_blueprint.seeds:
                    raise CampaignError(
                        "ranked candidate does not cover every blueprint seed"
                    )
                expected_median = float(median(recomputed_metrics))
                expected_worst = (
                    max(recomputed_metrics)
                    if policy.direction == "min"
                    else min(recomputed_metrics)
                )
                if not math.isclose(
                    float(candidate.get("median")),
                    expected_median,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ) or not math.isclose(
                    float(candidate.get("worst_seed")),
                    expected_worst,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise CampaignError(
                        "ranked candidate aggregate differs from authenticated metrics"
                    )
            if policy.retain_count != 1:
                raise CampaignError(
                    "adaptive Phase-2 selection chain must retain exactly one candidate"
                )
            is_validated_duplicate_substitution = (
                canonical_json(chain_item) in validated_duplicate_substitution_chains
            )
            retained_candidate_ids = {
                str(candidate.get("candidate_id", ""))
                for candidate in _policy_retained_candidates(
                    candidates,
                    policy,
                    smoke_protocol_only=smoke,
                )
            }
            if (
                str(chain_item.get("candidate_id", "")) not in retained_candidate_ids
                and not is_validated_duplicate_substitution
            ):
                raise CampaignError(
                    "selection chain does not name a policy-retained candidate"
                )
            if (
                chain_item.get("branch") == "comparator_family_duplicate_substitute"
                and not is_validated_duplicate_substitution
            ):
                raise CampaignError(
                    "selection chain names an unauthenticated duplicate substitute"
                )
            chosen = [
                candidate
                for candidate in candidates
                if isinstance(candidate, Mapping)
                and candidate.get("candidate_id") == chain_item.get("candidate_id")
            ]
            if len(chosen) != 1:
                raise CampaignError(
                    "selection chain does not name exactly one ranked candidate"
                )
            observations = chosen[0].get("observations")
            if not isinstance(observations, list) or not observations:
                raise CampaignError("selected ranked candidate lacks observations")
            selected_cells: list[CellSpec] = []
            metric_values: list[float] = []
            qualification_sha256s: list[str] = []
            for observation in observations:
                if not isinstance(observation, Mapping):
                    raise CampaignError(
                        "ranked candidate observation must be an object"
                    )
                cell_id = str(observation.get("cell_id", ""))
                cell = embedded_cells_by_id.get(cell_id)
                evidence = cells_by_id.get(cell_id)
                if (
                    cell is None
                    or evidence is None
                    or cell.stage != ranked.get("source_stage")
                    or cell.candidate_id != chosen[0].get("candidate_id")
                    or observation.get("seed") != cell.seed
                    or observation.get("qualification_sha256")
                    != qualification_hash(evidence)
                ):
                    raise CampaignError(
                        "ranked candidate observation lacks exact cell evidence"
                    )
                selected_cells.append(cell)
                metric_values.append(float(observation["metric"]))
                qualification_sha256s.append(str(observation["qualification_sha256"]))
            try:
                replayed_selection = FrozenSelection.from_cells(
                    policy,
                    selected_cells,
                    metric_values,
                    qualification_sha256s,
                    str(ranked["selection_universe_sha256"]),
                )
            except (KeyError, TypeError, ValueError, StudyError) as exc:
                raise CampaignError(
                    f"invalid ranked selection evidence: {exc}"
                ) from exc
            if (
                chain_item.get("selection_id") != replayed_selection.selection_id
                or chain_item.get("policy_id") != replayed_selection.policy_id
                or chain_item.get("candidate_id") != replayed_selection.candidate_id
                or tuple(chain_item.get("cell_ids", ())) != replayed_selection.cell_ids
                or tuple(chain_item.get("qualification_sha256s", ()))
                != replayed_selection.qualification_sha256s
            ):
                raise CampaignError(
                    "selection chain differs from its reconstructed frozen selection"
                )
            chain_bindings.append((chain_item, replayed_selection, policy))

        # Rebuild the entire adaptive Phase-2 plan from the authenticated
        # Phase-1 transfer, ranked selections, and family nominations.  The
        # content IDs in an envelope are claims, not evidence, until this
        # replay reaches the exact embedded cell universe.
        extension_bindings = [
            (index, chain_item, selection, policy)
            for index, (chain_item, selection, policy) in enumerate(chain_bindings)
            if chain_item.get("target_plan_id") is not None
        ]
        final_bindings = [
            (index, chain_item, selection, policy)
            for index, (chain_item, selection, policy) in enumerate(chain_bindings)
            if chain_item.get("target_plan_id") is None
        ]
        used_chain_indices: set[int] = set()
        used_nomination_sources: set[str] = set()
        replayed_extension_rows: list[Mapping[str, Any]] = []
        replayed_plans = [replay_plan]
        max_extensions = len(extension_bindings) + len(nominations_by_source_plan)
        for _ in range(max_extensions + 1):
            if replay_plan.plan_id == decision.source_phase2_plan_id:
                break
            chain_matches = [
                item
                for item in extension_bindings
                if item[0] not in used_chain_indices
                and item[1].get("source_plan_id") == replay_plan.plan_id
            ]
            nomination_matches = [
                (source_plan_id, nomination)
                for source_plan_id, nomination in nominations_by_source_plan.items()
                if source_plan_id not in used_nomination_sources
                and source_plan_id == replay_plan.plan_id
            ]
            if len(chain_matches) + len(nomination_matches) != 1:
                raise CampaignError(
                    "Phase-2 selection lineage does not define one acyclic plan replay"
                )
            if chain_matches:
                index, chain_item, selection, policy = chain_matches[0]
                source_stages = [
                    stage
                    for stage in replay_plan.stages
                    if stage.name == chain_item.get("source_stage")
                ]
                if len(source_stages) != 1:
                    raise CampaignError(
                        "Phase-2 selection source stage is absent from its source plan"
                    )
                source_stage = source_stages[0]
                source_cells = {cell.cell_id: cell for cell in source_stage.cells}
                try:
                    replay_cells = tuple(
                        source_cells[cell_id] for cell_id in selection.cell_ids
                    )
                except KeyError as exc:
                    raise CampaignError(
                        "Phase-2 selection escaped its replayed source stage"
                    ) from exc
                if replay_cells != tuple(
                    embedded_cells_by_id[cell_id] for cell_id in selection.cell_ids
                ):
                    raise CampaignError(
                        "Phase-2 selected cells differ from blueprint replay"
                    )
                branch = chain_item.get("branch")
                try:
                    if branch == "main":
                        if (
                            chain_item.get("family_name") is not None
                            or source_stage.selection_policy != policy
                        ):
                            raise CampaignError(
                                "main-chain selection has the wrong policy or family"
                            )
                        extended = materialize_child_plan(
                            replay_plan, replay_blueprint, selection
                        )
                    elif branch == "comparator_family":
                        family_name = str(chain_item.get("family_name", ""))
                        family = family_blueprints.get(family_name)
                        if family is None:
                            raise CampaignError(
                                "family-chain selection names an unknown family"
                            )
                        expected_policy = (
                            family.root_selection_policy
                            if source_stage.name == replay_blueprint.initial_stage.name
                            else source_stage.selection_policy
                        )
                        if expected_policy != policy:
                            raise CampaignError(
                                "family-chain selection has the wrong policy"
                            )
                        extended = materialize_family_child_plan(
                            replay_plan,
                            replay_blueprint,
                            family_name,
                            selection,
                        )
                    else:
                        raise CampaignError(
                            "plan-extending selection has an invalid branch"
                        )
                except StudyError as exc:
                    raise CampaignError(
                        f"Phase-2 selection cannot replay its blueprint: {exc}"
                    ) from exc
                if (
                    chain_item.get("target_plan_id") != extended.plan_id
                    or chain_item.get("target_stage") != extended.stages[-1].name
                    or not isinstance(chain_item.get("selection_artifact_sha256"), str)
                    or not str(chain_item["selection_artifact_sha256"]).startswith(
                        "sha256:"
                    )
                ):
                    raise CampaignError(
                        "Phase-2 selection target differs from blueprint replay"
                    )
                used_chain_indices.add(index)
                replayed_extension_rows.append(chain_item)
            else:
                source_plan_id, nomination = nomination_matches[0]
                family_name = str(nomination.get("family_name", ""))
                family = family_blueprints.get(family_name)
                selections = nomination_selections_by_family.get(family_name)
                if family is None or selections is None:
                    raise CampaignError("family revisit lacks its canonical blueprint")
                for selection in selections:
                    matching_stages = [
                        stage
                        for stage in replay_plan.stages
                        if stage.name == selection.source_stage
                    ]
                    if len(matching_stages) != 1:
                        raise CampaignError(
                            "family nomination source stage is absent from replay"
                        )
                    source_cells = {
                        cell.cell_id: cell for cell in matching_stages[0].cells
                    }
                    try:
                        selected_source_cells = tuple(
                            source_cells[cell_id] for cell_id in selection.cell_ids
                        )
                    except KeyError as exc:
                        raise CampaignError(
                            "family nomination escaped its replayed source stage"
                        ) from exc
                    if selected_source_cells != tuple(
                        embedded_cells_by_id[cell_id] for cell_id in selection.cell_ids
                    ):
                        raise CampaignError(
                            "family nomination cells differ from blueprint replay"
                        )
                try:
                    extended = materialize_family_revisit_plan(
                        replay_plan,
                        replay_blueprint,
                        family_name,
                        selections,
                    )
                except StudyError as exc:
                    raise CampaignError(
                        f"family revisit cannot replay its blueprint: {exc}"
                    ) from exc
                artifact_hash = nomination.get("nomination_artifact_sha256")
                if (
                    nomination.get("target_plan_id") != extended.plan_id
                    or not isinstance(artifact_hash, str)
                    or not artifact_hash.startswith("sha256:")
                ):
                    raise CampaignError(
                        "family nomination target differs from blueprint replay"
                    )
                used_nomination_sources.add(source_plan_id)
            replay_plan = extended
            replayed_plans.append(replay_plan)
        else:  # pragma: no cover - bounded by the finite action count
            raise CampaignError("Phase-2 plan replay exceeded its extension bound")
        if (
            replay_plan.plan_id != decision.source_phase2_plan_id
            or len(used_chain_indices) != len(extension_bindings)
            or len(used_nomination_sources) != len(nominations_by_source_plan)
        ):
            raise CampaignError("Phase-2 plan lineage is incomplete or stale")
        replayed_cells = {cell.cell_id: cell for cell in replay_plan.cells}
        if replayed_cells != embedded_cells_by_id or [
            item.get("cell_id") for item in cells
        ] != [cell.cell_id for cell in replay_plan.cells]:
            raise CampaignError(
                "Phase-2 campaign cells differ from the exact replayed plan"
            )
        if campaign_manifest.get("plan_sha256") != _run_cell_json_sha256(
            replay_plan.to_manifest()
        ) or campaign_manifest.get("blueprint_sha256") != _run_cell_json_sha256(
            replay_blueprint.to_manifest()
        ):
            raise CampaignError("Phase-2 plan/blueprint file hash is stale")
        plan_history = campaign_manifest.get("plan_history")
        if not isinstance(plan_history, list) or len(plan_history) != len(
            replayed_plans
        ):
            raise CampaignError("Phase-2 plan history is incomplete")
        for item, replayed in zip(plan_history, replayed_plans, strict=True):
            if (
                not isinstance(item, Mapping)
                or item.get("plan_id") != replayed.plan_id
                or item.get("sha256") != _run_cell_json_sha256(replayed.to_manifest())
            ):
                raise CampaignError("Phase-2 plan history differs from exact replay")

        final_rows_by_family: dict[str, Mapping[str, Any]] = {}
        for index, chain_item, selection, policy in final_bindings:
            family_name = str(chain_item.get("family_name", ""))
            family = family_blueprints.get(family_name)
            matching_stages = [
                stage
                for stage in replay_plan.stages
                if stage.name == chain_item.get("source_stage")
            ]
            if (
                family is None
                or family_name in final_rows_by_family
                or len(matching_stages) != 1
                or matching_stages[0].name != family.revisit.name
                or matching_stages[0].selection_policy != policy
                or chain_item.get("source_plan_id") != replay_plan.plan_id
                or chain_item.get("target_stage") is not None
                or chain_item.get("selection_artifact_sha256") is not None
                or chain_item.get("branch")
                not in {
                    "comparator_family_final",
                    "comparator_family_duplicate_substitute",
                }
            ):
                raise CampaignError("family-final selection lineage is malformed")
            source_cells = {cell.cell_id: cell for cell in matching_stages[0].cells}
            try:
                selected_source_cells = tuple(
                    source_cells[cell_id] for cell_id in selection.cell_ids
                )
            except KeyError as exc:
                raise CampaignError(
                    "family-final selection escaped its replayed revisit stage"
                ) from exc
            if selected_source_cells != tuple(
                embedded_cells_by_id[cell_id] for cell_id in selection.cell_ids
            ):
                raise CampaignError(
                    "family-final selection cells differ from blueprint replay"
                )
            final_rows_by_family[family_name] = chain_item
            used_chain_indices.add(index)
        if set(final_rows_by_family) != set(family_blueprints) or len(
            used_chain_indices
        ) != len(chain_bindings):
            raise CampaignError("panel lacks one final selection per comparator family")

        expected_main_chain = [
            row for row in replayed_extension_rows if row.get("branch") == "main"
        ]
        expected_family_chains = {
            family.name: [
                *(
                    row
                    for row in replayed_extension_rows
                    if row.get("family_name") == family.name
                ),
                final_rows_by_family[family.name],
            ]
            for family in replay_blueprint.comparator_families
        }
        expected_flat_chain = [
            *replayed_extension_rows,
            *(
                final_rows_by_family[family.name]
                for family in replay_blueprint.comparator_families
            ),
        ]
        if (
            canonical_json(main_chain) != canonical_json(expected_main_chain)
            or canonical_json(family_chains) != canonical_json(expected_family_chains)
            or canonical_json(selection_chain) != canonical_json(expected_flat_chain)
        ):
            raise CampaignError(
                "main/family selection chains differ from exact plan replay"
            )

        main_ids = tuple(str(item["selection_id"]) for item in expected_main_chain)
        for entry in decision.entries:
            if entry.role == "selected_finalist":
                if entry.selection_ids != main_ids or any(
                    tuple(cell.decision_map["selection.upstream_selection_ids"])
                    != main_ids
                    for cell in entry.source_cells
                ):
                    raise CampaignError(
                        "panel finalist selection chain differs from main evidence"
                    )
                continue
            family_chain = expected_family_chains.get(entry.panel_slot)
            nominations = nomination_selections_by_family.get(entry.panel_slot)
            final_row = final_rows_by_family.get(entry.panel_slot)
            if family_chain is None or nominations is None or final_row is None:
                raise CampaignError("panel comparator lacks its family lineage")
            family_ids = tuple(str(item["selection_id"]) for item in family_chain)
            nomination_ids = tuple(selection.selection_id for selection in nominations)
            expected_ids = tuple(
                dict.fromkeys((*family_ids[:-1], *nomination_ids, family_ids[-1]))
            )
            final_selection = next(
                selection
                for row, selection, _policy in (
                    (item[1], item[2], item[3]) for item in final_bindings
                )
                if row is final_row
            )
            if (
                entry.selection_ids != expected_ids
                or entry.source_cell_ids != final_selection.cell_ids
                or entry.qualification_sha256s != final_selection.qualification_sha256s
            ):
                raise CampaignError(
                    "panel comparator differs from its exact family selection lineage"
                )
        duplicate_configurations = Campaign._duplicate_projected_configurations(
            decision, smoke=campaign_manifest.get("smoke") is True
        )
        if duplicate_configurations:
            raise CampaignError(
                "panel contains duplicate projected scientific configurations: "
                + canonical_json(duplicate_configurations)
            )
        return decision

    def _register_cells(
        self,
        cells: Sequence[CellSpec],
        *,
        plan_id: str,
    ) -> None:
        known = {
            str(event["cell_id"])
            for event in self._events_cached()
            if event.get("event") == "transition" and event.get("previous") is None
        }
        for cell in cells:
            path = self.cell_manifest_path(cell.cell_id)
            if path.exists():
                existing_payload = _read_json(path)
                existing_cell = CellSpec.from_manifest(existing_payload)
                if (
                    existing_cell != cell
                    or canonical_json(existing_payload)
                    != canonical_json(cell.to_manifest())
                ):
                    raise CampaignError(f"cell manifest mismatch at {path}")
            else:
                _atomic_json(path, cell.to_manifest())
            if cell.cell_id not in known:
                event = self._event(
                    "transition",
                    cell.cell_id,
                    previous=None,
                    target=RunState.PLANNED,
                    message="registered",
                    metadata={"plan_id": plan_id},
                    artifacts=(),
                )
                self._append_event(event)
                known.add(cell.cell_id)
            try:
                self._write_snapshot(self.record(cell.cell_id))
            except (CampaignError, OSError):
                # Registration is committed by the journal.  Disposable
                # snapshots are repaired by the runner/reconcile path.
                pass

    def _commit_plan_extension(
        self,
        *,
        current: StudyPlan,
        plan: StudyPlan,
        child_stage: Any,
        evidence_ref: ArtifactRef,
        frozen_evidence: Mapping[str, Any],
        live_evidence: Callable[[], Mapping[str, Any]],
        journal_sha256: str | None,
        message: str,
        metadata: Mapping[str, Any],
    ) -> None:
        """Atomically revalidate and commit one plan extension."""

        with self._campaign_mutation():
            if self.plan != current:
                raise CampaignError(
                    "active plan changed during plan-extension validation"
                )
            observed = (
                _sha256(self.journal_path) if self.journal_path.is_file() else None
            )
            if observed != journal_sha256:
                raise CampaignError(
                    "campaign journal changed while validating the plan extension"
                )
            self._verify_artifact(evidence_ref)
            if canonical_json(live_evidence()) != canonical_json(frozen_evidence):
                raise CampaignError(
                    "selection evidence changed during plan-extension validation"
                )
            if (
                _sha256(self.journal_path) if self.journal_path.is_file() else None
            ) != journal_sha256:
                raise CampaignError(
                    "campaign journal changed during final extension validation"
                )
            _write_immutable_json(
                self.plans_dir / f"{_slug(current.plan_id)}.json",
                current.to_manifest(),
            )
            _write_immutable_json(
                self.plans_dir / f"{_slug(plan.plan_id)}.json",
                plan.to_manifest(),
            )
            self._register_cells(child_stage.cells, plan_id=plan.plan_id)
            event = self._event(
                "plan_extension",
                "__campaign__",
                message=message,
                metadata=metadata,
                artifacts=(evidence_ref,),
            )
            self._append_event(event)
            try:
                _atomic_json(self.plan_path, plan.to_manifest())
            except (CampaignError, OSError):
                # The journal event is authoritative and makes an identical
                # retry observe the new plan rather than duplicating it.
                pass

    def extend(
        self,
        plan: StudyPlan,
        *,
        selection: FrozenSelection,
        selection_path: str | Path,
        family_name: str | None = None,
    ) -> None:
        """Append exactly one blueprint-derived stage to the active plan.

        The current plan must be an exact prefix, the frozen selection must be
        present in the immutable stage-selection artifact, and its cell/hash
        evidence is revalidated against the live journal before the new plan
        pointer is advanced.
        """

        durable_mkdir(self.root, parents=True, exist_ok=True)
        registration_lock = self.root / ".registration.lock"
        with registration_lock.open("a+", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            journal_sha256 = (
                _sha256(self.journal_path) if self.journal_path.is_file() else None
            )
            current = self.plan
            if plan.phase is not current.phase:
                raise CampaignError("a campaign extension cannot change phase")
            if len(plan.stages) != len(current.stages) + 1:
                raise CampaignError(
                    "a campaign extension must append exactly one stage"
                )
            if plan.stages[:-1] != current.stages:
                raise CampaignError(
                    "campaign extension does not preserve the exact plan prefix"
                )
            if not self.blueprint_path.is_file():
                raise CampaignError("campaign extension requires its frozen blueprint")
            blueprint_manifest = _read_json(self.blueprint_path)
            if current.phase.value == "phase1":
                if family_name is not None:
                    raise CampaignError("Phase 1 has no comparator-family branches")
                blueprint: Phase1Blueprint | Phase2Blueprint = (
                    Phase1Blueprint.from_manifest(blueprint_manifest)
                )
            elif current.phase.value == "phase2":
                blueprint = Phase2Blueprint.from_manifest(blueprint_manifest)
            else:
                raise CampaignError("Phase 3 has no adaptive campaign extension")

            if family_name is None:
                source_matches = [
                    stage
                    for stage in current.stages
                    if stage.name == selection.source_stage
                ]
                if len(source_matches) != 1:
                    raise CampaignError(
                        "main-chain selection source is absent or ambiguous"
                    )
                source_stage = source_matches[0]
                main_stage_names = (
                    {
                        blueprint.initial_stage.name,
                        *(round_spec.name for round_spec in blueprint.rounds),
                    }
                    if isinstance(blueprint, Phase2Blueprint)
                    else {
                        *(stage.name for stage in blueprint.initial_stages),
                        *(round_spec.name for round_spec in blueprint.rounds),
                    }
                )
                if source_stage.name not in main_stage_names:
                    raise CampaignError(
                        "main-chain selection comes from a comparator-family stage"
                    )
                source_policy = source_stage.selection_policy
            else:
                assert isinstance(blueprint, Phase2Blueprint)
                family = self._family_blueprint(blueprint, family_name)
                stage_matches = [
                    stage
                    for stage in current.stages
                    if stage.name == selection.source_stage
                ]
                if len(stage_matches) != 1:
                    raise CampaignError(
                        "family selection source stage is absent or ambiguous"
                    )
                source_stage = stage_matches[0]
                if source_stage.name == blueprint.initial_stage.name:
                    source_policy = family.root_selection_policy
                elif source_stage.name in {
                    round_spec.name for round_spec in family.rounds
                }:
                    source_policy = source_stage.selection_policy
                else:
                    raise CampaignError(
                        "family selection does not come from its anchor or calibration branch"
                    )
            if source_policy is None:
                raise CampaignError("the selected source stage is not selectable")
            if selection.policy_id != source_policy.policy_id:
                raise CampaignError("selection is bound to a different frozen policy")

            resolved_selection_path = Path(selection_path)
            if not resolved_selection_path.is_absolute():
                resolved_selection_path = self.root / resolved_selection_path
            selection_ref = ArtifactRef.from_path(
                "stage_selection", resolved_selection_path, root=self.root
            )
            selection_payload = _read_json(selection_ref.resolve(self.root))
            if selection_payload.get("schema") != SELECTION_SCHEMA:
                raise CampaignError("stage-selection artifact has the wrong schema")
            if (
                selection_payload.get("plan_id") != current.plan_id
                or selection_payload.get("source_stage") != source_stage.name
                or selection_payload.get("policy", {}).get("policy_id")
                != source_policy.policy_id
            ):
                raise CampaignError("stage-selection artifact binding mismatch")
            body = dict(selection_payload)
            declared_selection_artifact_id = body.pop("selection_id", None)
            if declared_selection_artifact_id != content_id(body, prefix="selection"):
                raise CampaignError("stage-selection artifact content ID mismatch")
            selected_payloads = selection_payload.get("selected")
            if not isinstance(selected_payloads, list):
                raise CampaignError(
                    "stage-selection artifact has no selected candidates"
                )
            selected = {
                item.selection_id: item
                for item in (
                    FrozenSelection.from_dict(payload) for payload in selected_payloads
                )
            }
            if selected.get(selection.selection_id) != selection:
                raise CampaignError(
                    "requested frozen selection is absent from the artifact"
                )
            if selection.selection_universe_sha256 != selection_payload.get(
                "selection_universe_sha256"
            ):
                raise CampaignError("selection universe hash mismatch")

            # A selection file is a snapshot of *every* ranked and excluded
            # candidate, not a durable lease on the selected row alone.  A
            # later failure/promotion/evidence edit anywhere in the stage must
            # therefore be visible and invalidate advancement.
            live_selection_payload = self._selection_payload(
                source_stage.name,
                policy_override=(
                    source_policy
                    if source_policy is not source_stage.selection_policy
                    else None
                ),
            )
            if canonical_json(live_selection_payload) != canonical_json(
                selection_payload
            ):
                raise CampaignError(
                    "stage selection is stale relative to the complete live universe"
                )

            source_cells = {cell.cell_id: cell for cell in source_stage.cells}
            try:
                parent_cells = tuple(source_cells[item] for item in selection.cell_ids)
            except KeyError as exc:
                raise CampaignError(
                    "selection names a cell outside the source stage"
                ) from exc
            if any(
                cell.candidate_id != selection.candidate_id for cell in parent_cells
            ):
                raise CampaignError("selection parent candidate identity mismatch")
            smoke_protocol_only = (
                selection_payload.get("smoke_protocol_only") is True
                and selection_payload.get("selection_mode") == "smoke_protocol_only"
            )
            for cell, expected_hash, expected_metric in zip(
                parent_cells,
                selection.qualification_sha256s,
                selection.metric_values,
                strict=True,
            ):
                record = self.record(cell.cell_id)
                if record.state not in {RunState.QUALIFIED, RunState.PROMOTED}:
                    raise CampaignError("a selected parent is no longer qualified")
                self._validate_artifact_gate(
                    cell.cell_id, RunState.QUALIFIED, record.artifact_map
                )
                actual_hash = "sha256:" + record.artifact_map["qualification"].sha256
                if actual_hash != expected_hash:
                    raise CampaignError("selected qualification hash changed")
                qualification = self._qualification_payload(record)
                if qualification is None:
                    raise CampaignError("selected parent lacks qualification evidence")
                if smoke_protocol_only:
                    if (
                        qualification.get("selection_eligibility_mode")
                        != "smoke_protocol_only"
                        or qualification.get("selection_eligible_for_protocol_test")
                        is not True
                    ):
                        raise CampaignError(
                            "selected smoke parent lacks protocol-test eligibility"
                        )
                    actual_metric = 0.0
                else:
                    amendment = self._active_phase2_gate_amendment()
                    interpretation = self._selection_qualification_interpretation(
                        cell,
                        qualification,
                        None if amendment is None else amendment[0],
                    )
                    if interpretation["scientific_outcome_passed"] is not True:
                        raise CampaignError(
                            "selected parent failed its scientific outcome"
                        )
                    if interpretation["promotion_eligible"] is not True:
                        raise CampaignError(
                            "selected parent is not promotion eligible"
                        )
                    actual_metric = self._policy_metric(
                        qualification["selection_metrics"],
                        source_policy,
                    )
                if actual_metric != expected_metric:
                    raise CampaignError("selected metric differs from frozen evidence")

            child_stage = plan.stages[-1]
            if child_stage.depends_on != (source_stage.name,):
                raise CampaignError(
                    "appended stage must depend only on its selected source"
                )
            blueprint_id = blueprint.blueprint_id
            expected_plan = (
                materialize_child_plan(current, blueprint, selection)
                if family_name is None
                else materialize_family_child_plan(
                    current, blueprint, family_name, selection
                )
            )
            if plan != expected_plan:
                raise CampaignError(
                    "campaign extension differs from the next frozen blueprint round"
                )
            expected_bindings = {
                "selection.id": selection.selection_id,
                "selection.parent_candidate_id": selection.candidate_id,
                "selection.parent_cell_ids": selection.cell_ids,
                "selection.source_plan_id": current.plan_id,
                "selection.source_blueprint_id": blueprint_id,
                "selection.qualification_sha256s": selection.qualification_sha256s,
                "selection.universe_sha256": selection.selection_universe_sha256,
            }
            for cell in child_stage.cells:
                values = cell.decision_map
                mismatched = {
                    name: values.get(name)
                    for name, expected in expected_bindings.items()
                    if values.get(name) != expected
                }
                if mismatched:
                    raise CampaignError(
                        "derived child lacks frozen selection bindings: "
                        + canonical_json(mismatched)
                    )

            event_metadata = {
                "previous_plan_id": current.plan_id,
                "plan_id": plan.plan_id,
                "stage": child_stage.name,
                "selection_id": selection.selection_id,
            }
            if family_name is not None:
                event_metadata.update(
                    {
                        "branch": "comparator_family",
                        "family_name": family_name,
                    }
                )
            self._commit_plan_extension(
                current=current,
                plan=plan,
                child_stage=child_stage,
                evidence_ref=selection_ref,
                frozen_evidence=selection_payload,
                live_evidence=lambda: self._selection_payload(
                    source_stage.name,
                    policy_override=(
                        source_policy
                        if source_policy is not source_stage.selection_policy
                        else None
                    ),
                ),
                journal_sha256=journal_sha256,
                message=f"appended stage {child_stage.name}",
                metadata=event_metadata,
            )

    def extend_family(
        self,
        plan: StudyPlan,
        *,
        family_name: str,
        selection: FrozenSelection,
        selection_path: str | Path,
    ) -> None:
        """Append one exact comparator-family branch round."""

        self.extend(
            plan,
            selection=selection,
            selection_path=selection_path,
            family_name=family_name,
        )

    def extend_family_revisit(
        self,
        plan: StudyPlan,
        *,
        family_name: str,
        selection_path: str | Path,
    ) -> None:
        """Append a family's fresh 16M top-two revisit from its union nomination."""

        durable_mkdir(self.root, parents=True, exist_ok=True)
        registration_lock = self.root / ".registration.lock"
        with registration_lock.open("a+", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            current = self.plan
            blueprint = self._phase2_blueprint()
            family = self._family_blueprint(blueprint, family_name)
            if (
                plan.phase is not current.phase
                or len(plan.stages) != len(current.stages) + 1
                or plan.stages[:-1] != current.stages
            ):
                raise CampaignError(
                    "family revisit must append exactly one stage to the active plan"
                )
            resolved_path = Path(selection_path)
            if not resolved_path.is_absolute():
                resolved_path = self.root / resolved_path
            nomination_ref = ArtifactRef.from_path(
                "family_nomination", resolved_path, root=self.root
            )
            nomination_payload = _read_json(nomination_ref.resolve(self.root))
            if nomination_payload.get("schema") != FAMILY_NOMINATION_SCHEMA:
                raise CampaignError("family nomination artifact has the wrong schema")
            body = dict(nomination_payload)
            nomination_id = body.pop("nomination_id", None)
            if nomination_id != content_id(body, prefix="family-nomination"):
                raise CampaignError("family nomination artifact content ID mismatch")
            expected_binding = {
                "plan_id": current.plan_id,
                "blueprint_id": blueprint.blueprint_id,
                "family_name": family.name,
                "family_id": family.family_id,
                "revisit_id": family.revisit.revisit_id,
            }
            if any(
                nomination_payload.get(key) != value
                for key, value in expected_binding.items()
            ):
                raise CampaignError("family nomination artifact binding mismatch")
            journal_sha256 = (
                _sha256(self.journal_path) if self.journal_path.is_file() else None
            )
            live_payload = self._family_nomination_payload(family_name)
            if canonical_json(nomination_payload) != canonical_json(live_payload):
                raise CampaignError(
                    "family nomination is stale relative to the complete live union"
                )
            selected_payloads = nomination_payload.get("selected")
            if not isinstance(selected_payloads, list):
                raise CampaignError("family nomination has no selected candidates")
            try:
                selections = tuple(
                    FrozenSelection.from_dict(item) for item in selected_payloads
                )
            except (KeyError, TypeError, ValueError, StudyError) as exc:
                raise CampaignError(
                    f"invalid family nomination selection: {exc}"
                ) from exc
            if len(selections) != family.revisit.top_k:
                raise CampaignError("family nomination does not bind its exact top-k")
            expected_plan = materialize_family_revisit_plan(
                current, blueprint, family_name, selections
            )
            if plan != expected_plan:
                raise CampaignError(
                    "family revisit differs from its frozen top-two nomination"
                )
            child_stage = plan.stages[-1]
            if child_stage.name != family.revisit.name:
                raise CampaignError(
                    "family revisit stage name differs from its blueprint"
                )
            expected_selection_ids = {
                selection.selection_id for selection in selections
            }
            for cell in child_stage.cells:
                values = cell.decision_map
                if (
                    values.get("selection.id") not in expected_selection_ids
                    or values.get("selection.comparator_family_name") != family.name
                    or values.get("selection.comparator_family_blueprint_id")
                    != family.family_id
                    or values.get("selection.family_revisit_id")
                    != family.revisit.revisit_id
                ):
                    raise CampaignError(
                        "family revisit child lacks frozen nomination lineage"
                    )
            self._commit_plan_extension(
                current=current,
                plan=plan,
                child_stage=child_stage,
                evidence_ref=nomination_ref,
                frozen_evidence=nomination_payload,
                live_evidence=lambda: self._family_nomination_payload(family_name),
                journal_sha256=journal_sha256,
                message=f"appended family revisit {child_stage.name}",
                metadata={
                    "previous_plan_id": current.plan_id,
                    "plan_id": plan.plan_id,
                    "stage": child_stage.name,
                    "branch": "comparator_family_revisit",
                    "family_name": family.name,
                    "nomination_id": nomination_id,
                    "selection_ids": [
                        selection.selection_id for selection in selections
                    ],
                },
            )

    def _require_cell(self, cell_id: str) -> CellSpec:
        _active, registrations = self._journal_authority()
        plan_id = registrations.get(cell_id)
        if plan_id is None:
            raise CampaignError(f"unknown cell {cell_id}")
        history = self._history_plan(plan_id)
        matches = tuple(cell for cell in history.cells if cell.cell_id == cell_id)
        if len(matches) != 1:
            raise CampaignError(
                f"journal-bound plan {plan_id} does not define cell {cell_id} exactly once"
            )
        authoritative = matches[0]
        path = self.cell_manifest_path(cell_id)
        if not path.is_file():
            raise ArtifactError(f"registered cell projection is missing: {path}")
        try:
            before = _ArtifactFingerprint.from_path(path)
            cached = self._cell_projection_cache.get(cell_id)
            if cached is not None and cached[0] == before:
                projected = cached[1]
            else:
                payload = _read_json(path)
                projected = CellSpec.from_manifest(payload)
                after = _ArtifactFingerprint.from_path(path)
                if after != before:
                    raise ArtifactError(
                        f"cell projection changed while reading {cell_id}"
                    )
                if canonical_json(payload) != canonical_json(projected.to_manifest()):
                    raise ArtifactError(
                        f"registered cell projection is noncanonical: {cell_id}"
                    )
                self._cell_projection_cache[cell_id] = (after, projected)
        except (KeyError, OSError, TypeError, ValueError, StudyError) as exc:
            raise ArtifactError(
                f"registered cell projection is invalid for {cell_id}: {exc}"
            ) from exc
        if projected != authoritative:
            raise ArtifactError(
                f"registered cell projection differs from journal-bound plan: {cell_id}"
            )
        return authoritative

    def _require_active_cell(self, cell_id: str) -> CellSpec:
        active, registrations = self._journal_authority()
        missing = {item.cell_id for item in active.cells}.difference(registrations)
        if missing:
            raise CampaignError(
                "active plan registration is incomplete; missing cells: "
                + ", ".join(sorted(missing))
            )
        cell = self._require_cell(cell_id)
        matches = tuple(item for item in active.cells if item.cell_id == cell_id)
        if len(matches) != 1 or matches[0] != cell:
            raise CampaignError(
                f"cell {cell_id} is not a member of the committed active plan"
            )
        return cell

    def _event(
        self,
        event_type: str,
        cell_id: str,
        *,
        previous: RunState | None = None,
        target: RunState | None = None,
        message: str = "",
        metadata: Mapping[str, Any] | None = None,
        artifacts: Iterable[ArtifactRef] = (),
    ) -> dict[str, Any]:
        timestamp = float(self.clock())
        event = {
            "schema": CAMPAIGN_SCHEMA,
            "event": event_type,
            "event_id": uuid.uuid4().hex,
            "timestamp": timestamp,
            "cell_id": cell_id,
            "previous": None if previous is None else previous.value,
            "target": None if target is None else target.value,
            "message": message,
            "metadata": dict(metadata or {}),
            "artifacts": [item.to_dict() for item in artifacts],
        }
        # Validate serializability and reject NaN before touching the journal.
        canonical_json(event)
        return event

    def _append_event(self, event: Mapping[str, Any]) -> None:
        with self._campaign_mutation():
            self._append_event_locked(event)

    def _append_event_locked(self, event: Mapping[str, Any]) -> None:
        durable_mkdir(self.root, parents=True, exist_ok=True)
        journal_existed = self.journal_path.exists()
        body = (canonical_json(event) + "\n").encode("utf-8")
        fd = os.open(
            self.journal_path,
            os.O_RDWR | os.O_CREAT | os.O_APPEND,
            0o600,
        )
        recovered_tail = False
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            initial_size = os.fstat(fd).st_size
            if initial_size and os.pread(fd, 1, initial_size - 1) != b"\n":
                position = initial_size
                complete_end = 0
                while position > 0:
                    start = max(0, position - 65_536)
                    chunk = os.pread(fd, position - start, start)
                    newline = chunk.rfind(b"\n")
                    if newline >= 0:
                        complete_end = start + newline + 1
                        break
                    position = start
                torn_hasher = hashlib.sha256()
                offset = complete_end
                while offset < initial_size:
                    chunk = os.pread(fd, min(1 << 20, initial_size - offset), offset)
                    if not chunk:
                        raise CampaignError("short read while recovering journal tail")
                    torn_hasher.update(chunk)
                    offset += len(chunk)
                tail_length = initial_size - complete_end
                complete_tail = False
                if tail_length <= 16 << 20:
                    try:
                        tail_event = json.loads(
                            os.pread(fd, tail_length, complete_end).decode("utf-8")
                        )
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        tail_event = None
                    complete_tail = bool(
                        isinstance(tail_event, Mapping)
                        and tail_event.get("schema") == CAMPAIGN_SCHEMA
                    )
                if complete_tail:
                    if os.write(fd, b"\n") != 1:
                        raise CampaignError("short newline repair in campaign journal")
                else:
                    os.ftruncate(fd, complete_end)
                    recovery = self._event(
                        "journal_tail_recovered",
                        "__campaign__",
                        message="discarded one incomplete final journal record",
                        metadata={
                            "discarded_bytes": tail_length,
                            "discarded_sha256": torn_hasher.hexdigest(),
                        },
                        artifacts=(),
                    )
                    body = (canonical_json(recovery) + "\n").encode("utf-8") + body
                recovered_tail = True
            before_stat = os.fstat(fd)
            before_signature = (
                before_stat.st_dev,
                before_stat.st_ino,
                before_stat.st_size,
                before_stat.st_mtime_ns,
            )
            offset = 0
            while offset < len(body):
                written = os.write(fd, body[offset:])
                if written <= 0:  # pragma: no cover - defensive short-write guard
                    raise CampaignError("short append to campaign journal")
                offset += written
            os.fsync(fd)
            after_stat = os.fstat(fd)
            after_signature = (
                after_stat.st_dev,
                after_stat.st_ino,
                after_stat.st_size,
                after_stat.st_mtime_ns,
            )
        finally:
            os.close(fd)
        if not journal_existed:
            # The journal fd is fsynced for every append.  Its parent needs an
            # additional fsync only when this call may have created the entry;
            # racing first writers may both flush the directory harmlessly.
            fsync_directory(self.root)
        cached_event = copy.deepcopy(dict(event))
        with self._events_cache_lock:
            if (
                not recovered_tail
                and self._events_cache is not None
                and self._events_cache_signature == before_signature
            ):
                self._events_cache = (*self._events_cache, cached_event)
                cached_cell_id = str(cached_event.get("cell_id"))
                self._events_by_cell_cache[cached_cell_id] = (
                    *self._events_by_cell_cache.get(cached_cell_id, ()),
                    cached_event,
                )
                self._events_cache_signature = after_signature
            else:
                self._events_cache = None
                self._events_by_cell_cache = {}
                self._events_cache_signature = None

    def _events_cached(
        self, cell_id: str | None = None
    ) -> tuple[dict[str, Any], ...]:
        if not self.journal_path.exists():
            return ()
        stat = self.journal_path.stat()
        signature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        with self._events_cache_lock:
            if (
                self._events_cache is not None
                and self._events_cache_signature == signature
            ):
                return (
                    self._events_cache
                    if cell_id is None
                    else self._events_by_cell_cache.get(cell_id, ())
                )
        result: list[dict[str, Any]] = []
        by_cell: dict[str, list[dict[str, Any]]] = {}
        with self.journal_path.open(encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    if not line.endswith("\n"):
                        break
                    raise CampaignError(
                        f"corrupt journal line {line_number}: {exc}"
                    ) from exc
                if event.get("schema") != CAMPAIGN_SCHEMA:
                    raise CampaignError(f"wrong journal schema on line {line_number}")
                result.append(event)
                by_cell.setdefault(str(event.get("cell_id")), []).append(event)
            locked_stat = os.fstat(handle.fileno())
            locked_signature = (
                locked_stat.st_dev,
                locked_stat.st_ino,
                locked_stat.st_size,
                locked_stat.st_mtime_ns,
            )
        frozen = tuple(result)
        frozen_by_cell = {key: tuple(events) for key, events in by_cell.items()}
        with self._events_cache_lock:
            self._events_cache = frozen
            self._events_by_cell_cache = frozen_by_cell
            self._events_cache_signature = locked_signature
        return frozen if cell_id is None else frozen_by_cell.get(cell_id, ())

    def events(self, cell_id: str | None = None) -> tuple[dict[str, Any], ...]:
        """Return detached journal events; callers cannot poison live caches."""

        return copy.deepcopy(self._events_cached(cell_id))

    def record(self, cell_id: str) -> CampaignRecord:
        self._require_cell(cell_id)
        state: RunState | None = None
        resume_state: RunState | None = None
        artifacts: dict[str, ArtifactRef] = {}
        transition_count = 0
        updated_at: float | None = None
        for event in self._events_cached(cell_id):
            if event.get("event") != "transition":
                continue
            previous_raw = event.get("previous")
            expected_previous = None if state is None else state.value
            if previous_raw != expected_previous:
                raise CampaignError(
                    f"journal transition chain broken for {cell_id}: "
                    f"expected previous {expected_previous!r}, got {previous_raw!r}"
                )
            target_raw = event.get("target")
            if target_raw is None:
                raise CampaignError(f"transition without target for {cell_id}")
            metadata = event.get("metadata") or {}
            if not isinstance(metadata, Mapping):
                raise CampaignError(
                    f"transition metadata is not an object for {cell_id}"
                )
            try:
                target = RunState(target_raw)
            except (TypeError, ValueError) as exc:
                raise CampaignError(
                    f"transition has invalid target {target_raw!r} for {cell_id}"
                ) from exc
            if state is None:
                if target is not RunState.PLANNED:
                    raise CampaignError(
                        f"first transition for {cell_id} must register planned state"
                    )
            elif state is RunState.FAILED:
                if (
                    resume_state is None
                    or target is not resume_state
                    or metadata.get("retry") is not True
                ):
                    raise CampaignError(
                        f"illegal retry transition failed -> {target.value} for {cell_id}"
                    )
            elif target not in LEGAL_TRANSITIONS[state]:
                raise CampaignError(
                    f"illegal journal transition {state.value} -> {target.value} "
                    f"for {cell_id}"
                )

            artifact_payloads = event.get("artifacts", ())
            if not isinstance(artifact_payloads, list):
                raise CampaignError(
                    f"transition artifacts are not a list for {cell_id}"
                )
            event_new_kinds: set[str] = set()
            for artifact_payload in artifact_payloads:
                if not isinstance(artifact_payload, Mapping):
                    raise CampaignError(
                        f"transition artifact is not an object for {cell_id}"
                    )
                try:
                    artifact = ArtifactRef.from_dict(artifact_payload)
                except (ArtifactError, KeyError, TypeError, ValueError) as exc:
                    raise CampaignError(
                        f"invalid transition artifact for {cell_id}: {exc}"
                    ) from exc
                existing = artifacts.get(artifact.kind)
                if existing is not None and existing != artifact:
                    raise CampaignError(
                        "append-only journal artifact replacement for "
                        f"{cell_id} kind {artifact.kind!r}"
                    )
                if existing is None:
                    event_new_kinds.add(artifact.kind)
                artifacts[artifact.kind] = artifact
            unexpected = event_new_kinds.difference(
                TRANSITION_ARTIFACT_KINDS.get(target, frozenset())
            )
            if unexpected:
                raise CampaignError(
                    f"journal transition to {target.value} introduces artifact kinds "
                    f"belonging to another stage for {cell_id}: {sorted(unexpected)}"
                )
            missing = REQUIRED_ARTIFACTS.get(target, frozenset()).difference(artifacts)
            if missing:
                raise CampaignError(
                    f"journal transition to {target.value} lacks required artifacts "
                    f"for {cell_id}: {sorted(missing)}"
                )

            if target is RunState.FAILED:
                resume_raw = metadata.get("resume_state")
                expected_resume = self._resume_target(state)
                try:
                    parsed_resume = RunState(resume_raw)
                except (TypeError, ValueError) as exc:
                    raise CampaignError(
                        f"failed transition lacks a valid resume state for {cell_id}"
                    ) from exc
                if parsed_resume is not expected_resume:
                    raise CampaignError(
                        f"failed transition has wrong resume state for {cell_id}"
                    )
                resume_state = parsed_resume
            else:
                resume_state = None
            state = target
            transition_count += 1
            try:
                updated_at = float(event["timestamp"])
            except (KeyError, TypeError, ValueError) as exc:
                raise CampaignError(
                    f"transition has invalid timestamp for {cell_id}"
                ) from exc
        if state is None:
            raise CampaignError(f"cell has no registration event: {cell_id}")
        return CampaignRecord(
            cell_id=cell_id,
            state=state,
            artifacts=tuple(artifacts[name] for name in sorted(artifacts)),
            resume_state=resume_state,
            event_count=transition_count,
            updated_at=updated_at,
        )

    def records(self) -> tuple[CampaignRecord, ...]:
        if not self._events_cached() and not self.plan_path.exists():
            return ()
        return tuple(self.record(cell.cell_id) for cell in self.plan.cells)

    def _write_snapshot(self, record: CampaignRecord) -> None:
        _atomic_json(self.state_path(record.cell_id), record.to_dict())

    def _ensure_snapshot(self, record: CampaignRecord) -> bool:
        """Repair one disposable state projection from the authoritative journal."""

        path = self.state_path(record.cell_id)
        try:
            current = _read_json(path)
        except (CampaignError, FileNotFoundError, OSError):
            current = None
        expected = record.to_dict()
        if current == expected:
            return False
        self._write_snapshot(record)
        return True

    @staticmethod
    def _resume_target(previous: RunState) -> RunState:
        # running means training had been claimed but not completed; the durable
        # predecessor is prepared.  Every later state is itself durable.
        if previous is RunState.RUNNING:
            return RunState.PREPARED
        return previous

    def transition(
        self,
        cell_id: str,
        target: RunState | str,
        *,
        artifacts: Iterable[ArtifactRef] = (),
        message: str = "",
        metadata: Mapping[str, Any] | None = None,
        assume_locked: bool = False,
    ) -> CampaignRecord:
        parsed_target = RunState(target)
        if assume_locked:
            return self._transition_locked(
                cell_id,
                parsed_target,
                artifacts=tuple(artifacts),
                message=message,
                metadata=metadata,
            )
        with self.lock(cell_id):
            return self._transition_locked(
                cell_id,
                parsed_target,
                artifacts=tuple(artifacts),
                message=message,
                metadata=metadata,
            )

    def _transition_locked(
        self,
        cell_id: str,
        target: RunState,
        *,
        artifacts: tuple[ArtifactRef, ...],
        message: str,
        metadata: Mapping[str, Any] | None,
    ) -> CampaignRecord:
        if target is RunState.PREPARED:
            durable_mkdir(self.root, parents=True, exist_ok=True)
            with self._campaign_mutation():
                with self.implementation_identity_lock_path.open(
                    "a+", encoding="utf-8"
                ) as lock_handle:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                    return self._transition_locked_commit_mutation(
                        cell_id,
                        target,
                        artifacts=artifacts,
                        message=message,
                        metadata=metadata,
                    )
        return self._transition_locked_commit(
            cell_id,
            target,
            artifacts=artifacts,
            message=message,
            metadata=metadata,
        )

    def _transition_locked_commit(
        self,
        cell_id: str,
        target: RunState,
        *,
        artifacts: tuple[ArtifactRef, ...],
        message: str,
        metadata: Mapping[str, Any] | None,
    ) -> CampaignRecord:
        with self._campaign_mutation():
            return self._transition_locked_commit_mutation(
                cell_id,
                target,
                artifacts=artifacts,
                message=message,
                metadata=metadata,
            )

    def _transition_locked_commit_mutation(
        self,
        cell_id: str,
        target: RunState,
        *,
        artifacts: tuple[ArtifactRef, ...],
        message: str,
        metadata: Mapping[str, Any] | None,
    ) -> CampaignRecord:
        cell = self._require_active_cell(cell_id)
        record = self.record(cell_id)
        if target not in LEGAL_TRANSITIONS[record.state]:
            raise InvalidTransition(
                f"illegal transition {record.state.value} -> {target.value}"
            )
        merged = record.artifact_map
        new_kinds: set[str] = set()
        authenticated: list[ArtifactRef] = []
        for unverified in artifacts:
            self._require_executor_stage_receipt(
                cell,
                target,
                unverified,
            )
            artifact = self._verify_artifact(unverified)
            authenticated.append(artifact)
            existing = merged.get(artifact.kind)
            if existing is not None and existing != artifact:
                raise ArtifactError(
                    "append-only campaign artifacts cannot replace an existing "
                    f"kind {artifact.kind!r}"
                )
            if existing is not None:
                continue
            new_kinds.add(artifact.kind)
            merged[artifact.kind] = artifact
        unexpected = new_kinds.difference(TRANSITION_ARTIFACT_KINDS[target])
        if unexpected:
            raise ArtifactError(
                f"transition to {target.value} emitted artifact kinds belonging "
                f"to another stage: {sorted(unexpected)}"
            )
        self._validate_artifact_gate(cell_id, target, merged)
        event_metadata = dict(metadata or {})
        if target is RunState.PREPARED:
            preparation = _read_json(merged["preparation"].resolve(self.root))
            identity = preparation.get("data_identity")
            if identity is not None:
                if "activation_content_identity" in event_metadata:
                    raise ArtifactError(
                        "caller cannot supply reserved activation identity metadata"
                    )
                event_metadata["activation_content_identity"] = {
                    "schema": ACTIVATION_CONTENT_IDENTITY_SCHEMA,
                    "raw_content_identity_sha256": identity[
                        "raw_content_identity_sha256"
                    ],
                    "view_key": identity["view_key"],
                    "view_content_identity_sha256": identity[
                        "view_content_identity_sha256"
                    ],
                }
        if target is RunState.FAILED:
            event_metadata["resume_state"] = self._resume_target(record.state).value
        event = self._event(
            "transition",
            cell_id,
            previous=record.state,
            target=target,
            message=message,
            metadata=event_metadata,
            artifacts=tuple(authenticated),
        )
        self._append_event(event)
        updated = self.record(cell_id)
        try:
            self._write_snapshot(updated)
            if (
                target is RunState.PREPARED
                and preparation.get("data_identity") is not None
            ):
                self._write_activation_identity_projection()
        except (CampaignError, OSError):
            # Journal append is the commit point; projections are repairable
            # and must never create a contradictory FAILED transition.
            pass
        return updated

    def _require_executor_stage_receipt(
        self,
        cell: CellSpec,
        target: RunState,
        artifact: ArtifactRef,
    ) -> None:
        stage = TARGET_STAGES.get(target)
        if stage not in {"train", "calibrate", "evaluate"}:
            return
        token = artifact._verification
        path = artifact.resolve(self.root).resolve()
        try:
            fingerprint = _ArtifactFingerprint.from_path(path)
        except OSError as exc:
            raise ArtifactError(f"cannot stat executor artifact {path}: {exc}") from exc
        key = self._artifact_cache_key(artifact, path)
        if (
            token is None
            or token.issuer is not self._artifact_verification_issuer
            or token.key != key
            or token.fingerprint != fingerprint
            or token.executor_cell_id != cell.cell_id
            or token.executor_stage != stage
        ):
            raise ArtifactError(
                f"transition to {target.value} requires an authenticated "
                f"executor-stage receipt for {artifact.kind}"
            )
        if (
            cell.decision_map.get("runtime.smoke") is False
            and token.canonical_executor is not True
        ):
            raise ArtifactError(
                "scientific stage admission requires the canonical executor"
            )

    def retry(self, cell_id: str, *, assume_locked: bool = False) -> CampaignRecord:
        if not assume_locked:
            with self.lock(cell_id):
                return self.retry(cell_id, assume_locked=True)
        with self._campaign_mutation():
            self._require_active_cell(cell_id)
            record = self.record(cell_id)
            if record.state is not RunState.FAILED or record.resume_state is None:
                raise InvalidTransition(
                    "only a failed cell with a durable resume point can retry"
                )
            event = self._event(
                "transition",
                cell_id,
                previous=RunState.FAILED,
                target=record.resume_state,
                message="explicit resume",
                metadata={"retry": True},
                artifacts=(),
            )
            self._append_event(event)
            updated = self.record(cell_id)
            try:
                self._write_snapshot(updated)
            except (CampaignError, OSError):
                pass
            return updated

    def _validate_artifact_gate(
        self,
        cell_id: str,
        target: RunState,
        artifacts: Mapping[str, ArtifactRef],
    ) -> None:
        required = REQUIRED_ARTIFACTS.get(target, frozenset())
        missing = required.difference(artifacts)
        if missing:
            raise ArtifactError(
                f"cannot enter {target.value}; missing content-addressed artifacts "
                f"{sorted(missing)}"
            )
        for kind in required:
            self._verify_artifact(artifacts[kind])
        if target in TARGET_STAGES:
            self._validate_stage_manifest(cell_id, target, artifacts)
        if target is RunState.PREPARED:
            self._validate_preparation_implementation(cell_id, artifacts)
        elif target is RunState.QUALIFIED:
            self._validate_qualification(cell_id, artifacts)
        elif target is RunState.PROMOTED:
            self._validate_promotion(cell_id, artifacts)

    def _validate_stage_manifest(
        self,
        cell_id: str,
        target: RunState,
        artifacts: Mapping[str, ArtifactRef],
    ) -> None:
        """Authenticate direct transitions as strictly as CampaignRunner ones."""

        stage = TARGET_STAGES[target]
        manifest_kind = f"{stage}_manifest"
        payload = _read_json(artifacts[manifest_kind].resolve(self.root))
        if (
            set(payload) != {"schema", "cell_id", "stage", "artifacts"}
            or payload.get("schema") != ARTIFACT_SCHEMA
            or payload.get("cell_id") != cell_id
            or payload.get("stage") != stage
            or not isinstance(payload.get("artifacts"), list)
        ):
            raise ArtifactError(f"{stage} stage manifest is noncanonical")
        entries: dict[str, Mapping[str, Any]] = {}
        for item in payload["artifacts"]:
            if (
                not isinstance(item, Mapping)
                or set(item) != {"kind", "path", "sha256", "size_bytes"}
                or not isinstance(item.get("kind"), str)
                or not isinstance(item.get("path"), str)
                or not isinstance(item.get("sha256"), str)
                or not isinstance(item.get("size_bytes"), int)
                or isinstance(item.get("size_bytes"), bool)
                or item["kind"] in entries
            ):
                raise ArtifactError(
                    f"{stage} stage manifest has a malformed artifact entry"
                )
            entries[item["kind"]] = item
        expected_kinds = EXPECTED_STAGE_ARTIFACTS[stage]
        if set(entries) != expected_kinds:
            raise ArtifactError(
                f"{stage} stage manifest artifact kinds must be exactly "
                f"{sorted(expected_kinds)}"
            )
        for kind in expected_kinds:
            ref = artifacts[kind]
            entry = entries[kind]
            reported_path = Path(entry["path"])
            if not reported_path.is_absolute():
                reported_path = self.root / reported_path
            if (
                reported_path.resolve() != ref.resolve(self.root).resolve()
                or entry["sha256"] != ref.sha256
                or entry["size_bytes"] != ref.size_bytes
            ):
                raise ArtifactError(
                    f"{stage} stage manifest differs from committed {kind} artifact"
                )

    def _validate_preparation_implementation(
        self,
        cell_id: str,
        artifacts: Mapping[str, ArtifactRef],
    ) -> None:
        """Audit the pre-registered implementation and activation identities."""

        payload = _read_json(artifacts["preparation"].resolve(self.root))
        implementation = payload.get("implementation")
        cell = self._require_cell(cell_id)
        cell_manifest_path = self.cell_manifest_path(cell_id)
        try:
            cell_manifest = _read_json(cell_manifest_path)
            parsed_cell = CellSpec.from_manifest(cell_manifest)
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactError(
                f"registered cell projection is invalid: {exc}"
            ) from exc
        if parsed_cell != cell or cell_manifest != cell.to_manifest():
            raise ArtifactError(
                "registered cell projection differs from the authoritative plan"
            )
        if (
            payload.get("schema") != PREPARATION_SCHEMA
            or payload.get("cell_id") != cell_id
            or not isinstance(implementation, Mapping)
        ):
            raise ArtifactError(
                "preparation artifact lacks its schema/cell/implementation binding"
            )
        observed_digest = _validate_implementation_identity(
            implementation,
            scientific=cell.decision_map["runtime.smoke"] is False,
        )
        if payload.get("implementation_sha256") != observed_digest:
            raise ArtifactError("preparation implementation digest is stale")
        events = self._events_cached()
        registration_index = next(
            (
                index
                for index, event in enumerate(events)
                if event.get("event") == "transition"
                and event.get("previous") is None
                and event.get("cell_id") == cell_id
            ),
            None,
        )
        if registration_index is None:
            raise ArtifactError("cell lacks its registration event")
        if not self.implementation_identity_path.exists():
            raise ArtifactError(
                "campaign lacks its registration-time implementation pin"
            )
        pinned = _read_json(self.implementation_identity_path)
        if (
            set(pinned)
            != {
                "schema",
                "implementation_identity",
                "implementation_identity_sha256",
            }
            or pinned.get("schema") != CAMPAIGN_IMPLEMENTATION_SCHEMA
        ):
            raise ArtifactError("campaign implementation pin is noncanonical")
        pinned_identity = pinned.get("implementation_identity")
        expected_pinned_digest = pinned.get("implementation_identity_sha256")
        gate = self._active_phase2_gate_amendment()
        if gate is not None:
            gate_index = next(
                index
                for index, event in enumerate(events)
                if event.get("event") == "design_amendment"
            )
            successor_pins: list[tuple[int, Mapping[str, Any], str]] = [
                (
                    gate_index,
                    gate[0]["implementation_identity"],
                    str(gate[0]["implementation_identity_sha256"]),
                )
            ]
            successor_pins.extend(
                (
                    event_index,
                    amendment_payload["implementation_identity"],
                    str(amendment_payload["implementation_identity_sha256"]),
                )
                for amendment_payload, _ref, event_index in (
                    self._phase2_implementation_amendments()
                )
            )
            applicable = [
                item for item in successor_pins if item[0] < registration_index
            ]
            if applicable:
                _index, pinned_identity, expected_pinned_digest = max(
                    applicable, key=lambda item: item[0]
                )
        if not isinstance(pinned_identity, Mapping):
            raise ArtifactError("campaign implementation pin lacks its identity")
        pinned_digest = _validate_implementation_identity(
            pinned_identity,
            scientific=False,
        )
        if (
            expected_pinned_digest != pinned_digest
            or pinned_digest != observed_digest
        ):
            raise ArtifactError(
                "preparation implementation identity differs from the campaign pin"
            )
        # Import lazily: the canonical executor imports Campaign for artifact
        # schemas, while Campaign must replay that same executor contract before
        # PREPARED becomes an authoritative journal event.
        from .cli.run_cell import (  # noqa: PLC0415
            CellExecutionError,
            _validate_preparation_contract,
        )

        try:
            _validate_preparation_contract(
                cell,
                payload,
                cell_manifest_sha256=_sha256(cell_manifest_path),
                verification_campaign_root=self.root,
            )
        except (CellExecutionError, OSError, ValueError) as exc:
            raise ArtifactError(
                f"preparation contract is invalid: {exc}"
            ) from exc
        self._validate_preparation_data_identity(cell, payload)

    def _validate_preparation_data_identity(
        self,
        cell: CellSpec,
        preparation: Mapping[str, Any],
    ) -> None:
        identity = preparation.get("data_identity")
        if cell.phase is Phase.PHASE1:
            if identity is not None:
                raise ArtifactError("synthetic preparation cannot bind activation data")
            return
        if (
            not isinstance(identity, Mapping)
            or set(identity)
            != {
                "schema",
                "raw_content_identity_sha256",
                "view_key",
                "view_content_identity_sha256",
                "raw_contract",
                "view_contract",
            }
            or identity.get("schema") != ACTIVATION_CONTENT_IDENTITY_SCHEMA
        ):
            raise ArtifactError(
                "real-data preparation has a noncanonical data identity"
            )
        data = preparation.get("data")
        if not isinstance(data, Mapping):
            raise ArtifactError(
                "real-data preparation lacks the data behind its content identity"
            )
        try:
            expected_identity = activation_content_identity(data)
        except ValueError as exc:
            raise ArtifactError(str(exc)) from exc
        if dict(identity) != expected_identity:
            raise ArtifactError(
                "real-data preparation data identity differs from its data payload"
            )
        raw_contract = identity.get("raw_contract")
        view_contract = identity.get("view_contract")
        raw_digest = identity.get("raw_content_identity_sha256")
        view_digest = identity.get("view_content_identity_sha256")
        view_key = identity.get("view_key")
        if (
            not isinstance(raw_contract, Mapping)
            or not isinstance(view_contract, Mapping)
            or not isinstance(view_key, str)
            or not view_key
            or not _is_sha256_hex(raw_digest)
            or not _is_sha256_hex(view_digest)
            or raw_digest != _sha256_canonical_payload(raw_contract)
            or view_digest != _sha256_canonical_payload(view_contract)
            or view_contract.get("raw_content_identity_sha256") != raw_digest
        ):
            raise ArtifactError("real-data preparation data identity is inconsistent")
        committed = self._committed_activation_identity_projection()
        if committed is None:
            return
        if committed["raw_content_identity_sha256"] != raw_digest:
            raise ArtifactError("campaign preparations use different raw captures")
        views = committed["views"]
        if view_key in views and views[view_key] != view_digest:
            raise ArtifactError(
                f"campaign preparations disagree on activation view {view_key!r}"
            )

    def _committed_activation_identity_projection(self) -> dict[str, Any] | None:
        with self._activation_projection_lock:
            return self._committed_activation_identity_projection_locked()

    def _committed_activation_identity_projection_locked(
        self,
    ) -> dict[str, Any] | None:
        events = self._events_cached()
        prefix_valid = (
            self._activation_projection_cursor == 0
            or (
                len(events) >= self._activation_projection_cursor
                and events[self._activation_projection_cursor - 1].get("event_id")
                == self._activation_projection_last_event_id
            )
        )
        if not prefix_valid:
            self._activation_projection_cursor = 0
            self._activation_projection_last_event_id = None
            self._activation_projection_raw_digest = None
            self._activation_projection_views = {}
        raw_digest = self._activation_projection_raw_digest
        views = dict(self._activation_projection_views)
        summary_keys = {
            "schema",
            "raw_content_identity_sha256",
            "view_key",
            "view_content_identity_sha256",
        }
        for event in events[self._activation_projection_cursor :]:
            if event.get("event") != "transition" or event.get("target") != "prepared":
                continue
            metadata = event.get("metadata")
            summary = (
                metadata.get("activation_content_identity")
                if isinstance(metadata, Mapping)
                else None
            )
            event_cell = self._require_cell(str(event.get("cell_id", "")))
            if summary is None:
                if event_cell.phase is not Phase.PHASE1:
                    raise ArtifactError(
                        "real-data prepared transition lacks activation identity metadata"
                    )
                continue
            if event_cell.phase is Phase.PHASE1:
                raise ArtifactError(
                    "synthetic prepared transition carries activation identity metadata"
                )
            if (
                not isinstance(summary, Mapping)
                or set(summary) != summary_keys
                or summary.get("schema") != ACTIVATION_CONTENT_IDENTITY_SCHEMA
                or not _is_sha256_hex(summary.get("raw_content_identity_sha256"))
                or not isinstance(summary.get("view_key"), str)
                or not summary["view_key"]
                or not _is_sha256_hex(summary.get("view_content_identity_sha256"))
            ):
                raise ArtifactError(
                    "journal contains malformed activation identity metadata"
                )
            candidate_raw = str(summary["raw_content_identity_sha256"])
            view_key = str(summary["view_key"])
            view_digest = str(summary["view_content_identity_sha256"])
            if raw_digest is None:
                raw_digest = candidate_raw
            elif raw_digest != candidate_raw:
                raise ArtifactError(
                    "journal commits preparations from different raw captures"
                )
            if view_key in views and views[view_key] != view_digest:
                raise ArtifactError(
                    f"journal commits conflicting activation view {view_key!r}"
                )
            views[view_key] = view_digest
        self._activation_projection_cursor = len(events)
        self._activation_projection_last_event_id = (
            None if not events else str(events[-1].get("event_id"))
        )
        self._activation_projection_raw_digest = raw_digest
        self._activation_projection_views = dict(views)
        if raw_digest is None:
            return None
        return {
            "schema": "bsc-campaign-activation-identity-v1",
            "raw_content_identity_sha256": raw_digest,
            "views": views,
        }

    def _write_activation_identity_projection(self) -> None:
        projection = self._committed_activation_identity_projection()
        if projection is None:
            return
        _atomic_json(self.activation_identity_path, projection)

    def _validate_qualification(
        self,
        cell_id: str,
        artifacts: Mapping[str, ArtifactRef],
    ) -> None:
        """Validate a qualification decision, not merely a metrics report.

        The qualification JSON must have schema ``bsc-qualification-v4``, the
        matching cell ID, ``qualified: true``, the complete all-true evidence
        integrity check set, a separately reported scientific outcome, an
        explicit boolean promotion-eligibility decision, and an ``inputs``
        mapping that exactly names the hashes of the preparation, checkpoint,
        calibration, deployable codec, schedules, and evaluation artifacts.
        The qualification also carries the exact preparation implementation
        identity.  A scientifically negative but complete control is admissible
        evidence and therefore may qualify; it cannot be selected or promoted.
        """

        cell = self._require_cell(cell_id)
        payload = _read_json(artifacts["qualification"].resolve(self.root))
        self._validate_preparation_implementation(cell_id, artifacts)
        preparation = _read_json(artifacts["preparation"].resolve(self.root))
        evaluation = _read_json(artifacts["evaluation"].resolve(self.root))
        _validate_qualification_payload(
            payload,
            cell=cell,
            expected_artifact_hashes={
                kind: artifacts[kind].sha256 for kind in QUALIFICATION_INPUT_KINDS
            },
            expected_implementation_identity=preparation["implementation"],
            evaluation=evaluation,
        )

    def _validate_promotion(
        self,
        cell_id: str,
        artifacts: Mapping[str, ArtifactRef],
    ) -> None:
        ref = artifacts["promotion"]
        payload = _read_json(ref.resolve(self.root))
        if payload.get("schema") != PROMOTION_SCHEMA:
            raise ArtifactError("promotion artifact has the wrong schema")
        if payload.get("cell_id") != cell_id or payload.get("approved") is not True:
            raise ArtifactError("promotion artifact does not approve this cell")
        if payload.get("qualification_sha256") != artifacts["qualification"].sha256:
            raise ArtifactError("promotion is not bound to the qualification artifact")
        qualification = _read_json(artifacts["qualification"].resolve(self.root))
        if qualification.get("promotion_eligible") is not True:
            raise ArtifactError("diagnostic-only qualification cannot be promoted")
        if qualification.get("scientific_outcome", {}).get("passed") is not True:
            raise ArtifactError("a failed scientific outcome cannot be promoted")
        cell = self._require_cell(cell_id)
        if cell.decision_map.get("runtime.smoke") is not False:
            raise ArtifactError("smoke cells cannot be promoted")

    def eligible_for_qualification(self, cell_id: str) -> bool:
        record = self.record(cell_id)
        if record.state is not RunState.EVALUATED:
            return False
        try:
            self._validate_artifact_gate(
                cell_id, RunState.EVALUATED, record.artifact_map
            )
        except ArtifactError:
            return False
        return True

    def eligible_for_promotion(self, cell_id: str) -> bool:
        record = self.record(cell_id)
        if record.state is not RunState.QUALIFIED:
            return False
        try:
            self._validate_artifact_gate(
                cell_id, RunState.QUALIFIED, record.artifact_map
            )
        except ArtifactError:
            return False
        qualification = self._qualification_payload(record)
        return bool(
            qualification is not None
            and qualification.get("promotion_eligible") is True
            and qualification.get("scientific_outcome", {}).get("passed") is True
        )

    def promote(self, cell_id: str, promotion_path: str | Path) -> CampaignRecord:
        artifact = ArtifactRef.from_path(
            "promotion", Path(promotion_path), root=self.root
        )
        return self.transition(
            cell_id,
            RunState.PROMOTED,
            artifacts=(artifact,),
            message="explicit promotion decision",
        )

    def _qualification_payload(self, record: CampaignRecord) -> dict[str, Any] | None:
        ref = record.artifact_map.get("qualification")
        if ref is None:
            return None
        ref.verify(self.root)
        return _read_json(ref.resolve(self.root))

    @staticmethod
    def _finite_metric(payload: Mapping[str, Any], key: str, *, context: str) -> float:
        value = payload.get(key)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math_isfinite(value)
        ):
            raise ArtifactError(f"{context} lacks finite sharing-guard metric {key!r}")
        return float(value)

    @classmethod
    def _sharing_metrics(
        cls, selection_metrics: Mapping[str, Any], *, context: str
    ) -> dict[str, float]:
        sharing = selection_metrics.get("sharing_guard")
        if not isinstance(sharing, Mapping):
            raise ArtifactError(f"{context} lacks authenticated sharing-guard metrics")
        return {
            key: cls._finite_metric(sharing, key, context=context)
            for key in (
                "all_site_fvu_mean",
                "site_only_heldout_fvu_mean",
                "leave_one_out_heldout_fvu_mean",
                "site_only_support_iou_mean",
                "leave_one_out_support_iou_mean",
                "site_only_coordinate_concordance_mean",
                "leave_one_out_coordinate_concordance_mean",
                "site_only_coordinate_concordance_min",
                "leave_one_out_coordinate_concordance_min",
                "site_only_intersection_recall_mean",
                "leave_one_out_intersection_recall_mean",
                "site_only_intersection_recall_min",
                "leave_one_out_intersection_recall_min",
                "site_only_intersection_energy_coverage_mean",
                "leave_one_out_intersection_energy_coverage_mean",
                "site_only_intersection_energy_coverage_min",
                "leave_one_out_intersection_energy_coverage_min",
            )
        }

    def _sharing_lineage_metrics(
        self,
        cell: CellSpec,
        *,
        selection_metrics: Mapping[str, Any],
    ) -> tuple[
        str | None,
        dict[str, float],
        str,
        dict[str, float],
        list[dict[str, Any]],
    ]:
        """Return authenticated immediate-parent and root sharing evidence.

        Every hop follows the same-seed member of the exact frozen parent-cell
        tuple.  Qualification hashes and metrics for the complete path are
        emitted into the selection artifact, so cumulative drift cannot hide
        behind individually acceptable one-round changes.
        """

        current = self._sharing_metrics(
            selection_metrics, context=f"cell {cell.cell_id}"
        )
        plan_cells = {item.cell_id: item for item in self.plan.cells}
        seen: set[str] = set()
        trace: list[dict[str, Any]] = []
        current_cell = cell
        current_metrics = current
        parent_cell_id: str | None = None
        parent_metrics = current
        while True:
            if current_cell.cell_id in seen:
                raise ArtifactError(
                    f"sharing lineage for {cell.cell_id} contains a cycle"
                )
            seen.add(current_cell.cell_id)
            record = self.record(current_cell.cell_id)
            if record.state not in {RunState.QUALIFIED, RunState.PROMOTED}:
                raise ArtifactError(
                    f"sharing-lineage cell {current_cell.cell_id} is not qualified"
                )
            self._validate_artifact_gate(
                current_cell.cell_id, RunState.QUALIFIED, record.artifact_map
            )
            trace.append(
                {
                    "cell_id": current_cell.cell_id,
                    "seed": current_cell.seed,
                    "qualification_sha256": (
                        "sha256:" + record.artifact_map["qualification"].sha256
                    ),
                    "metrics": current_metrics,
                }
            )
            parent_ids = current_cell.decision_map.get("selection.parent_cell_ids", ())
            if not isinstance(parent_ids, (tuple, list)):
                raise ArtifactError(
                    f"cell {current_cell.cell_id} has malformed selected-parent binding"
                )
            if not parent_ids:
                return (
                    parent_cell_id,
                    parent_metrics,
                    current_cell.cell_id,
                    current_metrics,
                    trace,
                )
            try:
                parents = [plan_cells[str(parent_id)] for parent_id in parent_ids]
            except KeyError as exc:
                raise ArtifactError(
                    f"cell {current_cell.cell_id} names a parent outside the active plan"
                ) from exc
            matching = [parent for parent in parents if parent.seed == cell.seed]
            if len(matching) != 1:
                raise ArtifactError(
                    f"cell {current_cell.cell_id} does not bind exactly one same-seed parent"
                )
            next_cell = matching[0]
            next_record = self.record(next_cell.cell_id)
            if next_record.state not in {RunState.QUALIFIED, RunState.PROMOTED}:
                raise ArtifactError(
                    f"selected parent {next_cell.cell_id} is not qualified"
                )
            self._validate_artifact_gate(
                next_cell.cell_id, RunState.QUALIFIED, next_record.artifact_map
            )
            payload = self._qualification_payload(next_record)
            if payload is None:  # pragma: no cover - artifact gate guarantees it
                raise ArtifactError(
                    f"selected parent {next_cell.cell_id} has no qualification"
                )
            parent_selection_metrics = payload.get("selection_metrics")
            if not isinstance(parent_selection_metrics, Mapping):
                raise ArtifactError(
                    f"selected parent {next_cell.cell_id} lacks authenticated selection metrics"
                )
            next_metrics = self._sharing_metrics(
                parent_selection_metrics,
                context=f"selected parent {next_cell.cell_id}",
            )
            if parent_cell_id is None:
                parent_cell_id = next_cell.cell_id
                parent_metrics = next_metrics
            current_cell = next_cell
            current_metrics = next_metrics

    @staticmethod
    def _sharing_guard_payload(
        current: Mapping[str, float],
        parent: Mapping[str, float],
        root: Mapping[str, float],
        policy: SelectionPolicy,
    ) -> dict[str, Any]:
        """Recompute the scientific guard from authenticated endpoint metrics."""

        thresholds = {
            "site_only_fvu_degradation_max": float(
                policy.sharing_site_only_fvu_degradation_max
            ),
            "leave_one_out_fvu_degradation_max": float(
                policy.sharing_leave_one_out_fvu_degradation_max
            ),
            "support_iou_drop_max": float(policy.sharing_support_iou_drop_max),
            "coordinate_concordance_min": float(
                policy.sharing_coordinate_concordance_min
            ),
            "intersection_recall_min": float(policy.sharing_intersection_recall_min),
            "intersection_energy_coverage_min": float(
                policy.sharing_intersection_energy_coverage_min
            ),
            "fvu_absolute_max": float(policy.sharing_fvu_absolute_max),
            "root_site_only_fvu_degradation_max": float(
                policy.sharing_root_site_only_fvu_degradation_max
            ),
            "root_leave_one_out_fvu_degradation_max": float(
                policy.sharing_root_leave_one_out_fvu_degradation_max
            ),
        }
        measurements = {
            "site_only_fvu_degradation": (
                current["site_only_heldout_fvu_mean"]
                - parent["site_only_heldout_fvu_mean"]
            ),
            "leave_one_out_fvu_degradation": (
                current["leave_one_out_heldout_fvu_mean"]
                - parent["leave_one_out_heldout_fvu_mean"]
            ),
            "site_only_support_iou_drop": (
                parent["site_only_support_iou_mean"]
                - current["site_only_support_iou_mean"]
            ),
            "leave_one_out_support_iou_drop": (
                parent["leave_one_out_support_iou_mean"]
                - current["leave_one_out_support_iou_mean"]
            ),
            "all_view_fvu_advantage_descriptive": (
                current["site_only_heldout_fvu_mean"] - current["all_site_fvu_mean"]
            ),
            "site_only_coordinate_concordance": current[
                "site_only_coordinate_concordance_min"
            ],
            "leave_one_out_coordinate_concordance": current[
                "leave_one_out_coordinate_concordance_min"
            ],
            "site_only_intersection_recall": current[
                "site_only_intersection_recall_min"
            ],
            "leave_one_out_intersection_recall": current[
                "leave_one_out_intersection_recall_min"
            ],
            "site_only_intersection_energy_coverage": current[
                "site_only_intersection_energy_coverage_min"
            ],
            "leave_one_out_intersection_energy_coverage": current[
                "leave_one_out_intersection_energy_coverage_min"
            ],
            "root_site_only_fvu_degradation": (
                current["site_only_heldout_fvu_mean"]
                - root["site_only_heldout_fvu_mean"]
            ),
            "root_leave_one_out_fvu_degradation": (
                current["leave_one_out_heldout_fvu_mean"]
                - root["leave_one_out_heldout_fvu_mean"]
            ),
            "site_only_fvu_absolute": current["site_only_heldout_fvu_mean"],
            "leave_one_out_fvu_absolute": current["leave_one_out_heldout_fvu_mean"],
        }
        checks = {
            "site_only_fvu_degradation": (
                measurements["site_only_fvu_degradation"]
                <= thresholds["site_only_fvu_degradation_max"]
            ),
            "leave_one_out_fvu_degradation": (
                measurements["leave_one_out_fvu_degradation"]
                <= thresholds["leave_one_out_fvu_degradation_max"]
            ),
            "site_only_support_iou_drop": (
                measurements["site_only_support_iou_drop"]
                <= thresholds["support_iou_drop_max"]
            ),
            "leave_one_out_support_iou_drop": (
                measurements["leave_one_out_support_iou_drop"]
                <= thresholds["support_iou_drop_max"]
            ),
            "site_only_coordinate_concordance": (
                measurements["site_only_coordinate_concordance"]
                >= thresholds["coordinate_concordance_min"]
            ),
            "leave_one_out_coordinate_concordance": (
                measurements["leave_one_out_coordinate_concordance"]
                >= thresholds["coordinate_concordance_min"]
            ),
            "site_only_intersection_recall": (
                measurements["site_only_intersection_recall"]
                >= thresholds["intersection_recall_min"]
            ),
            "leave_one_out_intersection_recall": (
                measurements["leave_one_out_intersection_recall"]
                >= thresholds["intersection_recall_min"]
            ),
            "site_only_intersection_energy_coverage": (
                measurements["site_only_intersection_energy_coverage"]
                >= thresholds["intersection_energy_coverage_min"]
            ),
            "leave_one_out_intersection_energy_coverage": (
                measurements["leave_one_out_intersection_energy_coverage"]
                >= thresholds["intersection_energy_coverage_min"]
            ),
            "root_site_only_fvu_degradation": (
                measurements["root_site_only_fvu_degradation"]
                <= thresholds["root_site_only_fvu_degradation_max"]
            ),
            "root_leave_one_out_fvu_degradation": (
                measurements["root_leave_one_out_fvu_degradation"]
                <= thresholds["root_leave_one_out_fvu_degradation_max"]
            ),
            "site_only_fvu_absolute": (
                measurements["site_only_fvu_absolute"] <= thresholds["fvu_absolute_max"]
            ),
            "leave_one_out_fvu_absolute": (
                measurements["leave_one_out_fvu_absolute"]
                <= thresholds["fvu_absolute_max"]
            ),
        }
        return {
            "thresholds": thresholds,
            "measurements": measurements,
            "checks": checks,
            "passed": all(checks.values()),
        }

    @staticmethod
    def _validate_recomputed_sharing_guard(
        guard: Mapping[str, Any],
        current: Mapping[str, float],
        parent: Mapping[str, float],
        root: Mapping[str, float],
        policy: SelectionPolicy,
    ) -> dict[str, Any]:
        """Reject a mutable guard summary not implied by authenticated metrics."""

        expected = Campaign._sharing_guard_payload(current, parent, root, policy)
        if (
            guard.get("thresholds") != expected["thresholds"]
            or guard.get("measurements") != expected["measurements"]
            or guard.get("checks") != expected["checks"]
            or guard.get("passed") is not expected["passed"]
        ):
            raise CampaignError(
                "confirmation sharing guard was not recomputed from authenticated metrics"
            )
        return expected

    def _sharing_guard_result(
        self,
        cell: CellSpec,
        selection_metrics: Mapping[str, Any],
        policy: SelectionPolicy,
    ) -> dict[str, Any]:
        current = self._sharing_metrics(
            selection_metrics, context=f"cell {cell.cell_id}"
        )
        (
            parent_cell_id,
            parent,
            root_cell_id,
            root,
            lineage_trace,
        ) = self._sharing_lineage_metrics(cell, selection_metrics=selection_metrics)
        return {
            "cell_id": cell.cell_id,
            "seed": cell.seed,
            "parent_cell_id": parent_cell_id,
            "root_cell_id": root_cell_id,
            "authenticated_lineage": lineage_trace,
            **self._sharing_guard_payload(current, parent, root, policy),
        }

    @staticmethod
    def _candidate_for_variant(
        stage_name: str,
        candidates: Sequence[Mapping[str, Any]],
        variant_name: str,
    ) -> Mapping[str, Any] | None:
        expected_recipe = f"derived_{stage_name}_{variant_name}"
        matches = [
            candidate
            for candidate in candidates
            if candidate.get("recipe_name") == expected_recipe
        ]
        if len(matches) > 1:
            raise CampaignError(
                f"stage {stage_name!r} repeats noninferiority variant {variant_name!r}"
            )
        return None if not matches else matches[0]

    @staticmethod
    def _apply_noninferiority_gate(
        stage_name: str,
        policy: SelectionPolicy,
        candidates: list[dict[str, Any]],
        excluded: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Gate a changed carrier against its exact selected-parent rerun.

        The named carrier is compared seed-for-seed with the required control.
        If it fails, all candidates on the changed free/factorized carrier are
        ineligible and only the exact selected-parent control may advance.  The
        gate is recorded in the ranked universe, so any frozen selection binds
        its full per-seed evidence rather than merely the winning score.
        """

        control_variant = policy.required_control_variant
        carrier_variant = policy.noninferiority_candidate_variant
        if control_variant is None:
            return candidates
        if carrier_variant is None:  # pragma: no cover - SelectionPolicy rejects it
            raise CampaignError("noninferiority policy omits its carrier variant")
        tolerance = float(policy.control_noninferiority_absolute_tolerance)
        control = Campaign._candidate_for_variant(
            stage_name, candidates, control_variant
        )
        if control is None:
            raise CampaignError(
                f"stage {stage_name!r} lacks eligible required control variant "
                f"{control_variant!r}"
            )
        carrier = Campaign._candidate_for_variant(
            stage_name, candidates, carrier_variant
        )
        if carrier is None:
            gate = {
                "required_control_variant": control_variant,
                "candidate_variant": carrier_variant,
                "absolute_tolerance": tolerance,
                "direction": policy.direction,
                "per_seed": [],
                "passed": False,
                "reason": "carrier_ineligible_before_noninferiority",
            }
        else:
            control_by_seed = {
                int(item["seed"]): float(item["metric"])
                for item in control["observations"]
            }
            carrier_by_seed = {
                int(item["seed"]): float(item["metric"])
                for item in carrier["observations"]
            }
            if set(control_by_seed) != set(carrier_by_seed):
                raise CampaignError(
                    "noninferiority control and carrier do not cover identical seeds"
                )
            comparisons: list[dict[str, Any]] = []
            for seed in sorted(control_by_seed):
                control_metric = control_by_seed[seed]
                carrier_metric = carrier_by_seed[seed]
                degradation = (
                    carrier_metric - control_metric
                    if policy.direction == "min"
                    else control_metric - carrier_metric
                )
                comparisons.append(
                    {
                        "seed": seed,
                        "control_metric": control_metric,
                        "carrier_metric": carrier_metric,
                        "degradation": degradation,
                        "passed": Campaign._meets_upper_bound(degradation, tolerance),
                    }
                )
            gate = {
                "required_control_variant": control_variant,
                "candidate_variant": carrier_variant,
                "absolute_tolerance": tolerance,
                "direction": policy.direction,
                "per_seed": comparisons,
                "passed": all(item["passed"] for item in comparisons),
            }
        for candidate in candidates:
            candidate["noninferiority_gate"] = gate
        if gate["passed"] is True:
            return candidates

        retained: list[dict[str, Any]] = []
        for candidate in candidates:
            if candidate["candidate_id"] == control["candidate_id"]:
                retained.append(candidate)
                continue
            excluded.append(
                {
                    **candidate,
                    "reason": "required_carrier_noninferiority_failed",
                }
            )
        return retained

    @staticmethod
    def _directional_improvement(
        candidate_metric: float,
        reference_metric: float,
        direction: str,
    ) -> float:
        return (
            candidate_metric - reference_metric
            if direction == "max"
            else reference_metric - candidate_metric
        )

    @staticmethod
    def _meets_lower_bound(value: float, bound: float) -> bool:
        return value >= bound or math.isclose(
            value,
            bound,
            rel_tol=0.0,
            abs_tol=1e-12,
        )

    @staticmethod
    def _meets_upper_bound(value: float, bound: float) -> bool:
        return value <= bound or math.isclose(
            value,
            bound,
            rel_tol=0.0,
            abs_tol=1e-12,
        )

    @staticmethod
    def _apply_minimum_effect_gate(
        stage_name: str,
        policy: SelectionPolicy,
        candidates: list[dict[str, Any]],
        excluded: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Retain the exact parent unless a challenger clears a real effect.

        Candidate and parent metrics are paired by seed.  A challenger must
        improve on every seed *and* on the policy's median and worst-seed
        aggregates by the frozen raw-FVU amount.  The apparently redundant
        aggregate checks are intentional: they make the declared aggregation
        contract explicit in the immutable selection evidence.
        """

        parent_variant = policy.default_parent_variant
        if parent_variant is None:
            return candidates
        threshold = float(policy.minimum_effect_absolute)
        parent = Campaign._candidate_for_variant(stage_name, candidates, parent_variant)
        if parent is None:
            if (
                Campaign._candidate_for_variant(stage_name, excluded, parent_variant)
                is not None
            ):
                # The exact parent existed but failed an earlier scientific or
                # integrity gate.  Preserve the generic empty-universe refusal
                # below; a missing blueprint arm remains a distinct hard error.
                return []
            raise CampaignError(
                f"stage {stage_name!r} lacks eligible default-parent variant "
                f"{parent_variant!r}"
            )
        parent_by_seed = {
            int(item["seed"]): float(item["metric"]) for item in parent["observations"]
        }
        parent_gate = {
            "default_parent_variant": parent_variant,
            "minimum_effect_absolute": threshold,
            "reduction": policy.minimum_effect_reduction,
            "direction": policy.direction,
            "role": "default_parent",
            "per_seed": [],
            "median_improvement": 0.0,
            "worst_seed_improvement": 0.0,
            "passed": True,
        }
        parent["minimum_effect_gate"] = parent_gate
        retained = [parent]
        for candidate in candidates:
            if candidate["candidate_id"] == parent["candidate_id"]:
                continue
            candidate_by_seed = {
                int(item["seed"]): float(item["metric"])
                for item in candidate["observations"]
            }
            if set(candidate_by_seed) != set(parent_by_seed):
                raise CampaignError(
                    "minimum-effect parent and challenger do not cover identical seeds"
                )
            comparisons: list[dict[str, Any]] = []
            for seed in sorted(parent_by_seed):
                improvement = Campaign._directional_improvement(
                    candidate_by_seed[seed],
                    parent_by_seed[seed],
                    policy.direction,
                )
                comparisons.append(
                    {
                        "seed": seed,
                        "parent_metric": parent_by_seed[seed],
                        "candidate_metric": candidate_by_seed[seed],
                        "improvement": improvement,
                        "passed": Campaign._meets_lower_bound(improvement, threshold),
                    }
                )
            median_improvement = Campaign._directional_improvement(
                float(candidate["median"]),
                float(parent["median"]),
                policy.direction,
            )
            worst_improvement = Campaign._directional_improvement(
                float(candidate["worst_seed"]),
                float(parent["worst_seed"]),
                policy.direction,
            )
            gate = {
                "default_parent_variant": parent_variant,
                "minimum_effect_absolute": threshold,
                "reduction": policy.minimum_effect_reduction,
                "direction": policy.direction,
                "role": "challenger",
                "per_seed": comparisons,
                "median_improvement": median_improvement,
                "median_passed": Campaign._meets_lower_bound(
                    median_improvement, threshold
                ),
                "worst_seed_improvement": worst_improvement,
                "worst_seed_passed": Campaign._meets_lower_bound(
                    worst_improvement, threshold
                ),
            }
            gate["passed"] = (
                all(item["passed"] for item in comparisons)
                and gate["median_passed"] is True
                and gate["worst_seed_passed"] is True
            )
            candidate["minimum_effect_gate"] = gate
            if gate["passed"] is True:
                retained.append(candidate)
            else:
                excluded.append(
                    {
                        **candidate,
                        "reason": "minimum_effect_not_met_against_default_parent",
                    }
                )
        return retained

    @staticmethod
    def _apply_rank_parsimony_gate(
        stage_name: str,
        policy: SelectionPolicy,
        candidates: list[dict[str, Any]],
        excluded: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Choose the lowest declared site rank noninferior to the full tensor."""

        order = policy.parsimony_order_variants
        if not order:
            return candidates
        control_variant = policy.required_control_variant
        reference_variant = policy.noninferiority_candidate_variant
        if control_variant is None or reference_variant is None:
            raise CampaignError("rank parsimony lacks its carrier/reference contract")
        carrier_gate = next(
            (
                candidate.get("noninferiority_gate")
                for candidate in candidates
                if isinstance(candidate.get("noninferiority_gate"), Mapping)
            ),
            None,
        )
        if carrier_gate is None:
            raise CampaignError("rank parsimony lacks carrier noninferiority evidence")
        if carrier_gate.get("passed") is not True:
            blocked = {
                "order_variants": list(order),
                "reference_variant": reference_variant,
                "absolute_tolerance": float(
                    policy.parsimony_noninferiority_absolute_tolerance
                ),
                "reduction": policy.parsimony_reduction,
                "direction": policy.direction,
                "enforced": False,
                "reason": "carrier_noninferiority_failed",
                "selected_variant": control_variant,
            }
            for candidate in candidates:
                candidate["parsimony_gate"] = blocked
            return candidates

        expected_variants = {*order, control_variant}
        for candidate in candidates:
            recipe_name = str(candidate["recipe_name"])
            prefix = f"derived_{stage_name}_"
            if not recipe_name.startswith(prefix):
                raise CampaignError("rank-parsimony candidate has an unexpected recipe")
            if recipe_name[len(prefix) :] not in expected_variants:
                raise CampaignError(
                    "rank-parsimony stage contains an undeclared variant"
                )

        reference = Campaign._candidate_for_variant(
            stage_name, candidates, reference_variant
        )
        if reference is None:
            raise CampaignError(
                "carrier passed noninferiority but its full-rank reference is ineligible"
            )
        control = Campaign._candidate_for_variant(
            stage_name, candidates, control_variant
        )
        if control is None:
            raise CampaignError(
                "carrier passed noninferiority without its exact-parent control"
            )
        reference_by_seed = {
            int(item["seed"]): float(item["metric"])
            for item in reference["observations"]
        }
        tolerance = float(policy.parsimony_noninferiority_absolute_tolerance)
        comparisons: list[dict[str, Any]] = []
        selected: dict[str, Any] | None = None
        selected_variant: str | None = None
        for variant_name in order:
            candidate = Campaign._candidate_for_variant(
                stage_name, candidates, variant_name
            )
            if candidate is None:
                comparisons.append(
                    {
                        "variant": variant_name,
                        "eligible": False,
                        "passed": False,
                        "reason": "ineligible_before_parsimony",
                    }
                )
                continue
            candidate_by_seed = {
                int(item["seed"]): float(item["metric"])
                for item in candidate["observations"]
            }
            if set(candidate_by_seed) != set(reference_by_seed):
                raise CampaignError(
                    "parsimony candidate and full reference do not cover identical seeds"
                )
            seed_rows: list[dict[str, Any]] = []
            for seed in sorted(reference_by_seed):
                degradation = -Campaign._directional_improvement(
                    candidate_by_seed[seed],
                    reference_by_seed[seed],
                    policy.direction,
                )
                seed_rows.append(
                    {
                        "seed": seed,
                        "reference_metric": reference_by_seed[seed],
                        "candidate_metric": candidate_by_seed[seed],
                        "degradation": degradation,
                        "passed": Campaign._meets_upper_bound(degradation, tolerance),
                    }
                )
            median_degradation = -Campaign._directional_improvement(
                float(candidate["median"]),
                float(reference["median"]),
                policy.direction,
            )
            worst_degradation = -Campaign._directional_improvement(
                float(candidate["worst_seed"]),
                float(reference["worst_seed"]),
                policy.direction,
            )
            comparison = {
                "variant": variant_name,
                "eligible": True,
                "per_seed": seed_rows,
                "median_degradation": median_degradation,
                "median_passed": Campaign._meets_upper_bound(
                    median_degradation, tolerance
                ),
                "worst_seed_degradation": worst_degradation,
                "worst_seed_passed": Campaign._meets_upper_bound(
                    worst_degradation, tolerance
                ),
            }
            comparison["passed"] = (
                all(item["passed"] for item in seed_rows)
                and comparison["median_passed"] is True
                and comparison["worst_seed_passed"] is True
            )
            comparisons.append(comparison)
            if selected is None and comparison["passed"] is True:
                selected = candidate
                selected_variant = variant_name
        if selected is None or selected_variant is None:
            raise CampaignError(
                "full-rank reference failed its own parsimony noninferiority check"
            )
        gate = {
            "order_variants": list(order),
            "reference_variant": reference_variant,
            "absolute_tolerance": tolerance,
            "reduction": policy.parsimony_reduction,
            "direction": policy.direction,
            "enforced": True,
            "comparisons": comparisons,
            "selected_variant": selected_variant,
            "passed": True,
        }
        for candidate in candidates:
            candidate["parsimony_gate"] = gate
        for candidate in candidates:
            if candidate["candidate_id"] == selected["candidate_id"]:
                continue
            excluded.append(
                {
                    **candidate,
                    "reason": (
                        "carrier_control_only_after_noninferiority_pass"
                        if candidate["candidate_id"] == control["candidate_id"]
                        else "site_rank_parsimony_not_selected"
                    ),
                }
            )
        return [selected]

    @staticmethod
    def _threshold_sensitivity_payload(
        policy: SelectionPolicy,
        population: Sequence[Mapping[str, Any]],
        *,
        smoke_protocol_only: bool,
    ) -> dict[str, Any]:
        """Evaluate each preregistered threshold marginally without retuning.

        The center policy remains authoritative.  This payload exposes how the
        same authenticated measurements would pass or fail at every declared
        threshold, so the sensitivity grid is an executed analysis rather than
        inert manifest prose.
        """

        if not policy.threshold_sensitivity:
            return {
                "applicable": False,
                "basis": policy.threshold_basis,
                "reason": "selection_policy_has_no_winner_changing_thresholds",
                "surfaces": [],
            }
        if smoke_protocol_only:
            return {
                "applicable": False,
                "basis": policy.threshold_basis,
                "reason": "smoke_protocol_does_not_evaluate_scientific_thresholds",
                "surfaces": [
                    {"name": name, "thresholds": list(values)}
                    for name, values in policy.threshold_sensitivity
                ],
            }

        candidates = {
            str(candidate["candidate_id"]): candidate
            for candidate in population
            if isinstance(candidate.get("candidate_id"), str)
        }
        center_values: dict[str, float | None] = {
            "minimum_effect_absolute": policy.minimum_effect_absolute,
            "noninferiority_absolute_tolerance": (
                policy.control_noninferiority_absolute_tolerance
                if policy.control_noninferiority_absolute_tolerance is not None
                else policy.parsimony_noninferiority_absolute_tolerance
            ),
            "sharing_fvu_degradation_max": (
                policy.sharing_site_only_fvu_degradation_max
            ),
            "sharing_support_iou_drop_max": policy.sharing_support_iou_drop_max,
            "sharing_coordinate_concordance_min": (
                policy.sharing_coordinate_concordance_min
            ),
            "sharing_intersection_recall_min": (policy.sharing_intersection_recall_min),
            "sharing_intersection_energy_coverage_min": (
                policy.sharing_intersection_energy_coverage_min
            ),
            "sharing_fvu_absolute_max": policy.sharing_fvu_absolute_max,
        }
        sharing_contract: dict[
            str, tuple[str, tuple[tuple[str, float | None], ...]]
        ] = {
            "sharing_fvu_degradation_max": (
                "upper",
                (
                    (
                        "site_only_fvu_degradation",
                        policy.sharing_site_only_fvu_degradation_max,
                    ),
                    (
                        "leave_one_out_fvu_degradation",
                        policy.sharing_leave_one_out_fvu_degradation_max,
                    ),
                    (
                        "root_site_only_fvu_degradation",
                        policy.sharing_root_site_only_fvu_degradation_max,
                    ),
                    (
                        "root_leave_one_out_fvu_degradation",
                        policy.sharing_root_leave_one_out_fvu_degradation_max,
                    ),
                ),
            ),
            "sharing_support_iou_drop_max": (
                "upper",
                (
                    (
                        "site_only_support_iou_drop",
                        policy.sharing_support_iou_drop_max,
                    ),
                    (
                        "leave_one_out_support_iou_drop",
                        policy.sharing_support_iou_drop_max,
                    ),
                ),
            ),
            "sharing_coordinate_concordance_min": (
                "lower",
                (
                    (
                        "site_only_coordinate_concordance",
                        policy.sharing_coordinate_concordance_min,
                    ),
                    (
                        "leave_one_out_coordinate_concordance",
                        policy.sharing_coordinate_concordance_min,
                    ),
                ),
            ),
            "sharing_intersection_recall_min": (
                "lower",
                (
                    (
                        "site_only_intersection_recall",
                        policy.sharing_intersection_recall_min,
                    ),
                    (
                        "leave_one_out_intersection_recall",
                        policy.sharing_intersection_recall_min,
                    ),
                ),
            ),
            "sharing_intersection_energy_coverage_min": (
                "lower",
                (
                    (
                        "site_only_intersection_energy_coverage",
                        policy.sharing_intersection_energy_coverage_min,
                    ),
                    (
                        "leave_one_out_intersection_energy_coverage",
                        policy.sharing_intersection_energy_coverage_min,
                    ),
                ),
            ),
            "sharing_fvu_absolute_max": (
                "upper",
                (
                    ("site_only_fvu_absolute", policy.sharing_fvu_absolute_max),
                    ("leave_one_out_fvu_absolute", policy.sharing_fvu_absolute_max),
                ),
            ),
        }
        surfaces: list[dict[str, Any]] = []
        for name, thresholds in policy.threshold_sensitivity:
            center = center_values.get(name)
            if name == "minimum_effect_absolute":
                challenger_gates = {
                    candidate_id: candidate.get("minimum_effect_gate")
                    for candidate_id, candidate in candidates.items()
                    if isinstance(candidate.get("minimum_effect_gate"), Mapping)
                    and candidate["minimum_effect_gate"].get("role") == "challenger"
                }
                rows = []
                for threshold in thresholds:
                    passed = []
                    for candidate_id, gate in challenger_gates.items():
                        assert isinstance(gate, Mapping)
                        per_seed = gate.get("per_seed", ())
                        if (
                            all(
                                Campaign._meets_lower_bound(
                                    float(item["improvement"]), threshold
                                )
                                for item in per_seed
                            )
                            and Campaign._meets_lower_bound(
                                float(gate["median_improvement"]), threshold
                            )
                            and Campaign._meets_lower_bound(
                                float(gate["worst_seed_improvement"]), threshold
                            )
                        ):
                            passed.append(candidate_id)
                    rows.append(
                        {
                            "threshold": threshold,
                            "passing_challenger_candidate_ids": sorted(passed),
                        }
                    )
                surfaces.append(
                    {
                        "name": name,
                        "applicable": center is not None,
                        "center": center,
                        "rows": rows,
                    }
                )
                continue
            if name == "noninferiority_absolute_tolerance":
                gate = next(
                    (
                        candidate.get("noninferiority_gate")
                        for candidate in candidates.values()
                        if isinstance(candidate.get("noninferiority_gate"), Mapping)
                    ),
                    None,
                )
                parsimony = next(
                    (
                        candidate.get("parsimony_gate")
                        for candidate in candidates.values()
                        if isinstance(candidate.get("parsimony_gate"), Mapping)
                        and candidate["parsimony_gate"].get("enforced") is True
                    ),
                    None,
                )
                rows = []
                for threshold in thresholds:
                    carrier_passed = None
                    if isinstance(gate, Mapping) and gate.get("per_seed"):
                        carrier_passed = all(
                            Campaign._meets_upper_bound(
                                float(item["degradation"]), threshold
                            )
                            for item in gate["per_seed"]
                        )
                    passing_ranks: list[str] = []
                    if isinstance(parsimony, Mapping):
                        for comparison in parsimony.get("comparisons", ()):
                            if not comparison.get("eligible"):
                                continue
                            if (
                                all(
                                    Campaign._meets_upper_bound(
                                        float(item["degradation"]), threshold
                                    )
                                    for item in comparison["per_seed"]
                                )
                                and Campaign._meets_upper_bound(
                                    float(comparison["median_degradation"]),
                                    threshold,
                                )
                                and Campaign._meets_upper_bound(
                                    float(comparison["worst_seed_degradation"]),
                                    threshold,
                                )
                            ):
                                passing_ranks.append(str(comparison["variant"]))
                    rows.append(
                        {
                            "threshold": threshold,
                            "carrier_passed": carrier_passed,
                            "passing_parsimony_variants": passing_ranks,
                        }
                    )
                surfaces.append(
                    {
                        "name": name,
                        "applicable": center is not None,
                        "center": center,
                        "rows": rows,
                    }
                )
                continue

            rows = []
            for threshold in thresholds:
                passed_candidates: list[str] = []
                evaluated_candidates: list[str] = []
                for candidate_id, candidate in candidates.items():
                    guards = [
                        observation.get("sharing_guard")
                        for observation in candidate.get("observations", ())
                        if isinstance(observation.get("sharing_guard"), Mapping)
                    ]
                    if not guards:
                        continue
                    evaluated_candidates.append(candidate_id)
                    passes_complete_policy = True
                    for contract_name, (
                        direction,
                        measurement_contracts,
                    ) in sharing_contract.items():
                        predicate = (
                            Campaign._meets_upper_bound
                            if direction == "upper"
                            else Campaign._meets_lower_bound
                        )
                        for measurement_name, center_threshold in measurement_contracts:
                            effective_threshold = (
                                threshold if contract_name == name else center_threshold
                            )
                            if effective_threshold is None or not all(
                                predicate(
                                    float(guard["measurements"][measurement_name]),
                                    float(effective_threshold),
                                )
                                for guard in guards
                            ):
                                passes_complete_policy = False
                                break
                        if not passes_complete_policy:
                            break
                    if passes_complete_policy:
                        passed_candidates.append(candidate_id)
                rows.append(
                    {
                        "threshold": threshold,
                        "evaluated_candidate_ids": sorted(evaluated_candidates),
                        "passing_candidate_ids": sorted(passed_candidates),
                    }
                )
            surfaces.append(
                {
                    "name": name,
                    "applicable": center is not None,
                    "center": center,
                    "rows": rows,
                }
            )
        return {
            "applicable": True,
            "basis": policy.threshold_basis,
            "mode": "marginal_counterfactuals_center_policy_not_retuned",
            "surfaces": surfaces,
        }

    @staticmethod
    def _selection_qualification_interpretation(
        cell: CellSpec,
        qualification: Mapping[str, Any],
        gate_amendment: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        scientific_outcome = qualification.get("scientific_outcome")
        scientific_passed = (
            scientific_outcome.get("passed")
            if isinstance(scientific_outcome, Mapping)
            else None
        )
        promotion_eligible = qualification.get("promotion_eligible")
        selection_mode = qualification.get("selection_eligibility_mode")
        result = {
            "scientific_outcome_passed": scientific_passed,
            "promotion_eligible": promotion_eligible,
            "selection_eligibility_mode": selection_mode,
            "amended": False,
            "amendment_id": None,
            "original_failed_scientific_checks": [],
        }
        if gate_amendment is None or cell.phase is not Phase.PHASE2:
            return result
        Campaign._validate_phase2_gate_amendment_manifest(gate_amendment)
        checks = (
            scientific_outcome.get("checks")
            if isinstance(scientific_outcome, Mapping)
            else None
        )
        if not isinstance(checks, Mapping):
            raise CampaignError(
                f"qualification for {cell.cell_id} lacks scientific checks"
            )
        failed = sorted(name for name, passed in checks.items() if passed is not True)
        result["amendment_id"] = gate_amendment["amendment_id"]
        result["original_failed_scientific_checks"] = failed
        if scientific_passed is True:
            return result
        can_reinterpret = bool(
            failed
            and set(failed).issubset(PHASE2_CODEC_DIAGNOSTIC_CHECKS)
            and qualification.get("qualified") is True
            and isinstance(qualification.get("checks"), Mapping)
            and all(value is True for value in qualification["checks"].values())
            and cell.decision_map.get("qualification.promotable") is True
            and qualification.get("promotion_ineligible_reasons")
            == ["scientific_outcome_failed"]
            and qualification.get("selection_eligible_for_protocol_test") is False
        )
        if can_reinterpret:
            result.update(
                {
                    "scientific_outcome_passed": True,
                    "promotion_eligible": True,
                    "selection_eligibility_mode": "scientific_promotion",
                    "amended": True,
                }
            )
        return result

    @staticmethod
    def _selection_universe_from_evidence(
        stage_name: str,
        stage_cells: Sequence[CellSpec],
        policy: SelectionPolicy,
        evidence_by_cell_id: Mapping[str, Mapping[str, Any]],
        *,
        sharing_guard_for_cell: Callable[
            [CellSpec, Mapping[str, Any], SelectionPolicy], Mapping[str, Any]
        ],
        gate_amendment: Mapping[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], bool]:
        """Replay a complete stage universe from authenticated qualifications.

        This is the single selection reducer used both while a live campaign
        emits an artifact and while a self-contained Phase-3 panel envelope is
        parsed elsewhere.  Keeping eligibility, exclusion reasons, scientific
        gates, aggregation, and ordering here prevents a content-addressed but
        self-authored envelope from redefining its own candidate universe.
        """

        cells = tuple(stage_cells)
        if not cells:
            raise CampaignError(f"stage {stage_name!r} has no declared cells")
        if any(cell.stage != stage_name for cell in cells):
            raise CampaignError(
                f"stage {stage_name!r} selection contains a foreign-stage cell"
            )
        smoke_values = {cell.decision_map.get("runtime.smoke") for cell in cells}
        if smoke_values not in ({False}, {True}):
            raise CampaignError(
                f"stage {stage_name!r} mixes smoke and scientific cells"
            )
        smoke_protocol_only = smoke_values == {True}
        stage_seed_universe = tuple(sorted({cell.seed for cell in cells}))
        by_candidate: dict[str, list[tuple[CellSpec, Mapping[str, Any]]]] = {}
        for cell in cells:
            evidence = evidence_by_cell_id.get(cell.cell_id)
            if not isinstance(evidence, Mapping):
                raise CampaignError(
                    f"stage {stage_name!r} lacks evidence for {cell.cell_id}"
                )
            state = evidence.get("state")
            if state not in {
                RunState.QUALIFIED.value,
                RunState.PROMOTED.value,
                RunState.FAILED.value,
            }:
                raise CampaignError(
                    f"stage {stage_name!r} is not terminal; {cell.cell_id} is {state}"
                )
            by_candidate.setdefault(cell.candidate_id, []).append((cell, evidence))

        candidates: list[dict[str, Any]] = []
        excluded: list[dict[str, Any]] = []
        sensitivity_population: list[dict[str, Any]] = []
        for candidate_id, entries in sorted(by_candidate.items()):
            entries = sorted(entries, key=lambda item: item[0].seed)
            recipe_names = {cell.recipe_name for cell, _ in entries}
            recipe_ids = {cell.recipe_id for cell, _ in entries}
            if len(recipe_names) != 1 or len(recipe_ids) != 1:
                raise CampaignError(f"candidate {candidate_id} mixes recipe identities")
            recipe_name = next(iter(recipe_names))
            recipe_id = next(iter(recipe_ids))
            expected_seeds = [cell.seed for cell, _ in entries]
            if len(expected_seeds) != len(set(expected_seeds)):
                raise CampaignError(f"candidate {candidate_id} repeats a seed")
            if (
                policy.require_all_seeds
                and tuple(expected_seeds) != stage_seed_universe
            ):
                excluded.append(
                    {
                        "candidate_id": candidate_id,
                        "recipe_name": recipe_name,
                        "recipe_id": recipe_id,
                        "expected_seeds": list(stage_seed_universe),
                        "observed_seeds": expected_seeds,
                        "reason": "missing_declared_seed_replicates",
                    }
                )
                continue
            if policy.eligible_recipe_names and recipe_name not in set(
                policy.eligible_recipe_names
            ):
                excluded.append(
                    {
                        "candidate_id": candidate_id,
                        "recipe_name": recipe_name,
                        "recipe_id": recipe_id,
                        "expected_seeds": expected_seeds,
                        "reason": "recipe_not_eligible_under_frozen_policy",
                    }
                )
                continue
            qualified = [
                (cell, evidence)
                for cell, evidence in entries
                if evidence.get("state")
                in {RunState.QUALIFIED.value, RunState.PROMOTED.value}
            ]
            if len(qualified) != len(entries):
                excluded.append(
                    {
                        "candidate_id": candidate_id,
                        "recipe_name": recipe_name,
                        "recipe_id": recipe_id,
                        "expected_seeds": expected_seeds,
                        "reason": "not_all_seeds_qualified",
                        "states": {
                            str(cell.seed): evidence.get("state")
                            for cell, evidence in entries
                        },
                    }
                )
                continue
            observations: list[dict[str, Any]] = []
            outcome_failure = False
            promotion_ineligible = False
            protocol_ineligible = False
            for cell, evidence in qualified:
                qualification = evidence.get("qualification")
                qualification_sha256 = evidence.get("qualification_sha256")
                if not isinstance(qualification, Mapping) or not isinstance(
                    qualification_sha256, str
                ):
                    raise CampaignError(
                        f"qualified cell {cell.cell_id} lacks authenticated qualification evidence"
                    )
                interpretation = Campaign._selection_qualification_interpretation(
                    cell,
                    qualification,
                    gate_amendment,
                )
                scientific_passed = interpretation["scientific_outcome_passed"]
                if scientific_passed is not True:
                    outcome_failure = True
                promotion_eligible = interpretation["promotion_eligible"]
                if promotion_eligible is not True:
                    promotion_ineligible = True
                selection_mode = interpretation["selection_eligibility_mode"]
                if smoke_protocol_only and (
                    qualification.get("selection_eligible_for_protocol_test")
                    is not True
                    or selection_mode != "smoke_protocol_only"
                ):
                    protocol_ineligible = True
                selection_metrics = qualification.get("selection_metrics")
                if not isinstance(selection_metrics, Mapping):
                    raise CampaignError(
                        f"qualification for {cell.cell_id} lacks bound selection metrics"
                    )
                value = (
                    0.0
                    if smoke_protocol_only
                    else Campaign._policy_metric(selection_metrics, policy)
                )
                sharing_guard = (
                    dict(sharing_guard_for_cell(cell, selection_metrics, policy))
                    if policy.require_sharing_guard and not smoke_protocol_only
                    else None
                )
                observation = {
                    "cell_id": cell.cell_id,
                    "seed": cell.seed,
                    "metric": value,
                    "sharing_guard": sharing_guard,
                    "scientific_outcome_passed": scientific_passed,
                    "promotion_eligible": promotion_eligible,
                    "selection_eligibility_mode": selection_mode,
                    "qualification_sha256": qualification_sha256,
                }
                if gate_amendment is not None:
                    observation["qualification_interpretation"] = interpretation
                observations.append(observation)
            if smoke_protocol_only and protocol_ineligible:
                excluded.append(
                    {
                        "candidate_id": candidate_id,
                        "recipe_name": recipe_name,
                        "recipe_id": recipe_id,
                        "expected_seeds": expected_seeds,
                        "reason": "not_smoke_protocol_eligible",
                        "observations": observations,
                    }
                )
                continue
            if outcome_failure and not smoke_protocol_only:
                excluded.append(
                    {
                        "candidate_id": candidate_id,
                        "recipe_name": recipe_name,
                        "recipe_id": recipe_id,
                        "expected_seeds": expected_seeds,
                        "reason": "scientific_outcome_failed",
                        "observations": observations,
                    }
                )
                continue
            if promotion_ineligible and not smoke_protocol_only:
                excluded.append(
                    {
                        "candidate_id": candidate_id,
                        "recipe_name": recipe_name,
                        "recipe_id": recipe_id,
                        "expected_seeds": expected_seeds,
                        "reason": "promotion_ineligible_diagnostic",
                        "observations": observations,
                    }
                )
                continue
            failed_sharing_guards = [
                item["sharing_guard"]
                for item in observations
                if item["sharing_guard"] is not None
                and item["sharing_guard"].get("passed") is not True
            ]
            sharing_gate_enforced = bool(
                policy.require_sharing_guard
                and not smoke_protocol_only
                and gate_amendment is None
            )
            values = [float(item["metric"]) for item in observations]
            candidate_record = {
                "candidate_id": candidate_id,
                "recipe_name": recipe_name,
                "recipe_id": recipe_id,
                "median": float(median(values)),
                "worst_seed": (
                    max(values) if policy.direction == "min" else min(values)
                ),
                "sharing_guard_passed": (
                    not failed_sharing_guards
                    if policy.require_sharing_guard and not smoke_protocol_only
                    else None
                ),
                "selection_mode": (
                    "smoke_protocol_only"
                    if smoke_protocol_only
                    else "scientific_promotion"
                ),
                "observations": observations,
            }
            if gate_amendment is not None:
                candidate_record["sharing_gate_enforced"] = sharing_gate_enforced
            sensitivity_population.append(candidate_record)
            if (
                failed_sharing_guards
                and not smoke_protocol_only
                and sharing_gate_enforced
            ):
                excluded.append(
                    {
                        **candidate_record,
                        "reason": "sharing_guard_failed",
                        "failed_guards": failed_sharing_guards,
                    }
                )
                continue
            candidates.append(candidate_record)
        if smoke_protocol_only:
            for candidate in candidates:
                if policy.required_control_variant is not None:
                    candidate["noninferiority_gate"] = {
                        "enforced": False,
                        "mode": "smoke_protocol_only",
                    }
                if policy.default_parent_variant is not None:
                    candidate["minimum_effect_gate"] = {
                        "enforced": False,
                        "mode": "smoke_protocol_only",
                    }
                if policy.parsimony_order_variants:
                    candidate["parsimony_gate"] = {
                        "enforced": False,
                        "mode": "smoke_protocol_only",
                    }
        else:
            candidates = Campaign._apply_noninferiority_gate(
                stage_name, policy, candidates, excluded
            )
            candidates = Campaign._apply_minimum_effect_gate(
                stage_name, policy, candidates, excluded
            )
            candidates = Campaign._apply_rank_parsimony_gate(
                stage_name, policy, candidates, excluded
            )
        threshold_sensitivity = Campaign._threshold_sensitivity_payload(
            policy,
            sensitivity_population,
            smoke_protocol_only=smoke_protocol_only,
        )
        if not candidates:
            raise CampaignError(
                f"stage {stage_name!r} has no seed-complete, scientifically passing candidates"
            )
        if smoke_protocol_only:
            candidates.sort(key=lambda item: str(item["candidate_id"]))
        else:
            sign = 1.0 if policy.direction == "min" else -1.0
            candidates.sort(
                key=lambda item: (
                    sign * float(item["median"]),
                    sign * float(item["worst_seed"]),
                    str(item["candidate_id"]),
                )
            )
        return candidates, excluded, threshold_sensitivity, smoke_protocol_only

    def _selection_payload(
        self,
        stage_name: str,
        *,
        source_plan_id: str | None = None,
        policy_override: SelectionPolicy | None = None,
        gate_amendment: Mapping[str, Any] | None | object = (
            _AUTO_PHASE2_GATE_AMENDMENT
        ),
    ) -> dict[str, Any]:
        """Compute a stage selection from the current complete live universe.

        There are deliberately no caller-supplied metric, direction, or retain
        arguments.  Those choices are content-addressed in ``StageSpec`` before
        any evidence exists.  Qualification establishes evidence integrity;
        selection additionally requires ``scientific_outcome.passed`` for every
        declared seed of a candidate.
        """

        stages = {stage.name: stage for stage in self.plan.stages}
        try:
            stage = stages[stage_name]
        except KeyError as exc:
            raise CampaignError(f"unknown stage {stage_name!r}") from exc
        policy = stage.selection_policy if policy_override is None else policy_override
        if policy is None:
            raise CampaignError(f"stage {stage_name!r} has no selection policy")
        active_amendment = self._active_phase2_gate_amendment()
        if gate_amendment is _AUTO_PHASE2_GATE_AMENDMENT:
            amendment_manifest = (
                None if active_amendment is None else active_amendment[0]
            )
        elif gate_amendment is None:
            amendment_manifest = None
        elif isinstance(gate_amendment, Mapping):
            amendment_manifest = self._validate_phase2_gate_amendment_manifest(
                gate_amendment
            )
            if (
                active_amendment is None
                or amendment_manifest != active_amendment[0]
            ):
                raise CampaignError(
                    "selection requested an amendment not active in this campaign"
                )
        else:
            raise CampaignError("selection gate amendment argument is malformed")
        amendment_envelope = (
            None
            if amendment_manifest is None
            else {
                "artifact_sha256": "sha256:" + active_amendment[1].sha256,
                "manifest": amendment_manifest,
            }
        )
        normalized_evidence: dict[str, dict[str, Any]] = {}
        for cell in stage.cells:
            record = self.record(cell.cell_id)
            normalized: dict[str, Any] = {"state": record.state.value}
            if record.state in {RunState.QUALIFIED, RunState.PROMOTED}:
                self._validate_artifact_gate(
                    cell.cell_id, RunState.QUALIFIED, record.artifact_map
                )
                qualification = self._qualification_payload(record)
                if qualification is None:  # pragma: no cover - gate guarantees it
                    raise ArtifactError("qualified cell has no qualification artifact")
                normalized.update(
                    {
                        "qualification": qualification,
                        "qualification_sha256": "sha256:"
                        + record.artifact_map["qualification"].sha256,
                    }
                )
            normalized_evidence[cell.cell_id] = normalized
        (
            candidates,
            excluded,
            threshold_sensitivity,
            smoke_protocol_only,
        ) = self._selection_universe_from_evidence(
            stage_name,
            stage.cells,
            policy,
            normalized_evidence,
            sharing_guard_for_cell=self._sharing_guard_result,
            gate_amendment=amendment_manifest,
        )
        selected_candidates = _policy_retained_candidates(
            candidates,
            policy,
            smoke_protocol_only=smoke_protocol_only,
        )

        universe_payload = {
            "plan_id": self.plan.plan_id if source_plan_id is None else source_plan_id,
            "source_stage": stage_name,
            "policy_id": policy.policy_id,
            "ranked_candidates": candidates,
            "excluded_candidates": excluded,
        }
        if amendment_envelope is not None:
            universe_payload["gate_amendment"] = amendment_envelope
        universe_sha256 = (
            "sha256:"
            + hashlib.sha256(
                canonical_json(universe_payload).encode("utf-8")
            ).hexdigest()
        )
        cells_by_id = {cell.cell_id: cell for cell in stage.cells}
        frozen: list[FrozenSelection] = []
        for candidate in selected_candidates:
            observations = candidate["observations"]
            selected_cells = [cells_by_id[item["cell_id"]] for item in observations]
            frozen.append(
                FrozenSelection.from_cells(
                    policy,
                    selected_cells,
                    [float(item["metric"]) for item in observations],
                    [str(item["qualification_sha256"]) for item in observations],
                    universe_sha256,
                )
            )
        body = {
            "schema": SELECTION_SCHEMA,
            "plan_id": self.plan.plan_id if source_plan_id is None else source_plan_id,
            "phase": self.plan.phase.value,
            "source_stage": stage_name,
            "policy": policy.to_dict(),
            "selection_universe_sha256": universe_sha256,
            "ranked_candidates": candidates,
            "selected": [item.to_dict() for item in frozen],
            "excluded_candidates": excluded,
            "threshold_sensitivity": threshold_sensitivity,
            "smoke": smoke_protocol_only,
            "smoke_protocol_only": smoke_protocol_only,
            "selection_mode": (
                "smoke_protocol_only" if smoke_protocol_only else "scientific_promotion"
            ),
        }
        if amendment_envelope is not None:
            body["gate_amendment"] = amendment_envelope
        payload = {
            **body,
            "selection_id": content_id(body, prefix="selection"),
        }
        return payload

    def select_stage(
        self,
        stage_name: str,
        *,
        out: str | Path | None = None,
    ) -> dict[str, Any]:
        """Freeze the stage's immutable seed-aggregated selection policy."""

        payload = self._selection_payload(stage_name)
        destination = _campaign_output_path(
            self.root,
            out,
            self.root / "selections" / f"{stage_name}.json",
        )
        _write_immutable_json(destination, payload)
        return payload

    @staticmethod
    def _validate_phase2_gate_amendment_manifest(
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        expected_keys = {
            "schema",
            "amendment_id",
            "source_phase2_blueprint_id",
            "source_plan_id_at_adoption",
            "journal_event_id_before_adoption",
            "selection_policy_revision",
            "qualification_reinterpretation",
            "ratified_prior_selections",
            "implementation_identity",
            "implementation_identity_sha256",
            "rationale",
        }
        if set(payload) != expected_keys:
            raise CampaignError("Phase-2 gate amendment has a noncanonical field set")
        if payload.get("schema") != PHASE2_GATE_AMENDMENT_SCHEMA:
            raise CampaignError("Phase-2 gate amendment has the wrong schema")
        if not isinstance(payload.get("source_phase2_blueprint_id"), str) or not str(
            payload["source_phase2_blueprint_id"]
        ).startswith("phase2-blueprint:"):
            raise CampaignError("Phase-2 gate amendment lacks its blueprint binding")
        if not isinstance(payload.get("source_plan_id_at_adoption"), str) or not str(
            payload["source_plan_id_at_adoption"]
        ).startswith("study:"):
            raise CampaignError("Phase-2 gate amendment lacks its adoption plan")
        prior_event_id = payload.get("journal_event_id_before_adoption")
        if prior_event_id is not None and (
            not isinstance(prior_event_id, str) or not prior_event_id
        ):
            raise CampaignError("Phase-2 gate amendment has an invalid journal cursor")
        if payload.get("selection_policy_revision") != {
            "common_promotion_contract": (
                "integrity_complete_and_fixed_rate_raw_fvu_under_the_declared_"
                "method_specific_validity_contract"
            ),
            "partial_view_role": "common_nonpromotional_diagnostic",
            "sharing_gate_enforced": False,
            "applies_to": "all_phase2_method_families",
        }:
            raise CampaignError("Phase-2 gate amendment has the wrong selection policy")
        if payload.get("qualification_reinterpretation") != {
            "allowed_original_failed_checks": sorted(
                PHASE2_CODEC_DIAGNOSTIC_CHECKS
            ),
            "codec_exclusion_role": (
                "diagnostic_because_excluded_events_are_priced_in_fixed_rate_distortion"
            ),
            "eligibility_rule": (
                "integrity_complete_resolved_promotable_and_no_failed_scientific_"
                "checks_outside_allowed_original_failed_checks"
            ),
        }:
            raise CampaignError(
                "Phase-2 gate amendment has the wrong qualification reinterpretation"
            )
        ratified = payload.get("ratified_prior_selections")
        if not isinstance(ratified, list):
            raise CampaignError("Phase-2 gate amendment lacks prior-selection evidence")
        for row in ratified:
            if (
                not isinstance(row, Mapping)
                or set(row)
                != {
                    "source_stage",
                    "selection_artifact_sha256",
                    "selection_ids",
                    "selected_candidate_ids",
                }
                or not isinstance(row.get("source_stage"), str)
                or not row["source_stage"]
                or not isinstance(row.get("selection_artifact_sha256"), str)
                or not str(row["selection_artifact_sha256"]).startswith("sha256:")
                or not _is_sha256_hex(
                    str(row["selection_artifact_sha256"]).removeprefix("sha256:")
                )
                or not isinstance(row.get("selection_ids"), list)
                or not all(
                    isinstance(item, str) and item.startswith("selection:")
                    for item in row["selection_ids"]
                )
                or not isinstance(row.get("selected_candidate_ids"), list)
                or not all(
                    isinstance(item, str) and item.startswith("candidate:")
                    for item in row["selected_candidate_ids"]
                )
            ):
                raise CampaignError(
                    "Phase-2 gate amendment has malformed prior-selection evidence"
                )
        implementation = payload.get("implementation_identity")
        if not isinstance(implementation, Mapping):
            raise CampaignError(
                "Phase-2 gate amendment lacks its successor implementation identity"
            )
        observed_implementation_sha256 = _validate_implementation_identity(
            implementation,
            scientific=True,
        )
        if (
            payload.get("implementation_identity_sha256")
            != observed_implementation_sha256
        ):
            raise CampaignError(
                "Phase-2 gate amendment implementation digest is invalid"
            )
        if payload.get("rationale") != (
            "BSC receives deeper tuning as the target method, but promotion uses "
            "the same reconstruction standard as controls; partial-view endpoints "
            "measure cross-site behavior without presuming single-site sufficiency."
        ):
            raise CampaignError("Phase-2 gate amendment rationale is noncanonical")
        body = dict(payload)
        amendment_id = body.pop("amendment_id", None)
        if amendment_id != content_id(body, prefix="phase2-gate-amendment"):
            raise CampaignError("Phase-2 gate amendment content ID mismatch")
        return dict(payload)

    def _active_phase2_gate_amendment(
        self,
    ) -> tuple[dict[str, Any], ArtifactRef] | None:
        matches = [
            event
            for event in self._events_cached()
            if event.get("event") == "design_amendment"
        ]
        if not matches:
            return None
        if len(matches) != 1:
            raise CampaignError("campaign contains multiple Phase-2 gate amendments")
        event = matches[0]
        artifacts = event.get("artifacts")
        if not isinstance(artifacts, list) or len(artifacts) != 1:
            raise CampaignError("design-amendment event has invalid artifact evidence")
        try:
            ref = self._verify_artifact(ArtifactRef.from_dict(artifacts[0]))
        except (ArtifactError, KeyError, TypeError, ValueError) as exc:
            raise CampaignError(f"invalid design-amendment artifact: {exc}") from exc
        if ref.kind != PHASE2_GATE_AMENDMENT_ARTIFACT_KIND:
            raise CampaignError("design-amendment event has the wrong artifact kind")
        payload = self._validate_phase2_gate_amendment_manifest(
            _read_json(ref.resolve(self.root))
        )
        metadata = event.get("metadata")
        expected_metadata = {
            "amendment_id": payload["amendment_id"],
            "source_phase2_blueprint_id": payload["source_phase2_blueprint_id"],
            "source_plan_id_at_adoption": payload["source_plan_id_at_adoption"],
        }
        if metadata != expected_metadata:
            raise CampaignError("design-amendment journal binding mismatch")
        blueprint = self._phase2_blueprint()
        if payload["source_phase2_blueprint_id"] != blueprint.blueprint_id:
            raise CampaignError("design amendment belongs to another blueprint")
        return payload, ref

    @staticmethod
    def _validate_phase2_implementation_amendment_manifest(
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        if set(payload) != {
            "schema",
            "amendment_id",
            "gate_amendment_id",
            "source_plan_id_at_adoption",
            "journal_event_id_before_adoption",
            "predecessor_implementation_identity_sha256",
            "implementation_identity",
            "implementation_identity_sha256",
            "reason",
        }:
            raise CampaignError(
                "Phase-2 implementation amendment has a noncanonical field set"
            )
        if payload.get("schema") != PHASE2_IMPLEMENTATION_AMENDMENT_SCHEMA:
            raise CampaignError("Phase-2 implementation amendment has the wrong schema")
        if not isinstance(payload.get("gate_amendment_id"), str) or not str(
            payload["gate_amendment_id"]
        ).startswith("phase2-gate-amendment:"):
            raise CampaignError(
                "Phase-2 implementation amendment lacks its gate binding"
            )
        if not isinstance(payload.get("source_plan_id_at_adoption"), str) or not str(
            payload["source_plan_id_at_adoption"]
        ).startswith("study:"):
            raise CampaignError(
                "Phase-2 implementation amendment lacks its plan binding"
            )
        prior_event_id = payload.get("journal_event_id_before_adoption")
        if not isinstance(prior_event_id, str) or not prior_event_id:
            raise CampaignError(
                "Phase-2 implementation amendment lacks its journal cursor"
            )
        predecessor = payload.get("predecessor_implementation_identity_sha256")
        if not _is_sha256_hex(predecessor):
            raise CampaignError(
                "Phase-2 implementation amendment predecessor is malformed"
            )
        implementation = payload.get("implementation_identity")
        if not isinstance(implementation, Mapping):
            raise CampaignError(
                "Phase-2 implementation amendment lacks its implementation"
            )
        observed = _validate_implementation_identity(implementation, scientific=True)
        if payload.get("implementation_identity_sha256") != observed:
            raise CampaignError(
                "Phase-2 implementation amendment digest is invalid"
            )
        if payload.get("reason") != (
            "orchestration_only_branch_advance_fix_no_cell_kernel_change"
        ):
            raise CampaignError(
                "Phase-2 implementation amendment reason is noncanonical"
            )
        body = dict(payload)
        amendment_id = body.pop("amendment_id", None)
        if amendment_id != content_id(
            body, prefix="phase2-implementation-amendment"
        ):
            raise CampaignError(
                "Phase-2 implementation amendment content ID mismatch"
            )
        return dict(payload)

    def _phase2_implementation_amendments(
        self,
    ) -> tuple[tuple[dict[str, Any], ArtifactRef, int], ...]:
        gate = self._active_phase2_gate_amendment()
        if gate is None:
            return ()
        gate_payload, _gate_ref = gate
        predecessor = str(gate_payload["implementation_identity_sha256"])
        result: list[tuple[dict[str, Any], ArtifactRef, int]] = []
        for index, event in enumerate(self._events_cached()):
            if event.get("event") != "implementation_amendment":
                continue
            artifacts = event.get("artifacts")
            if not isinstance(artifacts, list) or len(artifacts) != 1:
                raise CampaignError(
                    "implementation-amendment event has invalid artifact evidence"
                )
            ref = self._verify_artifact(ArtifactRef.from_dict(artifacts[0]))
            if ref.kind != PHASE2_IMPLEMENTATION_AMENDMENT_ARTIFACT_KIND:
                raise CampaignError(
                    "implementation-amendment event has the wrong artifact kind"
                )
            payload = self._validate_phase2_implementation_amendment_manifest(
                _read_json(ref.resolve(self.root))
            )
            if (
                payload["gate_amendment_id"] != gate_payload["amendment_id"]
                or payload["predecessor_implementation_identity_sha256"]
                != predecessor
            ):
                raise CampaignError(
                    "Phase-2 implementation amendments do not form one chain"
                )
            metadata = event.get("metadata")
            if metadata != {
                "amendment_id": payload["amendment_id"],
                "gate_amendment_id": payload["gate_amendment_id"],
                "source_plan_id_at_adoption": payload[
                    "source_plan_id_at_adoption"
                ],
            }:
                raise CampaignError(
                    "implementation-amendment journal binding mismatch"
                )
            predecessor = str(payload["implementation_identity_sha256"])
            result.append((payload, ref, index))
        return tuple(result)

    def apply_phase2_implementation_amendment(
        self,
        *,
        out: str | Path | None = None,
    ) -> dict[str, Any]:
        """Pin a successor executor after a non-kernel orchestration repair."""

        if self.plan.phase is not Phase.PHASE2:
            raise CampaignError("implementation amendments apply only to Phase 2")
        with self._registration_mutation():
            gate = self._active_phase2_gate_amendment()
            if gate is None:
                raise CampaignError(
                    "implementation amendment requires the common-gate amendment"
                )
            chain = self._phase2_implementation_amendments()
            predecessor = (
                str(gate[0]["implementation_identity_sha256"])
                if not chain
                else str(chain[-1][0]["implementation_identity_sha256"])
            )
            from .implementation import implementation_identity  # noqa: PLC0415

            implementation = implementation_identity()
            implementation_sha256 = _validate_implementation_identity(
                implementation,
                scientific=True,
            )
            if implementation_sha256 == predecessor:
                return gate[0] if not chain else chain[-1][0]
            events = self._events_cached()
            body = {
                "schema": PHASE2_IMPLEMENTATION_AMENDMENT_SCHEMA,
                "gate_amendment_id": gate[0]["amendment_id"],
                "source_plan_id_at_adoption": self.plan.plan_id,
                "journal_event_id_before_adoption": str(events[-1]["event_id"]),
                "predecessor_implementation_identity_sha256": predecessor,
                "implementation_identity": implementation,
                "implementation_identity_sha256": implementation_sha256,
                "reason": (
                    "orchestration_only_branch_advance_fix_no_cell_kernel_change"
                ),
            }
            payload = {
                **body,
                "amendment_id": content_id(
                    body, prefix="phase2-implementation-amendment"
                ),
            }
            self._validate_phase2_implementation_amendment_manifest(payload)
            destination = _campaign_output_path(
                self.root,
                out,
                self.amendments_dir
                / f"phase2-implementation-{implementation_sha256[:12]}.json",
            )
            _write_immutable_json(destination, payload)
            ref = self._verify_artifact(
                ArtifactRef.from_path(
                    PHASE2_IMPLEMENTATION_AMENDMENT_ARTIFACT_KIND,
                    destination,
                    root=self.root,
                )
            )
            event = self._event(
                "implementation_amendment",
                "__campaign__",
                message="pinned successor Phase-2 orchestration implementation",
                metadata={
                    "amendment_id": payload["amendment_id"],
                    "gate_amendment_id": payload["gate_amendment_id"],
                    "source_plan_id_at_adoption": self.plan.plan_id,
                },
                artifacts=(ref,),
            )
            self._append_event_locked(event)
            return payload

    def apply_phase2_gate_amendment(
        self,
        *,
        out: str | Path | None = None,
    ) -> dict[str, Any]:
        """Adopt the corrected common Phase-2 gate without discarding evidence."""

        if self.plan.phase is not Phase.PHASE2:
            raise CampaignError("the common-gate amendment applies only to Phase 2")
        with self._registration_mutation():
            existing = self._active_phase2_gate_amendment()
            if existing is not None:
                return existing[0]
            blueprint = self._phase2_blueprint()
            events = self._events_cached()
            ratified: list[dict[str, Any]] = []
            for event in events:
                if event.get("event") != "plan_extension":
                    continue
                artifacts = event.get("artifacts")
                if not isinstance(artifacts, list) or len(artifacts) != 1:
                    raise CampaignError(
                        "cannot ratify a malformed historical plan extension"
                    )
                ref = self._verify_artifact(ArtifactRef.from_dict(artifacts[0]))
                if ref.kind != "stage_selection":
                    continue
                selection = _read_json(ref.resolve(self.root))
                selected = selection.get("selected")
                if not isinstance(selected, list) or not selected:
                    raise CampaignError(
                        "cannot ratify a historical selection without candidates"
                    )
                ratified.append(
                    {
                        "source_stage": str(selection["source_stage"]),
                        "selection_artifact_sha256": "sha256:" + ref.sha256,
                        "selection_ids": [
                            str(item["selection_id"]) for item in selected
                        ],
                        "selected_candidate_ids": [
                            str(item["candidate_id"]) for item in selected
                        ],
                    }
                )
            from .implementation import (  # noqa: PLC0415
                implementation_identity,
            )

            implementation = implementation_identity()
            implementation_sha256 = _validate_implementation_identity(
                implementation,
                scientific=True,
            )
            body = {
                "schema": PHASE2_GATE_AMENDMENT_SCHEMA,
                "source_phase2_blueprint_id": blueprint.blueprint_id,
                "source_plan_id_at_adoption": self.plan.plan_id,
                "journal_event_id_before_adoption": (
                    None if not events else str(events[-1]["event_id"])
                ),
                "selection_policy_revision": {
                    "common_promotion_contract": (
                        "integrity_complete_and_fixed_rate_raw_fvu_under_the_declared_"
                        "method_specific_validity_contract"
                    ),
                    "partial_view_role": "common_nonpromotional_diagnostic",
                    "sharing_gate_enforced": False,
                    "applies_to": "all_phase2_method_families",
                },
                "qualification_reinterpretation": {
                    "allowed_original_failed_checks": sorted(
                        PHASE2_CODEC_DIAGNOSTIC_CHECKS
                    ),
                    "codec_exclusion_role": (
                        "diagnostic_because_excluded_events_are_priced_in_fixed_rate_distortion"
                    ),
                    "eligibility_rule": (
                        "integrity_complete_resolved_promotable_and_no_failed_"
                        "scientific_checks_outside_allowed_original_failed_checks"
                    ),
                },
                "ratified_prior_selections": ratified,
                "implementation_identity": implementation,
                "implementation_identity_sha256": implementation_sha256,
                "rationale": (
                    "BSC receives deeper tuning as the target method, but promotion "
                    "uses the same reconstruction standard as controls; partial-view "
                    "endpoints measure cross-site behavior without presuming "
                    "single-site sufficiency."
                ),
            }
            payload = {
                **body,
                "amendment_id": content_id(body, prefix="phase2-gate-amendment"),
            }
            self._validate_phase2_gate_amendment_manifest(payload)
            destination = _campaign_output_path(
                self.root,
                out,
                self.amendments_dir / "phase2-common-gates.json",
            )
            _write_immutable_json(destination, payload)
            ref = self._verify_artifact(
                ArtifactRef.from_path(
                    PHASE2_GATE_AMENDMENT_ARTIFACT_KIND,
                    destination,
                    root=self.root,
                )
            )
            event = self._event(
                "design_amendment",
                "__campaign__",
                message="adopted corrected common Phase-2 promotion gates",
                metadata={
                    "amendment_id": payload["amendment_id"],
                    "source_phase2_blueprint_id": blueprint.blueprint_id,
                    "source_plan_id_at_adoption": self.plan.plan_id,
                },
                artifacts=(ref,),
            )
            self._append_event_locked(event)
            return payload

    def _phase2_blueprint(self) -> Phase2Blueprint:
        if self.plan.phase.value != "phase2" or not self.blueprint_path.is_file():
            raise CampaignError(
                "comparator-family operations require a registered Phase-2 blueprint"
            )
        try:
            return Phase2Blueprint.from_manifest(_read_json(self.blueprint_path))
        except (KeyError, TypeError, ValueError, StudyError) as exc:
            raise CampaignError(f"invalid registered Phase-2 blueprint: {exc}") from exc

    @staticmethod
    def _family_blueprint(blueprint: Phase2Blueprint, family_name: str):
        matches = [
            family
            for family in blueprint.comparator_families
            if family.name == family_name
        ]
        if len(matches) != 1:
            raise CampaignError(f"unknown comparator family {family_name!r}")
        return matches[0]

    def select_family_root(
        self,
        family_name: str,
        *,
        out: str | Path | None = None,
    ) -> dict[str, Any]:
        """Freeze one family's anchor under its own root-only policy."""

        blueprint = self._phase2_blueprint()
        family = self._family_blueprint(blueprint, family_name)
        stages = {stage.name: stage for stage in self.plan.stages}
        if stages.get(blueprint.initial_stage.name) != blueprint.initial_stage:
            raise CampaignError("active Phase-2 plan lacks its exact anchor stage")
        payload = self._selection_payload(
            blueprint.initial_stage.name,
            policy_override=family.root_selection_policy,
        )
        destination = _campaign_output_path(
            self.root,
            out,
            self.root / "selections" / f"family_{family.name}_root.json",
        )
        _write_immutable_json(destination, payload)
        return payload

    @staticmethod
    def _deduplicate_family_nomination_candidates(
        candidates: Sequence[dict[str, Any]],
        source_rounds: Sequence[str],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Collapse execution aliases before outcome ranking.

        The canonical representative is chosen solely by declared source-round
        order and candidate ID.  A configuration repeated as a center in later
        rounds therefore gets no best-of-repeats advantage.  All alias metrics
        and their spread remain in the evidence.
        """

        source_order = {name: index for index, name in enumerate(source_rounds)}
        if len(source_order) != len(tuple(source_rounds)):
            raise CampaignError("family nomination repeats a source round")
        ordered = sorted(
            candidates,
            key=lambda item: (
                source_order.get(str(item.get("source_stage", "")), math.inf),
                str(item.get("candidate_id", "")),
            ),
        )
        representatives: dict[str, dict[str, Any]] = {}
        duplicates: list[dict[str, Any]] = []
        for candidate in ordered:
            source_stage = str(candidate.get("source_stage", ""))
            if source_stage not in source_order:
                raise CampaignError(
                    "family nomination candidate escaped its declared source rounds"
                )
            signature = str(candidate.get("execution_signature", ""))
            if not signature:
                raise CampaignError(
                    "family nomination candidate lacks an execution signature"
                )
            observations = candidate.get("observations")
            if not isinstance(observations, list) or not observations:
                raise CampaignError("family nomination candidate lacks observations")
            alias = {
                "candidate_id": str(candidate["candidate_id"]),
                "source_stage": source_stage,
                "median": float(candidate["median"]),
                "worst_seed": float(candidate["worst_seed"]),
                "per_seed_metrics": [
                    {
                        "seed": int(item["seed"]),
                        "metric": float(item["metric"]),
                    }
                    for item in observations
                ],
            }
            representative = representatives.get(signature)
            if representative is None:
                candidate["execution_aliases"] = [alias]
                candidate["execution_representative_policy"] = (
                    "earliest_declared_source_round_then_candidate_id_before_outcome_ranking"
                )
                representatives[signature] = candidate
                continue
            representative["execution_aliases"].append(alias)
            duplicates.append(
                {
                    **candidate,
                    "execution_aliases": [alias],
                    "exclusion_reason": "duplicate_resolved_execution_signature",
                    "representative_candidate_id": representative["candidate_id"],
                    "representative_source_stage": representative["source_stage"],
                }
            )
        for representative in representatives.values():
            aliases = representative["execution_aliases"]
            medians = [float(alias["median"]) for alias in aliases]
            worst = [float(alias["worst_seed"]) for alias in aliases]
            by_seed: dict[int, list[float]] = {}
            for alias in aliases:
                for item in alias["per_seed_metrics"]:
                    by_seed.setdefault(int(item["seed"]), []).append(
                        float(item["metric"])
                    )
            representative["execution_alias_metric_spread"] = {
                "median_max_minus_min": max(medians) - min(medians),
                "worst_seed_max_minus_min": max(worst) - min(worst),
                "maximum_per_seed_max_minus_min": max(
                    (max(values) - min(values) for values in by_seed.values()),
                    default=0.0,
                ),
            }
        return list(representatives.values()), duplicates

    def _family_nomination_payload(
        self,
        family_name: str,
        *,
        source_plan_id: str | None = None,
    ) -> dict[str, Any]:
        """Rank the complete union of one family's declared 4M candidates."""

        blueprint = self._phase2_blueprint()
        family = self._family_blueprint(blueprint, family_name)
        policy = family.revisit.nomination_policy
        bound_plan_id = self.plan.plan_id if source_plan_id is None else source_plan_id
        stages = {stage.name: stage for stage in self.plan.stages}
        missing = [name for name in family.revisit.source_rounds if name not in stages]
        if missing:
            raise CampaignError(
                f"family {family_name!r} lacks materialized nomination rounds {missing}"
            )
        ranked: list[dict[str, Any]] = []
        excluded: list[dict[str, Any]] = []
        source_threshold_sensitivity: dict[str, Any] = {}
        for stage_name in family.revisit.source_rounds:
            payload = self._selection_payload(
                stage_name,
                source_plan_id=bound_plan_id,
                policy_override=policy,
            )
            ranked.extend(
                {**candidate, "source_stage": stage_name}
                for candidate in payload["ranked_candidates"]
            )
            excluded.extend(
                {**candidate, "source_stage": stage_name}
                for candidate in payload["excluded_candidates"]
            )
            source_threshold_sensitivity[stage_name] = payload["threshold_sensitivity"]
        candidate_ids = [str(candidate["candidate_id"]) for candidate in ranked]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise CampaignError("family nomination union repeats a candidate identity")

        def resolved_cells(candidate: Mapping[str, Any]) -> tuple[CellSpec, ...]:
            stage = stages[str(candidate["source_stage"])]
            cells_by_id = {cell.cell_id: cell for cell in stage.cells}
            observations = candidate.get("observations")
            if not isinstance(observations, list) or not observations:
                raise CampaignError("family nomination candidate lacks observations")
            try:
                return tuple(cells_by_id[str(item["cell_id"])] for item in observations)
            except (KeyError, TypeError) as exc:
                raise CampaignError(
                    "family nomination observation escaped its source stage"
                ) from exc

        for candidate in ranked:
            try:
                signature = resolved_candidate_execution_signature(
                    resolved_cells(candidate)
                )
            except StudyError as exc:
                raise CampaignError(
                    f"invalid family nomination execution signature: {exc}"
                ) from exc
            candidate["execution_signature"] = signature
        ranked, duplicate_aliases = self._deduplicate_family_nomination_candidates(
            ranked,
            family.revisit.source_rounds,
        )
        excluded.extend(duplicate_aliases)
        sign = 1.0 if policy.direction == "min" else -1.0
        ranked.sort(
            key=lambda item: (
                sign * float(item["median"]),
                sign * float(item["worst_seed"]),
                str(item["candidate_id"]),
            )
        )
        if len(ranked) < family.revisit.top_k:
            raise CampaignError(
                f"family {family_name!r} has fewer than {family.revisit.top_k} "
                "distinct eligible nomination configurations"
            )
        universe_payload = {
            "plan_id": bound_plan_id,
            "source_stage": family.revisit.name,
            "policy_id": policy.policy_id,
            "family_name": family.name,
            "family_id": family.family_id,
            "source_rounds": list(family.revisit.source_rounds),
            "ranked_candidates": ranked,
            "excluded_candidates": excluded,
            "source_threshold_sensitivity": source_threshold_sensitivity,
        }
        universe_sha256 = _canonical_sha256(universe_payload)
        frozen: list[FrozenSelection] = []
        for candidate in ranked[: family.revisit.top_k]:
            observations = candidate["observations"]
            cells = resolved_cells(candidate)
            frozen.append(
                FrozenSelection.from_cells(
                    policy,
                    cells,
                    [float(item["metric"]) for item in observations],
                    [str(item["qualification_sha256"]) for item in observations],
                    universe_sha256,
                )
            )
        body = {
            "schema": FAMILY_NOMINATION_SCHEMA,
            "plan_id": bound_plan_id,
            "phase": self.plan.phase.value,
            "blueprint_id": blueprint.blueprint_id,
            "family_name": family.name,
            "family_id": family.family_id,
            "revisit_id": family.revisit.revisit_id,
            "source_rounds": list(family.revisit.source_rounds),
            "policy": policy.to_dict(),
            "selection_universe_sha256": universe_sha256,
            "ranked_candidates": ranked,
            "selected": [selection.to_dict() for selection in frozen],
            "excluded_candidates": excluded,
            "source_threshold_sensitivity": source_threshold_sensitivity,
            "smoke": any(
                cell.decision_map.get("runtime.smoke") is True
                for stage_name in family.revisit.source_rounds
                for cell in stages[stage_name].cells
            ),
        }
        return {
            **body,
            "nomination_id": content_id(body, prefix="family-nomination"),
        }

    def select_family_revisit_inputs(
        self,
        family_name: str,
        *,
        out: str | Path | None = None,
    ) -> dict[str, Any]:
        """Freeze a family's complete cross-round top-two nomination."""

        payload = self._family_nomination_payload(family_name)
        destination = _campaign_output_path(
            self.root,
            out,
            self.root / "selections" / f"family_{family_name}_revisit_nomination.json",
        )
        _write_immutable_json(destination, payload)
        return payload

    def freeze_phase1_decision(
        self,
        *,
        scope_narrowing: Mapping[str, str] | None = None,
        out: str | Path | None = None,
    ) -> dict[str, Any]:
        """Freeze a complete Phase-1 campaign into a Phase-2 authorization.

        The selected baseline must pass, and both content-declared negative
        controls must fail the identification conjunction on every seed.  Any
        other failed stress arm remains reportable evidence only when the
        decision records an explicit, nonempty claim-scope narrowing for it.
        """

        declared_scope = dict(scope_narrowing or {})
        durable_mkdir(self.root, parents=True, exist_ok=True)
        with self._registration_mutation():
            self._reconcile_plan_projection()
            if not self.plan_path.is_file() or not self.blueprint_path.is_file():
                raise CampaignError(
                    "Phase-1 freeze requires a registered plan and blueprint"
                )
            plan_sha256 = _sha256(self.plan_path)
            blueprint_sha256 = _sha256(self.blueprint_path)
            journal_sha256 = (
                _sha256(self.journal_path) if self.journal_path.is_file() else None
            )
            if journal_sha256 is None:
                raise CampaignError("Phase-1 freeze requires its append-only journal")
            plan_manifest = _read_json(self.plan_path)
            blueprint_manifest = _read_json(self.blueprint_path)
            try:
                plan = StudyPlan.from_manifest(plan_manifest)
                blueprint = Phase1Blueprint.from_manifest(blueprint_manifest)
            except (KeyError, TypeError, ValueError, StudyError) as exc:
                raise CampaignError(f"invalid Phase-1 plan/blueprint: {exc}") from exc
            if plan.phase.value != "phase1":
                raise CampaignError("only Phase 1 can emit a Phase-2 authorization")
            if canonical_json(plan_manifest) != canonical_json(
                plan.to_manifest()
            ) or canonical_json(blueprint_manifest) != canonical_json(
                blueprint.to_manifest()
            ):
                raise CampaignError("Phase-1 plan or blueprint is noncanonical")
            expected_stage_names = (
                *(stage.name for stage in blueprint.initial_stages),
                *(round_spec.name for round_spec in blueprint.rounds),
            )
            if tuple(stage.name for stage in plan.stages) != expected_stage_names:
                raise CampaignError(
                    "Phase-1 plan is not the fully materialized blueprint"
                )
            if plan.stages[: len(blueprint.initial_stages)] != blueprint.initial_stages:
                raise CampaignError(
                    "Phase-1 initial stages differ from their blueprint"
                )
            smoke_values = {
                cell.decision_map.get("runtime.smoke") for cell in plan.cells
            }
            if smoke_values not in ({False}, {True}):
                raise CampaignError("Phase-1 plan mixes smoke and scientific cells")
            smoke = smoke_values == {True}

            all_events = self._events_cached()
            known_cell_ids = {cell.cell_id for cell in plan.cells}
            unknown_transition_cells = {
                str(event.get("cell_id"))
                for event in all_events
                if event.get("event") == "transition"
                and event.get("cell_id") not in known_cell_ids
            }
            if unknown_transition_cells:
                raise CampaignError(
                    "Phase-1 journal contains transitions outside its exact plan"
                )
            records: dict[str, CampaignRecord] = {}
            cell_evidence: list[dict[str, Any]] = []
            for cell in plan.cells:
                record = self.record(cell.cell_id)
                if record.state not in {RunState.QUALIFIED, RunState.PROMOTED}:
                    raise CampaignError(
                        "Phase-1 decision requires every materialized cell to be qualified; "
                        f"{cell.cell_id} is {record.state.value}"
                    )
                self._validate_artifact_gate(
                    cell.cell_id, RunState.QUALIFIED, record.artifact_map
                )
                for artifact in record.artifacts:
                    artifact.verify(self.root)
                qualification = self._qualification_payload(record)
                if qualification is None:  # pragma: no cover - gate guarantees it
                    raise CampaignError("qualified Phase-1 cell lacks qualification")
                records[cell.cell_id] = record
                cell_evidence.append(
                    {
                        "cell_id": cell.cell_id,
                        "candidate_id": cell.candidate_id,
                        "stage": cell.stage,
                        "seed": cell.seed,
                        "recipe_name": cell.recipe_name,
                        "recipe_id": cell.recipe_id,
                        "cell": cell.to_manifest(),
                        "state": record.state.value,
                        "qualification_sha256": (
                            "sha256:" + record.artifact_map["qualification"].sha256
                        ),
                        "qualification": qualification,
                    }
                )
            evidence_by_id = {item["cell_id"]: item for item in cell_evidence}

            smoke_suffix = "_smoke" if smoke else ""
            expected = StudyPlan(
                f"phase1_synthetic_prefix_{len(blueprint.initial_stages)}{smoke_suffix}",
                plan.phase,
                blueprint.initial_stages,
            )
            extension_events = tuple(
                event for event in all_events if event.get("event") == "plan_extension"
            )
            if len(extension_events) != len(blueprint.rounds):
                raise CampaignError(
                    "Phase-1 journal does not contain exactly one extension per round"
                )
            selection_chain: list[dict[str, Any]] = []
            selection_refs: list[ArtifactRef] = []
            plan_history: list[dict[str, str]] = []

            def verify_history(prefix: StudyPlan) -> None:
                path = self.plans_dir / f"{_slug(prefix.plan_id)}.json"
                if not path.is_file() or canonical_json(
                    _read_json(path)
                ) != canonical_json(prefix.to_manifest()):
                    raise CampaignError(
                        f"missing or stale Phase-1 plan history for {prefix.plan_id}"
                    )
                plan_history.append(
                    {
                        "plan_id": prefix.plan_id,
                        "sha256": "sha256:" + _sha256(path),
                    }
                )

            verify_history(expected)
            for event in extension_events:
                metadata = event.get("metadata")
                if not isinstance(metadata, Mapping):
                    raise CampaignError("Phase-1 extension metadata must be an object")
                source_stage = expected.stages[-1]
                policy = source_stage.selection_policy
                if policy is None:
                    raise CampaignError(
                        "Phase-1 extension follows a nonselectable stage"
                    )
                try:
                    event_artifacts = tuple(
                        ArtifactRef.from_dict(item)
                        for item in event.get("artifacts", ())
                    )
                except (ArtifactError, KeyError, TypeError, ValueError) as exc:
                    raise CampaignError(
                        f"invalid Phase-1 selection artifact: {exc}"
                    ) from exc
                if (
                    len(event_artifacts) != 1
                    or event_artifacts[0].kind != "stage_selection"
                ):
                    raise CampaignError(
                        "Phase-1 extension must bind exactly one stage selection"
                    )
                selection_ref = event_artifacts[0]
                selection_ref.verify(self.root)
                selection_payload = _read_json(selection_ref.resolve(self.root))
                live_payload = self._selection_payload(
                    source_stage.name,
                    source_plan_id=expected.plan_id,
                )
                if canonical_json(selection_payload) != canonical_json(live_payload):
                    raise CampaignError(
                        f"Phase-1 selection for {source_stage.name} is stale"
                    )
                event_selection_id = metadata.get("selection_id")
                try:
                    matches = [
                        FrozenSelection.from_dict(item)
                        for item in selection_payload.get("selected", ())
                        if item.get("selection_id") == event_selection_id
                    ]
                except (KeyError, TypeError, ValueError, StudyError) as exc:
                    raise CampaignError(
                        f"invalid Phase-1 frozen selection: {exc}"
                    ) from exc
                if len(matches) != 1:
                    raise CampaignError(
                        "Phase-1 extension does not identify one selected candidate"
                    )
                selection = matches[0]
                extended = materialize_child_plan(expected, blueprint, selection)
                expected_metadata = {
                    "previous_plan_id": expected.plan_id,
                    "plan_id": extended.plan_id,
                    "stage": extended.stages[-1].name,
                    "selection_id": selection.selection_id,
                }
                if (
                    metadata != expected_metadata
                    or plan.stages[: len(extended.stages)] != extended.stages
                ):
                    raise CampaignError(
                        "Phase-1 extension differs from blueprint replay"
                    )
                selection_chain.append(
                    {
                        "source_plan_id": expected.plan_id,
                        "source_stage": source_stage.name,
                        "target_plan_id": extended.plan_id,
                        "target_stage": extended.stages[-1].name,
                        "policy_id": policy.policy_id,
                        "selection_id": selection.selection_id,
                        "selection_universe_sha256": (
                            selection.selection_universe_sha256
                        ),
                        "selection_artifact_sha256": ("sha256:" + selection_ref.sha256),
                        "selection_artifact_sha256_semantics": (
                            "opaque_historical_commitment_requires_trusted_origin"
                        ),
                        "selection": selection.to_dict(),
                    }
                )
                selection_refs.append(selection_ref)
                expected = extended
                verify_history(expected)
            if expected != plan:
                raise CampaignError(
                    "Phase-1 extension replay does not reach active plan"
                )

            final_stage = plan.stages[-1]
            by_variant: dict[str, list[CellSpec]] = {}
            for cell in final_stage.cells:
                variant = cell.decision_map.get("factor.robustness")
                if not isinstance(variant, str):
                    raise CampaignError(
                        "Phase-1 confirmation lacks a robustness-variant binding"
                    )
                by_variant.setdefault(variant, []).append(cell)
            confirmation_results: list[dict[str, Any]] = []
            for variant, cells in sorted(by_variant.items()):
                ordered = sorted(cells, key=lambda cell: cell.seed)
                if tuple(cell.seed for cell in ordered) != blueprint.seeds:
                    raise CampaignError(
                        f"Phase-1 robustness variant {variant!r} is not seed-complete"
                    )
                roles = {
                    cell.decision_map.get("qualification.phase1_confirmation_role")
                    for cell in ordered
                }
                if len(roles) != 1 or next(iter(roles)) not in {
                    "required_baseline_pass",
                    "required_negative_control_failure",
                    "claim_scope_stress",
                }:
                    raise CampaignError(
                        "Phase-1 robustness variant lacks one frozen decision role"
                    )
                role = next(iter(roles))
                per_seed = []
                for cell in ordered:
                    evidence = evidence_by_id[cell.cell_id]
                    per_seed.append(
                        {
                            "seed": cell.seed,
                            "cell_id": cell.cell_id,
                            "qualification_sha256": evidence["qualification_sha256"],
                            **self._phase1_claim_evidence(
                                evidence["qualification"], cell=cell, smoke=smoke
                            ),
                        }
                    )
                confirmation_results.append(
                    {
                        "variant": variant,
                        "candidate_id": ordered[0].candidate_id,
                        "required_baseline": role == "required_baseline_pass",
                        "negative_control": role == "required_negative_control_failure",
                        "negative_control_passed": (
                            None
                            if smoke or role != "required_negative_control_failure"
                            else all(
                                item["conjunction_passed"] is False for item in per_seed
                            )
                        ),
                        "passed_all_seeds": all(
                            item["conjunction_passed"] is True for item in per_seed
                        ),
                        "per_seed": per_seed,
                    }
                )
            result_by_variant = {item["variant"]: item for item in confirmation_results}
            baseline_results = [
                item for item in confirmation_results if item["required_baseline"]
            ]
            negative_control_results = [
                item for item in confirmation_results if item["negative_control"]
            ]
            if len(baseline_results) != 1 or len(negative_control_results) != 2:
                raise CampaignError(
                    "Phase-1 confirmation lacks its frozen baseline/negative-control roles"
                )
            baseline_variant = str(baseline_results[0]["variant"])
            negative_control_variants = {
                str(item["variant"]) for item in negative_control_results
            }
            stress_failures = sorted(
                variant
                for variant, result in result_by_variant.items()
                if variant != baseline_variant
                and variant not in negative_control_variants
                and result["passed_all_seeds"] is not True
            )
            if set(declared_scope) != set(stress_failures) or any(
                not isinstance(value, str) or not value.strip()
                for value in declared_scope.values()
            ):
                raise CampaignError(
                    "every failed Phase-1 stress requires exactly one explicit scope narrowing"
                )
            baseline_passed = (
                result_by_variant[baseline_variant]["passed_all_seeds"] is True
            )
            negative_controls_passed = smoke or all(
                result_by_variant[variant].get("negative_control_passed") is True
                for variant in negative_control_variants
            )
            if smoke:
                authorization_mode = "smoke_protocol_only"
                decision = "protocol_complete"
                authorizes_scientific = False
            else:
                scientific_go = baseline_passed and negative_controls_passed
                authorization_mode = (
                    "scientific_go" if scientific_go else "scientific_no_go"
                )
                decision = "go" if scientific_go else "no_go"
                authorizes_scientific = scientific_go
            campaign_manifest = {
                "schema": PHASE1_CAMPAIGN_MANIFEST_SCHEMA,
                "source_phase1_plan_id": plan.plan_id,
                "source_phase1_blueprint_id": blueprint.blueprint_id,
                "plan_content_sha256": _canonical_sha256(plan.to_manifest()),
                "blueprint_content_sha256": _canonical_sha256(blueprint.to_manifest()),
                "plan_sha256": "sha256:" + plan_sha256,
                "blueprint_sha256": "sha256:" + blueprint_sha256,
                "journal_sha256": "sha256:" + journal_sha256,
                "journal_sha256_semantics": (
                    "opaque_historical_commitment_requires_trusted_origin"
                ),
                "smoke": smoke,
                "plan": plan.to_manifest(),
                "blueprint": blueprint.to_manifest(),
                "plan_history": plan_history,
                "selection_chain": selection_chain,
                "cells": cell_evidence,
                "confirmation": {
                    "results": confirmation_results,
                    "stress_failures": stress_failures,
                    "scope_narrowing": declared_scope,
                },
            }
            body = {
                "schema": PHASE1_DECISION_SCHEMA,
                "source_phase1_plan_id": plan.plan_id,
                "source_phase1_blueprint_id": blueprint.blueprint_id,
                "authorization_mode": authorization_mode,
                "decision": decision,
                "authorizes_phase2_scientific": authorizes_scientific,
                "authorizes_phase2_smoke": (
                    baseline_passed if smoke else authorizes_scientific
                ),
                "phase1_campaign_manifest_sha256": _canonical_sha256(campaign_manifest),
                "phase1_campaign_manifest": campaign_manifest,
                "phase1_transfer": build_phase1_transfer(campaign_manifest),
            }
            payload = {
                **body,
                "decision_id": content_id(body, prefix="phase1-decision"),
            }
            self.phase1_decision_from_manifest(payload)
            if (
                _sha256(self.plan_path) != plan_sha256
                or _sha256(self.blueprint_path) != blueprint_sha256
                or _sha256(self.journal_path) != journal_sha256
            ):
                raise CampaignError("Phase-1 campaign changed while freezing decision")
            for record in records.values():
                for artifact in record.artifacts:
                    artifact.verify(self.root)
            for ref in selection_refs:
                ref.verify(self.root)
            for item in plan_history:
                path = self.plans_dir / f"{_slug(item['plan_id'])}.json"
                if "sha256:" + _sha256(path) != item["sha256"]:
                    raise CampaignError("Phase-1 plan history changed during freeze")
            destination = _campaign_output_path(
                self.root,
                out,
                self.root / "decisions" / "phase2-authorization.json",
            )
            _write_immutable_json(destination, payload)
            return payload

    def _confirmation_noninferiority_evidence(
        self,
        source_cells: Sequence[CellSpec],
        records: Mapping[str, CampaignRecord],
        policy: SelectionPolicy,
        *,
        smoke: bool,
    ) -> dict[str, Any]:
        """Authenticate the untouched scalar-RMS confirmation gate seedwise."""

        def frozen_contract_value(name: str) -> Any:
            values = [cell.decision_map.get(name) for cell in source_cells]
            if not values or any(value != values[0] for value in values[1:]):
                raise CampaignError(
                    f"confirmation cells disagree on frozen contract {name!r}"
                )
            return values[0]

        score_degradation_max = frozen_contract_value(
            "qualification.confirmation_score_degradation_max"
        )
        score_sensitivity = frozen_contract_value(
            "qualification.confirmation_score_degradation_sensitivity"
        )
        threshold_basis = frozen_contract_value(
            "qualification.confirmation_threshold_basis"
        )
        if (
            score_degradation_max != PHASE2_CONFIRMATION_SCORE_DEGRADATION_MAX
            or score_sensitivity != PHASE2_CONFIRMATION_SCORE_DEGRADATION_SENSITIVITY
            or threshold_basis != PHASE2_CONFIRMATION_THRESHOLD_BASIS
        ):
            raise CampaignError(
                "confirmation cells lack the canonical preregistered score guard"
            )
        plan_cells = {cell.cell_id: cell for cell in self.plan.cells}
        per_seed: list[dict[str, Any]] = []
        for cell in sorted(source_cells, key=lambda item: item.seed):
            record = records[cell.cell_id]
            qualification = self._qualification_payload(record)
            if qualification is None:  # pragma: no cover - prevalidated above
                raise CampaignError("confirmation finalist lacks qualification")
            parent_ids = cell.decision_map.get("selection.parent_cell_ids", ())
            if not isinstance(parent_ids, (tuple, list)):
                raise CampaignError(
                    "confirmation finalist has malformed parent binding"
                )
            try:
                parents = [plan_cells[str(parent_id)] for parent_id in parent_ids]
            except KeyError as exc:
                raise CampaignError(
                    "confirmation finalist names a parent outside Phase 2"
                ) from exc
            matching = [parent for parent in parents if parent.seed == cell.seed]
            if len(matching) != 1:
                raise CampaignError(
                    "confirmation finalist requires one same-seed development parent"
                )
            parent = matching[0]
            parent_record = records.get(parent.cell_id)
            if parent_record is None:
                raise CampaignError("confirmation parent lacks campaign evidence")
            parent_qualification = self._qualification_payload(parent_record)
            if parent_qualification is None:
                raise CampaignError("confirmation parent lacks qualification")
            if smoke:
                protocol_passed = bool(
                    qualification.get("selection_eligibility_mode")
                    == "smoke_protocol_only"
                    and qualification.get("selection_eligible_for_protocol_test")
                    is True
                )
                per_seed.append(
                    {
                        "seed": cell.seed,
                        "cell_id": cell.cell_id,
                        "parent_cell_id": parent.cell_id,
                        "qualification_sha256": (
                            "sha256:" + record.artifact_map["qualification"].sha256
                        ),
                        "parent_qualification_sha256": (
                            "sha256:"
                            + parent_record.artifact_map["qualification"].sha256
                        ),
                        "confirmation_score": None,
                        "parent_score": None,
                        "score_degradation": None,
                        "sharing_guard": None,
                        "qualification_passed": protocol_passed,
                        "score_noninferiority_passed": None,
                        "sharing_guard_passed": None,
                        "passed": protocol_passed,
                    }
                )
                continue
            selection_metrics = qualification.get("selection_metrics")
            parent_metrics = parent_qualification.get("selection_metrics")
            if not isinstance(selection_metrics, Mapping) or not isinstance(
                parent_metrics, Mapping
            ):
                raise CampaignError(
                    "confirmation candidate or parent lacks selection metrics"
                )
            confirmation_score = self._policy_metric(selection_metrics, policy)
            parent_score = self._policy_metric(parent_metrics, policy)
            degradation = parent_score - confirmation_score
            sharing_guard = (
                self._sharing_guard_result(cell, selection_metrics, policy)
                if policy.require_sharing_guard
                else None
            )
            qualification_passed = bool(
                qualification.get("scientific_outcome", {}).get("passed") is True
                and qualification.get("promotion_eligible") is True
                and qualification.get("selection_eligibility_mode")
                == "scientific_promotion"
            )
            score_passed = bool(
                degradation <= score_degradation_max
                or math.isclose(
                    degradation,
                    score_degradation_max,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            )
            sharing_passed = (
                sharing_guard.get("passed") is True
                if sharing_guard is not None
                else None
            )
            passed = qualification_passed and score_passed and (
                not policy.require_sharing_guard or sharing_passed is True
            )
            per_seed.append(
                {
                    "seed": cell.seed,
                    "cell_id": cell.cell_id,
                    "parent_cell_id": parent.cell_id,
                    "qualification_sha256": (
                        "sha256:" + record.artifact_map["qualification"].sha256
                    ),
                    "parent_qualification_sha256": (
                        "sha256:" + parent_record.artifact_map["qualification"].sha256
                    ),
                    "confirmation_score": confirmation_score,
                    "parent_score": parent_score,
                    "score_degradation": degradation,
                    "sharing_guard": sharing_guard,
                    "qualification_passed": qualification_passed,
                    "score_noninferiority_passed": score_passed,
                    "sharing_guard_passed": sharing_passed,
                    "passed": passed,
                }
            )
        return {
            "mode": "smoke_protocol_only" if smoke else "scientific_confirmation",
            "metric_path": policy.metric_path,
            "direction": policy.direction,
            "score_degradation_max": score_degradation_max,
            "score_degradation_threshold_basis": threshold_basis,
            "score_degradation_sensitivity": (
                _confirmation_score_sensitivity_payload(
                    per_seed,
                    score_sensitivity,
                    smoke=smoke,
                    require_sharing_guard=policy.require_sharing_guard,
                )
            ),
            "policy": policy.to_dict(),
            "per_seed": per_seed,
            "passed": all(item["passed"] is True for item in per_seed),
        }

    @staticmethod
    def _projected_scientific_configurations(
        decision: FrozenPanelDecision, *, smoke: bool
    ) -> dict[str, str]:
        """Fingerprint operational Phase-3 method choices after projection.

        Slot/contest/factor labels and random replicate streams are excluded;
        executable model, objective, optimizer, auxiliary, inference,
        precision, training, and implementation choices remain.  Thus two
        differently named Phase-2 lineages cannot occupy two publication slots
        when they project to the same actual scientific configuration.
        """

        projection_seeds = (0,) if smoke else (0, 1, 2, 3, 4)
        projected = build_phase3_plan(
            seeds=projection_seeds,
            smoke=smoke,
            panel_decision=decision,
        )
        prefixes = (
            "model.",
            "objective.",
            "optimizer.",
            "auxiliary.",
            "regularizer.",
            "inference.",
            "training.",
            "precision.",
            "implementation.",
        )
        result: dict[str, str] = {}
        for entry in decision.entries:
            expected_name = f"phase3.frozen_panel.{entry.panel_slot}.s0"
            matches = [cell for cell in projected.cells if cell.name == expected_name]
            if len(matches) != 1:
                raise CampaignError(
                    f"cannot resolve projected Phase-3 slot {entry.panel_slot!r}"
                )
            cell = matches[0]
            scientific_configuration = {
                "decisions": [
                    {"name": item.name, "value": item.value}
                    for item in sorted(cell.decisions, key=lambda item: item.name)
                    if item.domain.value == "scientific"
                    and item.name.startswith(prefixes)
                    and not item.name.startswith("random.")
                ]
            }
            result[entry.panel_slot] = content_id(
                scientific_configuration, prefix="phase3-scientific-config"
            )
        return result

    @classmethod
    def _duplicate_projected_configurations(
        cls, decision: FrozenPanelDecision, *, smoke: bool
    ) -> list[dict[str, Any]]:
        fingerprints = cls._projected_scientific_configurations(decision, smoke=smoke)
        by_fingerprint: dict[str, list[str]] = {}
        for slot, fingerprint in fingerprints.items():
            by_fingerprint.setdefault(fingerprint, []).append(slot)
        return [
            {
                "scientific_configuration_id": fingerprint,
                "panel_slots": sorted(slots),
            }
            for fingerprint, slots in sorted(by_fingerprint.items())
            if len(slots) > 1
        ]

    def freeze_panel(
        self,
        *,
        out: str | Path | None = None,
    ) -> dict[str, Any]:
        """Freeze the exact qualified Phase-2 campaign into a Phase-3 panel.

        This is intentionally stricter than ordinary selection.  Every cell in
        the fully materialized blueprint must have complete, live qualification
        evidence; every adaptive extension is replayed from its original
        selection artifact against the *current* complete ranked universe; and
        a scientific scalar-RMS confirmation finalist must pass its outcome
        for every declared seed.  A uniformly smoke campaign may freeze only
        a protocol-test panel, which remains permanently barred from non-smoke
        Phase 3.  Comparators come from their independently calibrated family
        revisits, never from uncalibrated development anchors.
        """

        durable_mkdir(self.root, parents=True, exist_ok=True)
        with self._registration_mutation():
            self._reconcile_plan_projection()
            if not self.plan_path.is_file() or not self.blueprint_path.is_file():
                raise CampaignError(
                    "panel freeze requires a registered Phase-2 plan and blueprint"
                )
            if not self.phase1_decision_path.is_file():
                raise CampaignError(
                    "panel freeze requires the Phase-1 decision that authorized Phase 2"
                )
            plan_path_sha256 = _sha256(self.plan_path)
            blueprint_path_sha256 = _sha256(self.blueprint_path)
            phase1_decision_path_sha256 = _sha256(self.phase1_decision_path)
            journal_path_sha256 = (
                _sha256(self.journal_path) if self.journal_path.is_file() else None
            )
            plan_manifest = _read_json(self.plan_path)
            blueprint_manifest = _read_json(self.blueprint_path)
            phase1_decision_manifest = _read_json(self.phase1_decision_path)
            try:
                plan = StudyPlan.from_manifest(plan_manifest)
                blueprint = Phase2Blueprint.from_manifest(blueprint_manifest)
            except (KeyError, TypeError, ValueError, StudyError) as exc:
                raise CampaignError(
                    f"invalid Phase-2 plan or blueprint manifest: {exc}"
                ) from exc
            if plan.phase.value != "phase2":
                raise CampaignError(
                    "only a Phase-2 campaign can freeze a Phase-3 panel"
                )
            phase1_decision = self.phase1_decision_from_manifest(
                phase1_decision_manifest
            )
            expected_bound_blueprint = build_phase2_blueprint(
                blueprint.seeds,
                smoke=(
                    blueprint.initial_stage.cells[0].decision_map.get("runtime.smoke")
                    is True
                ),
                phase1_decision=phase1_decision,
            )
            if blueprint != expected_bound_blueprint:
                raise CampaignError(
                    "Phase-2 blueprint no longer matches its bound Phase-1 transfer"
                )
            if canonical_json(plan_manifest) != canonical_json(plan.to_manifest()):
                raise CampaignError("active Phase-2 plan manifest is not canonical")
            if canonical_json(blueprint_manifest) != canonical_json(
                blueprint.to_manifest()
            ):
                raise CampaignError(
                    "active Phase-2 blueprint manifest is not canonical"
                )
            expected_stage_names = (
                blueprint.initial_stage.name,
                *(round_spec.name for round_spec in blueprint.rounds),
                *(
                    stage_name
                    for family in blueprint.comparator_families
                    for stage_name in (
                        *(round_spec.name for round_spec in family.rounds),
                        family.revisit.name,
                    )
                ),
            )
            observed_stage_names = tuple(stage.name for stage in plan.stages)
            if (
                len(observed_stage_names) != len(set(observed_stage_names))
                or set(observed_stage_names) != set(expected_stage_names)
                or observed_stage_names[0] != blueprint.initial_stage.name
            ):
                raise CampaignError(
                    "Phase-2 plan is not the fully materialized blueprint"
                )
            if plan.stages[0] != blueprint.initial_stage:
                raise CampaignError("Phase-2 initial stage differs from its blueprint")
            all_events = self._events_cached()
            known_cell_ids = {cell.cell_id for cell in plan.cells}
            unknown_transition_cells = {
                str(event.get("cell_id"))
                for event in all_events
                if event.get("event") == "transition"
                and event.get("cell_id") not in known_cell_ids
            }
            if unknown_transition_cells:
                raise CampaignError(
                    "Phase-2 journal contains transitions outside the exact plan: "
                    + canonical_json(sorted(unknown_transition_cells))
                )

            # Validate every recorded artifact, including non-gate reports, so
            # the campaign digest cannot silently bless a partially stale cell.
            records: dict[str, CampaignRecord] = {}
            cell_evidence: list[dict[str, Any]] = []
            for cell in plan.cells:
                record = self.record(cell.cell_id)
                if record.state not in {RunState.QUALIFIED, RunState.PROMOTED}:
                    raise CampaignError(
                        "Phase-2 panel freeze requires every materialized cell to be qualified; "
                        f"{cell.cell_id} is {record.state.value}"
                    )
                self._validate_artifact_gate(
                    cell.cell_id, RunState.QUALIFIED, record.artifact_map
                )
                for artifact in record.artifacts:
                    artifact.verify(self.root)
                records[cell.cell_id] = record
                cell_evidence.append(
                    {
                        "cell_id": cell.cell_id,
                        "candidate_id": cell.candidate_id,
                        "stage": cell.stage,
                        "seed": cell.seed,
                        "recipe_name": cell.recipe_name,
                        "recipe_id": cell.recipe_id,
                        "cell": cell.to_manifest(),
                        "state": record.state.value,
                        "qualification": self._qualification_payload(record),
                        "artifacts": [
                            artifact.to_dict() for artifact in record.artifacts
                        ],
                    }
                )

            smoke = bool(plan.cells[0].decision_map["runtime.smoke"])
            if smoke:
                if phase1_decision.get("authorizes_phase2_smoke") is not True:
                    raise CampaignError(
                        "the bound Phase-1 decision does not authorize smoke Phase 2"
                    )
            elif (
                phase1_decision.get("authorization_mode") != "scientific_go"
                or phase1_decision.get("authorizes_phase2_scientific") is not True
            ):
                raise CampaignError(
                    "scientific Phase 2 lacks a bound Phase-1 go decision"
                )
            expected = StudyPlan(
                "phase2_small_real_prefix_1_smoke"
                if smoke
                else "phase2_small_real_prefix_1",
                plan.phase,
                (blueprint.initial_stage,),
            )
            extension_events = tuple(
                event for event in all_events if event.get("event") == "plan_extension"
            )
            if len(extension_events) != len(expected_stage_names) - 1:
                raise CampaignError(
                    "Phase-2 journal does not contain exactly one extension per "
                    "main, family, and revisit stage"
                )
            selection_chain: list[dict[str, Any]] = []
            main_selection_chain: list[dict[str, Any]] = []
            family_selection_chains: dict[str, list[dict[str, Any]]] = {
                family.name: [] for family in blueprint.comparator_families
            }
            family_nomination_selections: dict[str, tuple[FrozenSelection, ...]] = {}
            family_nominations: list[dict[str, Any]] = []
            ranked_universes: list[dict[str, Any]] = []
            plan_history: list[dict[str, str]] = []
            selection_refs: list[ArtifactRef] = []
            main_round_index = 0

            def verify_plan_history(prefix: StudyPlan) -> None:
                history_path = self.plans_dir / f"{_slug(prefix.plan_id)}.json"
                if not history_path.is_file():
                    raise CampaignError(
                        f"missing immutable plan-history artifact for {prefix.plan_id}"
                    )
                history_payload = _read_json(history_path)
                if canonical_json(history_payload) != canonical_json(
                    prefix.to_manifest()
                ):
                    raise CampaignError(
                        f"plan-history artifact differs for {prefix.plan_id}"
                    )
                plan_history.append(
                    {
                        "plan_id": prefix.plan_id,
                        "sha256": "sha256:" + _sha256(history_path),
                    }
                )

            def stage_selection_from_event(
                event: Mapping[str, Any],
                source_stage: Any,
                policy: SelectionPolicy,
                source_plan_id: str,
            ) -> tuple[FrozenSelection, dict[str, Any], ArtifactRef]:
                try:
                    event_artifacts = tuple(
                        ArtifactRef.from_dict(item)
                        for item in event.get("artifacts", ())
                    )
                except (ArtifactError, KeyError, TypeError, ValueError) as exc:
                    raise CampaignError(
                        f"invalid plan-extension artifact: {exc}"
                    ) from exc
                if (
                    len(event_artifacts) != 1
                    or event_artifacts[0].kind != "stage_selection"
                ):
                    raise CampaignError(
                        "a selection extension must bind exactly one stage-selection artifact"
                    )
                selection_ref = event_artifacts[0]
                selection_ref.verify(self.root)
                selection_payload = _read_json(selection_ref.resolve(self.root))
                live_payload = self._selection_payload(
                    source_stage.name,
                    source_plan_id=source_plan_id,
                    policy_override=policy,
                )
                if canonical_json(selection_payload) != canonical_json(live_payload):
                    raise CampaignError(
                        f"selection for {source_stage.name} is stale or incomplete"
                    )
                selected_payloads = selection_payload.get("selected")
                if not isinstance(selected_payloads, list):
                    raise CampaignError("selection artifact has no selected candidates")
                event_selection_id = (event.get("metadata") or {}).get("selection_id")
                try:
                    chosen = [
                        FrozenSelection.from_dict(item)
                        for item in selected_payloads
                        if item.get("selection_id") == event_selection_id
                    ]
                except (KeyError, TypeError, ValueError, StudyError) as exc:
                    raise CampaignError(
                        f"invalid frozen stage selection: {exc}"
                    ) from exc
                if len(chosen) != 1:
                    raise CampaignError(
                        "extension event does not identify exactly one frozen candidate"
                    )
                selection_refs.append(selection_ref)
                return chosen[0], selection_payload, selection_ref

            def bind_stage_selection(
                *,
                source_plan_id: str,
                source_stage: Any,
                target_plan: StudyPlan | None,
                target_stage: str | None,
                selection: FrozenSelection,
                selection_payload: Mapping[str, Any],
                selection_ref: ArtifactRef | None,
                branch: str,
                family_name: str | None = None,
            ) -> dict[str, Any]:
                chain_item = {
                    "source_plan_id": source_plan_id,
                    "source_stage": source_stage.name,
                    "target_plan_id": (
                        None if target_plan is None else target_plan.plan_id
                    ),
                    "target_stage": target_stage,
                    "branch": branch,
                    "family_name": family_name,
                    "selection_id": selection.selection_id,
                    "selection_artifact_sha256": (
                        None
                        if selection_ref is None
                        else "sha256:" + selection_ref.sha256
                    ),
                    "selection_artifact_sha256_semantics": (
                        "not_applicable"
                        if selection_ref is None
                        else "opaque_historical_commitment_requires_trusted_origin"
                    ),
                    "selection_universe_sha256": (selection.selection_universe_sha256),
                    "policy_id": selection.policy_id,
                    "candidate_id": selection.candidate_id,
                    "cell_ids": list(selection.cell_ids),
                    "qualification_sha256s": list(selection.qualification_sha256s),
                }
                selection_chain.append(chain_item)
                ranked_universes.append(
                    {
                        "schema": selection_payload["schema"],
                        "source_plan_id": source_plan_id,
                        "source_stage": source_stage.name,
                        "phase": selection_payload["phase"],
                        "policy": selection_payload["policy"],
                        "selection_universe_sha256": selection_payload[
                            "selection_universe_sha256"
                        ],
                        "ranked_candidates": selection_payload["ranked_candidates"],
                        "excluded_candidates": selection_payload["excluded_candidates"],
                        "threshold_sensitivity": selection_payload[
                            "threshold_sensitivity"
                        ],
                        "smoke": selection_payload["smoke"],
                        "smoke_protocol_only": selection_payload["smoke_protocol_only"],
                        "selection_mode": selection_payload["selection_mode"],
                    }
                )
                return chain_item

            verify_plan_history(expected)
            for event in extension_events:
                metadata = event.get("metadata")
                if not isinstance(metadata, Mapping):
                    raise CampaignError("plan-extension metadata must be an object")
                branch = metadata.get("branch")
                if branch is None:
                    if main_round_index >= len(blueprint.rounds):
                        raise CampaignError("journal has an extra main-chain extension")
                    round_spec = blueprint.rounds[main_round_index]
                    source_stage = expected.stages[-1]
                    if (
                        source_stage.name != round_spec.source_stage
                        or source_stage.selection_policy is None
                    ):
                        raise CampaignError(
                            "Phase-2 main selection chain has the wrong source stage"
                        )
                    selection, selection_payload, selection_ref = (
                        stage_selection_from_event(
                            event,
                            source_stage,
                            source_stage.selection_policy,
                            expected.plan_id,
                        )
                    )
                    extended = materialize_child_plan(expected, blueprint, selection)
                    expected_metadata = {
                        "previous_plan_id": expected.plan_id,
                        "plan_id": extended.plan_id,
                        "stage": round_spec.name,
                        "selection_id": selection.selection_id,
                    }
                    chain_item = bind_stage_selection(
                        source_plan_id=expected.plan_id,
                        source_stage=source_stage,
                        target_plan=extended,
                        target_stage=round_spec.name,
                        selection=selection,
                        selection_payload=selection_payload,
                        selection_ref=selection_ref,
                        branch="main",
                    )
                    main_selection_chain.append(chain_item)
                    main_round_index += 1
                elif branch == "comparator_family":
                    family_name = str(metadata.get("family_name", ""))
                    family = self._family_blueprint(blueprint, family_name)
                    event_artifacts = event.get("artifacts", ())
                    if (
                        not isinstance(event_artifacts, list)
                        or len(event_artifacts) != 1
                    ):
                        raise CampaignError(
                            "family extension lacks its selection artifact"
                        )
                    preview_ref = ArtifactRef.from_dict(event_artifacts[0])
                    preview_payload = _read_json(preview_ref.resolve(self.root))
                    source_name = str(preview_payload.get("source_stage", ""))
                    source_matches = [
                        stage for stage in expected.stages if stage.name == source_name
                    ]
                    if len(source_matches) != 1:
                        raise CampaignError("family extension source stage is absent")
                    source_stage = source_matches[0]
                    policy = (
                        family.root_selection_policy
                        if source_stage.name == blueprint.initial_stage.name
                        else source_stage.selection_policy
                    )
                    if policy is None:
                        raise CampaignError("family extension source is not selectable")
                    selection, selection_payload, selection_ref = (
                        stage_selection_from_event(
                            event, source_stage, policy, expected.plan_id
                        )
                    )
                    extended = materialize_family_child_plan(
                        expected, blueprint, family_name, selection
                    )
                    expected_metadata = {
                        "previous_plan_id": expected.plan_id,
                        "plan_id": extended.plan_id,
                        "stage": extended.stages[-1].name,
                        "selection_id": selection.selection_id,
                        "branch": "comparator_family",
                        "family_name": family_name,
                    }
                    chain_item = bind_stage_selection(
                        source_plan_id=expected.plan_id,
                        source_stage=source_stage,
                        target_plan=extended,
                        target_stage=extended.stages[-1].name,
                        selection=selection,
                        selection_payload=selection_payload,
                        selection_ref=selection_ref,
                        branch="comparator_family",
                        family_name=family_name,
                    )
                    family_selection_chains[family_name].append(chain_item)
                elif branch == "comparator_family_revisit":
                    family_name = str(metadata.get("family_name", ""))
                    family = self._family_blueprint(blueprint, family_name)
                    try:
                        event_artifacts = tuple(
                            ArtifactRef.from_dict(item)
                            for item in event.get("artifacts", ())
                        )
                    except (ArtifactError, KeyError, TypeError, ValueError) as exc:
                        raise CampaignError(
                            f"invalid family nomination artifact: {exc}"
                        ) from exc
                    if (
                        len(event_artifacts) != 1
                        or event_artifacts[0].kind != "family_nomination"
                    ):
                        raise CampaignError(
                            "family revisit must bind exactly one nomination artifact"
                        )
                    nomination_ref = event_artifacts[0]
                    nomination_ref.verify(self.root)
                    nomination_payload = _read_json(nomination_ref.resolve(self.root))
                    live_nomination = self._family_nomination_payload(
                        family_name, source_plan_id=expected.plan_id
                    )
                    if canonical_json(nomination_payload) != canonical_json(
                        live_nomination
                    ):
                        raise CampaignError(
                            f"family nomination for {family_name} is stale or incomplete"
                        )
                    try:
                        selections = tuple(
                            FrozenSelection.from_dict(item)
                            for item in nomination_payload["selected"]
                        )
                    except (KeyError, TypeError, ValueError, StudyError) as exc:
                        raise CampaignError(
                            f"invalid family nomination selections: {exc}"
                        ) from exc
                    extended = materialize_family_revisit_plan(
                        expected, blueprint, family_name, selections
                    )
                    expected_metadata = {
                        "previous_plan_id": expected.plan_id,
                        "plan_id": extended.plan_id,
                        "stage": family.revisit.name,
                        "branch": "comparator_family_revisit",
                        "family_name": family_name,
                        "nomination_id": nomination_payload["nomination_id"],
                        "selection_ids": [
                            selection.selection_id for selection in selections
                        ],
                    }
                    selection_refs.append(nomination_ref)
                    family_nomination_selections[family_name] = selections
                    family_nominations.append(
                        {
                            "source_plan_id": expected.plan_id,
                            "target_plan_id": extended.plan_id,
                            "family_name": family_name,
                            "family_id": family.family_id,
                            "source_stage": family.revisit.name,
                            "source_rounds": list(family.revisit.source_rounds),
                            "nomination_id": nomination_payload["nomination_id"],
                            "nomination_artifact_sha256": (
                                "sha256:" + nomination_ref.sha256
                            ),
                            "policy": nomination_payload["policy"],
                            "selection_universe_sha256": nomination_payload[
                                "selection_universe_sha256"
                            ],
                            "ranked_candidates": nomination_payload[
                                "ranked_candidates"
                            ],
                            "excluded_candidates": nomination_payload[
                                "excluded_candidates"
                            ],
                            "source_threshold_sensitivity": nomination_payload[
                                "source_threshold_sensitivity"
                            ],
                            "selected": nomination_payload["selected"],
                            "nomination_payload": nomination_payload,
                        }
                    )
                else:
                    raise CampaignError(
                        f"unknown Phase-2 plan-extension branch {branch!r}"
                    )
                if metadata != expected_metadata:
                    raise CampaignError(
                        "Phase-2 plan-extension journal binding mismatch"
                    )
                if plan.stages[: len(extended.stages)] != extended.stages:
                    raise CampaignError(
                        f"materialized stage {extended.stages[-1].name} differs "
                        "from blueprint replay"
                    )
                expected = extended
                verify_plan_history(expected)
            if (
                expected != plan
                or main_round_index != len(blueprint.rounds)
                or set(family_nomination_selections)
                != {family.name for family in blueprint.comparator_families}
            ):
                raise CampaignError(
                    "active Phase-2 plan is not the exact final replayed blueprint"
                )

            panel_blueprint = build_phase3_blueprint(smoke=smoke)
            stages_by_name = {stage.name: stage for stage in plan.stages}
            final_stage = stages_by_name[blueprint.rounds[-1].name]
            chain_ids = tuple(item["selection_id"] for item in main_selection_chain)
            family_final_selections: dict[str, FrozenSelection] = {}
            family_final_payloads: dict[str, dict[str, Any]] = {}
            for family in blueprint.comparator_families:
                revisit_stage = stages_by_name[family.revisit.name]
                if revisit_stage.selection_policy != family.revisit.selection_policy:
                    raise CampaignError(
                        f"family revisit policy differs for {family.name}"
                    )
                selection_payload = self._selection_payload(
                    revisit_stage.name,
                    source_plan_id=plan.plan_id,
                )
                selected_payloads = selection_payload.get("selected")
                if (
                    not isinstance(selected_payloads, list)
                    or len(selected_payloads) != 1
                ):
                    raise CampaignError(
                        f"family revisit {family.name} does not select exactly one finalist"
                    )
                selection = FrozenSelection.from_dict(selected_payloads[0])
                family_final_selections[family.name] = selection
                family_final_payloads[family.name] = selection_payload
                chain_item = bind_stage_selection(
                    source_plan_id=plan.plan_id,
                    source_stage=revisit_stage,
                    target_plan=None,
                    target_stage=None,
                    selection=selection,
                    selection_payload=selection_payload,
                    selection_ref=None,
                    branch="comparator_family_final",
                    family_name=family.name,
                )
                family_selection_chains[family.name].append(chain_item)
            entries: list[FrozenPanelEntry] = []
            confirmation_noninferiority: dict[str, Any] | None = None
            for slot in panel_blueprint.panel_slots:
                if slot.role == "selected_finalist":
                    source_cells = tuple(
                        sorted(
                            (
                                cell
                                for cell in final_stage.cells
                                if cell.decision_map["data.normalization"]
                                == "scalar_rms"
                            ),
                            key=lambda cell: cell.seed,
                        )
                    )
                    if (
                        tuple(cell.seed for cell in source_cells) != blueprint.seeds
                        or len({cell.candidate_id for cell in source_cells}) != 1
                        or any(
                            cell.decision_map["evaluation.split"] != "confirmation"
                            for cell in source_cells
                        )
                    ):
                        raise CampaignError(
                            "scalar-RMS confirmation finalist is not one seed-complete candidate"
                        )
                    upstream_chains = {
                        tuple(cell.decision_map["selection.upstream_selection_ids"])
                        for cell in source_cells
                    }
                    if upstream_chains != {chain_ids}:
                        raise CampaignError(
                            "confirmation finalist does not bind the exact complete selection chain"
                        )
                    for cell in source_cells:
                        payload = self._qualification_payload(records[cell.cell_id])
                        if not smoke and (
                            payload is None
                            or payload.get("scientific_outcome", {}).get("passed")
                            is not True
                        ):
                            raise CampaignError(
                                "every scalar-RMS finalist seed must pass its scientific outcome"
                            )
                    confirmation_source_stage = stages_by_name.get(
                        blueprint.rounds[-1].source_stage
                    )
                    if (
                        confirmation_source_stage is None
                        or confirmation_source_stage.selection_policy is None
                    ):
                        raise CampaignError(
                            "confirmation round lacks its frozen development policy"
                        )
                    confirmation_noninferiority = (
                        self._confirmation_noninferiority_evidence(
                            source_cells,
                            records,
                            confirmation_source_stage.selection_policy,
                            smoke=smoke,
                        )
                    )
                    if confirmation_noninferiority.get("passed") is not True:
                        raise CampaignError(
                            "scalar-RMS confirmation failed seedwise score/sharing noninferiority"
                        )
                    entries.append(
                        FrozenPanelEntry.from_cells(
                            panel_slot=slot.name,
                            role=slot.role,
                            source_cells=source_cells,
                            selection_ids=chain_ids,
                            qualification_sha256s=tuple(
                                "sha256:"
                                + records[cell.cell_id]
                                .artifact_map["qualification"]
                                .sha256
                                for cell in source_cells
                            ),
                            confirmation_sha256s=tuple(
                                "sha256:"
                                + records[cell.cell_id]
                                .artifact_map["evaluation"]
                                .sha256
                                for cell in source_cells
                            ),
                        )
                    )
                    continue

                family = self._family_blueprint(blueprint, slot.name)
                final_selection = family_final_selections[family.name]
                revisit_stage = stages_by_name[family.revisit.name]
                cells_by_id = {cell.cell_id: cell for cell in revisit_stage.cells}
                source_cells = tuple(
                    sorted(
                        (cells_by_id[cell_id] for cell_id in final_selection.cell_ids),
                        key=lambda cell: cell.seed,
                    )
                )
                if (
                    tuple(cell.seed for cell in source_cells) != blueprint.seeds
                    or len({cell.candidate_id for cell in source_cells}) != 1
                    or any(
                        cell.stage != family.revisit.name
                        or cell.decision_map["evaluation.split"] != "development"
                        or cell.decision_map.get("selection.comparator_family_name")
                        != family.name
                        or cell.decision_map.get(
                            "selection.comparator_family_blueprint_id"
                        )
                        != family.family_id
                        or cell.decision_map.get("selection.family_root_recipe_id")
                        != family.root_recipe_id
                        for cell in source_cells
                    )
                ):
                    raise CampaignError(
                        f"calibrated comparator {slot.name} is missing or lineage-mismatched"
                    )
                nomination_ids = tuple(
                    selection.selection_id
                    for selection in family_nomination_selections[family.name]
                )
                standard_family_ids = tuple(
                    item["selection_id"]
                    for item in family_selection_chains[family.name]
                )
                family_chain_ids = tuple(
                    dict.fromkeys(
                        (
                            *standard_family_ids[:-1],
                            *nomination_ids,
                            standard_family_ids[-1],
                        )
                    )
                )
                entries.append(
                    FrozenPanelEntry.from_cells(
                        panel_slot=slot.name,
                        role=slot.role,
                        source_cells=source_cells,
                        selection_ids=family_chain_ids,
                        qualification_sha256s=tuple(
                            "sha256:"
                            + records[cell.cell_id].artifact_map["qualification"].sha256
                            for cell in source_cells
                        ),
                    )
                )

            if confirmation_noninferiority is None:
                raise CampaignError("panel lacks scalar-RMS confirmation evidence")

            substitution_evidence: list[dict[str, Any]] = []
            slot_order = [slot.name for slot in panel_blueprint.panel_slots]
            slot_policies = {
                slot.name: getattr(slot, "duplicate_policy", "fail")
                for slot in panel_blueprint.panel_slots
            }

            def provisional_decision(
                candidate_entries: Sequence[FrozenPanelEntry],
            ) -> FrozenPanelDecision:
                return FrozenPanelDecision(
                    source_phase2_plan_id=plan.plan_id,
                    source_phase2_blueprint_id=blueprint.blueprint_id,
                    phase2_campaign_manifest_sha256="sha256:" + "0" * 64,
                    selection_universe_sha256="sha256:" + "0" * 64,
                    entries=tuple(candidate_entries),
                )

            while True:
                provisional = provisional_decision(entries)
                duplicate_groups = self._duplicate_projected_configurations(
                    provisional, smoke=smoke
                )
                if not duplicate_groups:
                    break
                group = min(
                    duplicate_groups,
                    key=lambda item: min(
                        slot_order.index(slot) for slot in item["panel_slots"]
                    ),
                )
                group_slots = sorted(group["panel_slots"], key=slot_order.index)
                substitute_slot = next(
                    (
                        slot
                        for slot in group_slots[1:]
                        if slot_policies.get(slot) == "next_ranked_nonduplicate"
                    ),
                    None,
                )
                if substitute_slot is None:
                    raise CampaignError(
                        "Phase-3 panel contains duplicate projected scientific "
                        "configurations without a preregistered "
                        "next_ranked_nonduplicate policy: " + canonical_json(group)
                    )
                family = self._family_blueprint(blueprint, substitute_slot)
                revisit_stage = stages_by_name[family.revisit.name]
                payload = family_final_payloads[substitute_slot]
                policy = revisit_stage.selection_policy
                if policy is None:
                    raise CampaignError("duplicate substitute source is not selectable")
                current_selection = family_final_selections[substitute_slot]
                current_entry = next(
                    entry for entry in entries if entry.panel_slot == substitute_slot
                )
                cells_by_id = {cell.cell_id: cell for cell in revisit_stage.cells}
                accepted: (
                    tuple[
                        FrozenSelection,
                        FrozenPanelEntry,
                        int,
                        dict[str, str],
                    ]
                    | None
                ) = None
                ranked_candidates = payload.get("ranked_candidates")
                if not isinstance(ranked_candidates, list):
                    raise CampaignError("duplicate substitute lacks a ranked universe")
                before_fingerprints = self._projected_scientific_configurations(
                    provisional, smoke=smoke
                )
                for rank, candidate in enumerate(ranked_candidates, start=1):
                    if not isinstance(candidate, Mapping):
                        raise CampaignError(
                            "ranked substitute candidate must be an object"
                        )
                    if candidate.get("candidate_id") == current_selection.candidate_id:
                        continue
                    observations = candidate.get("observations")
                    if not isinstance(observations, list):
                        raise CampaignError("ranked substitute lacks seed observations")
                    try:
                        candidate_cells = tuple(
                            cells_by_id[str(item["cell_id"])] for item in observations
                        )
                        alternate = FrozenSelection.from_cells(
                            policy,
                            candidate_cells,
                            [float(item["metric"]) for item in observations],
                            [
                                str(item["qualification_sha256"])
                                for item in observations
                            ],
                            str(payload["selection_universe_sha256"]),
                        )
                    except (KeyError, TypeError, ValueError, StudyError) as exc:
                        raise CampaignError(
                            f"invalid ranked duplicate substitute: {exc}"
                        ) from exc
                    replacement_ids = tuple(
                        alternate.selection_id
                        if selection_id == current_selection.selection_id
                        else selection_id
                        for selection_id in current_entry.selection_ids
                    )
                    replacement = FrozenPanelEntry.from_cells(
                        panel_slot=current_entry.panel_slot,
                        role=current_entry.role,
                        source_cells=candidate_cells,
                        selection_ids=replacement_ids,
                        qualification_sha256s=alternate.qualification_sha256s,
                    )
                    trial_entries = [
                        replacement if entry.panel_slot == substitute_slot else entry
                        for entry in entries
                    ]
                    trial_decision = provisional_decision(trial_entries)
                    trial_groups = self._duplicate_projected_configurations(
                        trial_decision, smoke=smoke
                    )
                    if any(
                        substitute_slot in item["panel_slots"] for item in trial_groups
                    ):
                        continue
                    accepted = (
                        alternate,
                        replacement,
                        rank,
                        self._projected_scientific_configurations(
                            trial_decision, smoke=smoke
                        ),
                    )
                    break
                if accepted is None:
                    raise CampaignError(
                        f"family {substitute_slot!r} has no ranked nonduplicate "
                        "16M revisit substitute"
                    )
                alternate, replacement, rank, after_fingerprints = accepted
                entries = [
                    replacement if entry.panel_slot == substitute_slot else entry
                    for entry in entries
                ]
                final_chain_item = family_selection_chains[substitute_slot][-1]
                if (
                    final_chain_item.get("selection_id")
                    != current_selection.selection_id
                ):
                    raise CampaignError(
                        "duplicate substitute chain is already inconsistent"
                    )
                final_chain_item.update(
                    {
                        "branch": "comparator_family_duplicate_substitute",
                        "selection_id": alternate.selection_id,
                        "candidate_id": alternate.candidate_id,
                        "cell_ids": list(alternate.cell_ids),
                        "qualification_sha256s": list(alternate.qualification_sha256s),
                    }
                )
                family_final_selections[substitute_slot] = alternate
                substitution_evidence.append(
                    {
                        "panel_slot": substitute_slot,
                        "policy": "next_ranked_nonduplicate",
                        "reason": "projected_scientific_configuration_duplicate",
                        "collided_panel_slots": group_slots,
                        "original_candidate_id": current_selection.candidate_id,
                        "original_selection_id": current_selection.selection_id,
                        "original_scientific_configuration_id": before_fingerprints[
                            substitute_slot
                        ],
                        "substitute_candidate_id": alternate.candidate_id,
                        "substitute_selection_id": alternate.selection_id,
                        "substitute_rank": rank,
                        "substitute_scientific_configuration_id": after_fingerprints[
                            substitute_slot
                        ],
                    }
                )
            ordered_entries = tuple(sorted(entries, key=lambda item: item.panel_slot))
            panel_candidate_ids = {
                entry.panel_slot: entry.source_candidate_id for entry in ordered_entries
            }
            universe_payload = {
                "schema": SELECTION_UNIVERSE_SCHEMA,
                "source_phase2_plan_id": plan.plan_id,
                "source_phase2_blueprint_id": blueprint.blueprint_id,
                "selection_chain": selection_chain,
                "main_selection_chain": main_selection_chain,
                "family_selection_chains": family_selection_chains,
                "family_nominations": family_nominations,
                "ranked_stage_universes": ranked_universes,
                "panel_source_candidate_ids": panel_candidate_ids,
                "phase1_decision_id": phase1_decision["decision_id"],
                "phase1_transfer_id": phase1_decision["phase1_transfer"]["transfer_id"],
                "confirmation_noninferiority": confirmation_noninferiority,
                "duplicate_substitutions": substitution_evidence,
            }
            campaign_manifest_payload = {
                "schema": PHASE2_CAMPAIGN_MANIFEST_SCHEMA,
                "source_phase2_plan_id": plan.plan_id,
                "source_phase2_blueprint_id": blueprint.blueprint_id,
                "plan_sha256": "sha256:" + plan_path_sha256,
                "blueprint_sha256": "sha256:" + blueprint_path_sha256,
                "journal_sha256": (
                    None
                    if journal_path_sha256 is None
                    else "sha256:" + journal_path_sha256
                ),
                "journal_sha256_semantics": (
                    "opaque_historical_commitment_requires_trusted_origin"
                ),
                "smoke": smoke,
                "phase1_decision_sha256": ("sha256:" + phase1_decision_path_sha256),
                "phase1_decision": phase1_decision,
                "phase1_transfer_id": phase1_decision["phase1_transfer"]["transfer_id"],
                "plan_history": plan_history,
                "selection_chain": selection_chain,
                "main_selection_chain": main_selection_chain,
                "family_selection_chains": family_selection_chains,
                "family_nominations": family_nominations,
                "confirmation_noninferiority": confirmation_noninferiority,
                "duplicate_substitutions": substitution_evidence,
                "cells": cell_evidence,
                "panel_entries": [entry.to_dict() for entry in ordered_entries],
            }
            decision = FrozenPanelDecision(
                source_phase2_plan_id=plan.plan_id,
                source_phase2_blueprint_id=blueprint.blueprint_id,
                phase2_campaign_manifest_sha256=_canonical_sha256(
                    campaign_manifest_payload
                ),
                selection_universe_sha256=_canonical_sha256(universe_payload),
                entries=ordered_entries,
            )
            duplicate_configurations = self._duplicate_projected_configurations(
                decision, smoke=smoke
            )
            if duplicate_configurations:
                raise CampaignError(
                    "Phase-3 panel contains duplicate projected scientific "
                    "configurations and no declared substitute: "
                    + canonical_json(duplicate_configurations)
                )
            payload = {
                **decision.to_dict(),
                "producer_schema": PANEL_DECISION_PRODUCER_SCHEMA,
                "phase2_campaign_manifest": campaign_manifest_payload,
                "selection_universe": universe_payload,
            }
            # Exercise the same verifier used by Phase-3 registration before
            # touching the decision path.
            self.panel_decision_from_manifest(payload)
            if (
                _sha256(self.plan_path) != plan_path_sha256
                or _sha256(self.blueprint_path) != blueprint_path_sha256
                or _sha256(self.phase1_decision_path) != phase1_decision_path_sha256
                or not self.journal_path.is_file()
                or _sha256(self.journal_path) != journal_path_sha256
            ):
                raise CampaignError(
                    "Phase-2 campaign changed while freezing the panel decision"
                )
            for record in records.values():
                for artifact in record.artifacts:
                    artifact.verify(self.root)
            for selection_ref in selection_refs:
                selection_ref.verify(self.root)
            for item in plan_history:
                history_path = self.plans_dir / f"{_slug(item['plan_id'])}.json"
                if "sha256:" + _sha256(history_path) != item["sha256"]:
                    raise CampaignError(
                        "Phase-2 plan history changed while freezing the panel"
                    )
            if (
                _sha256(self.plan_path) != plan_path_sha256
                or _sha256(self.blueprint_path) != blueprint_path_sha256
                or _sha256(self.phase1_decision_path) != phase1_decision_path_sha256
                or _sha256(self.journal_path) != journal_path_sha256
            ):
                raise CampaignError(
                    "Phase-2 campaign changed during final panel validation"
                )
            destination = _campaign_output_path(
                self.root,
                out,
                self.root / "decisions" / "phase3-panel.json",
            )
            _write_immutable_json(destination, payload)
            return payload

    @classmethod
    def _policy_metric(
        cls, payload: Mapping[str, Any], policy: SelectionPolicy
    ) -> float:
        value: Any = payload
        for component in policy.metric_path.split("."):
            if not isinstance(value, Mapping) or component not in value:
                raise ArtifactError(
                    f"bound selection metrics lack {policy.metric_path!r}"
                )
            value = value[component]
        if policy.map_key is not None:
            if not isinstance(value, Mapping) or policy.map_key not in value:
                raise ArtifactError(
                    f"selection metric lacks map key {policy.map_key!r}"
                )
            value = value[policy.map_key]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math_isfinite(value)
        ):
            raise ArtifactError("selection metric must be finite numeric")
        return float(value)

    def stage_open(self, stage_name: str) -> bool:
        plan = self.plan
        by_name = {stage.name: stage for stage in plan.stages}
        try:
            stage = by_name[stage_name]
        except KeyError as exc:
            raise CampaignError(f"unknown stage {stage_name!r}") from exc
        if not stage.depends_on:
            return True
        records_by_stage = {
            name: [self.record(cell.cell_id) for cell in by_name[name].cells]
            for name in stage.depends_on
        }
        gate = stage.gate
        for dependency, records in records_by_stage.items():
            integrity_complete: list[CampaignRecord] = []
            scientific_passed: list[CampaignRecord] = []
            protocol_eligible: list[CampaignRecord] = []
            for record in records:
                if record.state not in {RunState.QUALIFIED, RunState.PROMOTED}:
                    continue
                try:
                    self._validate_artifact_gate(
                        record.cell_id, RunState.QUALIFIED, record.artifact_map
                    )
                except ArtifactError:
                    continue
                integrity_complete.append(record)
                payload = self._qualification_payload(record)
                if (
                    payload is not None
                    and payload.get("scientific_outcome", {}).get("passed") is True
                ):
                    scientific_passed.append(record)
                if (
                    payload is not None
                    and payload.get("selection_eligibility_mode")
                    == "smoke_protocol_only"
                    and payload.get("selection_eligible_for_protocol_test") is True
                ):
                    protocol_eligible.append(record)
            if gate is not None and dependency == gate.source_stage:
                dependency_is_smoke = all(
                    cell.decision_map.get("runtime.smoke") is True
                    for cell in by_name[dependency].cells
                )
                successful = (
                    integrity_complete
                    if gate.basis == "integrity_complete"
                    else protocol_eligible
                    if dependency_is_smoke
                    else scientific_passed
                )
                if len(successful) < gate.minimum_count:
                    return False
            elif len(integrity_complete) != len(records):
                return False
        return True

    def runnable_cell_ids(
        self,
        *,
        include_failed: bool = False,
        include_resume_required: bool = False,
    ) -> tuple[str, ...]:
        runnable: list[str] = []
        for stage in self.plan.stages:
            if not self.stage_open(stage.name):
                continue
            for cell in stage.cells:
                state = self.record(cell.cell_id).state
                if state in {RunState.QUALIFIED, RunState.PROMOTED}:
                    continue
                if state is RunState.FAILED and not include_failed:
                    continue
                if state is RunState.RUNNING and not include_resume_required:
                    continue
                runnable.append(cell.cell_id)
        return tuple(runnable)

    def status(self) -> dict[str, Any]:
        counts = {state.value: 0 for state in RunState}
        by_stage: dict[str, dict[str, int]] = {}
        plan = self.plan
        for stage in plan.stages:
            stage_counts = {state.value: 0 for state in RunState}
            for cell in stage.cells:
                state = self.record(cell.cell_id).state
                counts[state.value] += 1
                stage_counts[state.value] += 1
            by_stage[stage.name] = {
                key: value for key, value in stage_counts.items() if value
            }
        return {
            "schema": CAMPAIGN_SCHEMA,
            "plan_id": plan.plan_id,
            "phase": plan.phase.value,
            "cells": len(plan.cells),
            "counts": {key: value for key, value in counts.items() if value},
            "stages": by_stage,
            "runnable": len(self.runnable_cell_ids()),
            "resume_required": counts[RunState.RUNNING.value],
            "failed_retry_required": counts[RunState.FAILED.value],
        }

    def reconcile_stale_locks(self, max_age_seconds: float) -> tuple[str, ...]:
        if max_age_seconds < 0:
            raise CampaignError("max_age_seconds must be non-negative")
        reconciled: list[str] = []
        lock_root = self.root / ".locks"
        if not lock_root.exists():
            return ()
        try:
            _active, registrations = self._journal_authority()
            cell_ids_by_slug = {_slug(cell_id): cell_id for cell_id in registrations}
        except CampaignError:
            cell_ids_by_slug = {}
        now = float(self.clock())
        for path in sorted(lock_root.glob("*.lock")):
            guard_path = path.with_suffix(".guard")
            guard_handle = guard_path.open("a+", encoding="utf-8")
            try:
                try:
                    fcntl.flock(
                        guard_handle.fileno(),
                        fcntl.LOCK_EX | fcntl.LOCK_NB,
                    )
                except BlockingIOError:
                    continue
                malformed_fingerprint: _ArtifactFingerprint | None = None
                try:
                    payload = _read_json(path)
                    heartbeat = float(
                        payload.get("heartbeat_at", payload["acquired_at"])
                    )
                    cell_id = str(payload["cell_id"])
                except (CampaignError, KeyError, TypeError, ValueError):
                    try:
                        heartbeat = path.stat().st_mtime
                        malformed_fingerprint = _ArtifactFingerprint.from_path(path)
                    except FileNotFoundError:
                        continue
                    cell_id = cell_ids_by_slug.get(path.stem, path.stem)
                    payload = {}
                if now - heartbeat <= max_age_seconds:
                    continue
                worker_terminated = False
                if payload.get("host") == socket.gethostname():
                    try:
                        owner_pid = int(payload["pid"])
                    except (KeyError, TypeError, ValueError):
                        owner_pid = -1
                    if owner_pid > 0 and _process_matches(
                        owner_pid,
                        payload.get("owner_process_identity"),
                    ):
                        continue
                    try:
                        worker_pid = int(payload["worker_pid"])
                        worker_pgid = int(payload["worker_pgid"])
                    except (KeyError, TypeError, ValueError):
                        worker_pid = worker_pgid = -1
                    worker_identity = payload.get("worker_process_identity")
                    if (
                        worker_pid > 0
                        and worker_pgid > 0
                        and isinstance(worker_identity, str)
                        and worker_pgid != os.getpgrp()
                        and _process_matches_for_termination(
                            worker_pid, worker_identity
                        )
                    ):
                        try:
                            observed_pgid = os.getpgid(worker_pid)
                        except ProcessLookupError:
                            observed_pgid = -1
                        if observed_pgid == worker_pgid:
                            try:
                                os.killpg(worker_pgid, signal.SIGTERM)
                                worker_terminated = True
                            except ProcessLookupError:
                                pass
                            for _ in range(20):
                                if not _process_matches_for_termination(
                                    worker_pid, worker_identity
                                ):
                                    break
                                time.sleep(0.05)
                            else:
                                try:
                                    os.killpg(worker_pgid, signal.SIGKILL)
                                except ProcessLookupError:
                                    pass
                if payload:
                    try:
                        current = _read_json(path)
                    except (CampaignError, FileNotFoundError):
                        continue
                    if current.get("attempt_id") != payload.get("attempt_id"):
                        continue
                else:
                    try:
                        current_fingerprint = _ArtifactFingerprint.from_path(path)
                    except FileNotFoundError:
                        continue
                    if current_fingerprint != malformed_fingerprint:
                        continue
                try:
                    path.unlink()
                except FileNotFoundError:
                    continue
                event = self._event(
                    "lock_reconciled",
                    cell_id,
                    message="removed stale cell lock lease",
                    metadata={
                        "age_seconds": now - heartbeat,
                        "lock": str(path),
                        "guard": str(guard_path),
                        "worker_process_group_terminated": worker_terminated,
                    },
                    artifacts=(),
                )
                self._append_event(event)
                reconciled.append(cell_id)
            finally:
                try:
                    fcntl.flock(guard_handle.fileno(), fcntl.LOCK_UN)
                finally:
                    guard_handle.close()
        return tuple(reconciled)

    def _reconcile_plan_projection(self) -> str | None:
        """Republish ``plan.json`` from the complete journal-derived plan chain."""

        committed = self.plan
        try:
            payload = _read_json(self.plan_path)
            active = StudyPlan.from_manifest(payload)
            exact = (
                active == committed
                and canonical_json(payload) == canonical_json(committed.to_manifest())
            )
        except (
            CampaignError,
            FileNotFoundError,
            KeyError,
            OSError,
            TypeError,
            ValueError,
            StudyError,
        ):
            exact = False
        if exact:
            return None
        _atomic_json(self.plan_path, committed.to_manifest())
        return committed.plan_id

    def _repair_journal_tail_for_reconcile(self) -> bool:
        if not self.journal_path.is_file():
            return False
        try:
            size = self.journal_path.stat().st_size
            if size == 0:
                return False
            with self.journal_path.open("rb") as handle:
                handle.seek(-1, os.SEEK_END)
                if handle.read(1) == b"\n":
                    return False
        except OSError as exc:
            raise CampaignError(f"cannot inspect campaign journal tail: {exc}") from exc
        self._append_event(
            self._event(
                "journal_reconciled",
                "__campaign__",
                message="normalized an unterminated final journal record",
                metadata={},
                artifacts=(),
            )
        )
        return True

    @contextmanager
    def _reconciliation_cell_guards(self, cell_ids: Iterable[str]):
        durable_mkdir(self.root / ".locks", parents=True, exist_ok=True)
        handles: list[Any] = []
        try:
            for cell_id in sorted(set(cell_ids)):
                handle = self.lock_guard_path(cell_id).open("a+", encoding="utf-8")
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    handle.close()
                    raise CampaignLocked(
                        f"cannot reconcile while cell {cell_id} is active"
                    ) from exc
                handles.append(handle)
            yield
        finally:
            for handle in reversed(handles):
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                finally:
                    handle.close()

    def reconcile(self, max_age_seconds: float | None = None) -> dict[str, Any]:
        locks = (
            self.reconcile_stale_locks(max_age_seconds)
            if max_age_seconds is not None
            else ()
        )
        durable_mkdir(self.root, parents=True, exist_ok=True)
        registration_lock = self.root / ".registration.lock"
        with registration_lock.open("a+", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            with self._campaign_mutation():
                tail_repaired = self._repair_journal_tail_for_reconcile()
                plan = self.plan
                _active, registrations = self._journal_authority()
                missing = {cell.cell_id for cell in plan.cells}.difference(registrations)
                if missing:
                    raise CampaignError(
                        "active plan has unregistered cells: "
                        + ", ".join(sorted(missing))
                    )
                with self.implementation_identity_lock_path.open(
                    "a+", encoding="utf-8"
                ) as implementation_handle:
                    fcntl.flock(implementation_handle.fileno(), fcntl.LOCK_EX)
                    with self._reconciliation_cell_guards(
                        cell.cell_id for cell in plan.cells
                    ):
                        republished = self._reconcile_plan_projection()
                        rebuilt = 0
                        for cell in plan.cells:
                            _atomic_json(
                                self.cell_manifest_path(cell.cell_id),
                                cell.to_manifest(),
                            )
                            self._cell_projection_cache.pop(cell.cell_id, None)
                            self._write_snapshot(self.record(cell.cell_id))
                            rebuilt += 1
                        with self._activation_projection_lock:
                            self._activation_projection_cursor = 0
                            self._activation_projection_last_event_id = None
                            self._activation_projection_raw_digest = None
                            self._activation_projection_views = {}
                            self._write_activation_identity_projection()
        result: dict[str, Any] = {
            "stale_locks": list(locks),
            "snapshots_rebuilt": rebuilt,
        }
        if tail_repaired:
            result["journal_tail_repaired"] = True
        if republished is not None:
            result["plan_republished"] = republished
        return result


def math_isfinite(value: int | float) -> bool:
    # Kept local to avoid numpy scalar truth semantics in qualification JSON.
    return value == value and value not in {float("inf"), float("-inf")}


@dataclass(frozen=True, slots=True)
class RunSummary:
    selected_cells: int
    completed_cells: int
    failed_cells: int
    skipped_cells: int

    def to_dict(self) -> dict[str, int]:
        return {
            "selected_cells": self.selected_cells,
            "completed_cells": self.completed_cells,
            "failed_cells": self.failed_cells,
            "skipped_cells": self.skipped_cells,
        }


class _PersistentCellWorker:
    """One crash-isolated executor process shared by a cell's stage chain."""

    def __init__(
        self,
        *,
        command: Sequence[str],
        cwd: Path,
        environment: Mapping[str, str],
    ) -> None:
        self._stderr = tempfile.TemporaryFile(mode="w+t", encoding="utf-8")
        try:
            self._process = subprocess.Popen(
                command,
                cwd=cwd,
                env=dict(environment),
                text=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._stderr,
                bufsize=1,
                start_new_session=True,
            )
        except Exception:
            self._stderr.close()
            raise
        self._pgid = self._process.pid
        if self._process.stdin is None or self._process.stdout is None:
            self.close()
            raise CampaignError("persistent run_cell worker lacks control pipes")

    @property
    def pid(self) -> int:
        return int(self._process.pid)

    @property
    def pgid(self) -> int:
        return int(self._pgid)

    def _stderr_tail(self) -> str:
        self._stderr.flush()
        self._stderr.seek(0, os.SEEK_END)
        end = self._stderr.tell()
        self._stderr.seek(max(0, end - 4_000))
        return self._stderr.read()

    def invoke(self, *, stage: str, artifacts_out: Path, resume: bool) -> None:
        if self._process.poll() is not None:
            raise CampaignError(
                "persistent run_cell worker exited "
                f"{self._process.returncode} before {stage}: {self._stderr_tail()}"
            )
        request = json.dumps(
            {
                "stage": stage,
                "artifacts_out": str(artifacts_out),
                "resume": resume,
            },
            sort_keys=True,
        )
        try:
            assert self._process.stdin is not None
            self._process.stdin.write(request + "\n")
            self._process.stdin.flush()
            assert self._process.stdout is not None
            response_raw = self._process.stdout.readline()
        except (BrokenPipeError, OSError) as exc:
            raise CampaignError(
                f"persistent run_cell worker control failure during {stage}: "
                f"{exc}; {self._stderr_tail()}"
            ) from exc
        if not response_raw:
            returncode = self._process.poll()
            if returncode is None:
                returncode = self._process.wait()
            raise CampaignError(
                f"persistent run_cell worker exited {returncode} during {stage}: "
                f"{self._stderr_tail()}"
            )
        try:
            response = json.loads(response_raw)
        except json.JSONDecodeError as exc:
            raise CampaignError(
                f"persistent run_cell worker emitted malformed control data "
                f"during {stage}: {response_raw[-1_000:]!r}"
            ) from exc
        if (
            not isinstance(response, dict)
            or response.get("stage") != stage
            or not isinstance(response.get("ok"), bool)
        ):
            raise CampaignError(
                f"persistent run_cell worker response binding mismatch during {stage}"
            )
        if response["ok"] is not True:
            error_type = str(response.get("error_type", "CellExecutionError"))
            error = str(response.get("error", "unknown worker failure"))
            raise CampaignError(f"{error_type} during {stage}: {error}")

    def close(self) -> None:
        process = getattr(self, "_process", None)
        if process is None:
            return
        try:
            if process.poll() is None and process.stdin is not None:
                try:
                    process.stdin.write('{"command":"close"}\n')
                    process.stdin.flush()
                except (BrokenPipeError, OSError):
                    pass
                process.stdin.close()
            if process.poll() is None:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(self._pgid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(self._pgid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        process.wait()
            # The worker is the session leader.  If it exited without reaping a
            # descendant, terminate the remainder of its owned process group.
            try:
                os.killpg(self._pgid, 0)
            except ProcessLookupError:
                pass
            else:
                try:
                    os.killpg(self._pgid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                else:
                    for _ in range(20):
                        try:
                            os.killpg(self._pgid, 0)
                        except ProcessLookupError:
                            break
                        time.sleep(0.05)
                    else:
                        try:
                            os.killpg(self._pgid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
        finally:
            if process.stdout is not None:
                process.stdout.close()
            self._stderr.close()
            self._process = None


class CampaignRunner:
    """Drive cells through prepare/train/calibrate/evaluate/qualify.

    Promotion is intentionally absent.  A cell that merely emits an evaluation
    report remains ``evaluated``; only an independently bound all-true
    qualification artifact can move it to ``qualified``.
    """

    def __init__(
        self,
        campaign: Campaign,
        *,
        python: str = sys.executable,
        module: str = "block_crosscoder_experiment.cli.run_cell",
        env: Mapping[str, str] | None = None,
    ):
        self.campaign = campaign
        self.python = python
        self.module = module
        self.env = dict(env or {})

    def _validate_executor_module(self, cell_ids: Sequence[str]) -> None:
        if self.module == CANONICAL_CELL_MODULE:
            return
        scientific = [
            cell_id
            for cell_id in cell_ids
            if self.campaign._require_cell(cell_id).decision_map["runtime.smoke"]
            is False
        ]
        if scientific:
            raise CampaignError(
                "non-smoke scientific cells require the canonical cell executor "
                f"module {CANONICAL_CELL_MODULE!r}; custom modules are smoke-only"
            )

    def run(
        self,
        *,
        limit: int | None = None,
        resume: bool = False,
        cell_ids: Sequence[str] | None = None,
        stop_after: str | None = None,
    ) -> RunSummary:
        if limit is not None and limit <= 0:
            raise CampaignError("limit must be positive")
        if limit is not None and cell_ids is not None:
            raise CampaignError("limit cannot be combined with explicit cell IDs")
        if stop_after is not None and stop_after not in STAGE_TARGETS:
            raise CampaignError(f"unknown stop stage {stop_after!r}")
        if cell_ids is None:
            selected = list(
                self.campaign.runnable_cell_ids(
                    include_failed=resume,
                    include_resume_required=resume,
                )
            )
        else:
            selected = list(cell_ids)
            for cell_id in selected:
                cell = self.campaign._require_cell(cell_id)
                if not self.campaign.stage_open(cell.stage):
                    raise CampaignError(
                        f"cell {cell_id} belongs to unopened stage {cell.stage!r}"
                    )
        if limit is not None:
            selected = selected[:limit]
        self._validate_executor_module(selected)
        completed = failed = skipped = 0
        for cell_id in selected:
            try:
                result = self._run_cell(cell_id, resume=resume, stop_after=stop_after)
            except CampaignLocked:
                skipped += 1
                continue
            if result is RunState.FAILED:
                failed += 1
            elif result in {RunState.QUALIFIED, RunState.PROMOTED} or (
                stop_after is not None
                and self._state_reached(result, STAGE_TARGETS[stop_after])
            ):
                completed += 1
            else:
                skipped += 1
        return RunSummary(len(selected), completed, failed, skipped)

    def _run_cell(
        self,
        cell_id: str,
        *,
        resume: bool,
        stop_after: str | None,
    ) -> RunState:
        with self.campaign.lock(cell_id) as cell_lock:
            record = self.campaign.record(cell_id)
            # A power loss after the journal fsync but before the disposable
            # snapshot rename must not feed stale prerequisites to run_cell.
            self.campaign._ensure_snapshot(record)
            if record.state is RunState.FAILED:
                if not resume:
                    return record.state
                record = self.campaign.retry(cell_id, assume_locked=True)
            if record.state is RunState.RUNNING and not resume:
                return record.state
            if record.state in {RunState.QUALIFIED, RunState.PROMOTED}:
                return record.state
            if stop_after is not None and self._state_reached(
                record.state, STAGE_TARGETS[stop_after]
            ):
                return record.state

            stages = self._remaining_stages(record.state)
            worker: _PersistentCellWorker | None = None
            try:
                for stage in stages:
                    if (
                        stage == "train"
                        and self.campaign.record(cell_id).state is RunState.PREPARED
                    ):
                        self.campaign.transition(
                            cell_id,
                            RunState.RUNNING,
                            message="training process claimed",
                            assume_locked=True,
                        )
                    try:
                        if worker is None and self._supports_persistent_worker:
                            worker = self._start_worker(cell_id)
                            cell_lock.bind_worker(pid=worker.pid, pgid=worker.pgid)
                        artifacts = self._invoke(
                            cell_id,
                            stage,
                            resume=resume,
                            worker=worker,
                        )
                        target = STAGE_TARGETS[stage]
                        self.campaign.transition(
                            cell_id,
                            target,
                            artifacts=artifacts,
                            message=f"{stage} stage completed",
                            assume_locked=True,
                        )
                    except (
                        ArtifactError,
                        CampaignError,
                        OSError,
                        subprocess.SubprocessError,
                    ) as exc:
                        current = self.campaign.record(cell_id)
                        if current.state is not RunState.FAILED:
                            self.campaign.transition(
                                cell_id,
                                RunState.FAILED,
                                message=f"{stage} stage failed: {exc}",
                                metadata={
                                    "stage": stage,
                                    "error_type": type(exc).__name__,
                                },
                                assume_locked=True,
                            )
                        return RunState.FAILED
                    if stage == stop_after:
                        break
            finally:
                if worker is not None:
                    worker.close()
            return self.campaign.record(cell_id).state

    @property
    def _supports_persistent_worker(self) -> bool:
        return self.module == "block_crosscoder_experiment.cli.run_cell"

    def _start_worker(self, cell_id: str) -> "_PersistentCellWorker":
        environment = os.environ.copy()
        environment.update(self.env)
        environment["BSC_CAMPAIGN_ROOT"] = str(self.campaign.root.resolve())
        return _PersistentCellWorker(
            command=[
                self.python,
                "-m",
                self.module,
                "--cell",
                str(self.campaign.cell_manifest_path(cell_id)),
                "--worker",
            ],
            cwd=self.campaign.root,
            environment=environment,
        )

    @staticmethod
    def _state_reached(state: RunState, target: RunState) -> bool:
        order = {
            RunState.PLANNED: -1,
            RunState.PREPARED: 0,
            # Training has been claimed, so prepare is already durable even
            # though train has not completed.
            RunState.RUNNING: 0,
            RunState.TRAINED: 1,
            RunState.CALIBRATED: 2,
            RunState.EVALUATED: 3,
            RunState.QUALIFIED: 4,
            RunState.PROMOTED: 5,
        }
        return state in order and target in order and order[state] >= order[target]

    @staticmethod
    def _remaining_stages(state: RunState) -> tuple[str, ...]:
        start = {
            RunState.PLANNED: 0,
            RunState.PREPARED: 1,
            RunState.RUNNING: 1,
            RunState.TRAINED: 2,
            RunState.CALIBRATED: 3,
            RunState.EVALUATED: 4,
        }.get(state)
        if start is None:
            return ()
        return tuple(STAGE_TARGETS)[start:]

    def _invoke(
        self,
        cell_id: str,
        stage: str,
        *,
        resume: bool,
        worker: "_PersistentCellWorker | None" = None,
    ) -> tuple[ArtifactRef, ...]:
        cell_manifest = self.campaign.cell_manifest_path(cell_id)
        attempt = uuid.uuid4().hex
        artifacts_out = (
            self.campaign.cell_dir(cell_id)
            / "stage-artifacts"
            / f"{stage}-{attempt}.json"
        )
        durable_mkdir(artifacts_out.parent, parents=True, exist_ok=True)
        if worker is None:
            command = [
                self.python,
                "-m",
                self.module,
                "--cell",
                str(cell_manifest),
                "--stage",
                stage,
                "--artifacts-out",
                str(artifacts_out),
            ]
            if resume:
                command.append("--resume")
            environment = os.environ.copy()
            environment.update(self.env)
            environment["BSC_CAMPAIGN_ROOT"] = str(self.campaign.root.resolve())
            completed = subprocess.run(
                command,
                cwd=self.campaign.root,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            if completed.returncode != 0:
                tail = (completed.stderr or completed.stdout)[-4_000:]
                raise CampaignError(
                    f"run_cell exited {completed.returncode} during {stage}: {tail}"
                )
        else:
            worker.invoke(stage=stage, artifacts_out=artifacts_out, resume=resume)
        manifest_ref = self.campaign._verified_artifact_from_path(
            f"{stage}_manifest",
            artifacts_out,
        )
        manifest_verification = manifest_ref._verification
        if manifest_verification is None:  # pragma: no cover - private invariant
            raise AssertionError("verified manifest lacks its process-local receipt")
        manifest_fingerprint = manifest_verification.fingerprint
        refs = list(self._load_artifact_manifest(cell_id, stage, artifacts_out))
        manifest_ref = self.campaign._verify_artifact(manifest_ref)
        final_verification = manifest_ref._verification
        if (
            final_verification is None  # pragma: no cover - private invariant
            or final_verification.fingerprint != manifest_fingerprint
        ):
            raise ArtifactError("stage-artifact manifest changed while loading")
        refs.append(manifest_ref)
        return self.campaign._authorize_executor_stage_artifacts(
            cell_id,
            stage,
            refs,
            canonical_executor=(self.module == CANONICAL_CELL_MODULE),
        )

    def _load_artifact_manifest(
        self,
        cell_id: str,
        stage: str,
        path: Path,
    ) -> tuple[ArtifactRef, ...]:
        payload = _read_json(path)
        if payload.get("schema") != ARTIFACT_SCHEMA:
            raise ArtifactError(f"wrong stage-artifact schema at {path}")
        if payload.get("cell_id") != cell_id or payload.get("stage") != stage:
            raise ArtifactError("stage-artifact manifest binding mismatch")
        items = payload.get("artifacts")
        if not isinstance(items, list):
            raise ArtifactError("stage-artifact manifest needs an artifacts list")
        refs: list[ArtifactRef] = []
        seen: set[str] = set()
        for item in items:
            if not isinstance(item, Mapping):
                raise ArtifactError("artifact entries must be objects")
            kind = str(item.get("kind", ""))
            if kind in seen:
                raise ArtifactError(f"stage manifest repeats artifact kind {kind!r}")
            seen.add(kind)
            artifact_path = Path(str(item.get("path", "")))
            if not artifact_path.is_absolute():
                artifact_path = self.campaign.root / artifact_path
            claimed_hash = item.get("sha256")
            claimed_size = item.get("size_bytes")
            if not isinstance(claimed_hash, str):
                raise ArtifactError(f"child-reported hash is not a string for {kind}")
            if not isinstance(claimed_size, int) or isinstance(claimed_size, bool):
                raise ArtifactError(f"child-reported size is not an integer for {kind}")
            resolved = artifact_path.resolve()
            try:
                stored_path = str(resolved.relative_to(self.campaign.root.resolve()))
            except ValueError:
                stored_path = str(resolved)
            ref = ArtifactRef(kind, stored_path, claimed_hash, claimed_size)
            refs.append(ref)
        observed = frozenset(item.kind for item in refs)
        expected = EXPECTED_STAGE_ARTIFACTS[stage]
        if observed != expected:
            raise ArtifactError(
                f"{stage} stage artifact kinds must be exactly "
                f"{sorted(expected)}, got {sorted(observed)}"
            )
        verified: list[ArtifactRef] = []
        for ref in refs:
            try:
                verified.append(self.campaign._verify_artifact(ref))
            except ArtifactError as exc:
                raise ArtifactError(
                    f"child-reported artifact mismatch for {ref.kind}: {exc}"
                ) from exc
        return tuple(verified)


__all__ = [
    "ARTIFACT_SCHEMA",
    "ArtifactError",
    "ArtifactRef",
    "CAMPAIGN_SCHEMA",
    "Campaign",
    "CampaignError",
    "CampaignLocked",
    "CampaignRecord",
    "CampaignRunner",
    "EVALUATION_SCHEMA",
    "EXPECTED_STAGE_ARTIFACTS",
    "InvalidTransition",
    "LEGAL_TRANSITIONS",
    "PROMOTION_SCHEMA",
    "QUALIFICATION_SCHEMA",
    "REQUIRED_QUALIFICATION_CHECKS",
    "REQUIRED_ARTIFACTS",
    "SELECTION_SCHEMA",
    "RunState",
    "RunSummary",
]
