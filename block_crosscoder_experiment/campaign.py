"""Simple local campaign state and cell execution.

The campaign trusts the operator and the files under its root.  ``plan.json``
is the current plan, each cell has one ``state.json``, and artifacts are stored
as ordinary paths.  There is no journal replay, content addressing, source
fingerprinting, or artifact hashing.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

from .studies import (
    CellSpec,
    FrozenPanelDecision,
    FrozenPanelEntry,
    FrozenSelection,
    Phase,
    Phase1Blueprint,
    Phase2Blueprint,
    SelectionPolicy,
    StudyError,
    StudyPlan,
    build_phase1_transfer,
)


CAMPAIGN_SCHEMA = "bsc-campaign-simple-v1"
ARTIFACT_SCHEMA = "bsc-stage-artifacts-simple-v1"
QUALIFICATION_SCHEMA = "bsc-qualification-simple-v1"
PREPARATION_SCHEMA = "bsc-preparation-simple-v1"
EVALUATION_SCHEMA = "bsc-evaluation-simple-v1"
EVALUATION_EXECUTION_IMPLEMENTATION = "deployable_full_view"
PROMOTION_SCHEMA = "bsc-promotion-simple-v1"
SELECTION_SCHEMA = "bsc-stage-selection-simple-v1"
PHASE1_DECISION_SCHEMA = "bsc-phase1-decision-simple-v1"
PANEL_DECISION_SCHEMA = "bsc-phase3-panel-simple-v1"
CANONICAL_CELL_MODULE = "block_crosscoder_experiment.cli.run_cell"


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
    RunState.FAILED: frozenset(),
    RunState.PROMOTED: frozenset(),
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
        {"calibration", "deployment_codec", "calibration_record"}
    ),
    "evaluate": frozenset({"deployment_schedules", "evaluation"}),
    "qualify": frozenset({"qualification"}),
}

REQUIRED_ARTIFACTS: Mapping[RunState, frozenset[str]] = {
    RunState.PREPARED: frozenset({"preparation"}),
    RunState.TRAINED: frozenset(
        {"preparation", "checkpoint", "training_report"}
    ),
    RunState.CALIBRATED: frozenset(
        {
            "preparation",
            "checkpoint",
            "training_report",
            "calibration",
            "deployment_codec",
            "calibration_record",
        }
    ),
    RunState.EVALUATED: frozenset(
        {
            "preparation",
            "checkpoint",
            "training_report",
            "calibration",
            "deployment_codec",
            "calibration_record",
            "deployment_schedules",
            "evaluation",
        }
    ),
    RunState.QUALIFIED: frozenset(
        {
            "preparation",
            "checkpoint",
            "training_report",
            "calibration",
            "deployment_codec",
            "calibration_record",
            "deployment_schedules",
            "evaluation",
            "qualification",
        }
    ),
}

# Kept as a public name for callers that display the qualification surface.
REQUIRED_QUALIFICATION_CHECKS = frozenset(
    {"finite", "method_endpoints", "scientific_endpoint_complete", "split_integrity"}
)


def _slug(identifier: str) -> str:
    return identifier.replace(":", "_").replace("/", "_")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CampaignError(f"expected a JSON object at {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(body)
    os.replace(temporary, path)


def _relative_or_absolute(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(root.resolve()))
    except ValueError:
        return str(resolved)


def _lookup(payload: Mapping[str, Any], dotted: str) -> Any:
    value: Any = payload
    for part in dotted.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise CampaignError(f"qualification lacks selection metric {dotted!r}")
        value = value[part]
    return value


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    kind: str
    path: str

    def __post_init__(self) -> None:
        if not self.kind or not self.path:
            raise ArtifactError("artifact kind and path are required")

    @classmethod
    def from_path(cls, kind: str, path: str | Path, root: str | Path) -> "ArtifactRef":
        root_path = Path(root)
        artifact_path = Path(path)
        if not artifact_path.is_absolute():
            artifact_path = root_path / artifact_path
        if not artifact_path.is_file():
            raise ArtifactError(f"artifact does not exist: {artifact_path}")
        return cls(kind, _relative_or_absolute(artifact_path, root_path))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArtifactRef":
        return cls(str(payload["kind"]), str(payload["path"]))

    def resolve(self, root: str | Path) -> Path:
        path = Path(self.path)
        if not path.is_absolute():
            path = Path(root) / path
        return path

    def verify(self, root: str | Path) -> "ArtifactRef":
        if not self.resolve(root).is_file():
            raise ArtifactError(f"artifact does not exist: {self.resolve(root)}")
        return self

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "path": self.path}


@dataclass(frozen=True, slots=True)
class CampaignRecord:
    cell_id: str
    state: RunState
    artifacts: tuple[ArtifactRef, ...] = ()
    message: str | None = None
    resume_state: RunState | None = None

    @property
    def artifact_map(self) -> dict[str, ArtifactRef]:
        return {artifact.kind: artifact for artifact in self.artifacts}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CAMPAIGN_SCHEMA,
            "cell_id": self.cell_id,
            "state": self.state.value,
            "resume_state": (
                None if self.resume_state is None else self.resume_state.value
            ),
            "message": self.message,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CampaignRecord":
        resume = payload.get("resume_state")
        return cls(
            cell_id=str(payload["cell_id"]),
            state=RunState(payload["state"]),
            resume_state=None if resume is None else RunState(resume),
            message=None if payload.get("message") is None else str(payload["message"]),
            artifacts=tuple(
                ArtifactRef.from_dict(item) for item in payload.get("artifacts", ())
            ),
        )


class CellLock(AbstractContextManager["CellLock"]):
    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle = None

    def __enter__(self) -> "CellLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a+")
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._handle.close()
            self._handle = None
            raise CampaignLocked(f"cell is already running: {self.path.parent.name}") from exc
        return self

    def bind_worker(self, *, pid: int, pgid: int) -> None:
        del pid, pgid

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        if self._handle is not None:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            self._handle.close()
            self._handle = None


class Campaign:
    """A plan plus one current state file per cell."""

    def __init__(self, root: str | Path, **_: Any) -> None:
        self.root = Path(root)
        self.plan_path = self.root / "plan.json"
        self.blueprint_path = self.root / "blueprint.json"
        self.phase1_decision_path = self.root / "phase1-decision.json"
        self.panel_decision_path = self.root / "panel-decision.json"
        self._mutation_lock_path = self.root / ".campaign.lock"

    @contextmanager
    def _mutation(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with self._mutation_lock_path.open("a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            yield
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @property
    def plan(self) -> StudyPlan:
        if not self.plan_path.is_file():
            raise CampaignError(f"campaign has no plan: {self.plan_path}")
        try:
            return StudyPlan.from_manifest(_read_json(self.plan_path))
        except (KeyError, TypeError, ValueError, StudyError) as exc:
            raise CampaignError(f"invalid campaign plan: {exc}") from exc

    def cell_dir(self, cell_id: str) -> Path:
        return self.root / "cells" / _slug(cell_id)

    def cell_manifest_path(self, cell_id: str) -> Path:
        return self.cell_dir(cell_id) / "cell.json"

    def state_path(self, cell_id: str) -> Path:
        return self.cell_dir(cell_id) / "state.json"

    def lock_path(self, cell_id: str) -> Path:
        return self.cell_dir(cell_id) / ".run.lock"

    def lock(self, cell_id: str) -> CellLock:
        self._require_cell(cell_id)
        return CellLock(self.lock_path(cell_id))

    def register(
        self,
        plan: StudyPlan,
        *,
        blueprint_manifest: Mapping[str, Any],
        phase1_decision_manifest: Mapping[str, Any] | None = None,
        panel_decision_manifest: Mapping[str, Any] | None = None,
    ) -> None:
        with self._mutation():
            if self.plan_path.exists():
                raise CampaignError(f"campaign already exists: {self.root}")
            _write_json(self.plan_path, plan.to_manifest())
            _write_json(self.blueprint_path, blueprint_manifest)
            if phase1_decision_manifest is not None:
                _write_json(self.phase1_decision_path, phase1_decision_manifest)
            if panel_decision_manifest is not None:
                _write_json(self.panel_decision_path, panel_decision_manifest)
            self._register_cells(plan.cells)

    def _register_cells(self, cells: Iterable[CellSpec]) -> None:
        for cell in cells:
            directory = self.cell_dir(cell.cell_id)
            directory.mkdir(parents=True, exist_ok=True)
            manifest_path = self.cell_manifest_path(cell.cell_id)
            if not manifest_path.exists():
                _write_json(manifest_path, cell.to_manifest())
            state_path = self.state_path(cell.cell_id)
            if not state_path.exists():
                _write_json(
                    state_path,
                    CampaignRecord(cell.cell_id, RunState.PLANNED).to_dict(),
                )

    def _replace_plan(self, plan: StudyPlan) -> None:
        current = self.plan
        current_stages = [stage.to_dict() for stage in current.stages]
        incoming_prefix = [stage.to_dict() for stage in plan.stages[: len(current.stages)]]
        if current.phase is not plan.phase or current_stages != incoming_prefix:
            raise CampaignError("extended plan must preserve the current stage prefix")
        self._register_cells(plan.cells[len(current.cells) :])
        _write_json(self.plan_path, plan.to_manifest())

    def extend(
        self,
        plan: StudyPlan,
        *,
        selection: FrozenSelection,
        selection_path: str | Path,
    ) -> None:
        del selection, selection_path
        with self._mutation():
            self._replace_plan(plan)

    def extend_family(
        self,
        plan: StudyPlan,
        *,
        family_name: str,
        selection: FrozenSelection,
        selection_path: str | Path,
    ) -> None:
        del family_name
        self.extend(plan, selection=selection, selection_path=selection_path)

    def extend_family_revisit(
        self,
        plan: StudyPlan,
        *,
        family_name: str,
        selection_path: str | Path,
    ) -> None:
        del family_name, selection_path
        with self._mutation():
            self._replace_plan(plan)

    def _require_cell(self, cell_id: str) -> CellSpec:
        matches = [cell for cell in self.plan.cells if cell.cell_id == cell_id]
        if len(matches) != 1:
            raise CampaignError(f"plan does not define cell {cell_id!r}")
        return matches[0]

    def record(self, cell_id: str) -> CampaignRecord:
        self._require_cell(cell_id)
        path = self.state_path(cell_id)
        if not path.is_file():
            return CampaignRecord(cell_id, RunState.PLANNED)
        return CampaignRecord.from_dict(_read_json(path))

    def records(self) -> tuple[CampaignRecord, ...]:
        return tuple(self.record(cell.cell_id) for cell in self.plan.cells)

    def transition(
        self,
        cell_id: str,
        target: RunState,
        *,
        artifacts: Sequence[ArtifactRef] = (),
        message: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        assume_locked: bool = False,
    ) -> CampaignRecord:
        del metadata, assume_locked
        with self._mutation():
            current = self.record(cell_id)
            if target not in LEGAL_TRANSITIONS[current.state]:
                raise InvalidTransition(
                    f"illegal transition {current.state.value} -> {target.value}"
                )
            artifact_map = current.artifact_map
            for artifact in artifacts:
                artifact.verify(self.root)
                artifact_map[artifact.kind] = artifact
            resume_state = current.resume_state
            if target is RunState.FAILED:
                resume_state = (
                    RunState.PREPARED
                    if current.state is RunState.RUNNING
                    else current.state
                )
            required = REQUIRED_ARTIFACTS.get(target, frozenset())
            missing = required.difference(artifact_map)
            if missing:
                raise ArtifactError(
                    f"{target.value} is missing artifacts {sorted(missing)}"
                )
            updated = CampaignRecord(
                cell_id,
                target,
                tuple(artifact_map.values()),
                message,
                resume_state,
            )
            _write_json(self.state_path(cell_id), updated.to_dict())
            return updated

    def retry(self, cell_id: str, *, assume_locked: bool = False) -> CampaignRecord:
        del assume_locked
        with self._mutation():
            current = self.record(cell_id)
            if current.state is not RunState.FAILED:
                raise InvalidTransition("only failed cells can be retried")
            target = current.resume_state or RunState.PLANNED
            updated = CampaignRecord(
                cell_id,
                target,
                current.artifacts,
                "retry requested",
                None,
            )
            _write_json(self.state_path(cell_id), updated.to_dict())
            return updated

    def eligible_for_qualification(self, cell_id: str) -> bool:
        return self.record(cell_id).state is RunState.EVALUATED

    def eligible_for_promotion(self, cell_id: str) -> bool:
        record = self.record(cell_id)
        if record.state is not RunState.QUALIFIED:
            return False
        payload = self._qualification(record)
        return payload.get("promotion_eligible") is True

    def promote(
        self, cell_id: str, promotion: ArtifactRef
    ) -> CampaignRecord:
        if not self.eligible_for_promotion(cell_id):
            raise CampaignError("cell is not eligible for promotion")
        return self.transition(
            cell_id, RunState.PROMOTED, artifacts=(promotion,)
        )

    def _qualification(self, record: CampaignRecord) -> dict[str, Any]:
        ref = record.artifact_map.get("qualification")
        if ref is None:
            raise CampaignError(f"cell {record.cell_id} has no qualification")
        return _read_json(ref.resolve(self.root))

    def stage_open(self, stage_name: str) -> bool:
        stage = next(
            (item for item in self.plan.stages if item.name == stage_name), None
        )
        if stage is None:
            raise CampaignError(f"unknown stage {stage_name!r}")
        if not stage.depends_on:
            return True
        records = {record.cell_id: record for record in self.records()}
        stages = {item.name: item for item in self.plan.stages}
        for dependency in stage.depends_on:
            source = stages[dependency]
            qualified = sum(
                records[cell.cell_id].state
                in {RunState.QUALIFIED, RunState.PROMOTED}
                for cell in source.cells
            )
            minimum = (
                stage.gate.minimum_count
                if stage.gate is not None
                and stage.gate.source_stage == dependency
                else len(source.cells)
            )
            if qualified < minimum:
                return False
        return True

    def runnable_cell_ids(
        self,
        *,
        include_failed: bool = False,
        include_resume_required: bool = False,
    ) -> tuple[str, ...]:
        result: list[str] = []
        for cell in self.plan.cells:
            if not self.stage_open(cell.stage):
                continue
            state = self.record(cell.cell_id).state
            if state is RunState.PLANNED:
                result.append(cell.cell_id)
            elif include_resume_required and state is RunState.RUNNING:
                result.append(cell.cell_id)
            elif include_failed and state is RunState.FAILED:
                result.append(cell.cell_id)
        return tuple(result)

    @staticmethod
    def _candidate_for_variant(
        stage_name: str,
        candidates: Sequence[Mapping[str, Any]],
        variant: str,
    ) -> Mapping[str, Any] | None:
        recipe_name = f"derived_{stage_name}_{variant}"
        matches = [
            candidate
            for candidate in candidates
            if candidate.get("recipe_name") == recipe_name
        ]
        if len(matches) > 1:
            raise CampaignError(f"stage repeats variant {variant!r}")
        return None if not matches else matches[0]

    @staticmethod
    def _directional_improvement(
        candidate: float, reference: float, direction: str
    ) -> float:
        return candidate - reference if direction == "max" else reference - candidate

    @classmethod
    def _apply_policy_gates(
        cls,
        stage_name: str,
        policy: SelectionPolicy,
        candidates: list[dict[str, Any]],
        excluded: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if policy.required_control_variant is not None:
            control = cls._candidate_for_variant(
                stage_name, candidates, policy.required_control_variant
            )
            carrier = cls._candidate_for_variant(
                stage_name, candidates, str(policy.noninferiority_candidate_variant)
            )
            if control is None:
                raise CampaignError("selection lacks its declared control candidate")
            passed = carrier is not None
            if passed:
                tolerance = float(policy.control_noninferiority_absolute_tolerance)
                control_values = {
                    item["seed"]: item["metric"] for item in control["observations"]
                }
                carrier_values = {
                    item["seed"]: item["metric"] for item in carrier["observations"]
                }
                passed = set(control_values) == set(carrier_values) and all(
                    -cls._directional_improvement(
                        carrier_values[seed], control_values[seed], policy.direction
                    )
                    <= tolerance
                    for seed in control_values
                )
            if not passed:
                excluded.extend(
                    {**item, "reason": "noninferiority_failed"}
                    for item in candidates
                    if item["candidate_id"] != control["candidate_id"]
                )
                candidates = [control]

        if policy.default_parent_variant is not None:
            parent = cls._candidate_for_variant(
                stage_name, candidates, policy.default_parent_variant
            )
            if parent is None:
                raise CampaignError("selection lacks its declared parent candidate")
            threshold = float(policy.minimum_effect_absolute)
            parent_values = {
                item["seed"]: item["metric"] for item in parent["observations"]
            }
            retained = [parent]
            for candidate in candidates:
                if candidate["candidate_id"] == parent["candidate_id"]:
                    continue
                values = {
                    item["seed"]: item["metric"]
                    for item in candidate["observations"]
                }
                passed = set(values) == set(parent_values) and all(
                    cls._directional_improvement(
                        values[seed], parent_values[seed], policy.direction
                    )
                    >= threshold
                    for seed in parent_values
                )
                if passed:
                    retained.append(candidate)
                else:
                    excluded.append({**candidate, "reason": "minimum_effect_not_met"})
            candidates = retained

        if policy.parsimony_order_variants:
            reference = cls._candidate_for_variant(
                stage_name,
                candidates,
                str(policy.noninferiority_candidate_variant),
            )
            if reference is None:
                raise CampaignError("parsimony selection lacks its reference")
            tolerance = float(policy.parsimony_noninferiority_absolute_tolerance)
            reference_values = {
                item["seed"]: item["metric"] for item in reference["observations"]
            }
            selected = None
            for variant in policy.parsimony_order_variants:
                candidate = cls._candidate_for_variant(
                    stage_name, candidates, variant
                )
                if candidate is None:
                    continue
                values = {
                    item["seed"]: item["metric"]
                    for item in candidate["observations"]
                }
                if set(values) == set(reference_values) and all(
                    -cls._directional_improvement(
                        values[seed], reference_values[seed], policy.direction
                    )
                    <= tolerance
                    for seed in values
                ):
                    selected = candidate
                    break
            if selected is None:
                raise CampaignError("no rank satisfies the parsimony tolerance")
            excluded.extend(
                {**candidate, "reason": "parsimony"}
                for candidate in candidates
                if candidate["candidate_id"] != selected["candidate_id"]
            )
            candidates = [selected]
        return candidates

    def _selection_population(
        self, stage_name: str, policy: SelectionPolicy
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        stage = next(
            (item for item in self.plan.stages if item.name == stage_name), None
        )
        if stage is None:
            raise CampaignError(f"unknown stage {stage_name!r}")
        by_candidate: dict[str, list[tuple[CellSpec, CampaignRecord]]] = {}
        for cell in stage.cells:
            by_candidate.setdefault(cell.candidate_id, []).append(
                (cell, self.record(cell.cell_id))
            )
        expected_seeds = tuple(sorted({cell.seed for cell in stage.cells}))
        candidates: list[dict[str, Any]] = []
        excluded: list[dict[str, Any]] = []
        for candidate_id, pairs in sorted(by_candidate.items()):
            cells = tuple(sorted((cell for cell, _ in pairs), key=lambda item: item.seed))
            record_by_id = {record.cell_id: record for _, record in pairs}
            reason = None
            if tuple(cell.seed for cell in cells) != expected_seeds:
                reason = "incomplete_seeds"
            observations: list[dict[str, Any]] = []
            qualification_files: list[str] = []
            if reason is None:
                for cell in cells:
                    record = record_by_id[cell.cell_id]
                    if record.state not in {RunState.QUALIFIED, RunState.PROMOTED}:
                        reason = "not_qualified"
                        break
                    qualification = self._qualification(record)
                    smoke_ok = (
                        cell.decision_map["runtime.smoke"] is True
                        and qualification.get("selection_eligible_for_protocol_test")
                        is True
                    )
                    if qualification.get("qualified") is not True:
                        reason = "qualification_failed"
                        break
                    scientific = qualification.get("scientific_outcome")
                    if (
                        policy.require_scientific_outcome_pass
                        and not (
                            isinstance(scientific, Mapping)
                            and scientific.get("passed") is True
                        )
                        and not smoke_ok
                    ):
                        reason = "scientific_outcome_failed"
                        break
                    if (
                        qualification.get("promotion_eligible") is not True
                        and not smoke_ok
                    ):
                        reason = "promotion_ineligible"
                        break
                    value = _lookup(qualification, policy.metric_path)
                    if policy.map_key is not None:
                        if not isinstance(value, Mapping):
                            reason = "metric_is_not_a_map"
                            break
                        value = value.get(policy.map_key)
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(float(value))
                    ):
                        reason = "metric_not_finite"
                        break
                    ref = record.artifact_map["qualification"]
                    qualification_files.append(ref.path)
                    observations.append(
                        {
                            "cell_id": cell.cell_id,
                            "seed": cell.seed,
                            "metric": float(value),
                            "qualification_file": ref.path,
                        }
                    )
            entry: dict[str, Any] = {
                "candidate_id": candidate_id,
                "recipe_name": cells[0].recipe_name,
                "cell_ids": [cell.cell_id for cell in cells],
                "seeds": [cell.seed for cell in cells],
                "observations": observations,
                "qualification_files": qualification_files,
            }
            if reason is not None:
                excluded.append({**entry, "reason": reason})
                continue
            values = [item["metric"] for item in observations]
            entry["median"] = float(median(values))
            entry["worst_seed"] = (
                min(values) if policy.direction == "max" else max(values)
            )
            candidates.append(entry)
        candidates = self._apply_policy_gates(
            stage_name, policy, candidates, excluded
        )
        return candidates, excluded

    @staticmethod
    def _rank_key(
        candidate: Mapping[str, Any], direction: str
    ) -> tuple[float, float, str]:
        sign = -1.0 if direction == "max" else 1.0
        return (
            sign * float(candidate["median"]),
            sign * float(candidate["worst_seed"]),
            str(candidate["candidate_id"]),
        )

    def select_stage(
        self, stage_name: str, *, out: str | Path | None = None
    ) -> dict[str, Any]:
        stage = next(
            (item for item in self.plan.stages if item.name == stage_name), None
        )
        if stage is None or stage.selection_policy is None:
            raise CampaignError(f"stage {stage_name!r} has no selection policy")
        policy = stage.selection_policy
        candidates, excluded = self._selection_population(stage_name, policy)
        if not candidates:
            raise CampaignError(f"stage {stage_name!r} has no seed-complete candidate")
        ranked = sorted(candidates, key=lambda item: self._rank_key(item, policy.direction))
        keep = (
            policy.retain_count
            if policy.retain_count is not None
            else max(1, math.ceil(len(ranked) * float(policy.retain_fraction)))
        )
        selected_candidates = ranked[:keep]
        if policy.tie_policy == "retain_all_at_cutoff" and keep < len(ranked):
            cutoff = (
                float(ranked[keep - 1]["median"]),
                float(ranked[keep - 1]["worst_seed"]),
            )
            selected_candidates.extend(
                candidate
                for candidate in ranked[keep:]
                if (
                    float(candidate["median"]),
                    float(candidate["worst_seed"]),
                )
                == cutoff
            )
        cells_by_id = {cell.cell_id: cell for cell in stage.cells}
        selections = [
            FrozenSelection.from_cells(
                policy,
                [cells_by_id[cell_id] for cell_id in candidate["cell_ids"]],
                [item["metric"] for item in candidate["observations"]],
                candidate["qualification_files"],
            )
            for candidate in selected_candidates
        ]
        payload = {
            "schema": SELECTION_SCHEMA,
            "stage": stage_name,
            "policy": policy.to_dict(),
            "ranked": ranked,
            "excluded": excluded,
            "selected": [selection.to_dict() for selection in selections],
        }
        destination = (
            Path(out)
            if out is not None
            else self.root / "selections" / f"{stage_name}.json"
        )
        _write_json(destination, payload)
        return payload

    def select_family_root(
        self, family_name: str, *, out: str | Path | None = None
    ) -> dict[str, Any]:
        blueprint = Phase2Blueprint.from_manifest(_read_json(self.blueprint_path))
        matches = [
            family for family in blueprint.comparator_families if family.name == family_name
        ]
        if len(matches) != 1:
            raise CampaignError(f"unknown comparator family {family_name!r}")
        stage = blueprint.initial_stage
        original = next(
            item for item in self.plan.stages if item.name == stage.name
        )
        policy = matches[0].root_selection_policy
        if original.selection_policy == policy:
            return self.select_stage(original.name, out=out)
        # Root policies filter the shared anchor stage by recipe name.
        proxy = type(original)(
            original.name,
            original.cells,
            original.depends_on,
            original.gate,
            policy,
            original.execution_duplicate_policy,
            original.elided_execution_duplicates,
            original.conditional_elision_reason,
            original.elided_conditional_variants,
        )
        plan = self.plan
        stages = tuple(proxy if item.name == original.name else item for item in plan.stages)
        temporary = StudyPlan(plan.name, plan.phase, stages)
        original_plan = _read_json(self.plan_path)
        try:
            _write_json(self.plan_path, temporary.to_manifest())
            return self.select_stage(original.name, out=out)
        finally:
            _write_json(self.plan_path, original_plan)

    def select_family_revisit_inputs(
        self, family_name: str, *, out: str | Path | None = None
    ) -> dict[str, Any]:
        blueprint = Phase2Blueprint.from_manifest(_read_json(self.blueprint_path))
        family = next(
            (
                item
                for item in blueprint.comparator_families
                if item.name == family_name
            ),
            None,
        )
        if family is None:
            raise CampaignError(f"unknown comparator family {family_name!r}")
        all_candidates: list[dict[str, Any]] = []
        excluded: list[dict[str, Any]] = []
        for stage_name in family.revisit.source_rounds:
            candidates, rejected = self._selection_population(
                stage_name, family.revisit.nomination_policy
            )
            all_candidates.extend(candidates)
            excluded.extend(rejected)
        if not all_candidates:
            raise CampaignError(f"family {family_name!r} has no revisit candidates")
        ranked = sorted(
            all_candidates,
            key=lambda item: self._rank_key(
                item, family.revisit.nomination_policy.direction
            ),
        )
        unique: list[dict[str, Any]] = []
        seen: set[str] = set()
        for candidate in ranked:
            cells = [self._require_cell(cell_id) for cell_id in candidate["cell_ids"]]
            signature = cells[0].recipe_id
            if signature in seen:
                continue
            seen.add(signature)
            unique.append(candidate)
        selected_candidates = unique[: family.revisit.top_k]
        if len(selected_candidates) != family.revisit.top_k:
            raise CampaignError("family revisit lacks enough distinct candidates")
        selections = []
        for candidate in selected_candidates:
            cells = [self._require_cell(cell_id) for cell_id in candidate["cell_ids"]]
            selections.append(
                FrozenSelection.from_cells(
                    family.revisit.nomination_policy,
                    cells,
                    [item["metric"] for item in candidate["observations"]],
                    candidate["qualification_files"],
                )
            )
        payload = {
            "schema": SELECTION_SCHEMA,
            "family": family_name,
            "ranked": ranked,
            "excluded": excluded,
            "selected": [selection.to_dict() for selection in selections],
        }
        destination = (
            Path(out)
            if out is not None
            else self.root / "selections" / f"{family_name}-revisit.json"
        )
        _write_json(destination, payload)
        return payload

    @staticmethod
    def phase1_decision_from_manifest(
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        if payload.get("schema") != PHASE1_DECISION_SCHEMA:
            raise CampaignError("unsupported Phase-1 decision")
        if payload.get("go") is not True:
            raise CampaignError("Phase-1 decision does not authorize Phase 2")
        if not isinstance(payload.get("phase1_transfer"), Mapping):
            raise CampaignError("Phase-1 decision lacks its method transfer")
        return dict(payload)

    def freeze_phase1_decision(
        self,
        *,
        scope_narrowing: Mapping[str, str] | None = None,
        out: str | Path | None = None,
    ) -> dict[str, Any]:
        plan = self.plan
        if plan.phase is not Phase.PHASE1:
            raise CampaignError("freeze-phase1 requires a Phase-1 campaign")
        blueprint = Phase1Blueprint.from_manifest(_read_json(self.blueprint_path))
        smoke_values = {
            bool(cell.decision_map["runtime.smoke"]) for cell in plan.cells
        }
        if len(smoke_values) != 1:
            raise CampaignError("Phase-1 campaign mixes smoke and scientific cells")
        smoke = smoke_values.pop()
        records = {record.cell_id: record for record in self.records()}
        cells: list[dict[str, Any]] = []
        for cell in plan.cells:
            record = records[cell.cell_id]
            if record.state not in {RunState.QUALIFIED, RunState.PROMOTED}:
                raise CampaignError(f"Phase-1 cell is incomplete: {cell.cell_id}")
            qualification = self._qualification(record)
            ref = record.artifact_map["qualification"]
            cells.append(
                {
                    "cell_id": cell.cell_id,
                    "qualification_file": ref.path,
                    "qualification": qualification,
                }
            )
        final_stage = plan.stages[-1]
        baseline = [
            cell
            for cell in final_stage.cells
            if cell.decision_map.get("factor.robustness") == "baseline"
        ]
        if not baseline:
            raise CampaignError("Phase-1 confirmation has no baseline carrier")
        scientific_go = all(
            isinstance(self._qualification(records[cell.cell_id]).get("scientific_outcome"), Mapping)
            and self._qualification(records[cell.cell_id])["scientific_outcome"].get("passed")
            is True
            for cell in baseline
        )
        selection_chain = [
            item
            for path in sorted((self.root / "selections").glob("*.json"))
            for item in _read_json(path).get("selected", ())
            if isinstance(item, Mapping)
        ]
        campaign_manifest = {
            "source_phase1_plan_id": plan.plan_id,
            "source_phase1_blueprint_id": blueprint.blueprint_id,
            "plan": plan.to_manifest(),
            "blueprint": blueprint.to_manifest(),
            "cells": cells,
            "selection_chain": selection_chain,
            "confirmation": {
                "go": scientific_go,
                "scope_narrowing": dict(scope_narrowing or {}),
            },
        }
        transfer = build_phase1_transfer(campaign_manifest)
        payload = {
            "schema": PHASE1_DECISION_SCHEMA,
            "decision_id": f"phase1-decision:{plan.name}",
            "go": scientific_go or smoke,
            "scientific_go": scientific_go,
            "smoke": smoke,
            "phase1_campaign_manifest": campaign_manifest,
            "phase1_transfer": transfer,
        }
        destination = (
            Path(out)
            if out is not None
            else self.root / "decisions" / "phase2-authorization.json"
        )
        _write_json(destination, payload)
        return payload

    @staticmethod
    def panel_decision_from_manifest(
        payload: Mapping[str, Any],
    ) -> FrozenPanelDecision:
        raw = payload.get("panel", payload)
        if not isinstance(raw, Mapping):
            raise CampaignError("panel decision is malformed")
        try:
            return FrozenPanelDecision.from_dict(raw)
        except (KeyError, TypeError, ValueError, StudyError) as exc:
            raise CampaignError(f"invalid panel decision: {exc}") from exc

    def _qualified_candidate_groups(self) -> list[dict[str, Any]]:
        groups: dict[str, list[CellSpec]] = {}
        for cell in self.plan.cells:
            groups.setdefault(cell.candidate_id, []).append(cell)
        result: list[dict[str, Any]] = []
        for candidate_id, cells in groups.items():
            ordered = tuple(sorted(cells, key=lambda item: item.seed))
            records = [self.record(cell.cell_id) for cell in ordered]
            if any(
                record.state not in {RunState.QUALIFIED, RunState.PROMOTED}
                for record in records
            ):
                continue
            qualifications = [self._qualification(record) for record in records]
            if any(
                qualification.get("qualified") is not True
                or qualification.get("promotion_eligible") is not True
                for qualification in qualifications
            ):
                continue
            values: list[float] = []
            for qualification in qualifications:
                try:
                    value = _lookup(
                        qualification,
                        "selection_metrics.fixed_rate.negative_mean_raw_fvu",
                    )
                except CampaignError:
                    value = qualification.get("selection_metrics", {}).get(
                        "negative_mean_raw_fvu"
                    )
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    values.append(float(value))
            if len(values) != len(ordered):
                continue
            result.append(
                {
                    "candidate_id": candidate_id,
                    "cells": ordered,
                    "records": records,
                    "score": float(median(values)),
                }
            )
        return result

    def freeze_panel(
        self, *, out: str | Path | None = None
    ) -> dict[str, Any]:
        plan = self.plan
        if plan.phase is not Phase.PHASE2:
            raise CampaignError("freeze-panel requires a Phase-2 campaign")
        blueprint = Phase2Blueprint.from_manifest(_read_json(self.blueprint_path))
        candidates = self._qualified_candidate_groups()
        panel_slots = __import__(
            "block_crosscoder_experiment.studies",
            fromlist=["build_phase3_blueprint"],
        ).build_phase3_blueprint(smoke=bool(plan.cells[0].decision_map["runtime.smoke"])).panel_slots
        entries: list[FrozenPanelEntry] = []
        used: set[str] = set()
        for slot in panel_slots:
            if slot.role == "selected_finalist":
                matches = [
                    item
                    for item in candidates
                    if all(
                        cell.decision_map["evaluation.split"] == "confirmation"
                        and cell.decision_map["data.normalization"] == "scalar_rms"
                        for cell in item["cells"]
                    )
                ]
            else:
                family = str(slot.comparator_family_name)
                matches = [
                    item
                    for item in candidates
                    if family in item["cells"][0].stage
                    or family in item["cells"][0].recipe_name
                ]
            matches = [
                item for item in matches if item["candidate_id"] not in used
            ]
            if not matches:
                raise CampaignError(f"no qualified candidate for panel slot {slot.name!r}")
            chosen = max(matches, key=lambda item: item["score"])
            used.add(chosen["candidate_id"])
            cells = chosen["cells"]
            records = chosen["records"]
            qualification_files = [
                record.artifact_map["qualification"].path for record in records
            ]
            selection_ids = tuple(
                dict.fromkeys(
                    selection_id
                    for cell in cells
                    for selection_id in cell.decision_map[
                        "selection.upstream_selection_ids"
                    ]
                )
            )
            entries.append(
                FrozenPanelEntry.from_cells(
                    panel_slot=slot.name,
                    role=slot.role,
                    source_cells=cells,
                    selection_ids=selection_ids or ("selection:reviewed",),
                    qualification_files=qualification_files,
                    confirmation_files=(
                        qualification_files
                        if slot.role == "selected_finalist"
                        else ()
                    ),
                )
            )
        decision = FrozenPanelDecision(
            source_phase2_plan_id=plan.plan_id,
            source_phase2_blueprint_id=blueprint.blueprint_id,
            entries=tuple(entries),
        )
        payload = {
            "schema": PANEL_DECISION_SCHEMA,
            "panel": decision.to_dict(),
            "phase2_campaign_manifest": {
                "plan": plan.to_manifest(),
                "blueprint": blueprint.to_manifest(),
                "smoke": bool(plan.cells[0].decision_map["runtime.smoke"]),
                "qualifications": {
                    cell.cell_id: self._qualification(self.record(cell.cell_id))
                    for entry in entries
                    for cell in entry.source_cells
                },
            },
        }
        destination = (
            Path(out)
            if out is not None
            else self.root / "decisions" / "phase3-panel.json"
        )
        _write_json(destination, payload)
        return payload

    def status(self) -> dict[str, Any]:
        plan = self.plan
        counts: dict[str, int] = {}
        for record in self.records():
            counts[record.state.value] = counts.get(record.state.value, 0) + 1
        return {
            "root": str(self.root.resolve()),
            "plan_id": plan.plan_id,
            "phase": plan.phase.value,
            "counts": counts,
            "runnable": len(self.runnable_cell_ids()),
            "resume_required": sum(
                record.state is RunState.RUNNING for record in self.records()
            ),
            "failed_retry_required": counts.get(RunState.FAILED.value, 0),
        }

    def reconcile(self, stale_after: float = 0.0) -> dict[str, Any]:
        del stale_after
        rebuilt = 0
        for cell in self.plan.cells:
            if not self.state_path(cell.cell_id).exists():
                _write_json(
                    self.state_path(cell.cell_id),
                    CampaignRecord(cell.cell_id, RunState.PLANNED).to_dict(),
                )
                rebuilt += 1
        return {"state_files_rebuilt": rebuilt}


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
    def __init__(
        self,
        *,
        command: Sequence[str],
        cwd: Path,
        environment: Mapping[str, str],
    ) -> None:
        self._stderr = tempfile.TemporaryFile(mode="w+t", encoding="utf-8")
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
        self._pgid = self._process.pid

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
                f"cell worker exited before {stage}: {self._stderr_tail()}"
            )
        request = json.dumps(
            {
                "stage": stage,
                "artifacts_out": str(artifacts_out),
                "resume": resume,
            }
        )
        assert self._process.stdin is not None
        assert self._process.stdout is not None
        self._process.stdin.write(request + "\n")
        self._process.stdin.flush()
        response_raw = self._process.stdout.readline()
        if not response_raw:
            raise CampaignError(
                f"cell worker exited during {stage}: {self._stderr_tail()}"
            )
        response = json.loads(response_raw)
        if response.get("ok") is not True:
            raise CampaignError(
                f"{response.get('error_type', 'CellExecutionError')} during "
                f"{stage}: {response.get('error', 'unknown worker failure')}"
            )

    def close(self) -> None:
        if self._process.poll() is None and self._process.stdin is not None:
            try:
                self._process.stdin.write('{"command":"close"}\n')
                self._process.stdin.flush()
                self._process.wait(timeout=5)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                try:
                    os.killpg(self._pgid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(self._pgid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
        if self._process.stdout is not None:
            self._process.stdout.close()
        self._stderr.close()


class CampaignRunner:
    """Drive eligible cells through the five ordinary execution stages."""

    def __init__(
        self,
        campaign: Campaign,
        *,
        python: str = sys.executable,
        module: str = CANONICAL_CELL_MODULE,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self.campaign = campaign
        self.python = python
        self.module = module
        self.env = dict(env or {})

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
        selected = (
            list(
                self.campaign.runnable_cell_ids(
                    include_failed=resume,
                    include_resume_required=resume,
                )
            )
            if cell_ids is None
            else list(cell_ids)
        )
        if limit is not None:
            selected = selected[:limit]
        completed = failed = skipped = 0
        for cell_id in selected:
            try:
                result = self._run_cell(
                    cell_id, resume=resume, stop_after=stop_after
                )
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
        self, cell_id: str, *, resume: bool, stop_after: str | None
    ) -> RunState:
        with self.campaign.lock(cell_id) as cell_lock:
            record = self.campaign.record(cell_id)
            if record.state is RunState.FAILED:
                if not resume:
                    return record.state
                record = self.campaign.retry(cell_id)
            if record.state is RunState.RUNNING and not resume:
                return record.state
            if record.state in {RunState.QUALIFIED, RunState.PROMOTED}:
                return record.state
            worker: _PersistentCellWorker | None = None
            try:
                for stage in self._remaining_stages(record.state):
                    if stage == "train":
                        self.campaign.transition(
                            cell_id,
                            RunState.RUNNING,
                            message="training started",
                        )
                    try:
                        if worker is None and self.module == CANONICAL_CELL_MODULE:
                            worker = self._start_worker(cell_id)
                            cell_lock.bind_worker(pid=worker.pid, pgid=worker.pgid)
                        artifacts = self._invoke(
                            cell_id,
                            stage,
                            resume=resume,
                            worker=worker,
                        )
                        self.campaign.transition(
                            cell_id,
                            STAGE_TARGETS[stage],
                            artifacts=artifacts,
                            message=f"{stage} completed",
                        )
                    except (
                        ArtifactError,
                        CampaignError,
                        OSError,
                        subprocess.SubprocessError,
                    ) as exc:
                        self.campaign.transition(
                            cell_id,
                            RunState.FAILED,
                            message=f"{stage} failed: {exc}",
                        )
                        return RunState.FAILED
                    if stage == stop_after:
                        break
            finally:
                if worker is not None:
                    worker.close()
            return self.campaign.record(cell_id).state

    def _start_worker(self, cell_id: str) -> _PersistentCellWorker:
        environment = os.environ.copy()
        environment.update(self.env)
        environment["BSC_CAMPAIGN_ROOT"] = str(self.campaign.root.resolve())
        source_root = str(Path(__file__).resolve().parent.parent)
        environment["PYTHONPATH"] = os.pathsep.join(
            item
            for item in (source_root, environment.get("PYTHONPATH", ""))
            if item
        )
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
        return () if start is None else tuple(STAGE_TARGETS)[start:]

    def _invoke(
        self,
        cell_id: str,
        stage: str,
        *,
        resume: bool,
        worker: _PersistentCellWorker | None,
    ) -> tuple[ArtifactRef, ...]:
        output = (
            self.campaign.cell_dir(cell_id)
            / "stage-artifacts"
            / f"{stage}-{uuid.uuid4().hex}.json"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        if worker is None:
            command = [
                self.python,
                "-m",
                self.module,
                "--cell",
                str(self.campaign.cell_manifest_path(cell_id)),
                "--stage",
                stage,
                "--artifacts-out",
                str(output),
            ]
            if resume:
                command.append("--resume")
            environment = os.environ.copy()
            environment.update(self.env)
            environment["BSC_CAMPAIGN_ROOT"] = str(self.campaign.root.resolve())
            source_root = str(Path(__file__).resolve().parent.parent)
            environment["PYTHONPATH"] = os.pathsep.join(
                item
                for item in (source_root, environment.get("PYTHONPATH", ""))
                if item
            )
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
                    f"cell executor exited {completed.returncode} during {stage}: {tail}"
                )
        else:
            worker.invoke(stage=stage, artifacts_out=output, resume=resume)
        return self._load_artifact_manifest(cell_id, stage, output)

    def _load_artifact_manifest(
        self, cell_id: str, stage: str, path: Path
    ) -> tuple[ArtifactRef, ...]:
        payload = _read_json(path)
        if payload.get("schema") != ARTIFACT_SCHEMA:
            raise ArtifactError(f"wrong stage artifact schema at {path}")
        if payload.get("cell_id") != cell_id or payload.get("stage") != stage:
            raise ArtifactError("stage artifact manifest names the wrong cell or stage")
        items = payload.get("artifacts")
        if not isinstance(items, list):
            raise ArtifactError("stage artifact manifest needs an artifacts list")
        refs = tuple(ArtifactRef.from_dict(item) for item in items)
        kinds = frozenset(ref.kind for ref in refs)
        if kinds != EXPECTED_STAGE_ARTIFACTS[stage] or len(kinds) != len(refs):
            raise ArtifactError(
                f"{stage} artifacts must be exactly "
                f"{sorted(EXPECTED_STAGE_ARTIFACTS[stage])}"
            )
        for ref in refs:
            ref.verify(self.campaign.root)
        return refs


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
