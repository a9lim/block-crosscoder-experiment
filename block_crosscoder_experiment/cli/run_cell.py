"""Execute one immutable stage of a resolved study cell.

This is the implementation behind :class:`~block_crosscoder_experiment.campaign.CampaignRunner`.
Every scientific choice
comes from the content-addressed cell manifest, and every durable output is
written once and subsequently verified rather than overwritten.

The checkpoint is a *trained* artifact.  Calibration never mutates it.  The
calibration artifact is a separately serialized codec containing the frozen
inference threshold and a binding to the checkpoint.  Evaluation reloads both
artifacts, and qualification binds their externally recomputed hashes.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import random
import struct
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import torch
import numpy as np

from block_crosscoder_experiment.campaign import (
    ARTIFACT_SCHEMA,
    CAMPAIGN_SCHEMA,
    Campaign,
    CampaignError,
    QUALIFICATION_SCHEMA,
)
from block_crosscoder_experiment.codec import (
    Codec,
    CodecSpec,
    _decode_trusted_packet_events_q_chunks,
    _encode_batch_events,
    decode_batch,
    encode_batch,
    estimate_calibration_peak_bytes,
    evaluate_rd,
    fit_codec,
)
from block_crosscoder_experiment.evaluation import (
    evaluate_shared_code,
    load_trained_model,
)
from block_crosscoder_experiment.model import BSCConfig, BlockCrosscoder, bsc_loss
from block_crosscoder_experiment.phase1 import (
    FelSyntheticConfig,
    LadderSyntheticConfig,
    Phase1Dataset,
    make_fel_dataset,
    make_ladder_dataset,
)
from block_crosscoder_experiment.store import (
    MANIFEST_NAME,
    StoreReader,
    Whitener,
    prefetch_batches,
)
from block_crosscoder_experiment.studies import (
    CellSpec,
    Phase,
    Phase1Blueprint,
    Phase2Blueprint,
    Phase3Blueprint,
    PHASE2_INELIGIBLE_SELECTION_SCORE,
    PHASE2_SELECTION_METRIC_KEY,
    StudyError,
    StudyPlan,
    build_phase1_blueprint,
    build_phase1_plan,
    build_phase2_blueprint,
    build_phase2_plan,
    build_phase3_blueprint,
    build_phase3_plan,
    canonical_json,
)
from block_crosscoder_experiment.trainer import TrainConfig, Trainer


EVALUATION_SCHEMA = "bsc-evaluation-v1"
PREPARATION_SCHEMA = "bsc-preparation-v1"
TRAINING_REPORT_SCHEMA = "bsc-training-report-v1"
EXECUTOR_SCHEMA = "bsc-cell-executor-v2"
STAGES = ("prepare", "train", "calibrate", "evaluate", "qualify")
_VERIFIED_STORE_BINDINGS: set[tuple[str, str, str, str]] = set()
_SYNTHETIC_NORMALIZATION_CACHE: dict[
    tuple[int, torch.dtype, torch.device],
    tuple[Mapping[str, Any], torch.Tensor, torch.Tensor],
] = {}
_DEPLOYABLE_CODEC_KEYS = {
    "format_version",
    "schema",
    "cell_id",
    "checkpoint_sha256",
    "calibration_sha256",
    "preparation_sha256",
    "model_cfg",
    "model_state",
    "codec_payload",
    "training_summary",
    "normalization",
    "raw_calibration_mean",
    "raw_calibration_mean_fit_tokens",
    "rate_contract",
    "artifact_sha256",
}
TIME_SHARING_HEADER_BYTES = 32
TIME_SHARING_BUNDLE_SCHEMA = "bsc-deployment-schedule-bundle-v1"
TIME_SHARING_HEADER_LAYOUT = ">QQQII"
TIME_SHARING_HEADER_LAYOUT_DESCRIPTION = (
    "binding_magic_u64,horizon_u64,upper_count_u64,lower_mode_u32,upper_mode_u32"
)


class CellExecutionError(RuntimeError):
    """A resolved cell cannot safely complete the requested stage."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_payload_digest(value: Any) -> str:
    """Canonical digest for nested JSON scalars and dense tensor payloads."""

    digest = hashlib.sha256()

    def add(item: Any) -> None:
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor\0")
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(canonical_json(list(tensor.shape)).encode("ascii"))
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict\0")
            for key in sorted(item):
                digest.update(str(key).encode("utf-8") + b"\0")
                add(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(b"seq\0")
            for child in item:
                add(child)
        else:
            digest.update(
                json.dumps(item, sort_keys=True, allow_nan=False, default=str).encode(
                    "utf-8"
                )
            )
            digest.update(b"\0")

    add(value)
    return digest.hexdigest()


def _verify_store_reader_once(
    reader: StoreReader,
    root: Path,
    split: str,
    *,
    expected_row_identity: Mapping[str, int] | None = None,
) -> None:
    manifest = root / split / MANIFEST_NAME
    manifest_sha256 = _sha256(manifest)
    row_identity_digest = (
        "generic"
        if expected_row_identity is None
        else hashlib.sha256(
            canonical_json(dict(expected_row_identity)).encode("utf-8")
        ).hexdigest()
    )
    key = (str(root.resolve()), split, manifest_sha256, row_identity_digest)
    generic_key = (str(root.resolve()), split, manifest_sha256, "generic")
    if key in _VERIFIED_STORE_BINDINGS:
        return

    def stat_record(path: Path) -> dict[str, int | str]:
        status = path.stat()
        return {
            "path": str(path.resolve()),
            "size_bytes": status.st_size,
            "mtime_ns": status.st_mtime_ns,
            "ctime_ns": status.st_ctime_ns,
            "device": status.st_dev,
            "inode": status.st_ino,
        }

    fingerprint = {
        "manifest": stat_record(manifest),
        "shards": [
            stat_record(root / split / str(record["file"]))
            for record in reader.manifest["shards"]
        ],
    }
    cache_base = os.environ.get("BSC_VERIFICATION_CACHE_ROOT")
    if cache_base is None:
        campaign_root = os.environ.get("BSC_CAMPAIGN_ROOT")
        if campaign_root is None:
            # A persistent receipt in a shared temporary directory could be
            # forged by another local user. Ad-hoc callers therefore receive
            # only the process-local cache; registered campaigns persist
            # receipts under their authenticated root.
            reader.verify(expected_row_identity=expected_row_identity)
            _VERIFIED_STORE_BINDINGS.add(key)
            _VERIFIED_STORE_BINDINGS.add(generic_key)
            return
        cache_root = Path(campaign_root) / ".store-verification"
    else:
        cache_root = Path(cache_base)
    cache_key = hashlib.sha256(
        canonical_json(
            {
                "root": str(root.resolve()),
                "split": split,
                "manifest_sha256": manifest_sha256,
                "expected_row_identity": expected_row_identity,
            }
        ).encode("utf-8")
    ).hexdigest()
    receipt_path = cache_root / f"{cache_key}.json"
    expected_receipt = {
        "schema": "bsc-store-verification-receipt-v1",
        "root": str(root.resolve()),
        "split": split,
        "manifest_sha256": manifest_sha256,
        "manifest_content_sha256": reader.manifest.get("manifest_sha256"),
        "content_stream_sha256": reader.manifest.get("content_stream_sha256"),
        "row_stream_sha256": reader.manifest.get("row_stream_sha256"),
        "n_tokens": reader.n_tokens,
        "expected_row_identity": expected_row_identity,
        "stat_fingerprint": fingerprint,
    }
    if receipt_path.is_file():
        try:
            receipt = json.loads(receipt_path.read_text())
        except (OSError, json.JSONDecodeError):
            receipt = None
        if receipt == expected_receipt:
            _VERIFIED_STORE_BINDINGS.add(key)
            _VERIFIED_STORE_BINDINGS.add(generic_key)
            return
    verified_tokens = reader.verify(expected_row_identity=expected_row_identity)
    if verified_tokens != reader.n_tokens:
        raise CellExecutionError(
            f"store verification returned {verified_tokens} rows, expected "
            f"{reader.n_tokens}"
        )
    _atomic_bytes(receipt_path, _json_bytes(expected_receipt))
    _VERIFIED_STORE_BINDINGS.add(key)
    _VERIFIED_STORE_BINDINGS.add(generic_key)


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    # Indented output remains reviewable; canonical_json is used for IDs.
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _atomic_bytes(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    body = _json_bytes(payload)
    if path.exists():
        if path.read_bytes() != body:
            raise CellExecutionError(
                f"immutable artifact already exists with different content: {path}"
            )
        return
    _atomic_bytes(path, body)


def _write_immutable_bytes(path: Path, body: bytes) -> None:
    if path.exists():
        if path.read_bytes() != body:
            raise CellExecutionError(
                f"immutable artifact already exists with different content: {path}"
            )
        return
    _atomic_bytes(path, body)


def _read_object(path: Path, *, label: str = "JSON") -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CellExecutionError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CellExecutionError(f"{label} must be a JSON object: {path}")
    return payload


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(root.resolve()))
    except ValueError:
        return str(resolved)


def _artifact_entry(kind: str, path: Path, *, root: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CellExecutionError(f"stage did not produce {kind}: {path}")
    return {
        "kind": kind,
        "path": _relative(path, root),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _emit_stage_manifest(
    path: Path,
    *,
    cell_id: str,
    stage: str,
    root: Path,
    artifacts: Sequence[tuple[str, Path]],
) -> None:
    entries = [_artifact_entry(kind, item, root=root) for kind, item in artifacts]
    if len({entry["kind"] for entry in entries}) != len(entries):
        raise CellExecutionError("a stage cannot emit the same artifact kind twice")
    _atomic_bytes(
        path,
        _json_bytes(
            {
                "schema": ARTIFACT_SCHEMA,
                "cell_id": cell_id,
                "stage": stage,
                "artifacts": entries,
            }
        ),
    )


class _Context:
    def __init__(self, cell_path: Path, artifacts_out: Path, stage: str) -> None:
        manifest = _read_object(cell_path, label="cell manifest")
        try:
            self.cell = CellSpec.from_manifest(manifest)
        except (KeyError, TypeError, ValueError) as exc:
            raise CellExecutionError(
                f"invalid cell manifest {cell_path}: {exc}"
            ) from exc
        if self.cell.phase is Phase.PHASE2 and any(
            self.cell.decision_map.get(name) == "unbound-preview"
            for name in (
                "provenance.phase1_decision_id",
                "provenance.phase1_transfer_id",
            )
        ):
            raise CellExecutionError(
                "unbound-preview Phase-2 cells are planning estimates only; "
                "execute a campaign registered from an authenticated Phase-1 decision"
            )
        self.cell_path = cell_path.resolve()
        self.stage = stage
        self.artifacts_out = artifacts_out.resolve()
        root_raw = os.environ.get("BSC_CAMPAIGN_ROOT")
        if root_raw is None:
            raise CellExecutionError(
                "BSC_CAMPAIGN_ROOT is required; execute cells through `bsc matrix run`"
            )
        self.root = Path(root_raw).resolve()
        try:
            self.cell_path.relative_to(self.root)
            self.artifacts_out.relative_to(self.root)
        except ValueError as exc:
            raise CellExecutionError(
                "cell and stage manifest must live inside BSC_CAMPAIGN_ROOT"
            ) from exc
        campaign = Campaign(self.root)
        try:
            active_plan_payload = _read_object(
                self.root / "plan.json", label="campaign plan"
            )
            active_plan = StudyPlan.from_manifest(active_plan_payload)
            if canonical_json(active_plan_payload) != canonical_json(
                active_plan.to_manifest()
            ):
                raise CellExecutionError("active campaign plan is noncanonical")
            expected_cell_path = campaign.cell_manifest_path(
                self.cell.cell_id
            ).resolve()
            matching_cells = [
                cell for cell in active_plan.cells if cell.cell_id == self.cell.cell_id
            ]
            if (
                active_plan.phase is not self.cell.phase
                or matching_cells != [self.cell]
                or self.cell_path != expected_cell_path
            ):
                raise CellExecutionError(
                    "cell is not an exact member of the active campaign plan"
                )
            history_matches: list[StudyPlan] = []
            for history_path in campaign.plans_dir.glob("*.json"):
                history_payload = _read_object(
                    history_path, label="immutable plan history"
                )
                history = StudyPlan.from_manifest(history_payload)
                if history.plan_id == active_plan.plan_id:
                    if canonical_json(history_payload) != canonical_json(
                        active_plan.to_manifest()
                    ):
                        raise CellExecutionError(
                            "active plan differs from immutable plan history"
                        )
                    history_matches.append(history)
            if history_matches != [active_plan]:
                raise CellExecutionError(
                    "active campaign lacks one exact immutable plan-history binding"
                )
            extension_events = tuple(
                event
                for event in campaign.events()
                if event.get("event") == "plan_extension"
            )
            if extension_events:
                metadata = extension_events[-1].get("metadata")
                if (
                    not isinstance(metadata, Mapping)
                    or metadata.get("plan_id") != active_plan.plan_id
                ):
                    raise CellExecutionError(
                        "active plan is not the journal's committed extension tip"
                    )
        except (
            CampaignError,
            CellExecutionError,
            KeyError,
            OSError,
            StudyError,
            TypeError,
            ValueError,
        ) as exc:
            raise CellExecutionError(
                f"execution requires an exact active campaign binding: {exc}"
            ) from exc

        if self.cell.phase is Phase.PHASE1:
            try:
                blueprint_payload = _read_object(
                    self.root / "blueprint.json", label="Phase-1 blueprint"
                )
                blueprint = Phase1Blueprint.from_manifest(blueprint_payload)
                smoke = self.cell.decision_map.get("runtime.smoke") is True
                if smoke:
                    # Focused smoke fixtures remain non-scientific and
                    # nonpromotable, but still require an exact registered
                    # blueprint/plan/history/cell binding above.
                    expected_blueprint = blueprint
                    initial_stages = blueprint.initial_stages
                else:
                    expected_blueprint = build_phase1_blueprint(
                        blueprint.seeds, smoke=False
                    )
                    initial_stages = build_phase1_plan(
                        blueprint.seeds, smoke=False
                    ).stages
                expected_stage_names = (
                    *(stage.name for stage in blueprint.initial_stages),
                    *(round_spec.name for round_spec in blueprint.rounds),
                )
                if (
                    blueprint != expected_blueprint
                    or canonical_json(blueprint_payload)
                    != canonical_json(expected_blueprint.to_manifest())
                    or active_plan.stages[: len(initial_stages)] != initial_stages
                    or tuple(stage.name for stage in active_plan.stages)
                    != expected_stage_names[: len(active_plan.stages)]
                    or (not extension_events and active_plan.stages != initial_stages)
                ):
                    raise CellExecutionError(
                        "Phase-1 plan/blueprint is not its canonical campaign prefix"
                    )
            except (
                CellExecutionError,
                KeyError,
                OSError,
                StudyError,
                TypeError,
                ValueError,
            ) as exc:
                raise CellExecutionError(
                    f"Phase-1 cell is not bound to its canonical campaign: {exc}"
                ) from exc
        if self.cell.phase is Phase.PHASE2:
            try:
                phase1_decision = Campaign.phase1_decision_from_manifest(
                    _read_object(
                        self.root / "phase1-decision.json",
                        label="Phase-1 decision",
                    )
                )
                blueprint = Phase2Blueprint.from_manifest(
                    _read_object(
                        self.root / "blueprint.json", label="Phase-2 blueprint"
                    )
                )
                smoke = self.cell.decision_map.get("runtime.smoke") is True
                expected_blueprint = build_phase2_blueprint(
                    blueprint.seeds,
                    smoke=smoke,
                    phase1_decision=phase1_decision,
                )
                expected_initial_plan = build_phase2_plan(
                    blueprint.seeds,
                    smoke=smoke,
                    phase1_decision=phase1_decision,
                )
            except (
                CampaignError,
                CellExecutionError,
                KeyError,
                OSError,
                StudyError,
                TypeError,
                ValueError,
            ) as exc:
                raise CellExecutionError(
                    "Phase-2 execution requires a campaign registered from an "
                    f"authenticated Phase-1 decision: {exc}"
                ) from exc
            if (
                active_plan.phase is not Phase.PHASE2
                or blueprint != expected_blueprint
                or canonical_json(blueprint.to_manifest())
                != canonical_json(
                    _read_object(
                        self.root / "blueprint.json", label="Phase-2 blueprint"
                    )
                )
                or active_plan.stages[0] != expected_initial_plan.stages[0]
                or matching_cells != [self.cell]
                or (not extension_events and active_plan != expected_initial_plan)
                or (
                    smoke and phase1_decision.get("authorizes_phase2_smoke") is not True
                )
                or (
                    not smoke
                    and (
                        phase1_decision.get("authorization_mode") != "scientific_go"
                        or phase1_decision.get("authorizes_phase2_scientific")
                        is not True
                    )
                )
            ):
                raise CellExecutionError(
                    "Phase-2 cell is not bound to the active authenticated campaign"
                )
        if self.cell.phase is Phase.PHASE3:
            try:
                panel_payload = _read_object(
                    self.root / "panel-decision.json", label="Phase-3 panel decision"
                )
                panel = Campaign.panel_decision_from_manifest(panel_payload)
                blueprint_payload = _read_object(
                    self.root / "blueprint.json", label="Phase-3 blueprint"
                )
                blueprint = Phase3Blueprint.from_manifest(blueprint_payload)
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
                if (
                    blueprint != expected_blueprint
                    or active_plan != expected_plan
                    or canonical_json(blueprint_payload)
                    != canonical_json(expected_blueprint.to_manifest())
                    or extension_events
                ):
                    raise CellExecutionError(
                        "Phase-3 plan/blueprint differs from its verified frozen panel"
                    )
            except (
                CampaignError,
                CellExecutionError,
                KeyError,
                OSError,
                StudyError,
                TypeError,
                ValueError,
            ) as exc:
                raise CellExecutionError(
                    f"Phase-3 cell is not bound to its frozen panel campaign: {exc}"
                ) from exc
        self.cell_dir = self.cell_path.parent
        self.outputs = self.cell_dir / "outputs"
        self.preparation = self.outputs / "preparation.json"
        self.checkpoint = self.outputs / "checkpoint.pt"
        self.training_report = self.outputs / "training-report.json"
        self.progress = self.outputs / "training-progress.pt"
        self.calibration = self.outputs / "calibration-codec.pt"
        self.deployment_codec = self.outputs / "deployable-codec.pt"
        self.deployment_schedules = self.outputs / "deployment-schedules.bin"
        self.calibration_record = self.outputs / "calibration-record.json"
        self.evaluation = self.outputs / "evaluation.json"
        self.qualification = self.outputs / "qualification.json"

    @property
    def values(self) -> dict[str, Any]:
        return self.cell.decision_map

    def state(self) -> tuple[str, dict[str, dict[str, Any]]]:
        path = self.cell_dir / "state.json"
        payload = _read_object(path, label="campaign state")
        if payload.get("schema") != CAMPAIGN_SCHEMA:
            raise CellExecutionError(f"wrong campaign state schema at {path}")
        if payload.get("cell_id") != self.cell.cell_id:
            raise CellExecutionError("campaign state is bound to a different cell")
        artifacts: dict[str, dict[str, Any]] = {}
        for raw in payload.get("artifacts", ()):
            if not isinstance(raw, dict) or not raw.get("kind"):
                raise CellExecutionError(
                    "campaign state has a malformed artifact entry"
                )
            kind = str(raw["kind"])
            if kind in artifacts:
                raise CellExecutionError(
                    f"campaign state repeats artifact kind {kind!r}"
                )
            artifacts[kind] = raw
        return str(payload.get("state")), artifacts

    def verify_ref(self, raw: Mapping[str, Any]) -> Path:
        path = Path(str(raw.get("path", "")))
        if not path.is_absolute():
            path = self.root / path
        if not path.is_file():
            raise CellExecutionError(f"prerequisite artifact disappeared: {path}")
        expected_size = int(raw.get("size_bytes", -1))
        if path.stat().st_size != expected_size:
            raise CellExecutionError(f"prerequisite artifact size mismatch: {path}")
        expected_hash = str(raw.get("sha256", ""))
        if _sha256(path) != expected_hash:
            raise CellExecutionError(f"prerequisite artifact hash mismatch: {path}")
        return path

    def prerequisites(self) -> dict[str, tuple[Path, str]]:
        expected_state = {
            "prepare": "planned",
            "train": "running",
            "calibrate": "trained",
            "evaluate": "calibrated",
            "qualify": "evaluated",
        }[self.stage]
        state, raw_refs = self.state()
        if state != expected_state:
            raise CellExecutionError(
                f"{self.stage} requires campaign state {expected_state!r}, got {state!r}"
            )
        required = {
            "prepare": (),
            "train": ("preparation", "prepare_manifest"),
            "calibrate": (
                "preparation",
                "prepare_manifest",
                "checkpoint",
                "training_report",
                "train_manifest",
            ),
            "evaluate": (
                "preparation",
                "prepare_manifest",
                "checkpoint",
                "training_report",
                "train_manifest",
                "calibration",
                "deployment_codec",
                "calibration_record",
                "calibrate_manifest",
            ),
            "qualify": (
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
            ),
        }[self.stage]
        missing = set(required).difference(raw_refs)
        if missing:
            raise CellExecutionError(
                f"{self.stage} is missing prerequisite artifacts {sorted(missing)}"
            )
        return {
            kind: (self.verify_ref(raw_refs[kind]), str(raw_refs[kind]["sha256"]))
            for kind in required
        }


def _device(ctx: _Context) -> torch.device:
    raw = ctx.values["runtime.device"]
    if not isinstance(raw, str) or not raw:
        raise CellExecutionError("runtime.device must be a nonempty device string")
    try:
        device = torch.device(raw)
    except RuntimeError as exc:
        raise CellExecutionError(f"invalid runtime.device {raw!r}: {exc}") from exc
    if device.type == "cuda" and not torch.cuda.is_available():
        raise CellExecutionError(
            "runtime.device requests CUDA but torch.cuda is unavailable"
        )
    smoke = ctx.values["runtime.smoke"]
    if not isinstance(smoke, bool):
        raise CellExecutionError("runtime.smoke must be boolean")
    if ctx.cell.phase is not Phase.PHASE1 and not smoke and device.type != "cuda":
        raise CellExecutionError(
            "non-smoke Phase 2/3 cells require runtime.device='cuda'"
        )
    return device


def _declared_device(values: Mapping[str, Any]) -> str:
    """Validate a device declaration without requiring hardware at prepare time."""

    raw = values["runtime.device"]
    if not isinstance(raw, str) or not raw:
        raise CellExecutionError("runtime.device must be a nonempty device string")
    try:
        device = torch.device(raw)
    except RuntimeError as exc:
        raise CellExecutionError(f"invalid runtime.device {raw!r}: {exc}") from exc
    smoke = values["runtime.smoke"]
    if not isinstance(smoke, bool):
        raise CellExecutionError("runtime.smoke must be boolean")
    return str(device)


def _implementation_identity() -> dict[str, Any]:
    package_root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    source_files = sorted(package_root.rglob("*.py"))
    for path in source_files:
        relative = path.relative_to(package_root)
        digest.update(str(relative).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    git_commit: str | None = None
    git_dirty: bool | None = None
    try:
        top = subprocess.run(
            ["git", "-C", str(package_root), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        git_commit = subprocess.run(
            ["git", "-C", top, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", top, "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        git_dirty = bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        pass
    dependencies: dict[str, str | None] = {}
    for distribution in ("numpy", "safetensors", "torch"):
        try:
            dependencies[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            dependencies[distribution] = None
    return {
        "executor_schema": EXECUTOR_SCHEMA,
        "python_source_sha256": digest.hexdigest(),
        "python_source_files": len(source_files),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "dependencies": dependencies,
    }


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise CellExecutionError(f"{name} must be a positive integer")
    return value


def _synthetic_counts(
    values: Mapping[str, Any],
) -> tuple[int, int, int, int, int, int]:
    train = _positive_int(values["data.train_tokens"], "data.train_tokens")
    unique = _positive_int(values["data.unique_tokens"], "data.unique_tokens")
    if unique > train:
        raise CellExecutionError("data.unique_tokens cannot exceed data.train_tokens")
    factor_calibration = _positive_int(
        values["data.synthetic_factor_calibration_examples"],
        "data.synthetic_factor_calibration_examples",
    )
    calibration = _positive_int(
        values["data.synthetic_calibration_examples"],
        "data.synthetic_calibration_examples",
    )
    development = _positive_int(
        values["data.synthetic_development_examples"],
        "data.synthetic_development_examples",
    )
    confirmation = _positive_int(
        values["data.synthetic_confirmation_examples"],
        "data.synthetic_confirmation_examples",
    )
    return (
        train,
        unique,
        factor_calibration,
        calibration,
        development,
        confirmation,
    )


def _synthetic_dataset(cell: CellSpec, split: str) -> Phase1Dataset:
    values = cell.decision_map
    if split not in {"train", "eval", "confirmation"}:
        raise CellExecutionError(f"unknown synthetic stream {split!r}")
    generator_split = "train" if split == "train" else "eval"
    kind = str(values["data.kind"])
    dims = tuple(int(item) for item in values["data.site_dims"])
    # Planted truth is a data contract.  Learner width/support contests must
    # consume byte-identical examples rather than quietly changing the DGP.
    factor_coordinate_dim = _positive_int(
        values["data.factor_coordinate_dim"], "data.factor_coordinate_dim"
    )
    active = _positive_int(
        values["data.active_factors_per_example"],
        "data.active_factors_per_example",
    )
    rank_profile = str(values["data.factor_rank_profile"])
    (
        train,
        unique,
        factor_calibration_examples,
        calibration_examples,
        development_examples,
        confirmation_examples,
    ) = _synthetic_counts(values)
    eval_examples = (
        factor_calibration_examples
        + calibration_examples
        + development_examples
        + confirmation_examples
    )
    generator_version = values["data.generator_version"]
    supported_generators = {
        "fel-structuring-sparsity-v1",
        "multisite-block-manifold-v1",
        "paper-bridge-v1",
    }
    if generator_version not in supported_generators:
        raise CellExecutionError(
            "unsupported data.generator_version " + repr(generator_version)
        )
    dgp_step = str(values["data.dgp_step"])
    presentation_order = str(values["data.presentation_order"])
    if presentation_order not in {
        "deterministic_epoch_shuffle",
        "cyclic_unshuffled",
    }:
        raise CellExecutionError(
            f"unsupported data.presentation_order {presentation_order!r}"
        )
    n_factors = _positive_int(values["data.n_factors"], "data.n_factors")
    structure_seed = int(values["random.structure_seed"])
    train_seed = int(values["random.train_data_seed"])
    eval_seed = int(
        values[
            "random.confirmation_data_seed"
            if split == "confirmation"
            else "random.eval_data_seed"
        ]
    )
    stochastic_seeds = {
        int(values["random.structure_seed"]),
        train_seed,
        int(values["random.eval_data_seed"]),
        int(values["random.confirmation_data_seed"]),
    }
    if len(stochastic_seeds) != 4:
        raise CellExecutionError(
            "synthetic structure/train/development/confirmation seeds must differ"
        )

    if (
        kind == "synthetic_manifold_superposition"
        or dgp_step
        in {
            "fel_manifold",
            "single_site",
        }
        or (kind == "synthetic_paper_bridge" and len(dims) == 1)
    ):
        if len(dims) != 1:
            raise CellExecutionError(f"DGP step {dgp_step!r} requires one site")
        if n_factors % 2:
            raise CellExecutionError("the Fel generator requires even data.n_factors")
        config = FelSyntheticConfig(
            ambient_dim=dims[0],
            n_factors=n_factors,
            active_per_example=min(active, n_factors),
            calibration_examples=factor_calibration_examples,
            train_unique_examples=unique,
            train_presentations=train,
            # One stateless held-out domain contains four explicitly declared
            # identity ranges.  The factor-calibration prefix freezes matching
            # and code alignment before any candidate-ranking rows are read.
            eval_unique_examples=eval_examples,
            eval_presentations=eval_examples,
            presentation_order=presentation_order,
            structure_seed=structure_seed,
            train_seed=train_seed,
            eval_seed=eval_seed,
        )
        return make_fel_dataset(config, split=generator_split)  # type: ignore[arg-type]

    if kind not in {"synthetic_multisite_manifold", "synthetic_paper_bridge"}:
        raise CellExecutionError(f"{kind!r} is not a Phase-1 synthetic data kind")
    if len(set(dims)) != 1:
        raise CellExecutionError(
            "the declared manifold ladder currently requires equal site widths"
        )
    step_aliases = {
        "support_only": "shared_support",
        "shared_coordinates": "baseline",
    }
    step = step_aliases.get(dgp_step, dgp_step)
    if step == "missingness" or float(values["data.missing_probability"]) != 0.0:
        raise CellExecutionError(
            "synthetic missingness is deferred until saved-codec and raw-space "
            "evaluation are mask-aware"
        )
    expected_rank_profile = (
        "cycle_1_to_factor_coordinate_dim"
        if step == "rank_heterogeneity"
        else "uniform_rank_2"
    )
    if rank_profile != expected_rank_profile:
        raise CellExecutionError(
            f"DGP step {step!r} requires data.factor_rank_profile="
            f"{expected_rank_profile!r}, got {rank_profile!r}"
        )
    config = LadderSyntheticConfig(
        step=step,
        n_sites=len(dims),
        d_model=dims[0],
        n_factors=n_factors,
        block_dim=min(factor_coordinate_dim, dims[0]),
        base_rank=min(2, factor_coordinate_dim, dims[0]),
        active_per_example=min(active, n_factors),
        scale_ratio=float(values["data.site_scale_ratio"]),
        noise_std=float(values["data.noise_std"]),
        site_map_rank_family=str(values["data.site_map_rank_family"]),
        site_presence_span=str(values["data.site_presence_span"]),
        feature_frequency=str(values["data.feature_frequency"]),
        coactivation_probability=float(values["data.coactivation_probability"]),
        coordinate_amplitude_law=str(values["data.coordinate_amplitude_law"]),
        factor_subspace_overlap=str(values["data.factor_subspace_overlap"]),
        train_unique_examples=unique,
        train_presentations=train,
        # Keep all four declared held-out ranges materializable.  The split
        # reader selects one interval and never aliases identities across roles.
        eval_unique_examples=eval_examples,
        eval_presentations=eval_examples,
        presentation_order=presentation_order,
        structure_seed=structure_seed,
        train_seed=train_seed,
        eval_seed=eval_seed,
    )
    return make_ladder_dataset(config, split=generator_split)  # type: ignore[arg-type]


def _normalization_record(
    dataset: Phase1Dataset, values: Mapping[str, Any]
) -> dict[str, Any]:
    mode = str(values["data.normalization"])
    dims = tuple(int(item) for item in dataset.site_dims)
    d_model = int(dataset.padded_dim)
    fit_split = str(values["data.normalization_fit_split"])
    fit_count = int(values["data.normalization_fit_count"])
    fit_statistic = str(values["data.normalization_fit_statistic"])
    expected_fit = {
        "none": ("not_applicable", 0, "identity"),
        "layer": (
            "not_applicable",
            0,
            "per_token_layer_norm_no_dataset_fit",
        ),
        "scalar_rms": (
            "train_unique_prefix",
            None,
            "per_site_fp64_mean_and_centered_scalar_rms",
        ),
        "sqrt_d": (
            "train_unique_prefix",
            None,
            "per_site_fp64_mean_and_mean_centered_l2_norm",
        ),
        "whiten": (
            "train_unique_prefix",
            None,
            "per_site_fp64_shrinkage_covariance",
        ),
    }
    if mode not in expected_fit:
        raise CellExecutionError(f"unknown synthetic normalization {mode!r}")
    expected_split, expected_count, expected_statistic = expected_fit[mode]
    if (
        fit_split != expected_split
        or (expected_count is not None and fit_count != expected_count)
        or fit_statistic != expected_statistic
    ):
        raise CellExecutionError(
            "synthetic normalization fit decisions disagree with the mode: "
            + canonical_json(
                {
                    "mode": mode,
                    "fit_split": fit_split,
                    "fit_count": fit_count,
                    "fit_statistic": fit_statistic,
                }
            )
        )
    if expected_count is None and not 0 < fit_count <= dataset.unique_examples:
        raise CellExecutionError(
            "synthetic normalization fit count exceeds the unique train prefix"
        )
    if mode == "none":
        mean = torch.zeros(len(dims), d_model)
        scale = torch.ones(len(dims))
        kind = "identity"
    elif mode == "layer":
        mean = torch.zeros(len(dims), d_model)
        scale = torch.ones(len(dims))
        kind = "token_layer_norm"
    elif mode == "whiten":
        raise CellExecutionError(
            "synthetic shrinkage whitening is not implemented by the Phase-1 "
            "executor; refusing to substitute scalar normalization"
        )
    else:
        # Fit on the entire declared unique training stream.  Replayed
        # presentations do not receive extra weight in a preprocessing fit.
        sample_count = fit_count
        mean64 = torch.zeros(len(dims), d_model, dtype=torch.float64)
        for item in dataset.batches(
            min(8_192, sample_count), start=0, stop=sample_count
        ):
            x = item.x.double()
            for site, dim in enumerate(dims):
                mean64[site, :dim] += x[:, site, :dim].sum(dim=0)
        mean64 /= sample_count
        mean = mean64.float()
        scale = torch.ones(len(dims))
        denominators = torch.zeros(len(dims), dtype=torch.float64)
        for item in dataset.batches(
            min(8_192, sample_count), start=0, stop=sample_count
        ):
            x = item.x.double()
            for site, dim in enumerate(dims):
                centered = x[:, site, :dim] - mean64[site, :dim]
                if mode == "sqrt_d":
                    denominators[site] += centered.norm(dim=1).sum()
                elif mode == "scalar_rms":
                    denominators[site] += centered.square().sum()
                else:
                    raise CellExecutionError(
                        f"synthetic normalization {mode!r} is not implemented"
                    )
        for site, dim in enumerate(dims):
            if mode == "sqrt_d":
                denominator = denominators[site] / sample_count
                target = math.sqrt(dim)
            elif mode == "scalar_rms":
                denominator = (denominators[site] / (sample_count * dim)).sqrt()
                target = 1.0
            if not torch.isfinite(denominator) or denominator <= 0:
                raise CellExecutionError(f"degenerate normalization at site {site}")
            scale[site] = target / denominator
        kind = "frozen_affine"
    return {
        "mode": mode,
        "kind": kind,
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "site_dims": list(dims),
        "fit_split": fit_split,
        "fit_count": fit_count,
        "fit_statistic": fit_statistic,
        "transform_contract": values["data.normalization_transform_contract"],
    }


def _synthetic_source_contract(values: Mapping[str, Any]) -> dict[str, Any]:
    """Bind the stateless generator, synthetic gauge, and source sentinels."""

    site_dims = tuple(int(item) for item in values["data.site_dims"])
    sites = tuple(str(item) for item in values["data.sites"])
    store_sites = tuple(str(item) for item in values["data.store_sites"])
    capture_contract = _resolved_capture_contract(values)
    expected_capture_contract = {
        "version": "stateless-synthetic-row-v1",
        "row_identity_columns": ["example_id"],
        "dtype": "float32_generated",
    }
    if capture_contract != expected_capture_contract:
        raise CellExecutionError(
            "synthetic data.capture_contract does not match the implemented "
            "stateless generator: "
            + canonical_json(
                {
                    "expected": expected_capture_contract,
                    "actual": capture_contract,
                }
            )
        )
    expected = {
        "source_models": [],
        "source_model_revisions": [],
        "corpus": ["generated"],
        "corpus_revision": [str(values["data.generator_version"])],
        "corpus_config": [str(values["data.dgp_step"])],
        "corpus_split": ["generated"],
        "tokenizer_hashes": [],
        "tokenizer_contract": "not_applicable",
        "store_contract_version": "stateless-generator-v1",
        "store_view_policy": "stateless_generator",
        "normalization_fit_split": values["data.normalization_fit_split"],
        "normalization_transform_contract": "not_applicable",
        "context_length": 1,
        "context_drop_policy": "none",
        "alignment_version": "synthetic-row-identity-v1",
        "alignment_audit": "not_applicable:generator-constructed",
        "capture_contract": expected_capture_contract,
    }
    actual = {
        "source_models": list(values["data.source_models"]),
        "source_model_revisions": list(values["data.source_model_revisions"]),
        "corpus": list(values["data.corpus"]),
        "corpus_revision": list(values["data.corpus_revision"]),
        "corpus_config": list(values["data.corpus_config"]),
        "corpus_split": list(values["data.corpus_split"]),
        "tokenizer_hashes": list(values["data.tokenizer_hashes"]),
        "tokenizer_contract": values["data.tokenizer_contract"],
        "store_contract_version": values["data.store_contract_version"],
        "store_view_policy": values["data.store_view_policy"],
        "normalization_fit_split": values["data.normalization_fit_split"],
        "normalization_transform_contract": values[
            "data.normalization_transform_contract"
        ],
        "context_length": values["data.context_length"],
        "context_drop_policy": values["data.context_drop_policy"],
        "alignment_version": values["data.alignment_version"],
        "alignment_audit": values["data.alignment_audit"],
        "capture_contract": capture_contract,
    }
    if actual != expected:
        raise CellExecutionError(
            "synthetic source decisions do not match the stateless generator contract: "
            + canonical_json({"expected": expected, "actual": actual})
        )
    if sites != store_sites or len(sites) != len(site_dims):
        raise CellExecutionError(
            "synthetic data.sites/data.store_sites must match data.site_dims exactly"
        )
    contract = {
        **actual,
        "kind": str(values["data.kind"]),
        "generator_version": str(values["data.generator_version"]),
        "dgp_step": str(values["data.dgp_step"]),
        "n_factors": int(values["data.n_factors"]),
        "active_factors_per_example": int(values["data.active_factors_per_example"]),
        "factor_coordinate_dim": int(values["data.factor_coordinate_dim"]),
        "factor_rank_profile": str(values["data.factor_rank_profile"]),
        "coordinate_amplitude_law": str(values["data.coordinate_amplitude_law"]),
        "factor_subspace_overlap": str(values["data.factor_subspace_overlap"]),
        "site_scale_ratio": values["data.site_scale_ratio"],
        "noise_std": values["data.noise_std"],
        "missing_probability": values["data.missing_probability"],
        "presentation_order": str(values["data.presentation_order"]),
        "site_dims": list(site_dims),
        "sites": list(sites),
        "structure_seed": int(values["random.structure_seed"]),
        "train_data_seed": int(values["random.train_data_seed"]),
        "eval_data_seed": int(values["random.eval_data_seed"]),
        "confirmation_data_seed": int(values["random.confirmation_data_seed"]),
        "synthetic_split_ranges": [
            [str(name), int(start), int(stop)]
            for name, start, stop in values["data.synthetic_split_ranges"]
        ],
    }
    return {
        "contract": contract,
        "sha256": hashlib.sha256(canonical_json(contract).encode("utf-8")).hexdigest(),
    }


def _apply_normalization(x: torch.Tensor, record: Mapping[str, Any]) -> torch.Tensor:
    result = x.float()
    dims = tuple(int(item) for item in record["site_dims"])
    if record["kind"] == "token_layer_norm":
        output = torch.zeros_like(result)
        for site, dim in enumerate(dims):
            values = result[:, site, :dim]
            mean = values.mean(dim=-1, keepdim=True)
            variance = values.var(dim=-1, correction=0, keepdim=True)
            output[:, site, :dim] = (values - mean) / (variance + 1e-5).sqrt()
        return output
    cache_key = (id(record), result.dtype, result.device)
    cached = _SYNTHETIC_NORMALIZATION_CACHE.get(cache_key)
    if cached is None or cached[0] is not record:
        mean = torch.tensor(
            record["mean"],
            dtype=result.dtype,
            device=result.device,
        )
        scale = torch.tensor(
            record["scale"],
            dtype=result.dtype,
            device=result.device,
        ).view(1, -1, 1)
        _SYNTHETIC_NORMALIZATION_CACHE[cache_key] = (record, mean, scale)
    else:
        _, mean, scale = cached
    result = (result - mean.unsqueeze(0)) * scale
    for site, dim in enumerate(dims):
        result[:, site, dim:] = 0
    return result


def _row_interval(reader: StoreReader) -> dict[str, Any]:
    """Prove a split is strictly ordered and summarize its identity interval.

    Capture row identities begin with ``(sequence, position)``.  Strict
    lexicographic order makes interval separation an exact, constant-memory
    disjointness proof; an arbitrary/non-monotone external store is refused
    instead of receiving a probabilistic overlap check.
    """

    first: tuple[int, int] | None = None
    previous: tuple[int, int] | None = None
    count = 0
    for _, row_ids in reader.sequential_batches_with_ids(65_536):
        if row_ids.ndim != 2 or row_ids.shape[1] < 2:
            raise CellExecutionError(
                f"store split {reader.dir} lacks (sequence, position) row identities"
            )
        keys = row_ids[:, :2].to(torch.int64)
        if len(keys) > 1:
            increasing = (keys[1:, 0] > keys[:-1, 0]) | (
                (keys[1:, 0] == keys[:-1, 0]) & (keys[1:, 1] > keys[:-1, 1])
            )
            if not bool(increasing.all()):
                raise CellExecutionError(
                    f"row identities are not strictly ordered in {reader.dir}"
                )
        current_first = (int(keys[0, 0]), int(keys[0, 1]))
        current_last = (int(keys[-1, 0]), int(keys[-1, 1]))
        if previous is not None and not (current_first > previous):
            raise CellExecutionError(
                f"row identities repeat or regress across shards in {reader.dir}"
            )
        first = current_first if first is None else first
        previous = current_last
        count += len(keys)
    if first is None or previous is None or count != reader.n_tokens:
        raise CellExecutionError(f"invalid/empty row identity stream in {reader.dir}")
    return {"first": list(first), "last": list(previous), "count": count}


def _intervals_are_disjoint(intervals: Mapping[str, Mapping[str, Any]]) -> bool:
    ordered = sorted(
        (
            (
                tuple(int(v) for v in value["first"]),
                tuple(int(v) for v in value["last"]),
                name,
            )
            for name, value in intervals.items()
        ),
        key=lambda item: item[0],
    )
    # Token-disjoint is insufficient: adjacent positions from one language-
    # model context leak almost the whole receptive field across a split.
    # Scientific splits therefore require whole-sequence separation.
    return all(left[1][0] < right[0][0] for left, right in zip(ordered, ordered[1:]))


def _single_source_value(values: Mapping[str, Any], name: str) -> str:
    resolved = tuple(str(item) for item in values[name])
    if len(resolved) != 1:
        raise CellExecutionError(
            f"{name} must contain exactly one value for this capture contract, "
            f"got {resolved!r}"
        )
    return resolved[0]


def _per_site_source_values(
    values: Mapping[str, Any], name: str, *, n_sites: int
) -> tuple[str, ...]:
    resolved = tuple(str(item) for item in values[name])
    if len(resolved) == 1:
        return resolved * n_sites
    if len(resolved) != n_sites:
        raise CellExecutionError(
            f"{name} must have one shared value or one value per captured site; "
            f"got {len(resolved)} for {n_sites} sites"
        )
    return resolved


def _resolved_capture_contract(values: Mapping[str, Any]) -> dict[str, Any]:
    """Decode the cell's immutable capture decisions into their JSON shape.

    Study decisions use tuples so their content IDs cannot depend on mutable
    dictionaries.  Capture manifests are JSON, so tuple-valued fields become
    lists.  Reject malformed/duplicate keys here instead of allowing ``dict``
    to silently keep the last value.
    """

    raw = values["data.capture_contract"]
    if not isinstance(raw, (tuple, list)) or not raw:
        raise CellExecutionError(
            "data.capture_contract must be a nonempty sequence of key/value pairs"
        )
    result: dict[str, Any] = {}
    for item in raw:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise CellExecutionError(
                "data.capture_contract entries must be key/value pairs"
            )
        key, value = item
        if not isinstance(key, str) or not key:
            raise CellExecutionError(
                "data.capture_contract keys must be nonempty strings"
            )
        if key in result:
            raise CellExecutionError(
                f"data.capture_contract contains duplicate key {key!r}"
            )
        result[key] = list(value) if isinstance(value, (tuple, list)) else value
    return result


def _expected_real_source_contract(values: Mapping[str, Any]) -> dict[str, Any]:
    """Materialize the exact scientific capture contract from cell decisions."""

    hooks = tuple(str(item) for item in values["data.store_sites"])
    if not hooks:
        raise CellExecutionError("data.store_sites cannot be empty")
    models = _per_site_source_values(values, "data.source_models", n_sites=len(hooks))
    revisions = _per_site_source_values(
        values, "data.source_model_revisions", n_sites=len(hooks)
    )
    drop_policy = str(values["data.context_drop_policy"])
    if drop_policy == "none":
        drop_positions = 0
    elif drop_policy == "drop_bos_position_0":
        drop_positions = 1
    else:
        raise CellExecutionError(
            f"unsupported data.context_drop_policy {drop_policy!r}"
        )
    base = {
        "sources": [
            {"model": model, "revision": revision, "hook": hook}
            for model, revision, hook in zip(models, revisions, hooks)
        ],
        "corpus": _single_source_value(values, "data.corpus"),
        "corpus_config": _single_source_value(values, "data.corpus_config"),
        "corpus_revision": _single_source_value(values, "data.corpus_revision"),
        "corpus_split": _single_source_value(values, "data.corpus_split"),
        "context": int(values["data.context_length"]),
        "drop_positions": drop_positions,
        "tokenizer_hashes": list(str(item) for item in values["data.tokenizer_hashes"]),
        "tokenizer_contract": str(values["data.tokenizer_contract"]),
        "store_contract_version": str(values["data.store_contract_version"]),
        "alignment_version": str(values["data.alignment_version"]),
        "alignment_audit": str(values["data.alignment_audit"]),
    }
    capture = _resolved_capture_contract(values)
    overlap = set(base).intersection(capture)
    if overlap:
        raise CellExecutionError(
            "data.capture_contract duplicates separately resolved source fields: "
            + ", ".join(sorted(overlap))
        )
    return {**base, **capture}


def _verify_real_source_contract(
    source: Mapping[str, Any], values: Mapping[str, Any]
) -> dict[str, Any]:
    expected = _expected_real_source_contract(values)
    mismatches = {
        key: {
            "cell": expected.get(key, "<not-declared>"),
            "capture": source.get(key, "<missing>"),
        }
        for key in sorted(set(expected).union(source))
        if source.get(key, "<missing>") != expected.get(key, "<not-declared>")
    }
    if mismatches:
        raise CellExecutionError(
            "capture source contract does not match resolved cell decisions: "
            + canonical_json(mismatches)
        )
    return expected


def _expected_capture_allocation(
    values: Mapping[str, Any],
) -> tuple[tuple[str, ...], dict[str, dict[str, int]]]:
    """Rebuild the canonical whole-sequence allocation from cell decisions."""

    declared_items = tuple(values["data.split_sizes"])
    split_order = tuple(str(name) for name, _ in declared_items)
    drop_policy = str(values["data.context_drop_policy"])
    if drop_policy == "none":
        drop_positions = 0
    elif drop_policy == "drop_bos_position_0":
        drop_positions = 1
    else:
        raise CellExecutionError(
            f"unsupported data.context_drop_policy {drop_policy!r}"
        )
    tokens_per_sequence = int(values["data.context_length"]) - drop_positions
    if tokens_per_sequence <= 0:
        raise CellExecutionError("capture tokens per sequence must be positive")
    next_sequence = 0
    plan: dict[str, dict[str, int]] = {}
    for name, requested in declared_items:
        name = str(name)
        requested = int(requested)
        n_sequences = math.ceil(requested / tokens_per_sequence)
        sequence_stop = next_sequence + n_sequences
        plan[name] = {
            "requested_tokens": requested,
            "actual_tokens": n_sequences * tokens_per_sequence,
            "sequence_start": next_sequence,
            "sequence_stop_exclusive": sequence_stop,
            "tokens_per_sequence": tokens_per_sequence,
        }
        next_sequence = sequence_stop
    return split_order, plan


def _load_capture_contract(raw_root: Path, values: Mapping[str, Any]) -> dict[str, Any]:
    capture_path = raw_root / "capture.json"
    if not capture_path.is_file():
        raise CellExecutionError(
            f"raw activation store lacks immutable source contract {capture_path}"
        )
    capture = _read_object(capture_path, label="capture source contract")
    source = capture.get("source")
    source_hash = capture.get("source_hash")
    if not isinstance(source, dict) or not isinstance(source_hash, str):
        raise CellExecutionError("capture source contract is malformed")
    computed_source_hash = hashlib.sha256(
        canonical_json(source).encode("utf-8")
    ).hexdigest()
    if computed_source_hash != source_hash:
        raise CellExecutionError("capture source contract hash mismatch")
    declared = _verify_real_source_contract(source, values)
    split_order, split_plan = _expected_capture_allocation(values)
    if capture.get("schema") != "bsc-capture-manifest-v1":
        raise CellExecutionError("capture source contract has an unknown schema")
    if capture.get("split_order") != list(split_order):
        raise CellExecutionError(
            "capture split order differs from the canonical cell allocation"
        )
    if capture.get("split_plan") != split_plan or capture.get("splits") != split_plan:
        raise CellExecutionError(
            "capture split allocation differs from data.split_sizes"
        )
    capture_binding_sha256 = capture.get("capture_binding_sha256")
    if not isinstance(capture_binding_sha256, str) or len(capture_binding_sha256) != 64:
        raise CellExecutionError("capture binding digest is malformed")
    return {
        "path": str(capture_path),
        "sha256": _sha256(capture_path),
        "source_hash": source_hash,
        "source": source,
        "declared": declared,
        "split_order": split_order,
        "split_plan": split_plan,
        "capture_binding_sha256": capture_binding_sha256,
    }


def _site_selection(
    values: Mapping[str, Any],
) -> tuple[tuple[int, ...], tuple[str, ...], tuple[str, ...], tuple[int, ...]]:
    expected_dims = tuple(int(item) for item in values["data.site_dims"])
    store_site_names = tuple(str(item) for item in values["data.store_sites"])
    requested_site_names = tuple(str(item) for item in values["data.sites"])
    if not requested_site_names or len(requested_site_names) != len(expected_dims):
        raise CellExecutionError("data.sites must align one-to-one with data.site_dims")
    if len(set(store_site_names)) != len(store_site_names):
        raise CellExecutionError("data.store_sites contains duplicates")
    try:
        selected_sites = tuple(
            store_site_names.index(name) for name in requested_site_names
        )
    except ValueError as exc:
        raise CellExecutionError(
            "data.sites is not an ordered subset of data.store_sites"
        ) from exc
    if tuple(sorted(selected_sites)) != selected_sites:
        raise CellExecutionError("data.sites reorders the immutable store site axis")
    return expected_dims, store_site_names, requested_site_names, selected_sites


def _verify_declared_split_contract(
    root: Path,
    values: Mapping[str, Any],
    *,
    capture_contract: Mapping[str, Any],
    expected_store_axis: tuple[int, ...],
    expected_whitener_hash: str,
) -> dict[str, Any]:
    declared_items = tuple(values["data.split_sizes"])
    declared = {str(name): int(tokens) for name, tokens in declared_items}
    expected_order = tuple(str(item) for item in capture_contract["split_order"])
    if tuple(declared) != expected_order:
        raise CellExecutionError(
            "activation-store split order differs from the capture allocation"
        )
    split_plan = capture_contract["split_plan"]
    capture_binding_sha256 = str(capture_contract["capture_binding_sha256"])
    available = {
        path.name
        for path in root.iterdir()
        if path.is_dir() and (path / MANIFEST_NAME).is_file()
    }
    if available != set(declared):
        raise CellExecutionError(
            "activation-store split set differs from data.split_sizes: "
            + canonical_json({"cell": sorted(declared), "store": sorted(available)})
        )
    row_id_width = int(values["data.row_id_width"])
    if int(values["data.row_id_bytes"]) != 8:
        raise CellExecutionError("activation-store row identities must use int64")
    result: dict[str, Any] = {}
    for split, requested_tokens in declared_items:
        split = str(split)
        reader = StoreReader(root, split)
        allocation = split_plan[split]
        if reader.n_tokens != int(allocation["actual_tokens"]):
            raise CellExecutionError(
                f"store split {split!r} has {reader.n_tokens} rows, but the "
                f"canonical allocation requires {allocation['actual_tokens']}"
            )
        try:
            _verify_store_reader_once(
                reader,
                root,
                split,
                expected_row_identity={
                    "sequence_start": int(allocation["sequence_start"]),
                    "sequence_stop_exclusive": int(
                        allocation["sequence_stop_exclusive"]
                    ),
                    "tokens_per_sequence": int(allocation["tokens_per_sequence"]),
                    "position_start": int(capture_contract["source"]["drop_positions"]),
                },
            )
        except (OSError, ValueError) as exc:
            raise CellExecutionError(
                f"store split {split!r} failed canonical row verification: {exc}"
            ) from exc
        meta = reader.manifest.get("meta", {})
        if tuple(reader.sites) != expected_store_axis:
            raise CellExecutionError(
                f"store split {split!r} has an undeclared site axis"
            )
        if reader.whitener_hash != expected_whitener_hash:
            raise CellExecutionError(
                f"store split {split!r} is bound to another transform/source"
            )
        if reader.n_tokens < int(requested_tokens):
            raise CellExecutionError(
                f"store split {split!r} has {reader.n_tokens} rows, below declared "
                f"minimum {requested_tokens}"
            )
        if meta.get("split_requested_tokens") != int(requested_tokens):
            raise CellExecutionError(
                f"store split {split!r} does not bind its requested row count"
            )
        if meta.get("split_actual_tokens") != reader.n_tokens:
            raise CellExecutionError(
                f"store split {split!r} does not bind its actual row count"
            )
        expected_meta = {
            "sequence_start": allocation["sequence_start"],
            "sequence_stop_exclusive": allocation["sequence_stop_exclusive"],
            "tokens_per_sequence": allocation["tokens_per_sequence"],
            "ordered_split_allocation": list(expected_order),
            "capture_binding_sha256": capture_binding_sha256,
        }
        mismatched_meta = {
            key: {"expected": expected, "observed": meta.get(key, "<missing>")}
            for key, expected in expected_meta.items()
            if meta.get(key, "<missing>") != expected
        }
        if mismatched_meta:
            raise CellExecutionError(
                f"store split {split!r} allocation metadata differs from capture: "
                + canonical_json(mismatched_meta)
            )
        if any(
            int(shard.get("row_id_width", -1)) != row_id_width
            for shard in reader.manifest.get("shards", ())
        ):
            raise CellExecutionError(
                f"store split {split!r} row identity width differs from cell"
            )
        result[split] = {
            "requested_tokens": int(requested_tokens),
            "actual_tokens": reader.n_tokens,
            "manifest_sha256": reader.manifest["manifest_sha256"],
            "row_stream_sha256": reader.manifest["row_stream_sha256"],
            "content_stream_sha256": reader.manifest["content_stream_sha256"],
        }
    return result


def _resolve_single_raw_store(
    values: Mapping[str, Any], configured_root: Path
) -> dict[str, Any]:
    """Resolve the Phase-3 one-store policy and its transform-only artifact."""

    normalization = str(values["data.normalization"])
    if normalization == "layer":
        raise CellExecutionError(
            "Phase 3 single-view storage forbids non-invertible token LayerNorm"
        )
    raw_root = Path(
        os.environ.get("BSC_RAW_STORE_ROOT")
        or os.environ.get("BSC_RAW_STORE")
        or configured_root
    ).resolve()
    source_contract = _load_capture_contract(raw_root, values)
    source_hash = str(source_contract["source_hash"])
    expected_dims, store_site_names, requested_site_names, selected_sites = (
        _site_selection(values)
    )
    expected_store_axis = tuple(range(len(store_site_names)))
    declared_split_contract = _verify_declared_split_contract(
        raw_root,
        values,
        capture_contract=source_contract,
        expected_store_axis=expected_store_axis,
        expected_whitener_hash=f"raw:{source_hash}",
    )
    required_splits = {
        "train": "train",
        "calibration": str(values["evaluation.calibration_split"]),
        "evaluation": str(values["evaluation.split"]),
    }
    if len(set(required_splits.values())) != len(required_splits):
        raise CellExecutionError(
            "train, calibration, and evaluation split names must differ"
        )
    bindings: dict[str, Any] = {}
    intervals: dict[str, dict[str, Any]] = {}
    for role, split in required_splits.items():
        manifest_path = raw_root / split / MANIFEST_NAME
        if not manifest_path.is_file():
            raise CellExecutionError(
                f"raw activation store {raw_root} lacks exact {role} split {split!r}"
            )
        full_reader = StoreReader(raw_root, split)
        if tuple(full_reader.sites) != expected_store_axis:
            raise CellExecutionError(
                f"raw {role} store site axis {full_reader.sites} does not bind "
                "data.store_sites"
            )
        reader = StoreReader(raw_root, split, sites=selected_sites)
        _verify_store_reader_once(reader, raw_root, split)
        if reader.whitener_hash != f"raw:{source_hash}":
            raise CellExecutionError(
                f"raw {role} split is not bound to capture source {source_hash}"
            )
        if tuple(reader.site_dims) != expected_dims:
            raise CellExecutionError(
                f"raw {role} store site_dims {reader.site_dims} != cell {expected_dims}"
            )
        bindings[role] = {
            "split": split,
            "manifest": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "n_tokens": reader.n_tokens,
            "row_stream_sha256": reader.manifest.get("row_stream_sha256"),
            "content_stream_sha256": reader.manifest.get("content_stream_sha256"),
            "selected_site_indices": list(selected_sites),
            "selected_site_names": list(requested_site_names),
        }
        intervals[role] = _row_interval(reader)
    if not _intervals_are_disjoint(intervals):
        raise CellExecutionError(
            "train/calibration/evaluation row-identity intervals overlap"
        )
    unique_tokens = int(values["data.unique_tokens"])
    train_tokens = int(values["data.train_tokens"])
    if not 0 < unique_tokens <= train_tokens:
        raise CellExecutionError(
            "real cells require 0 < data.unique_tokens <= data.train_tokens"
        )
    if int(bindings["train"]["n_tokens"]) < unique_tokens:
        raise CellExecutionError(
            "activation store has fewer immutable training rows than declared "
            f"data.unique_tokens: {bindings['train']['n_tokens']} < {unique_tokens}"
        )

    fit_split = str(values["data.normalization_fit_split"])
    transform_contract = str(values["data.normalization_transform_contract"])
    if fit_split != "normalization_fit":
        raise CellExecutionError(
            f"unsupported data.normalization_fit_split {fit_split!r}"
        )
    if transform_contract != "content_addressed_transform_only-v1":
        raise CellExecutionError("Phase 3 requires content_addressed_transform_only-v1")
    fit_manifest_path = raw_root / fit_split / MANIFEST_NAME
    if not fit_manifest_path.is_file():
        raise CellExecutionError(
            "Phase 3 raw store lacks the dedicated normalization_fit split"
        )
    fit_reader = StoreReader(raw_root, fit_split)
    _verify_store_reader_once(fit_reader, raw_root, fit_split)
    if tuple(fit_reader.sites) != expected_store_axis:
        raise CellExecutionError(
            "normalization_fit site axis differs from the raw store"
        )
    if fit_reader.whitener_hash != f"raw:{source_hash}":
        raise CellExecutionError("normalization_fit is bound to another capture source")
    fit_selected_reader = StoreReader(raw_root, fit_split, sites=selected_sites)
    if tuple(fit_selected_reader.site_dims) != expected_dims:
        raise CellExecutionError(
            "normalization_fit selected dimensions differ from the cell"
        )
    fit_binding = {
        "split": fit_split,
        "manifest": str(fit_manifest_path),
        "manifest_sha256": _sha256(fit_manifest_path),
        "n_tokens": fit_selected_reader.n_tokens,
        "row_stream_sha256": fit_selected_reader.manifest.get("row_stream_sha256"),
        "content_stream_sha256": fit_selected_reader.manifest.get(
            "content_stream_sha256"
        ),
        "selected_site_indices": list(selected_sites),
        "selected_site_names": list(requested_site_names),
    }
    bindings["normalization_fit"] = fit_binding
    intervals["normalization_fit"] = _row_interval(fit_selected_reader)
    if not _intervals_are_disjoint(intervals):
        raise CellExecutionError(
            "train/normalization-fit/calibration/evaluation row-identity "
            "intervals overlap"
        )

    transform_root = Path(
        os.environ.get("BSC_TRANSFORM_ROOT") or (raw_root / "transforms")
    ).resolve()
    candidate_root = transform_root / normalization
    candidates = sorted(candidate_root.glob("*/whitener.pt"))
    accepted: list[tuple[Path, Whitener]] = []
    expected_meta = {
        "source_capture_sha256": source_contract["sha256"],
        "source_hash": source_hash,
        "source_fit_manifest_sha256": fit_reader.manifest["manifest_sha256"],
        "source_fit_row_stream_sha256": fit_reader.manifest["row_stream_sha256"],
        "source_fit_content_stream_sha256": fit_reader.manifest[
            "content_stream_sha256"
        ],
        "source_fit_requested_tokens": int(values["data.normalization_fit_count"]),
        "transform_contract": transform_contract,
    }
    rejected: list[str] = []
    for path in candidates:
        try:
            transform = Whitener.load(path)
            meta_mismatch = {
                key: {"expected": expected, "actual": transform.meta.get(key)}
                for key, expected in expected_meta.items()
                if transform.meta.get(key) != expected
            }
            if transform.mode != normalization:
                raise ValueError(f"mode {transform.mode!r} != {normalization!r}")
            if transform.n_fit_tokens != int(values["data.normalization_fit_count"]):
                raise ValueError(
                    "transform fit count differs from data.normalization_fit_count"
                )
            if path.parent.name != transform.hash:
                raise ValueError("parent directory is not the Whitener content hash")
            if tuple(transform.sites) != expected_store_axis:
                raise ValueError("transform site axis differs from data.store_sites")
            selected_transform_dims = tuple(
                transform.site_dims[index] for index in selected_sites
            )
            if selected_transform_dims != expected_dims:
                raise ValueError(
                    f"selected transform dims {selected_transform_dims} != {expected_dims}"
                )
            if meta_mismatch:
                raise ValueError(canonical_json(meta_mismatch))
        except Exception as exc:  # noqa: BLE001
            rejected.append(f"{path}: {exc}")
            continue
        accepted.append((path, transform))
    if len(accepted) != 1:
        detail = "" if not rejected else "; rejected: " + "; ".join(rejected)
        raise CellExecutionError(
            f"expected exactly one verified Phase-3 transform under {candidate_root}, "
            f"found {len(accepted)}{detail}"
        )
    transform_path, transform = accepted[0]
    transform_manifest_path = transform_path.with_name("transform.json")
    if not transform_manifest_path.is_file():
        raise CellExecutionError(
            f"Phase-3 transform lacks {transform_manifest_path.name}"
        )
    transform_manifest = _read_object(
        transform_manifest_path, label="transform artifact manifest"
    )
    expected_transform_manifest = {
        "schema": "bsc-transform-artifact-v1",
        "mode": normalization,
        "transform_hash": transform.hash,
        "whitener_sha256": _sha256(transform_path),
        **expected_meta,
        "source_raw_root": str(raw_root),
    }
    manifest_mismatch = {
        key: {"expected": expected, "actual": transform_manifest.get(key)}
        for key, expected in expected_transform_manifest.items()
        if transform_manifest.get(key) != expected
    }
    if manifest_mismatch:
        raise CellExecutionError(
            "Phase-3 transform manifest binding mismatch: "
            + canonical_json(manifest_mismatch)
        )
    raw_bindings = {role: dict(binding) for role, binding in bindings.items()}
    return {
        "root": str(raw_root),
        "splits": {role: binding["split"] for role, binding in bindings.items()},
        "bindings": bindings,
        "row_intervals": intervals,
        "row_intervals_disjoint": True,
        "declared_split_contract": declared_split_contract,
        "raw_root": str(raw_root),
        "raw_bindings": raw_bindings,
        "source_contract": source_contract,
        "store_view_policy": str(values["data.store_view_policy"]),
        "training_row_policy": {
            "kind": "immutable_prefix_then_deterministic_replay",
            "unique_tokens": unique_tokens,
            "train_tokens": train_tokens,
        },
        "normalization": {
            "mode": normalization,
            "application": "on_the_fly",
            "transform_path": str(transform_path),
            "transform_sha256": _sha256(transform_path),
            "transform_hash": transform.hash,
            "transform_manifest": str(transform_manifest_path),
            "transform_manifest_sha256": _sha256(transform_manifest_path),
            "selected_site_indices": list(selected_sites),
            "source_capture_sha256": source_contract["sha256"],
            "source_fit_manifest": str(fit_manifest_path),
            "source_fit_manifest_file_sha256": _sha256(fit_manifest_path),
            "source_fit_manifest_sha256": fit_reader.manifest["manifest_sha256"],
            "source_fit_row_stream_sha256": fit_reader.manifest["row_stream_sha256"],
            "source_fit_requested_tokens": transform.n_fit_tokens,
        },
    }


def _resolve_real_store(values: Mapping[str, Any]) -> dict[str, Any]:
    store_view_policy = str(values["data.store_view_policy"])
    configured = os.environ.get("BSC_ACTIVATION_STORE") or os.environ.get(
        "BSC_STORE_ROOT"
    )
    if (
        configured is None
        and store_view_policy
        == "single_bf16_raw_view_on_the_fly_invertible_normalization"
    ):
        configured = os.environ.get("BSC_RAW_STORE_ROOT") or os.environ.get(
            "BSC_RAW_STORE"
        )
    if not configured:
        raise CellExecutionError(
            "Phase 2/3 requires BSC_ACTIVATION_STORE (or BSC_STORE_ROOT) "
            "pointing at a verified store view; Phase 3 also accepts "
            "BSC_RAW_STORE_ROOT for its single-view policy"
        )
    base = Path(configured).resolve()
    if store_view_policy == "single_bf16_raw_view_on_the_fly_invertible_normalization":
        return _resolve_single_raw_store(values, base)
    if store_view_policy != "content_addressed_derived_view":
        raise CellExecutionError(
            f"unsupported real data.store_view_policy {store_view_policy!r}"
        )
    fit_split = str(values["data.normalization_fit_split"])
    if fit_split != "normalization_fit":
        raise CellExecutionError(
            f"unsupported data.normalization_fit_split {fit_split!r}"
        )
    if values["data.normalization_transform_contract"] != "persisted-derived-shards-v1":
        raise CellExecutionError(
            "Phase 2 derived views require persisted-derived-shards-v1"
        )
    normalization = str(values["data.normalization"])
    candidates = (base, base / normalization)
    root: Path | None = None
    train_manifest: dict[str, Any] | None = None
    candidate_errors: list[str] = []
    for candidate in candidates:
        manifest_path = candidate / "train" / MANIFEST_NAME
        if not manifest_path.is_file():
            continue
        try:
            reader = StoreReader(candidate, "train")
        except Exception as exc:  # noqa: BLE001 - report every rejected view
            candidate_errors.append(f"{candidate}: {exc}")
            continue
        meta = reader.manifest.get("meta", {})
        if meta.get("derived_view") is not True:
            candidate_errors.append(
                f"{candidate}: manifest is not a content-addressed derived view"
            )
            continue
        declared_mode = meta.get("normalization")
        if declared_mode is None and str(reader.whitener_hash).startswith("raw:"):
            declared_mode = "none"
        if declared_mode != normalization:
            candidate_errors.append(
                f"{candidate}: manifest normalization {declared_mode!r} != {normalization!r}"
            )
            continue
        root = candidate
        train_manifest = reader.manifest
        break
    if root is None or train_manifest is None:
        detail = (
            "" if not candidate_errors else "; rejected: " + "; ".join(candidate_errors)
        )
        raise CellExecutionError(
            f"configured activation store {base} has no train/{MANIFEST_NAME} "
            f"explicitly bound to normalization {normalization!r}{detail}"
        )
    transform: Whitener | None = None
    derived_train_view = train_manifest.get("meta", {}).get("derived_view") is True
    if not derived_train_view:
        raise CellExecutionError(
            "content_addressed_derived_view policy requires a persisted derived "
            "normalization view, including for the identity normalization"
        )
    if normalization != "none" or derived_train_view:
        transform_path = root / "whitener.pt"
        if not transform_path.is_file():
            raise CellExecutionError(
                f"normalized store view {root} lacks frozen transform {transform_path.name}"
            )
        try:
            transform = Whitener.load(transform_path)
        except Exception as exc:  # noqa: BLE001
            raise CellExecutionError(
                f"cannot verify frozen transform {transform_path}: {exc}"
            ) from exc
        if transform.mode != normalization:
            raise CellExecutionError(
                f"transform mode {transform.mode!r} != cell normalization {normalization!r}"
            )
        if transform.n_fit_tokens != int(
            values["data.normalization_fit_count"]
        ) or transform.meta.get("source_fit_requested_tokens") != int(
            values["data.normalization_fit_count"]
        ):
            raise CellExecutionError(
                "transform fit count differs from data.normalization_fit_count"
            )
        if train_manifest.get("whitener_hash") != transform.hash:
            raise CellExecutionError(
                "train manifest is not bound to the frozen transform"
            )
        if train_manifest.get("meta", {}).get("derived_view") is not True:
            raise CellExecutionError(
                "normalized store manifest is not marked derived_view"
            )
    required_splits = {
        "train": "train",
        "calibration": str(values["evaluation.calibration_split"]),
        "evaluation": str(values["evaluation.split"]),
    }
    splits: dict[str, str] = {}
    bindings: dict[str, Any] = {}
    intervals: dict[str, dict[str, Any]] = {}
    expected_dims = tuple(int(item) for item in values["data.site_dims"])
    store_site_names = tuple(str(item) for item in values["data.store_sites"])
    requested_site_names = tuple(str(item) for item in values["data.sites"])
    if not requested_site_names or len(requested_site_names) != len(expected_dims):
        raise CellExecutionError("data.sites must align one-to-one with data.site_dims")
    if len(set(store_site_names)) != len(store_site_names):
        raise CellExecutionError("data.store_sites contains duplicates")
    try:
        selected_sites = tuple(
            store_site_names.index(name) for name in requested_site_names
        )
    except ValueError as exc:
        raise CellExecutionError(
            "data.sites is not an ordered subset of data.store_sites"
        ) from exc
    if tuple(sorted(selected_sites)) != selected_sites:
        raise CellExecutionError("data.sites reorders the immutable store site axis")
    for role, split in required_splits.items():
        if not (root / split / MANIFEST_NAME).is_file():
            raise CellExecutionError(
                f"activation store {root} lacks exact {role} split {split!r}; "
                "split aliases/fallbacks are forbidden"
            )
        full_reader = StoreReader(root, split)
        expected_store_axis = tuple(range(len(store_site_names)))
        if tuple(full_reader.sites) != expected_store_axis:
            raise CellExecutionError(
                f"{role} store site axis {full_reader.sites} does not bind the "
                f"declared hook axis {expected_store_axis}"
            )
        reader = StoreReader(root, split, sites=selected_sites)
        _verify_store_reader_once(reader, root, split)
        meta = reader.manifest.get("meta", {})
        manifest_mode = meta.get("normalization")
        if manifest_mode is None and str(reader.whitener_hash).startswith("raw:"):
            manifest_mode = "none"
        if manifest_mode != normalization:
            raise CellExecutionError(
                f"{role} manifest normalization {manifest_mode!r} != {normalization!r}"
            )
        if transform is not None and reader.whitener_hash != transform.hash:
            raise CellExecutionError(f"{role} split is bound to a different transform")
        if tuple(reader.site_dims) != expected_dims:
            raise CellExecutionError(
                f"{role} store site_dims {reader.site_dims} != cell {expected_dims}"
            )
        splits[role] = split
        manifest_path = root / split / MANIFEST_NAME
        bindings[role] = {
            "split": split,
            "manifest": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "n_tokens": reader.n_tokens,
            "row_stream_sha256": reader.manifest.get("row_stream_sha256"),
            "content_stream_sha256": reader.manifest.get("content_stream_sha256"),
            "selected_site_indices": list(selected_sites),
            "selected_site_names": list(requested_site_names),
        }
        intervals[role] = _row_interval(reader)
    if len(set(required_splits.values())) != len(required_splits):
        raise CellExecutionError(
            "train, calibration, and evaluation split names must differ"
        )
    if not _intervals_are_disjoint(intervals):
        raise CellExecutionError(
            "train/calibration/evaluation row-identity intervals overlap"
        )
    unique_tokens = int(values["data.unique_tokens"])
    train_tokens = int(values["data.train_tokens"])
    if not 0 < unique_tokens <= train_tokens:
        raise CellExecutionError(
            "real cells require 0 < data.unique_tokens <= data.train_tokens"
        )
    if int(bindings["train"]["n_tokens"]) < unique_tokens:
        raise CellExecutionError(
            "activation store has fewer immutable training rows than declared "
            f"data.unique_tokens: {bindings['train']['n_tokens']} < {unique_tokens}"
        )

    raw_configured = os.environ.get("BSC_RAW_STORE_ROOT") or os.environ.get(
        "BSC_RAW_STORE"
    )
    derived_meta = train_manifest.get("meta", {})
    declared_raw_root = derived_meta.get("source_raw_root")
    if raw_configured is None and normalization == "none" and not derived_train_view:
        raw_root = root
    elif raw_configured is None and declared_raw_root:
        raw_root = Path(str(declared_raw_root)).resolve()
    elif raw_configured is None:
        raise CellExecutionError(
            "normalized Phase 2/3 data requires BSC_RAW_STORE_ROOT for paired "
            "row-identical raw-space evaluation"
        )
    else:
        raw_root = Path(raw_configured).resolve()
    source_contract = _load_capture_contract(raw_root, values)
    source_hash = source_contract["source_hash"]
    expected_store_axis = tuple(range(len(store_site_names)))
    raw_declared_split_contract = _verify_declared_split_contract(
        raw_root,
        values,
        capture_contract=source_contract,
        expected_store_axis=expected_store_axis,
        expected_whitener_hash=f"raw:{source_hash}",
    )
    if transform is None:
        raise CellExecutionError(
            "content-addressed Phase-2 policy requires a frozen transform"
        )
    declared_split_contract = _verify_declared_split_contract(
        root,
        values,
        capture_contract=source_contract,
        expected_store_axis=expected_store_axis,
        expected_whitener_hash=transform.hash,
    )
    expected_source_hash = values.get("data.source_contract_sha256")
    if expected_source_hash not in {None, "runtime_bound"} and (
        expected_source_hash != source_hash
    ):
        raise CellExecutionError(
            f"capture source hash {source_hash} != cell contract {expected_source_hash}"
        )
    raw_bindings: dict[str, Any] = {}
    for role, split in required_splits.items():
        if not (raw_root / split / MANIFEST_NAME).is_file():
            raise CellExecutionError(
                f"raw activation store {raw_root} lacks exact split {split!r}"
            )
        raw_reader = StoreReader(raw_root, split, sites=selected_sites)
        _verify_store_reader_once(raw_reader, raw_root, split)
        raw_full_reader = StoreReader(raw_root, split)
        if tuple(raw_full_reader.sites) != tuple(range(len(store_site_names))):
            raise CellExecutionError(
                f"raw {role} store site axis does not bind data.store_sites"
            )
        if raw_reader.whitener_hash != f"raw:{source_hash}":
            raise CellExecutionError(
                f"raw {role} split is not bound to capture source {source_hash}"
            )
        normalized = bindings[role]
        raw_manifest_sha256 = raw_reader.manifest.get("manifest_sha256")
        normalized_meta = StoreReader(root, split).manifest.get("meta", {})
        if (
            tuple(raw_reader.site_dims) != expected_dims
            or raw_reader.n_tokens != normalized["n_tokens"]
            or raw_reader.manifest.get("row_stream_sha256")
            != normalized["row_stream_sha256"]
        ):
            raise CellExecutionError(
                f"raw and normalized {role} splits are not row/site aligned"
            )
        if normalized_meta.get("derived_view") is True:
            if (
                normalized_meta.get("source_split_manifest_sha256")
                != raw_manifest_sha256
            ):
                raise CellExecutionError(
                    f"normalized {role} split does not bind the paired raw manifest"
                )
            if normalized_meta.get("source_raw_root") != str(raw_root.resolve()):
                raise CellExecutionError(
                    f"normalized {role} split names a different source_raw_root"
                )
        raw_manifest = raw_root / split / MANIFEST_NAME
        raw_bindings[role] = {
            "split": split,
            "manifest": str(raw_manifest),
            "manifest_sha256": _sha256(raw_manifest),
            "n_tokens": raw_reader.n_tokens,
            "row_stream_sha256": raw_reader.manifest.get("row_stream_sha256"),
            "content_stream_sha256": raw_reader.manifest.get("content_stream_sha256"),
            "selected_site_indices": list(selected_sites),
            "selected_site_names": list(requested_site_names),
        }
    if transform is not None:
        fit_path = raw_root / fit_split / MANIFEST_NAME
        if not fit_path.is_file():
            raise CellExecutionError(
                "derived normalization has no paired raw normalization_fit manifest"
            )
        fit_reader = StoreReader(raw_root, fit_split)
        _verify_store_reader_once(fit_reader, raw_root, fit_split)
        if transform.meta.get("source_fit_manifest_sha256") != fit_reader.manifest.get(
            "manifest_sha256"
        ) or transform.meta.get(
            "source_fit_row_stream_sha256"
        ) != fit_reader.manifest.get("row_stream_sha256"):
            raise CellExecutionError(
                "frozen transform is not bound to the paired raw normalization-fit rows"
            )
        raw_fit_selected = StoreReader(raw_root, fit_split, sites=selected_sites)
        fit_binding_common = {
            "split": fit_split,
            "n_tokens": raw_fit_selected.n_tokens,
            "row_stream_sha256": raw_fit_selected.manifest.get("row_stream_sha256"),
            "content_stream_sha256": raw_fit_selected.manifest.get(
                "content_stream_sha256"
            ),
            "selected_site_indices": list(selected_sites),
            "selected_site_names": list(requested_site_names),
        }
        raw_bindings["normalization_fit"] = {
            **fit_binding_common,
            "manifest": str(fit_path),
            "manifest_sha256": _sha256(fit_path),
        }
        view_fit_path = root / fit_split / MANIFEST_NAME
        if not view_fit_path.is_file():
            raise CellExecutionError("derived view lacks its normalization_fit split")
        view_fit_reader = StoreReader(root, fit_split, sites=selected_sites)
        if (
            view_fit_reader.whitener_hash != transform.hash
            or tuple(view_fit_reader.site_dims) != expected_dims
            or view_fit_reader.manifest.get("row_stream_sha256")
            != raw_fit_selected.manifest.get("row_stream_sha256")
            or view_fit_reader.manifest.get("meta", {}).get(
                "source_split_manifest_sha256"
            )
            != raw_fit_selected.manifest.get("manifest_sha256")
        ):
            raise CellExecutionError(
                "derived normalization_fit view is not bound to its raw split"
            )
        bindings["normalization_fit"] = {
            **fit_binding_common,
            "manifest": str(view_fit_path),
            "manifest_sha256": _sha256(view_fit_path),
            "content_stream_sha256": view_fit_reader.manifest.get(
                "content_stream_sha256"
            ),
        }
        splits["normalization_fit"] = fit_split
        intervals["normalization_fit"] = _row_interval(raw_fit_selected)
        if not _intervals_are_disjoint(intervals):
            raise CellExecutionError(
                "train/normalization-fit/calibration/evaluation row-identity "
                "intervals overlap"
            )
    return {
        "root": str(root),
        "splits": splits,
        "bindings": bindings,
        "row_intervals": intervals,
        "row_intervals_disjoint": True,
        "declared_split_contract": declared_split_contract,
        "raw_root": str(raw_root),
        "raw_bindings": raw_bindings,
        "raw_declared_split_contract": raw_declared_split_contract,
        "source_contract": {
            **source_contract,
        },
        "store_view_policy": store_view_policy,
        "training_row_policy": {
            "kind": "immutable_prefix_then_deterministic_replay",
            "unique_tokens": unique_tokens,
            "train_tokens": train_tokens,
        },
        "normalization": {
            "mode": normalization,
            "transform_sha256": (
                None if transform is None else _sha256(root / "whitener.pt")
            ),
            "transform_hash": None if transform is None else transform.hash,
        },
    }


def _prepare(ctx: _Context) -> tuple[tuple[str, Path], ...]:
    values = ctx.values
    if values.get("implementation.materializable") is False:
        raise CellExecutionError(
            "cell recipe is quarantined and cannot be executed: "
            + str(
                values.get(
                    "implementation.quarantine_reason",
                    "source-exact runtime adapter incomplete",
                )
            )
        )
    if values["data.observation_policy"] != "all_sites":
        raise CellExecutionError(
            f"unsupported data.observation_policy {values['data.observation_policy']!r}"
        )
    declared_device = _declared_device(values)
    implementation = _implementation_identity()
    if ctx.cell.phase is Phase.PHASE3 and (
        implementation["git_commit"] is None or implementation["git_dirty"] is not False
    ):
        raise CellExecutionError(
            "Phase 3 requires a clean committed implementation; source identity "
            f"is commit={implementation['git_commit']!r}, "
            f"dirty={implementation['git_dirty']!r}"
        )
    if ctx.cell.phase is Phase.PHASE3 or "confirmation" in ctx.cell.stage:
        if (
            not values["selection.id"]
            or not values["selection.parent_candidate_id"]
            or not values["selection.parent_cell_ids"]
        ):
            raise CellExecutionError(
                "Phase 2 confirmation and Phase 3 are not materializable without "
                "a frozen upstream selection decision and its parent cell IDs"
            )
    if ctx.cell.phase is Phase.PHASE1:
        train = _synthetic_dataset(ctx.cell, "train")
        evaluation_split = str(values["evaluation.split"])
        if evaluation_split == "synthetic_test":
            evaluation_stream = "eval"
        elif evaluation_split == "confirmation":
            evaluation_stream = "confirmation"
        else:
            raise CellExecutionError(
                "Phase-1 evaluation.split must be synthetic_test or confirmation, "
                f"got {evaluation_split!r}"
            )
        calibration_dataset = _synthetic_dataset(ctx.cell, "eval")
        evaluation = _synthetic_dataset(ctx.cell, evaluation_stream)
        ranges = {
            str(name): [int(start), int(stop)]
            for name, start, stop in values["data.synthetic_split_ranges"]
        }
        evaluation_role = (
            "development" if evaluation_split == "synthetic_test" else "confirmation"
        )
        normalization = _normalization_record(train, values)
        data = {
            "kind": "synthetic",
            "source_contract": _synthetic_source_contract(values),
            "train_protocol": train.protocol_dict(),
            "calibration_protocol": calibration_dataset.protocol_dict(),
            "evaluation_protocol": evaluation.protocol_dict(),
            "evaluation_stream": evaluation_stream,
            "normalization": normalization,
            "ranges": {
                "train": [0, int(values["data.train_tokens"])],
                "factor_calibration": ranges["factor_calibration"],
                "calibration": ranges["calibration"],
                "evaluation": ranges[evaluation_role],
            },
            "evaluation_role": evaluation_role,
        }
    else:
        data = {"kind": "activation_store", **_resolve_real_store(values)}
    payload = {
        "schema": PREPARATION_SCHEMA,
        "cell_id": ctx.cell.cell_id,
        "cell_manifest_sha256": _sha256(ctx.cell_path),
        "phase": ctx.cell.phase.value,
        "stage_family": ctx.cell.stage,
        "recipe_name": ctx.cell.recipe_name,
        "recipe_id": ctx.cell.recipe_id,
        "seed": ctx.cell.seed,
        "decisions_sha256": hashlib.sha256(
            canonical_json(ctx.cell.content_payload()).encode("utf-8")
        ).hexdigest(),
        "data": data,
        "runtime": {
            "smoke": values["runtime.smoke"],
            "device": declared_device,
            "torch_version": torch.__version__,
        },
        "random": {
            "model_seed": values["random.model_seed"],
            "structure_seed": values["random.structure_seed"],
            "train_data_seed": values["random.train_data_seed"],
            "eval_data_seed": values["random.eval_data_seed"],
            "confirmation_data_seed": values["random.confirmation_data_seed"],
        },
        "selection": {
            "id": values["selection.id"],
            "source_blueprint_id": values["selection.source_blueprint_id"],
            "source_plan_id": values["selection.source_plan_id"],
            "upstream_selection_ids": list(values["selection.upstream_selection_ids"]),
            "parent_candidate_id": values["selection.parent_candidate_id"],
            "parent_cell_ids": list(values["selection.parent_cell_ids"]),
            "delta_decision_names": list(values["selection.delta_decision_names"]),
            "confirmation_sha256s": list(values["selection.confirmation_sha256s"]),
            "qualification_sha256s": list(values["selection.qualification_sha256s"]),
            "universe_sha256": values["selection.universe_sha256"],
        },
        "implementation": implementation,
    }
    _write_immutable_json(ctx.preparation, payload)
    return (("preparation", ctx.preparation),)


def _unsupported_semantics(values: Mapping[str, Any]) -> None:
    unsupported: list[str] = []
    coefficient = values["objective.regularizer_coefficient"]
    if not isinstance(coefficient, (int, float)) or isinstance(coefficient, bool):
        unsupported.append(f"unresolved regularizer coefficient {coefficient!r}")
    if unsupported:
        raise CellExecutionError(
            "cell declares semantics absent from the shared engine: "
            + "; ".join(unsupported)
            + ". Refusing to run an unlabeled approximation."
        )


def _model_config(cell: CellSpec) -> BSCConfig:
    values = cell.decision_map
    _unsupported_semantics(values)
    site_dims = tuple(int(item) for item in values["data.site_dims"])
    total_groups = int(values["model.groups"])

    selector_name = str(values["model.selector"])
    selection = {
        "token_topk": "token_topk",
        "block_batchtopk": "batch_topk",
        "decoder_weighted_batchtopk": "batch_topk",
        "learned_group_threshold": "dense",
        # Anthropic's paper anchor is dense ReLU plus the explicit
        # activation-weighted decoder L1 objective below.  This is the sole
        # dense training rule retained by the comparison matrix.
        "dense_l1": "dense",
    }.get(selector_name)
    if selection is None:
        raise CellExecutionError(f"unknown resolved selector {selector_name!r}")

    decoder_name = str(values["model.decoder"])
    if decoder_name in {"per_block_stiefel", "concatenated_stiefel"}:
        constraint = "qr"
    elif decoder_name == "concatenated_stiefel_polar":
        constraint = "gram"
    elif decoder_name in {
        "per_block_frobenius_ball",
        "concatenated_frobenius_ball",
    }:
        constraint = "frobenius"
    elif decoder_name in {
        "unit_block_frobenius",
        "concatenated_unit_block_frobenius",
    }:
        constraint = "unit_frobenius"
    elif decoder_name == "unit_row_renorm":
        constraint = "unit_latent"
    elif decoder_name in {
        "free_scale_controlled",
        "free_weight_decay",
        "free_per_site_affine",
    }:
        constraint = "free"
    else:
        raise CellExecutionError(f"unknown resolved decoder {decoder_name!r}")

    regularizer_name = str(values["objective.regularizer"])
    regularizer = {
        "none": "none",
        "end_to_end_map_nuclear": "map_nuclear",
        "decoder_nuclear": "decoder_nuclear",
        "activation_weighted_site_decoder_l1": "crosscoder_l1",
        "group_l21": "group_l21",
    }.get(regularizer_name)
    if regularizer is None:
        raise CellExecutionError(f"unknown resolved regularizer {regularizer_name!r}")
    coefficient_mode = str(values["objective.regularizer_coefficient_mode"])
    target_initial_ratio = values["objective.regularizer_target_initial_ratio"]
    ratio_contract = str(values["objective.regularizer_calibration_contract"])
    if coefficient_mode == "absolute":
        if target_initial_ratio is not None or ratio_contract != "not_applicable":
            raise CellExecutionError(
                "absolute regularizer coefficient has a stray initial-ratio fit contract"
            )
    elif coefficient_mode == "initial_loss_ratio":
        if (
            regularizer == "none"
            or float(values["objective.regularizer_coefficient"]) != 0.0
            or not isinstance(target_initial_ratio, (int, float))
            or isinstance(target_initial_ratio, bool)
            or not math.isfinite(float(target_initial_ratio))
            or float(target_initial_ratio) < 0.0
            or ratio_contract != "post_init_train_prefix_true_observation_fp32_v1"
        ):
            raise CellExecutionError(
                "initial-loss-ratio regularization has an invalid resolved contract"
            )
    else:
        raise CellExecutionError(
            f"unknown objective.regularizer_coefficient_mode {coefficient_mode!r}"
        )

    activation = str(values["model.activation"])
    if activation not in {"signed", "relu", "group_soft_threshold"}:
        raise CellExecutionError(f"unknown resolved activation {activation!r}")
    encoder_name = str(values["model.encoder"])
    known_untied = {
        "untied_affine",
        "untied_affine_unit_column_renorm",
        "untied_linear",
        "joint_untied_affine",
        "joint_untied_linear",
    }
    if encoder_name == "tied_positive_global_scale":
        encoder_mode = "tied"
    elif encoder_name in known_untied:
        encoder_mode = "untied"
    else:
        raise CellExecutionError(f"unknown resolved encoder {encoder_name!r}")
    reconstruction = str(values["objective.reconstruction"])
    reconstruction_loss = {
        "mean_l2": "mean_l2",
        "squared_l2": "squared_l2",
    }.get(reconstruction)
    if reconstruction_loss is None:
        raise CellExecutionError(
            f"unknown resolved reconstruction objective {reconstruction!r}"
        )
    regularizer_schedule = values["objective.regularizer_schedule"]
    regularizer_target = values["objective.regularizer_target"]
    if regularizer_schedule == "always":
        group_lasso_target_k = None
        if regularizer_target is not None:
            raise CellExecutionError(
                "an always-on regularizer must have objective.regularizer_target=None"
            )
    elif regularizer_schedule == "above_target_only":
        if regularizer != "group_l21" or regularizer_target != "model.active_blocks":
            raise CellExecutionError(
                "above_target_only is implemented only for Group L21 targeting "
                "model.active_blocks"
            )
        group_lasso_target_k = float(values["model.active_blocks"])
    else:
        raise CellExecutionError(
            f"unknown objective.regularizer_schedule {regularizer_schedule!r}"
        )
    return BSCConfig(
        n_blocks=total_groups,
        block_dim=int(values["model.block_width"]),
        n_sites=len(site_dims),
        d_model=max(site_dims),
        site_dims=site_dims,
        k=float(values["model.active_blocks"]),
        lambda_regularizer=float(values["objective.regularizer_coefficient"]),
        eig_floor=float(values["model.eig_floor"]),
        sv_eps=float(values["regularizer.sv_eps"]),
        seed=int(values["random.model_seed"]),
        selection=selection,
        encoder_mode=encoder_mode,
        encoder_bias=bool(values["model.encoder_bias"]),
        encoder_constraint=str(values["model.encoder_constraint"]),
        encoder_fusion=str(values["model.encoder_fusion"]),
        encoder_init=str(values["model.encoder_init"]),
        encoder_scale_init=float(values["model.encoder_scale_init"]),
        source_site=int(values["model.source_site"]),
        code_activation=activation,
        selection_score=str(values["model.selection_score"]),
        selector_tie_break=str(values["model.selector_tie_break"]),
        site_rank=(
            None
            if values["model.site_rank"] is None
            else int(values["model.site_rank"])
        ),
        decoder_norm_geometry=str(values["model.decoder_norm_geometry"]),
        decoder_constraint=constraint,
        group_threshold_scope=str(values["model.threshold_scope"]),
        group_threshold_parameterization=str(
            values["model.threshold_parameterization"]
        ),
        group_threshold_raw_init=(
            None
            if values["model.threshold_raw_init"] is None
            else float(values["model.threshold_raw_init"])
        ),
        group_threshold_effective_init=float(values["model.threshold_effective_init"]),
        regularizer=regularizer,
        reconstruction_loss=reconstruction_loss,
        decoder_bias=bool(values["model.decoder_bias"]),
        decoder_bias_init=str(values["model.decoder_bias_init"]),
        apply_decoder_bias_to_input=bool(values["auxiliary.apply_b_dec_to_input"]),
        decoder_init_distribution=str(values["model.decoder_init_distribution"]),
        decoder_init_preconditioning=str(values["model.decoder_init_preconditioning"]),
        decoder_init_operation_order=str(values["model.decoder_init_operation_order"]),
        identical_site_init=bool(values["model.identical_site_init"]),
        group_lasso_target_k=group_lasso_target_k,
        map_nuclear_reduction=(
            "sum_blocks"
            if values["objective.regularizer_reduction"] == "sum_blocks"
            else "mean_normalized"
        ),
    )


def _train_config(cell: CellSpec) -> TrainConfig:
    values = cell.decision_map
    batch = int(values["optimizer.batch_tokens"])
    tokens = int(values["data.train_tokens"])
    total_steps = math.ceil(tokens / batch)
    schedule_name = str(values["optimizer.schedule"])
    schedule = {
        "cosine_to_1e-5": "cosine",
        "cosine": "cosine",
        "warmup_then_final_fifth_linear": "linear_fifth",
        "linear_warmup_1000_then_constant": "constant",
        "linear_warmup_then_constant": "constant",
    }.get(schedule_name)
    if schedule is None:
        raise CellExecutionError(f"unknown resolved schedule {schedule_name!r}")

    aux_name = str(values["objective.auxiliary"])
    aux_variant = {
        "none": "none",
        "runner_up_blocks": "fel",
        "frequency_dead_residual": "sasa",
        "sasa_release_coordinate": "sasa_release",
        "dead_latent_residual": "long_horizon",
        "decoder_weighted_token_horizon_residual": ("decoder_weighted_token_horizon"),
    }.get(aux_name)
    if aux_variant is None:
        raise CellExecutionError(f"unknown resolved auxiliary {aux_name!r}")
    aux_count = values["auxiliary.count"]
    if aux_count == "match_active_blocks":
        aux_count = int(round(float(values["model.active_blocks"])))
    aux_coefficient = values["auxiliary.coefficient"]
    if aux_coefficient == "1/active_blocks":
        aux_coefficient = 1.0 / float(values["model.active_blocks"])
    weight_decay = float(values["optimizer.weight_decay"])
    encoder_weight_decay = float(values["optimizer.encoder_weight_decay"])
    decoder_weight_decay = float(values["optimizer.decoder_weight_decay"])
    bias_weight_decay = float(values["optimizer.bias_weight_decay"])
    optimizer = str(values["optimizer.name"])
    # Adam cannot accept decoupled weight decay; this also catches an invalid
    # recipe rather than silently changing its optimizer identity.
    if optimizer == "adam" and any(
        (weight_decay, encoder_weight_decay, decoder_weight_decay, bias_weight_decay)
    ):
        raise CellExecutionError("an Adam recipe cannot request decoupled weight decay")
    if values["data.outlier_policy"] != "none":
        raise CellExecutionError(
            "outlier masking is outside the revised cross-layer project scope"
        )
    gradient_clip = values["optimizer.gradient_clip_norm"]
    if values["objective.auxiliary_reduction"] != "mean_batch":
        raise CellExecutionError(
            "the shared auxiliary engine implements only mean_batch reduction"
        )
    return TrainConfig(
        total_steps=total_steps,
        lr=float(values["optimizer.learning_rate"]),
        warmup_steps=int(values["optimizer.warmup_steps"]),
        schedule=schedule,
        min_lr_ratio=float(values["optimizer.min_lr_ratio"]),
        final_decay_fraction=float(values["optimizer.final_decay_fraction"]),
        betas=tuple(float(item) for item in values["optimizer.betas"]),
        eps=float(values["optimizer.epsilon"]),
        foreach=bool(values["optimizer.foreach"]),
        fused=bool(values["optimizer.fused"]),
        optimizer=optimizer,
        encoder_weight_decay=encoder_weight_decay,
        decoder_weight_decay=decoder_weight_decay,
        bias_weight_decay=bias_weight_decay,
        gradient_clip_norm=(None if gradient_clip is None else float(gradient_clip)),
        retract_every=int(values["optimizer.retract_every_steps"]),
        forward_dtype=str(values["precision.forward"]),
        aux_variant=aux_variant,
        aux_reconstruction=str(values["objective.auxiliary_reconstruction"]),
        s_aux=max(1, int(aux_count)),
        alpha_aux=float(aux_coefficient),
        dead_threshold=float(values["auxiliary.dead_frequency"]),
        dead_window_tokens=int(values["auxiliary.dead_window_tokens"]),
        dead_horizon_tokens=int(values["auxiliary.dead_after_tokens"]),
        dead_window_passes=int(values["auxiliary.dead_window_passes"]),
        encoder_site_mask_mode=str(
            values.get("objective.encoder_site_mask_mode", "bernoulli")
        ),
        encoder_site_mask_probability=float(
            values["objective.encoder_site_mask_probability"]
        ),
        log_every=int(values["runtime.log_every_steps"]),
    )


def validate_cell_config(cell: CellSpec) -> tuple[BSCConfig, TrainConfig]:
    """Pure, fail-closed resolution of one cell into shared engine configs.

    Matrix and blueprint tests use this without touching a store or allocating
    a model, so every materializable conditional child can be checked before a
    campaign spends GPU time.
    """

    values = cell.decision_map
    if values.get("implementation.materializable") is False:
        raise CellExecutionError(
            "cell recipe is quarantined and cannot be executed: "
            + str(
                values.get(
                    "implementation.quarantine_reason",
                    "source-exact runtime adapter incomplete",
                )
            )
        )
    if values["data.observation_policy"] != "all_sites":
        raise CellExecutionError(
            "the shared engine currently supports only all_sites observation policy"
        )
    if (
        tuple(values["qualification.endpoint_paths"])
        != (
            "native_training_rule",
            "saved_codec_deployment_rule",
        )
        or values["qualification.require_saved_codec_validation"] is not True
    ):
        raise CellExecutionError(
            "qualification must bind native and saved-codec endpoint validation"
        )
    model_cfg = _model_config(cell)
    train_cfg = _train_config(cell)
    expected_steps = math.ceil(
        int(values["data.train_tokens"]) / int(values["optimizer.batch_tokens"])
    )
    if train_cfg.total_steps != expected_steps:
        raise CellExecutionError("resolved trainer step count is inconsistent")
    if model_cfg.site_dims != tuple(int(item) for item in values["data.site_dims"]):
        raise CellExecutionError("resolved model dimensions differ from the cell")
    if train_cfg.aux_variant == "decoder_weighted_token_horizon" and not (
        model_cfg.block_dim == 1
        and values["model.selector"] == "decoder_weighted_batchtopk"
        and train_cfg.aux_reconstruction == "squared_l2_over_residual_variance"
        and values["objective.auxiliary_reduction"] == "mean_batch"
        and model_cfg.apply_decoder_bias_to_input is False
    ):
        raise CellExecutionError(
            "decoder-weighted token-horizon AuxK requires its declared scalar, "
            "score, residual, reduction, and bias-free auxiliary contract"
        )
    return model_cfg, train_cfg


def _load_preparation(path: Path, cell_id: str) -> dict[str, Any]:
    payload = _read_object(path, label="preparation artifact")
    if payload.get("schema") != PREPARATION_SCHEMA or payload.get("cell_id") != cell_id:
        raise CellExecutionError("preparation artifact binding mismatch")
    current_implementation = _implementation_identity()
    if payload.get("implementation") != current_implementation:
        raise CellExecutionError(
            "implementation changed after prepare; create a new content-addressed "
            "campaign cell before executing another stage"
        )
    return payload


def _synthetic_batches(
    cell: CellSpec,
    preparation: Mapping[str, Any],
    role: str,
    batch_size: int,
    *,
    start_override: int | None = None,
    include_observed: bool = False,
) -> Iterator[torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
    data = preparation["data"]
    if role == "train":
        split = "train"
    elif role in {"factor_calibration", "calibration"}:
        # Matching and codec calibration are frozen on the development seed;
        # a confirmation cell may change only its final evaluation stream.
        split = "eval"
    else:
        split = str(data["evaluation_stream"])
    dataset = _synthetic_dataset(cell, split)
    start, stop = (int(item) for item in data["ranges"][role])
    if start_override is not None:
        start = start_override
    for batch in dataset.batches(batch_size, start=start, stop=stop):
        x = _apply_normalization(batch.x, data["normalization"])
        yield (x, batch.observed) if include_observed else x


def _store_reader(
    preparation: Mapping[str, Any], role: str, *, raw: bool = False
) -> StoreReader:
    data = preparation["data"]
    root_key = "raw_root" if raw else "root"
    bindings_key = "raw_bindings" if raw else "bindings"
    root = Path(data[root_key])
    binding = data[bindings_key][role]
    split = str(binding["split"])
    selected_sites = tuple(int(item) for item in binding["selected_site_indices"])
    if not selected_sites:
        raise CellExecutionError(f"{role} store binding has no selected sites")
    manifest = root / split / MANIFEST_NAME
    if _sha256(manifest) != binding["manifest_sha256"]:
        raise CellExecutionError(
            f"{role} store manifest changed after prepare: {manifest}"
        )
    reader = StoreReader(root, split, sites=selected_sites)
    live = {
        "n_tokens": reader.n_tokens,
        "row_stream_sha256": reader.manifest.get("row_stream_sha256"),
        "content_stream_sha256": reader.manifest.get("content_stream_sha256"),
    }
    expected = {name: binding[name] for name in live}
    if live != expected:
        raise CellExecutionError(
            f"{role} store binding changed after prepare: "
            + canonical_json({"expected": expected, "actual": live})
        )
    _verify_store_reader_once(reader, root, split)
    return reader


def _prepared_transform(preparation: Mapping[str, Any]) -> Whitener | None:
    data = preparation["data"]
    record = data.get("normalization", {})
    if record.get("application") != "on_the_fly":
        return None
    path = Path(str(record["transform_path"]))
    if not path.is_file() or _sha256(path) != record["transform_sha256"]:
        raise CellExecutionError(
            f"Phase-3 frozen transform changed after prepare: {path}"
        )
    transform_manifest = Path(str(record["transform_manifest"]))
    if (
        not transform_manifest.is_file()
        or _sha256(transform_manifest) != record["transform_manifest_sha256"]
    ):
        raise CellExecutionError(
            "Phase-3 transform artifact manifest changed after prepare"
        )
    capture_path = Path(str(data["source_contract"]["path"]))
    if (
        not capture_path.is_file()
        or _sha256(capture_path) != record["source_capture_sha256"]
    ):
        raise CellExecutionError("Phase-3 capture contract changed after prepare")
    fit_manifest = Path(str(record["source_fit_manifest"]))
    if (
        not fit_manifest.is_file()
        or _sha256(fit_manifest) != record["source_fit_manifest_file_sha256"]
    ):
        raise CellExecutionError(
            "Phase-3 normalization-fit manifest changed after prepare"
        )
    fit_payload = _read_object(fit_manifest, label="normalization-fit manifest")
    if (
        fit_payload.get("manifest_sha256") != record["source_fit_manifest_sha256"]
        or fit_payload.get("row_stream_sha256")
        != record["source_fit_row_stream_sha256"]
    ):
        raise CellExecutionError("Phase-3 normalization-fit binding mismatch")
    try:
        transform = Whitener.load(path)
    except Exception as exc:  # noqa: BLE001
        raise CellExecutionError(
            f"cannot load frozen Phase-3 transform: {exc}"
        ) from exc
    if (
        transform.hash != record["transform_hash"]
        or transform.mode != record["mode"]
        or transform.meta.get("source_capture_sha256")
        != record["source_capture_sha256"]
        or transform.meta.get("source_fit_manifest_sha256")
        != record["source_fit_manifest_sha256"]
        or transform.meta.get("source_fit_row_stream_sha256")
        != record["source_fit_row_stream_sha256"]
        or transform.n_fit_tokens != record["source_fit_requested_tokens"]
        or transform.meta.get("source_fit_requested_tokens")
        != record["source_fit_requested_tokens"]
    ):
        raise CellExecutionError("Phase-3 frozen transform binding mismatch")
    return transform


def _apply_prepared_transform(
    x: torch.Tensor,
    preparation: Mapping[str, Any],
    transform: Whitener | None,
) -> torch.Tensor:
    if transform is None:
        # Persisted views are already the declared bf16 coordinates.  Keep
        # their compact dtype until the trainer's nonblocking device transfer.
        return x
    record = preparation["data"]["normalization"]
    selected = tuple(int(item) for item in record["selected_site_indices"])
    mode = transform.mode
    cache = getattr(transform, "_application_cache", None)
    if cache is None:
        cache = {}
        setattr(transform, "_application_cache", cache)
    device_key = (x.device.type, x.device.index)
    key = (selected, device_key)
    bound = cache.get(key)
    if bound is None:
        index = torch.tensor(selected, dtype=torch.long)
        mean = transform.mean.index_select(0, index).to(x.device)
        selected_dims = tuple(transform.site_dims[item] for item in selected)
        coordinate_mask = (
            torch.arange(x.shape[2], device=x.device).view(1, -1)
            < torch.tensor(selected_dims, device=x.device).view(-1, 1)
        )
        if mode in {"none", "scalar_rms", "sqrt_d"}:
            operator = torch.diagonal(
                transform.W,
                dim1=-2,
                dim2=-1,
            ).index_select(0, index).to(x.device)
        elif mode == "whiten":
            operator = transform.W.index_select(0, index).to(x.device)
        else:
            operator = None
        bound = (mean, operator, selected_dims, coordinate_mask)
        cache[key] = bound
    mean, operator, selected_dims, coordinate_mask = bound
    if mode == "layer":
        result = torch.zeros_like(x, dtype=torch.float32)
        for site, source_site in enumerate(selected):
            dim = int(transform.site_dims[source_site])
            result[:, site, :dim] = torch.nn.functional.layer_norm(
                x[:, site, :dim].float(),
                (dim,),
                eps=float(transform.meta.get("layer_norm_eps", 1e-5)),
            )
    elif mode in {"none", "scalar_rms", "sqrt_d"}:
        assert operator is not None
        result = (x.float() - mean) * operator.unsqueeze(0)
    else:
        assert operator is not None
        result = torch.einsum("sde,nse->nsd", operator, x.float() - mean)
    return result * coordinate_mask.unsqueeze(0)


def _transform_on_cuda(
    preparation: Mapping[str, Any],
    device: torch.device,
) -> bool:
    data = preparation["data"]
    return (
        device.type == "cuda"
        and data["kind"] == "activation_store"
        and data.get("normalization", {}).get("application") == "on_the_fly"
    )


def _decode_transform(
    preparation: Mapping[str, Any],
) -> tuple[Whitener | None, tuple[int, ...]]:
    """Load the exact inverse transform required by a deployed raw decoder."""

    data = preparation["data"]
    if data["kind"] == "synthetic":
        return None, tuple(range(len(data["normalization"]["site_dims"])))
    selected = tuple(
        int(item) for item in data["bindings"]["calibration"]["selected_site_indices"]
    )
    transform = _prepared_transform(preparation)
    if transform is None:
        normalized_root = Path(data["root"])
        raw_root = Path(data["raw_root"])
        transform_path = normalized_root / "whitener.pt"
        if transform_path.is_file():
            transform = Whitener.load(transform_path)
            normalized = _store_reader(preparation, "calibration")
            if transform.hash != normalized.whitener_hash:
                raise CellExecutionError(
                    "normalized calibration store/transform hash mismatch"
                )
        elif normalized_root != raw_root:
            raise CellExecutionError(
                "a persisted normalized view lacks its deployable inverse transform"
            )
    return transform, selected


def _raw_calibration_mean(
    ctx: _Context, preparation: Mapping[str, Any]
) -> tuple[torch.Tensor, int]:
    """Fit the zero-event reconstruction in raw coordinates on calibration only."""

    sites = len(tuple(ctx.values["data.site_dims"]))
    width = max(int(item) for item in ctx.values["data.site_dims"])
    total = torch.zeros(sites, width, dtype=torch.float64)
    count = 0
    batch_size = int(ctx.values["optimizer.batch_tokens"])
    if preparation["data"]["kind"] == "synthetic":
        dataset = _synthetic_dataset(ctx.cell, "eval")
        start, stop = preparation["data"]["ranges"]["calibration"]
        batches = (
            item.x
            for item in dataset.batches(batch_size, start=int(start), stop=int(stop))
        )
    else:
        batches = _store_reader(
            preparation, "calibration", raw=True
        ).sequential_batches(batch_size)
    for batch in batches:
        total += batch.double().sum(dim=0)
        count += len(batch)
    if count <= 0:
        raise CellExecutionError("raw calibration split is empty")
    return total / count, count


def _deployment_codec_payload(
    ctx: _Context,
    preparation: Mapping[str, Any],
    model: BlockCrosscoder,
    *,
    checkpoint_hash: str,
    checkpoint_metadata: Mapping[str, Any],
    calibration_hash: str,
    preparation_hash: str,
) -> dict[str, Any]:
    """Build the actual saved consumer artifact priced by the R-D policy.

    The optimizer checkpoint is intentionally absent.  The artifact contains
    every trained model tensor needed to encode/decode, every frozen codec
    tensor, and the raw-space inverse/zero-event mean.
    """

    codec_payload = torch.load(ctx.calibration, map_location="cpu", weights_only=True)
    if not isinstance(codec_payload, dict):
        raise CellExecutionError("calibration artifact is not a codec mapping")
    # Validate the exact nested bytes before incorporating them into the only
    # consumer artifact that evaluation is allowed to load.
    Codec.from_payload(codec_payload, source=str(ctx.calibration))
    raw_mean, raw_mean_count = _raw_calibration_mean(ctx, preparation)
    if preparation["data"]["kind"] == "synthetic":
        normalization: dict[str, Any] = {
            "kind": "synthetic",
            "record": dict(preparation["data"]["normalization"]),
        }
    else:
        transform, selected = _decode_transform(preparation)
        normalization = {
            "kind": "identity" if transform is None else "frozen_transform",
            "selected_site_indices": list(selected),
            "record": dict(preparation["data"].get("normalization", {})),
        }
        if transform is not None:
            index = torch.tensor(selected, dtype=torch.long)
            normalization.update(
                {
                    "mode": transform.mode,
                    "W": transform.W.index_select(0, index).cpu(),
                    "mean": transform.mean.index_select(0, index).cpu(),
                    "site_dims": [transform.site_dims[item] for item in selected],
                    "meta": dict(transform.meta),
                    "transform_hash": transform.hash,
                }
            )
    payload = {
        "format_version": 2,
        "schema": "bsc-deployable-codec-v2",
        "cell_id": ctx.cell.cell_id,
        "checkpoint_sha256": checkpoint_hash,
        "calibration_sha256": calibration_hash,
        "preparation_sha256": preparation_hash,
        "model_cfg": asdict(model.cfg),
        "model_state": {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in model.state_dict().items()
        },
        "codec_payload": codec_payload,
        "training_summary": {
            "step_idx": int(checkpoint_metadata["step_idx"]),
            "accepted_tokens": int(checkpoint_metadata["accepted_tokens"]),
        },
        "normalization": normalization,
        "raw_calibration_mean": raw_mean,
        "raw_calibration_mean_fit_tokens": raw_mean_count,
        "rate_contract": {
            "packet_contract": ctx.values["codec.packet_contract"],
            "side_information_contract": ctx.values["codec.side_information_contract"],
            "time_sharing_schedule_contract": ctx.values[
                "codec.time_sharing_schedule_contract"
            ],
            "artifact_includes_optimizer": False,
            "artifact_bytes_are_priced_exactly": True,
        },
    }
    payload["artifact_sha256"] = _tensor_payload_digest(payload)
    return payload


def _load_deployable_codec(
    path: Path,
    *,
    cell_id: str,
    checkpoint_hash: str,
    calibration_hash: str,
    preparation_hash: str,
    device: torch.device,
) -> tuple[dict[str, Any], BlockCrosscoder, Codec, dict[str, int]]:
    """Load the complete consumer path solely from the priced artifact."""

    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001
        raise CellExecutionError(f"cannot load deployable codec: {exc}") from exc
    if not isinstance(value, dict):
        raise CellExecutionError("deployable codec is not a mapping")
    if set(value) != _DEPLOYABLE_CODEC_KEYS:
        raise CellExecutionError(
            "deployable codec field set mismatch: "
            + canonical_json(
                {
                    "missing": sorted(_DEPLOYABLE_CODEC_KEYS - set(value)),
                    "extra": sorted(set(value) - _DEPLOYABLE_CODEC_KEYS),
                }
            )
        )
    payload = dict(value)
    claimed = payload.pop("artifact_sha256", None)
    if not isinstance(claimed, str) or claimed != _tensor_payload_digest(payload):
        raise CellExecutionError("deployable codec internal content hash mismatch")
    if (
        payload.get("format_version") != 2
        or payload.get("schema") != "bsc-deployable-codec-v2"
        or payload.get("cell_id") != cell_id
        or payload.get("checkpoint_sha256") != checkpoint_hash
        or payload.get("calibration_sha256") != calibration_hash
        or payload.get("preparation_sha256") != preparation_hash
    ):
        raise CellExecutionError("deployable codec/input binding mismatch")
    try:
        cfg_payload = payload["model_cfg"]
        state = payload["model_state"]
        if not isinstance(cfg_payload, dict) or not isinstance(state, dict):
            raise TypeError("model config/state must be mappings")
        model = BlockCrosscoder(BSCConfig(**cfg_payload))
        if set(cfg_payload) != set(asdict(model.cfg)):
            raise ValueError("model config does not contain the exact resolved fields")
        expected_state = model.state_dict()
        if set(state) != set(expected_state):
            raise ValueError("model state field set differs from the resolved model")
        for name, expected_tensor in expected_state.items():
            actual_tensor = state[name]
            if (
                not torch.is_tensor(actual_tensor)
                or actual_tensor.shape != expected_tensor.shape
                or actual_tensor.dtype != expected_tensor.dtype
            ):
                raise ValueError(f"model state tensor contract mismatch for {name}")
            if actual_tensor.is_floating_point() and not bool(
                torch.isfinite(actual_tensor).all()
            ):
                raise ValueError(f"model state tensor {name} is nonfinite")
        model.load_state_dict(state, strict=True)
        model = model.to(device).eval()
        codec_payload = payload["codec_payload"]
        if not isinstance(codec_payload, dict):
            raise TypeError("nested codec payload must be a mapping")
        codec = Codec.from_payload(codec_payload, source=f"{path}:codec_payload")
        summary = payload["training_summary"]
        if not isinstance(summary, dict) or set(summary) != {
            "step_idx",
            "accepted_tokens",
        }:
            raise TypeError("training summary must be a mapping")
        if any(
            not isinstance(summary[name], int)
            or isinstance(summary[name], bool)
            or summary[name] <= 0
            for name in ("step_idx", "accepted_tokens")
        ):
            raise ValueError("training summary values must be positive integers")
        training_summary = {
            "step_idx": int(summary["step_idx"]),
            "accepted_tokens": int(summary["accepted_tokens"]),
        }
        normalization = payload["normalization"]
        if not isinstance(normalization, dict):
            raise TypeError("normalization must be a mapping")
        kind = normalization.get("kind")
        if kind not in {"synthetic", "identity", "frozen_transform"}:
            raise ValueError("normalization has an unknown kind")
        if kind == "synthetic":
            if set(normalization) != {"kind", "record"}:
                raise ValueError("synthetic normalization field set mismatch")
            record = normalization.get("record")
            if not isinstance(record, dict):
                raise TypeError("synthetic normalization record must be a mapping")
            mean = torch.as_tensor(record["mean"], dtype=torch.float32)
            scale = torch.as_tensor(record["scale"], dtype=torch.float32)
            if mean.shape != (model.cfg.n_sites, model.cfg.d_model):
                raise ValueError("synthetic normalization mean shape mismatch")
            if scale.shape != (model.cfg.n_sites,):
                raise ValueError("synthetic normalization scale shape mismatch")
            if not bool(torch.isfinite(mean).all() and torch.isfinite(scale).all()):
                raise ValueError("synthetic normalization is nonfinite")
            if bool((scale <= 0).any()):
                raise ValueError("synthetic normalization scale must be positive")
        else:
            expected_normalization_keys = {
                "kind",
                "selected_site_indices",
                "record",
            }
            if kind == "frozen_transform":
                expected_normalization_keys |= {
                    "mode",
                    "W",
                    "mean",
                    "site_dims",
                    "meta",
                    "transform_hash",
                }
            if set(normalization) != expected_normalization_keys:
                raise ValueError("real normalization field set mismatch")
            selected = normalization.get("selected_site_indices")
            if (
                not isinstance(selected, list)
                or len(selected) != model.cfg.n_sites
                or not all(isinstance(item, int) and item >= 0 for item in selected)
                or len(set(selected)) != len(selected)
            ):
                raise ValueError("real normalization has invalid selected-site indices")
            if kind == "frozen_transform":
                mode = normalization.get("mode")
                if mode not in {"none", "scalar_rms", "sqrt_d", "whiten", "layer"}:
                    raise ValueError("frozen normalization has an unknown mode")
                W = normalization.get("W")
                mean = normalization.get("mean")
                site_dims = normalization.get("site_dims")
                if (
                    not torch.is_tensor(W)
                    or W.shape
                    != (
                        model.cfg.n_sites,
                        model.cfg.d_model,
                        model.cfg.d_model,
                    )
                    or W.dtype != torch.float32
                ):
                    raise ValueError("frozen normalization W shape mismatch")
                if (
                    not torch.is_tensor(mean)
                    or mean.shape
                    != (
                        model.cfg.n_sites,
                        model.cfg.d_model,
                    )
                    or mean.dtype != torch.float32
                ):
                    raise ValueError("frozen normalization mean shape mismatch")
                if list(model.cfg.site_dims) != site_dims:
                    raise ValueError("frozen normalization site dimensions mismatch")
                if not bool(torch.isfinite(W).all() and torch.isfinite(mean).all()):
                    raise ValueError("frozen normalization is nonfinite")
                if not isinstance(normalization.get("transform_hash"), str):
                    raise ValueError("frozen normalization lacks its transform hash")
                if mode in {"none", "scalar_rms", "sqrt_d"}:
                    diagonal = torch.diag_embed(torch.diagonal(W, dim1=-2, dim2=-1))
                    if not torch.equal(W, diagonal) or bool(
                        (torch.diagonal(W, dim1=-2, dim2=-1) <= 0).any()
                    ):
                        raise ValueError(
                            "scalar frozen normalization is not positive diagonal"
                        )
                elif mode == "whiten":
                    if not torch.allclose(W, W.transpose(-1, -2), atol=1e-5, rtol=1e-5):
                        raise ValueError("whitening matrices are not symmetric")
                    if bool((torch.linalg.eigvalsh(W.double()) <= 0).any()):
                        raise ValueError(
                            "whitening matrices are not invertible positive maps"
                        )
        raw_mean = payload["raw_calibration_mean"]
        if (
            not torch.is_tensor(raw_mean)
            or raw_mean.shape
            != (
                model.cfg.n_sites,
                model.cfg.d_model,
            )
            or raw_mean.dtype != torch.float64
        ):
            raise ValueError("raw calibration mean shape mismatch")
        if not bool(torch.isfinite(raw_mean).all()):
            raise ValueError("raw calibration mean is nonfinite")
        raw_fit_tokens = payload["raw_calibration_mean_fit_tokens"]
        if (
            not isinstance(raw_fit_tokens, int)
            or isinstance(raw_fit_tokens, bool)
            or raw_fit_tokens <= 0
        ):
            raise ValueError("raw calibration mean has no fit rows")
        rate_contract = payload["rate_contract"]
        if (
            not isinstance(rate_contract, dict)
            or set(rate_contract)
            != {
                "packet_contract",
                "side_information_contract",
                "time_sharing_schedule_contract",
                "artifact_includes_optimizer",
                "artifact_bytes_are_priced_exactly",
            }
            or (
                rate_contract.get("artifact_includes_optimizer") is not False
                or rate_contract.get("artifact_bytes_are_priced_exactly") is not True
                or rate_contract.get("packet_contract")
                != "fixed_width_count_compact_block_id_amplitude_v1"
                or rate_contract.get("side_information_contract")
                != "exact_deployable_saved_codec_bytes_v1"
                or rate_contract.get("time_sharing_schedule_contract")
                not in {
                    "not_applicable",
                    "balanced_global_token_counter_u64_v1",
                }
            )
        ):
            raise ValueError("deployable codec has an invalid rate contract")
    except Exception as exc:  # noqa: BLE001
        raise CellExecutionError(
            f"cannot reconstruct consumer from deployable codec: {exc}"
        ) from exc
    if codec.meta.get("cell_id") != cell_id:
        raise CellExecutionError("nested codec is bound to another cell")
    if codec.meta.get("checkpoint_sha256") != checkpoint_hash:
        raise CellExecutionError("nested codec checkpoint binding mismatch")
    if abs(float(model.theta) - float(codec.meta["theta"])) > 1e-12:
        raise CellExecutionError(
            "deployable model and nested codec thresholds disagree"
        )
    return value, model, codec, training_summary


def _save_immutable_torch(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        existing = torch.load(path, map_location="cpu", weights_only=True)
        if _tensor_payload_digest(existing) != _tensor_payload_digest(dict(payload)):
            raise CellExecutionError(
                f"immutable torch artifact changed binding: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _training_batches(
    ctx: _Context,
    preparation: Mapping[str, Any],
    *,
    start_token: int,
    apply_transform: bool = True,
) -> Iterator[torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
    batch_size = int(ctx.values["optimizer.batch_tokens"])
    target = int(ctx.values["data.train_tokens"])
    if preparation["data"]["kind"] == "synthetic":
        yield from _synthetic_batches(
            ctx.cell,
            preparation,
            "train",
            batch_size,
            start_override=start_token,
            include_observed=True,
        )
        return
    reader = _store_reader(preparation, "train")
    transform = _prepared_transform(preparation) if apply_transform else None
    unique_tokens = int(ctx.values["data.unique_tokens"])
    consumed = 0
    raw_stream = reader.shuffled_batches(
        batch_size,
        seed=int(ctx.values["random.train_data_seed"]),
        epochs=None,
        prefix_tokens=unique_tokens,
    )
    carry: torch.Tensor | None = None
    for chunk in raw_stream:
        carry = chunk if carry is None else torch.cat((carry, chunk), dim=0)
        if len(carry) < batch_size:
            continue
        batch, carry = carry[:batch_size], carry[batch_size:]
        if consumed + len(batch) <= start_token:
            consumed += len(batch)
            continue
        if consumed < start_token:
            batch = batch[start_token - consumed :]
            consumed = start_token
        remaining = target - consumed
        if remaining <= 0:
            return
        batch = batch[:remaining]
        yield (
            _apply_prepared_transform(batch, preparation, transform)
            if apply_transform
            else batch
        )
        consumed += len(batch)


def _evaluation_batches(
    ctx: _Context,
    preparation: Mapping[str, Any],
    role: str,
) -> Iterator[torch.Tensor]:
    batch_size = int(ctx.values["optimizer.batch_tokens"])
    if preparation["data"]["kind"] == "synthetic":
        for batch in _synthetic_batches(ctx.cell, preparation, role, batch_size):
            assert isinstance(batch, torch.Tensor)
            yield batch
    else:
        reader = _store_reader(preparation, role)
        transform = _prepared_transform(preparation)
        for batch in reader.sequential_batches(batch_size):
            yield _apply_prepared_transform(batch, preparation, transform)


def _rd_evaluation_batches(
    ctx: _Context,
    preparation: Mapping[str, Any],
) -> Iterator[torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
    """Use true stored sequence IDs for every real-model R-D bootstrap."""

    batch_size = int(ctx.values["optimizer.batch_tokens"])
    if preparation["data"]["kind"] == "synthetic":
        yield from _evaluation_batches(ctx, preparation, "evaluation")
        return
    reader = _store_reader(preparation, "evaluation")
    transform = _prepared_transform(preparation)
    for batch, row_ids in reader.sequential_batches_with_ids(batch_size):
        yield _apply_prepared_transform(batch, preparation, transform), row_ids


def _prefetched_evaluation_batches(
    ctx: _Context,
    preparation: Mapping[str, Any],
    role: str = "evaluation",
) -> Iterator[torch.Tensor]:
    batches: Iterator[torch.Tensor] = _evaluation_batches(ctx, preparation, role)
    if _device(ctx).type == "cuda":
        batches = prefetch_batches(batches, depth=2, pin_memory=True)
    return batches


def _prefetched_rd_evaluation_batches(
    ctx: _Context,
    preparation: Mapping[str, Any],
) -> Iterator[torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
    batches = _rd_evaluation_batches(ctx, preparation)
    if _device(ctx).type == "cuda":
        batches = prefetch_batches(batches, depth=2, pin_memory=True)
    return batches


def _unpack_training_batch(
    batch: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(batch, tuple):
        return batch
    return batch, None


def _tensor_digest(*tensors: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(canonical_json(list(value.shape)).encode("ascii"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _initialization_slice(
    ctx: _Context,
    preparation: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, Any]]:
    try:
        batch = next(_training_batches(ctx, preparation, start_token=0))
    except StopIteration as exc:
        raise CellExecutionError("training stream has no initialization batch") from exc
    x, observed = _unpack_training_batch(batch)
    digest_inputs = [x]
    if observed is not None:
        digest_inputs.append(observed)
    return (
        x,
        observed,
        {
            "split": "train",
            "start_token": 0,
            "tokens": len(x),
            "sha256": _tensor_digest(*digest_inputs),
            "observation_mask_bound": observed is not None,
        },
    )


def _encoder_scale_fit_batches(
    ctx: _Context,
    preparation: Mapping[str, Any],
) -> Iterator[torch.Tensor]:
    batch_size = int(ctx.values["optimizer.batch_tokens"])
    fit_split = str(ctx.values["model.encoder_scale_fit_split"])
    fit_count = int(ctx.values["model.encoder_scale_fit_count"])
    if preparation["data"]["kind"] == "synthetic":
        if fit_split != "train_unique_prefix":
            raise CellExecutionError(
                "synthetic encoder-scale fitting requires "
                "model.encoder_scale_fit_split='train_unique_prefix'"
            )
        dataset = _synthetic_dataset(ctx.cell, "train")
        if not 0 < fit_count <= dataset.unique_examples:
            raise CellExecutionError(
                "model.encoder_scale_fit_count exceeds the unique synthetic "
                "training prefix"
            )
        for item in dataset.batches(batch_size, start=0, stop=fit_count):
            yield _apply_normalization(item.x, preparation["data"]["normalization"])
        return
    if fit_split != "normalization_fit":
        raise CellExecutionError(
            "real encoder-scale fitting requires "
            "model.encoder_scale_fit_split='normalization_fit'"
        )
    reader = _store_reader(preparation, "normalization_fit")
    transform = _prepared_transform(preparation)
    remaining = fit_count
    for batch in reader.sequential_batches(batch_size):
        if remaining <= 0:
            return
        batch = batch[:remaining]
        yield _apply_prepared_transform(batch, preparation, transform)
        remaining -= len(batch)
    if remaining:
        raise CellExecutionError(
            "normalization-fit store has fewer rows than the resolved fit count"
        )


@torch.no_grad()
def _apply_encoder_scale_calibration(
    ctx: _Context,
    preparation: Mapping[str, Any],
    model: BlockCrosscoder,
) -> dict[str, Any]:
    strategy = str(ctx.values["model.encoder_scale_calibration"])
    if strategy == "fixed_init_no_data_fit":
        if (
            ctx.values["model.encoder_scale_fit_split"],
            ctx.values["model.encoder_scale_fit_count"],
            ctx.values["model.encoder_scale_fit_statistic"],
        ) != ("not_applicable", 0, "not_applicable"):
            raise CellExecutionError(
                "fixed encoder scale cannot declare a data-fit contract"
            )
        return {
            "strategy": strategy,
            "fitted": False,
            "scale_multiplier": 1.0,
        }
    if strategy != "fit_global_mean_block_norm_to_one_on_normalization_fit":
        raise CellExecutionError(
            f"unknown model.encoder_scale_calibration {strategy!r}"
        )
    if ctx.values["model.encoder_scale_fit_statistic"] != (
        "global_fp64_mean_block_score"
    ):
        raise CellExecutionError(
            "fitted encoder scale requires the declared global fp64 mean block "
            "score statistic"
        )
    total = torch.zeros((), dtype=torch.float64, device=model.parameter_device)
    count = 0
    decoder = model.decoder_tensor()
    encoder = (
        decoder * model.log_gamma.exp()
        if model.cfg.encoder_mode == "tied"
        else model.encoder_tensor()
    )
    score_geometry = model._frozen_score_geometry(decoder)
    for batch in _encoder_scale_fit_batches(ctx, preparation):
        x = batch.to(device=model.parameter_device, dtype=torch.float32)
        z, keep = model._encode_with_tensor(x, encoder)
        scores = model.scores(
            z,
            x=x,
            _decoder=decoder,
            _observation_keep=keep,
            _score_geometry=score_geometry,
        )
        total += scores.double().sum()
        count += scores.numel()
    if count == 0:
        raise CellExecutionError("encoder-scale normalization-fit stream is empty")
    mean_score = float(total / count)
    if not math.isfinite(mean_score) or mean_score <= 0:
        raise CellExecutionError(
            f"encoder-scale fit produced invalid mean block norm {mean_score}"
        )
    multiplier = 1.0 / mean_score
    model.scale_encoder_(multiplier)
    if preparation["data"]["kind"] == "synthetic":
        input_binding: dict[str, Any] = {
            "kind": "stateless_generator_prefix",
            "source_contract_sha256": preparation["data"]["source_contract"]["sha256"],
            "start": 0,
            "stop": int(ctx.values["model.encoder_scale_fit_count"]),
        }
    else:
        binding = preparation["data"]["bindings"]["normalization_fit"]
        input_binding = {
            "kind": "activation_store_split",
            "split": binding["split"],
            "manifest_sha256": binding["manifest_sha256"],
            "row_stream_sha256": binding["row_stream_sha256"],
            "content_stream_sha256": binding["content_stream_sha256"],
            "available_tokens": binding["n_tokens"],
            "prefix_tokens": int(ctx.values["model.encoder_scale_fit_count"]),
        }
    return {
        "strategy": strategy,
        "fitted": True,
        "input": input_binding,
        "events": count,
        "mean_block_norm_before": mean_score,
        "scale_multiplier": multiplier,
        "mean_block_norm_after": mean_score * multiplier,
    }


@torch.no_grad()
def _apply_regularizer_ratio_calibration(
    ctx: _Context,
    model: BlockCrosscoder,
    init_x: torch.Tensor,
    init_observed: torch.Tensor | None,
    initialization_input: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve an undisclosed penalty coefficient from a dimensionless ratio.

    The fit occurs after every declared initializer and encoder-scale fit but
    before optimizer state exists.  The first training batch is already
    content-bound by ``initialization_input``; clean true-observation targets
    are used without stochastic encoder-site masking.
    """

    values = ctx.values
    mode = str(values["objective.regularizer_coefficient_mode"])
    declared_coefficient = float(values["objective.regularizer_coefficient"])
    contract = str(values["objective.regularizer_calibration_contract"])
    if mode == "absolute":
        if contract != "not_applicable":
            raise CellExecutionError(
                "absolute regularizer coefficient has a non-inert fit contract"
            )
        if not math.isclose(
            float(model.cfg.lambda_regularizer),
            declared_coefficient,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise CellExecutionError(
                "absolute regularizer coefficient changed before training"
            )
        return {
            "mode": mode,
            "contract": contract,
            "fitted": False,
            "declared_absolute_coefficient": declared_coefficient,
            "target_initial_ratio": None,
            "resolved_coefficient": declared_coefficient,
        }
    if (
        mode != "initial_loss_ratio"
        or contract != "post_init_train_prefix_true_observation_fp32_v1"
    ):
        raise CellExecutionError(f"unsupported regularizer calibration mode {mode!r}")
    target_ratio = float(values["objective.regularizer_target_initial_ratio"])
    if declared_coefficient != 0.0 or float(model.cfg.lambda_regularizer) != 0.0:
        raise CellExecutionError(
            "initial-loss-ratio fit requires a zero absolute coefficient placeholder"
        )
    x = init_x.to(device=model.parameter_device, dtype=torch.float32)
    observed = (
        None
        if init_observed is None
        else init_observed.to(device=model.parameter_device, dtype=torch.bool)
    )
    model.cfg.lambda_regularizer = 1.0
    try:
        output = model(x, observed=observed)
        parts = bsc_loss(output, x, model, observation_mask=observed)
        raw_regularizer = parts.get("regularizer")
        if raw_regularizer is None:
            raise CellExecutionError(
                "initial-loss-ratio fit did not execute the declared regularizer"
            )
        reconstruction = float(parts["rec"].detach())
        regularizer = float(raw_regularizer.detach())
    finally:
        model.cfg.lambda_regularizer = 0.0
    if (
        not math.isfinite(reconstruction)
        or reconstruction <= 0.0
        or not math.isfinite(regularizer)
        or regularizer <= 0.0
    ):
        raise CellExecutionError(
            "initial-loss-ratio fit requires finite positive reconstruction and penalty"
        )
    resolved = target_ratio * reconstruction / regularizer
    if not math.isfinite(resolved) or resolved < 0.0:
        raise CellExecutionError(
            "initial-loss-ratio fit produced an invalid coefficient"
        )
    model.cfg.lambda_regularizer = resolved
    achieved_ratio = resolved * regularizer / reconstruction
    if not math.isclose(achieved_ratio, target_ratio, rel_tol=1e-12, abs_tol=1e-15):
        raise CellExecutionError(
            "initial-loss-ratio fit failed its arithmetic invariant"
        )
    return {
        "mode": mode,
        "contract": contract,
        "fitted": True,
        "input": dict(initialization_input),
        "declared_absolute_coefficient": declared_coefficient,
        "target_initial_ratio": target_ratio,
        "initial_reconstruction_loss": reconstruction,
        "initial_regularizer_unweighted": regularizer,
        "resolved_coefficient": resolved,
        "achieved_initial_ratio": achieved_ratio,
    }


@torch.no_grad()
def _production_precision_preflight(
    ctx: _Context,
    model: BlockCrosscoder,
    init_x: torch.Tensor,
    init_observed: torch.Tensor | None,
) -> dict[str, Any]:
    """Compare fp32 and bf16 initial forward paths before optimization."""

    contract = str(ctx.values["precision.preflight_contract"])
    if contract == "not_applicable":
        return {"applicable": False, "contract": contract, "passed": True}
    if contract != "fp32_bf16_initial_forward_v1":
        raise CellExecutionError(f"unsupported precision preflight {contract!r}")
    tokens = int(ctx.values["precision.preflight_tokens"])
    if tokens <= 0 or len(init_x) < tokens:
        raise CellExecutionError(
            "precision preflight token count exceeds the content-bound initial batch"
        )
    x_fp32 = init_x[:tokens].to(device=model.parameter_device, dtype=torch.float32)
    observed = (
        None
        if init_observed is None
        else init_observed[:tokens].to(device=model.parameter_device, dtype=torch.bool)
    )
    fp32_output = model(x_fp32, observed=observed)
    fp32_parts = bsc_loss(fp32_output, x_fp32, model, observation_mask=observed)
    bf16_model = copy.deepcopy(model).to(torch.bfloat16)
    try:
        x_bf16 = x_fp32.to(torch.bfloat16)
        bf16_output = bf16_model(x_bf16, observed=observed)
        bf16_parts = bsc_loss(
            bf16_output, x_bf16, bf16_model, observation_mask=observed
        )
        fp32_rec = float(fp32_parts["rec"].detach())
        bf16_rec = float(bf16_parts["rec"].detach())
        rec_relative_error = abs(bf16_rec - fp32_rec) / max(abs(fp32_rec), 1e-30)
        intersection = (fp32_output.mask & bf16_output.mask).sum(dim=1).float()
        union = (fp32_output.mask | bf16_output.mask).sum(dim=1).float()
        support_iou = float(
            torch.where(union > 0, intersection / union, torch.ones_like(union)).mean()
        )
        output_relative_error = float(
            (bf16_output.xhat.float() - fp32_output.xhat.float()).norm()
            / fp32_output.xhat.float().norm().clamp_min(1e-30)
        )
        finite = bool(
            torch.isfinite(fp32_output.xhat).all()
            and torch.isfinite(bf16_output.xhat).all()
            and math.isfinite(fp32_rec)
            and math.isfinite(bf16_rec)
            and math.isfinite(rec_relative_error)
            and math.isfinite(support_iou)
            and math.isfinite(output_relative_error)
        )
    finally:
        del bf16_model
    rec_max = float(ctx.values["precision.preflight_reconstruction_relative_error_max"])
    iou_min = float(ctx.values["precision.preflight_support_iou_min"])
    checks = {
        "finite": finite,
        "reconstruction_relative_error": rec_relative_error <= rec_max,
        "support_iou": support_iou >= iou_min,
    }
    return {
        "applicable": True,
        "contract": contract,
        "input": {
            "split": "train",
            "start_token": 0,
            "tokens": tokens,
            "sha256": _tensor_digest(
                x_fp32,
                *(() if observed is None else (observed,)),
            ),
            "observation_mask_bound": observed is not None,
        },
        "fp32_reconstruction_loss": fp32_rec,
        "bf16_reconstruction_loss": bf16_rec,
        "reconstruction_relative_error": rec_relative_error,
        "support_iou": support_iou,
        "output_relative_error": output_relative_error,
        "thresholds": {
            "reconstruction_relative_error_max": rec_max,
            "support_iou_min": iou_min,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def _binding(
    ctx: _Context,
    preparation: Mapping[str, Any],
    model_cfg: BSCConfig,
    train_cfg: TrainConfig,
    initialization: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "cell_id": ctx.cell.cell_id,
        "recipe_id": ctx.cell.recipe_id,
        "preparation_sha256": _sha256(ctx.preparation),
        "model_cfg": asdict(model_cfg),
        "train_cfg": asdict(train_cfg),
        "initialization": dict(initialization),
        "data": preparation["data"],
        "runtime": preparation["runtime"],
        "selection": preparation["selection"],
        "implementation": preparation["implementation"],
    }


def _validate_final_checkpoint(
    path: Path, binding: Mapping[str, Any]
) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 - convert corrupt torch files cleanly
        raise CellExecutionError(
            f"cannot load immutable checkpoint {path}: {exc}"
        ) from exc
    if payload.get("run_binding") != binding:
        raise CellExecutionError("existing checkpoint has a different run binding")
    data_cursor = payload.get("data_cursor")
    if not isinstance(data_cursor, dict):
        raise CellExecutionError("final checkpoint lacks exact data-cursor state")
    step_idx = payload.get("step_idx")
    accepted_tokens = payload.get("accepted_tokens")
    next_token = data_cursor.get("next_token")
    if (
        not isinstance(step_idx, int)
        or isinstance(step_idx, bool)
        or step_idx < 0
        or not isinstance(accepted_tokens, int)
        or isinstance(accepted_tokens, bool)
        or accepted_tokens < 0
        or not isinstance(next_token, int)
        or isinstance(next_token, bool)
        or next_token != accepted_tokens
        or data_cursor.get("stream") != "train"
    ):
        raise CellExecutionError("final checkpoint has an invalid execution cursor")
    optimizer_kind = payload.get("optimizer_kind")
    if not isinstance(optimizer_kind, str) or not optimizer_kind:
        raise CellExecutionError("final checkpoint lacks resolved optimizer identity")
    history = payload.get("history")
    previous_shares = payload.get("diagnostic_prev_shares")
    model_payload = payload.get("model_cfg")
    expected_share_shape = (
        (
            int(model_payload.get("n_sites", -1)),
            int(model_payload.get("n_blocks", -1)),
        )
        if isinstance(model_payload, dict)
        else (-1, -1)
    )
    if (
        not isinstance(history, list)
        or any(not isinstance(item, dict) for item in history)
        or not torch.is_tensor(previous_shares)
        or tuple(previous_shares.shape) != expected_share_shape
        or not bool(torch.isfinite(previous_shares).all())
    ):
        raise CellExecutionError("final checkpoint lacks exact diagnostic state")
    return {
        "model_cfg": payload.get("model_cfg"),
        "train_cfg": payload.get("train_cfg"),
        "run_binding": payload.get("run_binding"),
        "step_idx": step_idx,
        "accepted_tokens": accepted_tokens,
        "data_cursor": dict(data_cursor),
        "optimizer_kind": optimizer_kind,
        "terminal_log": dict(history[-1]) if history else None,
    }


def _training_report_payload(
    ctx: _Context,
    *,
    metadata: Mapping[str, Any],
    checkpoint_hash: str,
    preparation_hash: str,
    terminal_log: Mapping[str, Any] | None,
) -> dict[str, Any]:
    data_cursor = metadata["data_cursor"]
    run_binding = metadata.get("run_binding")
    if not isinstance(run_binding, Mapping):
        raise CellExecutionError("final checkpoint lacks its run binding")
    initialization = run_binding.get("initialization")
    if not isinstance(initialization, Mapping):
        raise CellExecutionError("final checkpoint lacks initialization provenance")
    regularizer_calibration = initialization.get("regularizer_calibration")
    precision_preflight = initialization.get("precision_preflight")
    if not isinstance(regularizer_calibration, Mapping):
        raise CellExecutionError(
            "final checkpoint lacks regularizer-calibration provenance"
        )
    if not isinstance(precision_preflight, Mapping):
        raise CellExecutionError(
            "final checkpoint lacks precision-preflight provenance"
        )
    return {
        "schema": TRAINING_REPORT_SCHEMA,
        "cell_id": ctx.cell.cell_id,
        "checkpoint_sha256": checkpoint_hash,
        "preparation_sha256": preparation_hash,
        "step_idx": metadata.get("step_idx"),
        "accepted_tokens": metadata.get("accepted_tokens"),
        "attempted_tokens": int(data_cursor["next_token"]),
        "optimizer_kind": metadata["optimizer_kind"],
        "terminal_log": terminal_log,
        "data_cursor": dict(data_cursor),
        "model_cfg": metadata["model_cfg"],
        "train_cfg": metadata["train_cfg"],
        "regularizer_calibration": dict(regularizer_calibration),
        "precision_preflight": dict(precision_preflight),
    }


def _seed_fresh_training_rng(seed: int) -> None:
    """Seed every RNG captured by Trainer before a fresh optimization run."""

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)


def _train(
    ctx: _Context,
    prerequisites: Mapping[str, tuple[Path, str]],
    *,
    resume: bool,
) -> tuple[tuple[str, Path], ...]:
    preparation = _load_preparation(prerequisites["preparation"][0], ctx.cell.cell_id)
    device = _device(ctx)
    model_cfg, train_cfg = validate_cell_config(ctx.cell)
    init_x, init_observed, initialization = _initialization_slice(ctx, preparation)
    initialization_input = dict(initialization)
    initialization.update(
        {
            "decoder_bias_init": model_cfg.decoder_bias_init,
            "encoder_scale_init": model_cfg.encoder_scale_init,
        }
    )
    model = BlockCrosscoder(model_cfg, device=device)
    if init_observed is not None and not bool(init_observed.all()):
        if model_cfg.decoder_bias_init != "zero":
            raise CellExecutionError(
                "data-derived initialization with missing sites requires a "
                "mask-aware initializer; refusing to treat missing values as observed"
            )
    init_x_device = init_x.to(device=device, dtype=torch.float32)
    model.initialize_decoder_bias_(init_x_device)
    model.project_decoder_()
    initialization["encoder_scale_calibration"] = _apply_encoder_scale_calibration(
        ctx, preparation, model
    )
    initialization["regularizer_calibration"] = _apply_regularizer_ratio_calibration(
        ctx,
        model,
        init_x,
        init_observed,
        initialization_input,
    )
    initialization["precision_preflight"] = _production_precision_preflight(
        ctx,
        model,
        init_x,
        init_observed,
    )
    if initialization["precision_preflight"].get("passed") is not True:
        raise CellExecutionError(
            "production precision preflight failed before optimizer construction"
        )
    binding = _binding(
        ctx,
        preparation,
        model.cfg,
        train_cfg,
        initialization,
    )

    if ctx.checkpoint.exists():
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        metadata = _validate_final_checkpoint(ctx.checkpoint, binding)
        final_cursor = metadata["data_cursor"].get("next_token")
        if final_cursor != int(ctx.values["data.train_tokens"]):
            raise CellExecutionError(
                "final checkpoint data cursor does not equal data.train_tokens"
            )
        if metadata["accepted_tokens"] != int(ctx.values["data.train_tokens"]):
            raise CellExecutionError(
                "final checkpoint did not consume every declared training token"
            )
        if metadata["step_idx"] != train_cfg.total_steps:
            raise CellExecutionError(
                "final checkpoint optimizer step does not equal the resolved budget"
            )
        if not ctx.training_report.exists():
            if not resume:
                raise CellExecutionError(
                    "an orphan final checkpoint requires --resume to reconstruct "
                    "its hash-bound training report"
                )
            _write_immutable_json(
                ctx.training_report,
                _training_report_payload(
                    ctx,
                    metadata=metadata,
                    checkpoint_hash=_sha256(ctx.checkpoint),
                    preparation_hash=prerequisites["preparation"][1],
                    terminal_log=metadata["terminal_log"],
                ),
            )
        report = _read_object(ctx.training_report, label="training report")
        if (
            report.get("checkpoint_sha256") != _sha256(ctx.checkpoint)
            or report.get("attempted_tokens") != int(final_cursor)
            or report.get("data_cursor") != metadata["data_cursor"]
            or report.get("accepted_tokens") != metadata["accepted_tokens"]
            or report.get("step_idx") != metadata["step_idx"]
            or report.get("optimizer_kind") != metadata["optimizer_kind"]
            or canonical_json(report.get("model_cfg"))
            != canonical_json(metadata["model_cfg"])
            or canonical_json(report.get("train_cfg"))
            != canonical_json(metadata["train_cfg"])
            or canonical_json(report.get("regularizer_calibration"))
            != canonical_json(
                metadata["run_binding"]["initialization"]["regularizer_calibration"]
            )
            or canonical_json(report.get("precision_preflight"))
            != canonical_json(
                metadata["run_binding"]["initialization"]["precision_preflight"]
            )
        ):
            raise CellExecutionError("training report/checkpoint binding mismatch")
        if ctx.progress.exists():
            ctx.progress.unlink()
        return (
            ("checkpoint", ctx.checkpoint),
            ("training_report", ctx.training_report),
        )

    if ctx.progress.exists():
        if not resume:
            raise CellExecutionError(
                f"partial checkpoint exists at {ctx.progress}; rerun with --resume"
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        try:
            trainer = Trainer.load_checkpoint(
                ctx.progress, device=device, expected_binding=binding
            )
        except Exception as exc:  # noqa: BLE001
            raise CellExecutionError(
                f"cannot resume partial checkpoint: {exc}"
            ) from exc
    else:
        _seed_fresh_training_rng(int(ctx.values["random.model_seed"]))
        trainer = Trainer(model, train_cfg, run_binding=binding)

    start_token = int(trainer.data_cursor.get("next_token", trainer.accepted_tokens))
    if start_token != trainer.accepted_tokens:
        raise CellExecutionError(
            "partial checkpoint accepted-token count differs from its data cursor"
        )
    expected_resume_steps = math.ceil(
        start_token / int(ctx.values["optimizer.batch_tokens"])
    )
    if trainer.step_idx != expected_resume_steps:
        raise CellExecutionError(
            "partial checkpoint optimizer step differs from its exact data cursor"
        )
    checkpoint_tokens = max(1, int(ctx.values["runtime.checkpoint_tokens"]))
    next_checkpoint = ((start_token // checkpoint_tokens) + 1) * checkpoint_tokens
    cuda_transform: Whitener | None = None
    transform_on_cuda = _transform_on_cuda(preparation, device)
    if transform_on_cuda:
        cuda_transform = _prepared_transform(preparation)
        if cuda_transform is None:
            raise CellExecutionError("on-the-fly normalization has no frozen transform")
    training_batches: Iterator = _training_batches(
        ctx,
        preparation,
        start_token=start_token,
        apply_transform=not transform_on_cuda,
    )
    if device.type == "cuda":
        # Keep two pinned raw batches ready so shard latency and the next H2D
        # transfer do not sit on the GPU's critical path.  On-the-fly Phase-3
        # transforms execute after that nonblocking transfer on CUDA.
        training_batches = prefetch_batches(
            training_batches,
            depth=2,
            pin_memory=True,
        )
    try:
        for batch in training_batches:
            if trainer.step_idx >= trainer.cfg.total_steps:
                break
            x, observed = _unpack_training_batch(batch)
            if cuda_transform is not None:
                x = x.to(device=device, non_blocking=True)
                x = _apply_prepared_transform(x, preparation, cuda_transform)
            trainer.step(x, observed=observed)
            start_token += int(x.shape[0])
            trainer.data_cursor = {
                "next_token": start_token,
                "stream": "train",
            }
            if (
                start_token >= next_checkpoint
                and trainer.step_idx < trainer.cfg.total_steps
            ):
                trainer.save_checkpoint(ctx.progress)
                next_checkpoint += checkpoint_tokens
    finally:
        close_batches = getattr(training_batches, "close", None)
        if close_batches is not None:
            close_batches()
    if trainer.step_idx != trainer.cfg.total_steps:
        raise CellExecutionError(
            f"training stream ended at step {trainer.step_idx}/{trainer.cfg.total_steps}"
        )
    if (
        start_token != int(ctx.values["data.train_tokens"])
        or trainer.accepted_tokens != start_token
    ):
        raise CellExecutionError(
            "training reached its step budget without consuming the exact "
            "declared token budget"
        )
    # The final durable payload replaces the progress payload atomically.  At
    # no point do we retain two complete optimizer checkpoints for one cell.
    trainer.save_checkpoint(ctx.progress)
    ctx.progress.replace(ctx.checkpoint)
    metadata = _validate_final_checkpoint(ctx.checkpoint, binding)
    if metadata["data_cursor"].get("next_token") != start_token:
        raise CellExecutionError("final checkpoint cursor differs from training stream")
    if metadata["accepted_tokens"] != start_token:
        raise CellExecutionError(
            "final checkpoint accepted-token count differs from training stream"
        )
    history = trainer.history
    report = _training_report_payload(
        ctx,
        metadata=metadata,
        checkpoint_hash=_sha256(ctx.checkpoint),
        preparation_hash=prerequisites["preparation"][1],
        terminal_log=history[-1] if history else None,
    )
    _write_immutable_json(ctx.training_report, report)
    return (
        ("checkpoint", ctx.checkpoint),
        ("training_report", ctx.training_report),
    )


def _calibrate(
    ctx: _Context,
    prerequisites: Mapping[str, tuple[Path, str]],
) -> tuple[tuple[str, Path], ...]:
    preparation = _load_preparation(prerequisites["preparation"][0], ctx.cell.cell_id)
    checkpoint_path, checkpoint_hash = prerequisites["checkpoint"]
    training_report = _read_object(
        prerequisites["training_report"][0], label="training report"
    )
    if (
        training_report.get("schema") != TRAINING_REPORT_SCHEMA
        or training_report.get("cell_id") != ctx.cell.cell_id
        or training_report.get("checkpoint_sha256") != checkpoint_hash
    ):
        raise CellExecutionError("training report/checkpoint binding mismatch")
    try:
        model, metadata = load_trained_model(checkpoint_path, device=_device(ctx))
    except Exception as exc:  # noqa: BLE001
        raise CellExecutionError(
            f"cannot load checkpoint for calibration: {exc}"
        ) from exc
    binding = metadata.get("run_binding") or {}
    if binding.get("cell_id") != ctx.cell.cell_id:
        raise CellExecutionError("checkpoint is not bound to this cell")
    target = float(ctx.values["model.active_blocks"])
    threshold_estimator = str(ctx.values["inference.threshold_estimator"])
    threshold_source_name = str(ctx.values["inference.threshold_source"])
    if (
        threshold_estimator != "heldout_target_rate"
        or threshold_source_name != "calibration_quantile"
    ):
        raise CellExecutionError(
            "inference threshold estimator/source disagree: "
            "the saved codec requires heldout_target_rate/calibration_quantile"
        )
    quantile_method = "exact" if ctx.cell.phase is Phase.PHASE1 else "streaming"
    model.fit_threshold_(
        _evaluation_batches(ctx, preparation, "calibration"),
        target_avg_blocks=target,
        method=quantile_method,
    )
    threshold_source: dict[str, Any] = {
        "split": str(ctx.values["evaluation.calibration_split"]),
        "quantile_method": quantile_method,
        "target_avg_blocks": target,
    }
    achieved_events = 0
    achieved_tokens = 0
    with torch.no_grad():
        calibration_decoder = model.decoder_tensor()
        calibration_encoder = (
            calibration_decoder * model.log_gamma.exp()
            if model.cfg.encoder_mode == "tied"
            else model.encoder_tensor()
        )
        calibration_score_geometry = model._frozen_score_geometry(
            calibration_decoder
        )
        for batch in _evaluation_batches(ctx, preparation, "calibration"):
            x = batch.to(device=_device(ctx), dtype=torch.float32)
            z, keep = model._encode_with_tensor(x, calibration_encoder)
            scores = model.scores(
                z,
                x=x,
                _decoder=calibration_decoder,
                _observation_keep=keep,
                _score_geometry=calibration_score_geometry,
            )
            selected = model._select_scores(scores, mode="threshold", z=z)
            achieved_events += int(selected.sum())
            achieved_tokens += len(x)
    if achieved_tokens == 0:
        raise CellExecutionError("calibration split is empty")
    achieved_avg_blocks = achieved_events / achieved_tokens
    threshold_error = abs(achieved_avg_blocks - target)
    if ctx.values["codec.bootstrap_seed_source"] != "random.eval_data_seed":
        raise CellExecutionError("unsupported codec bootstrap seed source")
    if (
        ctx.values["codec.packet_contract"]
        != "fixed_width_count_compact_block_id_amplitude_v1"
    ):
        raise CellExecutionError("unsupported codec packet contract")
    if (
        ctx.values["codec.side_information_contract"]
        != "exact_deployable_saved_codec_bytes_v1"
    ):
        raise CellExecutionError("unsupported codec side-information contract")
    spec = CodecSpec(
        qs=tuple(int(item) for item in ctx.values["codec.quantizer_bits"]),
        clip_lo=float(ctx.values["codec.clip_lower_quantile"]),
        clip_hi=float(ctx.values["codec.clip_upper_quantile"]),
        floor=int(ctx.values["codec.minimum_active_events_per_block"]),
        n_bootstrap=int(ctx.values["codec.bootstrap_replicates"]),
        bootstrap_seed=int(ctx.values["random.eval_data_seed"]),
        max_calibration_event_bytes=int(
            ctx.values["codec.max_calibration_event_bytes"]
        ),
    )
    preflight_event_bytes = estimate_calibration_peak_bytes(
        achieved_events,
        model.cfg.block_dim,
    )
    if preflight_event_bytes > spec.max_calibration_event_bytes:
        raise CellExecutionError(
            "exact codec calibration exceeds its resolved event-memory ceiling "
            "before event materialization: "
            f"estimated {preflight_event_bytes} > "
            f"{spec.max_calibration_event_bytes} bytes"
        )
    codec = fit_codec(
        model,
        _evaluation_batches(ctx, preparation, "calibration"),
        spec,
        device=str(_device(ctx)),
    )
    codec.meta.update(
        {
            "schema": "bsc-calibration-v1",
            "cell_id": ctx.cell.cell_id,
            "checkpoint_sha256": checkpoint_hash,
            "preparation_sha256": prerequisites["preparation"][1],
            "theta": float(model.theta),
            "target_avg_blocks": target,
            "achieved_avg_blocks": achieved_avg_blocks,
            "threshold_abs_error": threshold_error,
            "threshold_estimator": threshold_estimator,
            "threshold_source_name": threshold_source_name,
            "threshold_source": threshold_source,
            "calibration_event_memory_preflight_bytes": preflight_event_bytes,
            "packet_contract": ctx.values["codec.packet_contract"],
            "side_information_contract": ctx.values["codec.side_information_contract"],
        }
    )
    if ctx.calibration.exists():
        try:
            existing = Codec.load(ctx.calibration)
        except Exception as exc:  # noqa: BLE001
            raise CellExecutionError(f"existing calibration is corrupt: {exc}") from exc
        expected_binding = {
            key: codec.meta[key]
            for key in (
                "schema",
                "cell_id",
                "checkpoint_sha256",
                "preparation_sha256",
                "threshold_estimator",
                "threshold_source_name",
                "threshold_source",
            )
        }
        if any(
            existing.meta.get(key) != value for key, value in expected_binding.items()
        ):
            raise CellExecutionError("existing calibration has a different binding")
    else:
        codec.save(ctx.calibration)
    # Always reload the exact durable bytes before reporting calibration.
    frozen = Codec.load(ctx.calibration)
    deployment_payload = _deployment_codec_payload(
        ctx,
        preparation,
        model,
        checkpoint_hash=checkpoint_hash,
        checkpoint_metadata=metadata,
        calibration_hash=_sha256(ctx.calibration),
        preparation_hash=prerequisites["preparation"][1],
    )
    _save_immutable_torch(ctx.deployment_codec, deployment_payload)
    # Reconstruct the exact consumer before pricing; truncated, incomplete, or
    # internally inconsistent bytes never reach evaluation.
    frozen_deployment, _, frozen_consumer_codec, _ = _load_deployable_codec(
        ctx.deployment_codec,
        cell_id=ctx.cell.cell_id,
        checkpoint_hash=checkpoint_hash,
        calibration_hash=_sha256(ctx.calibration),
        preparation_hash=prerequisites["preparation"][1],
        device=torch.device("cpu"),
    )
    if (
        frozen_deployment.get("schema") != "bsc-deployable-codec-v2"
        or frozen_deployment.get("cell_id") != ctx.cell.cell_id
        or frozen_deployment.get("checkpoint_sha256") != checkpoint_hash
        or frozen_deployment.get("calibration_sha256") != _sha256(ctx.calibration)
    ):
        raise CellExecutionError("deployable codec artifact binding mismatch")
    if frozen_consumer_codec.meta != frozen.meta:
        raise CellExecutionError("deployable codec differs from calibrated codec")
    excluded_calibration_events = int(
        frozen.calib_events[~frozen.included].sum().item()
    )
    total_calibration_events = int(frozen.calib_events.sum().item())
    record = {
        "schema": "bsc-calibration-record-v1",
        "cell_id": ctx.cell.cell_id,
        "checkpoint_sha256": checkpoint_hash,
        "codec_sha256": _sha256(ctx.calibration),
        "deployment_codec_sha256": _sha256(ctx.deployment_codec),
        "deployment_codec_size_bytes": ctx.deployment_codec.stat().st_size,
        "theta": float(frozen.meta["theta"]),
        "target_avg_blocks": target,
        "achieved_avg_blocks": float(frozen.meta["achieved_avg_blocks"]),
        "threshold_abs_error": float(frozen.meta["threshold_abs_error"]),
        "threshold_estimator": frozen.meta["threshold_estimator"],
        "threshold_source_name": frozen.meta["threshold_source_name"],
        "threshold_source": frozen.meta["threshold_source"],
        "calibration_event_memory_preflight_bytes": frozen.meta[
            "calibration_event_memory_preflight_bytes"
        ],
        "calibration_event_memory_ceiling_bytes": (
            frozen.spec.max_calibration_event_bytes
        ),
        "calibration_tokens": frozen.calib_tokens,
        "included_blocks": frozen.n_included,
        "excluded_blocks": int((~frozen.included).sum()),
        "excluded_calibration_events": excluded_calibration_events,
        "excluded_calibration_event_fraction": (
            excluded_calibration_events / max(1, total_calibration_events)
        ),
    }
    _write_immutable_json(ctx.calibration_record, record)
    return (
        ("calibration", ctx.calibration),
        ("deployment_codec", ctx.deployment_codec),
        ("calibration_record", ctx.calibration_record),
    )


def _orthonormal_columns(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.numel() == 0:
        return matrix.new_zeros((matrix.shape[0], 0))
    q, r = torch.linalg.qr(matrix.double(), mode="reduced")
    keep = torch.diagonal(r).abs() > 1e-8
    return q[:, keep]


def _support_matched_subspace_overlap(
    overlaps: torch.Tensor,
    factor_to_group: torch.Tensor,
) -> torch.Tensor:
    """Read each factor's overlap from its support-selected learned group."""

    if overlaps.ndim != 2 or factor_to_group.shape != (overlaps.shape[0],):
        raise CellExecutionError("subspace matching tensors have incompatible shapes")
    if factor_to_group.dtype != torch.long:
        raise CellExecutionError("factor-to-group assignment must use integer indices")
    if bool(((factor_to_group < 0) | (factor_to_group >= overlaps.shape[1])).any()):
        raise CellExecutionError("factor-to-group assignment is outside learned groups")
    return overlaps.gather(1, factor_to_group.view(-1, 1)).squeeze(1)


def _support_confusion(
    predicted: torch.Tensor,
    truth: torch.Tensor,
) -> dict[str, float | int]:
    """Binary event metrics with false-discovery and false-positive separated."""

    tp = int((predicted & truth).sum())
    fp = int((predicted & ~truth).sum())
    fn = int((~predicted & truth).sum())
    tn = int((~predicted & ~truth).sum())
    return {
        "true_positive_events": tp,
        "false_positive_events": fp,
        "false_negative_events": fn,
        "true_negative_events": tn,
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "false_discovery_rate": fp / max(1, fp + tp),
        "false_positive_rate": fp / max(1, fp + tn),
    }


def _matching_pathologies(association: torch.Tensor) -> dict[str, float]:
    """Directional planted-factor/learned-group multiplicity diagnostics."""

    strong = association >= 0.5
    weak = association >= 0.25
    associated_groups = strong.any(dim=0)
    merged_groups = strong.sum(dim=0) > 1
    associated_group_count = int(associated_groups.sum())
    factors_in_merged_groups = (
        strong[:, merged_groups].any(dim=1)
        if bool(merged_groups.any())
        else torch.zeros(strong.shape[0], dtype=torch.bool)
    )
    # Rows are truth factors, columns are learned groups.
    return {
        "split_factor_fraction": float((strong.sum(dim=1) > 1).float().mean()),
        # Normalize over associated groups, not allocated dictionary capacity;
        # otherwise adding inactive groups mechanically improves this score.
        "merge_group_fraction": (
            float(merged_groups.sum()) / associated_group_count
            if associated_group_count
            else 0.0
        ),
        # This capacity-invariant factor-level view is the qualification gate.
        "merged_factor_fraction": float(factors_in_merged_groups.float().mean()),
        "shattering_factor_fraction": float((weak.sum(dim=1) > 1).float().mean()),
        "dilution_factor_fraction": float(
            (association.max(dim=1).values < 0.25).float().mean()
        ),
    }


@torch.no_grad()
def _synthetic_recovery(
    model: BlockCrosscoder,
    matching_dataset: Phase1Dataset,
    evaluation_dataset: Phase1Dataset,
    normalization: Mapping[str, Any],
    *,
    selection_mode: str,
    factor_calibration_range: tuple[int, int],
    evaluation_range: tuple[int, int],
    batch_size: int,
) -> dict[str, Any]:
    """Truth-aware recovery with matching frozen before scored rows are read."""

    device = next(model.parameters()).device
    if selection_mode not in {"topk", "threshold"}:
        raise CellExecutionError("synthetic recovery selection mode is invalid")
    if not matching_dataset.stream_digest or not evaluation_dataset.stream_digest:
        raise CellExecutionError("synthetic recovery requires finalized protocols")
    if (
        len(matching_dataset.factors) != len(evaluation_dataset.factors)
        or matching_dataset.site_dims != evaluation_dataset.site_dims
    ):
        raise CellExecutionError(
            "synthetic matching/evaluation truth structures differ"
        )
    calibration_start, calibration_stop = factor_calibration_range
    evaluation_start, evaluation_stop = evaluation_range
    if calibration_stop <= calibration_start or evaluation_stop <= evaluation_start:
        raise CellExecutionError("synthetic recovery ranges must be nonempty")

    materialized_decoder = model.decoder_tensor()
    materialized_encoder = (
        materialized_decoder * model.log_gamma.exp()
        if model.cfg.encoder_mode == "tied"
        else model.encoder_tensor()
    )
    score_geometry = model._frozen_score_geometry(materialized_decoder)

    def frozen_forward(
        x: torch.Tensor,
        *,
        observed: torch.Tensor | None = None,
    ):
        return model.forward_with_materialized(
            x,
            mode=selection_mode,
            observed=observed,
            _decoder=materialized_decoder,
            _encoder=materialized_encoder,
            _score_geometry=score_geometry,
        )[0]

    dataset = evaluation_dataset
    n_factors = len(dataset.factors)
    n_groups = model.cfg.n_blocks
    coactive = torch.zeros(n_factors, n_groups, dtype=torch.float64)
    truth_count = torch.zeros(n_factors, dtype=torch.float64)
    predicted_count = torch.zeros(n_groups, dtype=torch.float64)
    matching_examples = 0
    for matching_batch in matching_dataset.batches(
        batch_size,
        start=calibration_start,
        stop=calibration_stop,
    ):
        matching_x = _apply_normalization(matching_batch.x, normalization).to(device)
        matching_out = frozen_forward(
            matching_x,
            observed=matching_batch.observed.to(device),
        )
        truth = matching_batch.active.bool().cpu()
        predicted_blocks = matching_out.mask.bool().cpu()
        coactive += truth.double().T @ predicted_blocks.double()
        truth_count += truth.sum(dim=0).double()
        predicted_count += predicted_blocks.sum(dim=0).double()
        matching_examples += len(truth)
    if matching_examples != calibration_stop - calibration_start:
        raise CellExecutionError("factor-calibration stream ended early")
    association = (
        2
        * coactive
        / (truth_count.unsqueeze(1) + predicted_count.unsqueeze(0)).clamp_min(1)
    )
    best_association, factor_to_group = association.max(dim=1)
    group_to_factor = association.argmax(dim=0)

    claim_value = dataset.ground_truth.get("shared_feature_claim_eligible")
    shared_feature_claim_eligible = (
        bool(claim_value) if claim_value is not None else len(dataset.site_dims) > 1
    )
    if shared_feature_claim_eligible:
        shared_feature_claim_reason = (
            "shared_support_and_coordinate_across_multiple_sites"
        )
    elif len(dataset.site_dims) <= 1:
        shared_feature_claim_reason = "single_site_source_or_control"
    elif (
        getattr(dataset, "config", None) is not None
        and getattr(dataset.config, "step", None) == "shared_support"
    ):
        shared_feature_claim_reason = "support_only_without_shared_coordinates"
    else:
        shared_feature_claim_reason = "factor_present_at_only_one_site"

    D = model.decoder_tensor().detach().cpu().double()
    coordinate_mask = model.coordinate_mask[:, 0, 0].cpu().bool()
    learned: list[torch.Tensor] = []
    for group in range(model.cfg.n_blocks):
        columns = D[:, group].permute(0, 2, 1).reshape(-1, model.cfg.block_dim)
        learned.append(_orthonormal_columns(columns[coordinate_mask.reshape(-1)]))

    maps = dataset.contribution_maps.double()
    subspace_eligible = normalization["kind"] != "token_layer_norm"
    if subspace_eligible:
        scales = torch.tensor(normalization["scale"], dtype=torch.float64)
        maps = maps * scales.view(1, -1, 1, 1)
    overlaps = torch.zeros(n_factors, model.cfg.n_blocks)
    if subspace_eligible:
        for factor, metadata in enumerate(dataset.factors):
            rank = metadata.coordinate_dim
            columns = maps[factor, :, :, :rank].reshape(-1, rank)
            truth = _orthonormal_columns(columns[coordinate_mask.reshape(-1)])
            if truth.shape[1] == 0:
                continue
            for group, basis in enumerate(learned):
                if basis.shape[1]:
                    overlaps[factor, group] = (
                        (truth.T @ basis).square().sum() / truth.shape[1]
                    ).float()
    matched_overlap = (
        _support_matched_subspace_overlap(overlaps, factor_to_group)
        if subspace_eligible
        else None
    )
    # Fit one affine map from the selected learned block to each planted
    # coordinate using factor-calibration rows only.  Lists contain only codes
    # and intrinsic coordinates, never ambient contributions, so peak memory
    # is independent of the large source dimension.
    alignment_coefficients: list[torch.Tensor | None] = [None] * n_factors
    alignment_references: list[torch.Tensor | None] = [None] * n_factors
    if subspace_eligible:
        latent_parts: list[list[torch.Tensor]] = [[] for _ in range(n_factors)]
        target_parts: list[list[torch.Tensor]] = [[] for _ in range(n_factors)]
        zero_mean_normalization = {
            **normalization,
            "mean": torch.zeros_like(torch.tensor(normalization["mean"])).tolist(),
        }
        for matching_batch in matching_dataset.batches(
            batch_size,
            start=calibration_start,
            stop=calibration_stop,
        ):
            isolated = _apply_normalization(
                matching_batch.contributions, zero_mean_normalization
            ).to(device)
            isolated_out = frozen_forward(isolated)
            selected_codes = isolated_out.z_selected.detach().cpu().double()
            for factor in matching_batch.event_factor.unique().tolist():
                rows = torch.nonzero(
                    matching_batch.event_factor == factor, as_tuple=False
                ).flatten()
                group = int(factor_to_group[factor])
                latent_parts[factor].append(selected_codes[rows, group])
                target_parts[factor].append(
                    matching_batch.coordinates[rows].reshape(len(rows), -1).double()
                )
        for factor in range(n_factors):
            if not latent_parts[factor]:
                continue
            latent = torch.cat(latent_parts[factor])
            target = torch.cat(target_parts[factor])
            if len(latent) < 2 * (model.cfg.block_dim + 1):
                continue
            design = torch.cat(
                (latent, torch.ones(len(latent), 1, dtype=torch.float64)), dim=1
            )
            alignment_coefficients[factor] = torch.linalg.lstsq(design, target).solution
            alignment_references[factor] = target.mean(dim=0, keepdim=True)

    support_totals = {name: 0 for name in ("tp", "fp", "fn", "tn")}
    category_totals = {
        name: {key: 0 for key in ("tp", "fp", "fn", "true")}
        for name in dataset.category_names
    }
    category_masks = {
        name: torch.tensor(
            [factor.category == name for factor in dataset.factors],
            dtype=torch.bool,
        )
        for name in dataset.category_names
    }
    alive = torch.zeros(n_groups, dtype=torch.bool)
    isolated_error = torch.zeros(n_factors, dtype=torch.float64)
    isolated_total = torch.zeros(n_factors, dtype=torch.float64)
    code_error = torch.zeros(n_factors, dtype=torch.float64)
    code_total = torch.zeros(n_factors, dtype=torch.float64)
    selected_count = torch.zeros(n_factors, dtype=torch.float64)
    isolated_count = torch.zeros(n_factors, dtype=torch.float64)
    evaluation_examples = 0
    zero_mean_normalization = {
        **normalization,
        "mean": torch.zeros_like(torch.tensor(normalization["mean"])).tolist(),
    }
    zero_input = torch.zeros(
        1,
        model.cfg.n_sites,
        model.cfg.d_model,
        device=device,
    )
    zero_prediction = frozen_forward(zero_input).xhat
    for batch in evaluation_dataset.batches(
        batch_size,
        start=evaluation_start,
        stop=evaluation_stop,
    ):
        x = _apply_normalization(batch.x, normalization).to(device)
        out = frozen_forward(x, observed=batch.observed.to(device))
        truth_active = batch.active.bool().cpu()
        block_mask = out.mask.bool().cpu()
        alive |= block_mask.any(dim=0)
        nz = block_mask.nonzero(as_tuple=False)
        predicted = torch.zeros_like(truth_active)
        if len(nz):
            predicted[nz[:, 0], group_to_factor[nz[:, 1]]] = True
        support_totals["tp"] += int((predicted & truth_active).sum())
        support_totals["fp"] += int((predicted & ~truth_active).sum())
        support_totals["fn"] += int((~predicted & truth_active).sum())
        support_totals["tn"] += int((~predicted & ~truth_active).sum())
        for name, mask in category_masks.items():
            category_totals[name]["tp"] += int(
                (predicted[:, mask] & truth_active[:, mask]).sum()
            )
            category_totals[name]["fp"] += int(
                (predicted[:, mask] & ~truth_active[:, mask]).sum()
            )
            category_totals[name]["fn"] += int(
                (~predicted[:, mask] & truth_active[:, mask]).sum()
            )
            category_totals[name]["true"] += int(truth_active[:, mask].sum())

        if subspace_eligible and batch.n_events:
            isolated = _apply_normalization(
                batch.contributions, zero_mean_normalization
            ).to(device)
            isolated_out = frozen_forward(isolated)
            isolated_hat = isolated_out.xhat - zero_prediction
            event_error = (
                (isolated_hat - isolated).double().square().sum(dim=(1, 2)).cpu()
            )
            event_total = isolated.double().square().sum(dim=(1, 2)).cpu()
            isolated_error.index_add_(0, batch.event_factor, event_error)
            isolated_total.index_add_(0, batch.event_factor, event_total)
            selected_codes = isolated_out.z_selected.detach().cpu().double()
            selected_masks = isolated_out.mask.detach().cpu()
            for factor in batch.event_factor.unique().tolist():
                rows = torch.nonzero(
                    batch.event_factor == factor, as_tuple=False
                ).flatten()
                coefficient = alignment_coefficients[factor]
                reference = alignment_references[factor]
                if coefficient is None or reference is None:
                    continue
                group = int(factor_to_group[factor])
                latent = selected_codes[rows, group]
                design = torch.cat(
                    (latent, torch.ones(len(rows), 1, dtype=torch.float64)), dim=1
                )
                target = batch.coordinates[rows].reshape(len(rows), -1).double()
                prediction = design @ coefficient
                code_error[factor] += (prediction - target).square().sum()
                code_total[factor] += (target - reference).square().sum()
                selected_count[factor] += selected_masks[rows, group].sum()
                isolated_count[factor] += len(rows)
        evaluation_examples += len(batch.x)
    if evaluation_examples != evaluation_stop - evaluation_start:
        raise CellExecutionError("synthetic evaluation stream ended early")

    tp, fp = support_totals["tp"], support_totals["fp"]
    fn, tn = support_totals["fn"], support_totals["tn"]
    support = {
        "true_positive_events": tp,
        "false_positive_events": fp,
        "false_negative_events": fn,
        "true_negative_events": tn,
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "false_discovery_rate": fp / max(1, fp + tp),
        "false_positive_rate": fp / max(1, fp + tn),
    }
    category = {
        name: {
            "precision": counts["tp"] / max(1, counts["tp"] + counts["fp"]),
            "recall": counts["tp"] / max(1, counts["tp"] + counts["fn"]),
            "true_events": counts["true"],
        }
        for name, counts in category_totals.items()
    }
    global_isolated_r2_by_factor = [
        None
        if isolated_total[factor] <= 0
        else 1.0 - float(isolated_error[factor] / isolated_total[factor])
        for factor in range(n_factors)
    ]
    code_r2_by_factor = [
        None
        if code_total[factor] <= 0
        else 1.0 - float(code_error[factor] / code_total[factor])
        for factor in range(n_factors)
    ]
    selected_group_coverage_by_factor = [
        None
        if isolated_count[factor] <= 0
        else float(selected_count[factor] / isolated_count[factor])
        for factor in range(n_factors)
    ]
    global_isolated_r2 = (
        None
        if isolated_total.sum() <= 0
        else 1.0 - float(isolated_error.sum() / isolated_total.sum())
    )
    code_r2_after_alignment = (
        None
        if code_total.sum() <= 0
        else 1.0 - float(code_error.sum() / code_total.sum())
    )

    recovered = best_association >= 0.5
    pathologies = _matching_pathologies(association)
    return {
        "selection_mode": selection_mode,
        "shared_feature_claim_eligible": shared_feature_claim_eligible,
        "shared_feature_claim_reason": shared_feature_claim_reason,
        "n_factor_calibration_examples": matching_examples,
        "n_examples": evaluation_examples,
        "n_truth_factors": n_factors,
        "support_precision": support["precision"],
        "support_recall": support["recall"],
        "support_false_discovery_rate": support["false_discovery_rate"],
        "support_false_positive_rate": support["false_positive_rate"],
        "support_confusion": support,
        "support_association_f1_mean": float(best_association.mean()),
        "subspace_metrics_eligible": subspace_eligible,
        "subspace_overlap_mean": (
            None if matched_overlap is None else float(matched_overlap.mean())
        ),
        "subspace_overlap_median": (
            None if matched_overlap is None else float(matched_overlap.median())
        ),
        "recovered_factor_fraction_at_association_0.5": float(recovered.float().mean()),
        "global_isolated_input_r2": global_isolated_r2,
        "global_isolated_input_r2_by_factor": global_isolated_r2_by_factor,
        "category": category,
        **pathologies,
        "duplicate_block_fraction": pathologies["split_factor_fraction"],
        "cross_factor_mixing_fraction": pathologies["merged_factor_fraction"],
        "alive_block_fraction": float(alive.float().mean()),
        "matching": {
            "factor_to_group": factor_to_group.tolist(),
            "best_support_association_f1": best_association.tolist(),
            "matched_subspace_overlap": (
                None if matched_overlap is None else matched_overlap.tolist()
            ),
        },
        "code_r2_after_alignment": code_r2_after_alignment,
        "code_r2_after_alignment_by_factor": code_r2_by_factor,
        "selected_group_coverage_by_factor": selected_group_coverage_by_factor,
        "note": (
            "factor-to-group matching and affine code alignment are fit only on "
            "the declared factor-calibration range, then frozen for the complete "
            "development or confirmation range; unselected events retain zero "
            "codes and therefore penalize R2 rather than being conditioned away"
        ),
    }


def _phase1_identification_evidence(
    recovery: Mapping[str, Any],
    threshold_items: Sequence[Sequence[Any]],
) -> dict[str, Any]:
    """Evaluate the preregistered factor-level recovery conjunction."""

    thresholds = {str(item[0]): float(item[1]) for item in threshold_items}
    required = {
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
    if set(thresholds) != required:
        raise CellExecutionError(
            "unknown Phase-1 identification threshold table: "
            + canonical_json(sorted(thresholds))
        )
    matching = recovery.get("matching") or {}
    association = matching.get("best_support_association_f1")
    overlap = matching.get("matched_subspace_overlap")
    isolated = recovery.get("global_isolated_input_r2_by_factor")
    aligned = recovery.get("code_r2_after_alignment_by_factor")
    n_factors = int(recovery.get("n_truth_factors", -1))
    if not all(
        isinstance(values, list) and len(values) == n_factors
        for values in (association, isolated, aligned)
    ):
        raise CellExecutionError("Phase-1 recovery lacks factor-level evidence")
    subspace_eligible = recovery.get("subspace_metrics_eligible") is True
    if subspace_eligible and not (
        isinstance(overlap, list) and len(overlap) == n_factors
    ):
        raise CellExecutionError("eligible Phase-1 recovery lacks subspace evidence")
    per_factor: list[dict[str, Any]] = []
    factor_margins: list[float] = []

    def min_margin(value: Any, threshold: float) -> float:
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            return -1.0e9
        return (float(value) - threshold) / max(abs(threshold), 1.0e-12)

    def max_margin(value: Any, threshold: float) -> float:
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            return -1.0e9
        return (threshold - float(value)) / max(abs(threshold), 1.0e-12)

    for factor in range(n_factors):
        metrics = {
            "support_association": association[factor],
            "subspace_overlap": None if not subspace_eligible else overlap[factor],
            "global_isolated_input_r2": isolated[factor],
            "aligned_code_r2": aligned[factor],
        }
        component_margins = {
            "support_association": min_margin(
                metrics["support_association"],
                thresholds["per_factor.support_association_min"],
            ),
            "global_isolated_input_r2": min_margin(
                metrics["global_isolated_input_r2"],
                thresholds["per_factor.global_isolated_input_r2_min"],
            ),
            "aligned_code_r2": min_margin(
                metrics["aligned_code_r2"],
                thresholds["per_factor.aligned_code_r2_min"],
            ),
        }
        if subspace_eligible:
            component_margins["subspace_overlap"] = min_margin(
                metrics["subspace_overlap"],
                thresholds["per_factor.subspace_overlap_min_when_eligible"],
            )
        factor_margin = min(component_margins.values())
        factor_margins.append(factor_margin)
        per_factor.append(
            {
                "factor": factor,
                **metrics,
                "component_margins": component_margins,
                "margin": factor_margin,
                "identified": factor_margin >= 0.0,
            }
        )

    def nonfinite_count(value: Any) -> int:
        if isinstance(value, bool) or value is None or isinstance(value, str):
            return 0
        if isinstance(value, (int, float)):
            return int(not math.isfinite(float(value)))
        if isinstance(value, (list, tuple)):
            return sum(nonfinite_count(item) for item in value)
        if isinstance(value, dict):
            return sum(nonfinite_count(item) for item in value.values())
        return 1

    recovered_fraction = sum(item["identified"] for item in per_factor) / max(
        1, n_factors
    )
    aggregate = {
        "recovered_factor_fraction": recovered_fraction,
        "support_precision": recovery.get("support_precision"),
        "support_recall": recovery.get("support_recall"),
        # Unused dictionary capacity is descriptive, not a pathology gate.
        # A perfect one-block-per-factor solution is necessarily sparse in an
        # overcomplete dictionary, and capacity rounds deliberately change
        # this fraction.
        "inactive_dictionary_fraction_diagnostic": (
            1.0 - float(recovery["alive_block_fraction"])
        ),
        "duplicate_block_fraction": recovery.get("duplicate_block_fraction"),
        "cross_factor_mixing_fraction": recovery.get("cross_factor_mixing_fraction"),
        "nonfinite_count": nonfinite_count(recovery),
    }
    checks = {
        "recovered_factor_fraction": recovered_fraction
        >= thresholds["aggregate.recovered_factor_fraction_min"],
        "support_precision": isinstance(aggregate["support_precision"], (int, float))
        and float(aggregate["support_precision"])
        >= thresholds["aggregate.support_precision_min"],
        "support_recall": isinstance(aggregate["support_recall"], (int, float))
        and float(aggregate["support_recall"])
        >= thresholds["aggregate.support_recall_min"],
        "duplicate_block_fraction": isinstance(
            aggregate["duplicate_block_fraction"], (int, float)
        )
        and float(aggregate["duplicate_block_fraction"])
        <= thresholds["pathology.duplicate_block_fraction_max"],
        "cross_factor_mixing_fraction": isinstance(
            aggregate["cross_factor_mixing_fraction"], (int, float)
        )
        and float(aggregate["cross_factor_mixing_fraction"])
        <= thresholds["pathology.cross_factor_mixing_fraction_max"],
        "nonfinite_count": aggregate["nonfinite_count"]
        <= thresholds["pathology.nonfinite_count_max"],
    }
    normalized_margins = {
        "worst_eligible_per_factor": min(factor_margins),
        "recovered_factor_fraction": min_margin(
            recovered_fraction,
            thresholds["aggregate.recovered_factor_fraction_min"],
        ),
        "support_precision": min_margin(
            aggregate["support_precision"],
            thresholds["aggregate.support_precision_min"],
        ),
        "support_recall": min_margin(
            aggregate["support_recall"],
            thresholds["aggregate.support_recall_min"],
        ),
        "duplicate_block_fraction": max_margin(
            aggregate["duplicate_block_fraction"],
            thresholds["pathology.duplicate_block_fraction_max"],
        ),
        "cross_factor_mixing_fraction": max_margin(
            aggregate["cross_factor_mixing_fraction"],
            thresholds["pathology.cross_factor_mixing_fraction_max"],
        ),
    }
    margin = min(normalized_margins.values())
    return {
        "thresholds": thresholds,
        "per_factor": per_factor,
        "aggregate": aggregate,
        "checks": checks,
        "normalized_margins": normalized_margins,
        "margin": margin,
        "passed": all(checks.values()),
    }


def _finite_json(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, (list, tuple)):
        return all(_finite_json(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _finite_json(item) for key, item in value.items()
        )
    return False


@torch.no_grad()
def _evaluate_native_selector(
    model: BlockCrosscoder,
    batches: Iterable[torch.Tensor],
    *,
    device: torch.device,
    selection_mode: str = "topk",
) -> dict[str, Any]:
    """Evaluate the method's training-native selector, separately from codec threshold."""

    model = model.to(device).eval()
    materialized_decoder = model.decoder_tensor()
    materialized_encoder = (
        materialized_decoder * model.log_gamma.exp()
        if model.cfg.encoder_mode == "tied"
        else model.encoder_tensor()
    )
    score_geometry = model._frozen_score_geometry(materialized_decoder)
    sites, width = model.cfg.n_sites, model.cfg.d_model
    coordinate_mask = (
        model.coordinate_mask[:, 0, 0].to(device).double()
        if model._has_padded_coordinates
        else None
    )
    error = torch.zeros(sites, dtype=torch.float64, device=device)
    total_sum = torch.zeros(sites, width, dtype=torch.float64, device=device)
    total_square = torch.zeros_like(total_sum)
    support_counts = torch.zeros(
        model.cfg.n_blocks + 1,
        dtype=torch.int64,
        device=device,
    )
    gain_count_names = (
        "candidate_negative",
        "candidate_zero",
        "candidate_positive",
        "selected_negative",
        "selected_zero",
        "selected_positive",
    )
    gain_count_tensor = torch.zeros(
        len(gain_count_names),
        dtype=torch.int64,
        device=device,
    )
    tokens = 0
    for raw in batches:
        x = raw.to(device=device, dtype=torch.float32, non_blocking=True)
        if not x.numel():
            continue
        # Isolated-loss diagnostics explicitly exercise observed-site
        # exclusion. Other scores use the algebraically identical all-site
        # fast path without allocating or multiplying by an all-ones mask.
        observed = (
            torch.ones(x.shape[0], sites, dtype=torch.bool, device=x.device)
            if model.cfg.selection_score == "isolated_loss_decrease"
            else None
        )
        out = model.forward_with_materialized(
            x,
            mode=selection_mode,
            observed=observed,
            _decoder=materialized_decoder,
            _encoder=materialized_encoder,
            _score_geometry=score_geometry,
        )[0]
        residual = (x - out.xhat).double()
        if coordinate_mask is not None:
            residual = residual * coordinate_mask
        error += residual.square().sum(dim=(0, 2))
        masked = x.double()
        if coordinate_mask is not None:
            masked = masked * coordinate_mask
        total_sum += masked.sum(dim=0)
        total_square += masked.square().sum(dim=0)
        counts = out.mask.sum(dim=1)
        support_counts += torch.bincount(
            counts,
            minlength=model.cfg.n_blocks + 1,
        )
        if model.cfg.selection_score == "isolated_loss_decrease":
            scores = out.scores
            negative = scores < 0
            zero = scores == 0
            positive = scores > 0
            gain_count_tensor += torch.stack(
                (
                    negative.sum(),
                    zero.sum(),
                    positive.sum(),
                    (negative & out.mask).sum(),
                    (zero & out.mask).sum(),
                    (positive & out.mask).sum(),
                )
            )
        tokens += len(x)
    if tokens == 0:
        raise CellExecutionError("native-selector evaluation stream is empty")
    centered = total_square - total_sum.square() / tokens
    denominator = centered.sum(dim=1).clamp_min(1e-30)
    fvu = error / denominator
    support_histogram = {
        count: frequency
        for count, frequency in enumerate(support_counts.cpu().tolist())
        if frequency
    }
    gain_counts = dict(
        zip(gain_count_names, gain_count_tensor.cpu().tolist(), strict=True)
    )
    event_total = sum(
        count * frequency for count, frequency in support_histogram.items()
    )
    if model.cfg.selection_score == "isolated_loss_decrease":
        candidate_total = sum(
            gain_counts[f"candidate_{sign}"]
            for sign in ("negative", "zero", "positive")
        )
        selected_total = sum(
            gain_counts[f"selected_{sign}"] for sign in ("negative", "zero", "positive")
        )
        if candidate_total != tokens * model.cfg.n_blocks:
            raise CellExecutionError(
                "isolated-loss candidate diagnostic count does not cover every block"
            )
        if selected_total != event_total:
            raise CellExecutionError(
                "isolated-loss selected diagnostic count differs from selector support"
            )
        isolated_loss_diagnostics: dict[str, Any] = {
            "schema": "bsc-isolated-loss-gain-diagnostics-v1",
            "applicable": True,
            "observation_contract": "explicit_true_observed_sites_only_v1",
            "candidate_event_count": candidate_total,
            "candidate_negative_gain_count": gain_counts["candidate_negative"],
            "candidate_zero_gain_count": gain_counts["candidate_zero"],
            "candidate_positive_gain_count": gain_counts["candidate_positive"],
            "candidate_negative_gain_fraction": (
                gain_counts["candidate_negative"] / candidate_total
            ),
            "candidate_zero_gain_fraction": (
                gain_counts["candidate_zero"] / candidate_total
            ),
            "candidate_positive_gain_fraction": (
                gain_counts["candidate_positive"] / candidate_total
            ),
            "selected_event_count": selected_total,
            "selected_negative_gain_count": gain_counts["selected_negative"],
            "selected_zero_gain_count": gain_counts["selected_zero"],
            "selected_positive_gain_count": gain_counts["selected_positive"],
            "selected_negative_gain_fraction": (
                None
                if selected_total == 0
                else gain_counts["selected_negative"] / selected_total
            ),
            "selected_zero_gain_fraction": (
                None
                if selected_total == 0
                else gain_counts["selected_zero"] / selected_total
            ),
            "selected_positive_gain_fraction": (
                None
                if selected_total == 0
                else gain_counts["selected_positive"] / selected_total
            ),
        }
    else:
        isolated_loss_diagnostics = {
            "applicable": False,
            "reason": "selection_score_not_isolated_loss_decrease",
        }
    return {
        "selector": model.cfg.selection,
        "selection_score": model.cfg.selection_score,
        "mode": selection_mode,
        "n_tokens": tokens,
        "fvu_per_site": fvu.cpu().tolist(),
        "fvu_pooled": float(error.sum() / denominator.sum()),
        "avg_active_blocks": event_total / tokens,
        "active_block_count_histogram": {
            str(key): support_histogram[key] for key in sorted(support_histogram)
        },
        "isolated_loss_gain_diagnostics": isolated_loss_diagnostics,
    }


@torch.no_grad()
def _apply_saved_real_normalization(
    x: torch.Tensor,
    normalization: Mapping[str, Any],
    *,
    mean: torch.Tensor | None = None,
    operator: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply only normalization bytes carried by the deployable artifact."""

    kind = normalization["kind"]
    if kind == "identity":
        return x.float()
    if kind != "frozen_transform":
        raise CellExecutionError("real codec has a non-real normalization payload")
    mode = str(normalization["mode"])
    if mode == "layer":
        result = torch.zeros_like(x, dtype=torch.float32)
        eps = float(normalization.get("meta", {}).get("layer_norm_eps", 1e-5))
        for site, dim in enumerate(normalization["site_dims"]):
            result[:, site, :dim] = torch.nn.functional.layer_norm(
                x[:, site, :dim].float(), (int(dim),), eps=eps
            )
        return result
    if mean is None:
        mean = normalization["mean"].to(x.device, dtype=torch.float32)
    centered = x.float() - mean
    if operator is None:
        W = normalization["W"].to(x.device, dtype=torch.float32)
        operator = (
            torch.diagonal(W, dim1=-2, dim2=-1)
            if mode in {"none", "scalar_rms", "sqrt_d"}
            else W
        )
    if mode in {"none", "scalar_rms", "sqrt_d"}:
        return centered * operator.unsqueeze(0)
    return torch.einsum("sde,nse->nsd", operator, centered)


def _invert_saved_real_normalization(
    normalized: torch.Tensor,
    raw_source: torch.Tensor,
    normalization: Mapping[str, Any],
    *,
    inverse_W: torch.Tensor | None,
    mean: torch.Tensor | None = None,
    diagonal: torch.Tensor | None = None,
) -> torch.Tensor:
    """Invert the serialized transform; raw_source is used only by LayerNorm."""

    kind = normalization["kind"]
    if kind == "identity":
        return normalized.float()
    if kind != "frozen_transform":
        raise CellExecutionError("real codec has a non-real normalization payload")
    mode = str(normalization["mode"])
    if mode == "layer":
        result = torch.zeros_like(normalized, dtype=torch.float32)
        eps = float(normalization.get("meta", {}).get("layer_norm_eps", 1e-5))
        for site, dim in enumerate(normalization["site_dims"]):
            dim = int(dim)
            values = raw_source[:, site, :dim].float()
            token_mean = values.mean(dim=-1, keepdim=True)
            variance = values.var(dim=-1, correction=0, keepdim=True)
            result[:, site, :dim] = (
                normalized[:, site, :dim].float() * (variance + eps).sqrt() + token_mean
            )
        return result
    if mean is None:
        mean = normalization["mean"].to(normalized.device, dtype=torch.float32)
    if mode in {"none", "scalar_rms", "sqrt_d"}:
        if diagonal is None:
            W = normalization["W"].to(normalized.device, dtype=torch.float32)
            diagonal = torch.diagonal(W, dim1=-2, dim2=-1)
        return normalized.float() / diagonal.clamp_min(1e-30).unsqueeze(0) + mean
    if inverse_W is None:
        raise CellExecutionError("dense frozen transform inverse was not prepared")
    return torch.einsum("sde,nse->nsd", inverse_W, normalized.float()) + mean


def _time_sharing_plan_key(
    *,
    budget: float,
    lower_name: str,
    upper_name: str,
    upper_tokens: int,
    horizon_tokens: int,
) -> str:
    """Stable lookup key for a fully specified operational mixture."""

    return hashlib.sha256(
        canonical_json(
            {
                "budget_bits_per_token": float(budget),
                "lower_name": lower_name,
                "upper_name": upper_name,
                "upper_tokens": int(upper_tokens),
                "horizon_tokens": int(horizon_tokens),
            }
        ).encode("utf-8")
    ).hexdigest()


def _time_sharing_mode_code(name: str) -> int:
    if name == "zero_event_calibration_mean":
        return 0
    if name.startswith("q") and name[1:].isdigit():
        value = int(name[1:])
        if 0 < value < 2**32:
            return value
    raise CellExecutionError(f"invalid time-sharing endpoint name {name!r}")


def _time_sharing_header_binding(
    *,
    cell_id: str,
    deployment_codec_sha256: str,
    schedule_contract: str,
    budget: float,
    lower_name: str,
    upper_name: str,
    upper_tokens: int,
    horizon_tokens: int,
) -> dict[str, Any]:
    return {
        "schema": "bsc-deployment-schedule-header-v1",
        "cell_id": cell_id,
        "deployment_codec_sha256": deployment_codec_sha256,
        "schedule_contract": schedule_contract,
        "budget_bits_per_token": float(budget),
        "lower_name": lower_name,
        "upper_name": upper_name,
        "upper_tokens": int(upper_tokens),
        "horizon_tokens": int(horizon_tokens),
    }


def _time_sharing_binding_magic(binding: Mapping[str, Any]) -> int:
    return int.from_bytes(
        hashlib.sha256(canonical_json(dict(binding)).encode("utf-8")).digest()[:8],
        byteorder="big",
        signed=False,
    )


def _pack_time_sharing_header(
    *,
    cell_id: str,
    deployment_codec_sha256: str,
    schedule_contract: str,
    budget: float,
    lower_name: str,
    upper_name: str,
    upper_tokens: int,
    horizon_tokens: int,
) -> bytes:
    if schedule_contract != "balanced_global_token_counter_u64_v1":
        raise CellExecutionError("cannot serialize an unsupported schedule contract")
    if horizon_tokens <= 0 or upper_tokens <= 0 or upper_tokens >= horizon_tokens:
        raise CellExecutionError("time-sharing header has invalid token counts")
    lower_mode = _time_sharing_mode_code(lower_name)
    upper_mode = _time_sharing_mode_code(upper_name)
    if lower_mode == upper_mode:
        raise CellExecutionError("time-sharing header endpoints must differ")
    binding = _time_sharing_header_binding(
        cell_id=cell_id,
        deployment_codec_sha256=deployment_codec_sha256,
        schedule_contract=schedule_contract,
        budget=budget,
        lower_name=lower_name,
        upper_name=upper_name,
        upper_tokens=upper_tokens,
        horizon_tokens=horizon_tokens,
    )
    header = struct.pack(
        TIME_SHARING_HEADER_LAYOUT,
        _time_sharing_binding_magic(binding),
        horizon_tokens,
        upper_tokens,
        lower_mode,
        upper_mode,
    )
    if len(header) != TIME_SHARING_HEADER_BYTES:
        raise AssertionError("time-sharing header layout is not exactly 32 bytes")
    return header


def _unpack_time_sharing_header(
    header: bytes,
    *,
    cell_id: str,
    deployment_codec_sha256: str,
    schedule_contract: str,
    budget: float,
    lower_name: str,
    upper_name: str,
    upper_tokens: int,
    horizon_tokens: int,
) -> dict[str, int]:
    if len(header) != TIME_SHARING_HEADER_BYTES:
        raise CellExecutionError("deployment schedule record is not exactly 32 bytes")
    try:
        magic, horizon, upper_count, lower_mode, upper_mode = struct.unpack(
            TIME_SHARING_HEADER_LAYOUT, header
        )
    except struct.error as exc:
        raise CellExecutionError(
            f"cannot decode deployment schedule header: {exc}"
        ) from exc
    binding = _time_sharing_header_binding(
        cell_id=cell_id,
        deployment_codec_sha256=deployment_codec_sha256,
        schedule_contract=schedule_contract,
        budget=budget,
        lower_name=lower_name,
        upper_name=upper_name,
        upper_tokens=upper_tokens,
        horizon_tokens=horizon_tokens,
    )
    expected = {
        "binding_magic_u64": _time_sharing_binding_magic(binding),
        "horizon_tokens": int(horizon_tokens),
        "upper_tokens": int(upper_tokens),
        "lower_mode": _time_sharing_mode_code(lower_name),
        "upper_mode": _time_sharing_mode_code(upper_name),
    }
    observed = {
        "binding_magic_u64": magic,
        "horizon_tokens": horizon,
        "upper_tokens": upper_count,
        "lower_mode": lower_mode,
        "upper_mode": upper_mode,
    }
    if observed != expected:
        raise CellExecutionError(
            "deployment schedule header binding mismatch: "
            + canonical_json(
                {
                    name: {"expected": expected[name], "observed": observed[name]}
                    for name in expected
                    if observed[name] != expected[name]
                }
            )
        )
    return observed


def _write_deployment_schedule_bundle(
    path: Path,
    *,
    cell_id: str,
    deployment_codec_sha256: str,
    schedule_contract: str,
    plans: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Persist exact alternative 32-byte deployment headers in one artifact."""

    records: list[dict[str, Any]] = []
    body = bytearray()
    seen_budgets: set[float] = set()
    for schedule_key, raw_plan in sorted(
        plans.items(),
        key=lambda item: (float(item[1]["budget_bits_per_token"]), item[0]),
    ):
        plan = dict(raw_plan)
        budget = float(plan["budget_bits_per_token"])
        if not math.isfinite(budget):
            raise CellExecutionError("deployment schedule budget must be finite")
        if budget in seen_budgets:
            raise CellExecutionError(
                "deployment schedule bundle has more than one record for a budget"
            )
        seen_budgets.add(budget)
        expected_key = _time_sharing_plan_key(
            budget=budget,
            lower_name=str(plan["lower_name"]),
            upper_name=str(plan["upper_name"]),
            upper_tokens=int(plan["upper_tokens"]),
            horizon_tokens=int(plan["horizon_tokens"]),
        )
        if schedule_key != expected_key:
            raise CellExecutionError("deployment schedule key/content mismatch")
        header = _pack_time_sharing_header(
            cell_id=cell_id,
            deployment_codec_sha256=deployment_codec_sha256,
            schedule_contract=schedule_contract,
            budget=budget,
            lower_name=str(plan["lower_name"]),
            upper_name=str(plan["upper_name"]),
            upper_tokens=int(plan["upper_tokens"]),
            horizon_tokens=int(plan["horizon_tokens"]),
        )
        offset = len(body)
        body.extend(header)
        records.append(
            {
                "schedule_key": schedule_key,
                "budget_bits_per_token": budget,
                "lower_name": str(plan["lower_name"]),
                "lower_q": plan.get("lower_q"),
                "upper_name": str(plan["upper_name"]),
                "upper_q": plan.get("upper_q"),
                "upper_tokens": int(plan["upper_tokens"]),
                "horizon_tokens": int(plan["horizon_tokens"]),
                "upper_mixture_weight": float(plan["upper_mixture_weight"]),
                "achieved_total_bits_per_token": float(
                    plan["achieved_total_bits_per_token"]
                ),
                "offset_bytes": offset,
                "size_bytes": len(header),
                "sha256": hashlib.sha256(header).hexdigest(),
                "binding_magic_u64": struct.unpack(TIME_SHARING_HEADER_LAYOUT, header)[
                    0
                ],
            }
        )
    frozen_body = bytes(body)
    _write_immutable_bytes(path, frozen_body)
    return {
        "schema": TIME_SHARING_BUNDLE_SCHEMA,
        "cell_id": cell_id,
        "deployment_codec_sha256": deployment_codec_sha256,
        "schedule_contract": schedule_contract,
        "record_size_bytes": TIME_SHARING_HEADER_BYTES,
        "header_layout": TIME_SHARING_HEADER_LAYOUT_DESCRIPTION,
        "record_count": len(records),
        "artifact_size_bytes": len(frozen_body),
        "artifact_sha256": hashlib.sha256(frozen_body).hexdigest(),
        "records": records,
    }


def _load_deployment_schedule_bundle(
    path: Path,
    manifest: Mapping[str, Any],
    *,
    cell_id: str,
    deployment_codec_sha256: str,
    schedule_contract: str,
) -> dict[str, dict[str, Any]]:
    expected_keys = {
        "schema",
        "cell_id",
        "deployment_codec_sha256",
        "schedule_contract",
        "record_size_bytes",
        "header_layout",
        "record_count",
        "artifact_size_bytes",
        "artifact_sha256",
        "records",
    }
    if set(manifest) != expected_keys or (
        manifest.get("schema") != TIME_SHARING_BUNDLE_SCHEMA
        or manifest.get("cell_id") != cell_id
        or manifest.get("deployment_codec_sha256") != deployment_codec_sha256
        or manifest.get("schedule_contract") != schedule_contract
        or manifest.get("record_size_bytes") != TIME_SHARING_HEADER_BYTES
        or manifest.get("header_layout") != TIME_SHARING_HEADER_LAYOUT_DESCRIPTION
    ):
        raise CellExecutionError("deployment schedule bundle manifest binding mismatch")
    try:
        body = path.read_bytes()
    except OSError as exc:
        raise CellExecutionError(
            f"cannot read deployment schedule bundle: {exc}"
        ) from exc
    if (
        manifest.get("artifact_size_bytes") != len(body)
        or manifest.get("artifact_sha256") != hashlib.sha256(body).hexdigest()
        or manifest.get("record_count") != len(manifest.get("records", ()))
        or len(body)
        != int(manifest.get("record_count", -1)) * TIME_SHARING_HEADER_BYTES
    ):
        raise CellExecutionError("deployment schedule bundle bytes/manifest mismatch")
    records = manifest.get("records")
    if not isinstance(records, list):
        raise CellExecutionError("deployment schedule bundle records must be a list")
    plans: dict[str, dict[str, Any]] = {}
    seen_budgets: set[float] = set()
    expected_record_keys = {
        "schedule_key",
        "budget_bits_per_token",
        "lower_name",
        "lower_q",
        "upper_name",
        "upper_q",
        "upper_tokens",
        "horizon_tokens",
        "upper_mixture_weight",
        "achieved_total_bits_per_token",
        "offset_bytes",
        "size_bytes",
        "sha256",
        "binding_magic_u64",
    }
    for index, record in enumerate(records):
        if not isinstance(record, Mapping) or set(record) != expected_record_keys:
            raise CellExecutionError("deployment schedule record must be an object")
        offset = index * TIME_SHARING_HEADER_BYTES
        if (
            record.get("offset_bytes") != offset
            or record.get("size_bytes") != TIME_SHARING_HEADER_BYTES
        ):
            raise CellExecutionError("deployment schedule record offset/size mismatch")
        header = body[offset : offset + TIME_SHARING_HEADER_BYTES]
        if record.get("sha256") != hashlib.sha256(header).hexdigest():
            raise CellExecutionError("deployment schedule record hash mismatch")
        budget = float(record["budget_bits_per_token"])
        if not math.isfinite(budget):
            raise CellExecutionError("deployment schedule record budget is nonfinite")
        if budget in seen_budgets:
            raise CellExecutionError("deployment schedule bundle repeats a budget")
        seen_budgets.add(budget)
        decoded = _unpack_time_sharing_header(
            header,
            cell_id=cell_id,
            deployment_codec_sha256=deployment_codec_sha256,
            schedule_contract=schedule_contract,
            budget=budget,
            lower_name=str(record["lower_name"]),
            upper_name=str(record["upper_name"]),
            upper_tokens=int(record["upper_tokens"]),
            horizon_tokens=int(record["horizon_tokens"]),
        )
        if record.get("binding_magic_u64") != decoded["binding_magic_u64"]:
            raise CellExecutionError("deployment schedule record magic mismatch")
        expected_lower_q = (
            None
            if str(record["lower_name"]) == "zero_event_calibration_mean"
            else _time_sharing_mode_code(str(record["lower_name"]))
        )
        expected_upper_q = (
            None
            if str(record["upper_name"]) == "zero_event_calibration_mean"
            else _time_sharing_mode_code(str(record["upper_name"]))
        )
        if (
            record.get("lower_q") != expected_lower_q
            or record.get("upper_q") != expected_upper_q
        ):
            raise CellExecutionError("deployment schedule endpoint/q binding mismatch")
        schedule_key = str(record["schedule_key"])
        expected_key = _time_sharing_plan_key(
            budget=budget,
            lower_name=str(record["lower_name"]),
            upper_name=str(record["upper_name"]),
            upper_tokens=int(record["upper_tokens"]),
            horizon_tokens=int(record["horizon_tokens"]),
        )
        if schedule_key != expected_key or schedule_key in plans:
            raise CellExecutionError("deployment schedule record key mismatch")
        plans[schedule_key] = {
            "budget_bits_per_token": budget,
            "lower_name": str(record["lower_name"]),
            "lower_q": record.get("lower_q"),
            "upper_name": str(record["upper_name"]),
            "upper_q": record.get("upper_q"),
            "upper_tokens": int(record["upper_tokens"]),
            "horizon_tokens": int(record["horizon_tokens"]),
            "upper_mixture_weight": float(record["upper_mixture_weight"]),
            "achieved_total_bits_per_token": float(
                record["achieved_total_bits_per_token"]
            ),
            "deployment_schedule": {
                "artifact_sha256": manifest["artifact_sha256"],
                "record_index": index,
                "offset_bytes": offset,
                "size_bytes": TIME_SHARING_HEADER_BYTES,
                "record_sha256": record["sha256"],
                "binding_magic_u64": decoded["binding_magic_u64"],
            },
        }
    return plans


@torch.no_grad()
def _evaluate_raw_space(
    ctx: _Context,
    preparation: Mapping[str, Any],
    model: BlockCrosscoder,
    codec: Codec,
    deployment: Mapping[str, Any],
    *,
    time_sharing_plans: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Evaluate the serialized quantized codec in paired source coordinates."""

    data = preparation["data"]
    batch_size = int(ctx.values["optimizer.batch_tokens"])
    device = _device(ctx)
    model = model.to(device).eval()
    materialized_decoder = model.decoder_tensor()
    materialized_encoder = (
        materialized_decoder * model.log_gamma.exp()
        if model.cfg.encoder_mode == "tied"
        else model.encoder_tensor()
    )
    materialized_score_geometry = model._frozen_score_geometry(
        materialized_decoder
    )
    materialized_decoder_matrix = materialized_decoder.permute(1, 2, 0, 3).reshape(
        model.cfg.n_latents,
        model.cfg.n_sites * model.cfg.d_model,
    )
    sites, width = model.cfg.n_sites, model.cfg.d_model
    coordinate_mask = (
        model.coordinate_mask[:, 0, 0].to(device).double()
        if model._has_padded_coordinates
        else None
    )
    errors = {
        q: torch.zeros(sites, dtype=torch.float64, device=device) for q in codec.spec.qs
    }
    time_sharing_plans = {
        str(key): dict(plan) for key, plan in (time_sharing_plans or {}).items()
    }
    time_sharing_errors = {
        key: torch.zeros((), dtype=torch.float64, device=device)
        for key in time_sharing_plans
    }
    time_sharing_evaluation_upper_tokens = {key: 0 for key in time_sharing_plans}

    saved_normalization = deployment["normalization"]
    inverse_W: torch.Tensor | None = None
    saved_mean_device: torch.Tensor | None = None
    saved_diagonal_device: torch.Tensor | None = None
    saved_mean_cpu: torch.Tensor | None = None
    saved_operator_cpu: torch.Tensor | None = None
    synthetic_normalization: Mapping[str, Any] | None = None
    synthetic_mean_device: torch.Tensor | None = None
    synthetic_scale_device: torch.Tensor | None = None
    oracle_layer_inverse = False
    serialized_forward_verified = data["kind"] == "synthetic"
    persisted_view_max_abs_difference = 0.0
    if data["kind"] == "synthetic":
        synthetic_normalization = data["normalization"]
        if (
            saved_normalization.get("kind") != "synthetic"
            or saved_normalization.get("record") != synthetic_normalization
        ):
            raise CellExecutionError(
                "deployable synthetic normalization differs from preparation"
            )
        mode = str(synthetic_normalization["mode"])
        oracle_layer_inverse = synthetic_normalization["kind"] == "token_layer_norm"
        if not oracle_layer_inverse:
            synthetic_mean_device = torch.tensor(
                synthetic_normalization["mean"],
                device=device,
                dtype=torch.float32,
            )
            synthetic_scale_device = torch.tensor(
                synthetic_normalization["scale"],
                device=device,
                dtype=torch.float32,
            ).view(1, -1, 1)
        evaluation_dataset = _synthetic_dataset(
            ctx.cell, str(data["evaluation_stream"])
        )
        evaluation_start, evaluation_stop = data["ranges"]["evaluation"]

        def paired_stream() -> Iterator[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ]:
            for item in evaluation_dataset.batches(
                batch_size,
                start=int(evaluation_start),
                stop=int(evaluation_stop),
            ):
                if not bool(item.observed.all()):
                    raise CellExecutionError(
                        "codec evaluation for missing-site synthetic rows is not "
                        "implemented; refusing to fabricate observations"
                    )
                yield (
                    _apply_normalization(item.x, synthetic_normalization),
                    item.x.float(),
                    item.presentation_ids.view(-1, 1),
                )

    else:
        normalized = _store_reader(preparation, "evaluation")
        raw = _store_reader(preparation, "evaluation", raw=True)
        on_the_fly = data.get("normalization", {}).get("application") == "on_the_fly"
        saved_mode = str(saved_normalization.get("mode", "none"))
        oracle_layer_inverse = (
            saved_normalization.get("kind") == "frozen_transform"
            and saved_mode == "layer"
        )
        if (
            saved_normalization.get("kind") == "frozen_transform"
        ):
            saved_mean_cpu = saved_normalization["mean"].to(dtype=torch.float32)
            saved_mean_device = saved_mean_cpu.to(device=device)
            saved_W_cpu = saved_normalization["W"].to(dtype=torch.float32)
            if saved_mode in {"none", "scalar_rms", "sqrt_d"}:
                saved_operator_cpu = torch.diagonal(
                    saved_W_cpu, dim1=-2, dim2=-1
                )
                saved_diagonal_device = saved_operator_cpu.to(device=device)
            elif saved_mode == "whiten":
                saved_operator_cpu = saved_W_cpu
                inverse_W = (
                    torch.linalg.inv(saved_W_cpu.double()).float().to(device)
                )
            elif saved_mode != "layer":
                saved_operator_cpu = saved_W_cpu

        def paired_stream() -> Iterator[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ]:
            nonlocal serialized_forward_verified
            nonlocal persisted_view_max_abs_difference
            normalized_stream = normalized.sequential_batches_with_ids(batch_size)
            raw_stream = raw.sequential_batches_with_ids(batch_size)
            sentinel = object()
            for normalized_item, raw_item in itertools.zip_longest(
                normalized_stream, raw_stream, fillvalue=sentinel
            ):
                if normalized_item is sentinel or raw_item is sentinel:
                    raise CellExecutionError(
                        "normalized/raw evaluation streams have different lengths"
                    )
                x_normalized, normalized_ids = normalized_item
                x_raw, raw_ids = raw_item
                if not torch.equal(normalized_ids, raw_ids):
                    raise CellExecutionError(
                        "normalized/raw evaluation row identities diverged"
                    )
                # The evaluated encoder input is always reconstructed from the
                # transform bytes inside the priced deployment artifact.  A
                # persisted Phase-2 view remains useful as an independent
                # materialization check, but is never a shortcut around the
                # actual consumer preprocessing path.
                if on_the_fly and not torch.equal(x_normalized, x_raw):
                    raise CellExecutionError(
                        "single-view evaluation readers do not expose identical "
                        "raw activation bytes"
                    )
                encoder_input = _apply_saved_real_normalization(
                    x_raw,
                    saved_normalization,
                    mean=saved_mean_cpu,
                    operator=saved_operator_cpu,
                )
                if not on_the_fly:
                    difference = float(
                        (encoder_input - x_normalized.float()).abs().max()
                    )
                    persisted_view_max_abs_difference = max(
                        persisted_view_max_abs_difference, difference
                    )
                    if not torch.allclose(
                        encoder_input,
                        x_normalized.float(),
                        rtol=0.012,
                        atol=0.012,
                    ):
                        raise CellExecutionError(
                            "serialized deployment normalization does not reproduce "
                            "the bound persisted evaluation view"
                        )
                serialized_forward_verified = True
                yield (encoder_input, x_raw.float(), raw_ids)

    calibration_tokens = int(deployment["raw_calibration_mean_fit_tokens"])
    calibration_mean = deployment["raw_calibration_mean"].to(
        device=device, dtype=torch.float64
    )
    if calibration_tokens <= 0 or calibration_mean.shape != (sites, width):
        raise CellExecutionError("deployable codec has invalid raw zero-rate mean")
    denominator = torch.zeros(sites, dtype=torch.float64, device=device)
    row_sequences: list[int] = []
    row_denominators: list[float] = []
    row_errors: dict[int, list[float]] = {q: [] for q in codec.spec.qs}
    tokens = 0
    evaluation_stream: Iterator = paired_stream()
    if device.type == "cuda":
        evaluation_stream = prefetch_batches(
            evaluation_stream,
            depth=2,
            pin_memory=True,
        )
    try:
        evaluation_iterator = iter(evaluation_stream)
        for x_normalized, x_raw, row_ids in evaluation_iterator:
            x_normalized = x_normalized.to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            x_raw = x_raw.to(device=device, dtype=torch.float32, non_blocking=True)
            row_ids = row_ids.to(device=device, non_blocking=True)
            centered = x_raw.double() - calibration_mean
            if coordinate_mask is not None:
                centered = centered * coordinate_mask
            token_denominator = centered.square().sum(dim=2)
            denominator += token_denominator.sum(dim=0)
            threshold_output, packet_events = _encode_batch_events(
                model,
                codec,
                x_normalized,
                _decoder=materialized_decoder,
                _encoder=materialized_encoder,
                _score_geometry=materialized_score_geometry,
            )
            del threshold_output
            token_errors: dict[int, torch.Tensor] = {}
            for decoded_chunk in _decode_trusted_packet_events_q_chunks(
                model,
                codec,
                packet_events,
                qs=codec.spec.qs,
                _decoder=materialized_decoder,
                _decoder_matrix=materialized_decoder_matrix,
            ):
                for q, normalized_prediction in decoded_chunk.items():
                    if data["kind"] == "synthetic":
                        assert synthetic_normalization is not None
                        if synthetic_normalization["kind"] == "token_layer_norm":
                            raw_prediction = torch.zeros_like(normalized_prediction)
                            for site, dim in enumerate(model.cfg.site_dims):
                                values = x_raw[:, site, :dim]
                                mean = values.mean(dim=-1, keepdim=True)
                                variance = values.var(
                                    dim=-1, correction=0, keepdim=True
                                )
                                raw_prediction[:, site, :dim] = (
                                    normalized_prediction[:, site, :dim]
                                    * (variance + 1e-5).sqrt()
                                    + mean
                                )
                        else:
                            assert synthetic_mean_device is not None
                            assert synthetic_scale_device is not None
                            raw_prediction = (
                                normalized_prediction / synthetic_scale_device
                                + synthetic_mean_device.unsqueeze(0)
                            )
                    else:
                        raw_prediction = _invert_saved_real_normalization(
                            normalized_prediction,
                            x_raw,
                            saved_normalization,
                            inverse_W=inverse_W,
                            mean=saved_mean_device,
                            diagonal=saved_diagonal_device,
                        )
                    residual = (x_raw - raw_prediction).double()
                    if coordinate_mask is not None:
                        residual = residual * coordinate_mask
                    token_error = residual.square().sum(dim=2)
                    token_errors[q] = token_error
                    errors[q] += token_error.sum(dim=0)
                del normalized_prediction, raw_prediction, residual, token_error
                del decoded_chunk
            if time_sharing_plans:
                horizon = next(iter(time_sharing_plans.values()))["horizon_tokens"]
                if tokens + len(x_raw) > int(horizon):
                    raise CellExecutionError(
                        "raw evaluation exceeds the declared time-sharing horizon"
                    )
                token_indices = torch.arange(
                    tokens,
                    tokens + len(x_raw),
                    device=device,
                    dtype=torch.int64,
                )
                endpoint_errors: dict[str, torch.Tensor] = {
                    "zero_event_calibration_mean": token_denominator.sum(dim=1)
                }
                endpoint_errors.update(
                    {f"q{q}": token_errors[q].sum(dim=1) for q in codec.spec.qs}
                )
                masks: dict[int, torch.Tensor] = {}
                for key, plan in time_sharing_plans.items():
                    upper_tokens = int(plan["upper_tokens"])
                    mask = masks.get(upper_tokens)
                    if mask is None:
                        mask = ((token_indices + 1) * upper_tokens) // int(horizon) > (
                            token_indices * upper_tokens
                        ) // int(horizon)
                        masks[upper_tokens] = mask
                    time_sharing_errors[key] += torch.where(
                        mask,
                        endpoint_errors[str(plan["upper_name"])],
                        endpoint_errors[str(plan["lower_name"])],
                    ).sum()
                    time_sharing_evaluation_upper_tokens[key] += int(mask.sum())

            sequences = row_ids[:, 0].to(dtype=torch.int64)
            unique_sequences, inverse = torch.unique_consecutive(
                sequences, return_inverse=True
            )
            grouped_denominator = torch.zeros(
                len(unique_sequences), sites, dtype=torch.float64, device=device
            )
            grouped_denominator.index_add_(0, inverse, token_denominator)
            grouped_errors: dict[int, torch.Tensor] = {}
            for q in codec.spec.qs:
                grouped = torch.zeros_like(grouped_denominator)
                grouped.index_add_(0, inverse, token_errors[q])
                grouped_errors[q] = grouped
            sequence_values = unique_sequences.cpu().tolist()
            denominator_values = grouped_denominator.sum(dim=1).cpu().tolist()
            error_values = {
                q: grouped_errors[q].sum(dim=1).cpu().tolist()
                for q in codec.spec.qs
            }
            for group, sequence in enumerate(sequence_values):
                den_value = denominator_values[group]
                err_values = {q: error_values[q][group] for q in codec.spec.qs}
                if row_sequences and row_sequences[-1] == int(sequence):
                    row_denominators[-1] += den_value
                    for q in codec.spec.qs:
                        row_errors[q][-1] += err_values[q]
                else:
                    row_sequences.append(int(sequence))
                    row_denominators.append(den_value)
                    for q in codec.spec.qs:
                        row_errors[q].append(err_values[q])
            tokens += len(x_raw)
    finally:
        close_evaluation = getattr(evaluation_stream, "close", None)
        if close_evaluation is not None:
            close_evaluation()
    if tokens == 0:
        raise CellExecutionError("raw-space evaluation stream is empty")
    denominator = denominator.clamp_min(1e-30)
    row_denominator_tensor = torch.tensor(row_denominators, dtype=torch.float64)

    def bootstrap_ci(numerator_values: Sequence[float], *, seed: int) -> list[float]:
        numerator = torch.tensor(numerator_values, dtype=torch.float64)
        n_rows = len(numerator)
        if n_rows == 0:
            raise CellExecutionError("raw-space sequence bootstrap has no rows")
        generator = torch.Generator().manual_seed(seed)
        ratios: list[torch.Tensor] = []
        remaining = codec.spec.n_bootstrap
        # Exact ordinary sequence bootstrap, chunked across replicates so a
        # Phase-3 panel never materializes [1000, n_sequences] indices.
        while remaining:
            replicates = min(8, remaining)
            indices = torch.randint(
                0,
                n_rows,
                (replicates, n_rows),
                generator=generator,
            )
            ratios.append(
                numerator[indices].sum(dim=1)
                / row_denominator_tensor[indices].sum(dim=1).clamp_min(1e-30)
            )
            remaining -= replicates
        values = torch.cat(ratios)
        bounds = torch.quantile(
            values,
            torch.tensor([0.025, 0.975], dtype=values.dtype),
        )
        return [float(bounds[0]), float(bounds[1])]

    points = {}
    for q in codec.spec.qs:
        points[str(q)] = {
            "fvu_per_site": (errors[q] / denominator).cpu().tolist(),
            "fvu_pooled": float(errors[q].sum() / denominator.sum()),
            "fvu_pooled_ci95": bootstrap_ci(
                row_errors[q],
                seed=codec.spec.bootstrap_seed + q,
            ),
        }
    mode = (
        str(synthetic_normalization["mode"])
        if synthetic_normalization is not None
        else str(saved_normalization.get("mode", "none"))
    )
    pooled_denominator = denominator.sum()
    operational_time_sharing = {
        key: {
            **plan,
            "evaluation_tokens": tokens,
            "evaluation_upper_tokens": time_sharing_evaluation_upper_tokens[key],
            "raw_space_fvu": float(time_sharing_errors[key] / pooled_denominator),
            "distortion_measurement": (
                "executed_balanced_schedule_on_paired_raw_evaluation_rows"
            ),
        }
        for key, plan in time_sharing_plans.items()
    }
    return {
        "eligible": not oracle_layer_inverse,
        "mode": mode,
        "n_tokens": tokens,
        "n_sequences": len(row_sequences),
        "calibration_mean_fit_tokens": calibration_tokens,
        "codec_quantized": True,
        "source_free_sparse_decode": True,
        "serialized_forward_preprocessing_validated": serialized_forward_verified,
        "persisted_view_max_abs_difference": (
            persisted_view_max_abs_difference
            if data["kind"] != "synthetic" and not on_the_fly
            else None
        ),
        "points": points,
        "operational_time_sharing": operational_time_sharing,
        "oracle_side_information": oracle_layer_inverse,
        "reason": (
            "token LayerNorm inverse uses unpriced source-token mean/variance"
            if oracle_layer_inverse
            else "paired row-identical raw view and invertible frozen transform"
        ),
    }


def _lower_convex_rate_envelope(
    points: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return the operational lower convex envelope in increasing rate."""

    best_at_rate: dict[float, dict[str, Any]] = {}
    for raw in points:
        rate = float(raw["total_bits_per_token"])
        fvu = float(raw["raw_space_fvu"])
        candidate = dict(raw)
        current = best_at_rate.get(rate)
        if current is None or fvu < float(current["raw_space_fvu"]):
            best_at_rate[rate] = candidate
    ordered = [best_at_rate[rate] for rate in sorted(best_at_rate)]
    # The budgets are upper bounds.  A higher-rate point with distortion no
    # better than an already feasible lower-rate point is never operationally
    # optimal, even if it lies on a geometric convex hull.  Remove these
    # Pareto-dominated points before time-sharing convexification.
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


def _balanced_schedule_uses_upper(
    token_index: int,
    *,
    upper_tokens: int,
    horizon_tokens: int,
) -> bool:
    """Decoder-reproducible balanced rational time-sharing decision.

    Both endpoints derive the mode from the global token counter and the two
    uint64 fields in the 32-byte schedule header.  Exactly ``upper_tokens`` of
    one complete horizon use the upper-rate packet, with no per-token mode bit.
    """

    if (
        not isinstance(token_index, int)
        or not isinstance(upper_tokens, int)
        or not isinstance(horizon_tokens, int)
        or isinstance(token_index, bool)
        or isinstance(upper_tokens, bool)
        or isinstance(horizon_tokens, bool)
        or horizon_tokens <= 0
        or not 0 <= token_index < horizon_tokens
        or not 0 <= upper_tokens <= horizon_tokens
    ):
        raise ValueError("invalid balanced time-sharing schedule arguments")
    return ((token_index + 1) * upper_tokens) // horizon_tokens > (
        token_index * upper_tokens
    ) // horizon_tokens


def _fixed_rate_hull(
    *,
    rd: Mapping[str, Any],
    raw_space: Mapping[str, Any],
    deployment_artifact_size_bytes: int,
    horizon_tokens: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    if deployment_artifact_size_bytes <= 0 or horizon_tokens <= 0:
        raise CellExecutionError("fixed-rate hull has invalid deployment inputs")
    side_rate = 8.0 * deployment_artifact_size_bytes / horizon_tokens
    measured: list[dict[str, Any]] = [
        {
            "name": "zero_event_calibration_mean",
            "q": None,
            "packet_bits_per_token": 0.0,
            "side_information_bits_per_token": side_rate,
            "total_bits_per_token": side_rate,
            "raw_space_fvu": 1.0,
        }
    ]
    for q in sorted(int(item) for item in rd["points"]):
        packet_rate = float(rd["points"][str(q)]["rate_bits_per_token"])
        measured.append(
            {
                "name": f"q{q}",
                "q": q,
                "packet_bits_per_token": packet_rate,
                "side_information_bits_per_token": side_rate,
                "total_bits_per_token": packet_rate + side_rate,
                "raw_space_fvu": float(raw_space["points"][str(q)]["fvu_pooled"]),
            }
        )
    hull = _lower_convex_rate_envelope(measured)
    if not hull:
        raise CellExecutionError("fixed-rate measurement produced an empty hull")
    return measured, hull, side_rate


def _selected_time_sharing_plans(
    ctx: _Context,
    *,
    rd: Mapping[str, Any],
    raw_space: Mapping[str, Any],
    deployment_artifact_size_bytes: int,
) -> dict[str, dict[str, Any]]:
    """Resolve the one operational mixture, if any, for each fixed budget."""

    if ctx.cell.phase is Phase.PHASE1 or raw_space.get("eligible") is not True:
        return {}
    horizon = int(ctx.values["evaluation.side_information_amortization_tokens"])
    budgets = tuple(
        float(item)
        for item in ctx.values["evaluation.fixed_rate_budgets_bits_per_token"]
    )
    if horizon <= 0 or not budgets:
        raise CellExecutionError("real time-sharing plan has invalid rate inputs")
    _, hull, _ = _fixed_rate_hull(
        rd=rd,
        raw_space=raw_space,
        deployment_artifact_size_bytes=deployment_artifact_size_bytes,
        horizon_tokens=horizon,
    )
    schedule_rate = 8.0 * TIME_SHARING_HEADER_BYTES / horizon
    plans: dict[str, dict[str, Any]] = {}
    for budget in budgets:
        if budget < float(hull[0]["total_bits_per_token"]) or budget >= float(
            hull[-1]["total_bits_per_token"]
        ):
            continue
        upper_index = next(
            index
            for index, point in enumerate(hull)
            if float(point["total_bits_per_token"]) >= budget
        )
        lower = hull[max(0, upper_index - 1)]
        upper = hull[upper_index]
        lower_rate = float(lower["total_bits_per_token"])
        upper_rate = float(upper["total_bits_per_token"])
        if math.isclose(budget, upper_rate, rel_tol=0.0, abs_tol=1e-12):
            continue
        target_weight = (budget - schedule_rate - lower_rate) / (
            upper_rate - lower_rate
        )
        upper_tokens = max(0, min(horizon, math.floor(target_weight * horizon)))
        if upper_tokens <= 0:
            continue
        if upper_tokens >= horizon:
            raise CellExecutionError("time-sharing mixture selected no lower endpoint")
        weight = upper_tokens / horizon
        achieved_rate = (
            (1.0 - weight) * lower_rate + weight * upper_rate + schedule_rate
        )
        if achieved_rate > budget + 1e-12:
            raise CellExecutionError(
                "selected time-sharing plan exceeded its fixed-rate budget"
            )
        key = _time_sharing_plan_key(
            budget=budget,
            lower_name=str(lower["name"]),
            upper_name=str(upper["name"]),
            upper_tokens=upper_tokens,
            horizon_tokens=horizon,
        )
        plans[key] = {
            "budget_bits_per_token": budget,
            "lower_name": str(lower["name"]),
            "lower_q": lower["q"],
            "upper_name": str(upper["name"]),
            "upper_q": upper["q"],
            "upper_tokens": upper_tokens,
            "horizon_tokens": horizon,
            "upper_mixture_weight": weight,
            "achieved_total_bits_per_token": achieved_rate,
        }
    return plans


def _fixed_rate_raw_score(
    ctx: _Context,
    *,
    rd: Mapping[str, Any],
    raw_space: Mapping[str, Any],
    deployment_path: Path,
    deployment_hash: str,
    calibration_hash: str,
    deployment_schedule_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply the frozen total-rate policy to raw-space distortion."""

    values = ctx.values
    if ctx.cell.phase is Phase.PHASE1:
        phase1_contract = {
            "evaluation.rate_axis": "not_applicable_synthetic_identification",
            "evaluation.fixed_rate_budgets_bits_per_token": (),
            "evaluation.rate_interpolation": "not_applicable",
            "evaluation.rate_out_of_range": "not_applicable",
            "evaluation.side_information_amortization_tokens": 0,
            "evaluation.zero_rate_reconstruction": "not_applicable",
        }
        if any(
            values[name] != expected for name, expected in phase1_contract.items()
        ) or values["evaluation.selection_score"] not in {
            "minimum_normalized_identification_margin",
            "phase1_factor_identification_conjunction",
        }:
            raise CellExecutionError("Phase-1 rate decisions must be not-applicable")
        return {
            "schema": "bsc-fixed-rate-raw-selection-v1",
            "applicable": False,
            "eligible": False,
            "reason": "phase1_uses_truth_known_identification",
        }
    expected = {
        "evaluation.rate_axis": (
            "fixed_width_packet_plus_amortized_side_info_bits_per_token"
        ),
        "evaluation.rate_interpolation": ("lower_convex_envelope_linear_time_sharing"),
        "evaluation.rate_out_of_range": "ineligible_no_extrapolation",
        "evaluation.selection_score": "negative_mean_raw_space_fvu_at_fixed_rates",
        "evaluation.zero_rate_reconstruction": "calibration_mean_per_site",
        "codec.time_sharing_schedule_contract": (
            "balanced_global_token_counter_u64_v1"
        ),
    }
    mismatched = {
        name: values[name]
        for name, wanted in expected.items()
        if values[name] != wanted
    }
    if mismatched:
        raise CellExecutionError(
            "unsupported fixed-rate policy decisions: " + canonical_json(mismatched)
        )
    if (
        rd.get("codec_meta", {}).get("packet_contract")
        != values["codec.packet_contract"]
        or rd.get("codec_meta", {}).get("side_information_contract")
        != values["codec.side_information_contract"]
    ):
        raise CellExecutionError(
            "R-D result is not bound to the resolved codec contract"
        )
    horizon = int(values["evaluation.side_information_amortization_tokens"])
    budgets = tuple(
        float(item) for item in values["evaluation.fixed_rate_budgets_bits_per_token"]
    )
    if horizon <= 0 or not budgets:
        raise CellExecutionError("real fixed-rate policy requires horizon and budgets")
    artifact_bytes = deployment_path.stat().st_size
    if not isinstance(deployment_schedule_manifest, Mapping) or (
        deployment_schedule_manifest.get("schema") != TIME_SHARING_BUNDLE_SCHEMA
        or deployment_schedule_manifest.get("cell_id") != ctx.cell.cell_id
        or deployment_schedule_manifest.get("deployment_codec_sha256")
        != deployment_hash
        or deployment_schedule_manifest.get("schedule_contract")
        != values["codec.time_sharing_schedule_contract"]
        or deployment_schedule_manifest.get("record_size_bytes")
        != TIME_SHARING_HEADER_BYTES
    ):
        raise CellExecutionError(
            "fixed-rate selection lacks its deployment schedule bundle binding"
        )
    raw_schedule_records = deployment_schedule_manifest.get("records")
    if not isinstance(raw_schedule_records, list):
        raise CellExecutionError("deployment schedule bundle has no record manifest")
    schedule_records = {
        str(record.get("schedule_key")): record
        for record in raw_schedule_records
        if isinstance(record, Mapping)
    }
    if len(schedule_records) != len(raw_schedule_records):
        raise CellExecutionError("deployment schedule bundle repeats a schedule key")
    measured, hull, side_rate = _fixed_rate_hull(
        rd=rd,
        raw_space=raw_space,
        deployment_artifact_size_bytes=artifact_bytes,
        horizon_tokens=horizon,
    )
    schedule_rate = 8.0 * TIME_SHARING_HEADER_BYTES / horizon
    fixed: list[dict[str, Any]] = []
    raw_reason = (
        None
        if raw_space.get("eligible") is True
        else "raw_space_decoder_requires_unpriced_oracle_information"
    )
    any_out_of_range = False
    for budget in budgets:
        entry: dict[str, Any] = {"budget_bits_per_token": budget}
        if raw_reason is not None or budget < float(hull[0]["total_bits_per_token"]):
            out_of_range = budget < float(hull[0]["total_bits_per_token"])
            any_out_of_range = any_out_of_range or out_of_range
            entry.update(
                {
                    "eligible": False,
                    "raw_space_fvu": None,
                    "bracket": None,
                    "upper_mixture_weight": None,
                    "achieved_total_bits_per_token": None,
                    "mixing_schedule": None,
                    "reason": raw_reason or "budget_outside_measured_envelope",
                }
            )
            fixed.append(entry)
            continue
        if budget >= float(hull[-1]["total_bits_per_token"]):
            endpoint = hull[-1]
            entry.update(
                {
                    "eligible": True,
                    "raw_space_fvu": float(endpoint["raw_space_fvu"]),
                    "bracket": [endpoint["name"], endpoint["name"]],
                    "upper_mixture_weight": 0.0,
                    "achieved_total_bits_per_token": float(
                        endpoint["total_bits_per_token"]
                    ),
                    "mixing_schedule": None,
                    "reason": "best_measured_point_within_at_most_budget",
                }
            )
            fixed.append(entry)
            continue
        upper_index = next(
            index
            for index, point in enumerate(hull)
            if float(point["total_bits_per_token"]) >= budget
        )
        lower_index = max(0, upper_index - 1)
        lower, upper = hull[lower_index], hull[upper_index]
        lower_rate = float(lower["total_bits_per_token"])
        upper_rate = float(upper["total_bits_per_token"])
        if math.isclose(budget, upper_rate, rel_tol=0.0, abs_tol=1e-12):
            entry.update(
                {
                    "eligible": True,
                    "raw_space_fvu": float(upper["raw_space_fvu"]),
                    "bracket": [upper["name"], upper["name"]],
                    "upper_mixture_weight": 0.0,
                    "achieved_total_bits_per_token": upper_rate,
                    "mixing_schedule": None,
                    "reason": "measured_endpoint_exactly_matches_budget",
                }
            )
            fixed.append(entry)
            continue
        available_for_packets = budget - schedule_rate
        target_weight = (available_for_packets - lower_rate) / (upper_rate - lower_rate)
        upper_tokens = max(
            0,
            min(horizon, math.floor(target_weight * horizon)),
        )
        if upper_tokens == 0:
            entry.update(
                {
                    "eligible": True,
                    "raw_space_fvu": float(lower["raw_space_fvu"]),
                    "bracket": [lower["name"], lower["name"]],
                    "upper_mixture_weight": 0.0,
                    "achieved_total_bits_per_token": lower_rate,
                    "mixing_schedule": None,
                    "reason": "schedule_header_would_exceed_budget_use_lower_endpoint",
                }
            )
            fixed.append(entry)
            continue
        weight = upper_tokens / horizon
        achieved_rate = (
            (1.0 - weight) * lower_rate + weight * upper_rate + schedule_rate
        )
        if achieved_rate > budget + 1e-12:
            raise CellExecutionError(
                "balanced time-sharing arithmetic exceeded its fixed-rate budget"
            )
        schedule_key = _time_sharing_plan_key(
            budget=budget,
            lower_name=str(lower["name"]),
            upper_name=str(upper["name"]),
            upper_tokens=upper_tokens,
            horizon_tokens=horizon,
        )
        scheduled = raw_space.get("operational_time_sharing", {}).get(schedule_key)
        if not isinstance(scheduled, Mapping):
            raise CellExecutionError(
                "selected time-sharing bracket was not executed on raw evaluation rows"
            )
        expected_schedule = {
            "budget_bits_per_token": budget,
            "lower_name": str(lower["name"]),
            "upper_name": str(upper["name"]),
            "upper_tokens": upper_tokens,
            "horizon_tokens": horizon,
        }
        mismatched_schedule = {
            name: scheduled.get(name)
            for name, expected_value in expected_schedule.items()
            if scheduled.get(name) != expected_value
        }
        if mismatched_schedule:
            raise CellExecutionError(
                "executed time-sharing measurement has the wrong schedule binding: "
                + canonical_json(mismatched_schedule)
            )
        schedule_binding = scheduled.get("deployment_schedule")
        record = schedule_records.get(schedule_key)
        if not isinstance(schedule_binding, Mapping) or not isinstance(record, Mapping):
            raise CellExecutionError(
                "executed time-sharing measurement lacks its serialized header"
            )
        expected_serialized_binding = {
            "artifact_sha256": deployment_schedule_manifest["artifact_sha256"],
            "record_index": raw_schedule_records.index(record),
            "offset_bytes": record["offset_bytes"],
            "size_bytes": TIME_SHARING_HEADER_BYTES,
            "record_sha256": record["sha256"],
            "binding_magic_u64": record["binding_magic_u64"],
        }
        if dict(schedule_binding) != expected_serialized_binding:
            raise CellExecutionError(
                "executed time-sharing measurement/header artifact binding mismatch"
            )
        fvu = float(scheduled["raw_space_fvu"])
        if not math.isfinite(fvu) or fvu < 0:
            raise CellExecutionError(
                "executed time-sharing measurement has invalid raw distortion"
            )
        if fvu >= float(lower["raw_space_fvu"]):
            entry.update(
                {
                    "eligible": True,
                    "raw_space_fvu": float(lower["raw_space_fvu"]),
                    "bracket": [lower["name"], lower["name"]],
                    "upper_mixture_weight": 0.0,
                    "achieved_total_bits_per_token": lower_rate,
                    "mixing_schedule": None,
                    "rejected_mixing_schedule": {
                        "schedule_key": schedule_key,
                        "raw_space_fvu": fvu,
                        "reason": "executed_schedule_did_not_improve_lower_endpoint",
                    },
                    "reason": ("lower_endpoint_outperformed_executed_time_sharing"),
                }
            )
            fixed.append(entry)
            continue
        entry.update(
            {
                "eligible": True,
                "raw_space_fvu": fvu,
                "bracket": [lower["name"], upper["name"]],
                "upper_mixture_weight": weight,
                "achieved_total_bits_per_token": achieved_rate,
                "mixing_schedule": {
                    "contract": values["codec.time_sharing_schedule_contract"],
                    "header_bytes": TIME_SHARING_HEADER_BYTES,
                    "header_layout": TIME_SHARING_HEADER_LAYOUT_DESCRIPTION,
                    "artifact_sha256": schedule_binding["artifact_sha256"],
                    "record_index": schedule_binding["record_index"],
                    "record_offset_bytes": schedule_binding["offset_bytes"],
                    "record_sha256": schedule_binding["record_sha256"],
                    "binding_magic_u64": schedule_binding["binding_magic_u64"],
                    "horizon_tokens": horizon,
                    "upper_tokens": upper_tokens,
                    "rule": (
                        "upper iff floor((i+1)*upper_tokens/horizon) > "
                        "floor(i*upper_tokens/horizon)"
                    ),
                    "per_token_mode_bits": 0,
                    "header_bits_per_token": schedule_rate,
                    "evaluation_tokens": int(scheduled["evaluation_tokens"]),
                    "evaluation_upper_tokens": int(
                        scheduled["evaluation_upper_tokens"]
                    ),
                    "distortion_measurement": scheduled["distortion_measurement"],
                },
                "reason": ("adjacent_envelope_executed_balanced_rational_time_sharing"),
            }
        )
        fixed.append(entry)
    eligible = (
        raw_reason is None
        and not any_out_of_range
        and all(item["eligible"] for item in fixed)
    )
    score = (
        -sum(float(item["raw_space_fvu"]) for item in fixed) / len(fixed)
        if eligible
        else None
    )
    payload = {
        "schema": "bsc-fixed-rate-raw-selection-v1",
        "applicable": True,
        "cell_id": ctx.cell.cell_id,
        "deployment_codec_sha256": deployment_hash,
        "calibration_sha256": calibration_hash,
        "side_information": {
            "artifact": "deployable_codec",
            "artifact_size_bytes": artifact_bytes,
            "amortization_tokens": horizon,
            "bits_per_token": side_rate,
            "includes_optimizer_checkpoint": False,
            "time_sharing_schedule_contract": values[
                "codec.time_sharing_schedule_contract"
            ],
            "mixture_header_bytes_when_used": TIME_SHARING_HEADER_BYTES,
            "deployment_schedule_bundle_sha256": (
                deployment_schedule_manifest["artifact_sha256"]
            ),
            "deployment_schedule_bundle_size_bytes": (
                deployment_schedule_manifest["artifact_size_bytes"]
            ),
            "deployment_schedule_record_count": (
                deployment_schedule_manifest["record_count"]
            ),
        },
        "packet_formula": {
            "rate_model": rd["rate_model"],
            "count_width_bits": rd["support_count_width_bits"],
            "block_id_width_bits": rd["support_id_width_bits"],
            "included_block_alphabet": rd["codec_meta"]["count_alphabet_max"],
            "sparse_amplitude_bits": "q * block_width * selected_events",
        },
        "rate_axis": values["evaluation.rate_axis"],
        "interpolation": values["evaluation.rate_interpolation"],
        "out_of_range": values["evaluation.rate_out_of_range"],
        "zero_rate_reconstruction": values["evaluation.zero_rate_reconstruction"],
        "measured_points": measured,
        "lower_convex_envelope": hull,
        "fixed_budgets": fixed,
        "selection_score_name": values["evaluation.selection_score"],
        "selection_score": score,
        "eligible": eligible,
        "reason": (
            "eligible"
            if eligible
            else raw_reason or "one_or_more_budgets_outside_measured_envelope"
        ),
    }
    payload["content_sha256"] = hashlib.sha256(
        canonical_json(payload).encode("utf-8")
    ).hexdigest()
    return payload


def _selection_validation_metrics(
    phase: Phase,
    *,
    identification: Mapping[str, Any] | None,
    fixed_rate: Mapping[str, Any],
) -> dict[str, bool | float]:
    """Map executor endpoints into the schema consumed by live policies."""

    if phase is Phase.PHASE1:
        return {
            "phase1_identification_conjunction": bool(
                identification is not None
                and identification["native"]["passed"]
                and identification["deployed"]["passed"]
            ),
            "phase1_identification_margin": (
                -1.0e9
                if identification is None
                else min(
                    float(identification["native"]["margin"]),
                    float(identification["deployed"]["margin"]),
                )
            ),
        }
    selection_score = fixed_rate.get("selection_score")
    if selection_score is None and fixed_rate.get("eligible") is not True:
        selection_score = PHASE2_INELIGIBLE_SELECTION_SCORE
    if (
        not isinstance(selection_score, (int, float))
        or isinstance(selection_score, bool)
        or not math.isfinite(float(selection_score))
    ):
        raise CellExecutionError(
            "fixed-rate selection score must be finite or ineligible"
        )
    return {PHASE2_SELECTION_METRIC_KEY: float(selection_score)}


def _evaluate(
    ctx: _Context,
    prerequisites: Mapping[str, tuple[Path, str]],
) -> tuple[tuple[str, Path], ...]:
    preparation = _load_preparation(prerequisites["preparation"][0], ctx.cell.cell_id)
    checkpoint_path, checkpoint_hash = prerequisites["checkpoint"]
    calibration_path, calibration_hash = prerequisites["calibration"]
    deployment_path, deployment_hash = prerequisites["deployment_codec"]
    calibration_record = _read_object(
        prerequisites["calibration_record"][0], label="calibration record"
    )
    if (
        calibration_record.get("cell_id") != ctx.cell.cell_id
        or calibration_record.get("checkpoint_sha256") != checkpoint_hash
        or calibration_record.get("codec_sha256") != calibration_hash
    ):
        raise CellExecutionError("calibration record/input binding mismatch")
    deployment, model, codec, training_summary = _load_deployable_codec(
        deployment_path,
        cell_id=ctx.cell.cell_id,
        checkpoint_hash=checkpoint_hash,
        calibration_hash=calibration_hash,
        preparation_hash=prerequisites["preparation"][1],
        device=_device(ctx),
    )
    if (
        calibration_record.get("deployment_codec_sha256") != deployment_hash
        or calibration_record.get("deployment_codec_size_bytes")
        != deployment_path.stat().st_size
    ):
        raise CellExecutionError("deployable codec/input binding mismatch")

    device = _device(ctx)
    native = _evaluate_native_selector(
        model,
        _prefetched_evaluation_batches(ctx, preparation),
        device=device,
        selection_mode="topk",
    )
    deployed = _evaluate_native_selector(
        model,
        _prefetched_evaluation_batches(ctx, preparation),
        device=device,
        selection_mode="threshold",
    )
    shared_native = evaluate_shared_code(
        model,
        _prefetched_evaluation_batches(ctx, preparation),
        device=device,
        selection_mode="topk",
    )
    shared_deployed = evaluate_shared_code(
        model,
        _prefetched_evaluation_batches(ctx, preparation),
        device=device,
        selection_mode="threshold",
    )
    rd = evaluate_rd(
        model,
        codec,
        _prefetched_rd_evaluation_batches(ctx, preparation),
        row_len=1 if preparation["data"]["kind"] == "synthetic" else None,
        device=str(device),
    )
    first_batch = next(_evaluation_batches(ctx, preparation, "evaluation"))
    roundtrip_q = max(codec.spec.qs)
    packet = encode_batch(model, codec, first_batch, q=roundtrip_q)
    decoded = decode_batch(model, codec, packet)
    packet_error = (decoded.cpu().float() - first_batch.float()).double().square().sum()
    packet_centered = (
        (first_batch.float() - codec.calib_mean.float().unsqueeze(0))
        .double()
        .square()
        .sum()
        .clamp_min(1e-30)
    )
    roundtrip = {
        "source_free_decode": True,
        "tokens": packet.n_tokens,
        "events": int(packet.counts.sum()),
        "finite": bool(torch.isfinite(decoded).all()),
        "shape_matches": list(decoded.shape) == list(first_batch.shape),
        "quantizer_bits": roundtrip_q,
        "fvu_pooled": float(packet_error / packet_centered),
    }

    recovery: dict[str, Any] | None = None
    identification: dict[str, Any] | None = None
    if ctx.cell.phase is Phase.PHASE1:
        matching_dataset = _synthetic_dataset(ctx.cell, "eval")
        evaluation_dataset = _synthetic_dataset(
            ctx.cell, str(preparation["data"]["evaluation_stream"])
        )
        factor_calibration_range = tuple(
            int(item) for item in preparation["data"]["ranges"]["factor_calibration"]
        )
        evaluation_range = tuple(
            int(item) for item in preparation["data"]["ranges"]["evaluation"]
        )
        active = max(1, int(ctx.values["data.active_factors_per_example"]))
        ambient = max(1, sum(int(item) for item in model.cfg.site_dims))
        recovery_batch_size = max(
            1,
            min(
                int(ctx.values["optimizer.batch_tokens"]),
                16_000_000 // max(1, active * ambient),
            ),
        )
        recovery = {
            mode: _synthetic_recovery(
                model,
                matching_dataset,
                evaluation_dataset,
                preparation["data"]["normalization"],
                selection_mode=("topk" if mode == "native" else "threshold"),
                factor_calibration_range=factor_calibration_range,
                evaluation_range=evaluation_range,
                batch_size=recovery_batch_size,
            )
            for mode in ("native", "deployed")
        }
        identification = {
            endpoint: _phase1_identification_evidence(
                recovery[endpoint],
                ctx.values["qualification.phase1_identification_thresholds"],
            )
            for endpoint in ("native", "deployed")
        }
    raw_space = _evaluate_raw_space(
        ctx,
        preparation,
        model,
        codec,
        deployment,
        time_sharing_plans={},
    )
    schedule_plans = _selected_time_sharing_plans(
        ctx,
        rd=rd,
        raw_space=raw_space,
        deployment_artifact_size_bytes=deployment_path.stat().st_size,
    )
    deployment_schedule_manifest = _write_deployment_schedule_bundle(
        ctx.deployment_schedules,
        cell_id=ctx.cell.cell_id,
        deployment_codec_sha256=deployment_hash,
        schedule_contract=str(ctx.values["codec.time_sharing_schedule_contract"]),
        plans=schedule_plans,
    )
    loaded_schedule_plans = _load_deployment_schedule_bundle(
        ctx.deployment_schedules,
        deployment_schedule_manifest,
        cell_id=ctx.cell.cell_id,
        deployment_codec_sha256=deployment_hash,
        schedule_contract=str(ctx.values["codec.time_sharing_schedule_contract"]),
    )
    if loaded_schedule_plans:
        scheduled_raw = _evaluate_raw_space(
            ctx,
            preparation,
            model,
            codec,
            deployment,
            time_sharing_plans=loaded_schedule_plans,
        )
        raw_space["operational_time_sharing"] = scheduled_raw[
            "operational_time_sharing"
        ]
    fixed_rate = _fixed_rate_raw_score(
        ctx,
        rd=rd,
        raw_space=raw_space,
        deployment_path=deployment_path,
        deployment_hash=deployment_hash,
        calibration_hash=calibration_hash,
        deployment_schedule_manifest=deployment_schedule_manifest,
    )
    validation = _selection_validation_metrics(
        ctx.cell.phase,
        identification=identification,
        fixed_rate=fixed_rate,
    )
    site_only = shared_deployed["site_only_fvu"]
    loo = shared_deployed["leave_one_site_out_fvu"]
    coordinate = shared_deployed["partial_view_coordinate_concordance"]
    site_coordinate = coordinate["site_only"]
    loo_coordinate = coordinate["leave_one_site_out"]
    n_sites = len(site_only)
    site_only_heldout = [
        float(site_only[source][target])
        for source in range(n_sites)
        for target in range(n_sites)
        if source != target
    ]
    sharing_summary = {
        "all_site_fvu_mean": sum(
            float(item) for item in shared_deployed["full_fvu_per_site"]
        )
        / n_sites,
        "site_only_heldout_fvu_mean": (
            None
            if not site_only_heldout
            else sum(site_only_heldout) / len(site_only_heldout)
        ),
        "leave_one_out_heldout_fvu_mean": (
            sum(float(loo[site][site]) for site in range(n_sites)) / n_sites
        ),
        "site_only_support_iou_mean": (
            sum(float(item) for item in shared_deployed["site_only_support_iou"])
            / n_sites
        ),
        "leave_one_out_support_iou_mean": (
            sum(
                float(item)
                for item in shared_deployed["leave_one_site_out_support_iou"]
            )
            / n_sites
        ),
        "site_only_coordinate_concordance_mean": (
            sum(float(item) for item in site_coordinate["concordance"]) / n_sites
        ),
        "site_only_coordinate_concordance_min": min(
            float(item) for item in site_coordinate["concordance"]
        ),
        "leave_one_out_coordinate_concordance_mean": (
            sum(float(item) for item in loo_coordinate["concordance"]) / n_sites
        ),
        "leave_one_out_coordinate_concordance_min": min(
            float(item) for item in loo_coordinate["concordance"]
        ),
        "site_only_intersection_recall_mean": (
            sum(float(item) for item in site_coordinate["support_intersection_recall"])
            / n_sites
        ),
        "leave_one_out_intersection_recall_mean": (
            sum(float(item) for item in loo_coordinate["support_intersection_recall"])
            / n_sites
        ),
        "site_only_intersection_recall_min": min(
            float(item) for item in site_coordinate["support_intersection_recall"]
        ),
        "leave_one_out_intersection_recall_min": min(
            float(item) for item in loo_coordinate["support_intersection_recall"]
        ),
        "site_only_intersection_energy_coverage_mean": (
            sum(float(item) for item in site_coordinate["decoded_energy_coverage"])
            / n_sites
        ),
        "leave_one_out_intersection_energy_coverage_mean": (
            sum(float(item) for item in loo_coordinate["decoded_energy_coverage"])
            / n_sites
        ),
        "site_only_intersection_energy_coverage_min": min(
            float(item) for item in site_coordinate["decoded_energy_coverage"]
        ),
        "leave_one_out_intersection_energy_coverage_min": min(
            float(item) for item in loo_coordinate["decoded_energy_coverage"]
        ),
    }
    selection_metrics: dict[str, Any] = {
        "validation": validation,
        "native": {
            "fvu_pooled": native["fvu_pooled"],
            "avg_active_blocks": native["avg_active_blocks"],
            "isolated_loss_gain_diagnostics": native["isolated_loss_gain_diagnostics"],
        },
        "deployed": {
            "fvu_pooled": deployed["fvu_pooled"],
            "avg_active_blocks": deployed["avg_active_blocks"],
            "isolated_loss_gain_diagnostics": deployed[
                "isolated_loss_gain_diagnostics"
            ],
        },
        "shared": {
            endpoint: {
                "fvu_pooled": (
                    shared_native if endpoint == "native" else shared_deployed
                )["full_fvu_pooled"]
            }
            for endpoint in ("native", "deployed")
        },
        "sharing_guard": sharing_summary,
        "codec": {
            "excluded_calibration_event_fraction": calibration_record[
                "excluded_calibration_event_fraction"
            ],
            "excluded_evaluation_event_fraction": rd["eval_excluded_event_share"],
            "support_bits_ci95": rd["support_bits_ci95"],
            "points": {
                q: {
                    "fvu_pooled": rd["points"][q]["fvu_pooled"],
                    "fvu_ci95": rd["points"][q]["fvu_ci95"],
                    "rate_bits_per_token": rd["points"][q]["rate_bits_per_token"],
                }
                for q in [str(item) for item in codec.spec.qs]
            },
        },
        "raw_codec": {
            "eligible": raw_space.get("eligible") is True,
            "points": raw_space.get("points", {}),
        },
        "recovery": recovery,
        "identification": identification,
        "fixed_rate_raw_selection": fixed_rate,
    }
    selection_metrics_sha256 = hashlib.sha256(
        canonical_json(selection_metrics).encode("utf-8")
    ).hexdigest()
    payload = {
        "schema": EVALUATION_SCHEMA,
        "cell_id": ctx.cell.cell_id,
        "inputs": {
            "checkpoint": checkpoint_hash,
            "calibration": calibration_hash,
            "deployment_codec": deployment_hash,
            "deployment_schedules": deployment_schedule_manifest["artifact_sha256"],
        },
        "preparation_sha256": prerequisites["preparation"][1],
        "phase": ctx.cell.phase.value,
        "split": ctx.values["evaluation.split"],
        "endpoint_profile": ctx.values["evaluation.endpoint_profile"],
        "native_selector": native,
        "deployed_selector": deployed,
        "shared_code": {
            "native": shared_native,
            "deployed": shared_deployed,
        },
        "rate_distortion": rd,
        "fixed_rate_raw_selection": fixed_rate,
        "synthetic_recovery": recovery,
        "synthetic_identification": identification,
        "codec_roundtrip": roundtrip,
        "raw_space": raw_space,
        "deployment_schedules": deployment_schedule_manifest,
        "validation": validation,
        "selection_metrics": selection_metrics,
        "selection_metrics_sha256": selection_metrics_sha256,
        "checkpoint_metadata": {
            "step_idx": training_summary["step_idx"],
            "accepted_tokens": training_summary["accepted_tokens"],
        },
    }
    if not _finite_json(payload):
        raise CellExecutionError("evaluation contains a non-finite or non-JSON value")
    _write_immutable_json(ctx.evaluation, payload)
    return (
        ("deployment_schedules", ctx.deployment_schedules),
        ("evaluation", ctx.evaluation),
    )


def _qualify(
    ctx: _Context,
    prerequisites: Mapping[str, tuple[Path, str]],
) -> tuple[tuple[str, Path], ...]:
    evaluation = _read_object(
        prerequisites["evaluation"][0], label="evaluation artifact"
    )
    if evaluation.get("schema") != EVALUATION_SCHEMA:
        raise CellExecutionError("evaluation artifact has the wrong schema")
    input_hashes = {
        kind: prerequisites[kind][1]
        for kind in (
            "checkpoint",
            "calibration",
            "deployment_codec",
            "deployment_schedules",
            "evaluation",
        )
    }
    expected_eval_inputs = {
        "checkpoint": input_hashes["checkpoint"],
        "calibration": input_hashes["calibration"],
        "deployment_codec": input_hashes["deployment_codec"],
        "deployment_schedules": input_hashes["deployment_schedules"],
    }
    preparation = _load_preparation(prerequisites["preparation"][0], ctx.cell.cell_id)
    training_report = _read_object(
        prerequisites["training_report"][0], label="training report"
    )
    calibration_record = _read_object(
        prerequisites["calibration_record"][0], label="calibration record"
    )
    if ctx.values["qualification.thresholds_version"] != "2026-07-20.v1":
        raise CellExecutionError(
            "unsupported qualification.thresholds_version "
            + repr(ctx.values["qualification.thresholds_version"])
        )
    threshold_map = {
        "schema": "bsc-integrity-thresholds-2026-07-20.v1",
        "support_target_abs_error_max": 0.1,
        "codec_excluded_calibration_event_fraction_max": 0.01,
        "codec_excluded_evaluation_event_fraction_max": 0.01,
        "probability_metric_range": [0.0, 1.0],
        "required_quantizer_bits": list(ctx.values["codec.quantizer_bits"]),
        "phase1_identification_thresholds": list(
            ctx.values["qualification.phase1_identification_thresholds"]
        ),
        "phase1_identification_enforced": ctx.values["runtime.smoke"] is False,
        "fixed_rate_budget_scale_factor": ctx.values[
            "evaluation.fixed_rate_budget_scale_factor"
        ],
        "fixed_rate_budget_scale_contract": ctx.values[
            "evaluation.fixed_rate_budget_scale_contract"
        ],
        "production_min_nonzero_rate_endpoints": ctx.values[
            "precision.preflight_min_nonzero_rate_endpoints"
        ],
    }
    rd = evaluation.get("rate_distortion", {})
    raw = evaluation.get("raw_space", {})
    native = evaluation.get("native_selector", {})
    deployed = evaluation.get("deployed_selector", {})
    shared = evaluation.get("shared_code", {})
    recovery = evaluation.get("synthetic_recovery")
    identification = evaluation.get("synthetic_identification")
    fixed_rate = evaluation.get("fixed_rate_raw_selection", {})
    deployment_schedule_manifest = evaluation.get("deployment_schedules")
    required_qs = [str(item) for item in threshold_map["required_quantizer_bits"]]

    def finite_number(value: Any) -> bool:
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        )

    def valid_ci(value: Any) -> bool:
        return (
            isinstance(value, list)
            and len(value) == 2
            and all(finite_number(item) for item in value)
            and float(value[0]) <= float(value[1])
        )

    def valid_isolated_loss_diagnostics(endpoint: Mapping[str, Any]) -> bool:
        diagnostics = endpoint.get("isolated_loss_gain_diagnostics")
        if ctx.values["model.selection_score"] != "isolated_loss_decrease":
            return diagnostics == {
                "applicable": False,
                "reason": "selection_score_not_isolated_loss_decrease",
            }
        if not isinstance(diagnostics, Mapping) or (
            diagnostics.get("schema") != "bsc-isolated-loss-gain-diagnostics-v1"
            or diagnostics.get("applicable") is not True
            or diagnostics.get("observation_contract")
            != "explicit_true_observed_sites_only_v1"
        ):
            return False
        candidate_total = diagnostics.get("candidate_event_count")
        selected_total = diagnostics.get("selected_event_count")
        expected_candidate_total = int(endpoint.get("n_tokens", -1)) * int(
            ctx.values["model.groups"]
        )
        try:
            histogram_total = sum(
                int(count) * int(frequency)
                for count, frequency in endpoint.get(
                    "active_block_count_histogram", {}
                ).items()
            )
        except (AttributeError, TypeError, ValueError):
            return False
        if (
            not isinstance(candidate_total, int)
            or isinstance(candidate_total, bool)
            or candidate_total <= 0
            or candidate_total != expected_candidate_total
            or not isinstance(selected_total, int)
            or isinstance(selected_total, bool)
            or selected_total < 0
            or selected_total != histogram_total
        ):
            return False
        for prefix, total in (
            ("candidate", candidate_total),
            ("selected", selected_total),
        ):
            counts = [
                diagnostics.get(f"{prefix}_{sign}_gain_count")
                for sign in ("negative", "zero", "positive")
            ]
            if (
                any(
                    not isinstance(count, int) or isinstance(count, bool) or count < 0
                    for count in counts
                )
                or sum(counts) != total
            ):
                return False
            fractions = [
                diagnostics.get(f"{prefix}_{sign}_gain_fraction")
                for sign in ("negative", "zero", "positive")
            ]
            if total == 0:
                if fractions != [None, None, None]:
                    return False
            elif any(
                not finite_number(fraction) or not 0.0 <= float(fraction) <= 1.0
                for fraction in fractions
            ) or any(
                not math.isclose(
                    float(fraction), count / total, rel_tol=1e-12, abs_tol=1e-15
                )
                for fraction, count in zip(fractions, counts, strict=True)
            ):
                return False
        return True

    if not isinstance(deployment_schedule_manifest, Mapping):
        raise CellExecutionError("evaluation lacks a deployment schedule manifest")
    loaded_schedule_plans = _load_deployment_schedule_bundle(
        prerequisites["deployment_schedules"][0],
        deployment_schedule_manifest,
        cell_id=ctx.cell.cell_id,
        deployment_codec_sha256=input_hashes["deployment_codec"],
        schedule_contract=str(ctx.values["codec.time_sharing_schedule_contract"]),
    )
    expected_schedule_plans = _selected_time_sharing_plans(
        ctx,
        rd=rd,
        raw_space=raw,
        deployment_artifact_size_bytes=prerequisites["deployment_codec"][0]
        .stat()
        .st_size,
    )

    def schedule_core(plan: Mapping[str, Any]) -> dict[str, Any]:
        return {
            name: plan[name]
            for name in (
                "budget_bits_per_token",
                "lower_name",
                "lower_q",
                "upper_name",
                "upper_q",
                "upper_tokens",
                "horizon_tokens",
                "upper_mixture_weight",
                "achieved_total_bits_per_token",
            )
        }

    deployment_schedule_integrity = bool(
        deployment_schedule_manifest.get("artifact_sha256")
        == input_hashes["deployment_schedules"]
        and set(loaded_schedule_plans) == set(expected_schedule_plans)
        and all(
            canonical_json(schedule_core(loaded_schedule_plans[key]))
            == canonical_json(schedule_core(expected_schedule_plans[key]))
            for key in expected_schedule_plans
        )
        and set(raw.get("operational_time_sharing", {})) == set(loaded_schedule_plans)
    )
    selection_score_diagnostics_integrity = bool(
        valid_isolated_loss_diagnostics(native)
        and valid_isolated_loss_diagnostics(deployed)
    )

    regularizer_calibration = training_report.get("regularizer_calibration")
    reported_model_cfg = training_report.get("model_cfg")
    regularizer_calibration_integrity = False
    if isinstance(regularizer_calibration, dict) and isinstance(
        reported_model_cfg, dict
    ):
        coefficient_mode = ctx.values["objective.regularizer_coefficient_mode"]
        target_ratio = ctx.values["objective.regularizer_target_initial_ratio"]
        resolved = regularizer_calibration.get("resolved_coefficient")
        reported_resolved = reported_model_cfg.get("lambda_regularizer")
        common_calibration_fields = (
            regularizer_calibration.get("mode") == coefficient_mode
            and finite_number(resolved)
            and finite_number(reported_resolved)
            and float(resolved) == float(reported_resolved)
            and regularizer_calibration.get("declared_absolute_coefficient")
            == ctx.values["objective.regularizer_coefficient"]
        )
        if coefficient_mode == "absolute":
            regularizer_calibration_integrity = bool(
                common_calibration_fields
                and regularizer_calibration.get("contract") == "not_applicable"
                and regularizer_calibration.get("fitted") is False
                and regularizer_calibration.get("target_initial_ratio") is None
                and float(resolved)
                == float(ctx.values["objective.regularizer_coefficient"])
            )
        elif coefficient_mode == "initial_loss_ratio":
            reconstruction = regularizer_calibration.get("initial_reconstruction_loss")
            unweighted = regularizer_calibration.get("initial_regularizer_unweighted")
            achieved = regularizer_calibration.get("achieved_initial_ratio")
            input_binding = regularizer_calibration.get("input")
            regularizer_calibration_integrity = bool(
                common_calibration_fields
                and regularizer_calibration.get("contract")
                == "post_init_train_prefix_true_observation_fp32_v1"
                and regularizer_calibration.get("fitted") is True
                and finite_number(target_ratio)
                and regularizer_calibration.get("target_initial_ratio") == target_ratio
                and finite_number(reconstruction)
                and float(reconstruction) > 0.0
                and finite_number(unweighted)
                and float(unweighted) > 0.0
                and finite_number(achieved)
                and math.isclose(
                    float(resolved),
                    float(target_ratio) * float(reconstruction) / float(unweighted),
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                )
                and math.isclose(
                    float(achieved),
                    float(target_ratio),
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                )
                and isinstance(input_binding, dict)
                and input_binding.get("split") == "train"
                and input_binding.get("start_token") == 0
                and isinstance(input_binding.get("tokens"), int)
                and input_binding["tokens"] > 0
                and isinstance(input_binding.get("sha256"), str)
                and len(input_binding["sha256"]) == 64
            )

    precision_preflight = training_report.get("precision_preflight")
    precision_profile = (
        ctx.values["qualification.profile"]
        == "phase3_production_stability_guardrails_v1"
    )
    precision_preflight_integrity = False
    precision_reconstruction_passed = not precision_profile
    precision_support_passed = not precision_profile
    precision_finite_passed = not precision_profile
    if isinstance(precision_preflight, dict):
        if not precision_profile:
            precision_preflight_integrity = precision_preflight == {
                "applicable": False,
                "contract": "not_applicable",
                "passed": True,
            }
        else:
            rec_error = precision_preflight.get("reconstruction_relative_error")
            support_iou = precision_preflight.get("support_iou")
            output_error = precision_preflight.get("output_relative_error")
            precision_checks = precision_preflight.get("checks")
            precision_thresholds = precision_preflight.get("thresholds")
            precision_input = precision_preflight.get("input")
            rec_max = float(
                ctx.values["precision.preflight_reconstruction_relative_error_max"]
            )
            iou_min = float(ctx.values["precision.preflight_support_iou_min"])
            precision_finite_passed = bool(
                isinstance(precision_checks, dict)
                and precision_checks.get("finite") is True
            )
            precision_reconstruction_passed = bool(
                finite_number(rec_error) and float(rec_error) <= rec_max
            )
            precision_support_passed = bool(
                finite_number(support_iou) and float(support_iou) >= iou_min
            )
            precision_preflight_integrity = bool(
                precision_preflight.get("applicable") is True
                and precision_preflight.get("contract")
                == "fp32_bf16_initial_forward_v1"
                and finite_number(output_error)
                and isinstance(precision_checks, dict)
                and precision_checks
                == {
                    "finite": precision_finite_passed,
                    "reconstruction_relative_error": (precision_reconstruction_passed),
                    "support_iou": precision_support_passed,
                }
                and precision_preflight.get("passed") is all(precision_checks.values())
                and precision_thresholds
                == {
                    "reconstruction_relative_error_max": rec_max,
                    "support_iou_min": iou_min,
                }
                and isinstance(precision_input, dict)
                and precision_input.get("split") == "train"
                and precision_input.get("start_token") == 0
                and precision_input.get("tokens")
                == ctx.values["precision.preflight_tokens"]
                and isinstance(precision_input.get("sha256"), str)
                and len(precision_input["sha256"]) == 64
            )

    production_frontier_passed = not precision_profile
    production_frontier_endpoint_count: int | None = None
    if precision_profile:
        fixed_budget_points = fixed_rate.get("fixed_budgets")
        expected_budgets = list(
            ctx.values["evaluation.fixed_rate_budgets_bits_per_token"]
        )
        side_rate = fixed_rate.get("side_information", {}).get("bits_per_token")
        distinct_nonzero_endpoints: set[str] = set()
        point_contracts: list[bool] = []
        if isinstance(fixed_budget_points, list):
            for point in fixed_budget_points:
                bracket = point.get("bracket") if isinstance(point, dict) else None
                nonzero = (
                    {
                        str(name)
                        for name in bracket
                        if name != "zero_event_calibration_mean"
                    }
                    if isinstance(bracket, list)
                    else set()
                )
                distinct_nonzero_endpoints.update(nonzero)
                point_contracts.append(
                    isinstance(point, dict)
                    and point.get("eligible") is True
                    and finite_number(point.get("raw_space_fvu"))
                    and finite_number(point.get("achieved_total_bits_per_token"))
                    and finite_number(point.get("budget_bits_per_token"))
                    and finite_number(side_rate)
                    and float(point["achieved_total_bits_per_token"]) > float(side_rate)
                    and float(point["achieved_total_bits_per_token"])
                    <= float(point["budget_bits_per_token"]) + 1e-12
                    and bool(nonzero)
                )
        production_frontier_endpoint_count = len(distinct_nonzero_endpoints)
        production_frontier_passed = bool(
            fixed_rate.get("schema") == "bsc-fixed-rate-raw-selection-v1"
            and fixed_rate.get("eligible") is True
            and fixed_rate.get("applicable") is True
            and ctx.values["evaluation.fixed_rate_budget_scale_factor"] == 4.0
            and ctx.values["evaluation.fixed_rate_budget_scale_contract"]
            == "phase3_active_coordinate_ratio_128_over_32_v1"
            and isinstance(fixed_budget_points, list)
            and [point.get("budget_bits_per_token") for point in fixed_budget_points]
            == expected_budgets
            and len(point_contracts) == len(expected_budgets)
            and all(point_contracts)
            and production_frontier_endpoint_count
            >= int(ctx.values["precision.preflight_min_nonzero_rate_endpoints"])
        )

    method_endpoints = (
        evaluation.get("endpoint_profile") == ctx.values["evaluation.endpoint_profile"]
        and finite_number(native.get("fvu_pooled"))
        and finite_number(native.get("avg_active_blocks"))
        and finite_number(deployed.get("fvu_pooled"))
        and finite_number(deployed.get("avg_active_blocks"))
        and all(
            isinstance(shared.get(endpoint), dict)
            and finite_number(shared[endpoint].get("full_fvu_pooled"))
            and shared[endpoint].get("selection_mode")
            == ("topk" if endpoint == "native" else "threshold")
            and isinstance(shared[endpoint].get("used_contribution_eigenvalues"), list)
            for endpoint in ("native", "deployed")
        )
        and all(
            isinstance(rd.get("points", {}).get(q), dict)
            and finite_number(rd["points"][q].get("fvu_pooled"))
            and finite_number(rd["points"][q].get("rate_bits_per_token"))
            and valid_ci(rd["points"][q].get("fvu_ci95"))
            for q in required_qs
        )
        and valid_ci(rd.get("support_bits_ci95"))
        and raw.get("serialized_forward_preprocessing_validated") is True
        and raw.get("source_free_sparse_decode") is True
        and all(
            isinstance(raw.get("points", {}).get(q), dict)
            and finite_number(raw["points"][q].get("fvu_pooled"))
            and valid_ci(raw["points"][q].get("fvu_pooled_ci95"))
            for q in required_qs
        )
        and (
            ctx.cell.phase is not Phase.PHASE1
            or ctx.values["runtime.smoke"] is True
            or (
                isinstance(recovery, dict)
                and isinstance(identification, dict)
                and all(
                    isinstance(recovery.get(endpoint), dict)
                    and isinstance(identification.get(endpoint), dict)
                    for endpoint in ("native", "deployed")
                )
            )
        )
        and (
            ctx.cell.phase is Phase.PHASE1
            or (
                fixed_rate.get("schema") == "bsc-fixed-rate-raw-selection-v1"
                and fixed_rate.get("applicable") is True
                and fixed_rate.get("deployment_codec_sha256")
                == prerequisites["deployment_codec"][1]
            )
        )
    )
    phase1_ranges_ok = True
    if ctx.cell.phase is Phase.PHASE1 and isinstance(recovery, dict):
        expected_recovery_examples = int(
            ctx.values[
                "data.synthetic_confirmation_examples"
                if ctx.values["evaluation.split"] == "confirmation"
                else "data.synthetic_development_examples"
            ]
        )
        expected_factor_calibration_examples = int(
            ctx.values["data.synthetic_factor_calibration_examples"]
        )
        ranged_names = (
            "support_precision",
            "support_recall",
            "support_false_discovery_rate",
            "support_false_positive_rate",
            "support_association_f1_mean",
            "recovered_factor_fraction_at_association_0.5",
            "split_factor_fraction",
            "merge_group_fraction",
            "merged_factor_fraction",
            "shattering_factor_fraction",
            "dilution_factor_fraction",
            "alive_block_fraction",
        )
        phase1_ranges_ok = all(
            recovery.get(endpoint, {}).get("n_truth_factors")
            == ctx.values["data.n_factors"]
            and recovery[endpoint].get("n_factor_calibration_examples")
            == expected_factor_calibration_examples
            and recovery[endpoint].get("n_examples") == expected_recovery_examples
            and all(
                finite_number(recovery[endpoint].get(name))
                and 0.0 <= float(recovery[endpoint][name]) <= 1.0
                for name in ranged_names
            )
            for endpoint in ("native", "deployed")
        )

    calibration_excluded = calibration_record.get("excluded_calibration_event_fraction")
    evaluation_excluded = rd.get("eval_excluded_event_share")
    phase1_endpoint_complete = ctx.cell.phase is not Phase.PHASE1 or (
        isinstance(recovery, dict)
        and isinstance(identification, dict)
        and all(
            isinstance(recovery.get(endpoint), dict)
            and isinstance(identification.get(endpoint), dict)
            and isinstance(identification[endpoint].get("passed"), bool)
            and finite_number(identification[endpoint].get("margin"))
            and isinstance(identification[endpoint].get("checks"), dict)
            for endpoint in ("native", "deployed")
        )
    )
    scientific_endpoint_complete = (
        finite_number(calibration_record.get("threshold_abs_error"))
        and finite_number(calibration_excluded)
        and finite_number(evaluation_excluded)
        and all(
            evaluation.get("codec_roundtrip", {}).get(key) is True
            for key in ("source_free_decode", "finite", "shape_matches")
        )
        and phase1_ranges_ok
        and phase1_endpoint_complete
    )
    scientific_outcome_checks = {
        "support_target_calibration": (
            finite_number(calibration_record.get("threshold_abs_error"))
            and float(calibration_record["threshold_abs_error"])
            <= threshold_map["support_target_abs_error_max"]
        ),
        "codec_calibration_exclusion": (
            finite_number(calibration_excluded)
            and float(calibration_excluded)
            <= threshold_map["codec_excluded_calibration_event_fraction_max"]
        ),
        "codec_evaluation_exclusion": (
            finite_number(evaluation_excluded)
            and float(evaluation_excluded)
            <= threshold_map["codec_excluded_evaluation_event_fraction_max"]
        ),
        "phase1_identification": (
            ctx.cell.phase is not Phase.PHASE1
            or (
                isinstance(identification, dict)
                and all(
                    identification.get(endpoint, {}).get("passed") is True
                    for endpoint in ("native", "deployed")
                )
                and evaluation.get("validation", {}).get(
                    "phase1_identification_conjunction"
                )
                is True
            )
        ),
        "production_precision_finite": precision_finite_passed,
        "production_precision_reconstruction": (precision_reconstruction_passed),
        "production_precision_support": precision_support_passed,
        "production_fixed_rate_frontier": production_frontier_passed,
    }
    phase1_margins = {
        endpoint: (
            None
            if ctx.cell.phase is not Phase.PHASE1
            or not isinstance(identification, dict)
            or not finite_number(identification.get(endpoint, {}).get("margin"))
            else float(identification[endpoint]["margin"])
        )
        for endpoint in ("native", "deployed")
    }
    scientific_outcome = {
        "passed": all(scientific_outcome_checks.values()),
        "checks": scientific_outcome_checks,
        "margins": {
            "support_target_abs_error": (
                threshold_map["support_target_abs_error_max"]
                - float(calibration_record["threshold_abs_error"])
            ),
            "codec_calibration_excluded_fraction": (
                threshold_map["codec_excluded_calibration_event_fraction_max"]
                - float(calibration_excluded)
            ),
            "codec_evaluation_excluded_fraction": (
                threshold_map["codec_excluded_evaluation_event_fraction_max"]
                - float(evaluation_excluded)
            ),
            "phase1_native_identification": (phase1_margins["native"]),
            "phase1_deployed_identification": (phase1_margins["deployed"]),
            "production_precision_reconstruction": (
                None
                if not precision_profile
                else float(
                    ctx.values["precision.preflight_reconstruction_relative_error_max"]
                )
                - float(precision_preflight["reconstruction_relative_error"])
            ),
            "production_precision_support_iou": (
                None
                if not precision_profile
                else float(precision_preflight["support_iou"])
                - float(ctx.values["precision.preflight_support_iou_min"])
            ),
            "production_fixed_rate_nonzero_endpoints": (
                None
                if not precision_profile
                else production_frontier_endpoint_count
                - int(ctx.values["precision.preflight_min_nonzero_rate_endpoints"])
            ),
        },
    }
    provenance = (
        evaluation.get("inputs") == expected_eval_inputs
        and evaluation.get("preparation_sha256") == prerequisites["preparation"][1]
        and training_report.get("schema") == TRAINING_REPORT_SCHEMA
        and training_report.get("cell_id") == ctx.cell.cell_id
        and training_report.get("checkpoint_sha256") == input_hashes["checkpoint"]
        and calibration_record.get("cell_id") == ctx.cell.cell_id
        and calibration_record.get("checkpoint_sha256") == input_hashes["checkpoint"]
        and calibration_record.get("codec_sha256") == input_hashes["calibration"]
        and calibration_record.get("deployment_codec_sha256")
        == input_hashes["deployment_codec"]
        and isinstance(preparation.get("implementation"), dict)
        and preparation["implementation"].get("executor_schema") == EXECUTOR_SCHEMA
    )
    expected_steps = math.ceil(
        int(ctx.values["data.train_tokens"]) / int(ctx.values["optimizer.batch_tokens"])
    )
    resource_compliance = (
        training_report.get("attempted_tokens") == ctx.values["data.train_tokens"]
        and training_report.get("step_idx") == expected_steps
        and isinstance(training_report.get("accepted_tokens"), int)
        and training_report["accepted_tokens"] == training_report["attempted_tokens"]
        and training_report.get("data_cursor")
        == {"next_token": ctx.values["data.train_tokens"], "stream": "train"}
    )
    if preparation["data"]["kind"] == "synthetic":
        ranges = preparation["data"].get("ranges", {})
        factor_calibration_range = ranges.get("factor_calibration")
        calibration_range = ranges.get("calibration")
        evaluation_range = ranges.get("evaluation")
        evaluation_stream = preparation["data"].get("evaluation_stream")
        expected_stream = (
            "confirmation"
            if ctx.values["evaluation.split"] == "confirmation"
            else "eval"
        )
        expected_eval_seed = int(
            ctx.values[
                "random.confirmation_data_seed"
                if expected_stream == "confirmation"
                else "random.eval_data_seed"
            ]
        )
        declared_seeds = {
            int(ctx.values["random.structure_seed"]),
            int(ctx.values["random.train_data_seed"]),
            int(ctx.values["random.eval_data_seed"]),
            int(ctx.values["random.confirmation_data_seed"]),
        }
        split_integrity = (
            isinstance(factor_calibration_range, list)
            and isinstance(calibration_range, list)
            and isinstance(evaluation_range, list)
            and len(factor_calibration_range)
            == len(calibration_range)
            == len(evaluation_range)
            == 2
            and int(factor_calibration_range[1]) <= int(calibration_range[0])
            and int(calibration_range[1]) <= int(evaluation_range[0])
            and int(factor_calibration_range[1]) - int(factor_calibration_range[0])
            == int(ctx.values["data.synthetic_factor_calibration_examples"])
            and int(calibration_range[1]) - int(calibration_range[0])
            == int(ctx.values["data.synthetic_calibration_examples"])
            and int(evaluation_range[1]) - int(evaluation_range[0])
            == int(
                ctx.values[
                    "data.synthetic_confirmation_examples"
                    if expected_stream == "confirmation"
                    else "data.synthetic_development_examples"
                ]
            )
            and len(declared_seeds) == 4
            and evaluation_stream == expected_stream
            and preparation["data"].get("calibration_protocol", {}).get("split_seed")
            == int(ctx.values["random.eval_data_seed"])
            and preparation["data"].get("evaluation_protocol", {}).get("split_seed")
            == expected_eval_seed
        )
    else:
        row_policy = preparation["data"].get("training_row_policy", {})
        split_roles = preparation["data"].get("splits", {})
        split_integrity = (
            preparation["data"].get("row_intervals_disjoint") is True
            and set(split_roles)
            == {
                "train",
                "normalization_fit",
                "calibration",
                "evaluation",
            }
            and len(set(split_roles.values())) == 4
            and set(preparation["data"].get("row_intervals", {})) == set(split_roles)
            and isinstance(preparation["data"].get("source_contract"), dict)
            and preparation["data"].get("store_view_policy")
            == ctx.values["data.store_view_policy"]
            and row_policy.get("kind") == "immutable_prefix_then_deterministic_replay"
            and row_policy.get("unique_tokens") == ctx.values["data.unique_tokens"]
            and row_policy.get("train_tokens") == ctx.values["data.train_tokens"]
            and preparation["data"]["bindings"]["train"].get("n_tokens", 0)
            >= ctx.values["data.unique_tokens"]
            and isinstance(preparation["data"].get("declared_split_contract"), dict)
        )
    checks = {
        "deployment_schedule_integrity": deployment_schedule_integrity,
        "finite": _finite_json(evaluation),
        "method_endpoints": method_endpoints,
        "provenance": provenance,
        "regularizer_calibration_integrity": (regularizer_calibration_integrity),
        "precision_preflight_integrity": precision_preflight_integrity,
        "resource_compliance": resource_compliance,
        "selection_score_diagnostics_integrity": (
            selection_score_diagnostics_integrity
        ),
        "scientific_endpoint_complete": scientific_endpoint_complete,
        "split_integrity": split_integrity,
    }
    failed = sorted(name for name, passed in checks.items() if passed is not True)
    if failed:
        raise CellExecutionError(
            "cell cannot qualify; failed evidence gates " + ", ".join(failed)
        )
    selection_metrics = evaluation.get("selection_metrics")
    selection_metrics_sha256 = evaluation.get("selection_metrics_sha256")
    if not isinstance(selection_metrics, dict) or not isinstance(
        selection_metrics_sha256, str
    ):
        raise CellExecutionError("evaluation lacks canonical selection metrics")
    observed_selection_hash = hashlib.sha256(
        canonical_json(selection_metrics).encode("utf-8")
    ).hexdigest()
    if selection_metrics_sha256 != observed_selection_hash or selection_metrics.get(
        "validation"
    ) != evaluation.get("validation"):
        raise CellExecutionError("evaluation selection-metrics binding mismatch")
    promotion_reasons: list[str] = []
    if ctx.values["runtime.smoke"] is not False:
        promotion_reasons.append("runtime_smoke")
    if raw.get("eligible") is not True:
        promotion_reasons.append("raw_codec_requires_unpriced_side_information")
    if ctx.cell.phase is not Phase.PHASE1 and fixed_rate.get("eligible") is not True:
        promotion_reasons.append("fixed_rate_budget_ineligible")
    if ctx.cell.phase is Phase.PHASE1 and (
        not isinstance(recovery, dict)
        or recovery.get("deployed", {}).get("shared_feature_claim_eligible") is not True
    ):
        promotion_reasons.append("synthetic_shared_feature_claim_ineligible")
    if ctx.values["qualification.promotable"] is not True:
        promotion_reasons.append("resolved_nonpromotable_cell")
    if scientific_outcome["passed"] is not True:
        promotion_reasons.append("scientific_outcome_failed")
    if ctx.cell.phase is Phase.PHASE3 or "confirmation" in ctx.cell.stage:
        if not ctx.values["selection.parent_cell_ids"]:
            promotion_reasons.append("missing_frozen_phase2_selection_decision")
    promotion_eligible = not promotion_reasons
    # A smoke cell can exercise the conditional campaign protocol without
    # becoming scientific evidence.  Preserve the cell's underlying
    # promotable intent so declared controls remain ineligible, and bind this
    # separate capability explicitly for the campaign verifier.
    selection_eligible_for_protocol_test = bool(
        ctx.values["runtime.smoke"] is True
        and ctx.values["qualification.promotable"] is True
    )
    payload = {
        "schema": QUALIFICATION_SCHEMA,
        "cell_id": ctx.cell.cell_id,
        "qualified": True,
        "checks": checks,
        "scientific_outcome": scientific_outcome,
        "inputs": input_hashes,
        "validation": evaluation["validation"],
        "qualification_profile": ctx.values["qualification.profile"],
        "thresholds_version": ctx.values["qualification.thresholds_version"],
        "thresholds": threshold_map,
        "selection_metrics": selection_metrics,
        "selection_metrics_sha256": selection_metrics_sha256,
        "selection_metrics_evaluation_sha256": input_hashes["evaluation"],
        "promotion_eligible": promotion_eligible,
        "promotion_ineligible_reasons": promotion_reasons,
        "selection_eligible_for_protocol_test": (selection_eligible_for_protocol_test),
        "selection_eligibility_mode": (
            "scientific_promotion"
            if promotion_eligible
            else "smoke_protocol_only"
            if selection_eligible_for_protocol_test
            else "none"
        ),
    }
    _write_immutable_json(ctx.qualification, payload)
    return (("qualification", ctx.qualification),)


def execute(
    ctx: _Context,
    *,
    resume: bool,
    prerequisites: Mapping[str, tuple[Path, str]] | None = None,
) -> tuple[tuple[str, Path], ...]:
    if prerequisites is None:
        prerequisites = ctx.prerequisites()
    if ctx.stage == "prepare":
        return _prepare(ctx)
    if ctx.stage == "train":
        return _train(ctx, prerequisites, resume=resume)
    if ctx.stage == "calibrate":
        return _calibrate(ctx, prerequisites)
    if ctx.stage == "evaluate":
        return _evaluate(ctx, prerequisites)
    if ctx.stage == "qualify":
        return _qualify(ctx, prerequisites)
    raise AssertionError(ctx.stage)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", type=Path, required=True)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--artifacts-out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    try:
        ctx = _Context(args.cell, args.artifacts_out, args.stage)
        prerequisites = ctx.prerequisites()
        artifacts = execute(
            ctx,
            resume=args.resume,
            prerequisites=prerequisites,
        )
        complete_artifacts = (
            *((kind, value[0]) for kind, value in prerequisites.items()),
            *artifacts,
        )
        _emit_stage_manifest(
            ctx.artifacts_out,
            cell_id=ctx.cell.cell_id,
            stage=ctx.stage,
            root=ctx.root,
            artifacts=complete_artifacts,
        )
    except (CellExecutionError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()
