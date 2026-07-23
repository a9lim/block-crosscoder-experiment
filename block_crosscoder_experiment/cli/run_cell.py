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
from contextlib import closing, contextmanager
import functools
import gc
import hashlib
import itertools
import json
import math
import os
import random
import re
import struct
import sys
import tempfile
import weakref
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import torch
import numpy as np

from block_crosscoder_experiment.activation_identity import (
    activation_content_identity,
)
from block_crosscoder_experiment.cli.data import (
    CAPTURE_MANIFEST_SCHEMA,
    TRANSFORM_ARTIFACT_SCHEMA,
    VIEW_MANIFEST_NAME,
    capture_implementation_contract,
    expected_capture_allocation,
    expected_capture_source_contract,
    validate_capture_manifest,
    validate_derived_view_manifest,
    validate_transform_artifact_manifest,
)

from block_crosscoder_experiment.campaign import (
    ARTIFACT_SCHEMA,
    Campaign,
    CampaignError,
    EVALUATION_EXECUTION_IMPLEMENTATION,
    EVALUATION_SCHEMA,
    PREPARATION_SCHEMA,
    QUALIFICATION_SCHEMA,
)
from block_crosscoder_experiment.codec import (
    Codec,
    CodecSpec,
    _RDEvaluationBatch,
    _RDEvaluationInput,
    _RDEvaluationSelection,
    _RDEvaluationSession,
    _packet_from_events,
    decode_batch,
    estimate_calibration_peak_bytes,
    fit_codec,
)
from block_crosscoder_experiment.durability import (
    durable_create,
    durable_mkdir,
    durable_replace,
)
from block_crosscoder_experiment.evaluation import (
    EvaluationModeEndpoints,
    evaluate_selector_and_shared_code_modes,
    load_trained_model,
)
from block_crosscoder_experiment.implementation import (
    CANONICAL_EXECUTOR_PROCESS_MODEL,
    CANONICAL_EXECUTOR_SCHEMA,
    cuda_execution_lock_path,
    execution_identity_sha256,
    host_cuda_execution_lock,
    implementation_identity,
    validate_implementation_identity,
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
    cuda_prefetch_batches,
    prefetch_batches,
)
from block_crosscoder_experiment.runtime_limits import (
    CODE_NORM_CUDA_IMPLEMENTATION,
    CODE_NORM_NATIVE_IMPLEMENTATION,
    DECODER_RETRACTION_CHOLESKY_QR_IMPLEMENTATION,
    DECODER_RETRACTION_HOUSEHOLDER_QR_IMPLEMENTATION,
    DECODER_RETRACTION_NOT_APPLICABLE,
    DECODER_RETRACTION_SYMMETRIC_POLAR_IMPLEMENTATION,
    DECODER_RETRACTION_SYMMETRIC_POLAR_REFERENCE_IMPLEMENTATION,
    FACTORIZED_EXECUTION_DIRECT_RANK_SPACE_IMPLEMENTATION,
    FACTORIZED_EXECUTION_FACTOR_REGULARIZERS_IMPLEMENTATION,
    FACTORIZED_EXECUTION_MATERIALIZED_REFERENCE_IMPLEMENTATION,
    FACTORIZED_EXECUTION_NOT_APPLICABLE,
    DECODED_ENERGY_EXACT_IMPLEMENTATION,
    DECODED_ENERGY_STIEFEL_CODE_NORM_IMPLEMENTATION,
    ISOLATED_LOSS_EXACT_IMPLEMENTATION,
    ISOLATED_LOSS_MAPPED_IMPLEMENTATION,
    MAP_NUCLEAR_EINSUM_REFERENCE_IMPLEMENTATION,
    MAP_NUCLEAR_GUARDED_MATMUL_IMPLEMENTATION,
    MODEL_IMPLEMENTATION_IDENTITY_FIELDS,
    RD_EVALUATION_TOKEN_CHUNK,
    SPARSE_DECODE_CUDA_IMPLEMENTATION,
    SPARSE_DECODE_DENSE_REFERENCE_IMPLEMENTATION,
    decoded_energy_code_norm_eligible,
    isolated_loss_mapped_eligible,
)
from block_crosscoder_experiment.serialization import (
    MODEL_STATE_DIGEST_CONTRACT,
    model_state_digest,
    tensor_payload_digest as _tensor_payload_digest,
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
from block_crosscoder_experiment.trainer import (
    TrainConfig,
    Trainer,
    validate_optimizer_state_config,
    validate_run_binding,
)


_NATIVE_TORCH_SAVE = torch.save
TRAINING_REPORT_SCHEMA = "bsc-training-report-v1"
EXECUTOR_SCHEMA = CANONICAL_EXECUTOR_SCHEMA
EXECUTOR_PROCESS_MODEL = CANONICAL_EXECUTOR_PROCESS_MODEL
STAGES = ("prepare", "train", "calibrate", "evaluate", "qualify")
_VERIFIED_STORE_BINDINGS: set[tuple[str, str, str, str, str]] = set()
_VERIFICATION_PROBE_BYTES = 64 * 1024
_SYNTHETIC_NORMALIZATION_CACHE: dict[
    tuple[int, torch.dtype, torch.device],
    tuple[Mapping[str, Any], torch.Tensor, torch.Tensor],
] = {}
_RECOVERY_ASSOCIATION_GROUP_CHUNK = 256
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
TIME_SHARING_BUNDLE_SCHEMA = "bsc-deployment-operating-record-bundle-v2"
FIXED_RATE_OPERATING_POLICY_SCHEMA = "bsc-fixed-rate-operating-policy-v1"
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


@dataclass(frozen=True, slots=True)
class _FileFingerprint:
    device: int
    inode: int
    size_bytes: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def from_path(cls, path: Path) -> "_FileFingerprint":
        stat = path.stat()
        return cls(
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )


class _ArtifactDigestCache:
    """Reuse digests only while the exact local file instance is unchanged."""

    def __init__(self) -> None:
        self._entries: dict[str, tuple[_FileFingerprint, str]] = {}

    def digest(self, path: Path) -> str:
        resolved = path.resolve()
        if not resolved.is_file():
            raise CellExecutionError(f"artifact disappeared: {resolved}")
        try:
            before = _FileFingerprint.from_path(resolved)
        except OSError as exc:
            raise CellExecutionError(f"cannot stat artifact {resolved}: {exc}") from exc
        cached = self._entries.get(str(resolved))
        if cached is not None and cached[0] == before:
            return cached[1]
        try:
            digest = _sha256(resolved)
            after = _FileFingerprint.from_path(resolved)
        except OSError as exc:
            raise CellExecutionError(f"cannot hash artifact {resolved}: {exc}") from exc
        if after != before:
            raise CellExecutionError(f"artifact changed while hashing: {resolved}")
        self._entries[str(resolved)] = (after, digest)
        return digest

    def verify(
        self,
        path: Path,
        *,
        sha256: str,
        size_bytes: int,
    ) -> _FileFingerprint:
        resolved = path.resolve()
        try:
            actual_size = resolved.stat().st_size
        except OSError as exc:
            raise CellExecutionError(f"cannot stat artifact {resolved}: {exc}") from exc
        if actual_size != size_bytes:
            raise CellExecutionError(f"prerequisite artifact size mismatch: {resolved}")
        if self.digest(resolved) != sha256:
            raise CellExecutionError(f"prerequisite artifact hash mismatch: {resolved}")
        return self._entries[str(resolved)][0]


def _verify_store_reader_once(
    reader: StoreReader,
    root: Path,
    split: str,
    *,
    expected_row_identity: Mapping[str, int] | None = None,
    verification_campaign_root: Path | None = None,
) -> None:
    manifest = root / split / MANIFEST_NAME
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

    manifest_before_hash = stat_record(manifest)
    manifest_sha256 = _sha256(manifest)
    manifest_after_hash = stat_record(manifest)
    if manifest_after_hash != manifest_before_hash:
        raise CellExecutionError(
            f"store split {split!r} manifest changed while it was being hashed"
        )
    live_manifest = _read_object(manifest, label=f"store split {split!r} manifest")
    if dict(live_manifest) != reader.manifest:
        raise CellExecutionError(
            f"store split {split!r} manifest changed after reader construction"
        )
    row_identity_digest = (
        "generic"
        if expected_row_identity is None
        else hashlib.sha256(
            canonical_json(dict(expected_row_identity)).encode("utf-8")
        ).hexdigest()
    )

    def content_probe(path: Path) -> dict[str, int | str]:
        size = path.stat().st_size
        length = min(size, _VERIFICATION_PROBE_BYTES)
        offset_span = size - length
        offset_seed = hashlib.sha256(
            f"{manifest_sha256}:{path.name}".encode("utf-8")
        ).digest()
        offset = (
            int.from_bytes(offset_seed[:8], "big") % (offset_span + 1)
            if offset_span
            else 0
        )
        with path.open("rb") as handle:
            handle.seek(offset)
            body = handle.read(length)
        if len(body) != length:
            raise CellExecutionError(f"short verification probe read from {path}")
        return {
            "path": str(path.resolve()),
            "offset": offset,
            "length": length,
            "sha256": hashlib.sha256(body).hexdigest(),
        }

    fingerprint = {
        "manifest": manifest_after_hash,
        "shards": [
            stat_record(root / split / str(record["file"]))
            for record in reader.manifest["shards"]
        ],
    }
    shard_paths = [
        root / split / str(record["file"]) for record in reader.manifest["shards"]
    ]
    stat_fingerprint_sha256 = hashlib.sha256(
        canonical_json(fingerprint).encode("utf-8")
    ).hexdigest()
    key = (
        str(root.resolve()),
        split,
        manifest_sha256,
        row_identity_digest,
        stat_fingerprint_sha256,
    )
    generic_key = (
        str(root.resolve()),
        split,
        manifest_sha256,
        "generic",
        stat_fingerprint_sha256,
    )
    if key in _VERIFIED_STORE_BINDINGS:
        return
    if os.environ.get("BSC_VERIFICATION_CACHE_ROOT") is not None:
        raise CellExecutionError(
            "BSC_VERIFICATION_CACHE_ROOT is unsupported for canonical execution; "
            "verification receipts are bound to BSC_CAMPAIGN_ROOT"
        )
    configured_campaign_root = (
        verification_campaign_root
        if verification_campaign_root is not None
        else (
            None
            if os.environ.get("BSC_CAMPAIGN_ROOT") is None
            else Path(str(os.environ["BSC_CAMPAIGN_ROOT"]))
        )
    )
    if configured_campaign_root is None:
        # Ad-hoc callers receive only the process-local cache. Persistent
        # receipts belong to one authenticated registered campaign.
        verified_tokens = reader.verify(expected_row_identity=expected_row_identity)
        after_fingerprint = {
            "manifest": stat_record(manifest),
            "shards": [stat_record(path) for path in shard_paths],
        }
        if verified_tokens != reader.n_tokens or after_fingerprint != fingerprint:
            raise CellExecutionError(
                f"store split {split!r} changed while it was being verified"
            )
        _VERIFIED_STORE_BINDINGS.add(key)
        _VERIFIED_STORE_BINDINGS.add(generic_key)
        return
    campaign_path = configured_campaign_root.resolve()
    cache_root = campaign_path / ".store-verification"
    if cache_root.exists() and cache_root.resolve().parent != campaign_path:
        raise CellExecutionError(
            "campaign verification cache must not escape BSC_CAMPAIGN_ROOT"
        )
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
    expected_receipt_core = {
        "schema": "bsc-store-verification-receipt-v2",
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
        if (
            isinstance(receipt, dict)
            and {
                name: receipt.get(name) for name in expected_receipt_core
            }
            == expected_receipt_core
            and isinstance(receipt.get("content_probes"), list)
        ):
            _VERIFIED_STORE_BINDINGS.add(key)
            _VERIFIED_STORE_BINDINGS.add(generic_key)
            return
    probes = [content_probe(path) for path in shard_paths]
    expected_receipt = {
        **expected_receipt_core,
        "content_probes": probes,
    }
    verified_tokens = reader.verify(expected_row_identity=expected_row_identity)
    if verified_tokens != reader.n_tokens:
        raise CellExecutionError(
            f"store verification returned {verified_tokens} rows, expected "
            f"{reader.n_tokens}"
        )
    after_fingerprint = {
        "manifest": stat_record(manifest),
        "shards": [stat_record(path) for path in shard_paths],
    }
    if after_fingerprint != fingerprint:
        raise CellExecutionError(
            f"store split {split!r} changed while it was being verified"
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
    durable_mkdir(path.parent, parents=True, exist_ok=True)
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
        durable_replace(temporary, path, file_already_synced=True)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    _write_immutable_bytes(path, _json_bytes(payload))


def _write_immutable_bytes(path: Path, body: bytes) -> None:
    durable_mkdir(path.parent, parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
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
            durable_create(temporary, path, file_already_synced=True)
            temporary = None
        except FileExistsError:
            if path.read_bytes() != body:
                raise CellExecutionError(
                    f"immutable artifact already exists with different content: {path}"
                )
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


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


def _artifact_entry(
    kind: str,
    path: Path,
    *,
    root: Path,
    digest: Callable[[Path], str] = _sha256,
) -> dict[str, Any]:
    if not path.is_file():
        raise CellExecutionError(f"stage did not produce {kind}: {path}")
    return {
        "kind": kind,
        "path": _relative(path, root),
        "sha256": digest(path),
        "size_bytes": path.stat().st_size,
    }


def _emit_stage_manifest(
    path: Path,
    *,
    cell_id: str,
    stage: str,
    root: Path,
    artifacts: Sequence[tuple[str, Path]],
    digest: Callable[[Path], str] = _sha256,
) -> None:
    entries = [
        _artifact_entry(kind, item, root=root, digest=digest)
        for kind, item in artifacts
    ]
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
    def __init__(
        self,
        cell_path: Path,
        artifacts_out: Path,
        stage: str,
        *,
        artifact_digests: _ArtifactDigestCache | None = None,
    ) -> None:
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
        self._artifact_digests = (
            artifact_digests if artifact_digests is not None else _ArtifactDigestCache()
        )
        self._prerequisite_receipts: dict[
            str,
            tuple[str, int],
        ] = {}
        try:
            self.cell_path.relative_to(self.root)
            self.artifacts_out.relative_to(self.root)
        except ValueError as exc:
            raise CellExecutionError(
                "cell and stage manifest must live inside BSC_CAMPAIGN_ROOT"
            ) from exc
        campaign = Campaign(self.root)
        self.campaign = campaign
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

    def artifact_sha256(self, path: Path) -> str:
        return self._artifact_digests.digest(path)

    def prerequisite_fingerprint(
        self,
        path: Path,
        *,
        sha256: str,
    ) -> _FileFingerprint:
        resolved = path.resolve()
        receipt = self._prerequisite_receipts.get(str(resolved))
        if receipt is None or receipt[0] != sha256:
            raise CellExecutionError(
                f"artifact lacks a matching prerequisite receipt: {resolved}"
            )
        return self._artifact_digests.verify(
            resolved,
            sha256=sha256,
            size_bytes=receipt[1],
        )

    def state(self) -> tuple[str, dict[str, dict[str, Any]]]:
        try:
            record = self.campaign.record(self.cell.cell_id)
        except CampaignError as exc:
            raise CellExecutionError(
                f"cannot replay authoritative campaign state: {exc}"
            ) from exc
        return record.state.value, {
            artifact.kind: artifact.to_dict() for artifact in record.artifacts
        }

    def verify_ref(self, raw: Mapping[str, Any]) -> Path:
        path = Path(str(raw.get("path", "")))
        if not path.is_absolute():
            path = self.root / path
        expected_size = int(raw.get("size_bytes", -1))
        expected_hash = str(raw.get("sha256", ""))
        self._artifact_digests.verify(
            path,
            sha256=expected_hash,
            size_bytes=expected_size,
        )
        self._prerequisite_receipts[str(path.resolve())] = (
            expected_hash,
            expected_size,
        )
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


@dataclass(frozen=True, slots=True)
class _RetainedArtifactKey:
    cell_id: str
    producer_stage: str
    consumer_stage: str
    artifact_kind: str
    canonical_path: str
    sha256: str
    size_bytes: int
    fingerprint: _FileFingerprint
    model_config_sha256: str


@dataclass(frozen=True, slots=True)
class _TensorSnapshotDescriptor:
    name: str
    object_id: int
    object_type: str
    storage_identity: int
    storage_data_ptr: int
    storage_nbytes: int
    tensor_data_ptr: int
    storage_offset: int
    version: int
    device: str
    dtype: str
    layout: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    requires_grad: bool
    gradient_present: bool


@dataclass(frozen=True, slots=True)
class _ModelSnapshotLineage:
    model_object_id: int
    model_type: str
    model_config_sha256: str
    model_training: bool
    threshold_cache_empty: bool
    live_state: tuple[_TensorSnapshotDescriptor, ...]
    snapshot_mapping_id: int
    snapshot_state: tuple[_TensorSnapshotDescriptor, ...]
    snapshot_digest_contract: str | None
    snapshot_sha256: str | None


@dataclass(slots=True)
class _RetainedCheckpointModel:
    key: _RetainedArtifactKey
    model: BlockCrosscoder
    metadata: dict[str, Any]
    lineage: _ModelSnapshotLineage
    released_owner_refs: dict[str, weakref.ReferenceType[Any]]


@dataclass(slots=True)
class _RetainedDeploymentConsumer:
    key: _RetainedArtifactKey
    deployment: dict[str, Any]
    model: BlockCrosscoder
    codec: Codec
    training_summary: dict[str, int]
    lineage: _ModelSnapshotLineage
    discarded_validation_model_ref: weakref.ReferenceType[BlockCrosscoder]


class _StageExecutionCache:
    """Transfer the sole model owner, never optimizer/checkpoint RNG state."""

    def __init__(self) -> None:
        self.checkpoint: _RetainedCheckpointModel | None = None
        self.deployment: _RetainedDeploymentConsumer | None = None

    @staticmethod
    def _release_unused_model_memory() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def remember_checkpoint(
        self,
        key: _RetainedArtifactKey,
        model: BlockCrosscoder,
        metadata: Mapping[str, Any],
        lineage: _ModelSnapshotLineage,
        *,
        released_owner_refs: Mapping[str, weakref.ReferenceType[Any]],
    ) -> None:
        if self.checkpoint is not None or self.deployment is not None:
            raise CellExecutionError("retained-consumer cache is not empty after train")
        self.checkpoint = _RetainedCheckpointModel(
            key,
            model,
            dict(metadata),
            lineage,
            dict(released_owner_refs),
        )

    def take_checkpoint(
        self,
        expected: _RetainedArtifactKey,
    ) -> tuple[BlockCrosscoder, dict[str, Any]] | None:
        entry = self.checkpoint
        self.checkpoint = None
        if entry is None:
            return None
        if entry.key != expected:
            del entry
            self._release_unused_model_memory()
            raise CellExecutionError(
                "retained checkpoint model binding differs from journaled artifact"
            )
        gc.collect()
        live_owners = sorted(
            name
            for name, reference in entry.released_owner_refs.items()
            if reference() is not None
        )
        if live_owners:
            del entry
            self._release_unused_model_memory()
            raise CellExecutionError(
                "released training owners remain live at calibration handoff: "
                + ", ".join(live_owners)
            )
        try:
            _assert_model_snapshot_lineage_current(
                entry.model,
                entry.lineage,
                label="retained checkpoint model",
            )
        except CellExecutionError:
            del entry
            self._release_unused_model_memory()
            raise
        return entry.model, entry.metadata

    def remember_deployment(
        self,
        key: _RetainedArtifactKey,
        deployment: Mapping[str, Any],
        model: BlockCrosscoder,
        codec: Codec,
        training_summary: Mapping[str, int],
        lineage: _ModelSnapshotLineage,
        *,
        discarded_validation_model_ref: weakref.ReferenceType[BlockCrosscoder],
    ) -> None:
        if self.checkpoint is not None or self.deployment is not None:
            raise CellExecutionError(
                "retained-consumer cache is not empty after calibration"
            )
        heavy_fields = {"model_state", "codec_payload"} & set(deployment)
        if heavy_fields:
            raise CellExecutionError(
                "volatile retained deployment contains heavy durable fields: "
                + ", ".join(sorted(heavy_fields))
            )
        if not isinstance(deployment.get("model_cfg"), dict):
            raise CellExecutionError(
                "volatile retained deployment lacks model config provenance"
            )
        self.deployment = _RetainedDeploymentConsumer(
            key,
            dict(deployment),
            model,
            codec,
            dict(training_summary),
            lineage,
            discarded_validation_model_ref,
        )

    def take_deployment(
        self,
        expected: _RetainedArtifactKey,
    ) -> tuple[dict[str, Any], BlockCrosscoder, Codec, dict[str, int]] | None:
        entry = self.deployment
        self.deployment = None
        if entry is None:
            return None
        if entry.key != expected:
            del entry
            self._release_unused_model_memory()
            raise CellExecutionError(
                "retained deployment binding differs from journaled artifact"
            )
        gc.collect()
        if entry.discarded_validation_model_ref() is not None:
            del entry
            self._release_unused_model_memory()
            raise CellExecutionError(
                "durable validation model remains live before retained evaluation"
            )
        try:
            _assert_model_snapshot_lineage_current(
                entry.model,
                entry.lineage,
                label="retained deployment model",
            )
        except CellExecutionError:
            del entry
            self._release_unused_model_memory()
            raise
        return entry.deployment, entry.model, entry.codec, entry.training_summary


def _retained_artifact_key(
    ctx: _Context,
    *,
    producer_stage: str,
    consumer_stage: str,
    artifact_kind: str,
    path: Path,
    sha256: str,
    fingerprint: _FileFingerprint,
    model_cfg: Mapping[str, Any],
) -> _RetainedArtifactKey:
    return _RetainedArtifactKey(
        cell_id=ctx.cell.cell_id,
        producer_stage=producer_stage,
        consumer_stage=consumer_stage,
        artifact_kind=artifact_kind,
        canonical_path=str(path.resolve()),
        sha256=sha256,
        size_bytes=fingerprint.size_bytes,
        fingerprint=fingerprint,
        model_config_sha256=hashlib.sha256(
            canonical_json(dict(model_cfg)).encode("utf-8")
        ).hexdigest(),
    )


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


def _resolved_runtime_device(device: torch.device | str) -> torch.device:
    """Resolve shorthand CUDA declarations for exact residency checks."""

    resolved = torch.device(device)
    if resolved.type == "cuda" and resolved.index is None:
        resolved = torch.device("cuda", torch.cuda.current_device())
    return resolved


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
    return implementation_identity()


def _implementation_identity_sha256(identity: Mapping[str, Any]) -> str:
    return execution_identity_sha256(identity)


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

    try:
        return expected_capture_source_contract(values)
    except (KeyError, TypeError, ValueError) as exc:
        raise CellExecutionError(
            f"invalid resolved capture source contract: {exc}"
        ) from exc


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

    try:
        return expected_capture_allocation(values)
    except (KeyError, TypeError, ValueError) as exc:
        raise CellExecutionError(
            f"invalid resolved capture split allocation: {exc}"
        ) from exc


def _load_capture_contract(raw_root: Path, values: Mapping[str, Any]) -> dict[str, Any]:
    capture_path = raw_root / "capture.json"
    if not capture_path.is_file():
        raise CellExecutionError(
            f"raw activation store lacks immutable source contract {capture_path}"
        )
    capture = _read_object(capture_path, label="capture source contract")
    try:
        capture_binding = validate_capture_manifest(capture)
    except ValueError as exc:
        raise CellExecutionError(
            f"capture source contract is unauthenticated: {exc}"
        ) from exc
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
    capture_implementation = capture.get("capture_implementation")
    expected_implementation = capture_implementation_contract()
    if not isinstance(capture_implementation, dict):
        raise CellExecutionError("capture implementation contract is malformed")
    observed_core = {
        key: capture_implementation.get(key) for key in expected_implementation
    }
    if observed_core != expected_implementation:
        raise CellExecutionError(
            "capture implementation differs from the current reviewed producer: "
            + canonical_json(
                {"expected": expected_implementation, "observed": observed_core}
            )
        )
    runtime = capture_implementation.get("runtime")
    if (
        not isinstance(runtime, dict)
        or set(runtime)
        != {"requested_device", "torch_cuda_version", "cuda_device_name"}
        or not isinstance(runtime.get("requested_device"), str)
    ):
        raise CellExecutionError("capture runtime provenance is malformed")
    if values["runtime.smoke"] is False and not runtime["requested_device"].startswith(
        "cuda"
    ):
        raise CellExecutionError("scientific activation capture must run on CUDA")
    split_order, split_plan = _expected_capture_allocation(values)
    if capture.get("schema") != CAPTURE_MANIFEST_SCHEMA:
        raise CellExecutionError("capture source contract has an unknown schema")
    if capture.get("split_order") != list(split_order):
        raise CellExecutionError(
            "capture split order differs from the canonical cell allocation"
        )
    if capture.get("split_plan") != split_plan or set(capture.get("splits", {})) != set(
        split_plan
    ):
        raise CellExecutionError(
            "capture split allocation differs from data.split_sizes"
        )
    for split in split_order:
        manifest_path = raw_root / split / MANIFEST_NAME
        if not manifest_path.is_file():
            raise CellExecutionError(
                f"raw capture lacks authenticated split manifest {manifest_path}"
            )
        reader = StoreReader(raw_root, split)
        allocation = split_plan[split]
        expected_record = {
            "allocation": dict(allocation),
            "manifest_file_sha256": _sha256(manifest_path),
            "manifest_sha256": reader.manifest.get("manifest_sha256"),
            "content_stream_sha256": reader.manifest.get(
                "content_stream_sha256"
            ),
            "row_stream_sha256": reader.manifest.get("row_stream_sha256"),
            "n_tokens": reader.n_tokens,
            "sites": list(reader.sites),
            "site_dims": list(reader.site_dims),
            "d_model": reader.d_model,
            "row_id_width": int(reader.manifest.get("row_id_width", -1)),
            "whitener_hash": reader.whitener_hash,
        }
        if capture["splits"].get(split) != expected_record:
            raise CellExecutionError(
                f"raw split {split!r} differs from its authenticated capture record"
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
        "capture_binding": capture_binding,
        "capture_implementation": capture_implementation,
        "capture_content_sha256": capture["capture_content_sha256"],
        "splits": capture["splits"],
        "capture": capture,
    }


def _validate_derived_root_envelope(
    root: Path,
    *,
    source_contract: Mapping[str, Any],
    transform: Whitener,
) -> tuple[str, str]:
    """Replay the immutable derived-root envelope without rescanning shards."""

    manifest_path = root / VIEW_MANIFEST_NAME
    manifest = _read_object(manifest_path, label="derived-view root manifest")
    try:
        validated = validate_derived_view_manifest(manifest)
    except ValueError as exc:
        raise CellExecutionError(
            f"derived-view root manifest is unauthenticated: {exc}"
        ) from exc
    capture = source_contract["capture"]
    split_order = tuple(source_contract["split_order"])
    expected_entries = set(split_order) | {"whitener.pt", VIEW_MANIFEST_NAME}
    actual_entries = {path.name for path in root.iterdir()}
    if actual_entries != expected_entries:
        raise CellExecutionError(
            "derived-view root entries differ from its authenticated envelope"
        )
    if (
        validated.get("mode") != transform.mode
        or validated.get("transform_hash") != transform.hash
        or validated.get("whitener_sha256") != _sha256(root / "whitener.pt")
        or validated.get("source_capture_sha256") != source_contract["sha256"]
        or validated.get("source_capture") != capture
        or validated.get("source_hash") != source_contract["source_hash"]
        or validated.get("capture_binding_sha256")
        != source_contract["capture_binding_sha256"]
        or validated.get("split_order") != list(split_order)
    ):
        raise CellExecutionError(
            "derived-view root envelope differs from its capture or transform"
        )
    for split in split_order:
        reader = StoreReader(root, split)
        source_record = capture["splits"][split]
        expected_record = {
            "allocation": dict(capture["split_plan"][split]),
            "manifest_sha256": reader.manifest.get("manifest_sha256"),
            "content_stream_sha256": reader.manifest.get(
                "content_stream_sha256"
            ),
            "row_stream_sha256": reader.manifest.get("row_stream_sha256"),
            "n_tokens": reader.n_tokens,
            "source_manifest_file_sha256": source_record[
                "manifest_file_sha256"
            ],
            "source_manifest_sha256": source_record["manifest_sha256"],
            "source_content_stream_sha256": source_record[
                "content_stream_sha256"
            ],
            "source_row_stream_sha256": source_record["row_stream_sha256"],
        }
        if validated["splits"].get(split) != expected_record:
            raise CellExecutionError(
                f"derived split {split!r} differs from its authenticated root envelope"
            )
    return str(validated["view_manifest_sha256"]), _sha256(manifest_path)


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
    verification_campaign_root: Path | None = None,
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
                verification_campaign_root=verification_campaign_root,
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
        "source_capture_manifest_sha256": hashlib.sha256(
            canonical_json(source_contract["capture"]).encode("utf-8")
        ).hexdigest(),
        "source_hash": source_hash,
        "source_fit_manifest_file_sha256": source_contract["splits"][fit_split][
            "manifest_file_sha256"
        ],
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
    try:
        validate_transform_artifact_manifest(transform_manifest)
    except ValueError as exc:
        raise CellExecutionError(
            f"Phase-3 transform manifest is unauthenticated: {exc}"
        ) from exc
    expected_transform_manifest = {
        "schema": TRANSFORM_ARTIFACT_SCHEMA,
        "mode": normalization,
        "transform_hash": transform.hash,
        "whitener_sha256": _sha256(transform_path),
        **expected_meta,
        "source_capture": source_contract["capture"],
        "source_fit_manifest_file_sha256": source_contract["splits"][fit_split][
            "manifest_file_sha256"
        ],
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
        "raw_declared_split_contract": declared_split_contract,
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
    assert transform is not None
    view_manifest_sha256, view_manifest_file_sha256 = (
        _validate_derived_root_envelope(
            root,
            source_contract=source_contract,
            transform=transform,
        )
    )
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
            "transform_sha256": _sha256(root / "whitener.pt"),
            "transform_hash": transform.hash,
            "view_manifest_sha256": view_manifest_sha256,
            "view_manifest_file_sha256": view_manifest_file_sha256,
        },
    }


def _synthetic_preparation_data(cell: CellSpec) -> dict[str, Any]:
    """Materialize the one exact Phase-1 data payload from its cell."""

    values = cell.decision_map
    train = _synthetic_dataset(cell, "train")
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
    calibration_dataset = _synthetic_dataset(cell, "eval")
    evaluation = _synthetic_dataset(cell, evaluation_stream)
    ranges = {
        str(name): [int(start), int(stop)]
        for name, start, stop in values["data.synthetic_split_ranges"]
    }
    evaluation_role = (
        "development" if evaluation_split == "synthetic_test" else "confirmation"
    )
    return {
        "kind": "synthetic",
        "source_contract": _synthetic_source_contract(values),
        "train_protocol": train.protocol_dict(),
        "calibration_protocol": calibration_dataset.protocol_dict(),
        "evaluation_protocol": evaluation.protocol_dict(),
        "evaluation_stream": evaluation_stream,
        "normalization": _normalization_record(train, values),
        "ranges": {
            "train": [0, int(values["data.train_tokens"])],
            "factor_calibration": ranges["factor_calibration"],
            "calibration": ranges["calibration"],
            "evaluation": ranges[evaluation_role],
        },
        "evaluation_role": evaluation_role,
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
    try:
        validate_implementation_identity(
            implementation,
            scientific=values["runtime.smoke"] is False,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CellExecutionError(str(exc)) from exc
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
    data = (
        _synthetic_preparation_data(ctx.cell)
        if ctx.cell.phase is Phase.PHASE1
        else {"kind": "activation_store", **_resolve_real_store(values)}
    )
    try:
        data_identity = (
            None if data["kind"] == "synthetic" else activation_content_identity(data)
        )
    except ValueError as exc:
        raise CellExecutionError(str(exc)) from exc
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
        "data_identity": data_identity,
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
        "implementation_sha256": _implementation_identity_sha256(implementation),
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
        "mean_squared": "mean_squared",
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
    site_rank = (
        None if values["model.site_rank"] is None else int(values["model.site_rank"])
    )
    code_norm_implementation = str(values["implementation.code_norm_implementation"])
    if code_norm_implementation not in {
        CODE_NORM_CUDA_IMPLEMENTATION,
        CODE_NORM_NATIVE_IMPLEMENTATION,
    }:
        raise CellExecutionError("unknown code-norm implementation identity")
    factorized_execution_implementation = str(
        values["implementation.factorized_execution_implementation"]
    )
    sparse_decode_implementation = str(
        values["implementation.sparse_decode_implementation"]
    )
    if sparse_decode_implementation not in {
        SPARSE_DECODE_CUDA_IMPLEMENTATION,
        SPARSE_DECODE_DENSE_REFERENCE_IMPLEMENTATION,
    }:
        raise CellExecutionError("unknown sparse-decode implementation identity")
    map_nuclear_implementation = str(
        values["implementation.map_nuclear_implementation"]
    )
    if map_nuclear_implementation not in {
        MAP_NUCLEAR_GUARDED_MATMUL_IMPLEMENTATION,
        MAP_NUCLEAR_EINSUM_REFERENCE_IMPLEMENTATION,
    }:
        raise CellExecutionError("unknown map-nuclear implementation identity")
    known_factorized_implementations = {
        FACTORIZED_EXECUTION_DIRECT_RANK_SPACE_IMPLEMENTATION,
        FACTORIZED_EXECUTION_FACTOR_REGULARIZERS_IMPLEMENTATION,
        FACTORIZED_EXECUTION_MATERIALIZED_REFERENCE_IMPLEMENTATION,
        FACTORIZED_EXECUTION_NOT_APPLICABLE,
    }
    if factorized_execution_implementation not in known_factorized_implementations:
        raise CellExecutionError("unknown factorized-execution implementation identity")
    allowed_factorized_implementations = (
        {
            FACTORIZED_EXECUTION_DIRECT_RANK_SPACE_IMPLEMENTATION,
            FACTORIZED_EXECUTION_FACTOR_REGULARIZERS_IMPLEMENTATION,
            FACTORIZED_EXECUTION_MATERIALIZED_REFERENCE_IMPLEMENTATION,
        }
        if site_rank is not None
        else {FACTORIZED_EXECUTION_NOT_APPLICABLE}
    )
    if factorized_execution_implementation not in allowed_factorized_implementations:
        raise CellExecutionError(
            "factorized-execution implementation violates its carrier predicate"
        )
    factor_regularizer_eligible = site_rank in {1, 2} and regularizer in {
        "map_nuclear",
        "decoder_nuclear",
    }
    if factorized_execution_implementation in {
        FACTORIZED_EXECUTION_DIRECT_RANK_SPACE_IMPLEMENTATION,
        FACTORIZED_EXECUTION_FACTOR_REGULARIZERS_IMPLEMENTATION,
    } and (
        factorized_execution_implementation
        != (
            FACTORIZED_EXECUTION_FACTOR_REGULARIZERS_IMPLEMENTATION
            if factor_regularizer_eligible
            else FACTORIZED_EXECUTION_DIRECT_RANK_SPACE_IMPLEMENTATION
        )
    ):
        raise CellExecutionError(
            "factorized-execution implementation violates its objective predicate"
        )
    selection_score = str(values["model.selection_score"])
    decoded_energy_implementation = str(
        values["implementation.decoded_energy_implementation"]
    )
    eligible_score_specialization = decoded_energy_code_norm_eligible(
        selection_score=selection_score,
        decoder_constraint=constraint,
        training_selector=selection,
        site_rank=site_rank,
        retract_every=int(values["optimizer.retract_every_steps"]),
    )
    if decoded_energy_implementation not in {
        DECODED_ENERGY_EXACT_IMPLEMENTATION,
        DECODED_ENERGY_STIEFEL_CODE_NORM_IMPLEMENTATION,
    }:
        raise CellExecutionError("unknown decoded-energy implementation identity")
    if (
        decoded_energy_implementation == DECODED_ENERGY_STIEFEL_CODE_NORM_IMPLEMENTATION
        and not eligible_score_specialization
    ):
        raise CellExecutionError(
            "stiefel decoded-energy implementation violates its carrier predicate"
        )
    isolated_loss_implementation = str(
        values["implementation.isolated_loss_decrease_implementation"]
    )
    eligible_isolated_loss_specialization = isolated_loss_mapped_eligible(
        selection_score=selection_score,
        decoder_constraint=constraint,
        decoder_bias=bool(values["model.decoder_bias"]),
        reconstruction_loss=reconstruction_loss,
    )
    if isolated_loss_implementation not in {
        ISOLATED_LOSS_EXACT_IMPLEMENTATION,
        ISOLATED_LOSS_MAPPED_IMPLEMENTATION,
    }:
        raise CellExecutionError("unknown isolated-loss implementation identity")
    if (
        isolated_loss_implementation == ISOLATED_LOSS_MAPPED_IMPLEMENTATION
        and not eligible_isolated_loss_specialization
    ):
        raise CellExecutionError(
            "mapped isolated-loss implementation violates its carrier predicate"
        )
    decoder_retraction_implementation = str(
        values["implementation.decoder_retraction_implementation"]
    )
    known_retraction_implementations = {
        DECODER_RETRACTION_CHOLESKY_QR_IMPLEMENTATION,
        DECODER_RETRACTION_HOUSEHOLDER_QR_IMPLEMENTATION,
        DECODER_RETRACTION_SYMMETRIC_POLAR_IMPLEMENTATION,
        DECODER_RETRACTION_SYMMETRIC_POLAR_REFERENCE_IMPLEMENTATION,
        DECODER_RETRACTION_NOT_APPLICABLE,
    }
    if decoder_retraction_implementation not in known_retraction_implementations:
        raise CellExecutionError("unknown decoder-retraction implementation identity")
    allowed_retraction_implementations = {
        "qr": {
            DECODER_RETRACTION_CHOLESKY_QR_IMPLEMENTATION,
            DECODER_RETRACTION_HOUSEHOLDER_QR_IMPLEMENTATION,
        },
        "gram": {
            DECODER_RETRACTION_SYMMETRIC_POLAR_IMPLEMENTATION,
            DECODER_RETRACTION_SYMMETRIC_POLAR_REFERENCE_IMPLEMENTATION,
        },
    }.get(constraint, {DECODER_RETRACTION_NOT_APPLICABLE})
    if decoder_retraction_implementation not in allowed_retraction_implementations:
        raise CellExecutionError(
            "decoder-retraction implementation violates its carrier predicate"
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
        selection_score=selection_score,
        code_norm_implementation=code_norm_implementation,
        decoded_energy_implementation=decoded_energy_implementation,
        isolated_loss_decrease_implementation=isolated_loss_implementation,
        decoder_retraction_implementation=decoder_retraction_implementation,
        factorized_execution_implementation=factorized_execution_implementation,
        sparse_decode_implementation=sparse_decode_implementation,
        map_nuclear_implementation=map_nuclear_implementation,
        selector_tie_break=str(values["model.selector_tie_break"]),
        site_rank=site_rank,
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
        values["model.selector"] == "learned_group_threshold"
        and values["objective.auxiliary"] == "runner_up_blocks"
    ):
        raise CellExecutionError(
            "Appendix-D runner-ups are undefined for learned Group-Lasso "
            "support; use the primary Group-Lasso recipe or a hard-TopK carrier"
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


_PREPARATION_CONTRACT_CACHE: set[tuple[str, str]] = set()


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise CellExecutionError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _validate_real_preparation_data(
    cell: CellSpec,
    data: Mapping[str, Any],
    *,
    verification_campaign_root: Path | None = None,
) -> None:
    """Replay the current real-data topology against immutable cell decisions."""

    expected_keys = {
        "kind",
        "root",
        "splits",
        "bindings",
        "row_intervals",
        "row_intervals_disjoint",
        "declared_split_contract",
        "raw_root",
        "raw_bindings",
        "raw_declared_split_contract",
        "source_contract",
        "store_view_policy",
        "training_row_policy",
        "normalization",
    }
    if set(data) != expected_keys or data.get("kind") != "activation_store":
        raise CellExecutionError("real preparation data uses a noncanonical field set")
    values = cell.decision_map
    expected_source = _expected_real_source_contract(values)
    expected_order, expected_plan = _expected_capture_allocation(values)
    source = data.get("source_contract")
    source_keys = {
        "path",
        "sha256",
        "source_hash",
        "source",
        "declared",
        "split_order",
        "split_plan",
        "capture_binding_sha256",
        "capture_binding",
        "capture_implementation",
        "capture_content_sha256",
        "splits",
        "capture",
    }
    if not isinstance(source, Mapping) or set(source) != source_keys:
        raise CellExecutionError("real preparation source contract is noncanonical")
    expected_source_hash = hashlib.sha256(
        canonical_json(expected_source).encode("utf-8")
    ).hexdigest()
    if (
        source.get("source") != expected_source
        or source.get("declared") != expected_source
        or source.get("source_hash") != expected_source_hash
        or source.get("split_order") != list(expected_order)
        or source.get("split_plan") != expected_plan
    ):
        raise CellExecutionError(
            "real preparation source/corpus/model/split contract differs from the cell"
        )
    for name in (
        "sha256",
        "source_hash",
        "capture_binding_sha256",
        "capture_content_sha256",
    ):
        _require_sha256(source.get(name), label=f"capture {name}")
    capture = source.get("capture")
    try:
        live_binding = validate_capture_manifest(capture)
    except ValueError as exc:
        raise CellExecutionError(
            f"prepared capture manifest is invalid: {exc}"
        ) from exc
    if (
        source.get("capture_binding") != live_binding
        or source.get("capture_binding_sha256") != capture.get("capture_binding_sha256")
        or source.get("capture_content_sha256") != capture.get("capture_content_sha256")
    ):
        raise CellExecutionError("prepared capture binding is internally inconsistent")
    raw_root = Path(str(data.get("raw_root", ""))).resolve()
    source_path = Path(str(source.get("path", ""))).resolve()
    if (
        source_path != raw_root / "capture.json"
        or _sha256(source_path) != source["sha256"]
    ):
        raise CellExecutionError("prepared capture file/path binding changed")
    live_source = _load_capture_contract(raw_root, values)
    if canonical_json(live_source) != canonical_json(source):
        raise CellExecutionError(
            "prepared capture contract differs from the authenticated live capture"
        )

    normalization = data.get("normalization")
    view_policy = str(values["data.store_view_policy"])
    normalization_keys = (
        {
            "mode",
            "application",
            "transform_path",
            "transform_sha256",
            "transform_hash",
            "transform_manifest",
            "transform_manifest_sha256",
            "selected_site_indices",
            "source_capture_sha256",
            "source_fit_manifest",
            "source_fit_manifest_file_sha256",
            "source_fit_manifest_sha256",
            "source_fit_row_stream_sha256",
            "source_fit_requested_tokens",
        }
        if view_policy == "single_bf16_raw_view_on_the_fly_invertible_normalization"
        else {
            "mode",
            "transform_sha256",
            "transform_hash",
            "view_manifest_sha256",
            "view_manifest_file_sha256",
        }
    )
    if (
        not isinstance(normalization, Mapping)
        or set(normalization) != normalization_keys
        or normalization.get("mode") != values["data.normalization"]
        or data.get("store_view_policy") != view_policy
    ):
        raise CellExecutionError(
            "real preparation normalization/view policy differs from the cell"
        )
    _require_sha256(
        normalization.get("transform_sha256"), label="normalization transform file"
    )
    _require_sha256(
        normalization.get("transform_hash"), label="normalization transform content"
    )
    if view_policy == "single_bf16_raw_view_on_the_fly_invertible_normalization":
        if (
            normalization.get("application") != "on_the_fly"
            or normalization.get("selected_site_indices")
            != list(
                tuple(str(item) for item in values["data.store_sites"]).index(str(name))
                for name in values["data.sites"]
            )
            or normalization.get("source_capture_sha256") != source["sha256"]
            or normalization.get("source_fit_requested_tokens")
            != int(values["data.normalization_fit_count"])
        ):
            raise CellExecutionError(
                "on-the-fly normalization lineage differs from the cell"
            )
        for name in (
            "transform_manifest_sha256",
            "source_capture_sha256",
            "source_fit_manifest_file_sha256",
            "source_fit_manifest_sha256",
            "source_fit_row_stream_sha256",
        ):
            _require_sha256(normalization.get(name), label=f"normalization {name}")
    else:
        _require_sha256(
            normalization.get("view_manifest_sha256"),
            label="derived-view manifest content",
        )
        _require_sha256(
            normalization.get("view_manifest_file_sha256"),
            label="derived-view manifest file",
        )

    store_sites = tuple(str(item) for item in values["data.store_sites"])
    selected_names = tuple(str(item) for item in values["data.sites"])
    try:
        selected_indices = tuple(store_sites.index(name) for name in selected_names)
    except ValueError as exc:
        raise CellExecutionError(
            "cell site selection is absent from its store axis"
        ) from exc
    expected_roles = {
        "train": "train",
        "calibration": str(values["evaluation.calibration_split"]),
        "evaluation": str(values["evaluation.split"]),
        "normalization_fit": str(values["data.normalization_fit_split"]),
    }
    splits = data.get("splits")
    bindings = data.get("bindings")
    raw_bindings = data.get("raw_bindings")
    intervals = data.get("row_intervals")
    if (
        splits != expected_roles
        or not isinstance(bindings, Mapping)
        or not isinstance(raw_bindings, Mapping)
        or not isinstance(intervals, Mapping)
        or set(bindings) != set(expected_roles)
        or set(raw_bindings) != set(expected_roles)
        or set(intervals) != set(expected_roles)
        or data.get("row_intervals_disjoint") is not True
    ):
        raise CellExecutionError(
            "real preparation role/split topology differs from the cell"
        )

    root = Path(str(data.get("root", ""))).resolve()
    if str(root) != data.get("root") or str(raw_root) != data.get("raw_root"):
        raise CellExecutionError("real preparation store roots are not canonical")
    if view_policy == "content_addressed_derived_view":
        transform_path = root / "whitener.pt"
        try:
            transform = Whitener.load(transform_path)
        except Exception as exc:  # noqa: BLE001
            raise CellExecutionError(
                f"cannot verify prepared derived transform: {exc}"
            ) from exc
        if (
            transform.mode != values["data.normalization"]
            or transform.hash != normalization["transform_hash"]
            or _sha256(transform_path) != normalization["transform_sha256"]
            or transform.n_fit_tokens
            != int(values["data.normalization_fit_count"])
        ):
            raise CellExecutionError(
                "prepared derived transform differs from the cell or data binding"
            )
        view_manifest_sha256, view_manifest_file_sha256 = (
            _validate_derived_root_envelope(
                root,
                source_contract=source,
                transform=transform,
            )
        )
        if (
            normalization["view_manifest_sha256"] != view_manifest_sha256
            or normalization["view_manifest_file_sha256"]
            != view_manifest_file_sha256
        ):
            raise CellExecutionError(
                "prepared derived-view envelope digest is stale"
            )
        view_whitener_hash = transform.hash
    else:
        if root != raw_root:
            raise CellExecutionError(
                "single-view on-the-fly normalization must consume the raw root"
            )
        transform_path = Path(str(normalization["transform_path"])).resolve()
        transform_manifest_path = Path(
            str(normalization["transform_manifest"])
        ).resolve()
        try:
            transform = Whitener.load(transform_path)
        except Exception as exc:  # noqa: BLE001
            raise CellExecutionError(
                f"cannot verify prepared transform-only artifact: {exc}"
            ) from exc
        transform_manifest = _read_object(
            transform_manifest_path,
            label="prepared transform-only manifest",
        )
        try:
            validate_transform_artifact_manifest(transform_manifest)
        except ValueError as exc:
            raise CellExecutionError(
                f"prepared transform-only manifest is unauthenticated: {exc}"
            ) from exc
        if (
            transform_path.name != "whitener.pt"
            or transform_manifest_path != transform_path.with_name("transform.json")
            or transform.mode != values["data.normalization"]
            or transform.hash != normalization["transform_hash"]
            or _sha256(transform_path) != normalization["transform_sha256"]
            or _sha256(transform_manifest_path)
            != normalization["transform_manifest_sha256"]
            or transform.n_fit_tokens
            != int(values["data.normalization_fit_count"])
            or transform_manifest.get("transform_hash") != transform.hash
            or transform_manifest.get("whitener_sha256")
            != normalization["transform_sha256"]
            or transform_manifest.get("source_capture") != source["capture"]
            or transform_manifest.get("source_capture_sha256") != source["sha256"]
            or Path(str(transform_manifest.get("source_raw_root", ""))).resolve()
            != raw_root
        ):
            raise CellExecutionError(
                "prepared transform-only artifact differs from its source or cell"
            )
        view_whitener_hash = f"raw:{source['source_hash']}"

    expected_axis = tuple(range(len(store_sites)))
    expected_raw_declared = _verify_declared_split_contract(
        raw_root,
        values,
        capture_contract=source,
        expected_store_axis=expected_axis,
        expected_whitener_hash=f"raw:{source['source_hash']}",
        verification_campaign_root=verification_campaign_root,
    )
    expected_view_declared = (
        expected_raw_declared
        if root == raw_root
        else _verify_declared_split_contract(
            root,
            values,
            capture_contract=source,
            expected_store_axis=expected_axis,
            expected_whitener_hash=view_whitener_hash,
            verification_campaign_root=verification_campaign_root,
        )
    )
    if (
        data.get("raw_declared_split_contract") != expected_raw_declared
        or data.get("declared_split_contract") != expected_view_declared
    ):
        raise CellExecutionError(
            "prepared declared split contracts differ from the live stores"
        )

    binding_keys = {
        "split",
        "manifest",
        "manifest_sha256",
        "n_tokens",
        "row_stream_sha256",
        "content_stream_sha256",
        "selected_site_indices",
        "selected_site_names",
    }
    for role, split in expected_roles.items():
        view = bindings[role]
        raw = raw_bindings[role]
        view_manifest_path = root / split / MANIFEST_NAME
        raw_manifest_path = raw_root / split / MANIFEST_NAME
        raw_reader = StoreReader(raw_root, split, sites=selected_indices)
        view_reader = (
            raw_reader
            if root == raw_root
            else StoreReader(root, split, sites=selected_indices)
        )
        expected_view_binding = {
            "split": split,
            "manifest": str(view_manifest_path),
            "manifest_sha256": _sha256(view_manifest_path),
            "n_tokens": view_reader.n_tokens,
            "row_stream_sha256": view_reader.manifest.get("row_stream_sha256"),
            "content_stream_sha256": view_reader.manifest.get(
                "content_stream_sha256"
            ),
            "selected_site_indices": list(selected_indices),
            "selected_site_names": list(selected_names),
        }
        expected_raw_binding = {
            "split": split,
            "manifest": str(raw_manifest_path),
            "manifest_sha256": _sha256(raw_manifest_path),
            "n_tokens": raw_reader.n_tokens,
            "row_stream_sha256": raw_reader.manifest.get("row_stream_sha256"),
            "content_stream_sha256": raw_reader.manifest.get(
                "content_stream_sha256"
            ),
            "selected_site_indices": list(selected_indices),
            "selected_site_names": list(selected_names),
        }
        if (
            not isinstance(view, Mapping)
            or not isinstance(raw, Mapping)
            or set(view) != binding_keys
            or set(raw) != binding_keys
            or dict(view) != expected_view_binding
            or dict(raw) != expected_raw_binding
            or view.get("split") != split
            or raw.get("split") != split
            or view.get("selected_site_indices") != list(selected_indices)
            or raw.get("selected_site_indices") != list(selected_indices)
            or view.get("selected_site_names") != list(selected_names)
            or raw.get("selected_site_names") != list(selected_names)
            or view.get("n_tokens") != raw.get("n_tokens")
            or view.get("row_stream_sha256") != raw.get("row_stream_sha256")
        ):
            raise CellExecutionError(
                f"real preparation {role} raw/view binding is inconsistent"
            )
        for prefix, binding in (("view", view), ("raw", raw)):
            for name in (
                "manifest_sha256",
                "row_stream_sha256",
                "content_stream_sha256",
            ):
                _require_sha256(binding.get(name), label=f"{role} {prefix} {name}")
        interval = intervals[role]
        expected_interval = _row_interval(raw_reader)
        if (
            not isinstance(interval, Mapping)
            or set(interval) != {"first", "last", "count"}
            or interval.get("count") != view.get("n_tokens")
            or dict(interval) != expected_interval
            or not all(
                isinstance(item, list)
                and len(item) >= 2
                and all(type(value) is int for value in item)
                for item in (interval.get("first"), interval.get("last"))
            )
        ):
            raise CellExecutionError(
                f"real preparation {role} row interval is malformed"
            )

    if not _intervals_are_disjoint(dict(intervals)):
        raise CellExecutionError(
            "prepared train/normalization/calibration/evaluation rows overlap"
        )

    declared_keys = {
        "requested_tokens",
        "actual_tokens",
        "manifest_sha256",
        "row_stream_sha256",
        "content_stream_sha256",
    }
    declared_view = data.get("declared_split_contract")
    declared_raw = data.get("raw_declared_split_contract")
    if (
        not isinstance(declared_view, Mapping)
        or not isinstance(declared_raw, Mapping)
        or set(declared_view) != set(expected_order)
        or set(declared_raw) != set(expected_order)
    ):
        raise CellExecutionError("real preparation declared split grid is incomplete")
    requested_by_split = {
        str(name): int(count) for name, count in values["data.split_sizes"]
    }
    for split in expected_order:
        view = declared_view[split]
        raw = declared_raw[split]
        if (
            not isinstance(view, Mapping)
            or not isinstance(raw, Mapping)
            or set(view) != declared_keys
            or set(raw) != declared_keys
            or view.get("requested_tokens") != requested_by_split[split]
            or raw.get("requested_tokens") != requested_by_split[split]
            or view.get("actual_tokens") != raw.get("actual_tokens")
            or view.get("row_stream_sha256") != raw.get("row_stream_sha256")
        ):
            raise CellExecutionError(
                f"real preparation declared split {split!r} is inconsistent"
            )
        for record in (view, raw):
            if (
                type(record.get("actual_tokens")) is not int
                or record["actual_tokens"] < requested_by_split[split]
            ):
                raise CellExecutionError(
                    f"real preparation split {split!r} is undersized"
                )
            for name in (
                "manifest_sha256",
                "row_stream_sha256",
                "content_stream_sha256",
            ):
                _require_sha256(record.get(name), label=f"{split} {name}")

    if data.get("training_row_policy") != {
        "kind": "immutable_prefix_then_deterministic_replay",
        "unique_tokens": int(values["data.unique_tokens"]),
        "train_tokens": int(values["data.train_tokens"]),
    }:
        raise CellExecutionError(
            "real preparation training-row policy differs from cell"
        )
    try:
        expected_identity = activation_content_identity(data)
    except ValueError as exc:
        raise CellExecutionError(str(exc)) from exc
    if data.get("kind") != "activation_store" or expected_identity["view_key"] != str(
        values["data.normalization"]
    ):
        raise CellExecutionError("real preparation activation identity role is invalid")


def _validate_preparation_contract(
    cell: CellSpec,
    payload: Mapping[str, Any],
    *,
    cell_manifest_sha256: str,
    verification_campaign_root: Path | None = None,
) -> None:
    expected_keys = {
        "schema",
        "cell_id",
        "cell_manifest_sha256",
        "phase",
        "stage_family",
        "recipe_name",
        "recipe_id",
        "seed",
        "decisions_sha256",
        "data",
        "data_identity",
        "runtime",
        "random",
        "selection",
        "implementation",
        "implementation_sha256",
    }
    if set(payload) != expected_keys:
        raise CellExecutionError("preparation artifact uses a noncanonical field set")
    values = cell.decision_map
    expected_static = {
        "cell_id": cell.cell_id,
        "cell_manifest_sha256": cell_manifest_sha256,
        "phase": cell.phase.value,
        "stage_family": cell.stage,
        "recipe_name": cell.recipe_name,
        "recipe_id": cell.recipe_id,
        "seed": cell.seed,
        "decisions_sha256": hashlib.sha256(
            canonical_json(cell.content_payload()).encode("utf-8")
        ).hexdigest(),
    }
    if any(payload.get(name) != expected for name, expected in expected_static.items()):
        raise CellExecutionError("preparation artifact differs from its cell manifest")
    expected_runtime = {
        "smoke": values["runtime.smoke"],
        "device": _declared_device(values),
        "torch_version": payload.get("implementation", {}).get("torch"),
    }
    expected_random = {
        name.removeprefix("random."): values[name]
        for name in (
            "random.model_seed",
            "random.structure_seed",
            "random.train_data_seed",
            "random.eval_data_seed",
            "random.confirmation_data_seed",
        )
    }
    expected_selection = {
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
    }
    if (
        payload.get("runtime") != expected_runtime
        or payload.get("random") != expected_random
        or payload.get("selection") != expected_selection
    ):
        raise CellExecutionError(
            "preparation runtime/random/selection binding is stale"
        )
    data = payload.get("data")
    if not isinstance(data, Mapping):
        raise CellExecutionError("preparation data payload is missing")
    if cell.phase is Phase.PHASE1:
        if payload.get("data_identity") is not None:
            raise CellExecutionError(
                "synthetic preparation cannot bind activation data"
            )
        if dict(data) != _synthetic_preparation_data(cell):
            raise CellExecutionError(
                "synthetic preparation data differs from the registered cell"
            )
    else:
        _validate_real_preparation_data(
            cell,
            data,
            verification_campaign_root=verification_campaign_root,
        )
        try:
            expected_identity = activation_content_identity(data)
        except ValueError as exc:
            raise CellExecutionError(str(exc)) from exc
        if payload.get("data_identity") != expected_identity:
            raise CellExecutionError(
                "real preparation identity differs from its validated data"
            )


def _load_preparation(path: Path, ctx: _Context) -> dict[str, Any]:
    payload = _read_object(path, label="preparation artifact")
    if (
        payload.get("schema") != PREPARATION_SCHEMA
        or payload.get("cell_id") != ctx.cell.cell_id
    ):
        raise CellExecutionError("preparation artifact binding mismatch")
    current_implementation = _implementation_identity()
    if payload.get("implementation") != current_implementation:
        raise CellExecutionError(
            "implementation changed after prepare; create a new content-addressed "
            "campaign cell before executing another stage"
        )
    if payload.get("implementation_sha256") != _implementation_identity_sha256(
        current_implementation
    ):
        raise CellExecutionError("preparation implementation digest mismatch")
    cache_key = (ctx.cell.cell_id, ctx.artifact_sha256(path))
    if cache_key not in _PREPARATION_CONTRACT_CACHE:
        _validate_preparation_contract(
            ctx.cell,
            payload,
            cell_manifest_sha256=_sha256(ctx.cell_path),
        )
        _PREPARATION_CONTRACT_CACHE.add(cache_key)
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
    source_contract = data["source_contract"]
    allocation = source_contract["split_plan"][split]
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
            "position_start": int(source_contract["source"]["drop_positions"]),
        },
    )
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
        coordinate_mask = torch.arange(x.shape[2], device=x.device).view(
            1, -1
        ) < torch.tensor(selected_dims, device=x.device).view(-1, 1)
        if mode in {"none", "scalar_rms", "sqrt_d"}:
            operator = (
                torch.diagonal(
                    transform.W,
                    dim1=-2,
                    dim2=-1,
                )
                .index_select(0, index)
                .to(x.device)
            )
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
    model_state: Mapping[str, torch.Tensor] | None = None,
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
    serialized_model_state: Mapping[str, torch.Tensor]
    if model_state is None:
        serialized_model_state = {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in model.state_dict().items()
        }
    else:
        expected_state = model.state_dict()
        if set(model_state) != set(expected_state):
            raise CellExecutionError(
                "deployment model snapshot field set differs from live model"
            )
        serialized_model_state = model_state
    payload = {
        "format_version": 2,
        "schema": "bsc-deployable-codec-v2",
        "cell_id": ctx.cell.cell_id,
        "checkpoint_sha256": checkpoint_hash,
        "calibration_sha256": calibration_hash,
        "preparation_sha256": preparation_hash,
        "model_cfg": asdict(model.cfg),
        "model_state": serialized_model_state,
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
) -> tuple[dict[str, Any], BlockCrosscoder, Codec, dict[str, int], str]:
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
        for identity in MODEL_IMPLEMENTATION_IDENTITY_FIELDS:
            if identity not in cfg_payload:
                raise ValueError(f"model config lacks {identity} identity")
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
        model.validate_decoded_energy_implementation()
        codec_payload = payload["codec_payload"]
        if not isinstance(codec_payload, dict):
            raise TypeError("nested codec payload must be a mapping")
        codec = Codec.from_payload(codec_payload, source=f"{path}:codec_payload")
        nested_model_cfg = codec.meta.get("model_cfg")
        if not isinstance(nested_model_cfg, dict) or canonical_json(
            nested_model_cfg
        ) != canonical_json(cfg_payload):
            raise ValueError(
                "nested codec model config differs from the deployed model config"
            )
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
    return value, model, codec, training_summary, claimed


def _assert_deployment_snapshot_digest(
    *,
    expected: str,
    verified: str,
) -> None:
    def canonical(value: object) -> bool:
        return (
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
        )

    if not canonical(expected) or not canonical(verified) or verified != expected:
        raise CellExecutionError(
            "durable deployable payload differs from the exact pre-save snapshot"
        )


def _save_immutable_torch(
    path: Path,
    payload: Mapping[str, Any],
    *,
    model_lineage: _ModelSnapshotLineage | None = None,
    model_state_field: str | None = None,
) -> None:
    serialized_model_state: Mapping[str, torch.Tensor] | None = None
    if model_lineage is not None:
        if not isinstance(model_state_field, str) or not model_state_field:
            raise CellExecutionError("snapshot-bound save requires a model-state field")
        candidate = payload.get(model_state_field)
        if not isinstance(candidate, Mapping):
            raise CellExecutionError(
                "snapshot-bound save lacks its model-state mapping"
            )
        serialized_model_state = candidate
        _assert_serialized_snapshot_current(
            serialized_model_state,
            model_lineage,
            label="deployable artifact",
        )

    def verify_existing() -> None:
        existing = torch.load(path, map_location="cpu", weights_only=True)
        if _tensor_payload_digest(existing) != _tensor_payload_digest(dict(payload)):
            raise CellExecutionError(
                f"immutable torch artifact changed binding: {path}"
            )
        if serialized_model_state is not None:
            _assert_serialized_snapshot_current(
                serialized_model_state,
                model_lineage,
                label="deployable artifact",
            )

    if path.exists():
        verify_existing()
        return
    durable_mkdir(path.parent, parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        # Saving to a path bakes the random temporary basename into every ZIP
        # member (and therefore the externally bound artifact hash). A file
        # object gives PyTorch the canonical ``archive/`` member prefix, so the
        # exact same tensor/scalar payload is byte-identical across fresh and
        # resumed executor processes.
        if model_lineage is not None and torch.save is not _NATIVE_TORCH_SAVE:
            raise CellExecutionError(
                "snapshot lineage requires native blocking torch.save"
            )
        save = _NATIVE_TORCH_SAVE if model_lineage is not None else torch.save
        with temporary.open("wb") as handle:
            save(dict(payload), handle)
        if serialized_model_state is not None:
            _assert_serialized_snapshot_current(
                serialized_model_state,
                model_lineage,
                label="deployable artifact",
            )
        try:
            durable_create(temporary, path)
        except FileExistsError:
            verify_existing()
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
        batch = carry[:batch_size]
        remainder = carry[batch_size:]
        carry = remainder if len(remainder) else None
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


def _prefetched_evaluation_batches(
    ctx: _Context,
    preparation: Mapping[str, Any],
    role: str = "evaluation",
) -> Iterator[torch.Tensor]:
    batches: Iterator[torch.Tensor] = _evaluation_batches(ctx, preparation, role)
    device = _device(ctx)
    if device.type == "cuda":
        batches = prefetch_batches(batches, depth=2, pin_memory=True)
        batches = cuda_prefetch_batches(batches, device=device, depth=1)
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
            ctx.values["model.encoder_scale_fit_solver"],
            ctx.values["model.encoder_scale_fit_target"],
            ctx.values["model.encoder_scale_fit_tolerance"],
            ctx.values["model.encoder_scale_fit_max_iterations"],
        ) != ("not_applicable", 0, "not_applicable", "not_applicable", 0.0, 0.0, 0):
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
        "global_fp64_mean_postactivation_block_norm"
    ) or ctx.values["model.encoder_scale_fit_solver"] != (
        "positive_bracketed_bisection_remeasure_v1"
    ):
        raise CellExecutionError(
            "fitted encoder scale requires the declared remeasured global fp64 "
            "postactivation block-norm contract"
        )
    target = float(ctx.values["model.encoder_scale_fit_target"])
    tolerance = float(ctx.values["model.encoder_scale_fit_tolerance"])
    max_iterations = int(ctx.values["model.encoder_scale_fit_max_iterations"])
    if target != 1.0 or tolerance != 1.0e-3 or max_iterations != 32:
        raise CellExecutionError(
            "unsupported encoder-scale fit target/tolerance budget"
        )

    def measure_block_norm() -> tuple[float, int]:
        total = torch.zeros((), dtype=torch.float64, device=model.parameter_device)
        count = 0
        for batch in _encoder_scale_fit_batches(ctx, preparation):
            x = batch.to(device=model.parameter_device, dtype=torch.float32)
            # The fit statistic is deliberately independent of the selector's
            # score geometry.  It measures the postactivation code itself, so
            # group shrinkage is included and signed loss-decrease scores can
            # never enter the calibration solver.
            code = model.encode(x)
            norms = torch.linalg.vector_norm(code.double(), dim=-1)
            total += norms.sum()
            count += norms.numel()
        if count == 0:
            raise CellExecutionError("encoder-scale normalization-fit stream is empty")
        mean_norm = float(total / count)
        if not math.isfinite(mean_norm) or mean_norm < 0.0:
            raise CellExecutionError(
                f"encoder-scale fit produced invalid mean block norm {mean_norm}"
            )
        return mean_norm, count

    current_multiplier = 1.0

    def measure_at(multiplier: float) -> tuple[float, int]:
        nonlocal current_multiplier
        if not math.isfinite(multiplier) or multiplier <= 0.0:
            raise CellExecutionError(
                "encoder-scale solver proposed an invalid multiplier"
            )
        model.scale_encoder_(multiplier / current_multiplier)
        current_multiplier = multiplier
        return measure_block_norm()

    mean_before, count = measure_at(1.0)
    mean_after = mean_before
    iterations = 1
    lower: tuple[float, float] | None = None
    upper: tuple[float, float] | None = None
    if abs(mean_before - target) > tolerance:
        if mean_before < target:
            lower = (1.0, mean_before)
            trial = 1.0
            while iterations < max_iterations:
                trial *= 2.0
                observed, observed_count = measure_at(trial)
                iterations += 1
                mean_after = observed
                if observed_count != count or observed + 1.0e-12 < lower[1]:
                    raise CellExecutionError(
                        "encoder-scale fit is not monotone on its declared replay stream"
                    )
                if observed >= target:
                    upper = (trial, observed)
                    break
                lower = (trial, observed)
        else:
            upper = (1.0, mean_before)
            trial = 1.0
            while iterations < max_iterations:
                trial *= 0.5
                observed, observed_count = measure_at(trial)
                iterations += 1
                mean_after = observed
                if observed_count != count or observed - 1.0e-12 > upper[1]:
                    raise CellExecutionError(
                        "encoder-scale fit is not monotone on its declared replay stream"
                    )
                if observed <= target:
                    lower = (trial, observed)
                    break
                upper = (trial, observed)
        if lower is None or upper is None:
            raise CellExecutionError(
                "encoder-scale fit could not bracket its declared target"
            )
        while iterations < max_iterations:
            trial = 0.5 * (lower[0] + upper[0])
            observed, observed_count = measure_at(trial)
            iterations += 1
            if (
                observed_count != count
                or not lower[1] - 1.0e-12 <= observed <= upper[1] + 1.0e-12
            ):
                raise CellExecutionError(
                    "encoder-scale fit violated monotonicity during bisection"
                )
            mean_after = observed
            if abs(observed - target) <= tolerance:
                break
            if observed < target:
                lower = (trial, observed)
            else:
                upper = (trial, observed)
    if abs(mean_after - target) > tolerance:
        raise CellExecutionError(
            "encoder-scale fit failed its remeasured post-fit tolerance: "
            f"observed={mean_after}, target={target}, tolerance={tolerance}"
        )
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
        "statistic": ctx.values["model.encoder_scale_fit_statistic"],
        "solver": ctx.values["model.encoder_scale_fit_solver"],
        "target": target,
        "tolerance": tolerance,
        "max_iterations": max_iterations,
        "iterations": iterations,
        "mean_block_norm_before": mean_before,
        "scale_multiplier": current_multiplier,
        "mean_block_norm_after": mean_after,
        "remeasured_post_fit": True,
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


@torch.no_grad()
def _decoded_energy_specialization_preflight(
    model: BlockCrosscoder,
    train_cfg: TrainConfig,
) -> dict[str, Any]:
    """Bind the score specialization's master and forward-copy geometry."""

    master = model.validate_decoded_energy_implementation()
    if not master["applicable"]:
        return {
            "schema": "bsc-decoded-energy-specialization-v1",
            "applicable": False,
            "implementation": model.cfg.decoded_energy_implementation,
            "passed": True,
        }
    if not decoded_energy_code_norm_eligible(
        selection_score=model.cfg.selection_score,
        decoder_constraint=model.cfg.decoder_constraint,
        training_selector=model.cfg.selection,
        site_rank=model.cfg.site_rank,
        retract_every=train_cfg.retract_every,
    ):
        raise CellExecutionError(
            "decoded-energy specialization disagrees with resolved trainer cadence"
        )
    if train_cfg.forward_dtype == "bf16":
        forward_model = copy.deepcopy(model).to(torch.bfloat16)
        try:
            forward = forward_model.validate_decoded_energy_implementation()
        finally:
            del forward_model
    else:
        forward = dict(master)
    passed = bool(master["passed"] and forward["passed"])
    return {
        "schema": "bsc-decoded-energy-specialization-v1",
        "applicable": True,
        "implementation": model.cfg.decoded_energy_implementation,
        "master": master,
        "forward": forward,
        "retract_every_steps": train_cfg.retract_every,
        "passed": passed,
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
        "preparation_sha256": ctx.artifact_sha256(ctx.preparation),
        "model_cfg": asdict(model_cfg),
        "train_cfg": asdict(train_cfg),
        "initialization": dict(initialization),
        "data": preparation["data"],
        "runtime": preparation["runtime"],
        "selection": preparation["selection"],
        "implementation": preparation["implementation"],
    }


def _validate_final_checkpoint(
    path: Path,
    binding: Mapping[str, Any],
    *,
    retain_model_state: bool = False,
    expected_model_lineage: _ModelSnapshotLineage | None = None,
) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 - convert corrupt torch files cleanly
        raise CellExecutionError(
            f"cannot load immutable checkpoint {path}: {exc}"
        ) from exc
    model_state = payload.get("model")
    claimed_digest_contract = payload.get("model_state_digest_contract")
    claimed_model_state_sha256 = payload.get("model_state_sha256")
    if (
        not isinstance(model_state, Mapping)
        or not model_state
        or claimed_digest_contract != MODEL_STATE_DIGEST_CONTRACT
        or not isinstance(claimed_model_state_sha256, str)
        or len(claimed_model_state_sha256) != 64
        or model_state_digest(model_state) != claimed_model_state_sha256
    ):
        raise CellExecutionError("final checkpoint model-state digest mismatch")
    if expected_model_lineage is not None and (
        expected_model_lineage.snapshot_digest_contract != MODEL_STATE_DIGEST_CONTRACT
        or not isinstance(expected_model_lineage.snapshot_sha256, str)
        or claimed_digest_contract != expected_model_lineage.snapshot_digest_contract
        or claimed_model_state_sha256 != expected_model_lineage.snapshot_sha256
    ):
        raise CellExecutionError(
            "final checkpoint model-state digest differs from synchronous snapshot"
        )
    if payload.get("run_binding") != binding:
        raise CellExecutionError("existing checkpoint has a different run binding")
    try:
        validate_run_binding(
            payload.get("run_binding"),
            {
                "model_cfg": payload.get("model_cfg"),
                "train_cfg": payload.get("train_cfg"),
            },
            keys=("model_cfg", "train_cfg"),
        )
    except ValueError as exc:
        raise CellExecutionError(
            "checkpoint top-level configuration disagrees with its run binding"
        ) from exc
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
    train_payload = payload.get("train_cfg")
    if not isinstance(train_payload, dict):
        raise CellExecutionError("final checkpoint lacks train configuration")
    try:
        train_cfg = TrainConfig(
            **{
                **train_payload,
                "betas": tuple(train_payload["betas"]),
            }
        )
        validate_optimizer_state_config(
            payload.get("optimizer"),
            train_cfg,
            optimizer_kind,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CellExecutionError(
            f"final checkpoint optimizer contract is invalid: {exc}"
        ) from exc
    history = payload.get("history")
    previous_shares = payload.get("diagnostic_prev_shares")
    model_payload = payload.get("model_cfg")
    if not isinstance(model_payload, dict):
        raise CellExecutionError("final checkpoint lacks model configuration")
    if expected_model_lineage is not None:
        durable_config_sha256 = hashlib.sha256(
            canonical_json(model_payload).encode("utf-8")
        ).hexdigest()
        if durable_config_sha256 != expected_model_lineage.model_config_sha256:
            raise CellExecutionError(
                "final checkpoint model config differs from synchronous snapshot"
            )
        _assert_durable_snapshot_schema(
            payload.get("model"),
            expected_model_lineage,
            label="final checkpoint",
        )
    for identity in MODEL_IMPLEMENTATION_IDENTITY_FIELDS:
        if identity not in model_payload:
            raise CellExecutionError(f"final checkpoint lacks {identity} identity")
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
    metadata = {
        "model_cfg": payload.get("model_cfg"),
        "train_cfg": payload.get("train_cfg"),
        "run_binding": payload.get("run_binding"),
        "step_idx": step_idx,
        "accepted_tokens": accepted_tokens,
        "data_cursor": dict(data_cursor),
        "optimizer_kind": optimizer_kind,
        "terminal_log": dict(history[-1]) if history else None,
    }
    if retain_model_state:
        model_state = payload.get("model")
        if not isinstance(model_state, dict) or not model_state:
            raise CellExecutionError("final checkpoint lacks retained model state")
        metadata["_retained_model_state"] = dict(model_state)
    return metadata


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
    encoder_scale_calibration = initialization.get("encoder_scale_calibration")
    precision_preflight = initialization.get("precision_preflight")
    decoded_energy_preflight = initialization.get("decoded_energy_specialization")
    if not isinstance(regularizer_calibration, Mapping):
        raise CellExecutionError(
            "final checkpoint lacks regularizer-calibration provenance"
        )
    if not isinstance(encoder_scale_calibration, Mapping):
        raise CellExecutionError(
            "final checkpoint lacks encoder-scale-calibration provenance"
        )
    if not isinstance(precision_preflight, Mapping):
        raise CellExecutionError(
            "final checkpoint lacks precision-preflight provenance"
        )
    if not isinstance(decoded_energy_preflight, Mapping):
        raise CellExecutionError(
            "final checkpoint lacks decoded-energy specialization provenance"
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
        "encoder_scale_calibration": dict(encoder_scale_calibration),
        "regularizer_calibration": dict(regularizer_calibration),
        "precision_preflight": dict(precision_preflight),
        "decoded_energy_specialization": dict(decoded_energy_preflight),
    }


def _verified_training_report_model_cfg(
    ctx: _Context,
    prerequisites: Mapping[str, tuple[Path, str]],
    *,
    checkpoint_hash: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the exact runtime-resolved model config bound by training."""

    training_report = _read_object(
        prerequisites["training_report"][0], label="training report"
    )
    if (
        training_report.get("schema") != TRAINING_REPORT_SCHEMA
        or training_report.get("cell_id") != ctx.cell.cell_id
        or training_report.get("checkpoint_sha256") != checkpoint_hash
        or training_report.get("preparation_sha256")
        != prerequisites["preparation"][1]
    ):
        raise CellExecutionError("training report/input binding mismatch")
    model_cfg = training_report.get("model_cfg")
    if not isinstance(model_cfg, dict):
        raise CellExecutionError("training report lacks its resolved model config")
    try:
        resolved = BSCConfig(**model_cfg)
    except (KeyError, TypeError, ValueError) as exc:
        raise CellExecutionError(
            f"training report has an invalid resolved model config: {exc}"
        ) from exc
    resolved_payload = asdict(resolved)
    if set(model_cfg) != set(resolved_payload) or canonical_json(
        model_cfg
    ) != canonical_json(resolved_payload):
        raise CellExecutionError(
            "training report model config does not contain the exact resolved fields"
        )
    return training_report, dict(model_cfg)


def _execution_rng_snapshot() -> tuple[Any, ...]:
    """Snapshot process RNGs to prove validation itself consumes none.

    This is deliberately not a checkpoint-RNG handoff: only the exact model
    consumer crosses the parent-verified durable artifact boundary.
    """

    numpy_state = np.random.get_state()
    torch_mps = (
        torch.mps.get_rng_state().cpu().numpy().tobytes()
        if torch.backends.mps.is_available()
        else None
    )
    return (
        random.getstate(),
        (
            numpy_state[0],
            numpy_state[1].tobytes(),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        ),
        torch.get_rng_state().numpy().tobytes(),
        tuple(
            state.cpu().numpy().tobytes()
            for state in (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else ()
            )
        ),
        torch_mps,
    )


def _model_config_sha256(model: BlockCrosscoder) -> str:
    return hashlib.sha256(canonical_json(asdict(model.cfg)).encode("utf-8")).hexdigest()


def _assert_callback_free_state_dict(model: BlockCrosscoder) -> None:
    if type(model) is not BlockCrosscoder:
        raise CellExecutionError(
            "exact model snapshot requires the canonical BlockCrosscoder type"
        )
    callbacks: list[str] = []
    for module_name, module in model.named_modules():
        label = module_name or "<root>"
        state_dict_method = getattr(module.state_dict, "__func__", None)
        if state_dict_method is not torch.nn.Module.state_dict:
            callbacks.append(f"{label}:state_dict")
        save_to_state_dict = getattr(module._save_to_state_dict, "__func__", None)
        if save_to_state_dict is not torch.nn.Module._save_to_state_dict:
            callbacks.append(f"{label}:_save_to_state_dict")
        for attribute in ("_state_dict_pre_hooks", "_state_dict_hooks"):
            hooks = getattr(module, attribute, None)
            if hooks:
                callbacks.append(f"{label}:{attribute}")
    if callbacks:
        raise CellExecutionError(
            "exact model snapshot requires canonical callback-free state_dict: "
            + ", ".join(callbacks)
        )


def _synchronize_state_devices(state: Mapping[str, torch.Tensor]) -> None:
    devices = sorted({tensor.device for tensor in state.values()}, key=str)
    for device in devices:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()


def _tensor_snapshot_descriptor(
    name: str,
    tensor: torch.Tensor,
) -> _TensorSnapshotDescriptor:
    if type(tensor) not in {torch.Tensor, torch.nn.Parameter}:
        raise CellExecutionError(
            f"exact model snapshot forbids tensor subclass for {name}"
        )
    if tensor.layout != torch.strided:
        raise CellExecutionError(
            f"exact model snapshot requires dense strided tensor {name}"
        )
    storage = tensor.untyped_storage()
    try:
        version = int(tensor._version)
    except RuntimeError as exc:
        raise CellExecutionError(
            f"exact model snapshot cannot inspect version for {name}: {exc}"
        ) from exc
    return _TensorSnapshotDescriptor(
        name=name,
        object_id=id(tensor),
        object_type=f"{type(tensor).__module__}.{type(tensor).__qualname__}",
        storage_identity=int(storage._cdata),
        storage_data_ptr=int(storage.data_ptr()),
        storage_nbytes=int(storage.nbytes()),
        tensor_data_ptr=int(tensor.data_ptr()),
        storage_offset=int(tensor.storage_offset()),
        version=version,
        device=str(tensor.device),
        dtype=str(tensor.dtype),
        layout=str(tensor.layout),
        shape=tuple(tensor.shape),
        stride=tuple(tensor.stride()),
        requires_grad=bool(tensor.requires_grad),
        gradient_present=getattr(tensor, "grad", None) is not None,
    )


def _state_snapshot_descriptors(
    state: Mapping[str, torch.Tensor],
) -> tuple[_TensorSnapshotDescriptor, ...]:
    if not isinstance(state, Mapping) or not state:
        raise CellExecutionError("exact model snapshot state is empty")
    descriptors: list[_TensorSnapshotDescriptor] = []
    for name, tensor in state.items():
        if not isinstance(name, str) or not name:
            raise CellExecutionError("exact model snapshot has an invalid field name")
        if not torch.is_tensor(tensor):
            raise CellExecutionError(
                f"exact model snapshot field {name} is not a tensor"
            )
        descriptors.append(_tensor_snapshot_descriptor(name, tensor))
    return tuple(descriptors)


def _live_model_snapshot_descriptors(
    model: BlockCrosscoder,
) -> tuple[_TensorSnapshotDescriptor, ...]:
    _assert_callback_free_state_dict(model)
    state = model.state_dict(keep_vars=True)
    return _state_snapshot_descriptors(state)


def _assert_model_snapshot_lineage_current(
    model: BlockCrosscoder,
    lineage: _ModelSnapshotLineage,
    *,
    label: str,
) -> None:
    _assert_callback_free_state_dict(model)
    live_state = model.state_dict(keep_vars=True)
    _synchronize_state_devices(live_state)
    current = _state_snapshot_descriptors(live_state)
    drift: list[str] = []
    if id(model) != lineage.model_object_id:
        drift.append("model identity")
    if f"{type(model).__module__}.{type(model).__qualname__}" != lineage.model_type:
        drift.append("model type")
    if _model_config_sha256(model) != lineage.model_config_sha256:
        drift.append("model config")
    if bool(model.training) != lineage.model_training:
        drift.append("training mode")
    if (model._validated_theta_key is None) != lineage.threshold_cache_empty:
        drift.append("threshold cache")
    if current != lineage.live_state:
        drift.append("tensor identity/storage/version/schema")
    if drift:
        raise CellExecutionError(
            f"{label} drifted after its exact durable snapshot: " + ", ".join(drift)
        )


def _assert_serialized_snapshot_current(
    state: Mapping[str, torch.Tensor],
    lineage: _ModelSnapshotLineage,
    *,
    label: str,
) -> None:
    if id(state) != lineage.snapshot_mapping_id:
        raise CellExecutionError(f"{label} replaced the serialized state mapping")
    current = _state_snapshot_descriptors(state)
    if current != lineage.snapshot_state:
        raise CellExecutionError(
            f"{label} mutated or replaced the serialized snapshot tensors"
        )


def _assert_durable_snapshot_schema(
    state: Any,
    lineage: _ModelSnapshotLineage,
    *,
    label: str,
) -> None:
    if not isinstance(state, Mapping):
        raise CellExecutionError(f"{label} model state is not a mapping")
    expected = lineage.snapshot_state
    if tuple(state) != tuple(item.name for item in expected):
        raise CellExecutionError(
            f"{label} model state field order differs from snapshot"
        )
    for descriptor in expected:
        tensor = state[descriptor.name]
        if (
            not torch.is_tensor(tensor)
            or tensor.device.type != "cpu"
            or str(tensor.dtype) != descriptor.dtype
            or str(tensor.layout) != descriptor.layout
            or tuple(tensor.shape) != descriptor.shape
            or tuple(tensor.stride()) != descriptor.stride
            or int(tensor.storage_offset()) != descriptor.storage_offset
            or int(tensor.untyped_storage().nbytes()) != descriptor.storage_nbytes
        ):
            raise CellExecutionError(
                f"{label} model state schema differs for {descriptor.name}"
            )


def _synchronous_model_snapshot(
    model: BlockCrosscoder,
    *,
    include_model_digest: bool = True,
) -> tuple[dict[str, torch.Tensor], _ModelSnapshotLineage]:
    """Copy one quiescent model state to CPU and bind its exact live owner.

    Every package-owned tensor mutation is versioned. Raw-pointer writes and
    external concurrent writers are outside the executor contract; admitting
    either would require restoring a second full byte comparison.
    """

    rng_before = _execution_rng_snapshot()
    try:
        _assert_callback_free_state_dict(model)
        live_state = model.state_dict(keep_vars=True)
        _synchronize_state_devices(live_state)
        config_before = _model_config_sha256(model)
        live_before = _state_snapshot_descriptors(live_state)
        with torch.no_grad():
            snapshot = {
                name: tensor.detach().to(device="cpu", copy=True).contiguous()
                for name, tensor in live_state.items()
            }
        _synchronize_state_devices(live_state)
        config_after = _model_config_sha256(model)
        live_after = _live_model_snapshot_descriptors(model)
        if config_after != config_before or live_after != live_before:
            raise CellExecutionError(
                "live model changed while taking its synchronous durable snapshot"
            )
        lineage = _ModelSnapshotLineage(
            model_object_id=id(model),
            model_type=f"{type(model).__module__}.{type(model).__qualname__}",
            model_config_sha256=config_after,
            model_training=bool(model.training),
            threshold_cache_empty=model._validated_theta_key is None,
            live_state=live_after,
            snapshot_mapping_id=id(snapshot),
            snapshot_state=_state_snapshot_descriptors(snapshot),
            snapshot_digest_contract=(
                MODEL_STATE_DIGEST_CONTRACT if include_model_digest else None
            ),
            snapshot_sha256=(
                model_state_digest(snapshot) if include_model_digest else None
            ),
        )
        _assert_serialized_snapshot_current(
            snapshot,
            lineage,
            label="synchronous model snapshot",
        )
        return snapshot, lineage
    finally:
        if _execution_rng_snapshot() != rng_before:
            raise CellExecutionError("exact model snapshot perturbed global RNG state")


def _normalize_model_only_consumer_state(model: BlockCrosscoder) -> None:
    """Match a fresh artifact consumer's ephemeral autograd/cache state."""

    bad_gradients: list[str] = []
    bad_requires_grad: list[str] = []
    for name, parameter in model.named_parameters():
        parameter.grad = None
        expected_requires_grad = model.cfg.decoder_bias if name == "c" else True
        parameter.requires_grad_(expected_requires_grad)
        if parameter.grad is not None:
            bad_gradients.append(name)
        if parameter.requires_grad is not expected_requires_grad:
            bad_requires_grad.append(name)
    if bad_gradients:
        raise CellExecutionError(
            "model-only handoff retained parameter gradients: "
            + ", ".join(bad_gradients)
        )
    if bad_requires_grad:
        raise CellExecutionError(
            "model-only handoff differs from fresh requires-grad schema: "
            + ", ".join(bad_requires_grad)
        )
    model._validated_theta_key = None
    if model._validated_theta_key is not None:
        raise CellExecutionError("model-only handoff retained threshold cache state")


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
    execution_cache: _StageExecutionCache | None = None,
) -> tuple[tuple[str, Path], ...]:
    preparation = _load_preparation(prerequisites["preparation"][0], ctx)
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
    # Validate before any encoder/regularizer/precision calibration can invoke
    # the specialized score path.
    initialization["decoded_energy_specialization"] = (
        _decoded_energy_specialization_preflight(model, train_cfg)
    )
    if initialization["decoded_energy_specialization"].get("passed") is not True:
        raise CellExecutionError(
            "decoded-energy specialization preflight failed before optimizer construction"
        )
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
                    checkpoint_hash=ctx.artifact_sha256(ctx.checkpoint),
                    preparation_hash=prerequisites["preparation"][1],
                    terminal_log=metadata["terminal_log"],
                ),
            )
        report = _read_object(ctx.training_report, label="training report")
        if (
            report.get("checkpoint_sha256") != ctx.artifact_sha256(ctx.checkpoint)
            or report.get("attempted_tokens") != int(final_cursor)
            or report.get("data_cursor") != metadata["data_cursor"]
            or report.get("accepted_tokens") != metadata["accepted_tokens"]
            or report.get("step_idx") != metadata["step_idx"]
            or report.get("optimizer_kind") != metadata["optimizer_kind"]
            or canonical_json(report.get("model_cfg"))
            != canonical_json(metadata["model_cfg"])
            or canonical_json(report.get("train_cfg"))
            != canonical_json(metadata["train_cfg"])
            or canonical_json(report.get("encoder_scale_calibration"))
            != canonical_json(
                metadata["run_binding"]["initialization"]["encoder_scale_calibration"]
            )
            or canonical_json(report.get("regularizer_calibration"))
            != canonical_json(
                metadata["run_binding"]["initialization"]["regularizer_calibration"]
            )
            or canonical_json(report.get("precision_preflight"))
            != canonical_json(
                metadata["run_binding"]["initialization"]["precision_preflight"]
            )
            or canonical_json(report.get("decoded_energy_specialization"))
            != canonical_json(
                metadata["run_binding"]["initialization"][
                    "decoded_energy_specialization"
                ]
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
        model = None

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
        # Keep two pinned raw batches ready, then copy one device batch ahead
        # on a dedicated stream. Shard latency and H2D transfer therefore stay
        # off the GPU's compute critical path. On-the-fly Phase-3 transforms
        # execute after the per-batch copy event on the consumer stream.
        training_batches = prefetch_batches(
            training_batches,
            depth=2,
            pin_memory=True,
        )
        training_batches = cuda_prefetch_batches(
            training_batches,
            device=device,
            depth=1,
        )
    try:
        for batch in training_batches:
            if trainer.step_idx >= trainer.cfg.total_steps:
                break
            x, observed = _unpack_training_batch(batch)
            if cuda_transform is not None:
                x = x.to(device=device, non_blocking=True)
                x = _apply_prepared_transform(x, preparation, cuda_transform)
            trainer.step(x, observed=observed, materialize_record=False)
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
    retained_model = trainer.master
    _normalize_model_only_consumer_state(retained_model)
    retained_model.eval()
    checkpoint_model_state, checkpoint_lineage = _synchronous_model_snapshot(
        retained_model
    )
    checkpoint_model_sha256 = checkpoint_lineage.snapshot_sha256
    if not isinstance(checkpoint_model_sha256, str):
        raise CellExecutionError("final checkpoint snapshot lacks its model digest")
    # The final durable payload replaces the progress payload atomically.  At
    # no point do we retain two complete optimizer checkpoints for one cell.
    trainer.save_checkpoint(
        ctx.progress,
        model_state=checkpoint_model_state,
        model_state_sha256=checkpoint_model_sha256,
        require_native_blocking_save=True,
    )
    _assert_serialized_snapshot_current(
        checkpoint_model_state,
        checkpoint_lineage,
        label="final checkpoint",
    )
    _assert_model_snapshot_lineage_current(
        retained_model,
        checkpoint_lineage,
        label="final checkpoint source model",
    )
    durable_replace(ctx.progress, ctx.checkpoint, file_already_synced=True)
    checkpoint_before_validation = _FileFingerprint.from_path(ctx.checkpoint)
    del checkpoint_model_state
    gc.collect()
    metadata = _validate_final_checkpoint(
        ctx.checkpoint,
        binding,
        expected_model_lineage=checkpoint_lineage,
    )
    checkpoint_hash = ctx.artifact_sha256(ctx.checkpoint)
    checkpoint_after_validation = _FileFingerprint.from_path(ctx.checkpoint)
    if checkpoint_after_validation != checkpoint_before_validation:
        raise CellExecutionError("final checkpoint changed while validating")
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
        checkpoint_hash=checkpoint_hash,
        preparation_hash=prerequisites["preparation"][1],
        terminal_log=history[-1] if history else None,
    )
    _write_immutable_json(ctx.training_report, report)
    if execution_cache is not None:
        released_owner_refs = {
            "trainer": weakref.ref(trainer),
            "forward": (
                None if trainer.fwd is retained_model else weakref.ref(trainer.fwd)
            ),
            "optimizer": weakref.ref(trainer.opt),
        }
        del trainer
        gc.collect()
        live_released_owners = sorted(
            name
            for name, reference in released_owner_refs.items()
            if reference is not None and reference() is not None
        )
        if live_released_owners:
            raise CellExecutionError(
                "released training owners remain live before calibration handoff: "
                + ", ".join(live_released_owners)
            )
        if device.type == "cuda":
            torch.cuda.empty_cache()
        retained_metadata = dict(metadata)
        retained_metadata["checkpoint_sha256"] = checkpoint_hash
        execution_cache.remember_checkpoint(
            _retained_artifact_key(
                ctx,
                producer_stage="train",
                consumer_stage="calibrate",
                artifact_kind="checkpoint",
                path=ctx.checkpoint,
                sha256=checkpoint_hash,
                fingerprint=checkpoint_after_validation,
                model_cfg=metadata["model_cfg"],
            ),
            retained_model,
            retained_metadata,
            checkpoint_lineage,
            released_owner_refs={
                name: reference
                for name, reference in released_owner_refs.items()
                if reference is not None
            },
        )
    return (
        ("checkpoint", ctx.checkpoint),
        ("training_report", ctx.training_report),
    )


def _calibrate(
    ctx: _Context,
    prerequisites: Mapping[str, tuple[Path, str]],
    *,
    execution_cache: _StageExecutionCache | None = None,
) -> tuple[tuple[str, Path], ...]:
    preparation = _load_preparation(prerequisites["preparation"][0], ctx)
    checkpoint_path, checkpoint_hash = prerequisites["checkpoint"]
    _, resolved_model_cfg = _verified_training_report_model_cfg(
        ctx,
        prerequisites,
        checkpoint_hash=checkpoint_hash,
    )
    checkpoint_before_load = ctx.prerequisite_fingerprint(
        checkpoint_path,
        sha256=checkpoint_hash,
    )
    retained_checkpoint = (
        None
        if execution_cache is None
        else execution_cache.take_checkpoint(
            _retained_artifact_key(
                ctx,
                producer_stage="train",
                consumer_stage="calibrate",
                artifact_kind="checkpoint",
                path=checkpoint_path,
                sha256=checkpoint_hash,
                fingerprint=checkpoint_before_load,
                model_cfg=resolved_model_cfg,
            )
        )
    )
    if retained_checkpoint is None:
        try:
            model, metadata = load_trained_model(
                checkpoint_path,
                device=_device(ctx),
                verified_checkpoint_sha256=checkpoint_hash,
            )
        except Exception as exc:  # noqa: BLE001
            raise CellExecutionError(
                f"cannot load checkpoint for calibration: {exc}"
            ) from exc
    else:
        model, metadata = retained_checkpoint
        if next(model.parameters()).device != _resolved_runtime_device(_device(ctx)):
            raise CellExecutionError("retained checkpoint model is on the wrong device")
    checkpoint_after_load = ctx.prerequisite_fingerprint(
        checkpoint_path,
        sha256=checkpoint_hash,
    )
    if checkpoint_after_load != checkpoint_before_load:
        raise CellExecutionError("checkpoint changed while loading for calibration")
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
    with closing(
        _prefetched_evaluation_batches(ctx, preparation, "calibration")
    ) as batches:
        model.fit_threshold_(
            batches,
            target_avg_blocks=target,
            method=quantile_method,
        )
    threshold_source: dict[str, Any] = {
        "split": str(ctx.values["evaluation.calibration_split"]),
        "quantile_method": quantile_method,
        "target_avg_blocks": target,
    }
    achieved_events_device = torch.zeros(
        (),
        dtype=torch.int64,
        device=_device(ctx),
    )
    achieved_tokens = 0
    with torch.no_grad():
        calibration_decoder = model.decoder_tensor()
        calibration_encoder = (
            model._tied_encoder_tensor(calibration_decoder)
            if model.cfg.encoder_mode == "tied"
            else model.encoder_tensor()
        )
        calibration_score_geometry = model._frozen_score_geometry(calibration_decoder)
        with closing(
            _prefetched_evaluation_batches(ctx, preparation, "calibration")
        ) as batches:
            for batch in batches:
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
                achieved_events_device.add_(selected.sum(dtype=torch.int64))
                achieved_tokens += len(x)
    if achieved_tokens == 0:
        raise CellExecutionError("calibration split is empty")
    achieved_events = int(achieved_events_device)
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
    with closing(
        _prefetched_evaluation_batches(ctx, preparation, "calibration")
    ) as batches:
        codec = fit_codec(
            model,
            batches,
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
    calibration_hash = ctx.artifact_sha256(ctx.calibration)
    _normalize_model_only_consumer_state(model)
    model.eval()
    deployment_model_state, deployment_lineage = _synchronous_model_snapshot(
        model,
        include_model_digest=False,
    )
    deployment_payload = _deployment_codec_payload(
        ctx,
        preparation,
        model,
        checkpoint_hash=checkpoint_hash,
        checkpoint_metadata=metadata,
        calibration_hash=calibration_hash,
        preparation_hash=prerequisites["preparation"][1],
        model_state=deployment_model_state,
    )
    expected_deployment_snapshot_digest = deployment_payload.get("artifact_sha256")
    if not isinstance(expected_deployment_snapshot_digest, str):
        raise CellExecutionError("deployable snapshot lacks its full-payload digest")
    _assert_serialized_snapshot_current(
        deployment_payload["model_state"],
        deployment_lineage,
        label="deployable artifact",
    )
    _save_immutable_torch(
        ctx.deployment_codec,
        deployment_payload,
        model_lineage=deployment_lineage,
        model_state_field="model_state",
    )
    _assert_model_snapshot_lineage_current(
        model,
        deployment_lineage,
        label="deployable artifact source model",
    )
    # The immutable file now owns this CPU snapshot. Keeping the producer-side
    # construction payload alive while reloading validation bytes would triple
    # the model-state RSS during the handoff.
    del deployment_payload, deployment_model_state
    gc.collect()
    # Reconstruct the exact consumer before pricing; truncated, incomplete, or
    # internally inconsistent bytes never reach evaluation.
    deployment_before_load = _FileFingerprint.from_path(ctx.deployment_codec)
    (
        frozen_deployment,
        frozen_consumer_model,
        frozen_consumer_codec,
        frozen_training_summary,
        verified_deployment_snapshot_digest,
    ) = _load_deployable_codec(
        ctx.deployment_codec,
        cell_id=ctx.cell.cell_id,
        checkpoint_hash=checkpoint_hash,
        calibration_hash=calibration_hash,
        preparation_hash=prerequisites["preparation"][1],
        device=torch.device("cpu"),
    )
    _assert_deployment_snapshot_digest(
        expected=expected_deployment_snapshot_digest,
        verified=verified_deployment_snapshot_digest,
    )
    deployment_hash = ctx.artifact_sha256(ctx.deployment_codec)
    deployment_after_load = _FileFingerprint.from_path(ctx.deployment_codec)
    if deployment_after_load != deployment_before_load:
        raise CellExecutionError("deployable codec changed while reconstructing")
    if (
        frozen_deployment.get("schema") != "bsc-deployable-codec-v2"
        or frozen_deployment.get("cell_id") != ctx.cell.cell_id
        or frozen_deployment.get("checkpoint_sha256") != checkpoint_hash
        or frozen_deployment.get("calibration_sha256") != calibration_hash
    ):
        raise CellExecutionError("deployable codec artifact binding mismatch")
    _assert_durable_snapshot_schema(
        frozen_deployment.get("model_state"),
        deployment_lineage,
        label="deployable artifact",
    )
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
        "codec_sha256": calibration_hash,
        "deployment_codec_sha256": deployment_hash,
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
    if execution_cache is not None:
        if any(
            tensor.device.type != "cpu"
            for tensor in frozen_consumer_model.state_dict().values()
        ):
            raise CellExecutionError(
                "durable validation model unexpectedly occupies an accelerator"
            )
        frozen_model_cfg = asdict(frozen_consumer_model.cfg)
        discarded_validation_model_ref = weakref.ref(frozen_consumer_model)
        del frozen_consumer_model
        gc.collect()
        if discarded_validation_model_ref() is not None:
            raise CellExecutionError(
                "durable validation model remains live after exact comparison"
            )
        retained_deployment = dict(frozen_deployment)
        for heavy_field in ("model_state", "codec_payload"):
            del retained_deployment[heavy_field]
        if {"model_state", "codec_payload"} & set(retained_deployment):
            raise CellExecutionError(
                "volatile deployment retained a heavy durable consumer field"
            )
        del frozen_deployment
        gc.collect()
        execution_cache.remember_deployment(
            _retained_artifact_key(
                ctx,
                producer_stage="calibrate",
                consumer_stage="evaluate",
                artifact_kind="deployment_codec",
                path=ctx.deployment_codec,
                sha256=deployment_hash,
                fingerprint=deployment_after_load,
                model_cfg=frozen_model_cfg,
            ),
            retained_deployment,
            model,
            frozen_consumer_codec,
            frozen_training_summary,
            deployment_lineage,
            discarded_validation_model_ref=discarded_validation_model_ref,
        )
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


def _accumulate_chunked_recovery_association(
    truth_active: torch.Tensor,
    block_mask: torch.Tensor,
    coactive: torch.Tensor,
    truth_count: torch.Tensor,
    predicted_count: torch.Tensor,
) -> None:
    """Accumulate exact binary association counts without a full-width cast."""

    truth = truth_active.to(device=coactive.device, dtype=torch.float64)
    truth_count.add_(truth.sum(dim=0))
    for start in range(0, block_mask.shape[1], _RECOVERY_ASSOCIATION_GROUP_CHUNK):
        stop = min(
            start + _RECOVERY_ASSOCIATION_GROUP_CHUNK,
            block_mask.shape[1],
        )
        predicted = block_mask[:, start:stop].to(dtype=torch.float64)
        coactive[:, start:stop].addmm_(truth.T, predicted)
        predicted_count[start:stop].add_(predicted.sum(dim=0))


def _mapped_support_confusion_counts(
    truth_active: torch.Tensor,
    block_mask: torch.Tensor,
    group_to_factor: torch.Tensor,
    category_masks: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map learned groups to factors and return exact device-resident counts."""

    predicted_uint8 = torch.zeros_like(truth_active, dtype=torch.uint8)
    predicted_uint8.scatter_reduce_(
        1,
        group_to_factor.unsqueeze(0).expand_as(block_mask),
        block_mask.to(torch.uint8),
        reduce="amax",
        include_self=False,
    )
    predicted = predicted_uint8.bool()
    totals = torch.stack(
        (
            (predicted & truth_active).sum(),
            (predicted & ~truth_active).sum(),
            (~predicted & truth_active).sum(),
            (~predicted & ~truth_active).sum(),
        )
    )
    if not category_masks:
        return totals, totals.new_zeros((0, 4))
    category_totals = torch.stack(
        tuple(
            torch.stack(
                (
                    (predicted[:, mask] & truth_active[:, mask]).sum(),
                    (predicted[:, mask] & ~truth_active[:, mask]).sum(),
                    (~predicted[:, mask] & truth_active[:, mask]).sum(),
                    truth_active[:, mask].sum(),
                )
            )
            for mask in category_masks
        )
    )
    return totals, category_totals


def _matching_pathologies(
    association: torch.Tensor,
    *,
    strong_cutoff: float,
    weak_cutoff: float,
) -> dict[str, float]:
    """Directional planted-factor/learned-group multiplicity diagnostics."""

    if not (
        math.isfinite(strong_cutoff)
        and math.isfinite(weak_cutoff)
        and 0.0 <= weak_cutoff < strong_cutoff <= 1.0
    ):
        raise CellExecutionError("pathology association cutoffs are invalid")
    strong = association >= strong_cutoff
    weak = association >= weak_cutoff
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
            (association.max(dim=1).values < weak_cutoff).float().mean()
        ),
    }


@torch.no_grad()
def _gather_event_factor_blocks(
    selected_code: torch.Tensor,
    selected_mask: torch.Tensor | None,
    event_factor: torch.Tensor,
    factor_to_group: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Gather only each event's truth-matched learned block on its device."""

    if selected_code.ndim != 3 or (
        selected_mask is not None and selected_mask.shape != selected_code.shape[:2]
    ):
        raise CellExecutionError("selected code/mask shapes are inconsistent")
    event_factor_device = event_factor.to(
        device=selected_code.device,
        dtype=torch.long,
    )
    factor_to_group_device = factor_to_group.to(
        device=selected_code.device,
        dtype=torch.long,
    )
    if event_factor_device.shape != (len(selected_code),):
        raise CellExecutionError("event-factor rows do not match selected codes")
    event_group = factor_to_group_device[event_factor_device]
    event_rows = torch.arange(len(selected_code), device=selected_code.device)
    return (
        selected_code[event_rows, event_group],
        (None if selected_mask is None else selected_mask[event_rows, event_group]),
    )


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
    rank_mismatch_contract: str,
    pathology_association_contract: str,
    pathology_strong_cutoff: float,
    pathology_weak_cutoff: float,
    pathology_cutoff_sensitivity: Sequence[Sequence[float]],
) -> dict[str, Any]:
    """Truth-aware recovery with matching frozen before scored rows are read."""

    device = next(model.parameters()).device
    if selection_mode not in {"topk", "threshold"}:
        raise CellExecutionError("synthetic recovery selection mode is invalid")
    if rank_mismatch_contract != (
        "same_block_primary_plus_calibration_frozen_minimum_group_diagnostic_v1"
    ):
        raise CellExecutionError("unknown Phase-1 rank-mismatch contract")
    if pathology_association_contract != (
        "primary_cutoffs_plus_complete_reporting_only_grid_v1"
    ):
        raise CellExecutionError("unknown Phase-1 pathology association contract")
    sensitivity_pairs = tuple(
        (float(pair[0]), float(pair[1])) for pair in pathology_cutoff_sensitivity
    )
    expected_pairs = tuple(
        (strong, weak) for strong in (0.4, 0.5, 0.6) for weak in (0.2, 0.25, 0.3)
    )
    if (
        pathology_strong_cutoff != 0.5
        or pathology_weak_cutoff != 0.25
        or sensitivity_pairs != expected_pairs
    ):
        raise CellExecutionError("Phase-1 pathology cutoff grid is not the frozen grid")
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
        model._tied_encoder_tensor(materialized_decoder)
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

    def frozen_select(
        x: torch.Tensor,
        *,
        observed: torch.Tensor | None = None,
    ):
        return model.select_with_materialized(
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
    coactive_device = torch.zeros(
        n_factors,
        n_groups,
        dtype=torch.float64,
        device=device,
    )
    truth_count_device = torch.zeros(
        n_factors,
        dtype=torch.float64,
        device=device,
    )
    predicted_count_device = torch.zeros(
        n_groups,
        dtype=torch.float64,
        device=device,
    )
    matching_examples = 0
    for matching_batch in matching_dataset.batches(
        batch_size,
        start=calibration_start,
        stop=calibration_stop,
    ):
        matching_x = _apply_normalization(matching_batch.x, normalization).to(device)
        matching_out = frozen_select(
            matching_x,
            observed=matching_batch.observed.to(device),
        )
        if device.type == "cuda":
            _accumulate_chunked_recovery_association(
                matching_batch.active,
                matching_out.mask,
                coactive_device,
                truth_count_device,
                predicted_count_device,
            )
            matching_examples += len(matching_batch.active)
        else:
            truth = matching_batch.active.bool().cpu()
            predicted_blocks = matching_out.mask.bool().cpu()
            coactive_device += truth.double().T @ predicted_blocks.double()
            truth_count_device += truth.sum(dim=0).double()
            predicted_count_device += predicted_blocks.sum(dim=0).double()
            matching_examples += len(truth)
    if matching_examples != calibration_stop - calibration_start:
        raise CellExecutionError("factor-calibration stream ended early")
    association = (
        2
        * coactive_device
        / (
            truth_count_device.unsqueeze(1) + predicted_count_device.unsqueeze(0)
        ).clamp_min(1)
    ).cpu()
    best_association, factor_to_group = association.max(dim=1)
    group_to_factor = association.argmax(dim=0)
    factor_to_group_device = factor_to_group.to(device)
    group_to_factor_device = group_to_factor.to(device)
    truth_ranks = tuple(int(metadata.coordinate_dim) for metadata in dataset.factors)
    grouped_factor_to_groups = tuple(
        tuple(
            sorted(
                range(n_groups),
                key=lambda group: (-float(association[factor, group]), group),
            )[: math.ceil(truth_ranks[factor] / model.cfg.block_dim)]
        )
        for factor in range(n_factors)
    )
    max_grouped_width = max(len(groups) for groups in grouped_factor_to_groups)
    grouped_index = torch.zeros(
        n_factors,
        max_grouped_width,
        dtype=torch.long,
        device=device,
    )
    grouped_valid = torch.zeros_like(grouped_index, dtype=torch.bool)
    for factor, groups in enumerate(grouped_factor_to_groups):
        grouped_index[factor, : len(groups)] = torch.tensor(groups, device=device)
        grouped_valid[factor, : len(groups)] = True

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
    truth_bases: list[torch.Tensor | None] = [None] * n_factors
    if subspace_eligible:
        for factor, metadata in enumerate(dataset.factors):
            rank = metadata.coordinate_dim
            columns = maps[factor, :, :, :rank].reshape(-1, rank)
            truth = _orthonormal_columns(columns[coordinate_mask.reshape(-1)])
            if truth.shape[1] == 0:
                continue
            truth_bases[factor] = truth
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
    grouped_overlap: torch.Tensor | None = None
    if subspace_eligible:
        grouped_overlap = torch.zeros(n_factors)
        for factor, groups in enumerate(grouped_factor_to_groups):
            truth = truth_bases[factor]
            if truth is None:
                continue
            grouped_basis = _orthonormal_columns(
                torch.cat(tuple(learned[group] for group in groups), dim=1)
            )
            if grouped_basis.shape[1]:
                grouped_overlap[factor] = (
                    (truth.T @ grouped_basis).square().sum() / truth.shape[1]
                ).float()
    # Fit one affine map from the selected learned block to each planted
    # coordinate using factor-calibration rows only.  Lists contain only codes
    # and intrinsic coordinates, never ambient contributions, so peak memory
    # is independent of the large source dimension.
    alignment_coefficients: list[torch.Tensor | None] = [None] * n_factors
    alignment_references: list[torch.Tensor | None] = [None] * n_factors
    grouped_alignment_coefficients: list[torch.Tensor | None] = [None] * n_factors
    grouped_alignment_references: list[torch.Tensor | None] = [None] * n_factors
    if subspace_eligible:
        latent_parts: list[list[torch.Tensor]] = [[] for _ in range(n_factors)]
        grouped_latent_parts: list[list[torch.Tensor]] = [[] for _ in range(n_factors)]
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
            isolated_out = frozen_select(isolated)
            selected_codes, _ = _gather_event_factor_blocks(
                isolated_out.z_selected,
                None,
                matching_batch.event_factor,
                factor_to_group_device,
            )
            selected_codes = selected_codes.detach().cpu().double()
            for factor in matching_batch.event_factor.unique().tolist():
                rows = torch.nonzero(
                    matching_batch.event_factor == factor, as_tuple=False
                ).flatten()
                latent_parts[factor].append(selected_codes[rows])
                grouped_latent_parts[factor].append(
                    isolated_out.z_selected[rows.to(device),][
                        :, grouped_index[factor, grouped_valid[factor]]
                    ]
                    .reshape(len(rows), -1)
                    .detach()
                    .cpu()
                    .double()
                )
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
            # The default pivoted-QR driver can choose different solutions for
            # a rank-deficient calibration design in persistent versus fresh
            # worker processes.  The SVD driver fixes the canonical
            # minimum-norm solution, preserving byte-exact resume artifacts.
            alignment_coefficients[factor] = torch.linalg.lstsq(
                design,
                target,
                driver="gelsd",
            ).solution
            alignment_references[factor] = target.mean(dim=0, keepdim=True)
            grouped_latent = torch.cat(grouped_latent_parts[factor])
            if len(grouped_latent) < 2 * (grouped_latent.shape[1] + 1):
                continue
            grouped_design = torch.cat(
                (
                    grouped_latent,
                    torch.ones(len(grouped_latent), 1, dtype=torch.float64),
                ),
                dim=1,
            )
            grouped_alignment_coefficients[factor] = torch.linalg.lstsq(
                grouped_design,
                target,
                driver="gelsd",
            ).solution
            grouped_alignment_references[factor] = target.mean(dim=0, keepdim=True)

    category_masks = tuple(
        torch.tensor(
            [factor.category == name for factor in dataset.factors],
            dtype=torch.bool,
            device=device,
        )
        for name in dataset.category_names
    )
    alive = torch.zeros(n_groups, dtype=torch.bool, device=device)
    support_totals_device = torch.zeros(4, dtype=torch.int64, device=device)
    grouped_support_totals_device = torch.zeros(4, dtype=torch.int64, device=device)
    category_totals_device = torch.zeros(
        len(dataset.category_names),
        4,
        dtype=torch.int64,
        device=device,
    )
    isolated_error = torch.zeros(n_factors, dtype=torch.float64)
    isolated_total = torch.zeros(n_factors, dtype=torch.float64)
    code_error = torch.zeros(n_factors, dtype=torch.float64)
    code_total = torch.zeros(n_factors, dtype=torch.float64)
    grouped_code_error = torch.zeros(n_factors, dtype=torch.float64)
    grouped_code_total = torch.zeros(n_factors, dtype=torch.float64)
    grouped_selected_count = torch.zeros(n_factors, dtype=torch.float64)
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
        out = frozen_select(x, observed=batch.observed.to(device))
        truth_active = batch.active.bool().to(device)
        block_mask = out.mask.bool()
        alive |= block_mask.any(dim=0)
        support_batch, category_batch = _mapped_support_confusion_counts(
            truth_active,
            block_mask,
            group_to_factor_device,
            category_masks,
        )
        support_totals_device += support_batch
        category_totals_device += category_batch
        grouped_predicted = block_mask[:, grouped_index.reshape(-1)].reshape(
            len(block_mask), n_factors, max_grouped_width
        )
        grouped_predicted = (grouped_predicted & grouped_valid.unsqueeze(0)).any(dim=2)
        grouped_support_totals_device += torch.stack(
            (
                (grouped_predicted & truth_active).sum(),
                (grouped_predicted & ~truth_active).sum(),
                (~grouped_predicted & truth_active).sum(),
                (~grouped_predicted & ~truth_active).sum(),
            )
        )

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
            selected_codes, selected_masks = _gather_event_factor_blocks(
                isolated_out.z_selected,
                isolated_out.mask,
                batch.event_factor,
                factor_to_group_device,
            )
            selected_codes = selected_codes.detach().cpu().double()
            assert selected_masks is not None
            selected_masks = selected_masks.detach().cpu()
            for factor in batch.event_factor.unique().tolist():
                rows = torch.nonzero(
                    batch.event_factor == factor, as_tuple=False
                ).flatten()
                coefficient = alignment_coefficients[factor]
                reference = alignment_references[factor]
                if coefficient is None or reference is None:
                    continue
                latent = selected_codes[rows]
                design = torch.cat(
                    (latent, torch.ones(len(rows), 1, dtype=torch.float64)), dim=1
                )
                target = batch.coordinates[rows].reshape(len(rows), -1).double()
                prediction = design @ coefficient
                code_error[factor] += (prediction - target).square().sum()
                code_total[factor] += (target - reference).square().sum()
                selected_count[factor] += selected_masks[rows].sum()
                isolated_count[factor] += len(rows)
                grouped_coefficient = grouped_alignment_coefficients[factor]
                grouped_reference = grouped_alignment_references[factor]
                if grouped_coefficient is not None and grouped_reference is not None:
                    event_rows = rows.to(device)
                    grouped_latent = isolated_out.z_selected[event_rows][
                        :, grouped_index[factor, grouped_valid[factor]]
                    ].reshape(len(rows), -1)
                    grouped_design = torch.cat(
                        (
                            grouped_latent.detach().cpu().double(),
                            torch.ones(len(rows), 1, dtype=torch.float64),
                        ),
                        dim=1,
                    )
                    grouped_prediction = grouped_design @ grouped_coefficient
                    grouped_code_error[factor] += (
                        (grouped_prediction - target).square().sum()
                    )
                    grouped_code_total[factor] += (
                        (target - grouped_reference).square().sum()
                    )
                    grouped_selected_count[factor] += (
                        isolated_out.mask[event_rows][
                            :, grouped_index[factor, grouped_valid[factor]]
                        ]
                        .any(dim=1)
                        .sum()
                        .cpu()
                    )
        evaluation_examples += len(batch.x)
    if evaluation_examples != evaluation_stop - evaluation_start:
        raise CellExecutionError("synthetic evaluation stream ended early")

    support_values = support_totals_device.cpu().tolist()
    support_totals = dict(zip(("tp", "fp", "fn", "tn"), support_values, strict=True))
    category_values = category_totals_device.cpu().tolist()
    category_totals = {
        name: dict(zip(("tp", "fp", "fn", "true"), values, strict=True))
        for name, values in zip(
            dataset.category_names,
            category_values,
            strict=True,
        )
    }

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
    grouped_tp, grouped_fp, grouped_fn, grouped_tn = (
        int(item) for item in grouped_support_totals_device.cpu().tolist()
    )
    grouped_support = {
        "true_positive_events": grouped_tp,
        "false_positive_events": grouped_fp,
        "false_negative_events": grouped_fn,
        "true_negative_events": grouped_tn,
        "precision": grouped_tp / max(1, grouped_tp + grouped_fp),
        "recall": grouped_tp / max(1, grouped_tp + grouped_fn),
        "false_discovery_rate": grouped_fp / max(1, grouped_fp + grouped_tp),
        "false_positive_rate": grouped_fp / max(1, grouped_fp + grouped_tn),
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
    grouped_code_r2_by_factor = [
        None
        if grouped_code_total[factor] <= 0
        else 1.0 - float(grouped_code_error[factor] / grouped_code_total[factor])
        for factor in range(n_factors)
    ]
    grouped_selected_coverage_by_factor = [
        None
        if isolated_count[factor] <= 0
        else float(grouped_selected_count[factor] / isolated_count[factor])
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
    grouped_code_r2_after_alignment = (
        None
        if grouped_code_total.sum() <= 0
        else 1.0 - float(grouped_code_error.sum() / grouped_code_total.sum())
    )

    recovered = best_association >= 0.5
    pathologies = _matching_pathologies(
        association,
        strong_cutoff=pathology_strong_cutoff,
        weak_cutoff=pathology_weak_cutoff,
    )
    pathology_sensitivity = [
        {
            "strong_association_cutoff": strong,
            "weak_association_cutoff": weak,
            **_matching_pathologies(
                association,
                strong_cutoff=strong,
                weak_cutoff=weak,
            ),
        }
        for strong, weak in sensitivity_pairs
    ]
    rank_mismatch_factors = [
        {
            "factor": factor,
            "truth_rank": truth_ranks[factor],
            "learner_block_width": model.cfg.block_dim,
            "same_block_linear_information_ceiling": min(
                1.0,
                model.cfg.block_dim / truth_ranks[factor],
            ),
            "same_block_support_group": int(factor_to_group[factor]),
            "grouped_diagnostic_block_count": len(grouped_factor_to_groups[factor]),
            "grouped_diagnostic_groups": list(grouped_factor_to_groups[factor]),
            "same_block_subspace_overlap": (
                None if matched_overlap is None else float(matched_overlap[factor])
            ),
            "grouped_subspace_overlap": (
                None if grouped_overlap is None else float(grouped_overlap[factor])
            ),
            "same_block_aligned_code_r2": code_r2_by_factor[factor],
            "grouped_aligned_code_r2": grouped_code_r2_by_factor[factor],
            "same_block_selected_coverage": selected_group_coverage_by_factor[factor],
            "grouped_selected_coverage": grouped_selected_coverage_by_factor[factor],
        }
        for factor in range(n_factors)
    ]
    return {
        "selection_mode": selection_mode,
        "shared_feature_claim_eligible": shared_feature_claim_eligible,
        "shared_feature_claim_reason": shared_feature_claim_reason,
        "identification_metrics_eligible": subspace_eligible,
        "identification_metrics_ineligible_reason": (
            None
            if subspace_eligible
            else "token_layer_normalization_is_not_a_fixed_linear_factor_map"
        ),
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
        "pathology_association": {
            "contract": pathology_association_contract,
            "primary": {
                "strong_association_cutoff": pathology_strong_cutoff,
                "weak_association_cutoff": pathology_weak_cutoff,
                **pathologies,
            },
            "reporting_only_sensitivity": pathology_sensitivity,
            "sensitivity_changes_primary_gate": False,
        },
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
        "rank_mismatch": {
            "contract": rank_mismatch_contract,
            "same_block_metrics_are_primary": True,
            "same_block_gate_is_ceiling_adjusted": False,
            "grouped_diagnostic_is_promotable": False,
            "grouped_support_confusion": grouped_support,
            "grouped_code_r2_after_alignment": grouped_code_r2_after_alignment,
            "grouped_subspace_overlap_mean": (
                None if grouped_overlap is None else float(grouped_overlap.mean())
            ),
            "factors": rank_mismatch_factors,
        },
        "note": (
            "factor-to-group matching and affine code alignment are fit only on "
            "the declared factor-calibration range, then frozen for the complete "
            "development or confirmation range; unselected events retain zero "
            "codes and therefore penalize R2 rather than being conditioned away. "
            "Rank-mismatch grouped metrics are calibration-frozen diagnostics; "
            "raw same-block metrics remain the qualification headline and gate"
        ),
    }


def _phase1_identification_evidence(
    recovery: Mapping[str, Any],
    threshold_items: Sequence[Sequence[Any]],
    *,
    margin_normalization_contract: str,
) -> dict[str, Any]:
    """Evaluate the preregistered factor-level recovery conjunction."""

    if margin_normalization_contract != (
        "piecewise_available_headroom_signed_margin_v2"
    ):
        raise CellExecutionError("unknown Phase-1 margin-normalization contract")

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
    if recovery.get("identification_metrics_eligible") is False:
        reason = recovery.get("identification_metrics_ineligible_reason")
        if not isinstance(reason, str) or not reason:
            raise CellExecutionError(
                "ineligible Phase-1 identification evidence lacks a reason"
            )
        return {
            "applicable": False,
            "ineligible_reason": reason,
            "thresholds": thresholds,
            "per_factor": [],
            "aggregate": {
                "support_precision_diagnostic": recovery.get("support_precision"),
                "support_recall_diagnostic": recovery.get("support_recall"),
            },
            "checks": {},
            "normalized_margins": {},
            "margin": None,
            "passed": None,
        }
    matching = recovery.get("matching") or {}
    association = matching.get("best_support_association_f1")
    overlap = matching.get("matched_subspace_overlap")
    isolated = recovery.get("global_isolated_input_r2_by_factor")
    aligned = recovery.get("code_r2_after_alignment_by_factor")
    rank_mismatch = recovery.get("rank_mismatch")
    rank_mismatch_factors = (
        rank_mismatch.get("factors") if isinstance(rank_mismatch, Mapping) else None
    )
    n_factors = int(recovery.get("n_truth_factors", -1))
    if not all(
        isinstance(values, list) and len(values) == n_factors
        for values in (association, isolated, aligned)
    ):
        raise CellExecutionError("Phase-1 recovery lacks factor-level evidence")
    if (
        not isinstance(rank_mismatch_factors, list)
        or len(rank_mismatch_factors) != n_factors
    ):
        raise CellExecutionError("Phase-1 recovery lacks rank-mismatch evidence")
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
        observed = float(value)
        denominator = (
            max(1.0 - threshold, 1.0e-12)
            if observed >= threshold
            else max(abs(threshold), 1.0e-12)
        )
        return (observed - threshold) / denominator

    def max_margin(value: Any, threshold: float) -> float:
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            return -1.0e9
        observed = float(value)
        denominator = (
            max(abs(threshold), 1.0e-12)
            if observed <= threshold
            else max(1.0 - threshold, 1.0e-12)
        )
        return (threshold - observed) / denominator

    for factor in range(n_factors):
        metrics = {
            "support_association": association[factor],
            "subspace_overlap": None if not subspace_eligible else overlap[factor],
            "global_isolated_input_r2": isolated[factor],
            "aligned_code_r2": aligned[factor],
            "rank_mismatch": rank_mismatch_factors[factor],
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
                "same_block_gate_has_positive_headroom": bool(
                    float(
                        rank_mismatch_factors[factor][
                            "same_block_linear_information_ceiling"
                        ]
                    )
                    > max(
                        thresholds["per_factor.subspace_overlap_min_when_eligible"],
                        thresholds["per_factor.aligned_code_r2_min"],
                    )
                ),
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
        "applicable": True,
        "ineligible_reason": None,
        "thresholds": thresholds,
        "per_factor": per_factor,
        "aggregate": aggregate,
        "checks": checks,
        "normalized_margins": normalized_margins,
        "margin_normalization_contract": margin_normalization_contract,
        "margin": margin,
        "passed": all(checks.values()),
    }


def _phase1_counterfactual_endpoint(
    endpoint: Mapping[str, Any],
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    if endpoint.get("applicable") is not True:
        return {"passed": None, "recovered_factor_fraction": None}
    per_factor = endpoint["per_factor"]
    identified = 0
    for factor in per_factor:
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
    recovered_fraction = identified / len(per_factor)
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
    return {
        "passed": passed,
        "recovered_factor_fraction": recovered_fraction,
    }


def _phase1_threshold_sensitivity_payload(
    identification: Mapping[str, Any],
    sensitivity: Sequence[Sequence[Any]],
) -> dict[str, Any]:
    center = {
        str(name): float(value)
        for name, value in identification["native"]["thresholds"].items()
    }
    if identification["deployed"]["thresholds"] != center:
        raise CellExecutionError("Phase-1 endpoints disagree on center thresholds")
    rows: list[dict[str, Any]] = []
    if {str(item[0]) for item in sensitivity} != set(center):
        raise CellExecutionError("Phase-1 threshold sensitivity grid is incomplete")
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
            conjunction = (
                None
                if native["passed"] is None and deployed["passed"] is None
                else bool(native["passed"] and deployed["passed"])
            )
            value_rows.append(
                {
                    "value": float(value),
                    "native": native,
                    "deployed": deployed,
                    "conjunction_passed": conjunction,
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
        model._tied_encoder_tensor(materialized_decoder)
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


def _persisted_view_validation_kernel(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> torch.Tensor:
    """Return max absolute and allclose-normalized differences."""

    expected_fp32 = expected.float()
    difference = (actual - expected_fp32).abs()
    tolerance = 0.012 + 0.012 * expected_fp32.abs()
    return torch.stack((difference.amax(), (difference / tolerance).amax()))


@functools.lru_cache(maxsize=1)
def _compiled_cuda_persisted_view_validation():
    return torch.compile(
        _persisted_view_validation_kernel,
        fullgraph=True,
        dynamic=True,
    )


def _persisted_view_validation(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> tuple[float, bool]:
    kernel = (
        _compiled_cuda_persisted_view_validation()
        if actual.device.type == "cuda" and actual.numel() >= 65_536
        else _persisted_view_validation_kernel
    )
    maximum, normalized_maximum = kernel(actual, expected).cpu().tolist()
    return float(maximum), float(normalized_maximum) <= 1.0


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
        "schema": "bsc-deployment-operating-record-v2",
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
    if horizon_tokens <= 0 or not 0 <= upper_tokens < horizon_tokens:
        raise CellExecutionError("time-sharing header has invalid token counts")
    lower_mode = _time_sharing_mode_code(lower_name)
    upper_mode = _time_sharing_mode_code(upper_name)
    if (lower_mode == upper_mode) != (upper_tokens == 0):
        raise CellExecutionError(
            "pure operating records require one endpoint and mixtures require two"
        )
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


@dataclass(frozen=True)
class _RawEndpointErrorCache:
    endpoint_names: tuple[str, ...]
    chunks: tuple[torch.Tensor, ...]  # each [endpoints, batch_tokens] fp64 CPU
    tokens: int
    pooled_denominator: float


def _consume_chunked_rd_evaluation_batch(
    session: _RDEvaluationSession,
    rd_input: _RDEvaluationInput,
    selection: _RDEvaluationSelection,
) -> None:
    """Consume one exact outer selection in bounded, ordered R-D chunks."""

    batch_tokens = len(rd_input.transformed)
    if (
        selection.z.shape[0] != batch_tokens
        or selection.scores.shape[0] != batch_tokens
        or selection.mask.shape[0] != batch_tokens
    ):
        raise CellExecutionError("R-D threshold selection batch is misbound")
    if rd_input.row_ids is not None and len(rd_input.row_ids) != batch_tokens:
        raise CellExecutionError("R-D row IDs are misbound")
    if rd_input.context is not None and (
        not torch.is_tensor(rd_input.context) or len(rd_input.context) != batch_tokens
    ):
        raise CellExecutionError("R-D observer context is not a batch tensor")

    for start in range(0, batch_tokens, RD_EVALUATION_TOKEN_CHUNK):
        token_slice = slice(start, min(start + RD_EVALUATION_TOKEN_CHUNK, batch_tokens))
        session.consume(
            _RDEvaluationInput(
                transformed=rd_input.transformed[token_slice],
                row_ids=(
                    None if rd_input.row_ids is None else rd_input.row_ids[token_slice]
                ),
                context=(
                    None if rd_input.context is None else rd_input.context[token_slice]
                ),
            ),
            threshold_selection=_RDEvaluationSelection(
                selection.z[token_slice],
                selection.scores[token_slice],
                selection.mask[token_slice],
            ),
        )


@torch.no_grad()
def _evaluate_rate_distortion_and_raw_space(
    ctx: _Context,
    preparation: Mapping[str, Any],
    model: BlockCrosscoder,
    codec: Codec,
    deployment: Mapping[str, Any],
    *,
    retain_endpoint_errors: bool = False,
) -> tuple[
    EvaluationModeEndpoints,
    dict[str, Any],
    dict[str, Any],
    _RawEndpointErrorCache | None,
    dict[str, Any],
]:
    """Evaluate transformed and paired raw codec endpoints in one traversal."""

    data = preparation["data"]
    batch_size = int(ctx.values["optimizer.batch_tokens"])
    device = _device(ctx)
    model = model.to(device).eval()
    sites, width = model.cfg.n_sites, model.cfg.d_model
    coordinate_mask = (
        model.coordinate_mask[:, 0, 0].to(device).double()
        if model._has_padded_coordinates
        else None
    )
    errors = {
        q: torch.zeros(sites, dtype=torch.float64, device=device) for q in codec.spec.qs
    }
    endpoint_names = (
        "zero_event_calibration_mean",
        *(f"q{q}" for q in codec.spec.qs),
    )
    endpoint_error_chunks: list[torch.Tensor] = []

    saved_normalization = deployment["normalization"]
    inverse_W: torch.Tensor | None = None
    saved_mean_device: torch.Tensor | None = None
    saved_diagonal_device: torch.Tensor | None = None
    saved_forward_device: torch.Tensor | None = None
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
        on_the_fly = data.get("normalization", {}).get("application") == "on_the_fly"
        if on_the_fly and (
            Path(data["root"]).resolve() != Path(data["raw_root"]).resolve()
            or data["bindings"]["evaluation"] != data["raw_bindings"]["evaluation"]
        ):
            raise CellExecutionError(
                "single-view evaluation root/binding differs from its raw view"
            )
        normalized = None if on_the_fly else _store_reader(preparation, "evaluation")
        raw = _store_reader(preparation, "evaluation", raw=True)
        saved_mode = str(saved_normalization.get("mode", "none"))
        oracle_layer_inverse = (
            saved_normalization.get("kind") == "frozen_transform"
            and saved_mode == "layer"
        )
        if saved_normalization.get("kind") == "frozen_transform":
            saved_mean_device = saved_normalization["mean"].to(
                device=device,
                dtype=torch.float32,
            )
            saved_W_cpu = saved_normalization["W"].to(dtype=torch.float32)
            if saved_mode in {"none", "scalar_rms", "sqrt_d"}:
                saved_diagonal_device = torch.diagonal(
                    saved_W_cpu,
                    dim1=-2,
                    dim2=-1,
                ).to(device=device)
                saved_forward_device = saved_diagonal_device
            elif saved_mode == "whiten":
                saved_forward_device = saved_W_cpu.to(device=device)
                inverse_W = torch.linalg.inv(saved_W_cpu.double()).float().to(device)
            elif saved_mode != "layer":
                saved_forward_device = saved_W_cpu.to(device=device)

        def paired_stream() -> Iterator[tuple[torch.Tensor, ...]]:
            raw_stream = raw.sequential_batches_with_ids(batch_size)
            if on_the_fly:
                for x_raw, raw_ids in raw_stream:
                    yield (x_raw, raw_ids)
                return
            assert normalized is not None
            normalized_stream = normalized.sequential_batches_with_ids(batch_size)
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
                yield (x_normalized, x_raw, raw_ids)

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
    evaluation_stream: Iterator = paired_stream()
    if device.type == "cuda":

        def copy_activation_leaf(tensor: torch.Tensor) -> bool:
            return tensor.is_floating_point()

        evaluation_stream = prefetch_batches(
            evaluation_stream,
            depth=2,
            pin_memory=copy_activation_leaf,
        )
        evaluation_stream = cuda_prefetch_batches(
            evaluation_stream,
            device=device,
            depth=1,
            copy_policy=copy_activation_leaf,
        )

    class _RawObserver:
        def __init__(self) -> None:
            self.tokens = 0
            self.roundtrip: dict[str, Any] | None = None
            self._raw: torch.Tensor | None = None
            self._token_denominator: torch.Tensor | None = None
            self._token_errors: dict[int, torch.Tensor] = {}

        def begin_batch(self, batch: _RDEvaluationBatch) -> None:
            if self._raw is not None or not torch.is_tensor(batch.context):
                raise CellExecutionError("joint R-D observer batch state is invalid")
            x_raw = batch.context
            if (
                x_raw.shape != batch.transformed.shape
                or x_raw.device != _resolved_runtime_device(device)
            ):
                raise CellExecutionError("joint R-D paired raw tensor is misbound")
            centered = x_raw.double() - calibration_mean
            if coordinate_mask is not None:
                centered = centered * coordinate_mask
            token_denominator = centered.square().sum(dim=2)
            denominator.add_(token_denominator.sum(dim=0))
            self._raw = x_raw
            self._token_denominator = token_denominator
            self._token_errors = {}

            if self.roundtrip is None:
                q = max(codec.spec.qs)
                packet = _packet_from_events(codec, batch.packet_events, q)
                decoded = decode_batch(
                    model,
                    codec,
                    packet,
                    _decoder=batch.decoder,
                    _decoder_matrix=batch.decoder_matrix,
                )
                packet_error = (
                    (decoded.float() - batch.transformed.float())
                    .double()
                    .square()
                    .sum()
                )
                packet_centered = (
                    (
                        batch.transformed.float()
                        - codec.calib_mean.to(
                            device=batch.transformed.device,
                            dtype=torch.float32,
                        ).unsqueeze(0)
                    )
                    .double()
                    .square()
                    .sum()
                    .clamp_min(1e-30)
                )
                self.roundtrip = {
                    "source_free_decode": True,
                    "tokens": packet.n_tokens,
                    "events": int(packet.counts.sum()),
                    "finite": bool(torch.isfinite(decoded).all()),
                    "shape_matches": list(decoded.shape)
                    == list(batch.transformed.shape),
                    "quantizer_bits": q,
                    "fvu_pooled": float(packet_error / packet_centered),
                }

        def consume_decoded_chunk(
            self,
            batch: _RDEvaluationBatch,
            decoded_chunk: Mapping[int, torch.Tensor],
        ) -> None:
            del batch
            x_raw = self._raw
            if x_raw is None:
                raise CellExecutionError("joint R-D observer consumed before begin")
            for q, normalized_prediction in decoded_chunk.items():
                if data["kind"] == "synthetic":
                    assert synthetic_normalization is not None
                    if synthetic_normalization["kind"] == "token_layer_norm":
                        raw_prediction = torch.zeros_like(normalized_prediction)
                        for site, dim in enumerate(model.cfg.site_dims):
                            values = x_raw[:, site, :dim]
                            mean = values.mean(dim=-1, keepdim=True)
                            variance = values.var(
                                dim=-1,
                                correction=0,
                                keepdim=True,
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
                self._token_errors[q] = token_error
                errors[q].add_(token_error.sum(dim=0))

        def end_batch(self, batch: _RDEvaluationBatch) -> None:
            x_raw = self._raw
            token_denominator = self._token_denominator
            if x_raw is None or token_denominator is None:
                raise CellExecutionError("joint R-D observer ended before begin")
            if set(self._token_errors) != set(codec.spec.qs):
                raise CellExecutionError("joint R-D observer missed a quantizer")
            if retain_endpoint_errors:
                endpoint_error_chunks.append(
                    torch.stack(
                        (
                            token_denominator.sum(dim=1),
                            *(self._token_errors[q].sum(dim=1) for q in codec.spec.qs),
                        ),
                        dim=0,
                    ).cpu()
                )

            unique_sequences, inverse = torch.unique_consecutive(
                batch.sequence_ids,
                return_inverse=True,
            )
            inverse = inverse.to(device=device, non_blocking=True)
            grouped_denominator = torch.zeros(
                len(unique_sequences),
                sites,
                dtype=torch.float64,
                device=device,
            )
            grouped_denominator.index_add_(0, inverse, token_denominator)
            grouped_errors: dict[int, torch.Tensor] = {}
            for q in codec.spec.qs:
                grouped = torch.zeros_like(grouped_denominator)
                grouped.index_add_(0, inverse, self._token_errors[q])
                grouped_errors[q] = grouped
            sequence_values = unique_sequences.tolist()
            grouped_metrics = torch.stack(
                (
                    grouped_denominator.sum(dim=1),
                    *(grouped_errors[q].sum(dim=1) for q in codec.spec.qs),
                ),
                dim=1,
            ).cpu()
            denominator_values = grouped_metrics[:, 0].tolist()
            error_values = {
                q: grouped_metrics[:, index + 1].tolist()
                for index, q in enumerate(codec.spec.qs)
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
            self.tokens += len(x_raw)
            self._raw = None
            self._token_denominator = None
            self._token_errors = {}

    observer = _RawObserver()

    def joint_inputs() -> Iterator[_RDEvaluationInput]:
        nonlocal persisted_view_max_abs_difference
        nonlocal serialized_forward_verified
        for item in evaluation_stream:
            if data["kind"] == "synthetic":
                x_normalized, x_raw, row_ids = item
                x_raw_device = x_raw.to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
                encoder_input = x_normalized.to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
            elif on_the_fly:
                x_raw, row_ids = item
                x_raw_device = x_raw.to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
                encoder_input = _apply_saved_real_normalization(
                    x_raw_device,
                    saved_normalization,
                    mean=saved_mean_device,
                    operator=saved_forward_device,
                )
                serialized_forward_verified = True
            else:
                x_persisted, x_raw, row_ids = item
                x_raw_device = x_raw.to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
                persisted_device = x_persisted.to(
                    device=device,
                    non_blocking=True,
                )
                # Always reconstruct the encoder input from the priced
                # deployment bytes. The persisted Phase-2 view is only an
                # independent, bounded materialization check.
                encoder_input = _apply_saved_real_normalization(
                    x_raw_device,
                    saved_normalization,
                    mean=saved_mean_device,
                    operator=saved_forward_device,
                )
                difference, agrees = _persisted_view_validation(
                    encoder_input,
                    persisted_device,
                )
                persisted_view_max_abs_difference = max(
                    persisted_view_max_abs_difference,
                    difference,
                )
                if not agrees:
                    raise CellExecutionError(
                        "serialized deployment normalization does not reproduce "
                        "the bound persisted evaluation view"
                    )
                serialized_forward_verified = True
                del persisted_device
            yield _RDEvaluationInput(
                transformed=encoder_input,
                row_ids=row_ids,
                context=x_raw_device,
            )
            del encoder_input, x_raw_device

    evaluation_decoder = model.decoder_tensor()
    evaluation_encoder = (
        model._tied_encoder_tensor(evaluation_decoder)
        if model.cfg.encoder_mode == "tied"
        else model.encoder_tensor()
    )
    evaluation_score_geometry = model._frozen_score_geometry(evaluation_decoder)
    direct_factorized = model.uses_direct_factorized_execution
    rd_session = _RDEvaluationSession(
        model,
        codec,
        row_len=1 if data["kind"] == "synthetic" else None,
        device=str(device),
        observer=observer,
        materialized_decoder=None if direct_factorized else evaluation_decoder,
        materialized_encoder=None if direct_factorized else evaluation_encoder,
        score_geometry=None if direct_factorized else evaluation_score_geometry,
    )
    current_rd_input: _RDEvaluationInput | None = None

    def mode_inputs() -> Iterator[torch.Tensor]:
        nonlocal current_rd_input
        for rd_input in joint_inputs():
            if current_rd_input is not None:
                raise CellExecutionError(
                    "joint evaluator advanced before consuming its R-D batch"
                )
            current_rd_input = rd_input
            yield rd_input.transformed
            if current_rd_input is not None:
                raise CellExecutionError(
                    "joint evaluator omitted the current R-D batch"
                )

    def consume_threshold_batch(
        transformed: torch.Tensor,
        z: torch.Tensor,
        scores: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        nonlocal current_rd_input
        rd_input = current_rd_input
        if rd_input is None or transformed is not rd_input.transformed:
            raise CellExecutionError("joint selector/R-D batch identity diverged")
        _consume_chunked_rd_evaluation_batch(
            rd_session,
            rd_input,
            _RDEvaluationSelection(z, scores, mask),
        )
        current_rd_input = None

    try:
        mode_endpoints = evaluate_selector_and_shared_code_modes(
            model,
            mode_inputs(),
            device=device,
            selection_modes=("topk", "threshold"),
            _threshold_batch_consumer=consume_threshold_batch,
            _materialized_decoder=evaluation_decoder,
            _materialized_encoder=evaluation_encoder,
            _score_geometry=evaluation_score_geometry,
        )
        if current_rd_input is not None:
            raise CellExecutionError("joint evaluator left an R-D batch unconsumed")
        rd = rd_session.finalize()
    finally:
        rd_session.close()
        close_evaluation = getattr(evaluation_stream, "close", None)
        if close_evaluation is not None:
            close_evaluation()
    tokens = observer.tokens
    if tokens == 0:
        raise CellExecutionError("raw-space evaluation stream is empty")
    if observer.roundtrip is None:
        raise CellExecutionError("joint R-D evaluation omitted packet roundtrip")
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
    pooled_denominator = float(denominator.sum())
    payload = {
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
        "operational_time_sharing": {},
        "oracle_side_information": oracle_layer_inverse,
        "reason": (
            "token LayerNorm inverse uses unpriced source-token mean/variance"
            if oracle_layer_inverse
            else "paired row-identical raw view and invertible frozen transform"
        ),
    }
    cache = (
        _RawEndpointErrorCache(
            endpoint_names=endpoint_names,
            chunks=tuple(endpoint_error_chunks),
            tokens=tokens,
            pooled_denominator=pooled_denominator,
        )
        if retain_endpoint_errors
        else None
    )
    return mode_endpoints, rd, payload, cache, observer.roundtrip


@torch.no_grad()
def _evaluate_cached_time_sharing(
    cache: _RawEndpointErrorCache,
    plans: Mapping[str, Mapping[str, Any]],
    *,
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    """Execute balanced schedules from first-pass paired raw endpoint errors."""

    resolved = {str(key): dict(plan) for key, plan in plans.items()}
    if not resolved:
        return {}
    name_to_index = {name: index for index, name in enumerate(cache.endpoint_names)}
    if len(name_to_index) != len(cache.endpoint_names):
        raise CellExecutionError("raw endpoint cache repeats an endpoint name")
    horizons = {int(plan["horizon_tokens"]) for plan in resolved.values()}
    if len(horizons) != 1:
        raise CellExecutionError("deployment schedules do not share one common horizon")
    horizon = next(iter(horizons))
    if horizon <= 0 or cache.tokens > horizon:
        raise CellExecutionError(
            "cached raw evaluation exceeds the deployment schedule horizon"
        )
    for plan in resolved.values():
        if (
            str(plan["lower_name"]) not in name_to_index
            or str(plan["upper_name"]) not in name_to_index
        ):
            raise CellExecutionError(
                "deployment schedule names an endpoint absent from the raw cache"
            )

    errors = {
        key: torch.zeros((), dtype=torch.float64, device=device) for key in resolved
    }
    # The balanced predicate is the difference of consecutive floor values,
    # so its prefix sum is exact without synchronizing a CUDA count per chunk.
    upper_counts = {
        key: cache.tokens * int(plan["upper_tokens"]) // horizon
        for key, plan in resolved.items()
    }
    token_offset = 0
    for cpu_chunk in cache.chunks:
        if (
            cpu_chunk.device.type != "cpu"
            or cpu_chunk.dtype != torch.float64
            or cpu_chunk.ndim != 2
            or cpu_chunk.shape[0] != len(cache.endpoint_names)
        ):
            raise CellExecutionError("raw endpoint cache chunk is malformed")
        chunk = cpu_chunk.to(device=device, non_blocking=True)
        chunk_tokens = chunk.shape[1]
        indices = torch.arange(
            token_offset,
            token_offset + chunk_tokens,
            device=device,
            dtype=torch.int64,
        )
        masks: dict[int, torch.Tensor] = {}
        for key, plan in resolved.items():
            upper_tokens = int(plan["upper_tokens"])
            mask = masks.get(upper_tokens)
            if mask is None:
                mask = ((indices + 1) * upper_tokens) // horizon > (
                    indices * upper_tokens
                ) // horizon
                masks[upper_tokens] = mask
            lower = chunk[name_to_index[str(plan["lower_name"])]]
            upper = chunk[name_to_index[str(plan["upper_name"])]]
            errors[key] += torch.where(mask, upper, lower).sum()
        token_offset += chunk_tokens
    if token_offset != cache.tokens:
        raise CellExecutionError("raw endpoint cache token count is inconsistent")
    if not math.isfinite(cache.pooled_denominator) or cache.pooled_denominator <= 0:
        raise CellExecutionError("raw endpoint cache denominator is invalid")
    return {
        key: {
            **plan,
            "evaluation_tokens": cache.tokens,
            "evaluation_upper_tokens": upper_counts[key],
            "raw_space_fvu": float(errors[key] / cache.pooled_denominator),
            "distortion_measurement": (
                "executed_balanced_schedule_on_paired_raw_evaluation_rows"
            ),
        }
        for key, plan in resolved.items()
    }


def _finalize_development_time_sharing(
    plans: Mapping[str, Mapping[str, Any]],
    measurements: Mapping[str, Mapping[str, Any]],
    *,
    rd: Mapping[str, Any],
    raw_space: Mapping[str, Any],
    deployment_artifact_size_bytes: int,
    horizon_tokens: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Retain a feasible lower endpoint when its executed mixture is no better.

    The endpoint errors and the operational mixtures are measured on the same
    paired raw rows.  Reusing the already accumulated lower-endpoint result
    avoids a second traversal of the potentially large evaluation cache.
    """

    measured, _hull, _side_rate = _fixed_rate_hull(
        rd=rd,
        raw_space=raw_space,
        deployment_artifact_size_bytes=deployment_artifact_size_bytes,
        horizon_tokens=horizon_tokens,
    )
    endpoints = {str(point["name"]): point for point in measured}
    schedule_rate = 8.0 * TIME_SHARING_HEADER_BYTES / horizon_tokens
    finalized_plans: dict[str, dict[str, Any]] = {}
    finalized_measurements: dict[str, dict[str, Any]] = {}
    seen_budgets: set[float] = set()
    for provisional_key, raw_plan in plans.items():
        plan = dict(raw_plan)
        measured_schedule = measurements.get(str(provisional_key))
        if not isinstance(measured_schedule, Mapping):
            raise CellExecutionError(
                "development operating plan lacks its executed measurement"
            )
        expected_binding = {
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
        if any(
            measured_schedule.get(name) != expected
            for name, expected in expected_binding.items()
        ):
            raise CellExecutionError(
                "development operating measurement/plan binding mismatch"
            )
        budget = float(plan["budget_bits_per_token"])
        if budget in seen_budgets:
            raise CellExecutionError("development operating plans repeat a budget")
        seen_budgets.add(budget)
        lower_name = str(plan["lower_name"])
        lower = endpoints.get(lower_name)
        if lower is None or lower.get("q") != plan["lower_q"]:
            raise CellExecutionError(
                "development operating plan names an unavailable lower endpoint"
            )
        scheduled_fvu = float(measured_schedule["raw_space_fvu"])
        lower_fvu = float(lower["raw_space_fvu"])
        if str(plan["upper_name"]) != lower_name and not scheduled_fvu < lower_fvu:
            plan = {
                "budget_bits_per_token": budget,
                "lower_name": lower_name,
                "lower_q": lower["q"],
                "upper_name": lower_name,
                "upper_q": lower["q"],
                "upper_tokens": 0,
                "horizon_tokens": horizon_tokens,
                "upper_mixture_weight": 0.0,
                "achieved_total_bits_per_token": (
                    float(lower["total_bits_per_token"]) + schedule_rate
                ),
            }
            final_key = _time_sharing_plan_key(
                budget=budget,
                lower_name=lower_name,
                upper_name=lower_name,
                upper_tokens=0,
                horizon_tokens=horizon_tokens,
            )
            measured_schedule = {
                **plan,
                "evaluation_tokens": int(measured_schedule["evaluation_tokens"]),
                "evaluation_upper_tokens": 0,
                "raw_space_fvu": lower_fvu,
                "distortion_measurement": (
                    "retained_lower_endpoint_from_paired_raw_evaluation_rows"
                ),
            }
        else:
            final_key = str(provisional_key)
            measured_schedule = dict(measured_schedule)
        if final_key in finalized_plans:
            raise CellExecutionError("development operating plans repeat a key")
        if float(plan["achieved_total_bits_per_token"]) > budget + 1e-12:
            raise CellExecutionError(
                "development lower-endpoint fallback exceeded its fixed budget"
            )
        finalized_plans[final_key] = plan
        finalized_measurements[final_key] = dict(measured_schedule)
    return finalized_plans, finalized_measurements


def _bind_deployment_schedule_measurements(
    plans: Mapping[str, Mapping[str, Any]],
    measurements: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Attach reloaded immutable record bindings to prior raw measurements."""

    if set(plans) != set(measurements):
        raise CellExecutionError(
            "deployment operating plans and raw measurements cover different keys"
        )
    bound: dict[str, dict[str, Any]] = {}
    plan_fields = (
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
    measurement_fields = (
        "evaluation_tokens",
        "evaluation_upper_tokens",
        "raw_space_fvu",
        "distortion_measurement",
    )
    for key, raw_plan in plans.items():
        plan = dict(raw_plan)
        measurement = measurements[key]
        if any(measurement.get(name) != plan.get(name) for name in plan_fields):
            raise CellExecutionError(
                "deployment operating record differs from its raw measurement"
            )
        bound[key] = {
            **plan,
            **{name: measurement[name] for name in measurement_fields},
        }
    return bound


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


def _qualification_operating_policy(
    qualification: Mapping[str, Any],
    *,
    cell_id: str,
) -> tuple[float, dict[str, Any]]:
    if qualification.get("cell_id") != cell_id:
        raise CellExecutionError("frozen parent qualification/cell binding mismatch")
    validation = qualification.get("validation")
    policy = qualification.get("fixed_rate_operating_policy")
    if not isinstance(validation, Mapping) or not isinstance(policy, Mapping):
        raise CellExecutionError(
            "frozen parent qualification lacks its fixed-rate operating policy"
        )
    metric = validation.get(PHASE2_SELECTION_METRIC_KEY)
    if (
        not isinstance(metric, (int, float))
        or isinstance(metric, bool)
        or not math.isfinite(float(metric))
    ):
        raise CellExecutionError("frozen parent qualification has no finite metric")
    return float(metric), dict(policy)


def _load_parent_qualification_policies(
    ctx: _Context,
) -> list[tuple[float, str, dict[str, Any]]]:
    parent_ids = tuple(str(item) for item in ctx.values["selection.parent_cell_ids"])
    expected_hashes = {
        str(item).removeprefix("sha256:")
        for item in ctx.values["selection.qualification_sha256s"]
    }
    if not parent_ids or len(expected_hashes) != len(parent_ids):
        raise CellExecutionError(
            "frozen operating policy requires exact parent qualification hashes"
        )
    resolved: list[tuple[float, str, dict[str, Any]]] = []
    if ctx.cell.phase is Phase.PHASE3:
        panel = _read_object(
            ctx.root / "panel-decision.json", label="Phase-3 panel decision"
        )
        campaign_manifest = panel.get("phase2_campaign_manifest")
        cells = (
            campaign_manifest.get("cells")
            if isinstance(campaign_manifest, Mapping)
            else None
        )
        if not isinstance(cells, list):
            raise CellExecutionError("Phase-3 panel lacks embedded Phase-2 evidence")
        by_id = {
            str(item.get("cell_id")): item
            for item in cells
            if isinstance(item, Mapping)
        }
        for parent_id in parent_ids:
            evidence = by_id.get(parent_id)
            if not isinstance(evidence, Mapping):
                raise CellExecutionError(
                    "Phase-3 panel omits a frozen parent qualification"
                )
            qualification = evidence.get("qualification")
            if not isinstance(qualification, Mapping):
                raise CellExecutionError("Phase-3 parent qualification is malformed")
            metric, policy = _qualification_operating_policy(
                qualification, cell_id=parent_id
            )
            resolved.append((metric, parent_id, policy))
        return resolved

    for parent_id in parent_ids:
        try:
            parent_record = ctx.campaign.record(parent_id)
        except CampaignError as exc:
            raise CellExecutionError(
                f"cannot replay frozen parent campaign state: {exc}"
            ) from exc
        ref = parent_record.artifact_map.get("qualification")
        if ref is None:
            raise CellExecutionError("selected parent is not qualified")
        path = ref.resolve(ctx.root)
        observed_hash = _sha256(path)
        if observed_hash != ref.sha256 or observed_hash not in expected_hashes:
            raise CellExecutionError(
                "selected parent qualification hash differs from the frozen selection"
            )
        qualification = _read_object(path, label="parent qualification")
        metric, policy = _qualification_operating_policy(
            qualification, cell_id=parent_id
        )
        resolved.append((metric, parent_id, policy))
    return resolved


def _frozen_fixed_rate_operating_policy(
    ctx: _Context,
) -> dict[str, Any] | None:
    if ctx.cell.phase is Phase.PHASE1:
        return None
    if (
        ctx.cell.phase is not Phase.PHASE3
        and ctx.values["evaluation.split"] != "confirmation"
    ):
        return None
    policies = _load_parent_qualification_policies(ctx)
    if not policies:
        raise CellExecutionError("holdout evaluation has no frozen parent policy")
    # Higher selection metrics are better. Replaying the worst source seed's
    # endpoint policy is the preregistered conservative aggregate and is
    # deterministic under cell-ID ties.
    _metric, parent_id, selected = min(policies, key=lambda item: (item[0], item[1]))
    result = {
        "schema": FIXED_RATE_OPERATING_POLICY_SCHEMA,
        "source_cell_id": parent_id,
        "source_evidence_sha256": selected.get("source_evidence_sha256"),
        "aggregation": "worst_seed_frozen_parent",
        "rows": selected.get("rows"),
    }
    result["content_sha256"] = hashlib.sha256(
        canonical_json(result).encode("utf-8")
    ).hexdigest()
    return result


def _selected_time_sharing_plans(
    ctx: _Context,
    *,
    rd: Mapping[str, Any],
    raw_space: Mapping[str, Any],
    deployment_artifact_size_bytes: int,
    frozen_operating_policy: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Resolve one priced operating record for every eligible fixed budget.

    Development evidence may choose a lower-envelope bracket. Confirmation
    and final evaluation pass a frozen policy and may use current serialized
    rates only to derive the largest integer mixture that fits the budget.
    They never inspect current distortion to choose endpoint identities.
    """

    if ctx.cell.phase is Phase.PHASE1 or raw_space.get("eligible") is not True:
        return {}
    horizon = int(ctx.values["evaluation.side_information_amortization_tokens"])
    budgets = tuple(
        float(item)
        for item in ctx.values["evaluation.fixed_rate_budgets_bits_per_token"]
    )
    if horizon <= 0 or not budgets:
        raise CellExecutionError("real time-sharing plan has invalid rate inputs")
    measured, hull, _ = _fixed_rate_hull(
        rd=rd,
        raw_space=raw_space,
        deployment_artifact_size_bytes=deployment_artifact_size_bytes,
        horizon_tokens=horizon,
    )
    schedule_rate = 8.0 * TIME_SHARING_HEADER_BYTES / horizon
    endpoints = {str(point["name"]): point for point in measured}
    frozen_rows: dict[float, Mapping[str, Any]] = {}
    if frozen_operating_policy is not None:
        expected_policy_keys = {
            "schema",
            "source_cell_id",
            "source_evidence_sha256",
            "aggregation",
            "rows",
            "content_sha256",
        }
        raw_rows = frozen_operating_policy.get("rows")
        content = {
            key: frozen_operating_policy[key]
            for key in expected_policy_keys.difference({"content_sha256"})
            if key in frozen_operating_policy
        }
        if (
            set(frozen_operating_policy) != expected_policy_keys
            or frozen_operating_policy.get("schema")
            != FIXED_RATE_OPERATING_POLICY_SCHEMA
            or frozen_operating_policy.get("aggregation")
            not in {"development_cell", "worst_seed_frozen_parent"}
            or not isinstance(raw_rows, list)
            or frozen_operating_policy.get("content_sha256")
            != hashlib.sha256(canonical_json(content).encode("utf-8")).hexdigest()
        ):
            raise CellExecutionError("frozen fixed-rate operating policy is malformed")
        expected_row_keys = {
            "budget_bits_per_token",
            "lower_name",
            "lower_q",
            "upper_name",
            "upper_q",
        }
        for row in raw_rows:
            if not isinstance(row, Mapping) or set(row) != expected_row_keys:
                raise CellExecutionError(
                    "frozen fixed-rate operating policy has a malformed row"
                )
            budget = float(row["budget_bits_per_token"])
            if not math.isfinite(budget) or budget in frozen_rows:
                raise CellExecutionError(
                    "frozen fixed-rate operating policy repeats a budget"
                )
            frozen_rows[budget] = row
        if set(frozen_rows) != set(budgets):
            raise CellExecutionError(
                "frozen fixed-rate operating policy does not cover exact budgets"
            )
    plans: dict[str, dict[str, Any]] = {}
    for budget in budgets:
        available_rate = budget - schedule_rate
        if frozen_rows:
            row = frozen_rows[budget]
            lower = endpoints.get(str(row["lower_name"]))
            upper = endpoints.get(str(row["upper_name"]))
            if lower is None or upper is None:
                raise CellExecutionError(
                    "frozen operating policy names an unavailable codec endpoint"
                )
            if lower.get("q") != row["lower_q"] or upper.get("q") != row["upper_q"]:
                raise CellExecutionError(
                    "frozen operating policy endpoint/q binding mismatch"
                )
        else:
            if available_rate < float(hull[0]["total_bits_per_token"]):
                continue
            if available_rate >= float(hull[-1]["total_bits_per_token"]):
                lower = upper = hull[-1]
            else:
                upper_index = next(
                    index
                    for index, point in enumerate(hull)
                    if float(point["total_bits_per_token"]) >= available_rate
                )
                exact_endpoint = hull[upper_index]
                if math.isclose(
                    float(exact_endpoint["total_bits_per_token"]),
                    available_rate,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    lower = upper = exact_endpoint
                else:
                    lower = hull[max(0, upper_index - 1)]
                    upper = exact_endpoint
        lower_rate = float(lower["total_bits_per_token"])
        upper_rate = float(upper["total_bits_per_token"])
        if str(lower["name"]) == str(upper["name"]):
            upper_tokens = 0
            weight = 0.0
            achieved_rate = lower_rate + schedule_rate
        else:
            target_weight = (available_rate - lower_rate) / (upper_rate - lower_rate)
            upper_tokens = math.floor(target_weight * horizon)
            if upper_tokens <= 0:
                if frozen_rows:
                    raise CellExecutionError(
                        "frozen mixture no longer fits one upper-endpoint token"
                    )
                upper = lower
                upper_tokens = 0
                weight = 0.0
                achieved_rate = lower_rate + schedule_rate
            else:
                if upper_tokens >= horizon:
                    raise CellExecutionError(
                        "time-sharing mixture selected no lower endpoint"
                    )
                weight = upper_tokens / horizon
                achieved_rate = (
                    (1.0 - weight) * lower_rate + weight * upper_rate + schedule_rate
                )
        if achieved_rate > budget + 1e-12:
            raise CellExecutionError(
                "selected operating record exceeded its fixed-rate budget"
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
    frozen_operating_policy: Mapping[str, Any] | None = None,
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
            "schema": "bsc-fixed-rate-raw-selection-v2",
            "applicable": False,
            "eligible": False,
            "operating_policy": None,
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
    records_by_budget = {
        float(record["budget_bits_per_token"]): record
        for record in raw_schedule_records
        if isinstance(record, Mapping)
    }
    if len(records_by_budget) != len(raw_schedule_records):
        raise CellExecutionError("deployment operating records repeat a budget")
    scheduled_results = raw_space.get("operational_time_sharing", {})
    if not isinstance(scheduled_results, Mapping):
        raise CellExecutionError("raw evaluation lacks operating-record results")
    for budget in budgets:
        entry: dict[str, Any] = {"budget_bits_per_token": budget}
        record = records_by_budget.get(budget)
        if raw_reason is not None or record is None:
            entry.update(
                {
                    "eligible": False,
                    "raw_space_fvu": None,
                    "bracket": None,
                    "upper_mixture_weight": None,
                    "achieved_total_bits_per_token": None,
                    "operating_record": None,
                    "reason": raw_reason or "budget_outside_frozen_operating_policy",
                }
            )
            fixed.append(entry)
            continue
        schedule_key = str(record["schedule_key"])
        scheduled = scheduled_results.get(schedule_key)
        if not isinstance(scheduled, Mapping):
            raise CellExecutionError(
                "frozen operating policy was not executed on raw evaluation rows"
            )
        expected_schedule = {
            "budget_bits_per_token": budget,
            "lower_name": str(record["lower_name"]),
            "upper_name": str(record["upper_name"]),
            "upper_tokens": int(record["upper_tokens"]),
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
        keyed_record = schedule_records.get(schedule_key)
        if not isinstance(schedule_binding, Mapping) or keyed_record is not record:
            raise CellExecutionError(
                "executed operating policy lacks its serialized record"
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
        lower_endpoint = next(
            (
                point
                for point in measured
                if str(point["name"]) == str(record["lower_name"])
            ),
            None,
        )
        if lower_endpoint is None:
            raise CellExecutionError(
                "deployment operating record names an unmeasured lower endpoint"
            )
        if (
            frozen_operating_policy is None
            and record["lower_name"] != record["upper_name"]
            and not fvu < float(lower_endpoint["raw_space_fvu"])
        ):
            raise CellExecutionError(
                "development operating policy failed to retain its better "
                "lower endpoint"
            )
        entry.update(
            {
                "eligible": True,
                "raw_space_fvu": fvu,
                "bracket": [record["lower_name"], record["upper_name"]],
                "upper_mixture_weight": float(record["upper_mixture_weight"]),
                "achieved_total_bits_per_token": float(
                    record["achieved_total_bits_per_token"]
                ),
                "operating_record": {
                    "contract": values["codec.time_sharing_schedule_contract"],
                    "header_bytes": TIME_SHARING_HEADER_BYTES,
                    "header_layout": TIME_SHARING_HEADER_LAYOUT_DESCRIPTION,
                    "artifact_sha256": schedule_binding["artifact_sha256"],
                    "record_index": schedule_binding["record_index"],
                    "record_offset_bytes": schedule_binding["offset_bytes"],
                    "record_sha256": schedule_binding["record_sha256"],
                    "binding_magic_u64": schedule_binding["binding_magic_u64"],
                    "horizon_tokens": horizon,
                    "upper_tokens": int(record["upper_tokens"]),
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
                "reason": (
                    "frozen_parent_operating_policy_replayed"
                    if frozen_operating_policy is not None
                    else "development_envelope_operating_policy_selected"
                ),
            }
        )
        fixed.append(entry)
    eligible = (
        raw_reason is None
        and len(fixed) == len(budgets)
        and all(item["eligible"] for item in fixed)
    )
    score = (
        -sum(float(item["raw_space_fvu"]) for item in fixed) / len(fixed)
        if eligible
        else None
    )
    policy: dict[str, Any] | None
    if frozen_operating_policy is not None:
        policy = dict(frozen_operating_policy)
    elif eligible:
        rows = [
            {
                "budget_bits_per_token": float(item["budget_bits_per_token"]),
                "lower_name": str(item["bracket"][0]),
                "lower_q": (
                    None
                    if item["bracket"][0] == "zero_event_calibration_mean"
                    else _time_sharing_mode_code(str(item["bracket"][0]))
                ),
                "upper_name": str(item["bracket"][1]),
                "upper_q": (
                    None
                    if item["bracket"][1] == "zero_event_calibration_mean"
                    else _time_sharing_mode_code(str(item["bracket"][1]))
                ),
            }
            for item in fixed
        ]
        source_evidence = {
            "cell_id": ctx.cell.cell_id,
            "rate_distortion_points": rd["points"],
            "raw_space_points": raw_space["points"],
            "operating_policy_rows": rows,
        }
        policy = {
            "schema": FIXED_RATE_OPERATING_POLICY_SCHEMA,
            "source_cell_id": ctx.cell.cell_id,
            "source_evidence_sha256": hashlib.sha256(
                canonical_json(source_evidence).encode("utf-8")
            ).hexdigest(),
            "aggregation": "development_cell",
            "rows": rows,
        }
        policy["content_sha256"] = hashlib.sha256(
            canonical_json(policy).encode("utf-8")
        ).hexdigest()
    else:
        policy = None
    payload = {
        "schema": "bsc-fixed-rate-raw-selection-v2",
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
            "operating_record_bytes_per_budget": TIME_SHARING_HEADER_BYTES,
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
        "operating_policy": policy,
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
) -> dict[str, bool | float | None]:
    """Map executor endpoints into the schema consumed by live policies."""

    if phase is Phase.PHASE1:
        applicable = bool(
            identification is not None
            and all(
                identification.get(endpoint, {}).get("applicable") is True
                for endpoint in ("native", "deployed")
            )
        )
        return {
            "phase1_identification_applicable": applicable,
            "phase1_identification_conjunction": bool(
                applicable
                and identification["native"]["passed"]
                and identification["deployed"]["passed"]
            ),
            "phase1_identification_margin": (
                None
                if not applicable
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


def _phase1_identification_outcome(
    phase: Phase,
    identification: Mapping[str, Any] | None,
    validation: Mapping[str, Any],
) -> tuple[bool, dict[str, str]]:
    """Evaluate or explicitly neutralize the Phase-1 identification check."""

    if phase is not Phase.PHASE1:
        return True, {}
    if not isinstance(identification, Mapping):
        return False, {}
    endpoints = tuple(identification.get(name, {}) for name in ("native", "deployed"))
    if all(endpoint.get("applicable") is False for endpoint in endpoints):
        reasons = {endpoint.get("ineligible_reason") for endpoint in endpoints}
        if len(reasons) == 1:
            reason = next(iter(reasons))
            if isinstance(reason, str) and reason:
                return True, {"phase1_identification": reason}
        return False, {}
    passed = all(
        endpoint.get("applicable") is True and endpoint.get("passed") is True
        for endpoint in endpoints
    )
    return (
        bool(
            passed
            and validation.get("phase1_identification_applicable") is True
            and validation.get("phase1_identification_conjunction") is True
        ),
        {},
    )


def _evaluate(
    ctx: _Context,
    prerequisites: Mapping[str, tuple[Path, str]],
    *,
    execution_cache: _StageExecutionCache | None = None,
) -> tuple[tuple[str, Path], ...]:
    preparation = _load_preparation(prerequisites["preparation"][0], ctx)
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
    deployment_before_load = ctx.prerequisite_fingerprint(
        deployment_path,
        sha256=deployment_hash,
    )
    _, resolved_model_cfg = _verified_training_report_model_cfg(
        ctx,
        prerequisites,
        checkpoint_hash=checkpoint_hash,
    )
    retained_deployment = (
        None
        if execution_cache is None
        else execution_cache.take_deployment(
            _retained_artifact_key(
                ctx,
                producer_stage="calibrate",
                consumer_stage="evaluate",
                artifact_kind="deployment_codec",
                path=deployment_path,
                sha256=deployment_hash,
                fingerprint=deployment_before_load,
                model_cfg=resolved_model_cfg,
            )
        )
    )
    device = _device(ctx)
    if retained_deployment is None:
        (
            deployment,
            model,
            codec,
            training_summary,
            _verified_deployment_snapshot_digest,
        ) = _load_deployable_codec(
            deployment_path,
            cell_id=ctx.cell.cell_id,
            checkpoint_hash=checkpoint_hash,
            calibration_hash=calibration_hash,
            preparation_hash=prerequisites["preparation"][1],
            device=device,
        )
    else:
        deployment, model, codec, training_summary = retained_deployment
        if next(model.parameters()).device != _resolved_runtime_device(device):
            raise CellExecutionError("retained deployment model is on the wrong device")
        model.eval()
    deployment_after_load = ctx.prerequisite_fingerprint(
        deployment_path,
        sha256=deployment_hash,
    )
    if deployment_after_load != deployment_before_load:
        raise CellExecutionError(
            "deployable codec changed while loading for evaluation"
        )
    if (
        calibration_record.get("deployment_codec_sha256") != deployment_hash
        or calibration_record.get("deployment_codec_size_bytes")
        != deployment_path.stat().st_size
    ):
        raise CellExecutionError("deployable codec/input binding mismatch")

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
                rank_mismatch_contract=str(
                    ctx.values["evaluation.rank_mismatch_contract"]
                ),
                pathology_association_contract=str(
                    ctx.values["evaluation.pathology_association_contract"]
                ),
                pathology_strong_cutoff=float(
                    ctx.values["evaluation.pathology_strong_association_cutoff"]
                ),
                pathology_weak_cutoff=float(
                    ctx.values["evaluation.pathology_weak_association_cutoff"]
                ),
                pathology_cutoff_sensitivity=ctx.values[
                    "evaluation.pathology_association_cutoff_sensitivity"
                ],
            )
            for mode in ("native", "deployed")
        }
        identification = {
            endpoint: _phase1_identification_evidence(
                recovery[endpoint],
                ctx.values["qualification.phase1_identification_thresholds"],
                margin_normalization_contract=str(
                    ctx.values["evaluation.phase1_margin_normalization"]
                ),
            )
            for endpoint in ("native", "deployed")
        }
    mode_endpoints, rd, raw_space, raw_endpoint_cache, roundtrip = (
        _evaluate_rate_distortion_and_raw_space(
            ctx,
            preparation,
            model,
            codec,
            deployment,
            retain_endpoint_errors=True,
        )
    )
    native = mode_endpoints.selector["topk"]
    deployed = mode_endpoints.selector["threshold"]
    shared_native = mode_endpoints.shared_code["topk"]
    shared_deployed = mode_endpoints.shared_code["threshold"]
    assert raw_endpoint_cache is not None
    frozen_operating_policy = _frozen_fixed_rate_operating_policy(ctx)
    schedule_plans = _selected_time_sharing_plans(
        ctx,
        rd=rd,
        raw_space=raw_space,
        deployment_artifact_size_bytes=deployment_path.stat().st_size,
        frozen_operating_policy=frozen_operating_policy,
    )
    schedule_measurements: dict[str, dict[str, Any]] | None = None
    if schedule_plans and frozen_operating_policy is None:
        schedule_measurements = _evaluate_cached_time_sharing(
            raw_endpoint_cache,
            schedule_plans,
            device=device,
        )
        schedule_plans, schedule_measurements = _finalize_development_time_sharing(
            schedule_plans,
            schedule_measurements,
            rd=rd,
            raw_space=raw_space,
            deployment_artifact_size_bytes=deployment_path.stat().st_size,
            horizon_tokens=int(
                ctx.values["evaluation.side_information_amortization_tokens"]
            ),
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
        if schedule_measurements is None:
            raw_space["operational_time_sharing"] = _evaluate_cached_time_sharing(
                raw_endpoint_cache,
                loaded_schedule_plans,
                device=device,
            )
        else:
            raw_space["operational_time_sharing"] = (
                _bind_deployment_schedule_measurements(
                    loaded_schedule_plans,
                    schedule_measurements,
                )
            )
    del raw_endpoint_cache
    fixed_rate = _fixed_rate_raw_score(
        ctx,
        rd=rd,
        raw_space=raw_space,
        deployment_path=deployment_path,
        deployment_hash=deployment_hash,
        calibration_hash=calibration_hash,
        deployment_schedule_manifest=deployment_schedule_manifest,
        frozen_operating_policy=frozen_operating_policy,
    )
    validation = _selection_validation_metrics(
        ctx.cell.phase,
        identification=identification,
        fixed_rate=fixed_rate,
    )
    phase1_threshold_sensitivity = (
        None
        if identification is None
        else _phase1_threshold_sensitivity_payload(
            identification,
            ctx.values["qualification.phase1_threshold_sensitivity"],
        )
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
        "phase1_threshold_sensitivity": phase1_threshold_sensitivity,
        "fixed_rate_raw_selection": fixed_rate,
    }
    selection_metrics_sha256 = hashlib.sha256(
        canonical_json(selection_metrics).encode("utf-8")
    ).hexdigest()
    payload = {
        "schema": EVALUATION_SCHEMA,
        "evaluation_execution_implementation": (EVALUATION_EXECUTION_IMPLEMENTATION),
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
        "phase1_threshold_sensitivity": phase1_threshold_sensitivity,
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
    if (
        evaluation.get("schema") != EVALUATION_SCHEMA
        or evaluation.get("evaluation_execution_implementation")
        != EVALUATION_EXECUTION_IMPLEMENTATION
    ):
        raise CellExecutionError("evaluation artifact has the wrong implementation")
    input_hashes = {
        kind: prerequisites[kind][1]
        for kind in (
            "preparation",
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
    preparation = _load_preparation(prerequisites["preparation"][0], ctx)
    training_report = _read_object(
        prerequisites["training_report"][0], label="training report"
    )
    calibration_record = _read_object(
        prerequisites["calibration_record"][0], label="calibration record"
    )
    if ctx.values["qualification.thresholds_version"] != "2026-07-22.v2":
        raise CellExecutionError(
            "unsupported qualification.thresholds_version "
            + repr(ctx.values["qualification.thresholds_version"])
        )
    threshold_map = {
        "schema": "bsc-integrity-thresholds-2026-07-22.v2",
        "support_target_abs_error_max": 0.1,
        "codec_excluded_calibration_event_fraction_max": 0.01,
        "codec_excluded_evaluation_event_fraction_max": 0.01,
        "probability_metric_range": [0.0, 1.0],
        "required_quantizer_bits": list(ctx.values["codec.quantizer_bits"]),
        "phase1_identification_thresholds": list(
            ctx.values["qualification.phase1_identification_thresholds"]
        ),
        "phase1_identification_enforced": ctx.values["runtime.smoke"] is False,
        "phase1_margin_normalization_contract": ctx.values[
            "evaluation.phase1_margin_normalization"
        ],
        "phase1_rank_mismatch_contract": ctx.values[
            "evaluation.rank_mismatch_contract"
        ],
        "phase1_pathology_association_contract": ctx.values[
            "evaluation.pathology_association_contract"
        ],
        "phase1_pathology_strong_association_cutoff": ctx.values[
            "evaluation.pathology_strong_association_cutoff"
        ],
        "phase1_pathology_weak_association_cutoff": ctx.values[
            "evaluation.pathology_weak_association_cutoff"
        ],
        "phase1_pathology_association_cutoff_sensitivity": [
            list(item)
            for item in ctx.values[
                "evaluation.pathology_association_cutoff_sensitivity"
            ]
        ],
        "encoder_scale_fit_statistic": ctx.values["model.encoder_scale_fit_statistic"],
        "encoder_scale_fit_solver": ctx.values["model.encoder_scale_fit_solver"],
        "encoder_scale_fit_target": ctx.values["model.encoder_scale_fit_target"],
        "encoder_scale_fit_tolerance": ctx.values["model.encoder_scale_fit_tolerance"],
        "encoder_scale_fit_max_iterations": ctx.values[
            "model.encoder_scale_fit_max_iterations"
        ],
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
        frozen_operating_policy=fixed_rate.get("operating_policy"),
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

    encoder_scale_calibration = training_report.get("encoder_scale_calibration")
    encoder_scale_calibration_integrity = False
    if isinstance(encoder_scale_calibration, dict):
        strategy = ctx.values["model.encoder_scale_calibration"]
        if strategy == "fixed_init_no_data_fit":
            encoder_scale_calibration_integrity = encoder_scale_calibration == {
                "strategy": strategy,
                "fitted": False,
                "scale_multiplier": 1.0,
            }
        else:
            observed_after = encoder_scale_calibration.get("mean_block_norm_after")
            target = float(ctx.values["model.encoder_scale_fit_target"])
            tolerance = float(ctx.values["model.encoder_scale_fit_tolerance"])
            encoder_scale_calibration_integrity = bool(
                encoder_scale_calibration.get("strategy") == strategy
                and encoder_scale_calibration.get("fitted") is True
                and encoder_scale_calibration.get("statistic")
                == ctx.values["model.encoder_scale_fit_statistic"]
                and encoder_scale_calibration.get("solver")
                == ctx.values["model.encoder_scale_fit_solver"]
                and encoder_scale_calibration.get("target") == target
                and encoder_scale_calibration.get("tolerance") == tolerance
                and encoder_scale_calibration.get("max_iterations")
                == ctx.values["model.encoder_scale_fit_max_iterations"]
                and isinstance(encoder_scale_calibration.get("iterations"), int)
                and 0
                < encoder_scale_calibration["iterations"]
                <= ctx.values["model.encoder_scale_fit_max_iterations"]
                and encoder_scale_calibration.get("remeasured_post_fit") is True
                and finite_number(observed_after)
                and abs(float(observed_after) - target) <= tolerance
                and finite_number(
                    encoder_scale_calibration.get("mean_block_norm_before")
                )
                and finite_number(encoder_scale_calibration.get("scale_multiplier"))
                and float(encoder_scale_calibration["scale_multiplier"]) > 0.0
                and isinstance(encoder_scale_calibration.get("events"), int)
                and encoder_scale_calibration["events"] > 0
                and isinstance(encoder_scale_calibration.get("input"), dict)
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
            fixed_rate.get("schema") == "bsc-fixed-rate-raw-selection-v2"
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
                fixed_rate.get("schema") == "bsc-fixed-rate-raw-selection-v2"
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
            and recovery[endpoint].get("rank_mismatch", {}).get("contract")
            == threshold_map["phase1_rank_mismatch_contract"]
            and recovery[endpoint]
            .get("rank_mismatch", {})
            .get("same_block_metrics_are_primary")
            is True
            and recovery[endpoint]
            .get("rank_mismatch", {})
            .get("same_block_gate_is_ceiling_adjusted")
            is False
            and recovery[endpoint].get("pathology_association", {}).get("contract")
            == threshold_map["phase1_pathology_association_contract"]
            and recovery[endpoint]
            .get("pathology_association", {})
            .get("primary", {})
            .get("strong_association_cutoff")
            == threshold_map["phase1_pathology_strong_association_cutoff"]
            and recovery[endpoint]
            .get("pathology_association", {})
            .get("primary", {})
            .get("weak_association_cutoff")
            == threshold_map["phase1_pathology_weak_association_cutoff"]
            and [
                [
                    item.get("strong_association_cutoff"),
                    item.get("weak_association_cutoff"),
                ]
                for item in recovery[endpoint]
                .get("pathology_association", {})
                .get("reporting_only_sensitivity", ())
                if isinstance(item, dict)
            ]
            == threshold_map["phase1_pathology_association_cutoff_sensitivity"]
            and identification.get(endpoint, {}).get("margin_normalization_contract")
            == threshold_map["phase1_margin_normalization_contract"]
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
            and (
                (
                    identification[endpoint].get("applicable") is True
                    and isinstance(identification[endpoint].get("passed"), bool)
                    and finite_number(identification[endpoint].get("margin"))
                    and isinstance(identification[endpoint].get("checks"), dict)
                )
                or (
                    identification[endpoint].get("applicable") is False
                    and identification[endpoint].get("passed") is None
                    and identification[endpoint].get("margin") is None
                    and isinstance(
                        identification[endpoint].get("ineligible_reason"), str
                    )
                )
            )
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
    phase1_identification_passed, inapplicable_scientific_checks = (
        _phase1_identification_outcome(
            ctx.cell.phase,
            identification,
            evaluation.get("validation", {}),
        )
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
        "phase1_identification": phase1_identification_passed,
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
        "inapplicable_checks": inapplicable_scientific_checks,
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
        and preparation.get("implementation_sha256")
        == _implementation_identity_sha256(preparation["implementation"])
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
        "encoder_scale_calibration_integrity": encoder_scale_calibration_integrity,
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
        "implementation_identity": preparation["implementation"],
        "implementation_identity_sha256": preparation["implementation_sha256"],
        "validation": evaluation["validation"],
        "qualification_profile": ctx.values["qualification.profile"],
        "thresholds_version": ctx.values["qualification.thresholds_version"],
        "thresholds": threshold_map,
        "selection_metrics": selection_metrics,
        "selection_metrics_sha256": selection_metrics_sha256,
        "selection_metrics_evaluation_sha256": input_hashes["evaluation"],
        "fixed_rate_operating_policy": fixed_rate.get("operating_policy"),
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
    execution_cache: _StageExecutionCache | None = None,
) -> tuple[tuple[str, Path], ...]:
    if prerequisites is None:
        prerequisites = ctx.prerequisites()
    if ctx.stage == "prepare":
        return _prepare(ctx)
    if ctx.stage == "train":
        return _train(
            ctx,
            prerequisites,
            resume=resume,
            execution_cache=execution_cache,
        )
    if ctx.stage == "calibrate":
        return _calibrate(ctx, prerequisites, execution_cache=execution_cache)
    if ctx.stage == "evaluate":
        return _evaluate(ctx, prerequisites, execution_cache=execution_cache)
    if ctx.stage == "qualify":
        return _qualify(ctx, prerequisites)
    raise AssertionError(ctx.stage)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", type=Path, required=True)
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--artifacts-out", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--worker",
        action="store_true",
        help="serve multiple bound stage requests over stdin/stdout",
    )
    return parser


def _gpu_lock_path(device: torch.device) -> Path:
    try:
        return cuda_execution_lock_path(device)
    except (OSError, RuntimeError, ValueError) as exc:
        raise CellExecutionError(str(exc)) from exc


@contextmanager
def _host_gpu_execution_lock(cell_path: Path):
    """Serialize canonical CUDA workers across campaigns on one host device."""

    try:
        cell = CellSpec.from_manifest(json.loads(cell_path.read_text()))
        device = torch.device(str(cell.decision_map["runtime.device"]))
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise CellExecutionError(
            f"cannot resolve host GPU lock from cell manifest: {exc}"
        ) from exc
    if device.type != "cuda":
        yield
        return

    try:
        with host_cuda_execution_lock(
            device,
            operation="cell-stage",
            owner_id=cell.cell_id,
        ):
            yield
    except (OSError, RuntimeError, ValueError) as exc:
        raise CellExecutionError(str(exc)) from exc


def _execute_stage_request(
    cell: Path,
    *,
    stage: str,
    artifacts_out: Path,
    resume: bool,
    execution_cache: _StageExecutionCache | None = None,
    artifact_digests: _ArtifactDigestCache | None = None,
) -> None:
    # Journal receipts remain stage-local. A worker digest observation may
    # cross the parent handshake only while its complete stat fingerprint is
    # unchanged; the new context still requires the journal's expected hash
    # and size before trusting that observation.
    _VERIFIED_STORE_BINDINGS.clear()
    ctx = _Context(
        cell,
        artifacts_out,
        stage,
        artifact_digests=artifact_digests,
    )
    prerequisites = ctx.prerequisites()
    artifacts = execute(
        ctx,
        resume=resume,
        prerequisites=prerequisites,
        execution_cache=execution_cache,
    )
    _emit_stage_manifest(
        ctx.artifacts_out,
        cell_id=ctx.cell.cell_id,
        stage=ctx.stage,
        root=ctx.root,
        artifacts=artifacts,
        digest=ctx.artifact_sha256,
    )


def _worker_main(cell: Path) -> None:
    execution_cache = _StageExecutionCache()
    artifact_digests = _ArtifactDigestCache()
    for raw in sys.stdin:
        try:
            request = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"error: malformed worker request: {exc}") from exc
        if request == {"command": "close"}:
            return
        if not isinstance(request, dict) or set(request) != {
            "stage",
            "artifacts_out",
            "resume",
        }:
            raise SystemExit("error: malformed worker request fields")
        stage = request["stage"]
        artifacts_out = request["artifacts_out"]
        resume = request["resume"]
        if (
            stage not in STAGES
            or not isinstance(artifacts_out, str)
            or not isinstance(resume, bool)
        ):
            raise SystemExit("error: malformed worker request values")
        try:
            _execute_stage_request(
                cell,
                stage=stage,
                artifacts_out=Path(artifacts_out),
                resume=resume,
                execution_cache=execution_cache,
                artifact_digests=artifact_digests,
            )
            response = {"ok": True, "stage": stage}
        except (
            CellExecutionError,
            KeyError,
            TypeError,
            ValueError,
            RuntimeError,
        ) as exc:
            response = {
                "ok": False,
                "stage": stage,
                "error_type": type(exc).__name__,
                "error": str(exc)[-4_000:],
            }
        sys.stdout.write(json.dumps(response, sort_keys=True) + "\n")
        sys.stdout.flush()
        if response["ok"] is not True:
            return


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.worker:
        if args.stage is not None or args.artifacts_out is not None or args.resume:
            raise SystemExit("error: --worker cannot be combined with stage arguments")
        with _host_gpu_execution_lock(args.cell):
            _worker_main(args.cell)
        return
    if args.stage is None or args.artifacts_out is None:
        raise SystemExit("error: --stage and --artifacts-out are required")
    try:
        with _host_gpu_execution_lock(args.cell):
            _execute_stage_request(
                args.cell,
                stage=args.stage,
                artifacts_out=args.artifacts_out,
                resume=args.resume,
            )
    except (CellExecutionError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()
