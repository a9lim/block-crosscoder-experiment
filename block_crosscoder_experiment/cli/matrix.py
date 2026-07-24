"""Plan and operate the declarative three-phase BSC campaign.

This command intentionally does not promote cells.  Promotion consumes an
explicit, hash-bound decision artifact through the campaign API after review.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
import shutil
import signal
import sys
import tempfile
from pathlib import Path
from typing import Sequence

from block_crosscoder_experiment.cli.data import (
    DEFAULT_FREE_SPACE_FLOOR_FRAC,
    expected_capture_allocation,
    expected_capture_source_contract,
    validate_capture_manifest,
    validate_derived_view_manifest,
    validate_transform_artifact_manifest,
)
from block_crosscoder_experiment.campaign import (
    Campaign,
    CampaignError,
    CampaignRunner,
    RunSummary,
)
from block_crosscoder_experiment.durability import durable_mkdir, durable_replace
from block_crosscoder_experiment.store import NORMALIZATION_MODES, StoreReader, Whitener
from block_crosscoder_experiment.studies import (
    Budget,
    BudgetExceeded,
    FrozenSelection,
    Phase,
    Phase1Blueprint,
    Phase2Blueprint,
    StudyError,
    build_phase1_blueprint,
    build_phase1_plan,
    build_phase2_blueprint,
    build_phase2_plan,
    build_phase3_blueprint,
    build_phase3_plan,
    estimate_activation_store,
    estimate_plan,
    enforce_plan_resources,
    materialize_child_plan,
    materialize_family_child_plan,
    materialize_family_revisit_plan,
)

_VERIFICATION_PROBE_BYTES = 64 * 1024


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


@contextmanager
def _sigterm_unwinds_runner():
    """Turn SIGTERM into stack unwinding so worker groups close in ``finally``."""

    previous = signal.getsignal(signal.SIGTERM)

    def terminate(signum, frame):
        del frame
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def _string_mapping_json(value: str) -> dict[str, str]:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(
            f"must be a JSON object mapping stress IDs to claim narrowings: {exc}"
        ) from exc
    if not isinstance(payload, dict) or any(
        not isinstance(key, str)
        or not key
        or not isinstance(item, str)
        or not item.strip()
        for key, item in payload.items()
    ):
        raise argparse.ArgumentTypeError(
            "must map nonempty string stress IDs to nonempty string claim narrowings"
        )
    return {key: item.strip() for key, item in payload.items()}


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path.resolve()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_verification_receipt(path: Path, payload: dict[str, object]) -> None:
    durable_mkdir(path.parent, parents=True, exist_ok=True)
    body = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
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
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _verify_store_with_receipt(
    reader: StoreReader,
    *,
    cache_root: Path | None,
) -> None:
    if cache_root is None:
        reader.verify()
        return
    split_dir = reader.dir
    manifest_path = split_dir / "split.json"

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

    def content_probe(path: Path) -> dict[str, int | str]:
        size = path.stat().st_size
        length = min(size, _VERIFICATION_PROBE_BYTES)
        offset_span = size - length
        offset_seed = hashlib.sha256(
            f"{manifest_file_sha256}:{path.name}".encode("utf-8")
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
            raise StudyError(f"short verification probe read from {path}")
        return {
            "path": str(path.resolve()),
            "offset": offset,
            "length": length,
            "sha256": hashlib.sha256(body).hexdigest(),
        }

    manifest_file_sha256 = _sha256(manifest_path)
    shard_paths = [
        split_dir / str(record["file"]) for record in reader.manifest["shards"]
    ]
    fingerprint = {
        "manifest": stat_record(manifest_path),
        "shards": [stat_record(path) for path in shard_paths],
    }
    probes = [content_probe(path) for path in shard_paths]
    key = _canonical_hash(
        {
            "root": str(split_dir.parent.resolve()),
            "split": split_dir.name,
            "manifest_sha256": manifest_file_sha256,
        }
    )
    receipt_path = cache_root / f"{key}.json"
    expected = {
        "schema": "bsc-store-verification-receipt-v2",
        "root": str(split_dir.parent.resolve()),
        "split": split_dir.name,
        "manifest_sha256": manifest_file_sha256,
        "manifest_content_sha256": reader.manifest.get("manifest_sha256"),
        "content_stream_sha256": reader.manifest.get("content_stream_sha256"),
        "row_stream_sha256": reader.manifest.get("row_stream_sha256"),
        "n_tokens": reader.n_tokens,
        "stat_fingerprint": fingerprint,
        "content_probes": probes,
    }
    if receipt_path.is_file():
        try:
            existing = json.loads(receipt_path.read_text())
        except (OSError, json.JSONDecodeError):
            existing = None
        if existing == expected:
            return
    reader.verify()
    _write_verification_receipt(receipt_path, expected)


def _configured_input_roots(
    explicit_roots: Sequence[Path] = (),
) -> tuple[Path, ...]:
    names = (
        "BSC_VIEW_ROOT",
        "BSC_ACTIVATION_STORE",
        "BSC_STORE_ROOT",
        "BSC_RAW_STORE_ROOT",
        "BSC_RAW_STORE",
        "BSC_TRANSFORM_ROOT",
    )
    roots: list[Path] = []
    seen: set[Path] = set()
    values = [str(root) for root in explicit_roots]
    values.extend(os.environ.get(name, "") for name in names)
    for value in values:
        if not value:
            continue
        root = Path(value).expanduser().resolve()
        if root not in seen:
            seen.add(root)
            roots.append(root)
    return tuple(roots)


def _verified_existing_input_storage(
    *,
    verification_cache_root: Path | None = None,
    plan=None,
    input_roots: Sequence[Path] = (),
) -> dict[str, object]:
    """Hash-verify configured immutable inputs and count their physical bytes.

    Only files reached through a verified split or transform manifest count.
    Merely pointing an environment variable at a directory never buys storage
    credit, and overlapping environment roots are deduplicated by resolved path.
    """

    if plan is not None and plan.phase is Phase.PHASE1:
        return {
            "verified_existing_input_bytes": 0,
            "inputs": [],
            "plan_input_contract": "stateless_phase1_no_input_credit",
        }

    expected_source: dict[str, object] | None = None
    expected_source_hash: str | None = None
    expected_split_plan: dict[str, dict[str, int]] | None = None
    expected_split_order: tuple[str, ...] | None = None
    expected_normalizations: set[str] | None = None
    if plan is not None:
        cells = [cell for stage in plan.stages for cell in stage.cells]
        if not cells:
            raise StudyError("cannot price inputs for an empty materialized plan")
        contracts = []
        allocations = []
        for cell in cells:
            values = cell.decision_map
            try:
                contracts.append(expected_capture_source_contract(values))
                allocations.append(expected_capture_allocation(values))
            except (KeyError, TypeError, ValueError) as exc:
                raise StudyError(
                    f"cannot resolve plan-bound activation input for {cell.cell_id}: {exc}"
                ) from exc
        expected_source = contracts[0]
        expected_split_order, expected_split_plan = allocations[0]
        if any(contract != expected_source for contract in contracts[1:]) or any(
            allocation != allocations[0] for allocation in allocations[1:]
        ):
            raise StudyError(
                "materialized plan does not share one immutable capture contract"
            )
        expected_source_hash = _canonical_hash(expected_source)
        expected_normalizations = {
            str(cell.decision_map["data.normalization"]) for cell in cells
        }

    counted_files: set[Path] = set()
    records: list[dict[str, object]] = []
    for root in _configured_input_roots(input_roots):
        if not root.is_dir():
            raise StudyError(f"configured activation input is not a directory: {root}")
        split_manifests = sorted(root.rglob("split.json"))
        view_manifests = sorted(root.rglob("view.json"))
        transform_manifests = sorted(root.rglob("transform.json"))
        root_files: set[Path] = set()
        verified_splits: list[str] = []
        verified_transforms: list[str] = []
        eligible_capture_files: set[Path] = set()
        eligible_split_envelopes: dict[Path, tuple[str, dict[str, object]]] = {}
        for capture_path in root.rglob("capture.json"):
            try:
                capture = json.loads(capture_path.read_text())
                if not isinstance(capture, dict):
                    raise ValueError("manifest must be an object")
                validate_capture_manifest(capture)
            except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
                raise StudyError(
                    f"invalid capture manifest {capture_path}: {exc}"
                ) from exc
            if expected_source is None or (
                capture.get("source") == expected_source
                and capture.get("split_order") == list(expected_split_order or ())
                and capture.get("split_plan") == expected_split_plan
            ):
                eligible_capture_files.add(capture_path.resolve())
                for split in capture["split_order"]:
                    eligible_split_envelopes[
                        (capture_path.parent / split).resolve()
                    ] = ("raw", dict(capture["splits"][split]))
        for view_path in view_manifests:
            try:
                view = json.loads(view_path.read_text())
                if not isinstance(view, dict):
                    raise ValueError("manifest must be an object")
                view = validate_derived_view_manifest(view)
            except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
                raise StudyError(
                    f"invalid derived-view manifest {view_path}: {exc}"
                ) from exc
            eligible_view = expected_source is None or (
                view["source_capture"]["source"] == expected_source
                and view["source_capture"]["split_order"]
                == list(expected_split_order or ())
                and view["source_capture"]["split_plan"] == expected_split_plan
                and view["mode"] in (expected_normalizations or set())
            )
            if not eligible_view:
                continue
            whitener_path = view_path.parent / "whitener.pt"
            if (
                not whitener_path.is_file()
                or _sha256(whitener_path) != view["whitener_sha256"]
            ):
                raise StudyError(f"derived-view whitener is unverified at {view_path}")
            root_files.update((view_path.resolve(), whitener_path.resolve()))
            for split in view["split_order"]:
                split_dir = (view_path.parent / split).resolve()
                if split_dir in eligible_split_envelopes:
                    raise StudyError(
                        f"multiple root envelopes claim activation split {split_dir}"
                    )
                eligible_split_envelopes[split_dir] = (
                    "derived",
                    dict(view["splits"][split]),
                )
        for manifest_path in split_manifests:
            split_dir = manifest_path.parent
            envelope = eligible_split_envelopes.get(split_dir.resolve())
            if envelope is None:
                continue
            try:
                reader = StoreReader(split_dir.parent, split_dir.name)
                _verify_store_with_receipt(
                    reader,
                    cache_root=verification_cache_root,
                )
            except (OSError, KeyError, TypeError, ValueError) as exc:
                raise StudyError(
                    f"unverified activation split at {split_dir}: {exc}"
                ) from exc
            envelope_kind, envelope_record = envelope
            if envelope_kind == "raw":
                envelope_matches = (
                    _sha256(manifest_path) == envelope_record["manifest_file_sha256"]
                    and reader.manifest.get("manifest_sha256")
                    == envelope_record["manifest_sha256"]
                    and reader.manifest.get("content_stream_sha256")
                    == envelope_record["content_stream_sha256"]
                    and reader.manifest.get("row_stream_sha256")
                    == envelope_record["row_stream_sha256"]
                    and reader.n_tokens == envelope_record["n_tokens"]
                )
            else:
                envelope_matches = (
                    reader.manifest.get("manifest_sha256")
                    == envelope_record["manifest_sha256"]
                    and reader.manifest.get("content_stream_sha256")
                    == envelope_record["content_stream_sha256"]
                    and reader.manifest.get("row_stream_sha256")
                    == envelope_record["row_stream_sha256"]
                    and reader.n_tokens == envelope_record["n_tokens"]
                )
            if not envelope_matches:
                raise StudyError(
                    f"activation split differs from its root envelope: {split_dir}"
                )
            meta = reader.manifest.get("meta", {})
            eligible = expected_source is None
            if expected_source is not None and expected_split_plan is not None:
                allocation = expected_split_plan.get(split_dir.name)
                source_matches = meta.get("source_hash") == expected_source_hash
                if not source_matches and str(reader.whitener_hash) == (
                    f"raw:{expected_source_hash}"
                ):
                    source_matches = True
                eligible = bool(
                    allocation is not None
                    and source_matches
                    and meta.get("split_requested_tokens")
                    == allocation["requested_tokens"]
                    and meta.get("split_actual_tokens") == allocation["actual_tokens"]
                    and reader.n_tokens == allocation["actual_tokens"]
                )
                normalization = meta.get("normalization")
                if meta.get("derived_view") is True:
                    eligible = eligible and normalization in expected_normalizations
            if eligible:
                root_files.add(manifest_path.resolve())
                for shard in reader.manifest["shards"]:
                    root_files.add((split_dir / shard["file"]).resolve())
                verified_splits.append(str(split_dir.relative_to(root)))
        for manifest_path in transform_manifests:
            try:
                manifest = json.loads(manifest_path.read_text())
                if not isinstance(manifest, dict):
                    raise ValueError("manifest must be an object")
                validate_transform_artifact_manifest(manifest)
            except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
                raise StudyError(
                    f"invalid transform manifest {manifest_path}: {exc}"
                ) from exc
            transform_path = manifest_path.parent / "whitener.pt"
            if not transform_path.is_file() or manifest.get(
                "whitener_sha256"
            ) != _sha256(transform_path):
                raise StudyError(
                    f"unverified transform artifact at {manifest_path.parent}"
                )
            try:
                transform = Whitener.load(transform_path)
            except (OSError, KeyError, TypeError, ValueError) as exc:
                raise StudyError(
                    f"unverified transform artifact at {manifest_path.parent}: {exc}"
                ) from exc
            if transform.hash != manifest.get("transform_hash"):
                raise StudyError(f"transform hash mismatch at {manifest_path.parent}")
            eligible = expected_source is None or (
                manifest.get("source_hash") == expected_source_hash
                and manifest.get("mode") in expected_normalizations
                and manifest.get("source_fit_requested_tokens")
                == (expected_split_plan or {})
                .get("normalization_fit", {})
                .get("requested_tokens")
            )
            if eligible:
                root_files.update((manifest_path.resolve(), transform_path.resolve()))
                verified_transforms.append(str(manifest_path.parent.relative_to(root)))
        root_files.update(eligible_capture_files)
        if expected_source is None and not verified_splits and not verified_transforms:
            raise StudyError(
                f"configured input contains no verified split or transform artifact: {root}"
            )
        new_files = root_files - counted_files
        byte_count = sum(path.stat().st_size for path in new_files)
        counted_files.update(root_files)
        records.append(
            {
                "root": str(root),
                "verified_bytes": byte_count,
                "splits": verified_splits,
                "transforms": verified_transforms,
                "eligible_for_plan": bool(root_files),
            }
        )
    return {
        "verified_existing_input_bytes": sum(
            path.stat().st_size for path in counted_files
        ),
        "inputs": records,
    }


def _estimated_plan_input_storage_bytes(plan) -> int:
    """Return the exact activation-store portion of ``estimate_plan(plan)``."""

    if plan is None:
        return 0
    stores: dict[tuple[object, ...], int] = {}
    raw_stores: dict[tuple[object, ...], int] = {}
    for store_bytes, key in (estimate_activation_store(cell) for cell in plan.cells):
        stores[key] = max(stores.get(key, 0), store_bytes)
        if key[12] == "content_addressed_derived_view":
            raw_key = (*key[:13], "raw_source_view", *key[14:])
            raw_stores[raw_key] = max(raw_stores.get(raw_key, 0), store_bytes)
    return sum(stores.values()) + sum(raw_stores.values())


def _storage_preflight(
    root: Path,
    estimated_storage_bytes: int,
    *,
    plan=None,
    input_roots: Sequence[Path] = (),
) -> dict[str, object]:
    existing = _verified_existing_input_storage(
        verification_cache_root=root / ".store-verification",
        plan=plan,
        input_roots=input_roots,
    )
    campaign_artifact_files: set[Path] = set()
    if (root / "plan.json").is_file():
        existing_campaign = Campaign(root)
        for record in existing_campaign.records():
            for artifact in record.artifact_map.values():
                artifact.verify(root)
                campaign_artifact_files.add(artifact.resolve(root).resolve())
    campaign_artifact_bytes = sum(
        path.stat().st_size for path in campaign_artifact_files
    )
    estimated_input_bytes = min(
        estimated_storage_bytes,
        (
            _estimated_plan_input_storage_bytes(plan)
            if plan is not None
            else int(existing["verified_existing_input_bytes"])
        ),
    )
    estimated_campaign_bytes = estimated_storage_bytes - estimated_input_bytes
    input_credit = min(
        estimated_input_bytes,
        int(existing["verified_existing_input_bytes"]),
    )
    campaign_artifact_credit = min(
        estimated_campaign_bytes,
        campaign_artifact_bytes,
    )
    credited = input_credit + campaign_artifact_credit
    missing_input = estimated_input_bytes - input_credit
    missing_campaign = estimated_campaign_bytes - campaign_artifact_credit

    requirements: dict[int, dict[str, object]] = {}

    def add_requirement(path: Path, required: int, role: str) -> None:
        parent = _nearest_existing_parent(path)
        device = int(parent.stat().st_dev)
        usage = shutil.disk_usage(parent)
        record = requirements.setdefault(
            device,
            {
                "device": device,
                "filesystem_path": str(parent),
                "raw_free_bytes": int(usage.free),
                "free_space_floor_bytes": int(
                    usage.total * DEFAULT_FREE_SPACE_FLOOR_FRAC
                ),
                "required_bytes": 0,
                "roles": [],
            },
        )
        record["required_bytes"] = int(record["required_bytes"]) + required
        roles = record["roles"]
        assert isinstance(roles, list)
        roles.append(role)

    add_requirement(root, missing_campaign, "campaign_artifacts")
    configured_roots = _configured_input_roots(input_roots)
    if missing_input:
        if configured_roots:
            # Output placement is not encoded in a scientific plan. Requiring
            # the complete unmaterialized input remainder on every declared
            # destination filesystem is conservative and prevents aggregate
            # free space on another device from authorizing this one.
            seen_devices: set[int] = set()
            for configured_root in configured_roots:
                parent = _nearest_existing_parent(configured_root)
                device = int(parent.stat().st_dev)
                if device in seen_devices:
                    continue
                seen_devices.add(device)
                add_requirement(
                    configured_root,
                    missing_input,
                    "unmaterialized_activation_inputs",
                )
        else:
            add_requirement(root, missing_input, "unmaterialized_activation_inputs")
    filesystem_preflights = sorted(
        requirements.values(), key=lambda item: item["device"]
    )
    for record in filesystem_preflights:
        record["available_above_floor_bytes"] = max(
            0,
            int(record["raw_free_bytes"]) - int(record["free_space_floor_bytes"]),
        )
        record["sufficient"] = int(record["required_bytes"]) <= int(
            record["available_above_floor_bytes"]
        )
    additional = missing_input + missing_campaign
    sufficient = all(bool(record["sufficient"]) for record in filesystem_preflights)
    campaign_parent = _nearest_existing_parent(root)
    campaign_usage = shutil.disk_usage(campaign_parent)
    campaign_free = int(campaign_usage.free)
    campaign_floor = int(campaign_usage.total * DEFAULT_FREE_SPACE_FLOOR_FRAC)
    return {
        "estimate_scope": "materialized_plan_prefix_or_frozen_panel",
        "estimated_storage_bytes": estimated_storage_bytes,
        "estimated_input_storage_bytes": estimated_input_bytes,
        "estimated_campaign_artifact_bytes": estimated_campaign_bytes,
        **existing,
        "verified_existing_campaign_artifact_bytes": campaign_artifact_bytes,
        "credited_existing_input_bytes": input_credit,
        "credited_existing_campaign_artifact_bytes": campaign_artifact_credit,
        "credited_existing_storage_bytes": credited,
        "additional_storage_bytes_required": additional,
        "additional_input_storage_bytes_required": missing_input,
        "additional_campaign_storage_bytes_required": missing_campaign,
        "free_bytes": campaign_free,
        "free_space_floor_bytes": campaign_floor,
        "available_above_floor_bytes": max(0, campaign_free - campaign_floor),
        "filesystem_path": str(campaign_parent),
        "filesystem_preflights": filesystem_preflights,
        "sufficient": sufficient,
    }


def _estimate_label(plan, blueprint) -> dict[str, object]:
    materialized_cells = sum(len(stage.cells) for stage in plan.stages)
    if isinstance(blueprint, Phase2Blueprint):
        cell_count_ceiling = blueprint.declared_cell_ceiling
        required_stages = {
            blueprint.initial_stage.name,
            *(round_spec.name for round_spec in blueprint.rounds),
            *(
                round_spec.name
                for family in blueprint.comparator_families
                for round_spec in family.rounds
            ),
            *(family.revisit.name for family in blueprint.comparator_families),
        }
        complete = required_stages.issubset({stage.name for stage in plan.stages})
        count_contract = "declared_pre_elision_ceiling"
    else:
        cell_count_ceiling = blueprint.projected_cells
        complete = materialized_cells == cell_count_ceiling
        count_contract = "exact_frozen_projection"
    frozen = plan.phase is Phase.PHASE3
    return {
        "scope": (
            "complete_frozen_plan"
            if frozen
            else (
                "complete_materialized_campaign"
                if complete
                else "materialized_conditional_prefix"
            )
        ),
        "materialized_cells": materialized_cells,
        "cell_count_ceiling": cell_count_ceiling,
        "cell_count_contract": count_contract,
        "materialized_campaign_complete": complete,
        "materialized_total_priced": complete,
    }


def _cell_count_ceiling(blueprint) -> int:
    return (
        blueprint.declared_cell_ceiling
        if isinstance(blueprint, Phase2Blueprint)
        else blueprint.projected_cells
    )


def _resolve_phase2_view_dispatch(
    view_root: Path,
    cells: dict[str, object],
) -> dict[str, Path]:
    """Resolve and manifest-check each cell's exact normalization view.

    Payload checks remain the cell executor's responsibility.  This preflight
    is intentionally cheap enough to run before every campaign invocation: it
    validates complete self-hashed manifests, exact file sets, the frozen
    Whitener, and cross-mode row-stream identity without rereading all payload
    bytes before any campaign state transition occurs.
    """

    root = view_root.expanduser().resolve()
    if not root.is_dir():
        raise StudyError(f"Phase-2 --view-root is not a directory: {root}")
    by_mode: dict[
        str,
        tuple[Path, dict[str, tuple[object, ...]], dict[str, int], str],
    ] = {}
    dispatched: dict[str, Path] = {}
    for cell_id, cell in cells.items():
        values = cell.decision_map
        mode = str(values["data.normalization"])
        if mode not in NORMALIZATION_MODES:
            raise StudyError(f"cell {cell_id} has unsupported normalization {mode!r}")
        declared_items = tuple(values["data.split_sizes"])
        declared = {str(name): int(tokens) for name, tokens in declared_items}
        if mode not in by_mode:
            mode_root = root / mode
            if not mode_root.is_dir():
                raise StudyError(f"Phase-2 view {mode!r} does not exist under {root}")
            view_manifest_path = mode_root / "view.json"
            try:
                view_manifest = json.loads(view_manifest_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise StudyError(
                    f"Phase-2 view {mode!r} lacks a valid root manifest: {exc}"
                ) from exc
            if not isinstance(view_manifest, dict):
                raise StudyError(
                    f"Phase-2 view {mode!r} root manifest is not an object"
                )
            try:
                view_manifest = validate_derived_view_manifest(view_manifest)
            except (TypeError, ValueError) as exc:
                raise StudyError(
                    f"Phase-2 view {mode!r} has an invalid root manifest: {exc}"
                ) from exc
            expected_order, expected_plan = expected_capture_allocation(values)
            expected_source = expected_capture_source_contract(values)
            source_capture = view_manifest["source_capture"]
            if (
                view_manifest.get("mode") != mode
                or view_manifest.get("split_order") != list(declared)
                or (
                    source_capture.get("source") != expected_source
                    or source_capture.get("split_order") != list(expected_order)
                    or source_capture.get("split_plan") != expected_plan
                )
            ):
                raise StudyError(f"Phase-2 view {mode!r} has a divergent root manifest")
            expected_root_entries = set(declared) | {"whitener.pt", "view.json"}
            actual_root_entries = {path.name for path in mode_root.iterdir()}
            if actual_root_entries != expected_root_entries:
                raise StudyError(
                    f"Phase-2 view {mode!r} root entries differ from its manifest"
                )
            transform_path = mode_root / "whitener.pt"
            if not transform_path.is_file():
                raise StudyError(f"Phase-2 view {mode!r} lacks whitener.pt")
            try:
                transform = Whitener.load(transform_path)
            except (OSError, KeyError, TypeError, ValueError) as exc:
                raise StudyError(
                    f"Phase-2 view {mode!r} has an invalid whitener: {exc}"
                ) from exc
            if transform.mode != mode:
                raise StudyError(
                    f"Phase-2 view {mode!r} contains transform mode {transform.mode!r}"
                )
            if transform.hash != view_manifest.get("transform_hash") or _sha256(
                transform_path
            ) != view_manifest.get("whitener_sha256"):
                raise StudyError(
                    f"Phase-2 view {mode!r} transform differs from its root manifest"
                )
            available = {
                path.name
                for path in mode_root.iterdir()
                if path.is_dir() and (path / "split.json").is_file()
            }
            if available != set(declared):
                raise StudyError(
                    f"Phase-2 view {mode!r} split set differs from the cell: "
                    f"expected={sorted(declared)}, actual={sorted(available)}"
                )
            signatures: dict[str, tuple[object, ...]] = {}
            for split, requested in declared.items():
                try:
                    reader = StoreReader(
                        mode_root,
                        split,
                        expected_whitener_hash=transform.hash,
                    )
                except (OSError, KeyError, TypeError, ValueError) as exc:
                    raise StudyError(
                        f"Phase-2 view {mode!r}/{split} has an invalid manifest: {exc}"
                    ) from exc
                meta = reader.manifest.get("meta", {})
                if (
                    meta.get("derived_view") is not True
                    or meta.get("normalization") != mode
                    or meta.get("split_requested_tokens") != requested
                    or reader.n_tokens < requested
                ):
                    raise StudyError(
                        f"Phase-2 view {mode!r}/{split} does not bind its cell contract"
                    )
                root_record = view_manifest["splits"].get(split)
                if not isinstance(root_record, dict) or any(
                    root_record.get(key) != expected
                    for key, expected in {
                        "manifest_sha256": reader.manifest["manifest_sha256"],
                        "content_stream_sha256": reader.manifest[
                            "content_stream_sha256"
                        ],
                        "row_stream_sha256": reader.manifest["row_stream_sha256"],
                        "n_tokens": reader.n_tokens,
                    }.items()
                ):
                    raise StudyError(
                        f"Phase-2 view {mode!r}/{split} differs from its root manifest"
                    )
                signatures[split] = (
                    reader.n_tokens,
                    reader.manifest["row_stream_sha256"],
                    tuple(reader.manifest["sites"]),
                    reader.d_model,
                )
            by_mode[mode] = (
                mode_root,
                signatures,
                declared,
                str(source_capture["capture_content_sha256"]),
            )
        elif by_mode[mode][2] != declared:
            raise StudyError(
                f"cells using Phase-2 view {mode!r} declare different split contracts"
            )
        dispatched[cell_id] = by_mode[mode][0]

    mode_items = list(by_mode.items())
    if mode_items:
        reference_mode, (_, reference, _, reference_capture) = mode_items[0]
        for mode, (_, signatures, _, capture_identity) in mode_items[1:]:
            if signatures != reference or capture_identity != reference_capture:
                raise StudyError(
                    f"Phase-2 views {reference_mode!r} and {mode!r} do not share "
                    "one exact content-addressed raw capture"
                )
    return dispatched


def _run_with_optional_view_dispatch(
    campaign: Campaign,
    args: argparse.Namespace,
) -> RunSummary:
    common = {
        "resume": args.resume,
        "stop_after": args.stop_after,
    }
    if args.view_root is None:
        return CampaignRunner(
            campaign,
            python=args.python,
            module=args.module,
        ).run(
            limit=args.limit,
            cell_ids=args.cells,
            **common,
        )
    if campaign.plan.phase is not Phase.PHASE2:
        raise StudyError("--view-root is only valid for a Phase-2 campaign")
    if args.cells is None:
        selected = list(
            campaign.runnable_cell_ids(
                include_failed=args.resume,
                include_resume_required=args.resume,
            )
        )
    else:
        selected = list(args.cells)
        for cell_id in selected:
            cell = campaign._require_cell(cell_id)
            if not campaign.stage_open(cell.stage):
                raise CampaignError(
                    f"cell {cell_id} belongs to unopened stage {cell.stage!r}"
                )
    if args.limit is not None:
        selected = selected[: args.limit]
    cells = {cell_id: campaign._require_cell(cell_id) for cell_id in selected}
    dispatch = _resolve_phase2_view_dispatch(args.view_root, cells)
    totals = {
        "selected_cells": 0,
        "completed_cells": 0,
        "failed_cells": 0,
        "skipped_cells": 0,
    }
    for cell_id in selected:
        summary = CampaignRunner(
            campaign,
            python=args.python,
            module=args.module,
            env={"BSC_ACTIVATION_STORE": str(dispatch[cell_id])},
        ).run(cell_ids=[cell_id], **common)
        for key, value in summary.to_dict().items():
            totals[key] += value
    return RunSummary(**totals)


def _phase(value: str) -> Phase:
    try:
        return Phase.parse(value)
    except StudyError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _add_phase(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--phase",
        required=True,
        type=_phase,
        metavar="PHASE",
        help="phase1/synthetic, phase2/pilot, or phase3/publishable",
    )
    parser.add_argument(
        "--seeds",
        type=_nonnegative_int,
        nargs="+",
        default=None,
        help=(
            "override replicate seeds only for smoke profiles; scientific "
            "Phase 1 is fixed to 0 1 2, Phase 2 to 0 1, and Phase 3 to "
            "0 1 2 3 4"
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="materialize the schema-complete tiny execution profile",
    )
    parser.add_argument(
        "--phase1-decision",
        type=Path,
        help=(
            "required authenticated Phase-1 go/no-go decision when registering Phase 2"
        ),
    )
    parser.add_argument(
        "--panel-decision",
        type=Path,
        help="required hash-bound Phase-2 panel decision for Phase 3",
    )


def _add_budget(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-training-tokens", type=_nonnegative_int)
    parser.add_argument("--max-parameters", type=_nonnegative_int)
    parser.add_argument("--max-storage-bytes", type=_nonnegative_int)
    parser.add_argument("--max-compute-flops", type=_nonnegative_int)
    parser.add_argument("--max-peak-vram-bytes", type=_nonnegative_int)
    parser.add_argument("--max-peak-host-ram-bytes", type=_nonnegative_int)


def _budget(args: argparse.Namespace) -> Budget | None:
    values = {
        "max_training_tokens": args.max_training_tokens,
        "max_parameters": args.max_parameters,
        "max_storage_bytes": args.max_storage_bytes,
        "max_compute_flops": args.max_compute_flops,
        "max_peak_vram_bytes": args.max_peak_vram_bytes,
        "max_peak_host_ram_bytes": args.max_peak_host_ram_bytes,
    }
    return None if all(value is None for value in values.values()) else Budget(**values)


def _print(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _read_object(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise StudyError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise StudyError(f"expected a JSON object at {path}")
    return payload


def _frozen_selection(
    path: Path,
    *,
    selection_id: str | None = None,
) -> FrozenSelection:
    payload = _read_object(path)
    candidates = payload.get("selected")
    if not isinstance(candidates, list) or not candidates:
        raise StudyError("selection artifact has no frozen candidates")
    try:
        selections = [FrozenSelection.from_dict(item) for item in candidates]
    except (KeyError, TypeError, ValueError, StudyError) as exc:
        raise StudyError(f"invalid frozen selection artifact: {exc}") from exc
    if selection_id is None:
        if len(selections) != 1:
            raise StudyError(
                "selection retained multiple candidates; pass --selection-id"
            )
        return selections[0]
    matches = [item for item in selections if item.selection_id == selection_id]
    if len(matches) != 1:
        raise StudyError("--selection-id is absent or ambiguous")
    return matches[0]


def _registered_blueprint(campaign: Campaign):
    payload = _read_object(campaign.blueprint_path)
    if campaign.plan.phase is Phase.PHASE1:
        return Phase1Blueprint.from_manifest(payload)
    if campaign.plan.phase is Phase.PHASE2:
        return Phase2Blueprint.from_manifest(payload)
    raise StudyError("Phase 3 is confirmatory and has no tuning advance")


def _checked_storage_extension(
    root: Path,
    estimate,
    *,
    allow_insufficient: bool,
    plan=None,
    input_roots: Sequence[Path] = (),
) -> dict[str, object]:
    preflight = _storage_preflight(
        root,
        estimate.storage_bytes,
        plan=plan,
        input_roots=input_roots,
    )
    if not allow_insufficient and not preflight["sufficient"]:
        failed_filesystems = [
            item
            for item in preflight["filesystem_preflights"]
            if not item["sufficient"]
        ]
        raise BudgetExceeded(
            f"storage_bytes: conservative cumulative estimate {estimate.storage_bytes}; "
            "incremental requirement "
            f"{preflight['additional_storage_bytes_required']} is not available on "
            "every bound destination filesystem: "
            f"{json.dumps(failed_filesystems, sort_keys=True)}; after crediting "
            f"{preflight['credited_existing_storage_bytes']} bytes of hash-verified "
            "configured inputs and existing campaign artifacts; choose a larger "
            "filesystem or pass "
            "--allow-insufficient-local-storage for planning only"
        )
    return preflight


def _build_plan_and_blueprint(args: argparse.Namespace):
    seeds = args.seeds
    phase1_manifest: dict[str, object] | None = None
    panel_manifest: dict[str, object] | None = None
    if args.phase is not Phase.PHASE2 and args.phase1_decision is not None:
        raise StudyError("--phase1-decision is valid only for Phase 2")
    if args.phase is not Phase.PHASE3 and args.panel_decision is not None:
        raise StudyError("--panel-decision is valid only for Phase 3")
    if args.phase is Phase.PHASE1:
        blueprint = build_phase1_blueprint(
            (0, 1, 2) if seeds is None else seeds, smoke=args.smoke
        )
        plan = build_phase1_plan(blueprint.seeds, smoke=args.smoke)
    elif args.phase is Phase.PHASE2:
        if args.command == "plan" and args.phase1_decision is None:
            raise StudyError(
                "Phase 2 registration requires --phase1-decision from a "
                "completed Phase-1 campaign"
            )
        if args.phase1_decision is not None:
            phase1_manifest = _read_object(args.phase1_decision)
            phase1_manifest = Campaign.phase1_decision_from_manifest(phase1_manifest)
        blueprint = build_phase2_blueprint(
            (0, 1) if seeds is None else seeds,
            smoke=args.smoke,
            phase1_decision=phase1_manifest,
        )
        plan = build_phase2_plan(
            blueprint.seeds,
            smoke=args.smoke,
            phase1_decision=phase1_manifest,
        )
    else:
        if args.panel_decision is None:
            raise StudyError(
                "Phase 3 requires --panel-decision from frozen Phase-2 evidence"
            )
        panel_manifest = _read_object(args.panel_decision)
        panel = Campaign.panel_decision_from_manifest(panel_manifest)
        source_manifest = panel_manifest["phase2_campaign_manifest"]
        if not isinstance(source_manifest, dict) or (
            source_manifest.get("smoke") is True and not args.smoke
        ):
            raise StudyError(
                "a smoke Phase-2 panel cannot open a production Phase-3 profile"
            )
        blueprint = build_phase3_blueprint(
            tuple(range(5)) if seeds is None else seeds,
            smoke=args.smoke,
            panel_decision=panel,
        )
        plan = build_phase3_plan(
            blueprint.seeds,
            smoke=args.smoke,
            panel_decision=panel,
        )
    budget = _budget(args)
    if budget is not None:
        budget.enforce(estimate_plan(plan))
    return plan, blueprint, phase1_manifest, panel_manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser(
        "plan", help="materialize and register an immutable plan"
    )
    plan.add_argument("--root", type=Path, required=True)
    plan.add_argument(
        "--allow-insufficient-local-storage",
        action="store_true",
        help="register even when incremental materialization exceeds local free space",
    )
    _add_phase(plan)
    _add_budget(plan)

    estimate = subparsers.add_parser(
        "estimate", help="estimate a plan without writing it"
    )
    _add_phase(estimate)
    _add_budget(estimate)

    run = subparsers.add_parser("run", help="run eligible cells through qualification")
    run.add_argument("--root", type=Path, required=True)
    selection = run.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=_positive_int)
    run.add_argument("--resume", action="store_true")
    selection.add_argument("--cell", action="append", dest="cells")
    run.add_argument(
        "--view-root",
        type=Path,
        help=(
            "Phase-2 parent whose <normalization>/ children are aligned derived "
            "views; dispatch each cell to its exact view"
        ),
    )
    run.add_argument(
        "--stop-after",
        choices=("prepare", "train", "calibrate", "evaluate", "qualify"),
    )
    run.add_argument("--python", default=sys.executable)
    run.add_argument(
        "--module",
        default="block_crosscoder_experiment.cli.run_cell",
        help="generic cell implementation module",
    )

    status = subparsers.add_parser(
        "status", help="show journal-derived campaign status"
    )
    status.add_argument("--root", type=Path, required=True)

    reconcile = subparsers.add_parser(
        "reconcile",
        help="remove stale locks and rebuild atomic snapshots from the journal",
    )
    reconcile.add_argument("--root", type=Path, required=True)
    reconcile.add_argument(
        "--stale-after",
        type=_positive_float,
        default=3600.0,
        help="lock age in seconds after which explicit reconciliation removes it",
    )

    select = subparsers.add_parser(
        "select",
        help="apply a stage's frozen policy and bind the complete ranked universe",
    )
    select.add_argument("--root", type=Path, required=True)
    select.add_argument("--stage", required=True)
    select.add_argument("--out", type=Path)

    select_family_root = subparsers.add_parser(
        "select-family-root",
        help="select one comparator's anchor with its root-only family policy",
    )
    select_family_root.add_argument("--root", type=Path, required=True)
    select_family_root.add_argument("--family", required=True)
    select_family_root.add_argument("--out", type=Path)

    amend_phase2_gates = subparsers.add_parser(
        "amend-phase2-gates",
        help=(
            "adopt the corrected common Phase-2 promotion gates while retaining "
            "all existing trial evidence"
        ),
    )
    amend_phase2_gates.add_argument("--root", type=Path, required=True)
    amend_phase2_gates.add_argument("--out", type=Path)

    advance = subparsers.add_parser(
        "advance",
        help="append the next blueprint round from an immutable frozen selection",
    )
    advance.add_argument("--root", type=Path, required=True)
    advance.add_argument("--selection", type=Path, required=True)
    advance.add_argument(
        "--selection-id",
        help="choose one frozen candidate if the policy retained multiple ties",
    )
    advance.add_argument(
        "--allow-insufficient-local-storage",
        action="store_true",
        help="append even when incremental materialization exceeds local free space",
    )

    advance_family = subparsers.add_parser(
        "advance-family",
        help="append the next declared round on one independent comparator branch",
    )
    advance_family.add_argument("--root", type=Path, required=True)
    advance_family.add_argument("--family", required=True)
    advance_family.add_argument("--selection", type=Path, required=True)
    advance_family.add_argument("--selection-id")
    advance_family.add_argument(
        "--allow-insufficient-local-storage",
        action="store_true",
        help="append even when incremental materialization exceeds local free space",
    )

    nominate_family = subparsers.add_parser(
        "nominate-family-revisit",
        help="freeze a family's top two over its complete 4M-round universe",
    )
    nominate_family.add_argument("--root", type=Path, required=True)
    nominate_family.add_argument("--family", required=True)
    nominate_family.add_argument("--out", type=Path)

    revisit_family = subparsers.add_parser(
        "revisit-family",
        help="materialize the fresh 16M rerun of a family's frozen top two",
    )
    revisit_family.add_argument("--root", type=Path, required=True)
    revisit_family.add_argument("--family", required=True)
    revisit_family.add_argument("--selection", type=Path, required=True)
    revisit_family.add_argument(
        "--allow-insufficient-local-storage",
        action="store_true",
        help="append even when incremental materialization exceeds local free space",
    )

    freeze_phase1 = subparsers.add_parser(
        "freeze-phase1",
        help="freeze complete Phase-1 evidence into a Phase-2 go/no-go decision",
    )
    freeze_phase1.add_argument("--root", type=Path, required=True)
    freeze_phase1.add_argument(
        "--scope-narrowing",
        type=_string_mapping_json,
        default={},
        metavar="JSON",
        help=(
            "JSON object mapping each failed non-control robustness stress ID "
            "to its explicit claim narrowing"
        ),
    )
    freeze_phase1.add_argument(
        "--out",
        type=Path,
        help=(
            "immutable decision path (default: "
            "ROOT/decisions/phase2-authorization.json)"
        ),
    )

    freeze_panel = subparsers.add_parser(
        "freeze-panel",
        help="freeze a complete qualified Phase-2 campaign for Phase 3",
    )
    freeze_panel.add_argument("--root", type=Path, required=True)
    freeze_panel.add_argument(
        "--out",
        type=Path,
        help="immutable decision path (default: ROOT/decisions/phase3-panel.json)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command in {"plan", "estimate"}:
            plan, blueprint, phase1_manifest, panel_manifest = (
                _build_plan_and_blueprint(args)
            )
            estimate = estimate_plan(plan)
            estimate_label = _estimate_label(plan, blueprint)
            if args.command == "estimate":
                _print(
                    {
                        "plan_id": plan.plan_id,
                        "phase": plan.phase.value,
                        "stages": len(plan.stages),
                        "blueprint_id": blueprint.blueprint_id,
                        "cell_count_ceiling": _cell_count_ceiling(blueprint),
                        "estimate_label": estimate_label,
                        "estimate": estimate.to_dict(),
                    }
                )
                return
            storage_preflight = _checked_storage_extension(
                args.root,
                estimate,
                allow_insufficient=args.allow_insufficient_local_storage,
                plan=plan,
            )
            campaign = Campaign(args.root)
            campaign.register(
                plan,
                blueprint_manifest=blueprint.to_manifest(),
                phase1_decision_manifest=phase1_manifest,
                panel_decision_manifest=panel_manifest,
            )
            _print(
                {
                    "root": str(args.root.resolve()),
                    "plan_id": plan.plan_id,
                    "phase": plan.phase.value,
                    "stages": len(plan.stages),
                    "blueprint_id": blueprint.blueprint_id,
                    "cell_count_ceiling": _cell_count_ceiling(blueprint),
                    "estimate_label": estimate_label,
                    "estimate": estimate.to_dict(),
                    "storage_preflight": storage_preflight,
                    "status": campaign.status(),
                }
            )
            return
        campaign = Campaign(args.root)
        if args.command == "run":
            enforce_plan_resources(campaign.plan)
            estimate = estimate_plan(campaign.plan)
            _checked_storage_extension(
                args.root,
                estimate,
                allow_insufficient=False,
                plan=campaign.plan,
                input_roots=(args.view_root,) if args.view_root is not None else (),
            )
            with _sigterm_unwinds_runner():
                summary = _run_with_optional_view_dispatch(campaign, args)
            _print({"run": summary.to_dict(), "status": campaign.status()})
            if summary.failed_cells:
                raise SystemExit(1)
        elif args.command == "status":
            _print(campaign.status())
        elif args.command == "reconcile":
            _print(
                {
                    "reconcile": campaign.reconcile(args.stale_after),
                    "status": campaign.status(),
                }
            )
        elif args.command == "select":
            _print(campaign.select_stage(args.stage, out=args.out))
        elif args.command == "select-family-root":
            _print(campaign.select_family_root(args.family, out=args.out))
        elif args.command == "amend-phase2-gates":
            _print(campaign.apply_phase2_gate_amendment(out=args.out))
        elif args.command == "nominate-family-revisit":
            _print(campaign.select_family_revisit_inputs(args.family, out=args.out))
        elif args.command == "freeze-phase1":
            _print(
                campaign.freeze_phase1_decision(
                    scope_narrowing=args.scope_narrowing,
                    out=args.out,
                )
            )
        elif args.command == "freeze-panel":
            _print(campaign.freeze_panel(out=args.out))
        elif args.command == "advance":
            selection = _frozen_selection(
                args.selection,
                selection_id=args.selection_id,
            )
            blueprint = _registered_blueprint(campaign)
            extended = materialize_child_plan(campaign.plan, blueprint, selection)
            estimate = estimate_plan(extended)
            storage_preflight = _checked_storage_extension(
                args.root,
                estimate,
                allow_insufficient=args.allow_insufficient_local_storage,
                plan=extended,
            )
            campaign.extend(
                extended,
                selection=selection,
                selection_path=args.selection,
            )
            _print(
                {
                    "selection_id": selection.selection_id,
                    "plan_id": extended.plan_id,
                    "appended_stage": extended.stages[-1].name,
                    "estimate_label": _estimate_label(extended, blueprint),
                    "estimate": estimate.to_dict(),
                    "storage_preflight": storage_preflight,
                    "status": campaign.status(),
                }
            )
        elif args.command == "advance-family":
            selection = _frozen_selection(
                args.selection,
                selection_id=args.selection_id,
            )
            blueprint = _registered_blueprint(campaign)
            if not isinstance(blueprint, Phase2Blueprint):
                raise StudyError("comparator-family branches belong only to Phase 2")
            extended = materialize_family_child_plan(
                campaign.plan,
                blueprint,
                args.family,
                selection,
            )
            estimate = estimate_plan(extended)
            storage_preflight = _checked_storage_extension(
                args.root,
                estimate,
                allow_insufficient=args.allow_insufficient_local_storage,
                plan=extended,
            )
            campaign.extend_family(
                extended,
                family_name=args.family,
                selection=selection,
                selection_path=args.selection,
            )
            _print(
                {
                    "family": args.family,
                    "selection_id": selection.selection_id,
                    "plan_id": extended.plan_id,
                    "appended_stage": extended.stages[-1].name,
                    "estimate_label": _estimate_label(extended, blueprint),
                    "estimate": estimate.to_dict(),
                    "storage_preflight": storage_preflight,
                    "status": campaign.status(),
                }
            )
        elif args.command == "revisit-family":
            blueprint = _registered_blueprint(campaign)
            if not isinstance(blueprint, Phase2Blueprint):
                raise StudyError("comparator-family revisits belong only to Phase 2")
            nomination = _read_object(args.selection)
            selected = nomination.get("selected")
            if not isinstance(selected, list):
                raise StudyError("family-revisit nomination lacks selected candidates")
            try:
                selections = tuple(FrozenSelection.from_dict(item) for item in selected)
            except (KeyError, TypeError, ValueError, StudyError) as exc:
                raise StudyError(f"invalid family-revisit nomination: {exc}") from exc
            extended = materialize_family_revisit_plan(
                campaign.plan,
                blueprint,
                args.family,
                selections,
            )
            estimate = estimate_plan(extended)
            storage_preflight = _checked_storage_extension(
                args.root,
                estimate,
                allow_insufficient=args.allow_insufficient_local_storage,
                plan=extended,
            )
            campaign.extend_family_revisit(
                extended,
                family_name=args.family,
                selection_path=args.selection,
            )
            _print(
                {
                    "family": args.family,
                    "selection_ids": [item.selection_id for item in selections],
                    "plan_id": extended.plan_id,
                    "appended_stage": extended.stages[-1].name,
                    "estimate_label": _estimate_label(extended, blueprint),
                    "estimate": estimate.to_dict(),
                    "storage_preflight": storage_preflight,
                    "status": campaign.status(),
                }
            )
        else:  # pragma: no cover - argparse enforces the command set
            parser.error(f"unknown command {args.command}")
    except (BudgetExceeded, CampaignError, StudyError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
