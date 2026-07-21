"""Plan and operate the declarative three-phase BSC campaign.

This command intentionally does not promote cells.  Promotion consumes an
explicit, hash-bound decision artifact through the campaign API after review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Sequence

from block_crosscoder_experiment.campaign import (
    Campaign,
    CampaignError,
    CampaignRunner,
    RunSummary,
)
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
    estimate_plan,
    materialize_child_plan,
    materialize_family_child_plan,
    materialize_family_revisit_plan,
)


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


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
    path.parent.mkdir(parents=True, exist_ok=True)
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
        os.replace(temporary, path)
    finally:
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

    manifest_file_sha256 = _sha256(manifest_path)
    fingerprint = {
        "manifest": stat_record(manifest_path),
        "shards": [
            stat_record(split_dir / str(record["file"]))
            for record in reader.manifest["shards"]
        ],
    }
    key = _canonical_hash(
        {
            "root": str(split_dir.parent.resolve()),
            "split": split_dir.name,
            "manifest_sha256": manifest_file_sha256,
        }
    )
    receipt_path = cache_root / f"{key}.json"
    expected = {
        "schema": "bsc-store-verification-receipt-v1",
        "root": str(split_dir.parent.resolve()),
        "split": split_dir.name,
        "manifest_sha256": manifest_file_sha256,
        "manifest_content_sha256": reader.manifest.get("manifest_sha256"),
        "content_stream_sha256": reader.manifest.get("content_stream_sha256"),
        "row_stream_sha256": reader.manifest.get("row_stream_sha256"),
        "n_tokens": reader.n_tokens,
        "stat_fingerprint": fingerprint,
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


def _configured_input_roots() -> tuple[Path, ...]:
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
    for name in names:
        value = os.environ.get(name)
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
) -> dict[str, object]:
    """Hash-verify configured immutable inputs and count their physical bytes.

    Only files reached through a verified split or transform manifest count.
    Merely pointing an environment variable at a directory never buys storage
    credit, and overlapping environment roots are deduplicated by resolved path.
    """

    counted_files: set[Path] = set()
    records: list[dict[str, object]] = []
    for root in _configured_input_roots():
        if not root.is_dir():
            raise StudyError(f"configured activation input is not a directory: {root}")
        split_manifests = sorted(root.rglob("split.json"))
        transform_manifests = sorted(root.rglob("transform.json"))
        root_files: set[Path] = set()
        verified_splits: list[str] = []
        verified_transforms: list[str] = []
        for manifest_path in split_manifests:
            split_dir = manifest_path.parent
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
            root_files.add(manifest_path.resolve())
            for shard in reader.manifest["shards"]:
                root_files.add((split_dir / shard["file"]).resolve())
            verified_splits.append(str(split_dir.relative_to(root)))
        for manifest_path in transform_manifests:
            try:
                manifest = json.loads(manifest_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise StudyError(
                    f"invalid transform manifest {manifest_path}: {exc}"
                ) from exc
            transform_path = manifest_path.parent / "whitener.pt"
            if (
                manifest.get("schema") != "bsc-transform-artifact-v1"
                or not transform_path.is_file()
                or manifest.get("whitener_sha256") != _sha256(transform_path)
            ):
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
            root_files.update((manifest_path.resolve(), transform_path.resolve()))
            verified_transforms.append(str(manifest_path.parent.relative_to(root)))
        for capture_path in root.rglob("capture.json"):
            try:
                capture = json.loads(capture_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise StudyError(
                    f"invalid capture manifest {capture_path}: {exc}"
                ) from exc
            source = capture.get("source")
            if not isinstance(source, dict) or capture.get(
                "source_hash"
            ) != _canonical_hash(source):
                raise StudyError(f"capture source hash mismatch at {capture_path}")
            root_files.add(capture_path.resolve())
        if not verified_splits and not verified_transforms:
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
            }
        )
    return {
        "verified_existing_input_bytes": sum(
            path.stat().st_size for path in counted_files
        ),
        "inputs": records,
    }


def _storage_preflight(root: Path, estimated_storage_bytes: int) -> dict[str, object]:
    existing = _verified_existing_input_storage(
        verification_cache_root=root / ".store-verification"
    )
    checkpoint_files: set[Path] = set()
    if (root / "plan.json").is_file():
        existing_campaign = Campaign(root)
        for record in existing_campaign.records():
            checkpoint = record.artifact_map.get("checkpoint")
            if checkpoint is None:
                continue
            checkpoint.verify(root)
            checkpoint_files.add(checkpoint.resolve(root).resolve())
    checkpoint_bytes = sum(path.stat().st_size for path in checkpoint_files)
    input_credit = min(
        estimated_storage_bytes,
        int(existing["verified_existing_input_bytes"]),
    )
    checkpoint_credit = min(
        max(0, estimated_storage_bytes - input_credit),
        checkpoint_bytes,
    )
    credited = input_credit + checkpoint_credit
    additional = max(0, estimated_storage_bytes - credited)
    parent = _nearest_existing_parent(root)
    free = shutil.disk_usage(parent).free
    return {
        "estimate_scope": "materialized_plan_prefix_or_frozen_panel",
        "estimated_storage_bytes": estimated_storage_bytes,
        **existing,
        "verified_existing_campaign_checkpoint_bytes": checkpoint_bytes,
        "credited_existing_input_bytes": input_credit,
        "credited_existing_campaign_checkpoint_bytes": checkpoint_credit,
        "credited_existing_storage_bytes": credited,
        "additional_storage_bytes_required": additional,
        "free_bytes": free,
        "filesystem_path": str(parent),
        "sufficient": additional <= free,
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
        tuple[Path, dict[str, tuple[object, ...]], dict[str, int]],
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
                signatures[split] = (
                    reader.n_tokens,
                    reader.manifest["row_stream_sha256"],
                    tuple(reader.manifest["sites"]),
                    reader.d_model,
                )
            by_mode[mode] = (mode_root, signatures, declared)
        elif by_mode[mode][2] != declared:
            raise StudyError(
                f"cells using Phase-2 view {mode!r} declare different split contracts"
            )
        dispatched[cell_id] = by_mode[mode][0]

    mode_items = list(by_mode.items())
    if mode_items:
        reference_mode, (_, reference, _) = mode_items[0]
        for mode, (_, signatures, _) in mode_items[1:]:
            if signatures != reference:
                raise StudyError(
                    f"Phase-2 views {reference_mode!r} and {mode!r} do not share "
                    "one exact row stream"
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
        selected = list(campaign.runnable_cell_ids(include_failed=args.resume))
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
) -> dict[str, object]:
    preflight = _storage_preflight(root, estimate.storage_bytes)
    if not allow_insufficient and not preflight["sufficient"]:
        raise BudgetExceeded(
            f"storage_bytes: conservative cumulative estimate {estimate.storage_bytes}; "
            "incremental requirement "
            f"{preflight['additional_storage_bytes_required']} exceeds "
            f"{preflight['free_bytes']} free bytes at "
            f"{preflight['filesystem_path']} after crediting "
            f"{preflight['credited_existing_storage_bytes']} bytes of hash-verified "
            "configured inputs and existing campaign checkpoints; choose a larger "
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
    run.add_argument("--limit", type=_nonnegative_int)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--cell", action="append", dest="cells")
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
            summary = _run_with_optional_view_dispatch(campaign, args)
            _print({"run": summary.to_dict(), "status": campaign.status()})
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
