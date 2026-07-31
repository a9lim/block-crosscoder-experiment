"""Plan and operate the declarative three-phase BSC campaign.

Selections and phase transitions are explicit JSON files intended for review.
The operator may edit or replace them before continuing.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
import shutil
import signal
import sys
from pathlib import Path
from typing import Sequence

from block_crosscoder_experiment.cli.data import DEFAULT_FREE_SPACE_FLOOR_FRAC
from block_crosscoder_experiment.campaign import (
    Campaign,
    CampaignError,
    CampaignRunner,
    RunSummary,
)
from block_crosscoder_experiment.store import NORMALIZATION_MODES, StoreReader
from block_crosscoder_experiment.studies import (
    Budget,
    BudgetExceeded,
    FrozenSelection,
    Phase,
    Phase1Blueprint,
    Phase2Blueprint,
    StudyPlan,
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


def _existing_input_storage(
    *,
    plan=None,
    input_roots: Sequence[Path] = (),
) -> dict[str, object]:
    """Count ordinary files under the activation roots selected by the operator."""

    if plan is not None and plan.phase is Phase.PHASE1:
        return {
            "existing_input_bytes": 0,
            "inputs": [],
            "plan_input_contract": "stateless_phase1",
        }

    counted: set[Path] = set()
    records: list[dict[str, object]] = []
    for root in _configured_input_roots(input_roots):
        if not root.is_dir():
            raise StudyError(f"configured activation input is not a directory: {root}")
        files = {
            path.resolve()
            for path in root.rglob("*")
            if path.is_file() and ".store-verification" not in path.parts
        }
        new_files = files - counted
        counted.update(files)
        records.append(
            {
                "root": str(root),
                "bytes": sum(path.stat().st_size for path in new_files),
                "files": len(files),
            }
        )
    return {
        "existing_input_bytes": sum(path.stat().st_size for path in counted),
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
        if key[12] == "derived_views":
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
    existing = _existing_input_storage(
        plan=plan,
        input_roots=input_roots,
    )
    campaign_artifact_files: set[Path] = set()
    if (root / "plan.json").is_file():
        existing_campaign = Campaign(root)
        for record in existing_campaign.records():
            for artifact in record.artifact_map.values():
                try:
                    artifact.verify(root)
                    resolved = artifact.resolve(root).resolve()
                except (CampaignError, OSError):
                    # Missing outputs simply do not earn storage credit.
                    continue
                campaign_artifact_files.add(resolved)
    campaign_artifact_bytes = sum(
        path.stat().st_size for path in campaign_artifact_files
    )
    estimated_input_bytes = min(
        estimated_storage_bytes,
        (
            _estimated_plan_input_storage_bytes(plan)
            if plan is not None
            else int(existing["existing_input_bytes"])
        ),
    )
    estimated_campaign_bytes = estimated_storage_bytes - estimated_input_bytes
    input_credit = min(
        estimated_input_bytes,
        int(existing["existing_input_bytes"]),
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
        "existing_campaign_artifact_bytes": campaign_artifact_bytes,
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
    """Resolve each cell to the normalization directory chosen by the operator."""

    root = view_root.expanduser().resolve()
    if not root.is_dir():
        raise StudyError(f"Phase-2 --view-root is not a directory: {root}")

    checked: dict[str, tuple[Path, dict[str, tuple[int, tuple[int, ...], int]]]] = {}
    dispatched: dict[str, Path] = {}
    for cell_id, cell in cells.items():
        values = cell.decision_map
        mode = str(values["data.normalization"])
        if mode not in NORMALIZATION_MODES:
            raise StudyError(f"cell {cell_id} has unsupported normalization {mode!r}")
        declared = {
            str(name): int(tokens) for name, tokens in values["data.split_sizes"]
        }
        if mode not in checked:
            mode_root = root / mode
            if not mode_root.is_dir():
                raise StudyError(f"Phase-2 view {mode!r} does not exist under {root}")
            signatures: dict[str, tuple[int, tuple[int, ...], int]] = {}
            for split, requested in declared.items():
                if not (mode_root / split / "split.json").is_file():
                    raise StudyError(f"Phase-2 view {mode!r} lacks split {split!r}")
                reader = StoreReader(mode_root, split)
                if reader.n_tokens < requested:
                    raise StudyError(
                        f"Phase-2 view {mode!r}/{split} has {reader.n_tokens} rows; "
                        f"{requested} are required"
                    )
                signatures[split] = (
                    reader.n_tokens,
                    tuple(reader.site_dims),
                    reader.d_model,
                )
            checked[mode] = (mode_root, signatures)
        elif set(checked[mode][1]) != set(declared):
            raise StudyError(
                f"cells using Phase-2 view {mode!r} declare different split roles"
            )
        dispatched[cell_id] = checked[mode][0]

    if checked:
        reference_mode, (_, reference) = next(iter(checked.items()))
        for mode, (_, signatures) in list(checked.items())[1:]:
            if signatures != reference:
                raise StudyError(
                    f"Phase-2 views {reference_mode!r} and {mode!r} have "
                    "different row counts or geometry"
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
        help="reviewed Phase-1 go/no-go decision required for Phase 2",
    )
    parser.add_argument(
        "--panel-decision",
        type=Path,
        help="reviewed Phase-2 panel decision required for Phase 3",
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
    if candidates is None and isinstance(payload.get("selection"), dict):
        candidates = [payload["selection"]]
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
            f"{preflight['credited_existing_storage_bytes']} bytes of existing "
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

    plan = subparsers.add_parser("plan", help="materialize and register a plan")
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

    status = subparsers.add_parser("status", help="show current campaign status")
    status.add_argument("--root", type=Path, required=True)

    select = subparsers.add_parser(
        "select",
        help="rank a completed stage with its declared policy",
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
        help="append the next blueprint round from a reviewed selection",
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
        help="decision path (default: ROOT/decisions/phase2-authorization.json)",
    )

    freeze_panel = subparsers.add_parser(
        "freeze-panel",
        help="freeze a complete qualified Phase-2 campaign for Phase 3",
    )
    freeze_panel.add_argument("--root", type=Path, required=True)
    freeze_panel.add_argument(
        "--out",
        type=Path,
        help="decision path (default: ROOT/decisions/phase3-panel.json)",
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
