"""Run the bounded pre-NVMe manifold-fidelity screen on the 4B pilot store.

The matrix is a response-surface design, not a Cartesian explosion.  It adds
the missing causal comparisons around the existing G4096/b4/k32/lr3e-4
center: k at fixed G, G at fixed k and fixed density, the safe side of the
learning-rate cliff, and a width sweep that jointly fixes ``G*b=16384``,
``k*b=128``, and ``k/G=1/128``.

Every cell is isolated under ``OUT_ROOT/<cell>/seed<seed>``.  Completed cells
are skipped, interrupted cells resume, and high-risk G/LR cells first stop at
a diagnostic checkpoint.  The runner continues after a failed cell so one
G8192 OOM cannot erase an overnight campaign.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path


def cell_grid() -> dict[str, dict]:
    return {
        # Reopen the untested interval below the demonstrated 6e-4 cliff.
        # 1800 steps crosses warmup and the batch-locked step-1600 stressor.
        "lr3.5e-4": dict(G=4096, b=4, k=32, lr=3.5e-4, diagnostic_steps=1800),
        "lr4e-4": dict(G=4096, b=4, k=32, lr=4e-4, diagnostic_steps=1800),
        "lr4.5e-4": dict(G=4096, b=4, k=32, lr=4.5e-4, diagnostic_steps=1800),
        "lr5e-4": dict(G=4096, b=4, k=32, lr=5e-4, diagnostic_steps=1800),
        "lr5.5e-4": dict(G=4096, b=4, k=32, lr=5.5e-4, diagnostic_steps=1800),
        "linear3e-4": dict(
            G=4096, b=4, k=32, lr=3e-4, schedule="linear_fifth",
            diagnostic_steps=1800,
        ),
        # Follow-up interaction surface after linear k32 and cosine k40/48/64
        # emerged as complementary geometry controls.
        "linear_k40": dict(
            G=4096, b=4, k=40, lr=3e-4, schedule="linear_fifth",
            diagnostic_steps=1800,
        ),
        "linear_k48": dict(
            G=4096, b=4, k=48, lr=3e-4, schedule="linear_fifth",
            diagnostic_steps=1800,
        ),
        "linear_k64": dict(
            G=4096, b=4, k=64, lr=3e-4, schedule="linear_fifth",
            diagnostic_steps=1800,
        ),
        # Honest lambda=1e-3 k sweep (the surviving off-points are lambda=0).
        "k16": dict(G=4096, b=4, k=16, lr=3e-4),
        "k24": dict(G=4096, b=4, k=24, lr=3e-4),
        "k40": dict(G=4096, b=4, k=40, lr=3e-4),
        "k48": dict(G=4096, b=4, k=48, lr=3e-4),
        "k64": dict(G=4096, b=4, k=64, lr=3e-4),
        # G x k disentangling around the existing (4096,32) center.
        "G2048_k16": dict(G=2048, b=4, k=16, lr=3e-4),
        "G2048_k32": dict(G=2048, b=4, k=32, lr=3e-4),
        "G8192_k32": dict(
            G=8192, b=4, k=32, lr=3e-4, diagnostic_steps=500,
            hardware_rejected=True,
        ),
        "G8192_k64": dict(
            G=8192, b=4, k=64, lr=3e-4, diagnostic_steps=500,
            hardware_rejected=True,
        ),
        # Width sweep: all three matching quantities above are held fixed.
        "b2_G8192_k64": dict(G=8192, b=2, k=64, lr=3e-4, diagnostic_steps=500),
        "b8_G2048_k16": dict(G=2048, b=8, k=16, lr=3e-4),
    }


def _event(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps({"time": time.time(), **payload}) + "\n")


def _run(command: list[str], *, dry_run: bool) -> int:
    print("+", " ".join(command), flush=True)
    return 0 if dry_run else subprocess.run(command, check=False).returncode


def _diagnostic_health(cell_root: Path) -> tuple[bool, dict]:
    """Cheap preregistered gate before an expensive diagnostic is resumed."""

    logs = list(cell_root.glob("*/steps.jsonl"))
    if len(logs) != 1:
        return False, {"reason": f"expected one steps log, found {len(logs)}"}
    metrics = []
    skipped = 0
    with logs[0].open() as handle:
        for line in handle:
            row = json.loads(line)
            event = row.get("guard_event")
            if event is not None:
                skipped += int(bool(event.get("skipped")))
            elif "rec" in row:
                metrics.append(row)
    if len(metrics) < 2:
        return False, {"reason": "diagnostic emitted fewer than two metric rows"}
    last = metrics[-1]
    finite_keys = ("rec", "total", "grad_norm", "gram_residual_postcast")
    finite = all(math.isfinite(float(last.get(key, float("nan")))) for key in finite_keys)
    steps = int(last["step"]) + 1
    skip_rate = skipped / max(steps, 1)
    rec_ratio = float(last["rec"]) / max(float(metrics[0]["rec"]), 1e-12)
    healthy = (
        finite
        and skip_rate <= 0.001
        and rec_ratio <= 1.2
        and float(last.get("gram_residual_postcast", float("inf"))) <= 0.01
        and int(last.get("floor_hits", 0)) == 0
        and float(last.get("dead_frac_window", 1.0)) < 0.25
    )
    return healthy, {
        "reason": "healthy" if healthy else "stability gate failed",
        "last_step": int(last["step"]),
        "rec_ratio_vs_first": rec_ratio,
        "skip_rate": skip_rate,
        "gram_residual_postcast": last.get("gram_residual_postcast"),
        "dead_frac_window": last.get("dead_frac_window"),
        "floor_hits": last.get("floor_hits"),
    }


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)


def _cell_fingerprint(args, cell_name: str, cell: dict, seed: int) -> dict:
    return {
        "format": 1,
        "cell": cell_name,
        "seed": seed,
        "G": cell["G"],
        "b": cell["b"],
        "k": cell["k"],
        "lr": cell["lr"],
        "schedule": cell.get("schedule", "cosine"),
        "lambda_rank": args.lam,
        "epochs": args.epochs,
        "train_split": args.train_split,
        "warmup_steps": args.warmup_steps,
        "store": str(args.store.resolve()),
        "site_renorm": True,
        "guard": True,
        "aux_ratio_cap": 1.0,
        "diagnostic_steps": cell.get("diagnostic_steps"),
    }


def _verify_or_initialize_fingerprint(
    fingerprint_path: Path, fingerprint: dict, *, dry_run: bool
) -> None:
    if fingerprint_path.exists():
        recorded = json.loads(fingerprint_path.read_text())
        if recorded != fingerprint:
            raise SystemExit(
                f"configuration mismatch at {fingerprint_path}; use a "
                "distinct horizon/configuration root\n"
                f"recorded={recorded}\nrequested={fingerprint}"
            )
        return

    cell_root = fingerprint_path.parent
    legacy_artifact = (
        next((path for path in cell_root.rglob("*") if path.is_file()), None)
        if cell_root.exists() else None
    )
    if legacy_artifact is not None:
        raise SystemExit(
            "refusing to adopt artifact-bearing cell without an immutable "
            f"fingerprint: {legacy_artifact}; use a distinct campaign root"
        )
    if not dry_run:
        _write_json_atomic(fingerprint_path, fingerprint)


def main() -> None:
    grid = cell_grid()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--cells", nargs="*", choices=list(grid), default=None)
    parser.add_argument("--seeds", type=int, nargs="*", default=[0])
    parser.add_argument(
        "--epochs", type=int, default=2,
        help="matched passes over the 6M pilot train split (default 2 = 12M); "
        "finalists use 4 = 24M",
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--lam", type=float, default=1e-3)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--initialize-only", action="store_true",
        help="write/verify immutable cell fingerprints without training; useful "
        "when adopting an already-running campaign started by this runner",
    )
    args = parser.parse_args()

    if args.cells:
        selected = args.cells
    else:
        # Pay the VRAM uncertainty first, before an overnight sequence of
        # already-known-to-fit G4096 cells.
        high_risk = ["b2_G8192_k64"]
        selected = [
            *high_risk,
            *(
                name for name, cell in grid.items()
                if name not in high_risk and not cell.get("hardware_rejected")
            ),
        ]
    manifest = args.out_root / "campaign.jsonl"
    failures = []
    for cell_name in selected:
        cell = grid[cell_name]
        for seed in args.seeds:
            cell_root = args.out_root / cell_name / f"seed{seed}"
            fingerprint = _cell_fingerprint(args, cell_name, cell, seed)
            fingerprint_path = cell_root / "config.json"
            _verify_or_initialize_fingerprint(
                fingerprint_path, fingerprint, dry_run=args.dry_run
            )
            if args.initialize_only:
                print(f"[configured] {cell_name} seed {seed}", flush=True)
                continue
            reports = list(cell_root.glob("*/report.json"))
            if reports:
                print(f"[skip] {cell_name} seed {seed}: {reports[0]}", flush=True)
                continue
            checkpoints = list(cell_root.glob("*/latest.pt"))
            diagnostic_marker = cell_root / "diagnostic.json"
            common = [
                sys.executable,
                "-m",
                "block_crosscoder_experiment.cli",
                "train",
                "--arm", "bsc",
                "--lam", str(args.lam),
                "--seed", str(seed),
                "--lr", str(cell["lr"]),
                "--schedule", cell.get("schedule", "cosine"),
                "--blocks", str(cell["G"]),
                "--block-dim", str(cell["b"]),
                "--k", str(cell["k"]),
                "--warmup-steps", str(args.warmup_steps),
                "--site-renorm",
                "--epochs", str(args.epochs),
                "--guard",
                "--aux-ratio-cap", "1",
                "--train-split", args.train_split,
                "--store", str(args.store),
                "--out-root", str(cell_root),
            ]
            if not args.dry_run:
                _event(manifest, {
                    "event": "start", "cell": cell_name, "seed": seed,
                    "config": cell,
                })
            if "diagnostic_steps" in cell:
                marker = (
                    json.loads(diagnostic_marker.read_text())
                    if diagnostic_marker.exists() else None
                )
                if marker is not None and marker.get("config") != fingerprint:
                    raise SystemExit(
                        f"diagnostic marker/config mismatch at {diagnostic_marker}"
                    )
                if marker is not None and not marker.get("healthy", False):
                    print(
                        f"[reject] {cell_name} seed {seed}: persisted diagnostic "
                        f"failure {marker}",
                        flush=True,
                    )
                    code = 86
                elif marker is not None:
                    code = _run([*common, "--resume"], dry_run=args.dry_run)
                else:
                    diagnostic_command = [
                        *common,
                        "--max-steps", str(cell["diagnostic_steps"]),
                    ]
                    if checkpoints:
                        diagnostic_command.append("--resume")
                    code = _run(diagnostic_command, dry_run=args.dry_run)
                    if code == 0 and not args.dry_run:
                        healthy, diagnostic = _diagnostic_health(cell_root)
                        marker = {
                            "healthy": healthy, "config": fingerprint, **diagnostic
                        }
                        _write_json_atomic(diagnostic_marker, marker)
                        _event(manifest, {
                            "event": "diagnostic", "cell": cell_name, "seed": seed,
                            **marker,
                        })
                        if not healthy:
                            print(
                                f"[reject] {cell_name} seed {seed}: {diagnostic}",
                                flush=True,
                            )
                            code = 86
                    if code == 0:
                        code = _run([*common, "--resume"], dry_run=args.dry_run)
            elif checkpoints:
                code = _run([*common, "--resume"], dry_run=args.dry_run)
            else:
                code = _run(common, dry_run=args.dry_run)
            if not args.dry_run:
                _event(manifest, {
                    "event": "finish" if code == 0 else "failure",
                    "cell": cell_name, "seed": seed, "exit_code": code,
                })
            if code != 0:
                failures.append((cell_name, seed, code))

    if failures:
        print("\nfailed cells:", flush=True)
        for cell, seed, code in failures:
            print(f"  {cell} seed {seed}: exit {code}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
