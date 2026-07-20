"""Plan, run, and inspect the Phase-0.5 paper-bridge training matrix.

The declared full profile is the Cartesian product of every finite factor
level in this module, filtered only by recipe-valid regularizer/Aux choices.
The screen profile runs one lower-LR, four-epoch representative of every
recipe in every normalization gauge before the costly optimizer factorial.
Successful non-final cells keep reports/logs/manifests and discard their
large checkpoints so the campaign can run on jobe's current 1 TB data disk.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


MATRIX_VERSION = "phase05-v2"
NORMALIZATIONS = ("none", "scalar", "layer", "whiten", "whiten_renorm")
LEARNING_RATES = (1e-4, 2e-4, 3e-4)
SCHEDULES = ("cosine", "linear_fifth")
EPOCHS = (2, 4, 8)
SITE_LAMBDAS = (0.0, 3e-4, 1e-3, 3e-3)
CROSSCODER_LAMBDAS = (1e-6, 3e-6, 1e-5, 3e-5, 1e-4)
GROUP_LASSO_LAMBDAS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
DEAD_WINDOW_TOKENS = (1_000, 32_768, 409_600)
AUX_RATIO_CAPS = (0.25, 1.0)
SEEDS = (0, 1)


@dataclass(frozen=True)
class Recipe:
    name: str
    arm: str
    blocks: int = 4096
    block_dim: int = 4
    k: float = 32.0
    selection: str = "batch_topk"
    encoder_mode: str = "untied"
    encoder_bias: bool = False
    code_activation: str = "signed"
    selection_score: str = "code_norm"
    decoder_constraint: str = "gram"
    regularizer: str = "site_profile"
    aux_variants: tuple[str, ...] = ("none", "sasa", "long_horizon", "fel")
    sites: tuple[int, ...] | None = None

    @property
    def lambdas(self) -> tuple[float, ...]:
        if self.regularizer in {"site_profile", "map_nuclear"}:
            return SITE_LAMBDAS
        if self.regularizer == "crosscoder_l1":
            return CROSSCODER_LAMBDAS
        if self.regularizer == "group_l21":
            return GROUP_LASSO_LAMBDAS
        return (0.0,)

    @property
    def primary_lambda(self) -> float:
        if self.regularizer in {"site_profile", "map_nuclear"}:
            return 1e-3
        if self.regularizer == "crosscoder_l1":
            return 1e-5
        if self.regularizer == "group_l21":
            return 1e-3
        return 0.0

    @property
    def primary_aux(self) -> str:
        return "sasa" if "sasa" in self.aux_variants else self.aux_variants[0]


RECIPES = (
    Recipe("bsc_batch", "bsc"),
    Recipe("bsc_batch_bias", "bsc", encoder_bias=True),
    Recipe("bsc_token", "bsc", selection="token_topk"),
    Recipe("bsc_token_bias", "bsc", selection="token_topk", encoder_bias=True),
    Recipe("bsc_threshold", "bsc", selection="threshold"),
    Recipe("bsc_threshold_bias", "bsc", selection="threshold", encoder_bias=True),
    Recipe("bsc_map_batch", "bsc", regularizer="map_nuclear"),
    Recipe("bsc_map_token", "bsc", selection="token_topk", regularizer="map_nuclear"),
    Recipe(
        "fel_grassmannian", "bsc", selection="token_topk", encoder_mode="tied",
        regularizer="none", aux_variants=("none", "fel"), sites=(18,),
    ),
    Recipe(
        "fel_vanilla", "bsc", selection="token_topk", encoder_bias=True,
        decoder_constraint="frobenius", regularizer="none",
        aux_variants=("none", "fel"), sites=(18,),
    ),
    Recipe(
        "fel_group_lasso", "bsc", selection="dense", encoder_bias=True,
        code_activation="group_soft_threshold", decoder_constraint="free",
        regularizer="group_l21", aux_variants=("none",), sites=(18,),
    ),
    Recipe(
        "scalar_signed_batch", "scalar", regularizer="none",
        aux_variants=("none", "sasa", "fel"),
    ),
    Recipe(
        "scalar_signed_token", "scalar", selection="token_topk",
        regularizer="none", aux_variants=("none", "sasa", "fel"),
    ),
    Recipe(
        "crosscoder_original", "scalar", selection="dense", encoder_bias=True,
        code_activation="relu", decoder_constraint="free",
        regularizer="crosscoder_l1", aux_variants=("none",),
    ),
    Recipe(
        "relu_batchtopk_bridge", "scalar", encoder_bias=True,
        code_activation="relu", selection_score="decoder_weighted",
        decoder_constraint="free",
        regularizer="none", aux_variants=("none",),
    ),
    Recipe(
        "sasa_paper_bridge", "bsc", block_dim=8, k=10,
        selection="token_topk", decoder_constraint="free",
        regularizer="map_nuclear", aux_variants=("none", "sasa"), sites=(18,),
    ),
)


def _job_id(config: dict) -> str:
    body = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(body).hexdigest()[:16]


def _job(recipe: Recipe, *, stage: str, normalization: str, lr: float,
         schedule: str, epochs: int, lam: float, aux_variant: str,
         dead_window_tokens: int, aux_ratio_cap: float | None, seed: int) -> dict:
    cfg = {
        "matrix_version": MATRIX_VERSION,
        "stage": stage,
        "recipe": recipe.name,
        "normalization": normalization,
        "lr": lr,
        "schedule": schedule,
        "epochs": epochs,
        "lam": lam,
        "aux_variant": aux_variant,
        "dead_window_tokens": dead_window_tokens,
        "aux_ratio_cap": aux_ratio_cap,
        "seed": seed,
        **asdict(recipe),
    }
    cfg.pop("name")
    cfg.pop("aux_variants")
    cfg["job_id"] = _job_id(cfg)
    return cfg


def build_jobs(profile: str) -> list[dict]:
    jobs: list[dict] = []
    if profile in {"screen", "all"}:
        for r, norm in itertools.product(RECIPES, NORMALIZATIONS):
            jobs.append(_job(
                r, stage="screen", normalization=norm, lr=1e-4,
                schedule="cosine", epochs=4, lam=r.primary_lambda,
                aux_variant=r.primary_aux, dead_window_tokens=1_000,
                aux_ratio_cap=1.0 if r.primary_aux in {"sasa", "long_horizon"} else None,
                seed=0,
            ))
    if profile in {"full", "all"}:
        for r in RECIPES:
            for norm, lr, sched, epochs, lam, aux, seed in itertools.product(
                NORMALIZATIONS, LEARNING_RATES, SCHEDULES, EPOCHS,
                r.lambdas, r.aux_variants, SEEDS,
            ):
                windows = DEAD_WINDOW_TOKENS if aux == "sasa" else (1_000,)
                caps = AUX_RATIO_CAPS if aux in {"sasa", "long_horizon"} else (None,)
                for window, cap in itertools.product(windows, caps):
                    jobs.append(_job(
                        r, stage="full", normalization=norm, lr=lr,
                        schedule=sched, epochs=epochs, lam=lam, aux_variant=aux,
                        dead_window_tokens=window, aux_ratio_cap=cap,
                        seed=seed,
                    ))
    # Stable order and no duplicate configurations across repeated generation.
    return sorted({j["job_id"]: j for j in jobs}.values(), key=lambda j: j["job_id"])


def _atomic_json(path: Path, payload: dict | list) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _load_state(root: Path) -> dict:
    path = root / "state.json"
    return json.loads(path.read_text()) if path.exists() else {"jobs": {}}


def _save_state(root: Path, state: dict) -> None:
    _atomic_json(root / "state.json", state)


def plan(root: Path, profile: str) -> list[dict]:
    root.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(profile)
    _atomic_json(root / "matrix.json", {
        "matrix_version": MATRIX_VERSION,
        "profile": profile,
        "n_jobs": len(jobs),
        "factors": {
            "normalization": NORMALIZATIONS,
            "lr": LEARNING_RATES,
            "schedule": SCHEDULES,
            "epochs": EPOCHS,
            "site_lambdas": SITE_LAMBDAS,
            "crosscoder_lambdas": CROSSCODER_LAMBDAS,
            "group_lasso_lambdas": GROUP_LASSO_LAMBDAS,
            "dead_window_tokens": DEAD_WINDOW_TOKENS,
            "aux_ratio_caps": AUX_RATIO_CAPS,
            "seeds": SEEDS,
            "recipes": [asdict(r) for r in RECIPES],
        },
        "jobs": jobs,
    })
    state = _load_state(root)
    for job in jobs:
        state["jobs"].setdefault(job["job_id"], {"status": "pending"})
    _save_state(root, state)
    return jobs


def harvest_stores(
    root: Path,
    store_root: Path,
    *,
    whitener_tokens: int,
    calibration_tokens: int,
    eval_tokens: int,
    train_tokens: int,
    device: str,
    model_revision: str | None,
) -> None:
    """Harvest every normalization from the identical pinned stream prefix."""
    root.mkdir(parents=True, exist_ok=True)
    store_root.mkdir(parents=True, exist_ok=True)
    state_path = root / "harvest_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    for normalization in NORMALIZATIONS:
        rec = state.setdefault(normalization, {"status": "pending"})
        if rec["status"] == "complete":
            continue
        out = store_root / normalization
        if out.exists() and any(out.iterdir()):
            rec.update(
                status="blocked",
                error=f"nonempty incomplete store requires explicit review: {out}",
            )
            _atomic_json(state_path, state)
            continue
        out.mkdir(parents=True, exist_ok=True)
        mode = "whiten" if normalization == "whiten_renorm" else normalization
        cmd = [
            sys.executable, "-m", "block_crosscoder_experiment.cli.harvest_store",
            "--out", str(out), "--device", device,
            "--normalization", mode,
            "--whitener-tokens", str(whitener_tokens),
            "--calib-tokens", str(calibration_tokens),
            "--eval-tokens", str(eval_tokens),
            "--train-tokens", str(train_tokens),
        ]
        if normalization == "whiten_renorm":
            cmd.append("--site-renorm")
        if model_revision is not None:
            cmd += ["--model-revision", model_revision]
        rec.update(status="running", started=time.time(), command=cmd)
        _atomic_json(state_path, state)
        with (root / f"harvest_{normalization}.log").open("a") as log:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        required = [out / "whitener.pt"] + [
            out / split / "split.json" for split in ("calibration", "eval", "train")
        ]
        if proc.returncode == 0 and all(p.exists() for p in required):
            rec.update(status="complete", finished=time.time(), returncode=0)
        else:
            rec.update(
                status="failed", finished=time.time(), returncode=proc.returncode,
                error="harvest failed or required manifests are missing",
            )
        _atomic_json(state_path, state)


def _command(job: dict, store_root: Path, run_root: Path, device: str) -> list[str]:
    store = store_root / job["normalization"]
    out = run_root / job["job_id"]
    cmd = [
        sys.executable, "-m", "block_crosscoder_experiment.cli.train_bsc",
        "--arm", job["arm"], "--store", str(store), "--out-root", str(out),
        "--raw-store", str(store_root / "none"),
        "--device", device, "--lam", str(job["lam"]), "--lr", str(job["lr"]),
        "--seed", str(job["seed"]),
        "--schedule", job["schedule"], "--epochs", str(job["epochs"]),
        "--blocks", str(job["blocks"]), "--block-dim", str(job["block_dim"]),
        "--k", str(job["k"]), "--selection", job["selection"],
        "--encoder-mode", job["encoder_mode"],
        "--code-activation", job["code_activation"],
        "--selection-score", job["selection_score"],
        "--decoder-constraint", job["decoder_constraint"],
        "--regularizer", job["regularizer"], "--aux-variant", job["aux_variant"],
        "--warmup-steps", "100", "--checkpoint-every", "1000",
    ]
    cmd.append("--encoder-bias" if job["encoder_bias"] else "--no-encoder-bias")
    if job["sites"] is not None:
        cmd += ["--sites", *(str(site) for site in job["sites"])]
    # Equal auxiliary coefficient capacity: block arm s_aux*b = scalar s_aux.
    if job["aux_variant"] != "none":
        s_aux = max(1, 256 // int(job["block_dim"])) if job["arm"] == "bsc" else 256
        if job["aux_variant"] == "fel":
            s_aux = max(1, round(job["k"] * (job["block_dim"] if job["arm"] == "scalar" else 1)))
        cmd += [
            "--s-aux", str(s_aux),
            "--dead-window-tokens", str(job["dead_window_tokens"]),
        ]
        if job["aux_ratio_cap"] is not None:
            cmd += ["--aux-ratio-cap", str(job["aux_ratio_cap"])]
    return cmd


def run(root: Path, store_root: Path, *, stage: str, device: str,
        retain_checkpoints: bool, limit: int | None) -> None:
    matrix_path = root / "matrix.json"
    if not matrix_path.exists():
        raise SystemExit(f"missing {matrix_path}; run `bsc phase05-matrix plan` first")
    jobs = json.loads(matrix_path.read_text())["jobs"]
    lock = root / "runner.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise SystemExit(f"matrix runner already active ({lock})") from exc
    os.write(fd, f"pid={os.getpid()} started={time.time()}\n".encode())
    os.close(fd)
    state = _load_state(root)
    ran = 0
    try:
        for job in jobs:
            if stage != "all" and job["stage"] != stage:
                continue
            rec = state["jobs"].setdefault(job["job_id"], {"status": "pending"})
            if rec["status"] == "complete":
                continue
            if limit is not None and ran >= limit:
                break
            store = store_root / job["normalization"]
            if not (store / "whitener.pt").exists():
                rec.update(status="blocked", error=f"missing store {store}")
                _save_state(root, state)
                continue
            run_root = root / "runs" / job["job_id"]
            run_root.mkdir(parents=True, exist_ok=True)
            _atomic_json(run_root / "job.json", job)
            cmd = _command(job, store_root, root / "runs", device)
            rec.update(status="running", started=time.time(), command=cmd)
            _save_state(root, state)
            with (run_root / "runner.log").open("a") as log:
                proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
            reports = list(run_root.rglob("report.json"))
            if proc.returncode == 0 and len(reports) == 1:
                rec.update(
                    status="complete", finished=time.time(),
                    report=str(reports[0]), returncode=0,
                )
                if not retain_checkpoints:
                    for checkpoint in run_root.rglob("latest.pt"):
                        checkpoint.unlink()
            else:
                rec.update(
                    status="failed", finished=time.time(), returncode=proc.returncode,
                    error=f"expected one report, found {len(reports)}",
                )
            _save_state(root, state)
            ran += 1
    finally:
        lock.unlink(missing_ok=True)


def status(root: Path) -> dict:
    state = _load_state(root)
    counts: dict[str, int] = {}
    for rec in state["jobs"].values():
        counts[rec["status"]] = counts.get(rec["status"], 0) + 1
    out = {"matrix_version": MATRIX_VERSION, "counts": counts}
    running = [k for k, v in state["jobs"].items() if v["status"] == "running"]
    failed = [k for k, v in state["jobs"].items() if v["status"] == "failed"]
    out["running"] = running
    out["failed"] = failed[:20]
    for name in ("harvest_state.json", "campaign_state.json"):
        path = root / name
        if path.exists():
            out[name.removesuffix(".json")] = json.loads(path.read_text())
    return out


def campaign(
    root: Path,
    store_root: Path,
    *,
    whitener_tokens: int,
    calibration_tokens: int,
    eval_tokens: int,
    train_tokens: int,
    device: str,
    model_revision: str | None,
) -> None:
    """Run the complete resumable campaign: harvest, screen, then factorial."""
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "campaign_state.json"

    def mark(status_value: str, **extra) -> None:
        payload = {
            "status": status_value,
            "updated": time.time(),
            "matrix_version": MATRIX_VERSION,
            **extra,
        }
        _atomic_json(state_path, payload)

    mark("planning")
    plan(root, "all")
    try:
        mark("harvesting")
        harvest_stores(
            root, store_root,
            whitener_tokens=whitener_tokens,
            calibration_tokens=calibration_tokens,
            eval_tokens=eval_tokens,
            train_tokens=train_tokens,
            device=device,
            model_revision=model_revision,
        )
        harvest = json.loads((root / "harvest_state.json").read_text())
        incomplete = {
            key: value for key, value in harvest.items()
            if value.get("status") != "complete"
        }
        if incomplete:
            raise RuntimeError(f"normalization harvest incomplete: {incomplete}")
        mark("screen")
        run(
            root, store_root, stage="screen", device=device,
            retain_checkpoints=False, limit=None,
        )
        failed_screen = [
            job_id for job_id, record in _load_state(root)["jobs"].items()
            if record.get("status") == "failed"
        ]
        if failed_screen:
            raise RuntimeError(
                f"screen produced {len(failed_screen)} failed cells; "
                "full factorial requires review"
            )
        mark("full_factorial")
        run(
            root, store_root, stage="full", device=device,
            retain_checkpoints=False, limit=None,
        )
        failed_full = [
            job_id for job_id, record in _load_state(root)["jobs"].items()
            if record.get("status") == "failed"
        ]
        if failed_full:
            raise RuntimeError(f"full factorial produced {len(failed_full)} failed cells")
    except BaseException as exc:
        mark("failed", error=repr(exc))
        raise
    mark("complete")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--profile", choices=("screen", "full", "all"), default="all")
    h = sub.add_parser("harvest")
    h.add_argument("--root", type=Path, required=True)
    h.add_argument("--store-root", type=Path, required=True)
    h.add_argument("--whitener-tokens", type=int, default=250_000)
    h.add_argument("--calib-tokens", type=int, default=500_000)
    h.add_argument("--eval-tokens", type=int, default=250_000)
    h.add_argument("--train-tokens", type=int, default=1_000_000)
    h.add_argument("--device", default="cuda")
    h.add_argument("--model-revision", default=None)
    c = sub.add_parser("campaign")
    c.add_argument("--root", type=Path, required=True)
    c.add_argument("--store-root", type=Path, required=True)
    c.add_argument("--whitener-tokens", type=int, default=250_000)
    c.add_argument("--calib-tokens", type=int, default=500_000)
    c.add_argument("--eval-tokens", type=int, default=250_000)
    c.add_argument("--train-tokens", type=int, default=1_000_000)
    c.add_argument("--device", default="cuda")
    c.add_argument("--model-revision", default=None)
    r = sub.add_parser("run")
    r.add_argument("--root", type=Path, required=True)
    r.add_argument("--store-root", type=Path, required=True)
    r.add_argument("--stage", choices=("screen", "full", "all"), default="screen")
    r.add_argument("--device", default="cuda")
    r.add_argument("--retain-checkpoints", action="store_true")
    r.add_argument("--limit", type=int, default=None)
    s = sub.add_parser("status")
    s.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "plan":
        jobs = plan(args.root, args.profile)
        print(json.dumps({"planned": len(jobs), "root": str(args.root)}))
    elif args.command == "harvest":
        harvest_stores(
            args.root, args.store_root,
            whitener_tokens=args.whitener_tokens,
            calibration_tokens=args.calib_tokens,
            eval_tokens=args.eval_tokens,
            train_tokens=args.train_tokens,
            device=args.device, model_revision=args.model_revision,
        )
    elif args.command == "run":
        run(
            args.root, args.store_root, stage=args.stage, device=args.device,
            retain_checkpoints=args.retain_checkpoints, limit=args.limit,
        )
    elif args.command == "campaign":
        campaign(
            args.root, args.store_root,
            whitener_tokens=args.whitener_tokens,
            calibration_tokens=args.calib_tokens,
            eval_tokens=args.eval_tokens,
            train_tokens=args.train_tokens,
            device=args.device, model_revision=args.model_revision,
        )
    else:
        print(json.dumps(status(args.root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
