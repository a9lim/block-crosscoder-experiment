"""Train and evaluate one production BSC or matched scalar control.

The production defaults are the ratified Phase-1 operating point: 4096
four-dimensional blocks, k=32, lambda=1e-3, cosine 3e-4 learning rate,
streaming threshold calibration, the mandatory spike guard, AuxK gradient
ratio cap 1.0, and four-batch prefetch.  The scalar control keeps the same
latent and training-average-L0 budget.  Store and output locations are always
explicit because production storage is host-specific.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

G, B_DIM, K = 4096, 4, 32.0
BATCH = 4096
EPOCHS = 2
SHUFFLE_SEED = 0  # recorded; shared verbatim by BSC and baseline (design)


def _json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _git_revision() -> str:
    repo = Path(__file__).resolve().parents[2]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("training requires a resolvable git revision") from exc


def _code_sha256() -> str:
    """Fingerprint the Python implementation actually present on disk."""
    repo = Path(__file__).resolve().parents[2]
    paths = sorted((repo / "block_crosscoder_experiment").rglob("*.py"))
    pyproject = repo / "pyproject.toml"
    if pyproject.exists():
        paths.append(pyproject)
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(repo)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _prepare_run_dir(run_dir: Path, manifest: dict, *, resume: bool) -> None:
    """Create one uncontaminated run directory or validate an exact resume."""
    # Compare the same JSON-domain value that is persisted. Dataclass tuples
    # (notably Adam betas) otherwise deserialize as lists and make every valid
    # resume look mismatched.
    canonical = json.loads(json.dumps(manifest, sort_keys=True))
    path = run_dir / "run_manifest.json"
    if resume:
        if not path.exists():
            raise SystemExit(
                f"--resume requires bound run manifest at {path}; legacy/unbound "
                "run directories are refused"
            )
        existing = json.loads(path.read_text())
        if existing != canonical:
            raise SystemExit(
                "resume run-manifest mismatch: "
                + json.dumps(
                    {"existing": existing, "expected": canonical}, sort_keys=True
                )
            )
        return
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(
            f"fresh run refuses non-empty directory {run_dir}; use --resume only "
            "for an exact bound continuation"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(canonical, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _calibration_extent(
    n_tokens: int, batch_size: int, cap_batches: int | None
) -> tuple[int, int]:
    """Return (batches, tokens) without dropping the final partial batch."""
    if n_tokens <= 0 or batch_size <= 0:
        raise ValueError("calibration token and batch counts must be positive")
    if cap_batches is not None and cap_batches <= 0:
        raise ValueError("calibration batch cap must be positive")
    available = (n_tokens + batch_size - 1) // batch_size
    batches = available if cap_batches is None else min(cap_batches, available)
    return batches, min(n_tokens, batches * batch_size)


def _validate_raw_alignment(normalized, raw, normalized_whitener, raw_whitener) -> None:
    """Fail closed unless two eval stores describe the same token stream."""
    mismatches: dict[str, object] = {}
    if raw_whitener.mode != "none":
        mismatches["raw_normalization"] = raw_whitener.mode
    for name in ("n_tokens", "sites", "d_model"):
        left, right = getattr(normalized, name), getattr(raw, name)
        if left != right:
            mismatches[name] = {"normalized": left, "raw": right}
    normalized_meta = normalized.manifest.get("meta", {})
    raw_meta = raw.manifest.get("meta", {})
    provenance = (
        "model",
        "model_revision",
        "corpus",
        "corpus_config",
        "corpus_revision",
        "corpus_split",
        "context_size",
        "prepend_bos",
        "dropped_positions",
        "pack_convention",
        "hook_names",
    )
    for name in provenance:
        if name not in normalized_meta or name not in raw_meta:
            mismatches[f"meta.{name}"] = "missing alignment provenance"
        elif normalized_meta[name] != raw_meta[name]:
            mismatches[f"meta.{name}"] = {
                "normalized": normalized_meta[name],
                "raw": raw_meta[name],
            }
    if normalized_whitener.meta.get("model") != raw_whitener.meta.get("model"):
        mismatches["whitener.model"] = {
            "normalized": normalized_whitener.meta.get("model"),
            "raw": raw_whitener.meta.get("model"),
        }
    if mismatches:
        raise ValueError(
            "raw eval store is not token/site/model aligned: "
            + json.dumps(mismatches, sort_keys=True, default=list)
        )


def _raw_reconstruct(
    whitener, xhat_normalized, raw_x, *, linear_inverse=None
):
    """Map a normalized reconstruction to paired raw activation coordinates."""
    if whitener.mode != "layer":
        if linear_inverse is not None:
            winv, mean = linear_inverse
            # Exact Whitener.unapply contraction with its matrix inverse
            # cached once rather than recomputed for every 4096-token batch.
            return (
                xhat_normalized.transpose(0, 1) @ winv.transpose(1, 2)
            ).transpose(0, 1) + mean
        return whitener.unapply(xhat_normalized)
    raw_mu = raw_x.mean(dim=-1, keepdim=True)
    raw_var = raw_x.var(dim=-1, correction=0, keepdim=True)
    eps = float(whitener.meta.get("layer_norm_eps", 1e-5))
    return xhat_normalized * (raw_var + eps).sqrt() + raw_mu


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("bsc", "scalar"), required=True)
    parser.add_argument(
        "--lam", type=float, default=None,
        help="coefficient for --regularizer (default: 1e-3 for BSC, 0 for scalar)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--schedule", choices=("cosine", "linear_fifth"), default="cosine",
        help="lr decay: cosine-to-zero (default) or SASA B.3 linear-last-fifth",
    )
    parser.add_argument("--encoder-wd", type=float, default=0.0)
    parser.add_argument(
        "--blocks", type=int, default=G,
        help="block count G (production default: 4096)",
    )
    parser.add_argument(
        "--block-dim", type=int, default=B_DIM,
        help="block width b (production default: 4); width sweeps should state "
        "which latent, coefficient, density, and rate quantities are matched",
    )
    parser.add_argument("--k", type=float, default=K)
    parser.add_argument(
        "--selection",
        choices=("batch_topk", "token_topk", "threshold", "dense"),
        default="batch_topk",
    )
    parser.add_argument(
        "--selection-score",
        choices=("code_norm", "decoder_weighted"),
        default="code_norm",
    )
    parser.add_argument(
        "--encoder-mode", choices=("untied", "tied"), default="untied"
    )
    parser.add_argument(
        "--encoder-bias", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--code-activation",
        choices=("signed", "relu", "group_soft_threshold"),
        default="signed",
    )
    parser.add_argument(
        "--decoder-constraint",
        choices=("gram", "frobenius", "free"),
        default="gram",
    )
    parser.add_argument(
        "--regularizer",
        choices=(
            "none",
            "site_profile",
            "map_nuclear",
            "crosscoder_l1",
            "group_l21",
        ),
        default=None,
        help="regularizer family; --lam is its coefficient",
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=1000,
        help="linear warmup length (production default: 1000); exposed so "
        "short diagnostics do not silently spend most of training in warmup",
    )
    parser.add_argument(
        "--site-renorm", action="store_true",
        help="compatibility path for shrinkage-only pilot stores; production "
        "stores already fold site-renorm into their transform. Applies after "
        "whitening at batch load (train, calibration, and eval). "
        "Not stored in the checkpoint — pass again on --resume.",
    )
    parser.add_argument(
        "--epochs", type=int, default=EPOCHS,
        help="passes over the training split",
    )
    parser.add_argument(
        "--calib-batches", type=int, default=None,
        help="optional calibration-batch cap; default streams the full split",
    )
    parser.add_argument(
        "--theta-method", choices=("exact", "streaming"), default="streaming",
        help="threshold estimator; production uses bounded-memory streaming",
    )
    parser.add_argument(
        "--guard", action=argparse.BooleanOptionalAction, default=True,
        help="batch-skip loss-spike guard (mandatory for production)",
    )
    parser.add_argument("--guard-factor", type=float, default=20.0)
    parser.add_argument("--guard-loss-factor", type=float, default=5.0)
    parser.add_argument("--guard-window", type=int, default=50)
    parser.add_argument("--guard-max-consecutive", type=int, default=5)
    parser.add_argument("--guard-max-skip-rate", type=float, default=1e-3)
    parser.add_argument(
        "--aux-variant",
        choices=("none", "sasa", "long_horizon", "fel"),
        default="sasa",
        help="dead-latent recovery rule (production default: SASA-style AuxK)",
    )
    parser.add_argument("--s-aux", type=int, default=256)
    parser.add_argument("--alpha-aux", type=float, default=1.0,
                        help="global aux weight (SASA-faithful 1.0)")
    parser.add_argument("--dead-threshold", type=float, default=1e-4)
    parser.add_argument(
        "--dead-window-tokens",
        type=int,
        default=409_600,
        help="accepted-token window for SASA dead-frequency classification",
    )
    parser.add_argument(
        "--dead-horizon-tokens",
        type=int,
        default=2_048_000,
        help="accepted-token horizon for the long-horizon AuxK variant",
    )
    parser.add_argument("--aux-frac-cap", type=float, default=None,
                        help="cap revived blocks/step at ceil(frac x dead-set)")
    parser.add_argument(
        "--aux-ratio-cap",
        type=float,
        default=None,
        help="optional cap on aux grad norm / main grad norm; production SASA "
        "passes 1.0 explicitly, paper-faithful no-Aux/Fel leaves it unset",
    )
    parser.add_argument("--sites", type=int, nargs="*", default=None,
                        help="E4 site-subset view: train on a subset of the "
                        "stored sites (layer numbers, stored order; a single "
                        "site = the factorial's S=1 cells). At matched seed "
                        "the stream is the joint run's, sliced. Default: all.")
    parser.add_argument("--prefetch", type=int, default=4,
                        help="training-stream prefetch depth (0 = off); "
                        "overlaps shard reads with GPU steps, order-preserving")
    parser.add_argument(
        "--train-split", default="train",
        help="store split to train on")
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument(
        "--raw-store",
        type=Path,
        default=None,
        help="aligned normalization=none store used for raw-coordinate FVU",
    )
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="stop early after N steps (resume gate: run with --max-steps, "
        "then again with --resume)",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.lam is None:
        args.lam = (
            1e-3
            if args.arm == "bsc" or args.regularizer not in (None, "none")
            else 0.0
        )
    if args.regularizer == "crosscoder_l1":
        bridge = {
            "arm": args.arm,
            "selection": args.selection,
            "selection_score": args.selection_score,
            "encoder_mode": args.encoder_mode,
            "encoder_bias": args.encoder_bias,
            "code_activation": args.code_activation,
            "decoder_constraint": args.decoder_constraint,
        }
        required = {
            "arm": "scalar",
            "selection": "dense",
            "selection_score": "code_norm",
            "encoder_mode": "untied",
            "encoder_bias": True,
            "code_activation": "relu",
            "decoder_constraint": "free",
        }
        if bridge != required:
            raise SystemExit(
                "crosscoder_l1 is the paper bridge and requires "
                + json.dumps(required, sort_keys=True)
            )

    import torch

    from block_crosscoder_experiment.model import BlockCrosscoder, BSCConfig
    from block_crosscoder_experiment.store import StoreReader, Whitener, prefetch_batches
    from block_crosscoder_experiment.trainer import TrainConfig, Trainer

    whitener = Whitener.load(args.store / "whitener.pt")
    train_reader = StoreReader(
        args.store, args.train_split,
        expected_whitener_hash=whitener.hash, sites=args.sites,
    )
    eval_reader = StoreReader(
        args.store, "eval", expected_whitener_hash=whitener.hash, sites=args.sites
    )
    raw_whitener = None
    raw_eval_reader = None
    if args.raw_store is not None:
        raw_whitener = Whitener.load(args.raw_store / "whitener.pt")
        raw_eval_reader = StoreReader(
            args.raw_store,
            "eval",
            expected_whitener_hash=raw_whitener.hash,
            sites=args.sites,
        )
        _validate_raw_alignment(
            eval_reader, raw_eval_reader, whitener, raw_whitener
        )
    n_sites = train_reader.n_sites
    d_model = train_reader.d_model
    # Whitener-side indices of the (possibly subset) site axis — the
    # renorm scalars must follow the view (E4).
    site_idx = [whitener.sites.index(s) for s in train_reader.sites]
    eval_whitener = Whitener(
        mean=whitener.mean[site_idx],
        W=whitener.W[site_idx],
        ridge=whitener.ridge[site_idx],
        eigenvalues=whitener.eigenvalues[site_idx],
        sites=tuple(train_reader.sites),
        n_fit_tokens=whitener.n_fit_tokens,
        meta=whitener.meta,
    )

    # Pilot-compatible renorm arm: dtype-preserving per-site scalar
    # multiply at batch load — equivalent to a renormed store up to one
    # bf16 rounding. Every data path below flows through renormed().
    renorm_scale = None
    folded_site_renorm = bool(whitener.meta.get("site_rms_renorm_folded"))
    if args.site_renorm:
        if folded_site_renorm:
            raise SystemExit(
                "--site-renorm would double-scale this production whitener; "
                "the transform already has site RMS renormalization folded in"
            )
        renorm_scale = whitener.site_rms_scalars()[site_idx].view(1, -1, 1)
        print(
            "site-renorm scalars (F7 arm): "
            f"{[round(float(v), 3) for v in renorm_scale.flatten()]}",
            flush=True,
        )

    def renormed(it):
        if renorm_scale is None:
            return it
        return (x * renorm_scale.to(x.dtype) for x in it)

    if args.arm == "bsc":
        model_cfg = BSCConfig(
            n_blocks=args.blocks, block_dim=args.block_dim,
            n_sites=n_sites, d_model=d_model,
            k=args.k, lambda_regularizer=args.lam, seed=args.seed,
            selection=args.selection, selection_score=args.selection_score,
            encoder_mode=args.encoder_mode,
            encoder_bias=args.encoder_bias, code_activation=args.code_activation,
            decoder_constraint=args.decoder_constraint, regularizer=args.regularizer,
        )
    else:
        if args.lam != 0.0 and args.regularizer in (None, "site_profile"):
            raise SystemExit(
                "scalar site_profile runs at lambda=0 (at b=1 it is not a "
                "site-profile penalty); select a scalar-compatible regularizer"
            )
        model_cfg = BSCConfig(
            n_blocks=args.blocks * args.block_dim, block_dim=1,
            n_sites=n_sites, d_model=d_model,
            k=args.k * args.block_dim,  # matched L0: E[l] = b*E[k]
            lambda_regularizer=args.lam, seed=args.seed,
            selection=args.selection, selection_score=args.selection_score,
            encoder_mode=args.encoder_mode,
            encoder_bias=args.encoder_bias, code_activation=args.code_activation,
            decoder_constraint=args.decoder_constraint, regularizer=args.regularizer,
        )

    steps_per_epoch = train_reader.n_tokens // BATCH
    total_steps = steps_per_epoch * args.epochs
    tag = ""
    if args.lr != 3e-4:
        tag += f"_lr{args.lr:g}"
    if args.schedule != "cosine":
        tag += f"_{args.schedule}"
    if args.encoder_wd:
        tag += f"_wd{args.encoder_wd:g}"
    if args.blocks != G:
        tag += f"_G{args.blocks}"
    if args.block_dim != B_DIM:
        tag += f"_b{args.block_dim}"
    if args.k != K:
        tag += f"_k{args.k:g}"
    tag += (
        f"_sel-{model_cfg.selection}_score-{model_cfg.selection_score}"
        f"_enc-{model_cfg.encoder_mode}"
        f"_eb{int(model_cfg.encoder_bias)}_act-{model_cfg.code_activation}"
        f"_dc-{model_cfg.decoder_constraint}_reg-{model_cfg.regularizer}"
        f"_aux-{args.aux_variant}_saux{args.s_aux}"
        f"_dt{args.dead_threshold:g}_dwt{args.dead_window_tokens}"
        f"_dht{args.dead_horizon_tokens}"
    )
    if args.warmup_steps != 1000:
        tag += f"_warm{args.warmup_steps}"
    if args.site_renorm or folded_site_renorm:
        tag += "_renorm"
    if args.epochs != EPOCHS:
        tag += f"_ep{args.epochs}"
    if args.calib_batches is not None:
        tag += f"_diagnostic-calib{args.calib_batches}"
    if args.train_split != "train":
        tag += f"_{args.train_split}"
    if args.guard:
        tag += "_guard"
    if args.aux_frac_cap is not None:
        tag += f"_fcap{args.aux_frac_cap:g}"
    if args.aux_ratio_cap is not None:
        tag += f"_rcap{args.aux_ratio_cap:g}"
    if args.alpha_aux != 1.0:
        tag += f"_aaux{args.alpha_aux:g}"
    if args.sites is not None:
        tag += "_site" + "-".join(str(s) for s in train_reader.sites)
    run_name = f"{args.arm}_lam{args.lam:g}_seed{args.seed}{tag}"
    run_dir = args.out_root / run_name
    ckpt = run_dir / "latest.pt"
    print(
        f"config: arm={args.arm} lam={args.lam:g} seed={args.seed} "
        f"G={model_cfg.n_blocks} b={model_cfg.block_dim} k={model_cfg.k} "
        f"sites={n_sites} d={d_model} steps={total_steps} batch={BATCH} "
        f"epochs={args.epochs} store={args.store} whitener={whitener.hash[:12]}… "
        f"run_dir={run_dir}",
        flush=True,
    )

    train_cfg = TrainConfig(
        total_steps=total_steps, lr=args.lr, schedule=args.schedule,
        warmup_steps=args.warmup_steps,
        encoder_weight_decay=args.encoder_wd, log_every=10,
        guard=args.guard, guard_factor=args.guard_factor,
        guard_loss_factor=args.guard_loss_factor,
        guard_window=args.guard_window,
        guard_max_consecutive=args.guard_max_consecutive,
        guard_max_skip_rate=args.guard_max_skip_rate,
        aux_variant=args.aux_variant, s_aux=args.s_aux,
        alpha_aux=args.alpha_aux, aux_frac_cap=args.aux_frac_cap,
        aux_ratio_cap=args.aux_ratio_cap,
        dead_threshold=args.dead_threshold,
        dead_window_tokens=args.dead_window_tokens,
        dead_horizon_tokens=args.dead_horizon_tokens,
    )

    binding = {
        "format_version": 1,
        "store_root": str(args.store.resolve()),
        "train_manifest_sha256": _json_sha256(train_reader.manifest),
        "whitener_hash": whitener.hash,
        "sites": list(train_reader.sites),
        "gauge": {
            "normalization": whitener.mode,
            "site_rms_renorm": bool(args.site_renorm or folded_site_renorm),
            "site_renorm_at_load": bool(args.site_renorm),
            "site_renorm_folded": folded_site_renorm,
        },
        "model_id": whitener.meta.get("model"),
        "train_split": args.train_split,
        "train_tokens": train_reader.n_tokens,
        "shuffle_seed": SHUFFLE_SEED,
        "code_revision": _git_revision(),
        "code_sha256": _code_sha256(),
        "model_cfg": asdict(model_cfg),
        "train_cfg": asdict(train_cfg),
    }
    run_manifest = {
        "format_version": 1,
        "run_name": run_name,
        "binding": binding,
        "postprocess": {
            "theta_method": args.theta_method,
            "calib_batches": args.calib_batches,
            "raw_store": str(args.raw_store.resolve()) if args.raw_store else None,
            "raw_eval_manifest_sha256": (
                _json_sha256(raw_eval_reader.manifest)
                if raw_eval_reader is not None else None
            ),
        },
    }
    _prepare_run_dir(run_dir, run_manifest, resume=args.resume)

    if args.resume:
        if not ckpt.exists():
            raise SystemExit(f"--resume but no checkpoint at {ckpt}")
        trainer = Trainer.load_checkpoint(
            ckpt, device=args.device, expected_binding=binding
        )
        trainer._log_file = (run_dir / "steps.jsonl").open("a")
        print(f"resumed at step {trainer.step_idx}", flush=True)
    else:
        model = BlockCrosscoder(model_cfg, device=args.device)
        # Encoder scale calibration on the first training batch (Fel-inspired).
        first = next(iter(renormed(train_reader.shuffled_batches(BATCH, seed=SHUFFLE_SEED))))
        first_device = first.to(args.device, torch.float32)
        model.calibrate_encoder_scale_(first_device)
        if model.cfg.selection == "threshold":
            model.fit_threshold_(
                [first_device], target_avg_blocks=model.cfg.k, method="exact"
            )
        trainer = Trainer(
            model,
            train_cfg,
            log_path=run_dir / "steps.jsonl",
            run_binding=binding,
        )

    batches = renormed(train_reader.shuffled_batches(
        BATCH, seed=SHUFFLE_SEED, epochs=args.epochs + 1  # +1: margin for skips
    ))
    if trainer.step_idx:
        batches = itertools.islice(batches, trainer.step_idx, None)
        # islice is lazy; the skip cost lands on the first next() below.
        print(f"fast-forwarding stream by {trainer.step_idx} batches", flush=True)
    if args.prefetch:
        batches = prefetch_batches(batches, depth=args.prefetch)

    stop_at = min(total_steps, args.max_steps or total_steps)
    t0 = time.time()
    data_wait = 0.0
    try:
        while trainer.step_idx < stop_at:
            t_io = time.time()
            x = next(batches)
            data_wait += time.time() - t_io
            rec = trainer.step(x)
            if (
                trainer.step_idx % args.checkpoint_every == 0
                or trainer.step_idx == stop_at
            ):
                trainer.save_checkpoint(ckpt)
            if trainer.step_idx % 200 == 0:
                dt = time.time() - t0
                print(
                    f"  step {trainer.step_idx}/{total_steps} rec {rec['rec']:.4f} "
                    f"total {rec['total']:.4f} "
                    f"({trainer.step_idx * BATCH / max(dt, 1e-9):,.0f} tok/s, "
                    f"data-wait {data_wait / max(dt, 1e-9):.0%})",
                    flush=True,
                )
    finally:
        close_batches = getattr(batches, "close", None)
        if close_batches is not None:
            close_batches()
    trainer.save_checkpoint(ckpt)
    train_minutes = (time.time() - t0) / 60
    print(f"training done at step {trainer.step_idx} ({train_minutes:.1f} min)",
          flush=True)
    if trainer.step_idx < total_steps:
        print("stopped at --max-steps; rerun with --resume to finish", flush=True)
        return

    trainer.validate_run_gates()

    model = trainer.master.to(args.device)

    # ---- threshold calibration on the full calibration split -------------
    calib_reader = StoreReader(
        args.store, "calibration", expected_whitener_hash=whitener.hash,
        sites=args.sites,
    )
    n_calib_batches, n_calib_tokens = _calibration_extent(
        calib_reader.n_tokens, BATCH, args.calib_batches
    )
    calibration_complete = n_calib_tokens == calib_reader.n_tokens
    calibration_manifest_sha256 = _json_sha256(calib_reader.manifest)
    theta = model.fit_threshold_(
        renormed(itertools.islice(calib_reader.sequential_batches(BATCH), n_calib_batches)),
        target_avg_blocks=model.cfg.k,
        method=args.theta_method,
    )
    print(f"calibrated theta {theta:.5f} ({args.theta_method}, target avg "
          f"blocks {model.cfg.k}, {n_calib_tokens:,} calib tokens)",
          flush=True)
    # Re-save so the calibrated theta is serialized with the checkpoint
    # (design: theta frozen and serialized with the codec — sol S1; the
    # pre-calibration save above holds theta = NaN).
    trainer.save_checkpoint(ckpt)

    # ---- eval: normalized + optional raw-coordinate FVU -----------------
    raw_linear_inverse = None
    if raw_eval_reader is not None and whitener.mode != "layer":
        raw_linear_inverse = (
            torch.linalg.inv(eval_whitener.W.double()).float().to(args.device),
            eval_whitener.mean.to(args.device),
        )

    @torch.no_grad()
    def eval_pass(mode: str, m=model, dtype=torch.float32) -> dict:
        sq_err = torch.zeros(n_sites, dtype=torch.float64)
        sq_tot = torch.zeros(n_sites, dtype=torch.float64)
        mean_acc = torch.zeros(n_sites, d_model, dtype=torch.float64)
        raw_sq_err = torch.zeros(n_sites, dtype=torch.float64)
        raw_sq_tot = torch.zeros(n_sites, dtype=torch.float64)
        raw_mean_acc = torch.zeros(n_sites, d_model, dtype=torch.float64)
        n = 0
        active = 0.0
        normalized_it = renormed(eval_reader.sequential_batches(BATCH))
        paired_it = (
            ((x, None) for x in normalized_it)
            if raw_eval_reader is None
            else zip(
                normalized_it,
                raw_eval_reader.sequential_batches(BATCH),
                strict=True,
            )
        )
        for x, raw_x in paired_it:
            x = x.to(args.device, dtype)
            out = m(x, mode=mode)
            sq_err += (x - out.xhat).double().pow(2).sum(dim=(0, 2)).cpu()
            mean_acc += x.double().sum(dim=0).cpu()
            sq_tot += x.double().pow(2).sum(dim=(0, 2)).cpu()
            if raw_x is not None:
                if raw_x.shape != x.shape:
                    raise RuntimeError(
                        "aligned raw eval yielded a different batch shape: "
                        f"normalized={tuple(x.shape)} raw={tuple(raw_x.shape)}"
                    )
                raw_x = raw_x.to(args.device, torch.float32)
                xhat_normalized = out.xhat.float()
                if renorm_scale is not None:
                    xhat_normalized = xhat_normalized / renorm_scale.to(
                        args.device, torch.float32
                    )
                raw_xhat = _raw_reconstruct(
                    eval_whitener,
                    xhat_normalized,
                    raw_x,
                    linear_inverse=raw_linear_inverse,
                )
                raw_sq_err += (
                    (raw_x - raw_xhat).double().pow(2).sum(dim=(0, 2)).cpu()
                )
                raw_mean_acc += raw_x.double().sum(dim=0).cpu()
                raw_sq_tot += raw_x.double().pow(2).sum(dim=(0, 2)).cpu()
            active += float(out.mask.sum())
            n += x.shape[0]
        mu = mean_acc / n
        centered_tot = sq_tot - n * mu.pow(2).sum(dim=1)
        fvu = (sq_err / centered_tot).tolist()
        result = {
            "mode": mode,
            "fvu_per_site": [round(v, 6) for v in fvu],
            "fvu_pooled": round(float(sq_err.sum() / centered_tot.sum()), 6),
            "avg_active_blocks": round(active / n, 3),
            "n_tokens": n,
        }
        if raw_eval_reader is None:
            result["raw_fvu_available"] = False
            result["raw_fvu_per_site"] = None
            result["raw_fvu_pooled"] = None
        else:
            raw_mu = raw_mean_acc / n
            raw_centered_tot = raw_sq_tot - n * raw_mu.pow(2).sum(dim=1)
            raw_fvu = raw_sq_err / raw_centered_tot
            result["raw_fvu_available"] = True
            result["raw_fvu_per_site"] = [
                round(float(value), 6) for value in raw_fvu
            ]
            result["raw_fvu_pooled"] = round(
                float(raw_sq_err.sum() / raw_centered_tot.sum()), 6
            )
        return result

    results = {}
    for mode in ("topk", "threshold"):
        first = eval_pass(mode)
        second = eval_pass(mode)
        deterministic = first == second
        results[mode] = first
        results[mode]["deterministic"] = deterministic
        print(f"  eval[{mode}]: pooled FVU {first['fvu_pooled']} "
              f"per-site {first['fvu_per_site']} "
              f"raw FVU {first['raw_fvu_pooled']} "
              f"avg blocks {first['avg_active_blocks']} "
              f"deterministic {deterministic}", flush=True)

    # bf16 shadow eval (sol S2): the passes above run the fp32 master — the
    # declared codec primary; this single diagnostic pass per mode runs the
    # training/deployment precision (theta coarsens to bf16 with the buffer,
    # which is part of what the shadow measures).
    shadow = copy.deepcopy(model).to(torch.bfloat16)
    for mode in ("topk", "threshold"):
        s = eval_pass(mode, m=shadow, dtype=torch.bfloat16)
        results[mode]["bf16_shadow"] = {
            k: s[k]
            for k in (
                "fvu_per_site",
                "fvu_pooled",
                "raw_fvu_available",
                "raw_fvu_per_site",
                "raw_fvu_pooled",
                "avg_active_blocks",
            )
        }
        print(f"  eval[{mode}] bf16 shadow: pooled FVU {s['fvu_pooled']} "
              f"avg blocks {s['avg_active_blocks']}", flush=True)

    dead_frac = float(
        (trainer.tracker.frequency(trainer.cfg.dead_window_tokens)
         <= trainer.cfg.dead_threshold).float().mean()
    )
    loaded_model_cfg = asdict(trainer.master.cfg)
    loaded_train_cfg = asdict(trainer.cfg)
    report = {
        "arm": args.arm,
        "lam": args.lam,
        "seed": args.seed,
        "model_cfg": loaded_model_cfg,
        "train_cfg": loaded_train_cfg,
        "run_binding": trainer.run_binding,
        "sites": list(train_reader.sites),
        "raw_store": str(args.raw_store) if args.raw_store else None,
        "raw_fvu_available": raw_eval_reader is not None,
        "prefetch": args.prefetch,
        "total_steps": trainer.cfg.total_steps,
        "epochs": args.epochs,
        "calib_batches": n_calib_batches,
        "calib_tokens": n_calib_tokens,
        "calibration_complete": calibration_complete,
        "calibration_manifest_sha256": calibration_manifest_sha256,
        "promotion_eligible": bool(calibration_complete and raw_eval_reader is not None),
        "lr": trainer.cfg.lr,
        "warmup_steps": trainer.cfg.warmup_steps,
        "schedule": trainer.cfg.schedule,
        "encoder_wd": trainer.cfg.encoder_weight_decay,
        "site_renorm": bool(args.site_renorm or folded_site_renorm),
        "normalization": whitener.mode,
        "site_renorm_at_load": args.site_renorm,
        "site_renorm_folded": folded_site_renorm,
        "site_renorm_scalars": (
            [round(float(v), 6) for v in renorm_scale.flatten()]
            if renorm_scale is not None else None
        ) if not folded_site_renorm else whitener.meta.get("site_rms_scalars"),
        "shuffle_seed": SHUFFLE_SEED,
        "whitener_hash": whitener.hash,
        "theta": theta,
        "theta_method": args.theta_method,
        # The skip rate is a run gate; guard events carry the postmortem.
        "guard": args.guard,
        "skipped_steps": trainer.skipped_steps,
        "skip_rate": round(trainer.skipped_steps / max(1, trainer.step_idx), 6),
        "guard_events": trainer.guard_events,
        "aux_variant": trainer.cfg.aux_variant,
        "s_aux": trainer.cfg.s_aux,
        "alpha_aux": trainer.cfg.alpha_aux,
        "dead_threshold": trainer.cfg.dead_threshold,
        "dead_window_tokens": trainer.cfg.dead_window_tokens,
        "dead_horizon_tokens": trainer.cfg.dead_horizon_tokens,
        "aux_frac_cap": trainer.cfg.aux_frac_cap,
        "aux_ratio_cap": trainer.cfg.aux_ratio_cap,
        "eval": results,
        "dead_frac_final_window": dead_frac,
        "train_minutes": round(train_minutes, 2),
        "optimizer": trainer.optimizer_kind,
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"-> {run_dir / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
