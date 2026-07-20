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
import itertools
import json
import time
from pathlib import Path

G, B_DIM, K = 4096, 4, 32.0
BATCH = 4096
EPOCHS = 2
SHUFFLE_SEED = 0  # recorded; shared verbatim by BSC and baseline (design)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("bsc", "scalar"), required=True)
    parser.add_argument(
        "--lam", type=float, default=None,
        help="rank penalty (default: 1e-3 for BSC, 0 for scalar)",
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
    parser.add_argument("--k", type=float, default=K)
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
    parser.add_argument("--alpha-aux", type=float, default=1.0,
                        help="global aux weight (SASA-faithful 1.0)")
    parser.add_argument("--aux-frac-cap", type=float, default=None,
                        help="cap revived blocks/step at ceil(frac x dead-set)")
    parser.add_argument("--aux-ratio-cap", type=float, default=1.0,
                        help="cap aux grad norm at ratio x main grad norm")
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
        args.lam = 1e-3 if args.arm == "bsc" else 0.0

    import torch

    from block_crosscoder_experiment.model import BlockCrosscoder, BSCConfig
    from block_crosscoder_experiment.store import StoreReader, Whitener, prefetch_batches
    from block_crosscoder_experiment.trainer import TrainConfig, Trainer

    whitener = Whitener.load(args.store / "whitener.pt")
    train_reader = StoreReader(
        args.store, args.train_split,
        expected_whitener_hash=whitener.hash, sites=args.sites,
    )
    n_sites = train_reader.n_sites
    d_model = train_reader.d_model
    # Whitener-side indices of the (possibly subset) site axis — the
    # renorm scalars must follow the view (E4).
    site_idx = [whitener.sites.index(s) for s in train_reader.sites]

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
            n_blocks=args.blocks, block_dim=B_DIM, n_sites=n_sites, d_model=d_model,
            k=args.k, lambda_rank=args.lam, seed=args.seed,
        )
    else:
        if args.lam != 0.0:
            raise SystemExit("scalar baseline runs at lambda=0 (design: at b=1 "
                             "the nuclear term is not a rank penalty)")
        model_cfg = BSCConfig(
            n_blocks=args.blocks * B_DIM, block_dim=1, n_sites=n_sites, d_model=d_model,
            k=args.k * B_DIM,  # matched training-average L0: E[l] = b*E[k]
            lambda_rank=0.0, seed=args.seed,
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
    if args.k != K:
        tag += f"_k{args.k:g}"
    if args.site_renorm or folded_site_renorm:
        tag += "_renorm"
    if args.epochs != EPOCHS:
        tag += f"_ep{args.epochs}"
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
    run_dir.mkdir(parents=True, exist_ok=True)
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
        encoder_weight_decay=args.encoder_wd, log_every=10,
        guard=args.guard, guard_factor=args.guard_factor,
        guard_loss_factor=args.guard_loss_factor,
        guard_window=args.guard_window,
        guard_max_consecutive=args.guard_max_consecutive,
        alpha_aux=args.alpha_aux, aux_frac_cap=args.aux_frac_cap,
        aux_ratio_cap=args.aux_ratio_cap,
    )

    if args.resume:
        if not ckpt.exists():
            raise SystemExit(f"--resume but no checkpoint at {ckpt}")
        trainer = Trainer.load_checkpoint(ckpt, device=args.device)
        trainer._log_file = (run_dir / "steps.jsonl").open("a")
        print(f"resumed at step {trainer.step_idx}", flush=True)
    else:
        model = BlockCrosscoder(model_cfg, device=args.device)
        # Encoder scale calibration on the first training batch (Fel-inspired).
        first = next(iter(renormed(train_reader.shuffled_batches(BATCH, seed=SHUFFLE_SEED))))
        model.calibrate_encoder_scale_(first.to(args.device, torch.float32))
        trainer = Trainer(model, train_cfg, log_path=run_dir / "steps.jsonl")

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
    while trainer.step_idx < stop_at:
        t_io = time.time()
        x = next(batches)
        data_wait += time.time() - t_io
        rec = trainer.step(x)
        if trainer.step_idx % args.checkpoint_every == 0 or trainer.step_idx == stop_at:
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
    trainer.save_checkpoint(ckpt)
    train_minutes = (time.time() - t0) / 60
    print(f"training done at step {trainer.step_idx} ({train_minutes:.1f} min)",
          flush=True)
    if trainer.step_idx < total_steps:
        print("stopped at --max-steps; rerun with --resume to finish", flush=True)
        return

    model = trainer.master.to(args.device)

    # ---- threshold calibration on the full calibration split -------------
    calib_reader = StoreReader(
        args.store, "calibration", expected_whitener_hash=whitener.hash,
        sites=args.sites,
    )
    available_calib_batches = calib_reader.n_tokens // BATCH
    n_calib_batches = (
        available_calib_batches
        if args.calib_batches is None
        else min(args.calib_batches, available_calib_batches)
    )
    theta = model.fit_threshold_(
        renormed(itertools.islice(calib_reader.sequential_batches(BATCH), n_calib_batches)),
        target_avg_blocks=model.cfg.k,
        method=args.theta_method,
    )
    print(f"calibrated theta {theta:.5f} ({args.theta_method}, target avg "
          f"blocks {model.cfg.k}, {n_calib_batches * BATCH:,} calib tokens)",
          flush=True)
    # Re-save so the calibrated theta is serialized with the checkpoint
    # (design: theta frozen and serialized with the codec — sol S1; the
    # pre-calibration save above holds theta = NaN).
    trainer.save_checkpoint(ckpt)

    # ---- eval: per-site whitened FVU, run twice (determinism gate) -------
    eval_reader = StoreReader(
        args.store, "eval", expected_whitener_hash=whitener.hash, sites=args.sites
    )

    @torch.no_grad()
    def eval_pass(mode: str, m=model, dtype=torch.float32) -> dict:
        sq_err = torch.zeros(n_sites, dtype=torch.float64)
        sq_tot = torch.zeros(n_sites, dtype=torch.float64)
        mean_acc = torch.zeros(n_sites, d_model, dtype=torch.float64)
        n = 0
        active = 0.0
        for x in renormed(eval_reader.sequential_batches(BATCH)):
            x = x.to(args.device, dtype)
            out = m(x, mode=mode)
            sq_err += (x - out.xhat).double().pow(2).sum(dim=(0, 2)).cpu()
            mean_acc += x.double().sum(dim=0).cpu()
            sq_tot += x.double().pow(2).sum(dim=(0, 2)).cpu()
            active += float(out.mask.sum())
            n += x.shape[0]
        mu = mean_acc / n
        centered_tot = sq_tot - n * mu.pow(2).sum(dim=1)
        fvu = (sq_err / centered_tot).tolist()
        return {
            "mode": mode,
            "fvu_per_site": [round(v, 6) for v in fvu],
            "fvu_pooled": round(float(sq_err.sum() / centered_tot.sum()), 6),
            "avg_active_blocks": round(active / n, 3),
            "n_tokens": n,
        }

    results = {}
    for mode in ("topk", "threshold"):
        first = eval_pass(mode)
        second = eval_pass(mode)
        deterministic = first == second
        results[mode] = first
        results[mode]["deterministic"] = deterministic
        print(f"  eval[{mode}]: pooled FVU {first['fvu_pooled']} "
              f"per-site {first['fvu_per_site']} "
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
            k: s[k] for k in ("fvu_per_site", "fvu_pooled", "avg_active_blocks")
        }
        print(f"  eval[{mode}] bf16 shadow: pooled FVU {s['fvu_pooled']} "
              f"avg blocks {s['avg_active_blocks']}", flush=True)

    dead_frac = float(
        (trainer.tracker.frequency(train_cfg.dead_window_batches)
         <= train_cfg.dead_threshold).float().mean()
    )
    report = {
        "arm": args.arm,
        "lam": args.lam,
        "seed": args.seed,
        "model_cfg": {
            "n_blocks": model_cfg.n_blocks, "block_dim": model_cfg.block_dim,
            "k": model_cfg.k, "n_sites": n_sites, "d_model": d_model,
        },
        "sites": list(train_reader.sites),
        "prefetch": args.prefetch,
        "total_steps": total_steps,
        "epochs": args.epochs,
        "calib_batches": n_calib_batches,
        "lr": args.lr,
        "schedule": args.schedule,
        "encoder_wd": args.encoder_wd,
        "site_renorm": bool(args.site_renorm or folded_site_renorm),
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
        "alpha_aux": args.alpha_aux,
        "aux_frac_cap": args.aux_frac_cap,
        "aux_ratio_cap": args.aux_ratio_cap,
        "eval": results,
        "dead_frac_final_window": dead_frac,
        "train_minutes": round(train_minutes, 2),
        "optimizer": trainer.optimizer_kind,
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"-> {run_dir / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
