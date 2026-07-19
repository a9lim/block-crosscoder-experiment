"""Tranche 2 (runbook-phase099): the factorial's single-site cells.

Trains ONE single-site model per stored site — all sites in a single
store pass (the batch iterator is shared; each model steps on its own
site's slice). With the joint arms this completes the 2x2:

  {block, cross-site}   = BSC (exists: pilot bsc arm)
  {scalar, cross-site}  = ordinary BatchTopK crosscoder (exists: scalar arm)
  {block, single-site}  = BSF cell        (--arm bsf, this script)
  {scalar, single-site} = per-site SAE    (--arm sae, this script)

S=1 deletes exactly one degree of freedom — code tying across sites. The
Gram constraint sum_s D_g^s D_g^s^T = I_b collapses to per-block
orthonormal decoder rows (the BSF parent's constraint); at b=1 it is
unit-norm rows (the standard SAE convention). Matching is exact and
config-only: per site, same G, b, k (selection rate per site per token),
lambda, lr/schedule, and — because the stream is the joint run's shuffle
seed, sliced (E4 guarantee) — the same tokens in the same order. Total
parameters across the S per-site models equal one joint run.

Site-renorm is intentionally absent: a per-site scalar on a single-site
model is a global input scale, absorbed by encoder-scale calibration and
theta; FVU is invariant. The bf16 shadow eval is also omitted — factorial
cells are compared on the fp32 primary.

The pooled FVU across per-site models (sum sq_err / sum centered_tot) is
directly comparable to the joint arms' pooled FVU at matched parameters
and rate; per-site FVU compares site-by-site.

  python -u scripts/run_phase099_single_site.py --arm bsf \
      --store /data/stores/bcc-pilot4b/gemma3_4b_8site_fineweb \
      --out-root /data/runs/bcc-phase099 --blocks 4096 --k 32 --guard
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

BATCH = 4096
SHUFFLE_SEED = 0  # the joint runs' seed, verbatim (design: shared stream)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("bsf", "sae"), required=True)
    parser.add_argument("--blocks", type=int, default=4096,
                        help="G of the JOINT bsc arm being matched")
    parser.add_argument("--k", type=float, default=32.0,
                        help="k of the JOINT bsc arm being matched")
    parser.add_argument("--lam", type=float, default=None,
                        help="default: 1e-3 for bsf (ratified joint value), "
                        "0 for sae (b=1: not a rank penalty)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--schedule", default="cosine")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--sites", type=int, nargs="*", default=None,
                        help="subset of stored sites (default: all)")
    parser.add_argument("--guard", action="store_true")
    parser.add_argument("--guard-factor", type=float, default=20.0)
    parser.add_argument("--guard-loss-factor", type=float, default=5.0)
    parser.add_argument("--guard-window", type=int, default=50)
    parser.add_argument("--guard-max-consecutive", type=int, default=5)
    parser.add_argument("--theta-method", default="streaming",
                        choices=("exact", "streaming"))
    parser.add_argument("--calib-batches", type=int, default=128)
    parser.add_argument("--prefetch", type=int, default=4)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--train-split", default="train")
    args = parser.parse_args()

    import torch

    from block_crosscoder_experiment.model import BlockCrosscoder, BSCConfig
    from block_crosscoder_experiment.store import StoreReader, Whitener, prefetch_batches
    from block_crosscoder_experiment.trainer import TrainConfig, Trainer

    if args.lam is None:
        args.lam = 1e-3 if args.arm == "bsf" else 0.0
    if args.arm == "sae" and args.lam != 0.0:
        raise SystemExit("sae cell runs at lambda=0 (b=1: nuclear term is "
                         "not a rank penalty — same rule as the scalar arm)")

    whitener = Whitener.load(args.store / "whitener.pt")
    train_reader = StoreReader(
        args.store, args.train_split, expected_whitener_hash=whitener.hash,
        sites=args.sites,
    )
    sites = list(train_reader.sites)
    d_model = train_reader.d_model

    if args.arm == "bsf":
        cell_G, cell_b, cell_k = args.blocks, 4, args.k
    else:
        cell_G, cell_b, cell_k = args.blocks * 4, 1, args.k * 4
    steps_per_epoch = train_reader.n_tokens // BATCH
    total_steps = steps_per_epoch * args.epochs

    tag = ""
    if args.lr != 3e-4:
        tag += f"_lr{args.lr:g}"
    if args.epochs != 2:
        tag += f"_ep{args.epochs}"
    if args.train_split != "train":
        tag += f"_{args.train_split}"
    if args.sites is not None:
        tag += "_site" + "-".join(str(s) for s in sites)
    run_name = f"{args.arm}_lam{args.lam:g}_seed{args.seed}{tag}"
    run_dir = args.out_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"config: arm={args.arm} cells={len(sites)} sites={sites} "
        f"G={cell_G} b={cell_b} k={cell_k:g} lam={args.lam:g} lr={args.lr:g} "
        f"steps={total_steps} batch={BATCH} epochs={args.epochs} "
        f"whitener={whitener.hash[:12]}… run_dir={run_dir}",
        flush=True,
    )

    train_cfg = TrainConfig(
        total_steps=total_steps, lr=args.lr, schedule=args.schedule,
        log_every=10, guard=args.guard, guard_factor=args.guard_factor,
        guard_loss_factor=args.guard_loss_factor,
        guard_window=args.guard_window,
        guard_max_consecutive=args.guard_max_consecutive,
    )

    site_dirs = {L: run_dir / f"site{L}" for L in sites}
    trainers: dict[int, Trainer] = {}
    if args.resume:
        for L in sites:
            ckpt = site_dirs[L] / "latest.pt"
            if not ckpt.exists():
                raise SystemExit(f"--resume but no checkpoint at {ckpt}")
            trainers[L] = Trainer.load_checkpoint(ckpt, device=args.device)
            trainers[L]._log_file = (site_dirs[L] / "steps.jsonl").open("a")
        steps = {L: t.step_idx for L, t in trainers.items()}
        if len(set(steps.values())) != 1:
            raise SystemExit(f"per-site checkpoints out of lockstep: {steps}")
        print(f"resumed all {len(sites)} cells at step {trainers[sites[0]].step_idx}",
              flush=True)
    else:
        first = next(iter(train_reader.shuffled_batches(BATCH, seed=SHUFFLE_SEED)))
        for i, L in enumerate(sites):
            site_dirs[L].mkdir(exist_ok=True)
            # Per-site seed offset: independent inits across cells, same
            # convention as seed sweeps (the joint arm's seed is the base).
            cfg = BSCConfig(
                n_blocks=cell_G, block_dim=cell_b, n_sites=1, d_model=d_model,
                k=cell_k, lambda_rank=args.lam, seed=args.seed * 1000 + L,
            )
            model = BlockCrosscoder(cfg, device=args.device)
            model.calibrate_encoder_scale_(
                first[:, i : i + 1].to(args.device, torch.float32)
            )
            trainers[L] = Trainer(
                model, train_cfg, log_path=site_dirs[L] / "steps.jsonl"
            )

    batches = train_reader.shuffled_batches(
        BATCH, seed=SHUFFLE_SEED, epochs=args.epochs + 1  # +1: skip margin
    )
    step0 = trainers[sites[0]].step_idx
    if step0:
        batches = itertools.islice(batches, step0, None)
        print(f"fast-forwarding stream by {step0} batches", flush=True)
    if args.prefetch:
        batches = prefetch_batches(batches, depth=args.prefetch)

    stop_at = min(total_steps, args.max_steps or total_steps)
    t0 = time.time()
    data_wait = 0.0
    while trainers[sites[0]].step_idx < stop_at:
        t_io = time.time()
        x = next(batches)
        data_wait += time.time() - t_io
        recs = {}
        for i, L in enumerate(sites):
            recs[L] = trainers[L].step(x[:, i : i + 1])
        n = trainers[sites[0]].step_idx
        if n % args.checkpoint_every == 0 or n == stop_at:
            for L in sites:
                trainers[L].save_checkpoint(site_dirs[L] / "latest.pt")
        if n % 200 == 0:
            dt = time.time() - t0
            worst = max(recs.values(), key=lambda r: r["rec"])
            print(
                f"  step {n}/{total_steps} rec(mean) "
                f"{sum(r['rec'] for r in recs.values()) / len(recs):.4f} "
                f"rec(max) {worst['rec']:.4f} "
                f"({n * BATCH / max(dt, 1e-9):,.0f} tok/s, "
                f"data-wait {data_wait / max(dt, 1e-9):.0%})",
                flush=True,
            )
    for L in sites:
        trainers[L].save_checkpoint(site_dirs[L] / "latest.pt")
    train_minutes = (time.time() - t0) / 60
    print(f"training done at step {trainers[sites[0]].step_idx} "
          f"({train_minutes:.1f} min)", flush=True)
    if trainers[sites[0]].step_idx < total_steps:
        print("stopped at --max-steps; rerun with --resume to finish", flush=True)
        return

    # ---- theta calibration, per cell on its site's calib slice -----------
    thetas = {}
    for L in sites:
        calib = StoreReader(
            args.store, "calibration", expected_whitener_hash=whitener.hash,
            sites=[L],
        )
        n_calib = min(args.calib_batches, calib.n_tokens // BATCH)
        model = trainers[L].master.to(args.device)
        thetas[L] = model.fit_threshold_(
            itertools.islice(calib.sequential_batches(BATCH), n_calib),
            target_avg_blocks=model.cfg.k,
            method=args.theta_method,
        )
        trainers[L].save_checkpoint(site_dirs[L] / "latest.pt")
        print(f"  theta[site {L}] {thetas[L]:.5f} ({args.theta_method}, "
              f"{n_calib * BATCH:,} tokens)", flush=True)

    # ---- eval: one pass over the eval split feeds every cell -------------
    eval_reader = StoreReader(
        args.store, "eval", expected_whitener_hash=whitener.hash, sites=args.sites
    )

    @torch.no_grad()
    def eval_pass(mode: str) -> dict:
        sq_err = {L: 0.0 for L in sites}
        sq_tot = {L: 0.0 for L in sites}
        mean_acc = {
            L: torch.zeros(d_model, dtype=torch.float64) for L in sites
        }
        active = {L: 0.0 for L in sites}
        n = 0
        for x in eval_reader.sequential_batches(BATCH):
            x = x.to(args.device, torch.float32)
            for i, L in enumerate(sites):
                xs = x[:, i : i + 1]
                out = trainers[L].master(xs, mode=mode)
                sq_err[L] += float((xs - out.xhat).double().pow(2).sum())
                sq_tot[L] += float(xs.double().pow(2).sum())
                mean_acc[L] += xs.double().sum(dim=(0, 1)).cpu()
                active[L] += float(out.mask.sum())
            n += x.shape[0]
        per_site, pooled_err, pooled_tot = {}, 0.0, 0.0
        for L in sites:
            mu = mean_acc[L] / n
            centered = sq_tot[L] - n * float(mu.pow(2).sum())
            per_site[L] = {
                "fvu": round(sq_err[L] / centered, 6),
                "avg_active_blocks": round(active[L] / n, 3),
            }
            pooled_err += sq_err[L]
            pooled_tot += centered
        return {
            "mode": mode,
            "fvu_per_site": [per_site[L]["fvu"] for L in sites],
            "fvu_pooled": round(pooled_err / pooled_tot, 6),
            "avg_active_blocks_per_site": [
                per_site[L]["avg_active_blocks"] for L in sites
            ],
            "n_tokens": n,
        }

    results = {}
    for mode in ("topk", "threshold"):
        first_pass = eval_pass(mode)
        deterministic = eval_pass(mode) == first_pass
        results[mode] = first_pass | {"deterministic": deterministic}
        print(f"  eval[{mode}]: pooled FVU {first_pass['fvu_pooled']} "
              f"per-site {first_pass['fvu_per_site']} "
              f"deterministic {deterministic}", flush=True)

    report = {
        "arm": args.arm,
        "factorial_cell": (
            "block,single-site (BSF)" if args.arm == "bsf"
            else "scalar,single-site (per-site SAE)"
        ),
        "matched_joint": {"G": args.blocks, "b": 4, "k": args.k},
        "cell_cfg": {"n_blocks": cell_G, "block_dim": cell_b, "k": cell_k,
                     "n_sites": 1, "d_model": d_model},
        "lam": args.lam,
        "seed": args.seed,
        "sites": sites,
        "lr": args.lr,
        "schedule": args.schedule,
        "epochs": args.epochs,
        "train_split": args.train_split,
        "total_steps": total_steps,
        "shuffle_seed": SHUFFLE_SEED,
        "whitener_hash": whitener.hash,
        "theta_method": args.theta_method,
        "thetas": {str(L): thetas[L] for L in sites},
        "guard": args.guard,
        "skipped_steps": {str(L): trainers[L].skipped_steps for L in sites},
        "guard_events": {str(L): trainers[L].guard_events for L in sites},
        "dead_frac_per_site": {
            str(L): float(
                (trainers[L].tracker.frequency(train_cfg.dead_window_batches)
                 <= train_cfg.dead_threshold).float().mean()
            )
            for L in sites
        },
        "train_minutes": round(train_minutes, 1),
        "results": results,
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"-> {run_dir / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
