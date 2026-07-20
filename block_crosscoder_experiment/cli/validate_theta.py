"""Validate full-split streaming threshold fitting against exact quantiles.

Gates, per checkpoint:
  1. estimator agreement — exact and streaming fit_threshold_ on the SAME
     calibration batches must land within |Δ avg-blocks| <= 0.1 realized
     on the eval split;
  2. bounded memory — streaming over the FULL calibration split must run
     inside host RAM (run FIRST so peak RSS is attributable to it; the
     scalar 16k-latent checkpoint is the arm that OOM'd 61 GB at 64
     batches under exact).

The full-split streaming theta also carries information the capped exact
theta cannot: it is the whole-split quantile, reported alongside.
"""

from __future__ import annotations

import argparse
import itertools
import json
import resource
import time
from pathlib import Path

import torch

BATCH = 4096


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # Linux: KB


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--site-renorm", action="store_true",
                    help="renorm arm: apply site-RMS scalars at batch load, "
                    "matching how the checkpoint was trained and calibrated")
    ap.add_argument("--exact-batches", type=int, default=32,
                    help="calib batches for the paired exact/streaming gate "
                    "(host-RAM-safe: 32 at G=4096 is ~8.6 GB x2 copies; use "
                    "16 for the 16k-latent scalar arm)")
    ap.add_argument("--eval-batches", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    from block_crosscoder_experiment.store import StoreReader, Whitener
    from block_crosscoder_experiment.trainer import Trainer

    whitener = Whitener.load(args.store / "whitener.pt")
    scale = (
        whitener.site_rms_scalars().view(1, -1, 1) if args.site_renorm else None
    )

    def renormed(it):
        if scale is None:
            return it
        return (x * scale.to(x.dtype) for x in it)

    trainer = Trainer.load_checkpoint(args.ckpt, device=args.device)
    model = trainer.master.to(args.device)
    target = float(model.cfg.k)
    calib = StoreReader(args.store, "calibration", expected_whitener_hash=whitener.hash)
    eval_r = StoreReader(args.store, "eval", expected_whitener_hash=whitener.hash)
    n_full = calib.n_tokens // BATCH

    def calib_batches(n: int | None):
        it = calib.sequential_batches(BATCH)
        if n is not None:
            it = itertools.islice(it, n)
        return renormed(it)

    @torch.no_grad()
    def avg_blocks() -> float:
        tot, cnt = 0.0, 0
        for x in itertools.islice(
            renormed(eval_r.sequential_batches(BATCH)), args.eval_batches
        ):
            out = model(x.to(args.device, torch.float32), mode="threshold")
            tot += float(out.mask.sum())
            cnt += x.shape[0]
        return tot / cnt

    def timed_fit(n: int | None, method: str) -> dict:
        t0 = time.time()
        theta = model.fit_threshold_(calib_batches(n), target, method=method)
        return {
            "theta": theta,
            "method": method,
            "batches": n if n is not None else n_full,
            "seconds": round(time.time() - t0, 1),
            "rss_gb_after": round(rss_gb(), 2),
            "avg_blocks_eval": round(avg_blocks(), 4),
        }

    res: dict = {
        "ckpt": str(args.ckpt),
        "n_latents": model.cfg.n_latents,
        "n_blocks": model.cfg.n_blocks,
        "site_renorm": args.site_renorm,
        "target_avg_blocks": target,
        "calib_batches_full": n_full,
        "rss_gb_baseline": round(rss_gb(), 2),
    }
    # Order matters: full-split streaming FIRST, so its peak RSS is not
    # contaminated by the exact pass's score matrix.
    res["streaming_full"] = timed_fit(None, "streaming")
    res["exact_capped"] = timed_fit(args.exact_batches, "exact")
    res["streaming_capped"] = timed_fit(args.exact_batches, "streaming")

    res["gate_delta_avg_blocks"] = round(
        abs(
            res["exact_capped"]["avg_blocks_eval"]
            - res["streaming_capped"]["avg_blocks_eval"]
        ),
        4,
    )
    res["gate_estimator_agreement"] = res["gate_delta_avg_blocks"] <= 0.1
    res["gate_bounded_memory"] = res["streaming_full"]["rss_gb_after"] < 30.0
    print(json.dumps(res, indent=2), flush=True)
    if args.out is not None:
        args.out.write_text(json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    main()
