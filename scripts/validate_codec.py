"""Tranche 3 (runbook-phase099): run the preregistered R-D codec on a
trained checkpoint — calibration-split fit, eval-split scoring.

Codec validation per the runbook: exercised end-to-end on existing pilot
checkpoints before the frontier trainings need it. The active-count
floor is applied and exclusions reported (pilot calib is 2M tokens vs
production 13M — the floor bites harder here; worn openly). Standing
caution: pooled FVU at matched activation count is NOT a bits-distortion
verdict; this codec is where the support-bit amortization the block bet
is priced on becomes measurable.

  python -u scripts/validate_codec.py \
      --ckpt /data/runs/bcc-pilot4b/bsc_lam0.001_seed0_G4096_k32/latest.pt \
      --store /data/stores/bcc-pilot4b/gemma3_4b_8site_fineweb \
      --out /data/runs/bcc-phase099/rd_bsc_primary.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

BATCH = 4096


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--site-renorm", action="store_true",
                    help="renorm arm: apply site-RMS scalars at batch load")
    ap.add_argument("--sites", type=int, nargs="*", default=None,
                    help="site-subset view (single-site cells)")
    ap.add_argument("--qs", type=int, nargs="*", default=[4, 6, 8])
    ap.add_argument("--floor", type=int, default=1000,
                    help="active-count floor (calib events) for codec inclusion")
    ap.add_argument("--calib-batches", type=int, default=None,
                    help="default: the full calibration split")
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import torch

    from block_crosscoder_experiment.codec import CodecSpec, evaluate_rd, fit_codec
    from block_crosscoder_experiment.store import StoreReader, Whitener
    from block_crosscoder_experiment.trainer import Trainer

    whitener = Whitener.load(args.store / "whitener.pt")
    trainer = Trainer.load_checkpoint(args.ckpt, device=args.device)
    model = trainer.master.to(args.device)
    if torch.isnan(model.theta):
        raise SystemExit(f"{args.ckpt} has no calibrated theta — the codec "
                         "prices the deployed threshold path")

    calib = StoreReader(args.store, "calibration",
                        expected_whitener_hash=whitener.hash, sites=args.sites)
    eval_r = StoreReader(args.store, "eval",
                         expected_whitener_hash=whitener.hash, sites=args.sites)
    site_idx = [whitener.sites.index(s) for s in calib.sites]
    scale = (
        whitener.site_rms_scalars()[site_idx].view(1, -1, 1)
        if args.site_renorm else None
    )

    def renormed(it):
        if scale is None:
            return it
        return (x * scale.to(x.dtype) for x in it)

    meta = calib.manifest.get("meta", {})
    row_len = int(meta.get("context_size", 1024)) - int(meta.get("dropped_positions", 2))

    spec = CodecSpec(qs=tuple(args.qs), floor=args.floor,
                     n_bootstrap=args.n_bootstrap)
    calib_it = calib.sequential_batches(BATCH)
    if args.calib_batches is not None:
        calib_it = itertools.islice(calib_it, args.calib_batches)

    t0 = time.time()
    codec = fit_codec(model, renormed(calib_it), spec, device=args.device)
    print(f"codec fit: {codec.calib_tokens:,} calib tokens, "
          f"{codec.n_included}/{model.cfg.n_blocks} blocks included "
          f"(excluded calib-event share "
          f"{codec.meta['excluded_calib_event_share']:.2e}), "
          f"{time.time() - t0:.0f}s", flush=True)

    t0 = time.time()
    res = evaluate_rd(model, codec, renormed(eval_r.sequential_batches(BATCH)),
                      row_len=row_len, device=args.device)
    print(f"eval: {res['n_tokens']:,} tokens in {res['n_rows']} sequences, "
          f"avg count {res['avg_count']:.2f}, "
          f"excluded eval-event share {res['eval_excluded_event_share']:.2e}, "
          f"{time.time() - t0:.0f}s", flush=True)
    for qk, p in res["points"].items():
        print(f"  q={qk}: FVU {p['fvu_pooled']:.4f} "
              f"[{p['fvu_ci95'][0]:.4f}, {p['fvu_ci95'][1]:.4f}]  "
              f"rate {p['rate_bits_per_token']:.1f} bits/tok "
              f"(support {res['support_bits_per_token']:.1f} + "
              f"amp {p['amplitude_bits_per_token']:.1f}; "
              f"bernoulli-rate {p['rate_bits_bernoulli']:.1f})", flush=True)

    payload = {
        "ckpt": str(args.ckpt),
        "model_cfg": {
            "n_blocks": model.cfg.n_blocks, "block_dim": model.cfg.block_dim,
            "n_sites": model.cfg.n_sites, "k": model.cfg.k,
        },
        "theta": float(model.theta),
        "site_renorm": args.site_renorm,
        "sites": list(calib.sites),
        "whitener_hash": whitener.hash,
        "spec": {"qs": list(spec.qs), "clip": [spec.clip_lo, spec.clip_hi],
                 "floor": spec.floor, "n_bootstrap": spec.n_bootstrap},
        "results": res,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
