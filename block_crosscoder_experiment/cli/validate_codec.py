"""Run the frozen R-D codec on a trained checkpoint.

Fit orientation, clipping ranges, count model, and quantizer on calibration;
freeze them; then score the eval split. Active-count exclusions and their
usage shares remain explicit. Pooled FVU at matched activation count is not
a bits–distortion verdict; this codec prices support amortization directly.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path

BATCH = 4096


def _json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


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
    ap.add_argument(
        "--codec-out", type=Path, default=None,
        help="serialized codec path (default: <out stem>.codec.pt)",
    )
    args = ap.parse_args()

    import torch

    from block_crosscoder_experiment.codec import Codec, CodecSpec, evaluate_rd, fit_codec
    from block_crosscoder_experiment.store import StoreReader, Whitener
    from block_crosscoder_experiment.trainer import Trainer, validate_run_binding

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
    folded_site_renorm = bool(whitener.meta.get("site_rms_renorm_folded"))
    if folded_site_renorm and args.site_renorm:
        raise SystemExit(
            "--site-renorm would double-identify a folded-renorm checkpoint; "
            "omit the flag for production folded stores"
        )
    if trainer.run_binding is None:
        raise SystemExit("codec evaluation refuses a legacy/unbound checkpoint")
    train_split = trainer.run_binding.get("train_split")
    if not train_split:
        raise SystemExit("checkpoint binding has no train_split")
    bound_train = StoreReader(
        args.store,
        train_split,
        expected_whitener_hash=whitener.hash,
        sites=args.sites,
    )
    expected_binding = {
        "whitener_hash": whitener.hash,
        "sites": list(calib.sites),
        "gauge": {
            "normalization": whitener.mode,
            "site_rms_renorm": bool(args.site_renorm or folded_site_renorm),
            "site_renorm_at_load": bool(args.site_renorm),
            "site_renorm_folded": folded_site_renorm,
        },
        "model_id": whitener.meta.get("model"),
        "train_manifest_sha256": _json_sha256(bound_train.manifest),
        "train_tokens": bound_train.n_tokens,
    }
    try:
        validate_run_binding(
            trainer.run_binding,
            expected_binding,
            keys=tuple(expected_binding),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
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
    codec.meta["run_binding"] = trainer.run_binding
    codec.meta["calibration_manifest_sha256"] = _json_sha256(calib.manifest)
    codec.meta["eval_manifest_sha256"] = _json_sha256(eval_r.manifest)
    codec_path = args.codec_out or args.out.with_suffix(".codec.pt")
    codec.save(codec_path)
    # Evaluate the artifact that was actually serialized, not the transient
    # fit object, so publication output exercises the reload path.
    codec = Codec.load(codec_path)
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
        "site_renorm": bool(args.site_renorm or folded_site_renorm),
        "site_renorm_at_load": args.site_renorm,
        "normalization": whitener.mode,
        "site_renorm_folded": folded_site_renorm,
        "sites": list(calib.sites),
        "whitener_hash": whitener.hash,
        "run_binding": trainer.run_binding,
        "codec": str(codec_path),
        "spec": {"qs": list(spec.qs), "clip": [spec.clip_lo, spec.clip_hi],
                 "floor": spec.floor, "n_bootstrap": spec.n_bootstrap},
        "results": res,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
