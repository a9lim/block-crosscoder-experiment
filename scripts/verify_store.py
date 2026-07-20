"""Phase-0.9 store verification: shard checksums + whitening round trip.

Re-hashes every shard in every split against its manifest checksum, then
measures the whitening round-trip error by re-whitening the retained raw
validation shard and comparing against the stored whitened calibration
prefix (same tokens; sequential reads preserve stream order).

  python -u scripts/verify_phase09_store.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--store", type=Path,
        default=Path("/data/stores/bcc-phase09/gemma3_1b_6site_fineweb"),
    )
    parser.add_argument(
        "--splits", nargs="+", default=None,
        help="verify only these whitened splits (checksums, no raw "
        "round-trip); default: the full 0.9-store protocol",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(f"config: store={args.store} device={args.device}", flush=True)

    import torch

    from block_crosscoder_experiment.store import StoreReader, Whitener

    whitener = Whitener.load(args.store / "whitener.pt")
    print(f"whitener hash {whitener.hash[:16]}…", flush=True)
    if args.splits is not None:
        for split in args.splits:
            reader = StoreReader(
                args.store, split, expected_whitener_hash=whitener.hash
            )
            t0 = time.time()
            n = reader.verify()
            dt = time.time() - t0
            gb = n * len(reader.manifest["sites"]) * reader.d_model * 2 / 1e9
            print(f"  {split}: {n:,} tokens ({gb:,.0f} GB, "
                  f"{len(reader.manifest['shards'])} shards) verified in "
                  f"{dt:.0f}s ({gb / max(dt, 1e-9):.2f} GB/s)", flush=True)
        print("store verified", flush=True)
        return
    readers = {}
    for split, expected in (
        ("calibration", whitener.hash),
        ("eval", whitener.hash),
        ("train", whitener.hash),
        ("raw_validation", "raw:" + whitener.hash),
    ):
        reader = StoreReader(args.store, split, expected_whitener_hash=expected)
        t0 = time.time()
        n = reader.verify()
        print(f"  {split}: {n:,} tokens, {len(reader.manifest['shards'])} "
              f"shards verified in {time.time() - t0:.0f}s", flush=True)
        readers[split] = reader

    raw = torch.cat(list(readers["raw_validation"].sequential_batches(16_384)))
    need, got = raw.shape[0], []
    for batch in readers["calibration"].sequential_batches(16_384):
        got.append(batch)
        if sum(g.shape[0] for g in got) >= need:
            break
    stored = torch.cat(got)[:need]
    w_dev = whitener.W.to(args.device)
    mu_dev = whitener.mean.to(args.device)
    xw = torch.einsum(
        "sde,nse->nsd", w_dev, raw.to(args.device).float() - mu_dev
    ).to(torch.bfloat16).cpu()
    rel = float((xw.float() - stored.float()).norm() / stored.float().norm())
    exact = float((xw == stored).float().mean())
    print(f"round trip: {need:,} tokens, rel err {rel:.3e}, "
          f"bf16 exact-match fraction {exact:.4f}", flush=True)
    print("store verified", flush=True)


if __name__ == "__main__":
    main()
