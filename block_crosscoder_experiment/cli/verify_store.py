"""Verify store manifests, shard checksums, and whitening round trip.

Re-hashes every shard in every split against its manifest checksum, then
measures the whitening round-trip error by re-whitening the retained raw
validation shard and comparing against the stored whitened calibration
prefix (same tokens; sequential reads preserve stream order).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path


def _verify_round_trip(
    whitener,
    raw_reader,
    stored_reader,
    *,
    device: str,
    batch_size: int = 8_192,
) -> tuple[int, float, float]:
    """Compare a raw prefix to its stored transform with bounded memory."""
    import torch

    stored_batches = iter(stored_reader.sequential_batches(batch_size))
    stored_carry = None
    n_tokens = 0
    diff_sq = 0.0
    denom_sq = 0.0
    exact_values = 0
    n_values = 0
    for raw in raw_reader.sequential_batches(batch_size):
        need = raw.shape[0]
        pieces = []
        while need:
            if stored_carry is None or stored_carry.shape[0] == 0:
                try:
                    stored_carry = next(stored_batches)
                except StopIteration as exc:
                    raise ValueError(
                        "stored calibration is shorter than raw-validation prefix"
                    ) from exc
            take = min(need, stored_carry.shape[0])
            pieces.append(stored_carry[:take])
            stored_carry = stored_carry[take:]
            need -= take
        stored = torch.cat(pieces) if len(pieces) > 1 else pieces[0]
        transformed = whitener.apply(raw.to(device)).to(torch.bfloat16).cpu()
        delta = transformed.float() - stored.float()
        diff_sq += float(delta.square().sum(dtype=torch.float64))
        denom_sq += float(stored.float().square().sum(dtype=torch.float64))
        exact_values += int((transformed == stored).sum())
        n_values += stored.numel()
        n_tokens += raw.shape[0]
    rel = (diff_sq / max(denom_sq, 1e-300)) ** 0.5
    exact = exact_values / max(n_values, 1)
    return n_tokens, rel, exact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--store", type=Path, required=True,
    )
    parser.add_argument(
        "--splits", nargs="+", default=None,
        help="verify only these whitened splits (checksums, no raw "
        "round-trip); default: the full production-store protocol",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(f"config: store={args.store} device={args.device}", flush=True)

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

    need, rel, exact = _verify_round_trip(
        whitener,
        readers["raw_validation"],
        readers["calibration"],
        device=args.device,
    )
    print(f"round trip: {need:,} tokens, rel err {rel:.3e}, "
          f"bf16 exact-match fraction {exact:.4f}", flush=True)
    print("store verified", flush=True)


if __name__ == "__main__":
    main()
