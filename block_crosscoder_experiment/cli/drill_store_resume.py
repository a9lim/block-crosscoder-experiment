"""Exercise interrupted-harvest recovery on a synthetic scratch store.

Rehearses, on synthetic scratch data (never a real store), the recovery
procedure for a harvest that dies mid-write — the failure mode the 2.17 TB
production harvest must survive. The drill:

1. writes shards through the real ``ShardWriter`` and "crashes" before
   ``close()`` (no manifest), leaving a stray ``.tmp`` to simulate a
   mid-write kill;
2. recovers: completed shards are exactly the atomically-renamed
   ``shard_*.safetensors`` (``.tmp`` files are incomplete by construction
   and quarantined); headers are self-describing, so the manifest is
   rebuilt from them after a contiguity check on ``shard_index``;
3. verifies the rebuilt split with ``StoreReader.verify()`` (full content
   checksums);
4. prints the resume rule: a resumed harvest must write a NEW split and
   merge manifests — a fresh ``ShardWriter``
   restarts shard numbering at 0, so in-place append would collide.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scratch", type=Path, default=None,
        help="scratch root; default is a fresh system temporary directory",
    )
    parser.add_argument("--shards", type=int, default=5)
    parser.add_argument("--tokens-per-shard", type=int, default=50_000)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()
    if args.scratch is None:
        args.scratch = Path(tempfile.mkdtemp(prefix="bsc-resume-drill-"))

    import torch
    from safetensors import safe_open

    from block_crosscoder_experiment.store import (
        MANIFEST_NAME,
        ShardWriter,
        StoreReader,
    )

    sites = (9, 12, 15, 18, 21, 24, 27, 30)
    split = "drill"
    if args.scratch.exists():
        shutil.rmtree(args.scratch)
    fake_hash = "drill-" + "0" * 58

    # ---- 1. interrupted harvest ------------------------------------------
    writer = ShardWriter(
        args.scratch, split,
        whitener_hash=fake_hash, sites=sites, d_model=args.d_model,
        meta={"drill": "interrupted-harvest recovery"},
        tokens_per_shard=args.tokens_per_shard,
    )
    gen = torch.Generator().manual_seed(0)
    total = args.shards * args.tokens_per_shard
    written = 0
    while written < total:
        n = min(65_536, total - written)
        writer.add(torch.randn(n, len(sites), args.d_model, generator=gen))
        written += n
    # crash before close(): no manifest; simulate a mid-write kill artifact
    stray = args.scratch / split / f"shard_{args.shards:05d}.tmp"
    stray.write_bytes(b"\x00" * 1024)
    print(f"harvest 'crashed': {args.shards} complete shards, no manifest, "
          f"1 stray .tmp", flush=True)

    # ---- 2. recovery: rebuild the manifest from shard headers ------------
    split_dir = args.scratch / split
    tmps = sorted(split_dir.glob("*.tmp"))
    for t in tmps:
        t.unlink()
    print(f"quarantined {len(tmps)} incomplete .tmp file(s)", flush=True)

    entries = []
    for path in sorted(split_dir.glob("shard_*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as f:
            h = f.metadata()
        entries.append({
            "file": path.name,
            "n_tokens": int(h["n_tokens"]),
            "index": int(h["shard_index"]),
            "whitener_hash": h["whitener_hash"],
            "sites": json.loads(h["sites"]),
            "d_model": int(h["d_model"]),
            "meta": json.loads(h["meta"]),
        })
    entries.sort(key=lambda e: e["index"])
    contiguous = []
    for want, e in enumerate(entries):
        if e["index"] != want:
            print(f"  gap at shard {want} — truncating recovery there", flush=True)
            break
        contiguous.append(e)
    if not contiguous:
        raise SystemExit("nothing recoverable")
    hashes = {e["whitener_hash"] for e in contiguous}
    if len(hashes) != 1:
        raise SystemExit(f"mixed whitener hashes in shard headers: {hashes}")
    manifest = {
        "split": split,
        "whitener_hash": contiguous[0]["whitener_hash"],
        "sites": contiguous[0]["sites"],
        "d_model": contiguous[0]["d_model"],
        "n_tokens": sum(e["n_tokens"] for e in contiguous),
        "shards": [{"file": e["file"], "n_tokens": e["n_tokens"]}
                   for e in contiguous],
        "meta": contiguous[0]["meta"] | {"recovered": "manifest rebuilt from shard headers"},
    }
    (split_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"manifest rebuilt: {len(contiguous)} shards, "
          f"{manifest['n_tokens']:,} tokens", flush=True)

    # ---- 3. full checksum verification of the recovered split ------------
    reader = StoreReader(args.scratch, split, expected_whitener_hash=fake_hash)
    n = reader.verify()
    assert n == total, (n, total)
    print(f"recovered split verified: {n:,} tokens, checksums OK", flush=True)

    # ---- 4. the resume rule ----------------------------------------------
    rows_consumed = -(-manifest["n_tokens"] // 1022)
    print(
        "resume rule: relaunch with a NEW split (fresh ShardWriter restarts "
        "shard numbering at 0 — in-place append collides), skip-rows "
        f">= original skip + ceil(recovered/1022) = +{rows_consumed:,} rows "
        "+ margin, then merge manifests.",
        flush=True,
    )

    if not args.keep:
        shutil.rmtree(args.scratch)
        print("scratch removed", flush=True)
    print("drill complete", flush=True)


if __name__ == "__main__":
    main()
