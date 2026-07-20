"""Dump per-site decoder/encoder frames for a named block list.

The compact export lets figure generation and cross-arm alignment run without
shipping full checkpoints. ``bsc refresh-analysis`` derives the block list
from the current capture tests.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--blocks", type=int, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    ckpt = torch.load(args.run / "latest.pt", map_location="cpu",
                      weights_only=False)
    report = json.loads((args.run / "report.json").read_text())
    blocks = sorted(set(args.blocks))
    D = ckpt["model"]["D"]  # [S, G, b, d] fp32 master
    E = ckpt["model"]["E"]
    np.savez_compressed(
        args.out,
        blocks=np.array(blocks, dtype=np.int64),
        frames=D[:, blocks].float().numpy(),
        enc_frames=E[:, blocks].float().numpy(),
        theta=np.float32(float(ckpt["model"]["theta"])),
        site_renorm=np.bool_(bool(report.get("site_renorm"))),
        meta=json.dumps({"run": str(args.run),
                         "model_cfg": ckpt["model_cfg"]}),
    )
    print(f"{args.run.name}: {len(blocks)} blocks -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
