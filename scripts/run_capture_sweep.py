#!/usr/bin/env python
"""Capture-conditions sweep (Phase -1, strict capture-as-written gate).

Factor-isolates the passing pinned-test config against the failing battery
core config, one flip at a time, 4 seeds per cell. The six candidate factors
(docs/findings-phase-minus1-battery.md section 3): zoo composition, feature
frequency, selection budget ratio, learner capacity G, optimizer (AdamW vs
8-bit Adam), and step count; plus aux pressure (s_aux) exposed as a seventh.

Full sweep on jobe:
    python scripts/run_capture_sweep.py --out data/capture_sweep.json
Quick plumbing smoke:
    python scripts/run_capture_sweep.py --cells A_base --seeds 0 --steps 200
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time

import torch

from block_crosscoder_experiment.battery import BatteryConfig, core_zoo, run_one_full
from block_crosscoder_experiment.synthetic import BlockSpec

OVERLAP_PASS = 0.9
R2_PASS = 0.8
SPAN_FLOOR = 0.5  # below: the learner never found the block at all


def core6_specs() -> list[BlockSpec]:
    return core_zoo()[0]


def core5_specs(frequency: float) -> list[BlockSpec]:
    """Core zoo minus the thickened shell (index 3), refrequenced.
    At f=0.25 this is exactly the pinned unit-test zoo
    (tests/test_metrics.py::test_end_to_end_recovery)."""
    specs = [s for i, s in enumerate(core6_specs()) if i != 3]
    return [dataclasses.replace(s, frequency=frequency) for s in specs]


ZOOS = {
    "core6": core6_specs,  # E[active] = 1.0
    "core5_f20": lambda: core5_specs(0.2),  # E[active] = 1.0
    "core5_f25": lambda: core5_specs(0.25),  # E[active] = 1.25 (pinned zoo)
}


def cell_grid() -> dict[str, dict]:
    """A = failing battery-core config; B = passing pinned-test config.
    OFAT flips from A toward B, plus the B side and its 8-bit bridge."""
    A = dict(zoo="core6", G=10, k=1.0, optimizer="auto", steps=3000, s_aux=4)
    B = dict(zoo="core5_f25", G=8, k=1.0, optimizer="adamw", steps=2000, s_aux=2)
    return {
        "A_base": A,
        "A_adamw": {**A, "optimizer": "adamw"},
        "A_budget08": {**A, "k": 0.8},
        "A_budget09": {**A, "k": 0.9},
        "A_G8": {**A, "G": 8},
        "A_zoo5_G10": {**A, "zoo": "core5_f20"},
        "A_zoo5_G8": {**A, "zoo": "core5_f20", "G": 8},
        "A_steps2k": {**A, "steps": 2000},
        "A_steps10k": {**A, "steps": 10000},
        "A_saux2": {**A, "s_aux": 2},
        "B_pinned": B,
        "B_8bit": {**B, "optimizer": "auto"},
    }


def classify(rec) -> str:
    """Basin label per planted block under the strict-capture reading."""
    if rec.matched is None or not math.isfinite(rec.overlap) or rec.overlap < SPAN_FLOOR:
        return "missing"
    if rec.overlap > OVERLAP_PASS:
        return "captured" if rec.code_r2 > R2_PASS else "tiled"
    return "partial"


def run_cell(name: str, cell: dict, seeds: list[int], bc0: BatteryConfig, device) -> dict:
    specs = ZOOS[cell["zoo"]]()
    bc = dataclasses.replace(bc0, steps=cell["steps"])
    runs = []
    for seed in seeds:
        t0 = time.time()
        rep, _, _ = run_one_full(
            specs, bc,
            n_blocks=cell["G"], k=cell["k"],
            learner_seed=seed, data_seed=seed,
            optimizer=cell["optimizer"], s_aux=cell["s_aux"],
            device=device,
        )
        labels = [classify(r) for r in rep.blocks]
        blocks = [
            {"planted": r.planted, "rank": r.rank,
             "geometry": specs[r.planted].geometry,
             "label": lab,
             "overlap": None if not math.isfinite(r.overlap) else round(r.overlap, 4),
             "code_r2": None if not math.isfinite(r.code_r2) else round(r.code_r2, 4),
             "support_size": r.support_size}
            for r, lab in zip(rep.blocks, labels)
        ]
        runs.append({
            "seed": seed,
            "captured_fraction": labels.count("captured") / len(labels),
            "labels": labels,
            "n_dead": rep.n_learned_dead,
            "n_alive": rep.n_learned_eligible,
            "blocks": blocks,
            "wall_s": round(time.time() - t0, 1),
        })
        print(f"[sweep] {name} seed {seed}: "
              f"cap {runs[-1]['captured_fraction']:.2f} "
              f"labels {labels} dead {rep.n_learned_dead} "
              f"({runs[-1]['wall_s']}s)", flush=True)
    caps = [r["captured_fraction"] for r in runs]
    return {
        "cell": name, "config": cell, "runs": runs,
        "capture_mean": sum(caps) / len(caps), "capture_min": min(caps),
        "n_missing": sum(r["labels"].count("missing") for r in runs),
        "n_tiled": sum(r["labels"].count("tiled") for r in runs),
        "n_partial": sum(r["labels"].count("partial") for r in runs),
    }


def run_replica(bc0: BatteryConfig, device) -> dict:
    """Exact reproduction of the pinned unit test through the sweep
    machinery: learner_seed=1, data stream seed 7 (= 1000 + (-993)).
    Expected 5/5 captured — validates the harness against the known pass."""
    bc = dataclasses.replace(bc0, steps=2000)
    rep, _, _ = run_one_full(
        ZOOS["core5_f25"](), bc, n_blocks=8, k=1.0,
        learner_seed=1, data_seed=-993,
        optimizer="adamw", s_aux=2, device=device,
    )
    labels = [classify(r) for r in rep.blocks]
    print(f"[sweep] B_replica (pinned test, seeds 1/7): labels {labels}", flush=True)
    return {"cell": "B_replica", "labels": labels,
            "captured_fraction": labels.count("captured") / len(labels)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    grid = cell_grid()
    p.add_argument("--cells", nargs="*", default=None, choices=list(grid))
    p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3])
    p.add_argument("--steps", type=int, default=None,
                   help="override steps in every cell (smoke runs)")
    p.add_argument("--skip-replica", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    bc0 = BatteryConfig()
    cells = {n: dict(c) for n, c in grid.items() if args.cells is None or n in args.cells}
    if args.steps is not None:
        for c in cells.values():
            c["steps"] = args.steps

    results = {}
    if not args.skip_replica:
        results["B_replica"] = run_replica(bc0, device)
    for name, cell in cells.items():
        results[name] = run_cell(name, cell, args.seeds, bc0, device)

    print(f"\n{'cell':<14} {'cap_mean':>8} {'cap_min':>8} "
          f"{'tiled':>5} {'partial':>7} {'missing':>7}")
    for name, r in results.items():
        if name == "B_replica":
            continue
        print(f"{name:<14} {r['capture_mean']:>8.2f} {r['capture_min']:>8.2f} "
              f"{r['n_tiled']:>5} {r['n_partial']:>7} {r['n_missing']:>7}")

    if args.out:
        payload = {"device": str(device), "seeds": args.seeds, "results": results}
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=1)
        print(f"[sweep] wrote {args.out}")


if __name__ == "__main__":
    main()
