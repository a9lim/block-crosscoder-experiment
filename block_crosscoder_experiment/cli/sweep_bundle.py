#!/usr/bin/env python
"""Round 7: bundle_null ring-capture sweep (Phase -1 capture campaign).

Battery run 5 (2026-07-16, the first run at the honest 10k operating point)
failed exactly one hard gate: bundle_null. Per-seed picture at G16 / k=1.2 /
10k: no-hallucination 4/4 (the co-active bundle is never reported as a ring
- it also never packs into one block); ring detection 1/4. Seeds 0 and 3
capture the ring *span* perfectly (overlap 0.9995) but soft phase-split it
across two co-firing blocks (cond_rate ~0.73 each, i.e. ~46% of ring tokens
shared; each block's magnitude varies with ring phase, norm_cv ~0.22 > 0.1).
Seed 1 is an ordinary miss (overlap 0.51). This refutes the in-code
assumption that ring splits stay norm-concentrated ("arcs of a hollow shell
are still norm-concentrated") - the split is by soft amplitude sharing, not
by hard arcs - and is the tiled-ring instrument caveat observed in vivo.

Under strict capture-as-written the gate stands; the fix must make training
capture the ring in a single block, seed-robustly. Levers, 4 seeds per cell
(k is absolute; zoo E[active] = 1.5, so k=1.2 is the battery's ratio-0.8
budget):

  - G8: remove the spare blocks the split needs (G16 gives it room);
  - 30k steps: the convergence force that merged the identical decoy twins
    (round 5) should merge the two half-rings - a rank-2 ring fits b=4, so
    here that force works *for* the gate;
  - k=0.9: tight enough to price the bundle's unpacked format (4 blocks) out
    of the budget - if the bundle packs, the ring faces less competition;
  - k=1.5: matched-unpacked budget - round 4 says loose budgets tile shells,
    check that holds in this zoo (expect worse, but the map has surprised us);
  - G16/k1.2 reference cell: exact battery run-5 config, doubling as an
    in-repo determinism probe now that configs are honest.

Run on jobe:
    bsc sweep-bundle --out data/capture_sweep_round7.json
Quick plumbing smoke:
    bsc sweep-bundle --cells H_bun_G16_k12 --seeds 0 --steps 300
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time

import torch

from block_crosscoder_experiment.battery import (
    BatteryConfig,
    _gate_associated_cv,
    bundle_zoo,
    run_one_full,
)

RING = 4  # planted index of the hollow ring in bundle_zoo


def cell_grid() -> dict[str, dict]:
    return {
        "H_bun_G16_k12": dict(G=16, k=1.2, steps=10000),  # run-5 reference
        "H_bun_G8_k12": dict(G=8, k=1.2, steps=10000),
        "H_bun_G8_k12_30k": dict(G=8, k=1.2, steps=30000),
        "H_bun_G16_k12_30k": dict(G=16, k=1.2, steps=30000),
        "H_bun_G16_k09": dict(G=16, k=0.9, steps=10000),
        "H_bun_G16_k15": dict(G=16, k=1.5, steps=10000),
        # Round 8: k=0.9 solved capture (4/4 single-block rings, code-R2
        # 0.9996, bundle packed 4/4) but budget slack junk-fills the ring
        # block on 2.3% of off-ring tokens at norm 0.074 vs ring norm 2.000,
        # inflating all-firings CV 0.010 -> 0.254 (bimodal arithmetic:
        # CV ~ 0.97*sqrt(junk fraction); < 0.1 needs junk < ~1% of firings).
        # Prediction: the gate-as-written window is k just above the packed
        # demand 0.75 - packing forced, slack ~0, junk ~0. Probe the window
        # and its starvation edge.
        "I_bun_G16_k070": dict(G=16, k=0.70, steps=10000),
        "I_bun_G16_k075": dict(G=16, k=0.75, steps=10000),
        "I_bun_G16_k080": dict(G=16, k=0.80, steps=10000),
    }


def run_cell(name: str, cell: dict, seeds: list[int], bc0: BatteryConfig, device) -> dict:
    specs, _ = bundle_zoo()
    bc = dataclasses.replace(bc0, steps=cell["steps"])
    runs = []
    for seed in seeds:
        t0 = time.time()
        rep, trainer, truth = run_one_full(
            specs, bc, n_blocks=cell["G"], k=cell["k"],
            learner_seed=seed, data_seed=seed, device=device,
        )
        # Mirror scenario_bundle_null exactly (same eval seed) so cell
        # verdicts transfer 1:1 to the battery gate.
        batch_gates = truth.sample(bc.n_eval, seed=98).active
        bundle_assoc = _gate_associated_cv(
            trainer.master, truth, batch_gates[:, 0], device, bc.n_eval, 98
        )
        ring_assoc = _gate_associated_cv(
            trainer.master, truth, batch_gates[:, RING], device, bc.n_eval, 98
        )
        no_hallucination = all(
            r["norm_cv"] >= bc.norm_cv_shell_max for r in bundle_assoc
        )
        ring_detected = any(
            r["norm_cv"] < bc.norm_cv_shell_max for r in ring_assoc
        )
        ring_row = rep.blocks[RING]
        runs.append({
            "seed": seed,
            "gate": no_hallucination and ring_detected,
            "no_hallucination": no_hallucination,
            "ring_detected": ring_detected,
            "ring_overlap": round(ring_row.overlap, 4),
            "ring_code_r2": round(ring_row.code_r2, 4),
            "n_ring_blocks": len(ring_assoc),
            "ring_associated": ring_assoc,
            "bundle_associated": bundle_assoc,
            "n_dead": rep.n_learned_dead,
            "wall_s": round(time.time() - t0, 1),
        })
        r = runs[-1]
        print(f"[sweep] {name} seed {seed}: gate={'PASS' if r['gate'] else 'FAIL'} "
              f"ring(det={r['ring_detected']}, ov={r['ring_overlap']}, "
              f"r2={r['ring_code_r2']}, blocks={r['n_ring_blocks']}, "
              f"cv={[round(x['norm_cv'], 3) for x in ring_assoc]}) "
              f"halluc_ok={r['no_hallucination']} ({r['wall_s']}s)", flush=True)
    return {
        "cell": name, "config": cell, "runs": runs,
        "n_gate_pass": sum(r["gate"] for r in runs),
        "n_ring_detected": sum(r["ring_detected"] for r in runs),
        "n_no_hallucination": sum(r["no_hallucination"] for r in runs),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    grid = cell_grid()
    p.add_argument("--cells", nargs="*", default=None, choices=list(grid))
    p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3])
    p.add_argument("--steps", type=int, default=None,
                   help="override steps in every cell (smoke runs)")
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
    for name, cell in cells.items():
        results[name] = run_cell(name, cell, args.seeds, bc0, device)

    print(f"\n{'cell':<20} {'gate':>6} {'ring_det':>8} {'no_halluc':>9}")
    for name, r in results.items():
        n = len(r["runs"])
        print(f"{name:<20} {r['n_gate_pass']}/{n:<4} {r['n_ring_detected']:>6}/{n} "
              f"{r['n_no_hallucination']:>7}/{n}")

    if args.out:
        payload = {"device": str(device), "seeds": args.seeds, "results": results}
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=1)
        print(f"[sweep] wrote {args.out}")


if __name__ == "__main__":
    main()
