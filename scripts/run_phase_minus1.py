#!/usr/bin/env python
"""Run the Phase -1 scenario battery (design v2.2).

Full battery on jobe:
    python scripts/run_phase_minus1.py --out data/phase_minus1_report.json
Quick plumbing smoke:
    python scripts/run_phase_minus1.py --scenario core --steps 300 --seeds 0
"""

import argparse

import torch

from block_crosscoder_experiment.battery import SCENARIOS, BatteryConfig, run_battery


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario", nargs="*", default=None, choices=list(SCENARIOS))
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--seeds", type=int, nargs="*", default=[0, 1])
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    bc = BatteryConfig(
        steps=args.steps, batch_size=args.batch_size, seeds=tuple(args.seeds)
    )
    payload = run_battery(bc, device=device, scenarios=args.scenario, out_path=args.out)
    verdict = payload["all_hard_gates_pass"]
    print(f"[battery] hard gates: {'PASS' if verdict else 'FAIL' if verdict is not None else 'n/a'}")


if __name__ == "__main__":
    main()
