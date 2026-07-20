"""Derive the showcase-block map for the current winner.

Block indices are checkpoint-specific: every promoted winner captures the
probe families in *different* blocks. This reads the family capture tests
(`zoo_block_tests.json` from probe_families, run over both arms) and
writes `showcase_blocks.json` into the winner's analysis dir — the single
place figure scripts look block identities up.

Qualification is the mega-block rule made mechanical — top-1 capture is
never read alone:

  consolidated   best block claims a majority of the family's classes
  ordered        the order statistic (ring hits / geo R^2 / line rho)
                 beats its permutation null at p <= --alpha
  qualified      both — figures skip unqualified families with a note

Per family the entry elects the better arm (qualified first, then lower
perm-p, then higher claim), keeping both arms' stats for transparency.

  python scripts/analysis/derive_showcase.py            # winner-scoped paths
  python scripts/analysis/derive_showcase.py --tests <path> --out <path>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from winner import analysis_dir, load_winner


def qualify(fam_entry: dict, n_classes: int, alpha: float) -> dict:
    order = fam_entry["order"]
    consolidated = fam_entry["top1_claimed"] * 2 > n_classes
    ordered = order["perm_p"] <= alpha
    return {
        "block": fam_entry["best_block"],
        "top1_claimed": fam_entry["top1_claimed"],
        "n_classes": n_classes,
        "distinct_top1": fam_entry["distinct_top1"],
        "order": order,
        "consolidated": consolidated,
        "ordered": ordered,
        "qualified": consolidated and ordered,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tests", type=Path, default=None,
                    help="zoo_block_tests.json (default: winner analysis dir)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--alpha", type=float, default=0.01)
    args = ap.parse_args()

    w = load_winner()
    adir = analysis_dir(w)
    tests_path = args.tests or adir / "zoo_block_tests.json"
    out_path = args.out or adir / "showcase_blocks.json"
    tests = json.loads(tests_path.read_text())

    # probe_families keys runs by the --runs names; the driver uses
    # 'winner' and 'primary'.
    arms = {name: entry for name, entry in tests.items()}
    families = sorted(
        k for entry in arms.values() for k in entry
        if isinstance(entry[k], dict) and "best_block" in entry[k]
    )

    out: dict = {
        "_derived_from": {
            "winner": w["run_name"],
            "tests": str(tests_path),
            "alpha": args.alpha,
        },
        "families": {},
    }
    for family in dict.fromkeys(families):
        per_arm = {}
        for arm, entry in arms.items():
            if family not in entry:
                continue
            n_classes = len(entry[family]["top1_map"])
            per_arm[arm] = qualify(entry[family], n_classes, args.alpha)
        if not per_arm:
            continue
        elected = sorted(
            per_arm,
            key=lambda a: (
                not per_arm[a]["qualified"],
                per_arm[a]["order"]["perm_p"],
                -per_arm[a]["top1_claimed"],
            ),
        )[0]
        out["families"][family] = {
            "arm": elected,
            **per_arm[elected],
            "arms": per_arm,
        }
        e = out["families"][family]
        o = e["order"]
        stat = (f"ring {o['hits']}/{o['max']}" if o["kind"] == "ring"
                else f"geo R2 {o['r2']}" if o["kind"] == "geo"
                else f"|rho| {o['spearman']}")
        flag = "OK " if e["qualified"] else "-- "
        print(f"{flag}{family}: {e['arm']} b{e['block']} "
              f"top1 {e['top1_claimed']}/{e['n_classes']} {stat} "
              f"(p {o['perm_p']:.2e})", flush=True)

    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
