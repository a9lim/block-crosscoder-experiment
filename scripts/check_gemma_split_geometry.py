"""Decoder geometry of the supervised per-class features: ring or simplex?

The 16k → 65k supervised probes invert: at 16k the partially-split month
features sit in calendar order (circ 0.52, p 0.015); at 65k almost every
month has its own selective feature, yet calendar order is gone (circ
0.55, p 0.29) and the union figure is a star of near-orthogonal rays.
Hypothesis: ring geometry is a property of *partial* splitting — split
siblings still correlated in a low-rank family subspace — and dissolves
into near-orthogonal per-class axes as splitting completes.

Decoder-level test, per store: take the top-1 feature per class from
supervised_ring.json, compute pairwise decoder cosines, and compare
adjacent-in-cycle pairs vs non-adjacent pairs (rings predict adjacent >
non-adjacent; simplex predicts no difference, all near zero).

  python scripts/check_gemma_split_geometry.py
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

RELEASE = "gemma-scope-2-4b-pt-res"
RUNS = (
    ("layer_22_width_16k_l0_medium",
     Path("/data/stores/bcc-phase0/gemma3_4b_l22_pile")),
    ("layer_22_width_65k_l0_medium",
     Path("/data/stores/bcc-phase0/gemma3_4b_l22_65k_pile")),
)
N_CLASSES = {"weekday": 7, "month": 12}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=Path("data/phase0/split_geometry.json")
    )
    args = parser.parse_args()

    from sae_lens import SAE

    report: dict = {}
    for sae_id, store in RUNS:
        probe = json.loads((store / "target_run" / "supervised_ring.json").read_text())
        dec = SAE.from_pretrained(RELEASE, sae_id, device="cpu", dtype="float32").W_dec
        dec = dec / dec.norm(dim=1, keepdim=True).clamp_min(1e-12)
        report[sae_id] = {}
        for family, res in probe.items():
            n = N_CLASSES[family]
            # top-1 feature per class; classes with no selective feature drop out
            top1 = {
                int(c): int(picks[0][0])
                for c, picks in res["per_class"].items()
                if picks
            }
            classes = sorted(top1)
            feats = [top1[c] for c in classes]
            cos = dec[feats] @ dec[feats].T
            adj, nonadj = [], []
            for i, j in itertools.combinations(range(len(classes)), 2):
                if top1[classes[i]] == top1[classes[j]]:
                    continue  # same feature picked twice: no geometry to compare
                d = abs(classes[i] - classes[j])
                pair = float(cos[i, j])
                (adj if min(d, n - d) == 1 else nonadj).append(pair)
            stats = {
                "top1": {str(c): top1[c] for c in classes},
                "n_distinct_feats": len(set(feats)),
                "adjacent_mean_cos": round(sum(adj) / len(adj), 4) if adj else None,
                "adjacent_max_cos": round(max(adj), 4) if adj else None,
                "nonadjacent_mean_cos": (
                    round(sum(nonadj) / len(nonadj), 4) if nonadj else None
                ),
                "nonadjacent_max_cos": round(max(nonadj), 4) if nonadj else None,
                "cos_matrix": [[round(float(v), 3) for v in row] for row in cos],
            }
            report[sae_id][family] = stats
            print(
                f"{sae_id} {family}: {len(set(feats))} distinct top-1 feats over "
                f"{len(classes)} classes | adjacent mean cos "
                f"{stats['adjacent_mean_cos']} (max {stats['adjacent_max_cos']}) "
                f"vs non-adjacent {stats['nonadjacent_mean_cos']} "
                f"(max {stats['nonadjacent_max_cos']})",
                flush=True,
            )
        del dec

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
