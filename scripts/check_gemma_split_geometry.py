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

Plus the Engels artifact form: PCA the per-class decoder vectors and test
calendar order of their ANGLES in the top plane (Fisher-Lee circular
correlation against class phase, permutation null) — with a figure per
family when --figures is set.

  python scripts/check_gemma_split_geometry.py --figures
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

RELEASE = "gemma-scope-2-4b-pt-res"
RUNS = (
    ("layer_22_width_16k_l0_medium",
     Path("/data/stores/bcc-phase0/gemma3_4b_l22_pile")),
    ("layer_22_width_65k_l0_medium",
     Path("/data/stores/bcc-phase0/gemma3_4b_l22_65k_pile")),
)
N_CLASSES = {"weekday": 7, "month": 12}


def _fisher_lee(a: list[float], b: list[float]) -> float:
    """Fisher-Lee circular-circular correlation of two angle lists."""
    num = s_a = s_b = 0.0
    for i, j in itertools.combinations(range(len(a)), 2):
        sa, sb = math.sin(a[i] - a[j]), math.sin(b[i] - b[j])
        num += sa * sb
        s_a += sa * sa
        s_b += sb * sb
    return num / math.sqrt(s_a * s_b) if s_a > 0 and s_b > 0 else 0.0


def decoder_angle_test(
    vecs, classes: list[int], n: int, *, n_perm: int = 20000, seed: int = 0
) -> dict:
    """PCA the per-class decoder vectors; test calendar order of PC1/2 angles.

    |r| is the statistic (a ring is a ring whichever way it winds).
    """
    import random

    import torch

    x = vecs - vecs.mean(dim=0, keepdim=True)
    _, s, v = torch.linalg.svd(x, full_matrices=False)
    proj = x @ v[:2].T
    angles = torch.atan2(proj[:, 1], proj[:, 0]).tolist()
    phase = [2 * math.pi * c / n for c in classes]
    obs = abs(_fisher_lee(angles, phase))
    rng = random.Random(seed)
    idx = list(range(len(classes)))
    ge = 0
    for _ in range(n_perm):
        rng.shuffle(idx)
        if abs(_fisher_lee(angles, [phase[i] for i in idx])) >= obs:
            ge += 1
    return {
        "explained_frac_pc12": round(
            float((s[:2] ** 2).sum() / (s**2).sum()), 4
        ),
        "angles": {str(c): round(a, 4) for c, a in zip(classes, angles)},
        "fisher_lee_r": round(obs, 4),
        "p_perm": round((ge + 1) / (n_perm + 1), 6),
        "proj": [[round(float(u), 5) for u in row] for row in proj],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=Path("data/phase0/split_geometry.json")
    )
    parser.add_argument(
        "--figures", action="store_true",
        help="render decoder PC1/2 maps into figures/phase0-gemma",
    )
    args = parser.parse_args()

    from sae_lens import SAE

    report: dict = {}
    for sae_id, store in RUNS:
        probe = json.loads((store / "target_run" / "supervised_ring.json").read_text())
        dec = SAE.from_pretrained(
            RELEASE, sae_id, device="cpu", dtype="float32"
        ).W_dec.detach()
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
            if len(set(feats)) >= 4:
                dedup = sorted(set(zip(feats, classes)))
                angle = decoder_angle_test(
                    dec[[f for f, _ in dedup]], [c for _, c in dedup], n
                )
                stats["decoder_angle_test"] = angle
                print(
                    f"  decoder PC1/2: explained {angle['explained_frac_pc12']:.2f}, "
                    f"|r| {angle['fisher_lee_r']:.3f}, p {angle['p_perm']:.4g}",
                    flush=True,
                )
                if args.figures:
                    import matplotlib

                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots(figsize=(6, 6))
                    order = sorted(range(len(dedup)), key=lambda i: dedup[i][1])
                    xs = [angle["proj"][i][0] for i in order]
                    ys = [angle["proj"][i][1] for i in order]
                    ax.plot(xs + xs[:1], ys + ys[:1], "-", color="0.8", zorder=1)
                    ax.scatter(xs, ys, c=[dedup[i][1] for i in order], cmap="hsv",
                               vmin=0, vmax=n, s=60, zorder=2)
                    for i in order:
                        ax.annotate(
                            f"{dedup[i][1]}", angle["proj"][i],
                            textcoords="offset points", xytext=(6, 4), fontsize=9,
                        )
                    ax.set_title(
                        f"{sae_id} {family}: top-1 decoder vectors, PC1/2 "
                        f"(|r| {angle['fisher_lee_r']:.2f}, p {angle['p_perm']:.4g})"
                    )
                    fig_dir = Path("figures/phase0-gemma")
                    fig_dir.mkdir(parents=True, exist_ok=True)
                    width = sae_id.split("width_")[1].split("_")[0]
                    fig.savefig(
                        fig_dir / f"{family}_decoder_ring_{width}.png", dpi=150
                    )
                    plt.close(fig)
        del dec

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
