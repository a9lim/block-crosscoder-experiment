"""Stream-side structure tests for the atlas zoo tranche (Mac-side).

The number-line depth table's analogue for the new families, from the
compact class-means export — no GPU, no token data:

  - color: hue-wheel adjacency ring hits + first-harmonic mass of the
    chromatic prefix, per depth (achromatic classes excluded from the
    ring, as in the block tests);
  - element / planet: Spearman |rho| of class order (atomic number /
    distance from sun) along PC1 of class means, per depth, classes
    with count >= 5 only;
  - country: LOO lat/lon decode R² per depth rides in
    `fig_worldmap_3d.py`; here just the continent silhouette per depth
    (do country means cluster by continent before they map?).

  python scripts/analysis/atlas_stream_tests.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

import sys

sys.path.insert(0, str(Path(__file__).parent))
from _geo import COUNTRY_GEO  # noqa: E402
from block_crosscoder_experiment.phase0.labels import (COUNTRIES,  # noqa: E402
                                                       FAMILIES)

DATA = Path("data/analysis")
SITES = [9, 12, 15, 18, 21, 24, 27, 30]
HUE_PREFIX = 6
MIN_COUNT = 5


def ring_hits(means: np.ndarray) -> tuple[int, float]:
    """Adjacency hits on the top-2 PCA plane + first-harmonic mass."""
    C = len(means)
    X = means - means.mean(0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    P = X @ Vt[:2].T
    ang = np.arctan2(P[:, 1], P[:, 0])
    order = np.argsort(ang)
    pos = np.empty(C, int)
    pos[order] = np.arange(C)
    hits = sum(min((pos[i] - pos[(i + 1) % C]) % C,
                   (pos[(i + 1) % C] - pos[i]) % C) == 1 for i in range(C))
    F = np.fft.fft(X, axis=0)
    power = (np.abs(F[1: C // 2 + 1]) ** 2).sum(1)
    return int(hits), float(power[0] / power.sum())


def silhouette(X: np.ndarray, labels: list[str]) -> float:
    from scipy.spatial.distance import cdist

    D = cdist(X, X)
    vals = []
    for i, li in enumerate(labels):
        same = [j for j, lj in enumerate(labels) if lj == li and j != i]
        if not same:
            continue
        a = D[i, same].mean()
        b = min(D[i, [j for j, lj in enumerate(labels) if lj == lc]].mean()
                for lc in set(labels) if lc != li)
        vals.append((b - a) / max(a, b))
    return float(np.mean(vals))


def main() -> None:
    zm = np.load(DATA / "zoo_means_atlas4b.npz")
    tests = json.loads((DATA / "zoo_block_tests_atlas4b.json").read_text())
    counts = {f: np.array(tests[next(iter(tests))][f]["class_counts"])
              for f in ("color", "country", "element", "planet")}
    out: dict = {"sites": SITES, "class_counts": {
        f: c.tolist() for f, c in counts.items()}}

    hue = zm["color_means"][:HUE_PREFIX]  # [6, S, d]
    out["color_hue_ring"] = []
    for s, L in enumerate(SITES):
        hits, h1 = ring_hits(hue[:, s])
        out["color_hue_ring"].append(
            {"site": L, "hits": hits, "h1": round(h1, 3)})
    print("color hue ring (6 chromatic):",
          " ".join(f"L{r['site']}:{r['hits']}/6({r['h1']:.2f})"
                   for r in out["color_hue_ring"]))

    for fam in ("element", "planet"):
        keep = counts[fam] >= MIN_COUNT
        M = zm[f"{fam}_means"][keep]
        idx = np.where(keep)[0]
        out[f"{fam}_line"] = []
        for s, L in enumerate(SITES):
            X = M[:, s] - M[:, s].mean(0)
            _, _, Vt = np.linalg.svd(X, full_matrices=False)
            rho = abs(spearmanr(idx, X @ Vt[0]).statistic)
            out[f"{fam}_line"].append({"site": L, "rho": round(float(rho), 3)})
        dropped = [FAMILIES[fam][i] for i in np.where(~keep)[0]]
        out[f"{fam}_dropped"] = dropped
        print(f"{fam} line |rho| ({int(keep.sum())} classes"
              + (f", dropped {dropped}" if dropped else "") + "):",
              " ".join(f"L{r['site']}:{r['rho']:.2f}"
                       for r in out[f"{fam}_line"]))

    keep = counts["country"] >= MIN_COUNT
    M = zm["country_means"][keep]
    conts = [COUNTRY_GEO[c][2] for c, k in zip(COUNTRIES, keep) if k]
    out["country_continent_silhouette"] = []
    for s, L in enumerate(SITES):
        X = M[:, s] - M[:, s].mean(0)
        sil = silhouette(X, conts)
        out["country_continent_silhouette"].append(
            {"site": L, "silhouette": round(sil, 3)})
    print(f"country continent silhouette ({int(keep.sum())} classes):",
          " ".join(f"L{r['site']}:{r['silhouette']:.2f}"
                   for r in out["country_continent_silhouette"]))

    (DATA / "atlas_stream_tests.json").write_text(
        json.dumps(out, indent=2) + "\n")
    print("-> data/analysis/atlas_stream_tests.json")


if __name__ == "__main__":
    main()
