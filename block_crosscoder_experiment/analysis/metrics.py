"""Shared topology statistics for descriptive and confirmatory probes."""

from __future__ import annotations

import numpy as np


def ring_stats(means: np.ndarray) -> tuple[int, float]:
    """Return cyclic-adjacency hits and top-plane explained variance."""

    centered = means - means.mean(0)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    plane = centered @ vt[:2].T
    angles = np.arctan2(plane[:, 1], plane[:, 0])
    n_classes = len(angles)
    order = np.argsort(angles)
    positions = np.empty(n_classes, int)
    positions[order] = np.arange(n_classes)
    distance = np.abs(np.diff(np.concatenate([positions, positions[:1]])))
    distance = np.minimum(distance, n_classes - distance)
    hits = int((distance == 1).sum())
    top_plane = float(
        (singular_values[:2] ** 2).sum()
        / max((singular_values**2).sum(), 1e-12)
    )
    return hits, top_plane


def ring_permutation_p(
    observed_hits: int,
    n_classes: int,
    n_permutations: int,
    seed: int = 0,
) -> float:
    """Permutation p-value for at least ``observed_hits`` ring adjacencies."""

    rng = np.random.default_rng(seed)
    greater_or_equal = 0
    indices = np.arange(n_classes)
    for _ in range(n_permutations):
        positions = rng.permutation(n_classes)
        distance = np.abs(
            np.diff(np.concatenate([positions[indices], positions[indices[:1]]]))
        )
        distance = np.minimum(distance, n_classes - distance)
        greater_or_equal += int(int((distance == 1).sum()) >= observed_hits)
    return (1 + greater_or_equal) / (n_permutations + 1)


__all__ = ["ring_permutation_p", "ring_stats"]
