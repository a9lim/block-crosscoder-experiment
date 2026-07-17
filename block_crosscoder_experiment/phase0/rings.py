"""Ring-structure evidence beyond the Engels indices (design §Phase 0).

Phase −1 lesson, binding here (findings §2.3): norm-concentration is never a
ring detector by itself. Ring claims need span-level, gate-conditional
evidence — these are the added instruments:

- circular_decoding: held-out test that known cyclic labels sit in cyclic
  ORDER around the cluster plane (fit θ = ±φ + δ on train folds, score
  mean-cosine on held-out tokens). Chance ≈ 0, perfect ring = 1.
- ngon_alignment: are the n class-mean angles equispaced (n-th harmonic
  alignment of class means), and how concentrated is each class
  (per-class resultant length)?
- angle_harmonic_power: label-free Fourier structure of code angle, for the
  unknown-cluster search where no cyclic labels exist.
- plane_scan / cone_normalize: the Engels PC-plane protocol (planes 1–2
  through 4–5, mean scores) plus the PC1-as-intensity cone check.
"""

from __future__ import annotations

import math

import torch

from block_crosscoder_experiment.phase0.indices import irreducibility_score

__all__ = [
    "angle_harmonic_power",
    "circular_decoding",
    "cone_normalize",
    "ngon_alignment",
    "pca_projections",
    "plane_scan",
]

DEFAULT_PLANES: tuple[tuple[int, int], ...] = ((0, 1), (1, 2), (2, 3), (3, 4))


def pca_projections(
    points: torch.Tensor, k: int = 5
) -> tuple[torch.Tensor, torch.Tensor]:
    """Center and project onto the top-k PCs: (proj (N, k), explained (k,)).

    `explained` is each kept PC's share of total variance — plane_scan uses
    it to skip numerically-degenerate PCs.
    """
    x = points.to(torch.float32)
    x = x - x.mean(dim=0)
    k = min(k, *x.shape)
    _, svals, vt = torch.linalg.svd(x, full_matrices=False)
    explained = svals.pow(2) / svals.pow(2).sum().clamp_min(1e-30)
    return x @ vt[:k].T, explained[:k]


def plane_scan(
    proj: torch.Tensor,
    *,
    planes: tuple[tuple[int, int], ...] = DEFAULT_PLANES,
    explained: torch.Tensor | None = None,
    min_variance_ratio: float = 0.01,
    mixture_steps: int = 10_000,
    seed: int = 0,
) -> dict:
    """Engels protocol: score PC planes, report per-plane + mean scores.

    When `explained` is given, planes touching a PC below
    `min_variance_ratio` of total variance are skipped — on an exactly
    low-rank reconstruction the trailing PCs are float noise that the
    index normalization would blow up to unit scale. Falls back to the
    top plane if the filter empties the scan.
    """
    usable = [p for p in planes if max(p) < proj.shape[1]]
    if explained is not None:
        usable = [
            p
            for p in usable
            if float(explained[list(p)].min()) >= min_variance_ratio
        ] or [planes[0]]
    per_plane = {
        p: irreducibility_score(
            proj[:, list(p)], mixture_steps=mixture_steps, seed=seed
        )
        for p in usable
    }
    mean = {
        key: sum(s[key] for s in per_plane.values()) / len(per_plane)
        for key in ("separability", "mixture", "score")
    }
    best = max(per_plane, key=lambda p: per_plane[p]["score"])
    return {"planes": per_plane, "mean": mean, "best_plane": best}


def cone_normalize(proj: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """PC1-as-intensity: divide the remaining PCs by |PC1| (cone → ring).

    Engels: the ring may live in PCs 2–3 of a cone whose radius is the
    activation intensity along PC1.
    """
    return proj[:, 1:] / proj[:, :1].abs().clamp_min(eps)


def _angles(plane_points: torch.Tensor) -> torch.Tensor:
    centered = plane_points - plane_points.mean(dim=0)
    return torch.atan2(centered[:, 1], centered[:, 0])


def circular_decoding(
    plane_points: torch.Tensor,
    class_ids: torch.Tensor,
    n_classes: int,
    *,
    folds: int = 5,
    seed: int = 0,
) -> float:
    """Held-out circular decoding score in [−1, 1].

    Model: target angle θ_k = 2πk/n; fit θ = s·φ + δ (s ∈ {±1}, δ closed
    form) on train folds; score = mean cos(θ − s·φ − δ) on the held-out
    fold. Tests cyclic ORDER, not just class separation: arbitrarily placed
    class clumps fit ± rotation poorly.
    """
    phi = _angles(plane_points)
    theta = class_ids.to(torch.float32) * (2 * math.pi / n_classes)
    n = phi.shape[0]
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen)
    scores = []
    for f in range(folds):
        test = perm[f * n // folds : (f + 1) * n // folds]
        train = torch.cat([perm[: f * n // folds], perm[(f + 1) * n // folds :]])
        best = None
        for s in (1.0, -1.0):
            resid = theta[train] - s * phi[train]
            delta = torch.atan2(resid.sin().sum(), resid.cos().sum())
            train_score = (theta[train] - s * phi[train] - delta).cos().mean()
            if best is None or train_score > best[0]:
                best = (train_score, s, delta)
        assert best is not None
        _, s, delta = best
        scores.append(float((theta[test] - s * phi[test] - delta).cos().mean()))
    return sum(scores) / len(scores)


def ngon_alignment(
    plane_points: torch.Tensor,
    class_ids: torch.Tensor,
    n_classes: int,
) -> dict[str, float]:
    """Equispacing + concentration of class-mean angles on the plane.

    alignment = |Σ_k e^{i·n·μ_k}| / n ∈ [0, 1]: 1 iff the n class means sit
    on an equispaced n-gon grid. concentration = mean per-class resultant
    length (how tightly each class clusters around its mean angle).
    """
    phi = _angles(plane_points)
    mean_angles, resultants = [], []
    for k in range(n_classes):
        member = class_ids == k
        if not bool(member.any()):
            continue
        c, s = phi[member].cos().mean(), phi[member].sin().mean()
        mean_angles.append(math.atan2(float(s), float(c)))
        resultants.append(float((c * c + s * s) ** 0.5))
    n = len(mean_angles)
    align_c = sum(math.cos(n_classes * m) for m in mean_angles) / max(n, 1)
    align_s = sum(math.sin(n_classes * m) for m in mean_angles) / max(n, 1)
    return {
        "alignment": (align_c**2 + align_s**2) ** 0.5,
        "concentration": sum(resultants) / max(n, 1),
        "classes_present": n,
    }


def angle_harmonic_power(
    plane_points: torch.Tensor,
    *,
    harmonics: range = range(1, 13),
) -> dict[int, float]:
    """Label-free Fourier structure of code angle: m ↦ |E[e^{i·m·φ}]|.

    For an isotropic cloud every harmonic is O(1/√N); a hidden n-class ring
    concentrates power at m = n. Feeds the unknown-cluster search, where
    p-values come from the random-cluster null, BH-corrected.
    """
    phi = _angles(plane_points)
    return {
        m: float(
            torch.complex((m * phi).cos().mean(), (m * phi).sin().mean()).abs()
        )
        for m in harmonics
    }
