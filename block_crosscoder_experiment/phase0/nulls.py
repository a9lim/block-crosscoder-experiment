"""Null calibration for the ring hunt (design §Phase 0).

Every positive claim in Phase 0 is read against a null run through the SAME
estimator: label-permutation for labeled tests (circular decoding,
n-gon alignment), random matched-size feature clusters for the unknown-
cluster search, and Benjamini–Hochberg over however many clusters the
search touched. A null result is informative at every gate; these tools are
what make it well-calibrated.
"""

from __future__ import annotations

from typing import Callable, Sequence

import torch

__all__ = [
    "benjamini_hochberg",
    "class_permutation_pvalue",
    "empirical_pvalue",
    "permutation_pvalue",
    "random_member_sets",
]


def empirical_pvalue(observed: float, null_scores: Sequence[float]) -> float:
    """P(null >= observed), add-one smoothed (Phipson–Smyth: never 0)."""
    exceed = sum(1 for s in null_scores if s >= observed)
    return (exceed + 1) / (len(null_scores) + 1)


def permutation_pvalue(
    stat_fn: Callable[[torch.Tensor], float],
    labels: torch.Tensor,
    *,
    n_perm: int = 200,
    seed: int = 0,
) -> tuple[float, float]:
    """Observed stat + p-value under label permutation.

    stat_fn maps a label vector to a scalar statistic; the observed labels
    are scored once, then n_perm shuffles build the null.
    """
    observed = stat_fn(labels)
    gen = torch.Generator().manual_seed(seed)
    null = [
        stat_fn(labels[torch.randperm(labels.shape[0], generator=gen)])
        for _ in range(n_perm)
    ]
    return observed, empirical_pvalue(observed, null)


def class_permutation_pvalue(
    stat_fn: Callable[[torch.Tensor], float],
    labels: torch.Tensor,
    n_classes: int,
    *,
    n_perm: int = 200,
    seed: int = 0,
) -> tuple[float, float]:
    """Observed stat + p-value under CLASS-IDENTITY permutation.

    Relabels classes consistently (k → π(k) for all tokens at once), so
    class clumps survive but their assignment to target positions
    randomizes. This is the null for cyclic-ORDER claims: token-level
    shuffling would reward any consistent clump layout, ring or not — a
    lesson a planted non-ring control taught us.
    """
    observed = stat_fn(labels)
    gen = torch.Generator().manual_seed(seed)
    null = []
    for _ in range(n_perm):
        perm = torch.randperm(n_classes, generator=gen).to(labels.device)
        null.append(stat_fn(perm[labels]))
    return observed, empirical_pvalue(observed, null)


def random_member_sets(
    n_features: int,
    size: int,
    *,
    n_draws: int = 100,
    seed: int = 0,
    exclude: torch.Tensor | None = None,
    frequencies: torch.Tensor | None = None,
    match_to: torch.Tensor | None = None,
    n_buckets: int = 10,
) -> list[torch.Tensor]:
    """Matched random feature subsets — the random-cluster null.

    `exclude` masks features (e.g. the candidate cluster itself) out of the
    draw pool. With `frequencies` (per-feature firing counts) and
    `match_to` (the candidate's members), each null member is drawn from
    the same log-frequency quantile bucket as the member it replaces —
    without this, random draws over a production-sparse dictionary almost
    never co-fire and the null starves.
    """
    pool_mask = torch.ones(n_features, dtype=torch.bool)
    if exclude is not None:
        pool_mask[exclude] = False
    pool = torch.arange(n_features)[pool_mask]
    if size > pool.shape[0]:
        raise ValueError(f"size {size} exceeds null pool {pool.shape[0]}")
    gen = torch.Generator().manual_seed(seed)

    if frequencies is None or match_to is None:
        return [
            pool[torch.randperm(pool.shape[0], generator=gen)[:size]]
            for _ in range(n_draws)
        ]

    logf = torch.log10(frequencies.to(torch.float64) + 1.0)
    edges = torch.quantile(logf, torch.linspace(0, 1, n_buckets + 1, dtype=torch.float64))
    bucket = torch.bucketize(logf, edges[1:-1])
    pools = [pool[bucket[pool] == b] for b in range(n_buckets)]
    member_buckets = bucket[match_to]
    draws = []
    for _ in range(n_draws):
        picks = []
        for b in member_buckets.tolist():
            candidates = pools[b] if pools[b].numel() else pool
            picks.append(
                int(candidates[torch.randint(candidates.shape[0], (1,), generator=gen)])
            )
        draws.append(torch.tensor(picks, dtype=torch.long))
    return draws


def benjamini_hochberg(
    pvalues: Sequence[float], *, alpha: float = 0.05
) -> torch.Tensor:
    """BH step-up: boolean rejection mask in the input order."""
    p = torch.tensor(list(pvalues), dtype=torch.float64)
    m = p.shape[0]
    order = p.argsort()
    ranked = p[order]
    thresh = alpha * torch.arange(1, m + 1, dtype=torch.float64) / m
    below = (ranked <= thresh).nonzero()
    mask = torch.zeros(m, dtype=torch.bool)
    if below.numel():
        cutoff = int(below.max())
        mask[order[: cutoff + 1]] = True
    return mask
