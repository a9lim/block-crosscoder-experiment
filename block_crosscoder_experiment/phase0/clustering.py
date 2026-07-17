"""Decoder-direction clustering for post-hoc blockification (design §Phase 0).

Two Engels-fidelity geometric methods plus the activation-dependence branch
(P15) that runs beside them:

- spectral_clusters: Engels App. F.1 — spectral clustering on pairwise
  *angular* similarity (1 − arccos(cos)/π) of decoder rows, n_clusters=1000
  at the GPT-2 control scale. NJW recipe, full eigendecomposition (24k×24k
  is minutes on CUDA), seeded k-means.
- knn_graph_clusters: Engels App. F.2 — the Mistral-scale method: k=2
  nearest-neighbor graph, undirected, prune cosine < τ=0.5, connected
  components. The escape hatch for widths where a dense eigh won't fit.
- coactivation_similarity: P15 (Bhalla) — decoder-cosine correlation "need
  not suffice" exactly in shattering/dilution regimes, so a similarity
  built from *firing patterns* (co-occurrence cosine on binarized codes)
  feeds the same spectral machinery as a parallel branch. Ising fit stays
  the documented escalation if the branches disagree.

Plus the stability instrument the design requires for any cluster claim:
subsample/reseed reruns scored by best-match Jaccard.
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "angular_similarity",
    "cluster_sizes",
    "cluster_stability",
    "coactivation_similarity",
    "kmeans",
    "knn_graph_clusters",
    "spectral_clusters",
]


def _unit_rows(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)


def angular_similarity(decoder: torch.Tensor) -> torch.Tensor:
    """Pairwise angular similarity 1 − arccos(cos θ)/π of decoder rows."""
    u = _unit_rows(decoder.to(torch.float32))
    cos = (u @ u.T).clamp(-1.0, 1.0)
    return 1.0 - torch.arccos(cos) / math.pi


def coactivation_similarity(
    firings: torch.Tensor, *, chunk: int = 65536
) -> torch.Tensor:
    """Co-occurrence cosine n_ij / √(n_i n_j) over binarized codes (T, F).

    The P15 activation-dependence branch: two features are similar when
    they fire on the same tokens, regardless of decoder geometry.
    """
    if firings.ndim != 2:
        raise ValueError(f"expected (T, F) firings, got {tuple(firings.shape)}")
    f = firings.shape[1]
    counts = torch.zeros(f, f, dtype=torch.float32, device=firings.device)
    for start in range(0, firings.shape[0], chunk):
        block = firings[start : start + chunk].to(torch.float32)
        counts += block.T @ block
    diag = counts.diagonal().clamp_min(1.0)
    return counts / (diag.sqrt().unsqueeze(0) * diag.sqrt().unsqueeze(1))


def kmeans(
    points: torch.Tensor,
    n_clusters: int,
    *,
    seed: int = 0,
    n_init: int = 10,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> torch.Tensor:
    """Seeded k-means++ / Lloyd with restarts (sklearn-default n_init=10).

    Returns the lowest-inertia labeling; single-init k-means is unreliable
    at Engels' cluster counts and sklearn's SpectralClustering restarts too.
    """
    best_labels, best_inertia = None, float("inf")
    for init in range(n_init):
        labels, inertia = _kmeans_once(
            points, n_clusters, seed=seed * 1000 + init, max_iter=max_iter, tol=tol
        )
        if inertia < best_inertia:
            best_labels, best_inertia = labels, inertia
    assert best_labels is not None
    return best_labels


def _kmeans_once(
    points: torch.Tensor,
    n_clusters: int,
    *,
    seed: int,
    max_iter: int,
    tol: float,
) -> tuple[torch.Tensor, float]:
    n = points.shape[0]
    if n_clusters > n:
        raise ValueError(f"n_clusters {n_clusters} > n_points {n}")
    points = points.to(torch.float32)
    gen = torch.Generator().manual_seed(seed)

    # k-means++ init (distances computed on-device, draws on CPU generator).
    first = int(torch.randint(n, (1,), generator=gen))
    centers = [points[first]]
    d2 = (points - centers[0]).pow(2).sum(dim=1)
    for _ in range(1, n_clusters):
        total = d2.sum()
        if float(total) <= 0:  # duplicate-point degeneracy: any point works
            probs = torch.full((n,), 1.0 / n)
        else:
            probs = (d2 / total).cpu()
        nxt = int(torch.multinomial(probs, 1, generator=gen))
        centers.append(points[nxt])
        d2 = torch.minimum(d2, (points - centers[-1]).pow(2).sum(dim=1))
    c = torch.stack(centers)

    labels = torch.zeros(n, dtype=torch.long, device=points.device)
    dist = torch.zeros(n, n_clusters, device=points.device)
    for _ in range(max_iter):
        dist = (
            points.pow(2).sum(1, keepdim=True)
            - 2 * points @ c.T
            + c.pow(2).sum(1).unsqueeze(0)
        )
        labels = dist.argmin(dim=1)
        new_c = torch.zeros_like(c)
        new_c.index_add_(0, labels, points)
        sizes = torch.bincount(labels, minlength=n_clusters).unsqueeze(1)
        empty = sizes.squeeze(1) == 0
        new_c = new_c / sizes.clamp_min(1)
        if empty.any():
            # Reseed empty clusters at the currently worst-fit points.
            far = dist.gather(1, labels.unsqueeze(1)).squeeze(1)
            new_c[empty] = points[far.topk(int(empty.sum())).indices]
        shift = (new_c - c).pow(2).sum(1).max()
        c = new_c
        if float(shift) < tol:
            break
    inertia = float(dist.gather(1, labels.unsqueeze(1)).sum())
    return labels, inertia


def spectral_clusters(
    similarity: torch.Tensor,
    n_clusters: int = 1000,
    *,
    seed: int = 0,
) -> torch.Tensor:
    """Spectral clustering on a precomputed similarity, sklearn semantics.

    Engels ran sklearn's SpectralClustering; this reproduces its math with
    a full eigh (dense and exact — 24k×24k is fine on CUDA): top
    n_clusters eigenvectors of D^{-1/2} S D^{-1/2}, mapped back through
    D^{-1/2} (random-walk embedding, trivial eigenvector kept, no row
    normalization), then k-means with restarts.
    """
    s = similarity.to(torch.float32)
    if s.shape[0] != s.shape[1]:
        raise ValueError(f"similarity must be square, got {tuple(s.shape)}")
    d_inv_sqrt = s.sum(dim=1).clamp_min(1e-12).rsqrt()
    m = d_inv_sqrt.unsqueeze(1) * s * d_inv_sqrt.unsqueeze(0)
    # eigh returns ascending eigenvalues; the top block is the last columns.
    _, vecs = torch.linalg.eigh(m)
    emb = d_inv_sqrt.unsqueeze(1) * vecs[:, -n_clusters:]
    return kmeans(emb, n_clusters, seed=seed)


def knn_graph_clusters(
    decoder: torch.Tensor,
    *,
    k: int = 2,
    tau: float = 0.5,
) -> torch.Tensor:
    """Engels App. F.2 graph clustering: kNN edges, prune cos < τ, components."""
    u = _unit_rows(decoder.to(torch.float32))
    cos = u @ u.T
    cos.fill_diagonal_(-2.0)
    _, nbrs = cos.topk(k, dim=1)  # (F, k)

    n = u.shape[0]
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    rows = torch.arange(n, device=cos.device).repeat_interleave(k)
    cols = nbrs.reshape(-1)
    keep = (cos[rows, cols] >= tau).cpu()
    rows, cols = rows.cpu(), cols.cpu()
    for i, j in zip(rows[keep].tolist(), cols[keep].tolist()):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    roots = [find(i) for i in range(n)]
    remap: dict[int, int] = {}
    return torch.tensor([remap.setdefault(r, len(remap)) for r in roots])


def cluster_sizes(labels: torch.Tensor) -> torch.Tensor:
    return torch.bincount(labels)


def _best_match_jaccard(
    base: torch.Tensor, other: torch.Tensor, cluster_id: int
) -> float:
    members = base == cluster_id
    other_ids = other[members].unique()
    best = 0.0
    for oid in other_ids.tolist():
        o = other == oid
        inter = (members & o).sum()
        union = (members | o).sum()
        best = max(best, float(inter) / float(union))
    return best


def cluster_stability(
    similarity: torch.Tensor,
    labels: torch.Tensor,
    *,
    n_clusters: int,
    n_runs: int = 5,
    subsample: float = 0.9,
    seed: int = 0,
) -> dict[int, float]:
    """Per-cluster stability: best-match Jaccard against subsampled reruns.

    Each rerun spectrally reclusters a random `subsample` fraction of
    features with a fresh k-means seed; a cluster's stability is its mean
    best-match Jaccard (computed on the surviving members) across runs.
    """
    n = similarity.shape[0]
    gen = torch.Generator().manual_seed(seed)
    ids = labels.unique().tolist()
    totals = {i: 0.0 for i in ids}
    for run in range(n_runs):
        keep = torch.randperm(n, generator=gen)[: int(subsample * n)].sort().values
        sub = spectral_clusters(
            similarity[keep][:, keep], n_clusters, seed=seed + 1 + run
        )
        base_sub = labels[keep]
        for i in ids:
            totals[i] += _best_match_jaccard(base_sub, sub.to(base_sub.device), i)
    return {i: t / n_runs for i, t in totals.items()}
