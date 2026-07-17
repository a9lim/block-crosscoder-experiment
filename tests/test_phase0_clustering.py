"""Clustering recovery on planted decoder geometry + firing patterns.

Ground truth: G groups of decoder rows spanning random 2-D planes (high
intra-group cosine), plus lone isotropic directions. The P15 case plants
groups that are *invisible* to decoder cosine (orthogonal rows) but co-fire
— the co-activation branch must find them where the geometric branch can't.
"""

import pytest
import torch

from block_crosscoder_experiment.phase0.clustering import (
    angular_similarity,
    cluster_stability,
    coactivation_similarity,
    knn_graph_clusters,
    spectral_clusters,
)

G, PER, BACKGROUND, DIM = 6, 10, 20, 64
N_CLUSTERS = 10


def _planted_decoder(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Decoder rows in G tight planar fans + isotropic background.

    Fans are near-duplicate directions (10° arc, evenly spaced): the Engels
    premise is that the sparsity penalty tiles a multi-d feature's plane
    densely, so real fans are tight. Background rows are labeled −1; no
    assertion is made about how k-means buckets them — battery + nulls own
    that judgement downstream, not the clustering step.
    """
    gen = torch.Generator().manual_seed(7)
    rows, labels = [], []
    for g in range(G):
        basis = torch.linalg.qr(torch.randn(DIM, 2, generator=gen)).Q  # (DIM, 2)
        theta = torch.linspace(0, torch.pi / 18, PER)
        theta = theta + 0.002 * torch.randn(PER, generator=gen)
        coords = torch.stack([theta.cos(), theta.sin()], dim=1)
        rows.append(coords @ basis.T)
        labels += [g] * PER
    noise = torch.randn(BACKGROUND, DIM, generator=gen)
    rows.append(noise / noise.norm(dim=1, keepdim=True))
    labels += [-1] * BACKGROUND
    return torch.cat(rows).to(device), torch.tensor(labels, device=device)


def _group_recovery(pred: torch.Tensor, truth: torch.Tensor, group: int) -> float:
    members = truth == group
    ids, counts = pred[members].unique(return_counts=True)
    best = ids[counts.argmax()]
    o = pred == best
    return float((members & o).sum()) / float((members | o).sum())


def test_spectral_recovers_planted_groups(device):
    decoder, truth = _planted_decoder(device)
    labels = spectral_clusters(angular_similarity(decoder), N_CLUSTERS, seed=0)
    labels = labels.to(device)
    for g in range(G):
        assert _group_recovery(labels, truth, g) >= 0.9, f"group {g}"


def test_knn_graph_recovers_planted_groups(device):
    decoder, truth = _planted_decoder(device)
    labels = knn_graph_clusters(decoder).to(device)
    for g in range(G):
        assert _group_recovery(labels, truth, g) >= 0.9, f"group {g}"
    # Background must not glue to the planted fans under the τ prune.
    background_ids = labels[truth < 0]
    fan_ids = labels[truth >= 0]
    assert not set(background_ids.tolist()) & set(fan_ids.tolist())


def test_coactivation_branch_finds_cosine_invisible_groups(device):
    """P15: orthogonal decoders, correlated firings — geometry is blind."""
    gen = torch.Generator().manual_seed(11)
    n_feat, n_tok, n_groups = 24, 4000, 4
    decoder = torch.linalg.qr(torch.randn(n_feat, n_feat, generator=gen)).Q
    truth = torch.arange(n_feat) // (n_feat // n_groups)
    gate = torch.rand(n_tok, n_groups, generator=gen) < 0.15  # (T, G)
    firings = gate[:, truth]  # members of a group co-fire
    firings &= torch.rand(n_tok, n_feat, generator=gen) < 0.9  # per-feature dropout

    labels = spectral_clusters(
        coactivation_similarity(firings.to(device)), n_groups, seed=0
    ).to(device)
    truth = truth.to(device)
    for g in range(n_groups):
        assert _group_recovery(labels, truth, g) >= 0.9, f"group {g}"

    # And the geometric branch is indeed blind here: angular similarity of
    # an orthogonal decoder is flat, so recovery should fail for some group.
    geo = spectral_clusters(angular_similarity(decoder.to(device)), n_groups, seed=0)
    geo_scores = [_group_recovery(geo.to(device), truth, g) for g in range(n_groups)]
    assert min(geo_scores) < 0.9


def test_stability_high_for_planted_low_for_noise(device):
    decoder, truth = _planted_decoder(device)
    sim = angular_similarity(decoder)
    labels = spectral_clusters(sim, N_CLUSTERS, seed=0)
    stab = cluster_stability(sim, labels, n_clusters=N_CLUSTERS, n_runs=3, seed=0)

    fan_ids = {int(labels[(truth >= 0).cpu() & (truth == g).cpu()].mode().values) for g in range(G)}
    fan_stability = min(stab[i] for i in fan_ids)
    assert fan_stability >= 0.7

    gen = torch.Generator().manual_seed(13)
    noise = torch.randn(60, DIM, generator=gen).to(device)
    nsim = angular_similarity(noise)
    nlabels = spectral_clusters(nsim, 10, seed=0)
    nstab = cluster_stability(nsim, nlabels, n_clusters=10, n_runs=3, seed=0)
    assert sum(nstab.values()) / len(nstab) < fan_stability


def test_spectral_deterministic(device):
    decoder, _ = _planted_decoder(device)
    sim = angular_similarity(decoder)
    a = spectral_clusters(sim, N_CLUSTERS, seed=3)
    b = spectral_clusters(sim, N_CLUSTERS, seed=3)
    assert torch.equal(a, b)


def test_rejects_bad_shapes(device):
    with pytest.raises(ValueError):
        spectral_clusters(torch.randn(10, 12, device=device), 3)
    with pytest.raises(ValueError):
        coactivation_similarity(torch.zeros(10, device=device))
