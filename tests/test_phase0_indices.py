"""Engels indices against synthetic ground truth (design §Phase 0).

The battery must separate irreducible rings from every reducible look-alike
*before* it touches a real SAE — a failure downstream then implicates the
pipeline, not the metrics. Clouds mirror the paper's toy validation
(App. B.2 / Fig. 11): ring, independent Gaussian, independent uniform
square, sparse-spike mixture, 1-D line.
"""

import math

import pytest
import torch

from block_crosscoder_experiment.phase0.indices import (
    epsilon_mixture_index,
    irreducibility_score,
    normalize_cloud,
    separability_index,
)

N = 20_000
# Annealed GD on clean synthetic clouds converges long before the paper's
# 10k steps; tests dial down for speed, defaults stay paper-exact.
FAST_MIXTURE = dict(steps=1500, restarts=4, seed=0)


def _ring(n: int, device: torch.device, noise: float = 0.05) -> torch.Tensor:
    gen = torch.Generator().manual_seed(1)
    theta = torch.rand(n, generator=gen) * 2 * math.pi
    pts = torch.stack([theta.cos(), theta.sin()], dim=1)
    return (pts + noise * torch.randn(n, 2, generator=gen)).to(device)


def _gaussian(n: int, device: torch.device) -> torch.Tensor:
    gen = torch.Generator().manual_seed(2)
    return torch.randn(n, 2, generator=gen).to(device)


def _uniform_square(n: int, device: torch.device) -> torch.Tensor:
    gen = torch.Generator().manual_seed(3)
    return (torch.rand(n, 2, generator=gen) * 2 - 1).to(device)


def _sparse_spike(n: int, device: torch.device) -> torch.Tensor:
    """Feature off 90% of the time, strong activation 10% — a mixture."""
    gen = torch.Generator().manual_seed(4)
    direction = torch.tensor([0.8, 0.6])
    amp = (torch.rand(n, 1, generator=gen) < 0.1).float() * 5.0
    noise = 0.02 * torch.randn(n, 2, generator=gen)
    return (amp * direction + noise).to(device)


def _line(n: int, device: torch.device) -> torch.Tensor:
    gen = torch.Generator().manual_seed(5)
    t = torch.rand(n, 1, generator=gen) * 2 - 1
    direction = torch.tensor([0.6, -0.8])
    return (t * direction + 0.02 * torch.randn(n, 2, generator=gen)).to(device)


def test_normalize_cloud_matches_recipe(device):
    pts = _gaussian(N, device) * 3.7 + torch.tensor([5.0, -2.0], device=device)
    out = normalize_cloud(pts)
    assert torch.allclose(out.mean(dim=0), torch.zeros(2, device=device), atol=1e-4)
    rms = out.norm(dim=1).pow(2).mean().sqrt()
    assert abs(float(rms) - math.sqrt(2)) < 1e-4


def test_separability_low_for_independent_axes(device):
    assert separability_index(_gaussian(N, device)) < 0.1
    assert separability_index(_uniform_square(N, device)) < 0.1


def test_separability_high_for_ring(device):
    assert separability_index(_ring(N, device)) > 0.3


def test_mixture_high_for_sparse_spike(device):
    assert epsilon_mixture_index(_sparse_spike(N, device), **FAST_MIXTURE) > 0.8


def test_mixture_moderate_for_ring(device):
    # Engels Fig. 2: the weekday ring scores M_ε ≈ 0.4 — low relative to
    # typical clusters, not near zero (the band can sit on the ring's edge).
    m = epsilon_mixture_index(_ring(N, device), **FAST_MIXTURE)
    assert m < 0.6


def test_mixture_deterministic(device):
    pts = _ring(N, device)
    a = epsilon_mixture_index(pts, **FAST_MIXTURE)
    b = epsilon_mixture_index(pts, **FAST_MIXTURE)
    assert a == b


def test_ring_tops_irreducibility_ranking(device):
    """The paper's discriminator: rings outrank every reducible cloud."""
    clouds = {
        "ring": _ring(N, device),
        "gaussian": _gaussian(N, device),
        "square": _uniform_square(N, device),
        "spike": _sparse_spike(N, device),
        "line": _line(N, device),
    }
    scores = {
        name: irreducibility_score(pts, mixture_steps=1500)["score"]
        for name, pts in clouds.items()
    }
    ring = scores.pop("ring")
    worst_margin = min(ring - s for s in scores.values())
    assert worst_margin > 0.1, f"ring={ring:.3f}, others={scores}"


def test_rejects_degenerate_input(device):
    with pytest.raises(ValueError):
        separability_index(torch.randn(50, 2, device=device))
    with pytest.raises(ValueError):
        separability_index(torch.randn(500, 3, device=device))
