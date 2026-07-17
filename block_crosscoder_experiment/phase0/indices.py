"""Engels irreducibility indices (arXiv 2405.14860, App. B.2), exact recipes.

Two scores over a 2-D point cloud (a cluster's activations projected onto a
PC plane):

- separability index S(f): min over a rotation grid of the mutual
  information between the two axes of the normalized cloud. High S means no
  rotation makes the axes independent — the two dimensions carry joint
  structure (a ring cannot be split).
- ε-mixture index M_ε(f): max over affine functionals (v, c) of the fraction
  of points inside the relative band |v·f + c| < ε·rms(v·f + c). High M
  means the cloud is mostly a "feature off + occasional spike" mixture.

Engels ranks candidate clusters by (1 − M_ε)·S averaged over PC planes; the
weekday/month/year rings rank 9/28/15 of 1000 on GPT-2-small layer 7 —
Phase 0's positive-control target.

Fidelity notes: parameter defaults are the paper's (1000-angle grid, 6×6
clip, 40×40 histogram; GD 10k steps, lr 0.1, T annealed 1→0, ε = 0.1, hard
score at T = 0). The plug-in MI estimator is biased up at finite N exactly
as in the paper — do not "fix" it; comparisons are against nulls run through
the same estimator. Two documented deviations, both conservative: the
rotation grid spans [0, π/2) (MI is invariant under axis swap/negation, so
the paper's unstated 1000-angle range can only be redundant), and the
ε-mixture max is taken over several random restarts (M_ε is defined as a
max; restarts only tighten the estimate toward it).
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "normalize_cloud",
    "separability_index",
    "epsilon_mixture_index",
    "irreducibility_score",
]

_MIN_POINTS = 100


def _check_cloud(points: torch.Tensor, name: str) -> torch.Tensor:
    if points.ndim != 2:
        raise ValueError(f"{name} expects (N, d) points, got {tuple(points.shape)}")
    if points.shape[0] < _MIN_POINTS:
        raise ValueError(
            f"{name} needs >= {_MIN_POINTS} points for a stable estimate, "
            f"got {points.shape[0]}"
        )
    return points.to(torch.float32)


def normalize_cloud(points: torch.Tensor) -> torch.Tensor:
    """Center, scale to unit RMS norm, and (in 2-D) multiply by √2.

    Engels B.2.1: subtract the mean, divide by the root-mean-squared norm,
    ×√d so per-coordinate variance is O(1) against the fixed 6×6 clip box.
    """
    points = points - points.mean(dim=0)
    rms = points.norm(dim=1).pow(2).mean().clamp_min(1e-12).sqrt()
    return points / rms * math.sqrt(points.shape[1])


def separability_index(
    points: torch.Tensor,
    *,
    n_angles: int = 1000,
    clip: float = 3.0,
    n_bins: int = 40,
    angle_chunk: int = 128,
) -> float:
    """Min over a rotation grid of I(a; b) for a 2-D cloud, in nats.

    MI under 2-D rotation is invariant to θ → θ + π/2 (axis swap) and
    θ → θ + π (negation), so the grid covers [0, π/2).
    """
    points = _check_cloud(points, "separability_index")
    if points.shape[1] != 2:
        raise ValueError("separability_index is defined on 2-D clouds")
    points = normalize_cloud(points)
    device = points.device
    n = points.shape[0]

    angles = torch.arange(n_angles, device=device) * (math.pi / 2) / n_angles
    best = torch.tensor(float("inf"), device=device)
    edges = 2 * clip / n_bins
    for start in range(0, n_angles, angle_chunk):
        theta = angles[start : start + angle_chunk]
        cos, sin = torch.cos(theta), torch.sin(theta)
        rot = torch.stack(
            [torch.stack([cos, -sin], -1), torch.stack([sin, cos], -1)], -2
        )  # (A, 2, 2)
        rotated = torch.einsum("aij,nj->ani", rot, points)  # (A, N, 2)
        idx = ((rotated.clamp(-clip, clip - 1e-6) + clip) / edges).long()
        flat = idx[..., 0] * n_bins + idx[..., 1]  # (A, N)
        a_count = theta.shape[0]
        offset = torch.arange(a_count, device=device).unsqueeze(1) * n_bins * n_bins
        hist = torch.bincount(
            (flat + offset).reshape(-1), minlength=a_count * n_bins * n_bins
        ).reshape(a_count, n_bins, n_bins)
        p_ab = hist.float() / n
        p_a = p_ab.sum(dim=2, keepdim=True)
        p_b = p_ab.sum(dim=1, keepdim=True)
        ratio = p_ab / (p_a * p_b).clamp_min(1e-12)
        mi = torch.where(
            p_ab > 0, p_ab * ratio.clamp_min(1e-12).log(), p_ab.new_zeros(())
        ).sum(dim=(1, 2))
        best = torch.minimum(best, mi.min())
    return float(best)


def epsilon_mixture_index(
    points: torch.Tensor,
    *,
    epsilon: float = 0.1,
    steps: int = 10_000,
    lr: float = 0.1,
    restarts: int = 4,
    seed: int = 0,
) -> float:
    """Max fraction of points in a relative ε-band, Engels B.2.2.

    Full-batch GD on the sigmoid relaxation
    E[σ((ε − |v·f + c| / rms(v·f + c)) / T)] with T annealed linearly 1 → 0,
    then scored hard at T = 0. Restarts run as one batched optimization; the
    returned value is the max over restarts.
    """
    points = _check_cloud(points, "epsilon_mixture_index")
    points = normalize_cloud(points)
    device, d = points.device, points.shape[1]

    gen = torch.Generator(device="cpu").manual_seed(seed)
    v = torch.randn(restarts, d, generator=gen).to(device)
    v = v / v.norm(dim=1, keepdim=True)
    c = torch.zeros(restarts, device=device)
    v.requires_grad_(True)
    c.requires_grad_(True)

    def band_ratio() -> torch.Tensor:
        z = points @ v.T + c  # (N, R)
        rms = z.pow(2).mean(dim=0).clamp_min(1e-12).sqrt()
        return z.abs() / rms

    opt = torch.optim.SGD([v, c], lr=lr)
    for step in range(steps):
        temp = max(1.0 - step / steps, 1.0 / steps)
        loss = -torch.sigmoid((epsilon - band_ratio()) / temp).mean(dim=0).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        hard = (band_ratio() < epsilon).float().mean(dim=0)
    return float(hard.max())


def irreducibility_score(
    points: torch.Tensor,
    *,
    epsilon: float = 0.1,
    mixture_steps: int = 10_000,
    restarts: int = 4,
    seed: int = 0,
) -> dict[str, float]:
    """Engels ranking metric for one 2-D cloud: (1 − M_ε)·S plus components."""
    s = separability_index(points)
    m = epsilon_mixture_index(
        points, epsilon=epsilon, steps=mixture_steps, restarts=restarts, seed=seed
    )
    return {"separability": s, "mixture": m, "score": (1.0 - m) * s}
