"""Battery integration on planted codes: recover the ring, pass the nulls.

The Phase −1 gate discipline, applied to the Phase-0 instruments: a planted
heptagon ring (SAE-like sparse nonneg codes over a tiled plane, with an
intensity cone) must light up every ring test; a matched control cluster
(same class→feature firing structure, arbitrary decoder geometry) must not;
and the label-free scan + BH must flag only the ring.
"""

import math

import torch

from block_crosscoder_experiment.discovery.battery import (
    cluster_restricted_reconstruction,
    run_cluster_battery,
    unknown_cluster_scan,
)
from block_crosscoder_experiment.discovery.nulls import benjamini_hochberg
from block_crosscoder_experiment.discovery.rings import angle_harmonic_power

N_CLASSES = 7
D, F = 32, 40
RING = torch.arange(0, 7)
CONTROL = torch.arange(7, 14)
T_RING, T_CONTROL, T_NOISE = 3000, 1500, 1500


def _fixture(device: torch.device):
    """Codes (T, F), decoder (F, D), per-token class ids for both clusters."""
    gen = torch.Generator().manual_seed(21)
    basis = torch.linalg.qr(torch.randn(D, 2, generator=gen)).Q  # ring plane

    theta = torch.arange(N_CLASSES) * (2 * math.pi / N_CLASSES)
    ring_rows = torch.stack([theta.cos(), theta.sin()], dim=1) @ basis.T
    control_rows = torch.randn(len(CONTROL), D, generator=gen)
    control_rows /= control_rows.norm(dim=1, keepdim=True)
    noise_rows = torch.randn(F - 14, D, generator=gen)
    noise_rows /= noise_rows.norm(dim=1, keepdim=True)
    decoder = torch.cat([ring_rows, control_rows, noise_rows])

    T = T_RING + T_CONTROL + T_NOISE
    codes = torch.zeros(T, F)
    ring_ids = torch.full((T,), -1, dtype=torch.long)
    control_ids = torch.full((T,), -1, dtype=torch.long)

    # Ring tokens: class angle + jitter, intensity cone, SAE-like tiling —
    # only the 1–2 nearest dictionary elements fire (nonneg codes).
    cls = torch.arange(T_RING) % N_CLASSES
    phi = theta[cls] + 0.1 * torch.randn(T_RING, generator=gen)
    intensity = 0.5 + 1.5 * torch.rand(T_RING, 1, generator=gen)
    cos_gap = math.cos(2 * math.pi / N_CLASSES)
    acts = ((phi.unsqueeze(1) - theta.unsqueeze(0)).cos() - cos_gap) / (1 - cos_gap)
    codes[:T_RING, RING] = intensity * acts.clamp_min(0.0)
    ring_ids[:T_RING] = cls

    # Control tokens: same class→feature structure, no ring geometry.
    ccls = torch.arange(T_CONTROL) % N_CLASSES
    rows = torch.arange(T_RING, T_RING + T_CONTROL)
    codes[rows, CONTROL[ccls]] = 0.5 + torch.rand(T_CONTROL, generator=gen)
    control_ids[rows] = ccls

    # Background: independent sparse firing across all tokens.
    bg = (torch.rand(T, F - 14, generator=gen) < 0.05).float()
    codes[:, 14:] = bg * (0.5 + torch.rand(T, F - 14, generator=gen))

    return codes.to(device), decoder.to(device), ring_ids.to(device), control_ids.to(device)


def test_discard_rule(device):
    codes, decoder, _, _ = _fixture(device)
    recon, kept = cluster_restricted_reconstruction(codes, decoder, RING.to(device))
    assert kept.shape[0] == T_RING  # every ring token fires >= 1 member
    assert int(kept.max()) < T_RING
    assert recon.shape == (T_RING, D)

    _, capped = cluster_restricted_reconstruction(
        codes, decoder, RING.to(device), max_tokens=500
    )
    assert capped.shape[0] == 500


def test_battery_confirms_planted_ring(device):
    codes, decoder, ring_ids, _ = _fixture(device)
    out = run_cluster_battery(
        codes,
        decoder,
        RING.to(device),
        class_ids=ring_ids,
        n_classes=N_CLASSES,
        mixture_steps=1500,
        n_perm=100,
    )
    assert out["n_tokens"] == T_RING
    assert out["circular"] > 0.8
    assert out["circular_p"] < 0.02
    assert out["ngon"]["alignment"] > 0.8
    assert out["ngon"]["classes_present"] == N_CLASSES
    best = out["plane_scan"]["planes"][out["plane_scan"]["best_plane"]]
    assert best["score"] > 0.3
    assert max(out["harmonics"], key=out["harmonics"].get) == N_CLASSES


def test_battery_rejects_control_cluster(device):
    codes, decoder, _, control_ids = _fixture(device)
    out = run_cluster_battery(
        codes,
        decoder,
        CONTROL.to(device),
        class_ids=control_ids,
        n_classes=N_CLASSES,
        mixture_steps=1500,
        n_perm=100,
    )
    assert out["n_tokens"] == T_CONTROL
    # The observed score can be legitimately medium for consistent class
    # clumps — the class-permutation p-value is the verdict, not the score.
    assert out["circular_p"] > 0.05


def test_unknown_scan_flags_only_ring(device):
    codes, decoder, _, _ = _fixture(device)
    gen = torch.Generator().manual_seed(33)
    clusters = {
        0: RING.to(device),
        1: CONTROL.to(device),
        2: (14 + torch.randperm(F - 14, generator=gen)[:7]).to(device),
    }
    # Null resolution must beat the BH threshold at this search width:
    # rank-1 rejection at width 3 needs p <= 0.05/3, i.e. > 60 null draws.
    results = unknown_cluster_scan(
        codes, decoder, clusters, n_null_draws=100, mixture_steps=400
    )
    pvals = [results[c]["p"] for c in sorted(clusters)]
    rejected = benjamini_hochberg(pvals, alpha=0.05)
    assert bool(rejected[0]), f"ring not flagged: {results}"
    assert not bool(rejected[1]) and not bool(rejected[2]), f"false flag: {results}"


def test_harmonic_power_isotropic_is_flat(device):
    gen = torch.Generator().manual_seed(5)
    cloud = torch.randn(5000, 2, generator=gen).to(device)
    power = angle_harmonic_power(cloud)
    assert max(power.values()) < 0.1


def test_benjamini_hochberg_basics():
    # Strong signals among uniforms are kept, uniforms alone are not.
    p = [0.0001, 0.0002, 0.6, 0.4, 0.9, 0.25, 0.7]
    mask = benjamini_hochberg(p, alpha=0.05)
    assert mask.tolist()[:2] == [True, True] and mask.sum() == 2
    assert not benjamini_hochberg([0.2, 0.8, 0.5, 0.9], alpha=0.05).any()
