"""Scientific contracts for trained shared-code and used-span endpoints."""

from __future__ import annotations

import numpy as np
import torch

from block_crosscoder_experiment.analysis.extract_geometry import (
    numerical_rank,
    pair_principal_cos,
    spectral_summaries,
    svd_chunked,
)
from block_crosscoder_experiment.analysis.fig_geometry import used_basis
from block_crosscoder_experiment.analysis.trained_endpoints import (
    _site_full_views,
    accumulate_code_moments,
    affine_r2,
    canonical_correlations,
    fit_affine_maps,
    fit_procrustes_maps,
    reconstruction_endpoints,
    sparse_decode,
)
from block_crosscoder_experiment.model import BSCConfig, BlockCrosscoder


def moments(x: torch.Tensor, y: torch.Tensor):
    count = torch.tensor(float(x.shape[0]), dtype=torch.float64)
    x, y = x.double(), y.double()
    return (
        count,
        x.sum(0),
        y.sum(0),
        x.T @ x,
        x.T @ y,
        y.T @ y,
    )


def test_affine_code_map_is_fit_on_calibration_and_scored_on_eval():
    gen = torch.Generator().manual_seed(3)
    A0 = torch.tensor([[1.5, -0.2], [0.3, 0.8]], dtype=torch.float64)
    t0 = torch.tensor([0.7, -1.1], dtype=torch.float64)
    x_cal = torch.randn(4000, 2, generator=gen, dtype=torch.float64)
    y_cal = x_cal @ A0 + t0
    A, t, valid = fit_affine_maps(*moments(x_cal, y_cal), min_count=100)
    assert valid
    assert torch.allclose(A, A0, atol=2e-5)
    assert torch.allclose(t, t0, atol=2e-5)

    x_eval = torch.randn(2000, 2, generator=gen, dtype=torch.float64)
    y_eval = x_eval @ A0 + t0
    assert affine_r2(*moments(x_eval, y_eval), A, t) > 1 - 1e-9


def test_procrustes_and_cca_are_rank_aware():
    gen = torch.Generator().manual_seed(4)
    x = torch.randn(5000, 2, generator=gen, dtype=torch.float64)
    q, _ = torch.linalg.qr(torch.randn(2, 2, generator=gen, dtype=torch.float64))
    y = x @ q + torch.tensor([2.0, -0.5])
    R, t, valid = fit_procrustes_maps(*moments(x, y), min_count=100)
    assert valid
    assert affine_r2(*moments(x, y), R, t) > 1 - 1e-9
    cca = canonical_correlations(*moments(x, y))
    assert torch.allclose(cca, torch.ones(2, dtype=torch.float64), atol=1e-6)

    # A truly rank-one code has one correlation, not a fabricated second one.
    xr = torch.stack([x[:, 0], torch.zeros_like(x[:, 0])], 1)
    yr = torch.stack([2 * x[:, 0], torch.zeros_like(x[:, 0])], 1)
    cca_rank1 = canonical_correlations(*moments(xr, yr))
    assert abs(float(cca_rank1[0]) - 1) < 1e-6
    assert torch.isnan(cca_rank1[1])


def test_principal_cosines_do_not_complete_rank_deficient_frames():
    factors = torch.zeros(2, 1, 2, 4)
    factors[0, 0, 0, 0] = 1
    factors[1, 0, 0, 0] = 1
    # Parked decoder directions differ, but have zero code variance and must
    # never appear in the empirical used span.
    values, bases = svd_chunked(factors)
    ranks = numerical_rank(values)
    cosines, _ = pair_principal_cos(bases, ranks)
    assert ranks.tolist() == [[1], [1]]
    assert abs(float(cosines[0, 0, 0]) - 1) < 1e-6
    assert np.isnan(cosines[0, 0, 1])


def test_spectral_summaries_make_zero_energy_rank_zero():
    pr, rank95 = spectral_summaries(
        torch.tensor([[2.0, 2.0, 0.0], [0.0, 0.0, 0.0]])
    )
    assert torch.allclose(pr, torch.tensor([2.0, 0.0]))
    assert rank95.tolist() == [2, 0]


def test_figure_used_basis_excludes_parked_capacity():
    frame = np.zeros((2, 5))
    frame[0, 0] = 1
    frame[1, 4] = 1
    covariance = np.diag([2.0, 0.0])
    basis = used_basis(frame, covariance)
    assert basis.shape == (5, 1)
    assert abs(abs(basis[0, 0]) - 1) < 1e-8
    assert abs(basis[4, 0]) < 1e-8


def test_full_rank_truncation_recovers_the_unablated_model():
    model = BlockCrosscoder(
        BSCConfig(n_blocks=3, block_dim=2, n_sites=2, d_model=5, k=1),
    )
    model.theta.fill_(-1)  # every block active: deterministic support contract
    gen = torch.Generator().manual_seed(7)
    calibration = torch.randn(128, 2, 5, generator=gen)
    cal = accumulate_code_moments(model, [calibration])
    maps, offsets, _ = fit_affine_maps(*_site_full_views(cal), min_count=2)
    result, eval_moments = reconstruction_endpoints(
        model,
        [torch.randn(64, 2, 5, generator=gen)],
        maps,
        offsets,
        cal,
    )
    assert eval_moments.n_tokens == 64
    assert result["single_site_fvu"].shape == (2, 2)
    assert result["leave_one_out_fvu"].shape == (2, 2)
    assert torch.allclose(
        result["truncation_second_fvu"][-1], result["full_fvu"], atol=1e-7
    )
    assert torch.allclose(
        result["truncation_centered_fvu"][-1], result["full_fvu"], atol=1e-7
    )


def test_sparse_endpoint_decode_is_exactly_the_model_decoder():
    model = BlockCrosscoder(
        BSCConfig(n_blocks=5, block_dim=2, n_sites=3, d_model=7, k=1),
    )
    gen = torch.Generator().manual_seed(9)
    code = torch.randn(11, 5, 2, generator=gen)
    mask = torch.rand(11, 5, generator=gen) > 0.65
    selected = code * mask.unsqueeze(-1)
    assert torch.allclose(
        sparse_decode(model, code, mask, event_chunk=3),
        model.decode(selected),
        atol=1e-6,
    )
