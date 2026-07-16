"""Metrics checks: gauge invariance of the primitives (the O(b) rotation
must be invisible to every readout), Procrustes exactness, and an
end-to-end trained recovery — the pipeline finds what was planted."""

import math

import torch

from block_crosscoder_experiment.metrics import (
    block_site_spans,
    energy_rank,
    evaluate_recovery,
    norm_cv,
    participation_ratio,
    procrustes,
    subspace_overlap,
)
from block_crosscoder_experiment.model import BlockCrosscoder, BSCConfig
from block_crosscoder_experiment.synthetic import BlockSpec, PlantedModel
from block_crosscoder_experiment.trainer import TrainConfig, Trainer

S, B_DIM, D_MODEL = 4, 4, 32


def random_orthogonal(n, seed=0):
    gen = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(n, n, generator=gen))
    return q * torch.sign(torch.diagonal(r)).unsqueeze(0)


def test_spans_and_spectra_gauge_invariant():
    """A joint O(b) on (D_g, z_g) — the exact residual gauge — leaves
    used spans and contribution spectra unchanged."""
    gen = torch.Generator().manual_seed(0)
    truth = PlantedModel(
        [BlockSpec(rank=2, frequency=0.5, geometry="shell")],
        n_sites=S,
        d_model=D_MODEL,
        block_dim=B_DIM,
    )
    z = truth.sample(4000, seed=1)
    z_active = z.z[z.active[:, 0], 0]
    R = random_orthogonal(B_DIM, seed=2)
    D_rot = torch.einsum("bc,scd->sbd", R, truth.D[:, 0])
    z_rot = z_active @ R.T

    spans, spectra = block_site_spans(truth.D[:, 0], z_active)
    spans_r, spectra_r = block_site_spans(D_rot, z_rot)
    assert torch.allclose(spectra, spectra_r, atol=1e-4)
    for s in range(S):
        ov, ang = subspace_overlap(spans[s, :, :2], spans_r[s, :, :2])
        assert ov > 1 - 1e-5
        assert float(ang.max()) < 1e-2


def test_procrustes_recovers_planted_rotation():
    gen = torch.Generator().manual_seed(3)
    z_ref = torch.randn(2000, B_DIM, generator=gen)
    R0 = random_orthogonal(B_DIM, seed=4)
    R, r2, scale = procrustes(z_ref @ R0.T, z_ref)
    assert torch.allclose(R, R0, atol=1e-5)
    assert r2 > 1 - 1e-6
    assert abs(scale - 1.0) < 1e-6
    # Not fooled by an unrelated code: r2 collapses.
    z_other = torch.randn(2000, B_DIM, generator=gen)
    _, r2_null, _ = procrustes(z_other, z_ref)
    assert r2_null < 0.1


def test_rank_estimators():
    flat2 = torch.tensor([1.0, 1.0, 0.0, 0.0])
    assert abs(participation_ratio(flat2) - 2.0) < 1e-6
    assert energy_rank(flat2) == 2
    spread = torch.tensor([4.0, 2.0, 1.0, 0.5])
    assert energy_rank(spread) == 4
    assert 2.0 < participation_ratio(spread) < 3.0
    assert energy_rank(torch.tensor([1.0, 1e-8, 1e-8, 1e-8])) == 1


def test_norm_cv_separates_geometries():
    gen = torch.Generator().manual_seed(5)
    shell = torch.randn(5000, B_DIM, generator=gen)
    shell = shell / shell.norm(dim=1, keepdim=True)
    gauss = torch.randn(5000, B_DIM, generator=gen)
    assert norm_cv(shell) < 0.01
    assert norm_cv(gauss) > 0.25


def test_end_to_end_recovery(device):
    """The core Phase -1 property in miniature: train on planted data,
    match, align, and verify every planted block is recovered — subspaces,
    codes, ranks, geometry regime.

    Config pinned to a measured full-recovery basin (d=128, learner seed 1;
    calibration 2026-07-16): d >> total planted latent dims keeps
    superposition crosstalk low, budget k*B matches E[active] exactly, and
    G/F = 1.6. Multi-seed regime statistics (capture vs tiling vs mixing)
    are the battery's job, not a unit test's."""
    specs = [
        BlockSpec(rank=1, frequency=0.25, scale=2.0),
        BlockSpec(rank=2, frequency=0.25, spectrum=(2.4, 1.6)),
        BlockSpec(rank=2, frequency=0.25, geometry="shell", scale=2.0),
        BlockSpec(rank=3, frequency=0.25, spectrum=(2.0, 1.2, 0.8)),
        BlockSpec(rank=4, frequency=0.25, spectrum=(1.6, 1.2, 0.8, 0.6)),
    ]
    truth = PlantedModel(
        specs, n_sites=S, d_model=128, block_dim=B_DIM, noise_std=0.02, seed=5
    )
    cfg = BSCConfig(
        n_blocks=8, block_dim=B_DIM, n_sites=S, d_model=128, k=1, seed=1
    )
    learner = BlockCrosscoder(cfg).to(device)
    learner.calibrate_encoder_scale_(truth.sample(4096, seed=11).x.to(device))
    trainer = Trainer(
        learner,
        TrainConfig(
            total_steps=2000, lr=3e-3, warmup_steps=20, forward_dtype="fp32",
            optimizer="adamw", aux_variant="sasa", s_aux=2,
            dead_window_batches=10, log_every=500,
        ),
    )
    trainer.fit(truth.batches(1024, 2000, seed=7))
    report = evaluate_recovery(truth, learner, n_eval=32768, seed=99)

    assert len(report.blocks) == 5
    for rec in report.blocks:
        assert rec.matched is not None, rec.planted
        assert rec.overlap > 0.9, (rec.planted, rec.overlap)
        assert rec.code_r2 > 0.8, (rec.planted, rec.code_r2)
        assert rec.share_error < 0.1
        assert abs(rec.rank_95 - rec.rank) <= 1.0, (rec.planted, rec.rank_95)
        assert rec.capture_r2 > 0.75, (rec.planted, rec.capture_r2)
    by_g = {r.planted: r for r in report.blocks}
    # Geometry regime survives recovery: the hollow ring is norm-
    # concentrated, gaussian blocks are not.
    assert by_g[2].norm_cv_learned < 0.1
    assert by_g[1].norm_cv_learned > 0.2
    assert by_g[4].norm_cv_learned > 0.2
