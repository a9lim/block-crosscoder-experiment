"""Generator checks: the planted truth satisfies the Gram constraint
exactly, realizes its specified depth profiles / frequencies / ranks /
geometries, matches the learner's decode convention, and is learnable
end-to-end by the trainer."""

import torch

from block_crosscoder_experiment.gram import gram_residual, site_frobenius_shares
from block_crosscoder_experiment.model import BlockCrosscoder, BSCConfig
from block_crosscoder_experiment.synthetic import BlockSpec, PlantedModel
from block_crosscoder_experiment.trainer import TrainConfig, Trainer

S, B_DIM, D_MODEL = 4, 4, 32


def spec_zoo():
    return [
        BlockSpec(rank=2, frequency=0.3, geometry="shell"),  # hollow ring
        BlockSpec(rank=2, frequency=0.3, geometry="shell", radial_spread=0.3),
        BlockSpec(rank=4, frequency=0.2, spectrum=(4.0, 2.0, 1.0, 0.5)),
        BlockSpec(rank=1, frequency=0.05, depth_profile=(1.0, 0.0, 0.0, 0.0)),
        BlockSpec(rank=3, frequency=0.1, depth_profile=(0.5, 0.3, 0.2, 0.0)),
    ]


def planted(**overrides):
    kwargs = dict(n_sites=S, d_model=D_MODEL, block_dim=B_DIM, seed=0)
    return PlantedModel(spec_zoo(), **{**kwargs, **overrides})


def test_planted_decoders_satisfy_constraint_exactly():
    model = planted()
    assert float(gram_residual(model.D).max()) < 1e-6


def test_depth_profiles_realized():
    model = planted()
    shares = site_frobenius_shares(model.D)  # [S, G]
    assert torch.allclose(shares[:, 0], torch.full((S,), 0.25), atol=1e-6)
    assert torch.allclose(shares[:, 3], torch.tensor([1.0, 0.0, 0.0, 0.0]), atol=1e-6)
    assert torch.allclose(shares[:, 4], torch.tensor([0.5, 0.3, 0.2, 0.0]), atol=1e-6)


def test_frequencies_realized():
    model = planted()
    batch = model.sample(50_000, seed=1)
    freq = batch.active.float().mean(dim=0)
    expected = torch.tensor([s.frequency for s in model.specs])
    assert (freq - expected).abs().max() < 0.01


def test_planted_rank_lives_in_contributions_not_frames():
    model = planted()
    for g, spec in enumerate(model.specs):
        M = model.planted_second_moment(g, n=5000)
        profile = spec.depth_profile or (1.0 / S,) * S
        for s in range(S):
            if profile[s] == 0.0:
                assert float(M[s].abs().max()) < 1e-8
                continue
            # The frame at every carrying site is full capacity b (parked
            # capacity, unidentifiable)...
            assert int(torch.linalg.matrix_rank(model.D[s, g])) == B_DIM
            # ...while the contribution second moment has exactly rank r.
            evals = torch.linalg.eigvalsh(M[s])
            assert int((evals > 1e-6 * evals.max()).sum()) == spec.rank


def test_hollow_vs_thickened_shells():
    model = planted()
    batch = model.sample(20_000, seed=2)
    norms_hollow = batch.z[batch.active[:, 0], 0].norm(dim=-1)
    norms_thick = batch.z[batch.active[:, 1], 1].norm(dim=-1)
    # Hollow: codes live on the shell (constant norm, up to fp error).
    assert float(norms_hollow.std()) < 1e-5
    assert abs(float(norms_hollow.mean()) - 1.0) < 1e-5
    # Thickened: same geometry, radially filled.
    assert float(norms_thick.std()) > 0.2


def test_bundle_null_gate_coupling():
    bundle = [
        BlockSpec(rank=1, frequency=0.2, gate_group=0, gate_coupling=1.0)
        for _ in range(3)
    ] + [
        BlockSpec(rank=1, frequency=0.2, gate_group=1, gate_coupling=0.0),
        BlockSpec(rank=1, frequency=0.2, gate_group=1, gate_coupling=0.0),
    ]
    model = PlantedModel(bundle, n_sites=S, d_model=D_MODEL, block_dim=B_DIM)
    active = model.sample(50_000, seed=3).active.float()
    # Perfect coupling: identical gates (the weakened null, D11).
    assert torch.equal(active[:, 0], active[:, 1])
    assert torch.equal(active[:, 0], active[:, 2])
    # Zero coupling: independent gates.
    corr = torch.corrcoef(active[:, 3:].T)[0, 1]
    assert abs(float(corr)) < 0.02


def test_decode_convention_matches_learner():
    """The planted truth loaded into a BlockCrosscoder reproduces x — the
    generator and the learner agree on every transpose."""
    model = planted(noise_std=0.0)
    batch = model.sample(256, seed=4)
    cfg = BSCConfig(
        n_blocks=len(model.specs), block_dim=B_DIM, n_sites=S, d_model=D_MODEL, k=3
    )
    learner = BlockCrosscoder(cfg)
    with torch.no_grad():
        learner.D.copy_(model.D)
    xhat = learner.decode(batch.z)
    assert torch.allclose(xhat, batch.x, atol=1e-5)


def test_planted_config_is_learnable(device):
    """End-to-end smoke: an easy planted config trains to low held-out FVU.
    Full recovery scoring (assignment + global Procrustes) is metrics-side.
    """
    specs = [
        BlockSpec(rank=2, frequency=0.25, geometry="shell"),
        BlockSpec(rank=2, frequency=0.25, geometry="shell", radial_spread=0.3),
        BlockSpec(rank=4, frequency=0.25, spectrum=(4.0, 2.0, 1.0, 0.5)),
        BlockSpec(rank=1, frequency=0.25),
    ]
    truth = PlantedModel(
        specs, n_sites=S, d_model=D_MODEL, block_dim=B_DIM, noise_std=0.02, seed=5
    )
    cfg = BSCConfig(n_blocks=8, block_dim=B_DIM, n_sites=S, d_model=D_MODEL, k=2, seed=0)
    learner = BlockCrosscoder(cfg).to(device)
    trainer = Trainer(
        learner,
        TrainConfig(
            total_steps=300,
            lr=3e-3,
            warmup_steps=10,
            forward_dtype="fp32",
            optimizer="adamw",
            aux_variant="sasa",
            s_aux=4,
            dead_window_tokens=10 * 512,
            log_every=50,
        ),
    )
    trainer.fit(truth.batches(512, 300, seed=6))
    held_out = truth.sample(8192, seed=999).x.to(device)
    with torch.no_grad():
        out = learner(held_out)
        fvu = float((out.xhat - held_out).pow(2).mean() / held_out.float().var())
    assert fvu < 0.15
