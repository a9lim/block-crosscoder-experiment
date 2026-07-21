"""Executable contracts for the paper-method and frontier variant matrix.

These tests intentionally distinguish structural method guarantees from mere
configuration reachability.  They are CPU-only so every matrix change is
checked before an accelerator campaign can make an invalid recipe expensive.
"""

from dataclasses import asdict
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from block_crosscoder_experiment.codec import (
    CodecSpec,
    decode_batch,
    encode_batch,
    fit_codec,
)
from block_crosscoder_experiment.gram import gram_residual
from block_crosscoder_experiment.model import BSCConfig, BlockCrosscoder, bsc_loss
from block_crosscoder_experiment.trainer import (
    TrainConfig,
    Trainer,
    _lr_factor,
    aux_loss,
    build_optimizer,
)


def _model_cfg(**overrides) -> BSCConfig:
    base = dict(
        n_blocks=4,
        block_dim=2,
        n_sites=2,
        d_model=4,
        k=4,
        seed=17,
        decoder_constraint="free",
    )
    return BSCConfig(**{**base, **overrides})


def _train_cfg(**overrides) -> TrainConfig:
    base = dict(
        total_steps=8,
        lr=1e-3,
        warmup_steps=0,
        schedule="constant",
        optimizer="adamw",
        forward_dtype="fp32",
        aux_variant="none",
        log_every=1,
    )
    return TrainConfig(**{**base, **overrides})


def _assert_exact_zero(tensor: torch.Tensor) -> None:
    assert torch.count_nonzero(tensor).item() == 0


@pytest.mark.parametrize(
    "overrides",
    [
        {"site_dims": (2, 4)},
        {"identical_site_init": True},
    ],
)
def test_declared_gram_constraint_holds_before_the_first_forward(overrides):
    model = BlockCrosscoder(_model_cfg(decoder_constraint="gram", **overrides))
    assert float(gram_residual(model.decoder_tensor()).max().detach()) < 1e-5


def test_decoder_initialization_preconditioning_is_explicit():
    preconditioned = BlockCrosscoder(
        _model_cfg(
            decoder_constraint="free",
            decoder_init_preconditioning="concatenated_gram_retraction",
            decoder_init_operation_order=(
                "gaussian_precondition_mask_rescale_then_declared_constraint"
            ),
        )
    )
    raw = BlockCrosscoder(
        _model_cfg(
            decoder_constraint="free",
            decoder_init_preconditioning="none",
            decoder_init_operation_order=(
                "gaussian_mask_rescale_then_declared_constraint"
            ),
        )
    )
    assert float(gram_residual(preconditioned.D).max().detach()) < 1e-5
    assert float(gram_residual(raw.D).max().detach()) > 1e-2
    with pytest.raises(ValueError, match="operation order"):
        _model_cfg(
            decoder_init_preconditioning="none",
            decoder_init_operation_order=(
                "gaussian_precondition_mask_rescale_then_declared_constraint"
            ),
        )


def test_mean_fusion_divides_by_observed_sites_not_configured_sites():
    model = BlockCrosscoder(
        _model_cfg(
            n_blocks=1,
            block_dim=1,
            d_model=1,
            site_dims=(1, 1),
            k=1,
            encoder_fusion="mean",
        )
    )
    with torch.no_grad():
        model.E.fill_(1.0)
    x = torch.tensor([[[2.0], [100.0]]])
    observed = torch.tensor([[True, False]])
    assert torch.equal(model.encode(x, observed=observed), torch.tensor([[[2.0]]]))
    assert torch.equal(model.encode(x), torch.tensor([[[51.0]]]))


def test_availability_rescaled_sum_matches_full_sum_and_rescales_visible_sites():
    summed = BlockCrosscoder(
        _model_cfg(
            n_blocks=2,
            block_dim=1,
            d_model=1,
            site_dims=(1, 1),
            k=2,
            encoder_fusion="sum",
        )
    )
    rescaled = BlockCrosscoder(
        _model_cfg(
            n_blocks=2,
            block_dim=1,
            d_model=1,
            site_dims=(1, 1),
            k=2,
            encoder_fusion="availability_rescaled_sum",
        )
    )
    with torch.no_grad():
        summed.E.copy_(torch.tensor([[[[1.0]], [[2.0]]], [[[3.0]], [[4.0]]]]))
    rescaled.load_state_dict(summed.state_dict())
    x = torch.tensor([[[2.0], [5.0]], [[7.0], [11.0]]])

    # S / n_visible is exactly one on complete rows, so this is bit-exact.
    assert torch.equal(rescaled.encode(x), summed.encode(x))

    observed = torch.tensor([[True, False], [False, True]])
    expected = torch.tensor([[[4.0], [8.0]], [[66.0], [88.0]]])
    assert torch.equal(rescaled.encode(x, observed=observed), expected)
    with pytest.raises(ValueError, match="at least one site"):
        rescaled.encode(x, observed=torch.zeros_like(observed))


def test_removed_site_profile_regularizer_fails_closed():
    with pytest.raises(ValueError, match="unknown regularizer"):
        _model_cfg(regularizer="site_profile", lambda_regularizer=1.0)
    with pytest.raises(ValueError, match="explicit supported regularizer"):
        _model_cfg(lambda_regularizer=1.0)


def test_preencoder_decoder_bias_centering_matches_sasa_release():
    cfg = BSCConfig(
        1,
        1,
        1,
        3,
        1,
        decoder_constraint="free",
        decoder_bias=True,
        apply_decoder_bias_to_input=True,
    )
    model = BlockCrosscoder(cfg)
    with torch.no_grad():
        assert model.E is not None
        model.E.fill_(1.0)
        model.c.copy_(torch.tensor([[1.0, 2.0, 3.0]]))
    centered = model.encode(torch.tensor([[[2.0, 4.0, 6.0]]]))
    assert torch.equal(centered, torch.tensor([[[6.0]]]))
    uncentered_cfg = BSCConfig(
        1, 1, 1, 3, 1, decoder_constraint="free", decoder_bias=True
    )
    uncentered = BlockCrosscoder(uncentered_cfg)
    with torch.no_grad():
        assert uncentered.E is not None
        uncentered.E.fill_(1.0)
        uncentered.c.copy_(model.c)
    assert torch.equal(
        uncentered.encode(torch.tensor([[[2.0, 4.0, 6.0]]])),
        torch.tensor([[[12.0]]]),
    )


def test_geometric_median_bias_and_independent_encoder_init_are_executable():
    model = BlockCrosscoder(
        _model_cfg(
            d_model=1,
            site_dims=(1, 1),
            decoder_bias_init="geometric_median",
            encoder_init="independent",
        )
    )
    assert not torch.equal(model.E, model.D)
    x = torch.tensor([[[0.0], [0.0]], [[0.0], [0.0]], [[100.0], [100.0]]])
    model.initialize_decoder_bias_(x)
    assert float(model.c.abs().max().detach()) < 1e-3


def test_site_observation_mask_excludes_only_the_named_loss_terms():
    model = BlockCrosscoder(
        _model_cfg(
            n_blocks=1,
            block_dim=1,
            d_model=1,
            site_dims=(1, 1),
            k=1,
            reconstruction_loss="squared_l2",
        )
    )
    with torch.no_grad():
        model.D.zero_()
        model.E.zero_()
        model.c.zero_()
    x = torch.tensor([[[100.0], [2.0]], [[100.0], [3.0]]])
    out = model(x)
    all_loss = bsc_loss(out, x, model)["rec"]
    masked_loss = bsc_loss(
        out,
        x,
        model,
        observation_mask=torch.tensor([[False, True], [False, True]]),
    )["rec"]
    assert masked_loss == torch.tensor((4.0 + 9.0) / 2)
    assert masked_loss < all_loss


def test_unit_frobenius_release_scale_control_expands_and_contracts_blocks():
    model = BlockCrosscoder(_model_cfg(decoder_constraint="unit_frobenius"))
    with torch.no_grad():
        model.D[0, 0].mul_(0.1)
        model.D[0, 1].mul_(10.0)
        model.project_decoder_()
    norms = model.decoder_tensor().float().pow(2).sum(dim=(0, 2, 3)).sqrt()
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)


def test_rectangular_sites_ignore_padding_and_have_zero_padded_gradients():
    """Padding is storage only: it cannot affect codes, decodes, or updates."""
    model = BlockCrosscoder(_model_cfg(site_dims=(2, 4), decoder_bias=True))
    gen = torch.Generator().manual_seed(2)
    x = torch.randn(7, 2, 4, generator=gen)
    perturbed = x.clone()
    perturbed[:, 0, 2:] += 10_000 * torch.randn(7, 2, generator=gen)

    assert torch.equal(model.encode(x), model.encode(perturbed))
    out = model(x)
    out_perturbed = model(perturbed)
    assert torch.equal(out.xhat, out_perturbed.xhat)
    _assert_exact_zero(out.xhat[:, 0, 2:])

    encoder_mask = model.coordinate_mask.expand_as(model.E)
    decoder_mask = model.coordinate_mask.expand_as(model.D)
    _assert_exact_zero(model.encoder_tensor()[~encoder_mask])
    _assert_exact_zero(model.decoder_tensor()[~decoder_mask])

    bsc_loss(out, x, model)["total"].backward()
    assert model.E.grad is not None and model.D.grad is not None
    _assert_exact_zero(model.E.grad[~encoder_mask])
    _assert_exact_zero(model.D.grad[~decoder_mask])
    _assert_exact_zero(model.c.grad[~model.coordinate_mask[:, 0, 0]])
    assert model.E.grad[encoder_mask].abs().sum() > 0
    assert model.D.grad[decoder_mask].abs().sum() > 0


def test_sum_mean_and_source_encoder_fusion_are_distinct_exact_maps():
    common = _model_cfg(
        n_blocks=2,
        block_dim=1,
        d_model=2,
        k=2,
        decoder_bias=False,
    )
    sum_model = BlockCrosscoder(common)
    with torch.no_grad():
        sum_model.E.zero_()
        sum_model.E[0, :, 0, 0] = torch.tensor([1.0, 2.0])
        sum_model.E[1, :, 0, 0] = torch.tensor([3.0, 4.0])

    mean_model = BlockCrosscoder(
        _model_cfg(**{**asdict(common), "encoder_fusion": "mean"})
    )
    source_model = BlockCrosscoder(
        _model_cfg(**{**asdict(common), "encoder_fusion": "source", "source_site": 1})
    )
    mean_model.load_state_dict(sum_model.state_dict())
    source_model.load_state_dict(sum_model.state_dict())

    x = torch.tensor([[[2.0, 9.0], [5.0, -7.0]], [[-1.0, 2.0], [4.0, 3.0]]])
    per_site = torch.einsum("bsd,sgkd->bsgk", x, sum_model.E)
    expected_sum = per_site.sum(dim=1)
    assert torch.equal(sum_model.encode(x), expected_sum)
    assert torch.equal(mean_model.encode(x), expected_sum / 2)
    assert torch.equal(source_model.encode(x), per_site[:, 1])

    changed_non_source = x.clone()
    changed_non_source[:, 0] += 1000
    assert torch.equal(source_model.encode(x), source_model.encode(changed_non_source))


def test_mean_squared_and_mean_l2_reconstruction_forks_match_their_definitions():
    mean_squared = BlockCrosscoder(
        _model_cfg(
            n_blocks=2,
            block_dim=1,
            d_model=3,
            site_dims=(2, 3),
            k=2,
            decoder_bias=False,
            reconstruction_loss="mean_squared",
        )
    )
    mean_l2 = BlockCrosscoder(
        _model_cfg(
            n_blocks=2,
            block_dim=1,
            d_model=3,
            site_dims=(2, 3),
            k=2,
            decoder_bias=False,
            reconstruction_loss="mean_l2",
        )
    )
    mean_l2.load_state_dict(mean_squared.state_dict())
    with torch.no_grad():
        mean_squared.E.zero_()
        mean_l2.E.zero_()
    x = torch.tensor(
        [
            [[3.0, 4.0, 999.0], [0.0, 0.0, 12.0]],
            [[0.0, 5.0, -999.0], [8.0, 15.0, 0.0]],
        ]
    )
    masked = x * mean_squared.coordinate_mask[:, 0, 0]
    squared_expected = masked.square().sum() / (x.shape[0] * 5)
    mean_l2_expected = masked.norm(dim=-1).mean()

    squared_rec = bsc_loss(mean_squared(x), x, mean_squared)["rec"]
    mean_l2_rec = bsc_loss(mean_l2(x), x, mean_l2)["rec"]
    assert torch.equal(squared_rec, squared_expected)
    assert torch.equal(mean_l2_rec, mean_l2_expected)
    assert not torch.isclose(squared_rec, mean_l2_rec)


def test_squared_l2_and_mean_l1_reductions_are_explicit():
    common = dict(
        n_blocks=2,
        block_dim=1,
        d_model=3,
        site_dims=(2, 3),
        k=2,
        decoder_bias=False,
    )
    squared_l2 = BlockCrosscoder(_model_cfg(**common, reconstruction_loss="squared_l2"))
    mean_l1 = BlockCrosscoder(_model_cfg(**common, reconstruction_loss="mean_l1"))
    mean_l1.load_state_dict(squared_l2.state_dict())
    with torch.no_grad():
        squared_l2.E.zero_()
        mean_l1.E.zero_()
    x = torch.tensor(
        [
            [[3.0, 4.0, 999.0], [0.0, 0.0, 12.0]],
            [[0.0, 5.0, -999.0], [8.0, 15.0, 0.0]],
        ]
    )
    masked = x * squared_l2.coordinate_mask[:, 0, 0]
    assert torch.equal(
        bsc_loss(squared_l2(x), x, squared_l2)["rec"],
        masked.square().sum() / x.shape[0],
    )
    assert torch.equal(
        bsc_loss(mean_l1(x), x, mean_l1)["rec"],
        masked.abs().sum(dim=-1).mean(),
    )


def test_sasa_release_unit_latent_constraints_and_decoder_nuclear_loss():
    model = BlockCrosscoder(
        _model_cfg(
            n_sites=1,
            d_model=5,
            site_dims=(5,),
            n_blocks=3,
            block_dim=2,
            k=2,
            encoder_constraint="unit_latent",
            decoder_constraint="unit_latent",
            regularizer="decoder_nuclear",
            lambda_regularizer=3.0,
        )
    )
    assert torch.allclose(model.D.norm(dim=-1), torch.ones(1, 3, 2))
    assert torch.allclose(model.E.norm(dim=-1), torch.ones(1, 3, 2))
    with torch.no_grad():
        model.D.mul_(torch.linspace(0.2, 2.0, 6).view(1, 3, 2, 1))
        model.E.mul_(torch.linspace(2.0, 0.2, 6).view(1, 3, 2, 1))
    model.project_decoder_()
    assert torch.allclose(model.D.norm(dim=-1), torch.ones(1, 3, 2), atol=1e-6)
    assert torch.allclose(model.E.norm(dim=-1), torch.ones(1, 3, 2), atol=1e-6)
    parts = bsc_loss(model(torch.randn(7, 1, 5)), torch.randn(7, 1, 5), model)
    assert parts["regularizer"].isfinite()
    assert parts["regularizer"] > 0


def test_sasa_paper_sum_over_block_maps_has_exact_source_scaling():
    common = dict(
        n_sites=1,
        d_model=5,
        site_dims=(5,),
        n_blocks=3,
        block_dim=2,
        k=2,
        regularizer="map_nuclear",
        lambda_regularizer=1.0,
    )
    normalized = BlockCrosscoder(
        _model_cfg(**common, map_nuclear_reduction="mean_normalized")
    )
    paper_sum = BlockCrosscoder(
        _model_cfg(**common, map_nuclear_reduction="sum_blocks")
    )
    paper_sum.load_state_dict(normalized.state_dict())
    x = torch.randn(7, 1, 5, generator=torch.Generator().manual_seed(808))
    normal_reg = bsc_loss(normalized(x), x, normalized)["regularizer"]
    paper_reg = bsc_loss(paper_sum(x), x, paper_sum)["regularizer"]
    assert torch.allclose(paper_reg, normal_reg * 3 * 2)


def test_qr_and_polar_retractions_are_distinct_stiefel_contracts():
    polar = BlockCrosscoder(_model_cfg(decoder_constraint="gram"))
    qr = BlockCrosscoder(_model_cfg(decoder_constraint="qr"))
    qr.load_state_dict(polar.state_dict())
    with torch.no_grad():
        perturbation = torch.randn_like(polar.D) * 0.3
        polar.D.add_(perturbation)
        qr.D.add_(perturbation)
    polar.project_decoder_()
    qr.project_decoder_()
    eye = torch.eye(polar.cfg.block_dim).expand(polar.cfg.n_blocks, -1, -1)
    polar_gram = torch.einsum("sgbd,sgcd->gbc", polar.D, polar.D)
    qr_gram = torch.einsum("sgbd,sgcd->gbc", qr.D, qr.D)
    assert torch.allclose(polar_gram, eye, atol=1e-5)
    assert torch.allclose(qr_gram, eye, atol=1e-5)
    assert not torch.allclose(polar.D, qr.D)


def test_sasa_release_coordinate_aux_can_select_partial_blocks():
    model = BlockCrosscoder(
        _model_cfg(
            n_sites=1,
            d_model=4,
            site_dims=(4,),
            n_blocks=2,
            block_dim=2,
            k=1,
            decoder_bias=True,
            decoder_constraint="free",
            code_activation="signed",
            selection="token_topk",
        )
    )
    x = torch.randn(5, 1, 4, generator=torch.Generator().manual_seed(991))
    out = model(x)
    loss = aux_loss(
        model,
        x,
        out,
        "sasa_release",
        torch.tensor([[True, False], [False, False]]),
        s_aux=1,
    )
    assert loss is not None and loss.ndim == 0 and torch.isfinite(loss)


def test_bsf_release_shared_group_threshold_is_one_parameter():
    model = BlockCrosscoder(
        _model_cfg(
            n_sites=1,
            site_dims=(5,),
            d_model=5,
            n_blocks=4,
            block_dim=2,
            k=2,
            code_activation="group_soft_threshold",
            selection="dense",
            decoder_constraint="unit_frobenius",
            group_threshold_scope="shared_scalar",
            group_threshold_parameterization="exp",
            group_threshold_raw_init=0.0,
            group_threshold_effective_init=1.0,
        )
    )
    assert model.log_threshold is not None
    assert model.log_threshold.shape == torch.Size([])
    assert float(torch.exp(model.log_threshold).detach()) == pytest.approx(1.0)
    with torch.no_grad():
        model.log_threshold.zero_()
    z = model.encode(torch.randn(7, 1, 5))
    assert z.shape == (7, 4, 2)
    assert torch.isfinite(z).all()


def test_adamw_assigns_decay_by_parameter_role():
    model = BlockCrosscoder(
        _model_cfg(
            n_blocks=3,
            block_dim=1,
            d_model=3,
            k=3,
            code_activation="relu",
            selection="dense",
            encoder_bias=True,
        )
    )
    cfg = _train_cfg(
        optimizer="adamw",
        encoder_weight_decay=0.11,
        decoder_weight_decay=0.22,
        bias_weight_decay=0.33,
    )
    optimizer, kind = build_optimizer(model, cfg)
    assert kind == "adamw"
    assert isinstance(optimizer, torch.optim.AdamW)

    decay_by_parameter = {
        id(parameter): group["weight_decay"]
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    decoder_names = {"D"}
    bias_names = {"a", "c"}
    for name, parameter in model.named_parameters():
        expected = (
            0.22 if name in decoder_names else 0.33 if name in bias_names else 0.11
        )
        assert decay_by_parameter[id(parameter)] == pytest.approx(expected), name

    with pytest.raises(ValueError, match="Adam recipes cannot request"):
        build_optimizer(model, _train_cfg(optimizer="adam", encoder_weight_decay=1e-4))
    adam, resolved = build_optimizer(model, _train_cfg(optimizer="adam"))
    assert resolved == "adam" and isinstance(adam, torch.optim.Adam)


@pytest.mark.parametrize("schedule", ["cosine", "linear_fifth"])
def test_decaying_schedules_reach_the_declared_floor_on_last_optimizer_update(schedule):
    """A total_steps run updates at indices 0..total_steps-1, inclusively."""
    cfg = _train_cfg(
        total_steps=20,
        warmup_steps=4,
        schedule=schedule,
        min_lr_ratio=0.2,
        final_decay_fraction=0.25,
    )
    factor = _lr_factor(cfg)
    assert factor(0) == pytest.approx(0.25)
    assert factor(cfg.warmup_steps - 1) == pytest.approx(1.0)
    assert factor(cfg.total_steps - 1) == pytest.approx(cfg.min_lr_ratio)


def test_checkpoint_round_trips_variant_configs_and_all_cpu_rng_state(
    tmp_path, monkeypatch
):
    import block_crosscoder_experiment.trainer as trainer_module

    model_cfg = _model_cfg(
        n_blocks=8,
        block_dim=1,
        d_model=4,
        site_dims=(2, 4),
        k=3.5,
        encoder_fusion="source",
        source_site=1,
        code_activation="relu",
        selection="dense",
        reconstruction_loss="mean_l2",
        decoder_norm_geometry="concat_l2",
    )
    train_cfg = _train_cfg(
        total_steps=17,
        warmup_steps=3,
        schedule="linear_fifth",
        min_lr_ratio=0.1,
        final_decay_fraction=0.3,
        encoder_weight_decay=1e-4,
        decoder_weight_decay=2e-4,
        bias_weight_decay=3e-4,
    )
    trainer = Trainer(BlockCrosscoder(model_cfg), train_cfg)
    trainer.master.theta.fill_(0.6789)
    trainer.step_idx = 5
    trainer.accepted_tokens = 123
    trainer.data_cursor = {"shard": 4, "offset": 99}

    random.seed(101)
    np.random.seed(102)
    torch.manual_seed(103)
    monkeypatch.setattr(
        trainer_module.shutil,
        "disk_usage",
        lambda _: SimpleNamespace(total=10**12, used=0, free=10**12),
    )
    checkpoint = tmp_path / "variants.pt"
    trainer.save_checkpoint(checkpoint)
    expected_rng = (random.random(), np.random.random(), torch.rand(5))

    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    restored = Trainer.load_checkpoint(checkpoint)
    actual_rng = (random.random(), np.random.random(), torch.rand(5))

    assert asdict(restored.master.cfg) == asdict(trainer.master.cfg)
    assert asdict(restored.cfg) == asdict(trainer.cfg)
    assert restored.optimizer_kind == trainer.optimizer_kind
    assert restored.step_idx == trainer.step_idx
    assert restored.accepted_tokens == trainer.accepted_tokens
    assert restored.data_cursor == trainer.data_cursor
    assert torch.equal(restored.master.theta, trainer.master.theta)
    assert actual_rng[0] == expected_rng[0]
    assert actual_rng[1] == expected_rng[1]
    assert torch.equal(actual_rng[2], expected_rng[2])


def test_site_axis_factorization_full_rank_reconstructs_the_full_control():
    full_cfg = _model_cfg(
        n_sites=3,
        site_dims=(4, 4, 4),
        d_model=4,
        n_blocks=5,
        block_dim=2,
        decoder_constraint="free",
        encoder_init="independent",
    )
    full = BlockCrosscoder(full_cfg)
    factorized = BlockCrosscoder(_model_cfg(**{**asdict(full_cfg), "site_rank": 3}))

    assert factorized.D is None and factorized.E is None
    assert factorized.D_site is not None and factorized.D_core is not None
    assert factorized.E_site is not None and factorized.E_core is not None
    assert torch.equal(factorized.decoder_tensor(), full.decoder_tensor())
    assert torch.equal(factorized.encoder_tensor(), full.encoder_tensor())

    clone = BlockCrosscoder(factorized.cfg)
    clone.load_state_dict(factorized.state_dict())
    assert torch.equal(clone.decoder_tensor(), factorized.decoder_tensor())
    assert torch.equal(clone.encoder_tensor(), factorized.encoder_tensor())


def test_low_rank_site_axis_parameters_have_the_declared_shapes_and_rank():
    cfg = _model_cfg(
        n_sites=4,
        site_dims=(3, 3, 3, 3),
        d_model=3,
        n_blocks=5,
        block_dim=2,
        decoder_constraint="free",
        encoder_init="independent",
        site_rank=2,
    )
    model = BlockCrosscoder(cfg)
    assert model.D_site is not None and model.D_site.shape == (4, 2)
    assert model.D_core is not None and model.D_core.shape == (2, 5, 2, 3)
    assert model.E_site is not None and model.E_site.shape == (4, 2)
    assert model.E_core is not None and model.E_core.shape == (2, 5, 2, 3)
    expected_factor_parameters = 2 * (4 * 2 + 2 * 5 * 2 * 3)
    actual_factor_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.startswith(("D_", "E_"))
    )
    assert actual_factor_parameters == expected_factor_parameters
    assert torch.linalg.matrix_rank(model.decoder_tensor().reshape(4, -1)) <= 2
    assert torch.linalg.matrix_rank(model.encoder_tensor().reshape(4, -1)) <= 2


def test_factorized_parameter_roles_and_codec_path_are_operational():
    model = BlockCrosscoder(
        _model_cfg(
            n_sites=3,
            site_dims=(4, 4, 4),
            d_model=4,
            n_blocks=5,
            block_dim=2,
            k=2,
            decoder_constraint="free",
            site_rank=1,
        )
    )
    optimizer, _ = build_optimizer(
        model,
        _train_cfg(
            encoder_weight_decay=0.11,
            decoder_weight_decay=0.22,
            bias_weight_decay=0.33,
        ),
    )
    decay = {
        id(parameter): group["weight_decay"]
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    for name, parameter in model.named_parameters():
        if name.startswith("D_"):
            assert decay[id(parameter)] == pytest.approx(0.22)
        elif name in {"a", "c", "log_threshold"}:
            assert decay[id(parameter)] == pytest.approx(0.33)
        else:
            assert decay[id(parameter)] == pytest.approx(0.11)

    x = torch.randn(96, 3, 4, generator=torch.Generator().manual_seed(818))
    model.fit_threshold_([x], target_avg_blocks=2)
    codec = fit_codec(
        model,
        [x[:48], x[48:]],
        CodecSpec(qs=(4,), floor=1, n_bootstrap=2),
    )
    packet = encode_batch(model, codec, x[:9], q=4)
    decoded = decode_batch(model, codec, packet)
    assert decoded.shape == (9, 3, 4)
    assert torch.isfinite(decoded).all()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"site_rank": 0}, "site_rank"),
        ({"site_rank": 1.5}, "integer"),
        ({"site_rank": 3}, "site_rank"),
        ({"site_rank": 1, "site_dims": (2, 4)}, "equal site widths"),
        ({"site_rank": 1, "encoder_mode": "tied"}, "untied encoder"),
        ({"site_rank": 1, "decoder_constraint": "gram"}, "free decoder"),
        (
            {"site_rank": 1, "encoder_constraint": "unit_latent"},
            "encoder_constraint='none'",
        ),
    ],
)
def test_site_axis_factorization_invalid_combinations_fail_closed(overrides, message):
    with pytest.raises(ValueError, match=message):
        _model_cfg(**overrides)


def test_decoded_energy_is_reciprocal_within_block_gauge_invariant():
    common = _model_cfg(
        n_sites=2,
        site_dims=(3, 3),
        d_model=3,
        n_blocks=2,
        block_dim=2,
        k=2,
        decoder_constraint="free",
        decoder_bias=False,
        selection_score="decoded_energy",
    )
    original = BlockCrosscoder(common)
    transformed = BlockCrosscoder(common)
    transformed.load_state_dict(original.state_dict())
    x = torch.randn(17, 2, 3, generator=torch.Generator().manual_seed(244))
    original_code = original.encode(x)

    gauge = torch.tensor([[2.0, 0.0], [0.0, 0.5]])
    inverse_transpose = torch.linalg.inv(gauge).T
    with torch.no_grad():
        assert transformed.E is not None and transformed.D is not None
        transformed.E[0] = torch.einsum("bc,gcd->gbd", gauge, transformed.E[0])
        transformed.E[1] = torch.einsum("bc,gcd->gbd", gauge, transformed.E[1])
        transformed.D[0] = torch.einsum(
            "bc,gcd->gbd", inverse_transpose, transformed.D[0]
        )
        transformed.D[1] = torch.einsum(
            "bc,gcd->gbd", inverse_transpose, transformed.D[1]
        )
    transformed_code = transformed.encode(x)

    assert torch.allclose(
        transformed.scores(transformed_code),
        original.scores(original_code),
        rtol=1e-5,
        atol=1e-6,
    )
    assert not torch.allclose(transformed_code.norm(dim=-1), original_code.norm(dim=-1))
