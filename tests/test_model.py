"""Model-level checks: the selection-score identity, BatchTopK exactness
and gradient masking, init conventions, and a smoke test of the full
step -> retract loop on tiny synthetic data."""

import copy

import pytest
import torch

from block_crosscoder_experiment.gram import gram_residual, retract_
from block_crosscoder_experiment.model import (
    BlockCrosscoder,
    BSCConfig,
    SignedStreamingScoreQuantile,
    batch_topk_mask,
    bsc_loss,
    token_topk_mask,
)

CFG = BSCConfig(n_blocks=16, block_dim=4, n_sites=4, d_model=32, k=3, seed=0)


def make_model(device, **overrides):
    cfg = BSCConfig(**{**CFG.__dict__, **overrides})
    return BlockCrosscoder(cfg).to(device)


def whitened_batch(device, n=512, seed=1):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(n, CFG.n_sites, CFG.d_model, generator=gen).to(device)


def test_selection_score_identity(device):
    """Under the constraint, ||z_g||^2 is exactly the block's contribution
    energy sum_s ||D_g^s^T z_g||^2 — the score is not a proxy."""
    model = make_model(device)
    z = model.encode(whitened_batch(device))  # [B, G, b]
    contrib = torch.einsum("bgk,sgkd->bsgd", z, model.D)
    energy = contrib.pow(2).sum(dim=(1, 3))  # [B, G]
    assert torch.allclose(energy, z.pow(2).sum(dim=-1), rtol=1e-4, atol=1e-5)


def test_encode_matches_naive(device):
    model = make_model(device)
    x = whitened_batch(device, n=8)
    z = model.encode(x)
    naive = torch.einsum("bsd,sgkd->bgk", x, model.E)
    assert torch.allclose(z, naive, rtol=1e-4, atol=1e-5)


def test_decode_matches_naive_and_bias(device):
    model = make_model(device)
    z = torch.randn(8, CFG.n_blocks, CFG.block_dim, device=device)
    xhat = model.decode(z)
    naive = torch.einsum("bgk,sgkd->bsd", z, model.D) + model.c
    assert torch.allclose(xhat, naive, rtol=1e-4, atol=1e-5)
    # Zero code decodes to the bias.
    zero = model.decode(torch.zeros_like(z))
    assert torch.allclose(zero, model.c.expand_as(zero), atol=1e-6)


def test_pre_materialized_structured_weights_preserve_forward(device):
    model = make_model(
        device,
        site_rank=2,
        selection="token_topk",
        decoder_constraint="free",
        decoder_init_preconditioning="none",
        decoder_init_operation_order="gaussian_mask_rescale_then_declared_constraint",
    )
    x = whitened_batch(device, n=8)
    expected = model(x)
    decoder = model.decoder_tensor()
    encoder = model.encoder_tensor()
    actual, returned_decoder, returned_encoder = model.forward_with_materialized(
        x,
        _decoder=decoder,
        _encoder=encoder,
    )
    for expected_tensor, actual_tensor in zip(expected, actual, strict=True):
        assert torch.equal(expected_tensor, actual_tensor)
    assert returned_decoder is decoder
    assert returned_encoder is encoder

    selected, selected_decoder, selected_encoder = model.select_with_materialized(
        x,
        _decoder=decoder,
        _encoder=encoder,
    )
    for expected_tensor, selected_tensor in zip(
        actual[1:], selected, strict=True
    ):
        assert torch.equal(expected_tensor, selected_tensor)
    assert selected_decoder is decoder
    assert selected_encoder is encoder
    assert torch.equal(
        actual.xhat,
        model.decode(selected.z_selected, _decoder=selected_decoder),
    )


@pytest.mark.parametrize(
    "topology",
    ("factorized", "tied", "padded"),
)
@pytest.mark.parametrize(
    "fusion",
    ("sum", "mean", "availability_rescaled_sum", "source"),
)
@pytest.mark.parametrize(
    "activation",
    ("signed", "relu", "group_soft_threshold"),
)
def test_frozen_encoder_sites_preserve_every_view_exactly(
    device, topology, fusion, activation
):
    overrides = {
        "encoder_fusion": fusion,
        "source_site": 1,
        "code_activation": activation,
        "decoder_constraint": "free",
        "encoder_bias": True,
        "decoder_bias": True,
        "apply_decoder_bias_to_input": True,
    }
    if topology == "factorized":
        overrides["site_rank"] = 2
    elif topology == "tied":
        overrides["encoder_mode"] = "tied"
    else:
        overrides["site_dims"] = (32, 24, 16, 8)
    model = make_model(device, **overrides)
    x = whitened_batch(device, n=9, seed=417)
    mixed = torch.ones(9, CFG.n_sites, dtype=torch.bool, device=device)
    mixed[::2, 0] = False
    only_first = torch.zeros_like(mixed)
    only_first[:, 0] = True
    without_first = torch.ones_like(mixed)
    without_first[:, 0] = False
    with torch.no_grad():
        model.c.copy_(torch.randn_like(model.c))
        decoder = model.decoder_tensor()
        encoder = (
            decoder * model.log_gamma.exp()
            if model.cfg.encoder_mode == "tied"
            else model.encoder_tensor()
        )
        frozen_sites = model._frozen_encoder_sites(x, encoder)
        for observed in (None, mixed, only_first, without_first):
            reference = model.forward_with_materialized(
                x,
                observed=observed,
                validate_observed=False,
                _decoder=decoder,
                _encoder=encoder,
            )[0]
            cached = model.forward_with_materialized(
                x,
                observed=observed,
                validate_observed=False,
                _decoder=decoder,
                _encoder=encoder,
                _encoder_sites=frozen_sites,
            )[0]
            for actual, expected in zip(cached, reference, strict=True):
                assert torch.equal(actual, expected)


def test_frozen_encoder_sites_fail_closed_on_stale_state(device):
    model = make_model(
        device,
        decoder_constraint="free",
        encoder_bias=True,
        decoder_bias=True,
        apply_decoder_bias_to_input=True,
        code_activation="group_soft_threshold",
        site_dims=(32, 24, 16, 8),
    )
    x = whitened_batch(device, n=7, seed=418)
    encoder = model.encoder_tensor()
    state_keys = set(model.state_dict())
    with pytest.raises(RuntimeError, match="no-grad"):
        model._frozen_encoder_sites(x, encoder)
    with torch.no_grad():
        frozen_sites = model._frozen_encoder_sites(x, encoder)
    with pytest.raises(RuntimeError, match="no-grad"):
        model._encode_from_frozen_sites(x, encoder, frozen_sites)

    def fresh_cache():
        with torch.no_grad():
            return model._frozen_encoder_sites(x, encoder)

    with torch.no_grad():
        x.add_(1)
        with pytest.raises(ValueError, match="input"):
            model._encode_from_frozen_sites(x, encoder, frozen_sites)
        frozen_sites = fresh_cache()
        encoder.add_(1)
        with pytest.raises(ValueError, match="encoder"):
            model._encode_from_frozen_sites(x, encoder, frozen_sites)
        frozen_sites = fresh_cache()
        model.c.add_(1)
        with pytest.raises(ValueError, match="preprocessing"):
            model._encode_from_frozen_sites(x, encoder, frozen_sites)
        frozen_sites = fresh_cache()
        assert model.a is not None
        model.a.add_(1)
        with pytest.raises(ValueError, match="postprocessing"):
            model._encode_from_frozen_sites(x, encoder, frozen_sites)
        frozen_sites = fresh_cache()
        assert model.log_threshold is not None
        model.log_threshold.add_(1)
        with pytest.raises(ValueError, match="postprocessing"):
            model._encode_from_frozen_sites(x, encoder, frozen_sites)
        frozen_sites = fresh_cache()
        model.coordinate_mask[0, 0, 0, 0].logical_not_()
        with pytest.raises(ValueError, match="preprocessing"):
            model._encode_from_frozen_sites(x, encoder, frozen_sites)
    assert set(model.state_dict()) == state_keys


@pytest.mark.parametrize(
    ("overrides", "expected_mutated"),
    (
        ({"decoder_constraint": "free"}, ()),
        ({"decoder_constraint": "free", "decoder_bias": False}, ("c",)),
        (
            {
                "decoder_constraint": "free",
                "site_dims": (32, 24, 16, 8),
            },
            ("D", "E", "c"),
        ),
        ({"decoder_constraint": "gram"}, ("D",)),
        (
            {
                "decoder_constraint": "free",
                "encoder_constraint": "unit_latent",
            },
            ("E",),
        ),
        (
            {
                "decoder_constraint": "unit_latent",
                "encoder_constraint": "unit_latent",
            },
            ("D", "E"),
        ),
        (
            {"decoder_constraint": "free", "site_rank": 2},
            (),
        ),
        (
            {
                "decoder_constraint": "free",
                "site_rank": 2,
                "site_dims": (24, 24, 24, 24),
            },
            ("c",),
        ),
        (
            {
                "decoder_constraint": "free",
                "encoder_mode": "tied",
                "site_dims": (32, 24, 16, 8),
            },
            ("D", "c"),
        ),
    ),
)
def test_projection_reports_exact_mutated_parameter_set(
    device, overrides, expected_mutated
):
    model = make_model(device, **overrides)
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    count, mutated = model._project_decoder_with_state_()
    assert count.shape == ()
    assert count.dtype == torch.int64
    assert count.device == next(model.parameters()).device
    assert tuple(names[id(parameter)] for parameter in mutated) == expected_mutated
    assert isinstance(model.project_decoder_(), int)


def test_batchtopk_exact_count_and_variable_per_token(device):
    B, G = 64, CFG.n_blocks
    gen = torch.Generator(device="cpu").manual_seed(2)
    scores = torch.randn(B, G, generator=gen).abs().to(device)
    mask = batch_topk_mask(scores, CFG.k)
    assert int(mask.sum().item()) == CFG.k * B
    per_token = mask.sum(dim=1).float()
    assert per_token.min() != per_token.max()  # counts vary by design
    # Kept scores dominate dropped scores globally.
    assert scores[mask].min() >= scores[~mask].max()
    # Fractional budget (the capture-sweep under-provisioned regime):
    # round(k*B) selections batch-wide.
    frac = batch_topk_mask(scores, 0.8)
    assert int(frac.sum().item()) == round(0.8 * B)


def test_token_topk_exact_per_token_count(device):
    scores = torch.rand(64, CFG.n_blocks, device=device)
    mask = token_topk_mask(scores, 3)
    assert torch.equal(mask.sum(dim=1), torch.full((64,), 3, device=device))


def test_topk_exact_ties_use_lowest_declared_candidate_index(device):
    token_scores = torch.zeros(2, 5, device=device)
    token_mask = token_topk_mask(token_scores, 2)
    assert torch.equal(
        token_mask,
        torch.tensor(
            [[True, True, False, False, False], [True, True, False, False, False]],
            device=device,
        ),
    )

    batch_scores = torch.zeros(3, 4, device=device)
    batch_mask = batch_topk_mask(batch_scores, 1)
    assert torch.equal(
        batch_mask.reshape(-1),
        torch.tensor(
            [
                True,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
            device=device,
        ),
    )


def test_selector_tie_policy_is_content_bound():
    assert CFG.selector_tie_break == "lowest_flat_index_at_cutoff"
    with pytest.raises(ValueError, match="selector_tie_break"):
        BSCConfig(**{**CFG.__dict__, "selector_tie_break": "runtime_default"})


def test_encoder_bias_breaks_antipodal_support(device):
    model = make_model(device, encoder_bias=True, selection="token_topk")
    with torch.no_grad():
        model.a.normal_(mean=0.5, std=0.1)
    x = whitened_batch(device, n=64)
    assert not torch.allclose(
        model.scores(model.encode(x)), model.scores(model.encode(-x))
    )


def test_tied_grassmannian_uses_single_gamma(device):
    model = make_model(device, encoder_mode="tied")
    assert model.E is None and model.log_gamma.shape == ()
    x = whitened_batch(device, n=8)
    expected = torch.einsum("bsd,sgkd->bgk", x, model.D) * model.log_gamma.exp()
    assert torch.allclose(model.encode(x), expected, atol=1e-5)


def test_relu_dense_crosscoder_bridge(device):
    model = make_model(
        device,
        block_dim=1,
        code_activation="relu",
        selection="dense",
        regularizer="crosscoder_l1",
        lambda_regularizer=1e-4,
        encoder_bias=True,
        decoder_constraint="frobenius",
    )
    x = whitened_batch(device, n=32)
    out = model(x)
    assert (out.z >= 0).all()
    assert torch.equal(out.mask, out.scores > 0)
    parts = bsc_loss(out, x, model)
    assert parts["regularizer"] >= 0


def test_anthropic_crosscoder_l1_sums_per_site_decoder_norms(device):
    model = make_model(
        device,
        n_blocks=2,
        block_dim=1,
        n_sites=2,
        d_model=2,
        site_dims=(2, 2),
        k=2,
        code_activation="relu",
        selection="dense",
        regularizer="crosscoder_l1",
        lambda_regularizer=1e-4,
        encoder_bias=True,
        decoder_constraint="free",
        decoder_norm_geometry="sum_l2",
        decoder_init_preconditioning="none",
        decoder_init_operation_order="gaussian_mask_rescale_then_declared_constraint",
    )
    with torch.no_grad():
        model.E.fill_(0.5)
        model.a.fill_(0.25)
        model.D.zero_()
        # Per-site norms are (3, 4) for block 0 and (5, 12) for block 1.
        model.D[0, 0, 0, 0] = 3.0
        model.D[1, 0, 0, 0] = 4.0
        model.D[0, 1, 0, 0] = 5.0
        model.D[1, 1, 0, 0] = 12.0
    x = torch.ones(3, 2, 2, device=device)
    out = model(x)
    observed = bsc_loss(out, x, model)["regularizer"]
    summed_site_cost = torch.tensor((7.0, 17.0), device=device)
    expected = (out.scores.float() * summed_site_cost).sum(dim=1).mean()
    concatenated_cost = torch.tensor((5.0, 13.0), device=device)
    negative_control = (out.scores.float() * concatenated_cost).sum(dim=1).mean()
    assert torch.allclose(observed, expected)
    assert not torch.allclose(observed, negative_control)


def test_decoder_weighted_batchtopk_score_matches_minder(device):
    model = make_model(
        device,
        block_dim=1,
        code_activation="relu",
        selection_score="decoder_weighted",
        decoder_constraint="free",
    )
    x = whitened_batch(device, n=32)
    z = model.encode(x)
    expected = z.squeeze(-1) * model.D.float().norm(dim=-1).squeeze(-1).sum(dim=0)
    assert torch.allclose(model.scores(z), expected.to(z.dtype), atol=1e-5)


def test_group_lasso_bridge_has_positive_learned_threshold(device):
    model = make_model(
        device,
        selection="dense",
        code_activation="group_soft_threshold",
        decoder_constraint="free",
        regularizer="group_l21",
        lambda_regularizer=1e-3,
        encoder_bias=True,
    )
    x = whitened_batch(device, n=32)
    out = model(x)
    assert torch.nn.functional.softplus(model.log_threshold).min() > 0
    assert torch.equal(out.mask, out.scores > 0)
    parts = bsc_loss(out, x, model)
    parts["total"].backward()
    assert model.log_threshold.grad is not None


def test_group_lasso_target_gate_keeps_exact_boundary_semantics(device):
    model = make_model(
        device,
        selection="dense",
        code_activation="group_soft_threshold",
        decoder_constraint="free",
        regularizer="group_l21",
        lambda_regularizer=1e-3,
        group_lasso_target_k=0.01,
    )
    x = whitened_batch(device, n=32)
    out = model(x)
    active = float(out.mask.float().sum(dim=1).mean())
    model.cfg.group_lasso_target_k = 0.01
    above = bsc_loss(out, x, model)["regularizer"]
    model.cfg.group_lasso_target_k = active
    at_boundary = bsc_loss(out, x, model)["regularizer"]
    model.cfg.group_lasso_target_k = active + 1.0
    below = bsc_loss(out, x, model)["regularizer"]
    assert above > 0
    assert at_boundary == 0
    assert below == 0


def test_group_threshold_training_mask_is_independent_of_endpoint_score(device):
    model = BlockCrosscoder(
        BSCConfig(
            n_blocks=1,
            block_dim=1,
            n_sites=1,
            d_model=1,
            k=1,
            selection="dense",
            code_activation="group_soft_threshold",
            selection_score="isolated_loss_decrease",
            decoder_constraint="free",
            decoder_bias=False,
            reconstruction_loss="squared_l2",
            decoder_init_preconditioning="none",
            decoder_init_operation_order=(
                "gaussian_mask_rescale_then_declared_constraint"
            ),
            group_threshold_effective_init=1.0,
        )
    ).to(device)
    with torch.no_grad():
        assert model.E is not None and model.D is not None
        model.E.fill_(-2.0)
        model.D.fill_(1.0)
    out = model(torch.ones(1, 1, 1, device=device))
    assert out.z.item() < 0
    assert out.scores.item() < 0
    assert out.mask.item() is True
    assert torch.equal(out.z_selected, out.z)


def test_free_decoder_projection_is_noop(device):
    model = make_model(device, decoder_constraint="free")
    before = model.D.detach().clone()
    assert model.project_decoder_() == 0
    assert torch.equal(model.D, before)


def test_frobenius_decoder_projection(device):
    model = make_model(device, decoder_constraint="frobenius")
    with torch.no_grad():
        model.D.mul_(5)
    hits = model.project_decoder_()
    norms = model.D.float().pow(2).sum(dim=(0, 2, 3)).sqrt()
    assert hits > 0 and norms.max() <= 1 + 1e-5


def test_map_nuclear_regularizer(device):
    model = make_model(device, regularizer="map_nuclear", lambda_regularizer=1e-3)
    x = whitened_batch(device, n=32)
    parts = bsc_loss(model(x), x, model)
    assert parts["regularizer"] > 0
    parts["total"].backward()
    assert model.E.grad is not None


def test_gradient_only_through_selected(device):
    model = make_model(device)
    x = whitened_batch(device, n=64)
    z = model.encode(x)
    z_leaf = z.detach().requires_grad_(True)
    mask = model.select(z_leaf)
    out = model.decode(z_leaf * mask.unsqueeze(-1))
    out.pow(2).sum().backward()
    grad_norms = z_leaf.grad.norm(dim=-1)  # [B, G]
    assert grad_norms[~mask].max().item() == 0.0
    assert grad_norms[mask].min().item() > 0.0


def test_threshold_mode_requires_calibration(device):
    model = make_model(device)
    x = whitened_batch(device, n=8)
    with pytest.raises(RuntimeError):
        model(x, mode="threshold")
    model.theta.fill_(model.scores(model.encode(x)).median())
    out = model(x, mode="threshold")
    assert out.mask.any() and not out.mask.all()


def test_threshold_validation_cache_tracks_buffer_lifecycle(device, monkeypatch):
    model = BlockCrosscoder(CFG, device=device)
    assert model.theta.device == model.parameter_device
    state_keys = set(model.state_dict())
    scores = torch.ones(2, CFG.n_blocks, device=device)
    original_isfinite = torch.isfinite
    validation_calls = 0

    def tracked_isfinite(value):
        nonlocal validation_calls
        if value is model.theta:
            validation_calls += 1
        return original_isfinite(value)

    monkeypatch.setattr(torch, "isfinite", tracked_isfinite)
    with pytest.raises(RuntimeError, match="not calibrated"):
        model._select_scores(scores, mode="threshold")
    assert validation_calls == 1

    model.theta.fill_(0.0)
    model._select_scores(scores, mode="threshold")
    model._select_scores(scores, mode="threshold")
    assert validation_calls == 2

    model.theta.fill_(float("inf"))
    with pytest.raises(RuntimeError, match="not calibrated"):
        model._select_scores(scores, mode="threshold")
    model.theta.fill_(0.0)
    model._select_scores(scores, mode="threshold")
    assert validation_calls == 4

    nan_state = {name: value.clone() for name, value in model.state_dict().items()}
    nan_state["theta"].fill_(float("nan"))
    model.load_state_dict(nan_state)
    with pytest.raises(RuntimeError, match="not calibrated"):
        model._select_scores(scores, mode="threshold")
    finite_state = {name: value.clone() for name, value in nan_state.items()}
    finite_state["theta"].fill_(0.25)
    model.load_state_dict(finite_state)
    model._select_scores(scores, mode="threshold")
    assert validation_calls == 6

    cloned = copy.deepcopy(model)
    cloned._select_scores(scores, mode="threshold")
    assert cloned._validated_theta_key is not None
    assert cloned._validated_theta_key[0] == id(cloned.theta)
    converted = copy.deepcopy(model).to(dtype=torch.float64)
    converted._select_scores(scores.double(), mode="threshold")
    assert converted._validated_theta_key is not None
    assert converted._validated_theta_key[3] == torch.float64
    assert set(model.state_dict()) == state_keys


@pytest.mark.parametrize(
    "selection_score",
    ("code_norm", "decoder_weighted", "decoded_energy", "isolated_loss_decrease"),
)
def test_frozen_score_geometry_is_exact_and_decoder_bound(device, selection_score):
    overrides = {"selection_score": selection_score}
    if selection_score == "decoder_weighted":
        overrides["code_activation"] = "relu"
    if selection_score == "isolated_loss_decrease":
        overrides["decoder_bias"] = False
    model = make_model(device, **overrides)
    x = whitened_batch(device, n=17, seed=313)
    observed = torch.ones(17, CFG.n_sites, dtype=torch.bool, device=device)
    observed[::2, -1] = False

    with torch.no_grad():
        decoder = model.decoder_tensor()
        encoder = model.encoder_tensor()
        reference = model.forward_with_materialized(
            x,
            observed=observed,
            _decoder=decoder,
            _encoder=encoder,
        )[0]
        geometry = model._frozen_score_geometry(decoder)
        encoder_sites = model._frozen_encoder_sites(x, encoder)
        cached = model.forward_with_materialized(
            x,
            observed=observed,
            _decoder=decoder,
            _encoder=encoder,
            _score_geometry=geometry,
        )[0]
        for actual, expected in zip(cached, reference, strict=True):
            assert torch.equal(actual, expected)
        selected = model.select_with_materialized(
            x,
            observed=observed,
            _decoder=decoder,
            _encoder=encoder,
            _score_geometry=geometry,
        )[0]
        for actual, expected in zip(selected, cached[1:], strict=True):
            assert torch.equal(actual, expected)
        encoder_cached = model.forward_with_materialized(
            x,
            observed=observed,
            _decoder=decoder,
            _encoder=encoder,
            _score_geometry=geometry,
            _encoder_sites=encoder_sites,
        )[0]
        for actual, expected in zip(encoder_cached, cached, strict=True):
            assert torch.equal(actual, expected)
        with pytest.raises(ValueError, match="not bound"):
            model.forward_with_materialized(
                x,
                observed=observed,
                _decoder=decoder.clone(),
                _encoder=encoder,
                _score_geometry=geometry,
            )

    with pytest.raises(RuntimeError, match="no-grad"):
        model._frozen_score_geometry(model.decoder_tensor())


def test_threshold_fit_builds_score_geometry_once(device, monkeypatch):
    model = make_model(device, selection_score="decoded_energy")
    original = model._frozen_score_geometry
    calls = 0

    def counted(decoder):
        nonlocal calls
        calls += 1
        return original(decoder)

    monkeypatch.setattr(model, "_frozen_score_geometry", counted)
    x = whitened_batch(device, n=48, seed=317)
    model.fit_threshold_(list(x.split(12)), CFG.k, method="exact")
    assert calls == 1


def test_fixed_threshold_training_selector_has_variable_counts(device):
    model = make_model(device, selection="threshold")
    x = whitened_batch(device, n=128)
    model.fit_threshold_([x], target_avg_blocks=CFG.k, method="exact")
    out = model(x)
    assert abs(float(out.mask.float().sum(dim=1).mean()) - CFG.k) < 0.1
    assert out.mask.sum(dim=1).min() != out.mask.sum(dim=1).max()


def test_init_tied_and_score_comparability(device):
    model = make_model(device)
    assert torch.equal(model.E, model.D)  # transpose-tied at init
    x = whitened_batch(device, n=2048)
    model.calibrate_encoder_scale_(x)
    p = model.scores(model.encode(x)).mean(dim=0)  # [G]
    spread = (p.max() / p.min()).item()
    assert spread < 1.01  # per-block means equalized on the calib batch


def planted_lowrank_batch(device, n=1024, rank=8, seed=3):
    """Data with structure a sparse model can actually fit: a low-rank
    linear source (rank < k*b) plus small noise — a micro-preview of the
    Phase-1 generator."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    u = torch.randn(n, rank, generator=gen)
    P = torch.randn(rank, CFG.n_sites * CFG.d_model, generator=gen) / rank**0.5
    x = (u @ P).view(n, CFG.n_sites, CFG.d_model)
    x = x + 0.01 * torch.randn(x.shape, generator=gen)
    return x.to(device)


def test_train_smoke_loss_decreases(device):
    """Full ordering on tiny data: optimizer step -> retract -> next step.
    Loss must fall and the constraint must hold at every step."""
    torch.manual_seed(0)
    model = make_model(device)
    x = planted_lowrank_batch(device)
    opt = torch.optim.Adam(
        [
            {"params": [model.E], "weight_decay": 1e-4},
            {"params": [model.D, model.c], "weight_decay": 0.0},  # decoders: 0
        ],
        lr=3e-3,
    )
    losses = []
    for _ in range(60):
        out = model(x)
        parts = bsc_loss(out, x, model)
        opt.zero_grad()
        parts["total"].backward()
        opt.step()
        retract_(model.D.data, eig_floor=model.cfg.eig_floor)
        losses.append(parts["rec"].item())
        assert gram_residual(model.D).max().item() < 1e-4
    assert losses[-1] < 0.5 * losses[0]
    assert all(torch.isfinite(torch.tensor(losses)))


# -- streaming theta quantile ------------------------------------------------


def test_streaming_quantile_matches_exact_kthvalue(device):
    from block_crosscoder_experiment.model import StreamingScoreQuantile

    gen = torch.Generator().manual_seed(5)
    scores = torch.rand(200_000, generator=gen).pow(2) * 30  # skewed, positive
    hist = StreamingScoreQuantile(device=device)
    for chunk in scores.to(device).split(4096):
        hist.update(chunk)
    for q in (0.5, 0.99, 0.9921875):  # last = 1 - 32/4096, the pilot target
        n = scores.numel()
        idx = min(max(int(round(q * n)), 1), n)
        exact = float(scores.kthvalue(idx).values)
        assert abs(hist.quantile(q) - exact) / exact < 1e-3


def test_streaming_quantile_batch_order_independent(device):
    from block_crosscoder_experiment.model import StreamingScoreQuantile

    gen = torch.Generator().manual_seed(6)
    chunks = [torch.rand(1000, generator=gen) * 50 for _ in range(20)]
    a, b = StreamingScoreQuantile(device=device), StreamingScoreQuantile(device=device)
    for c in chunks:
        a.update(c.to(device))
    for c in reversed(chunks):
        b.update(c.to(device))
    assert torch.equal(a.counts, b.counts)
    assert a.quantile(0.99) == b.quantile(0.99)


def test_streaming_quantile_rejects_nonfinite(device):
    from block_crosscoder_experiment.model import StreamingScoreQuantile

    hist = StreamingScoreQuantile(device=device)
    bad = torch.tensor([1.0, float("nan")], device=device)
    with pytest.raises(ValueError, match="non-finite"):
        hist.update(bad)


def test_fit_threshold_streaming_matches_exact(device):
    """The E1 validation gate in miniature: theta and realized avg-blocks
    agree between methods on the same calibration batches."""
    model = BlockCrosscoder(CFG).to(device)
    gen = torch.Generator().manual_seed(7)
    calib = [
        torch.randn(512, CFG.n_sites, CFG.d_model, generator=gen).to(device)
        for _ in range(8)
    ]
    theta_exact = model.fit_threshold_(calib, float(CFG.k), method="exact")
    counts_exact = torch.cat(
        [model(x, mode="threshold").mask.sum(dim=1).float() for x in calib]
    )
    theta_stream = model.fit_threshold_(calib, float(CFG.k), method="streaming")
    counts_stream = torch.cat(
        [model(x, mode="threshold").mask.sum(dim=1).float() for x in calib]
    )
    assert abs(theta_stream - theta_exact) / theta_exact < 1e-3
    assert abs(float(counts_stream.mean()) - float(counts_exact.mean())) <= 0.1


def test_decoded_energy_score_equals_isolated_decoded_contribution_norm():
    cfg = BSCConfig(
        n_blocks=5,
        block_dim=3,
        n_sites=2,
        d_model=7,
        k=2,
        decoder_constraint="free",
        selection_score="decoded_energy",
    )
    model = BlockCrosscoder(cfg)
    z = torch.randn(11, 5, 3, generator=torch.Generator().manual_seed(481))
    contribution = torch.einsum("ngb,sgbd->nsgd", z, model.decoder_tensor())
    expected = contribution.float().pow(2).sum(dim=(1, 3)).sqrt()
    assert torch.allclose(model.scores(z), expected, rtol=1e-6, atol=1e-6)


def test_decoded_energy_reduces_to_code_norm_on_concatenated_stiefel_blocks():
    decoded = BlockCrosscoder(
        BSCConfig(
            n_blocks=7,
            block_dim=3,
            n_sites=4,
            d_model=11,
            k=2,
            decoder_constraint="qr",
            selection_score="decoded_energy",
        )
    )
    conventional = BlockCrosscoder(
        BSCConfig(
            n_blocks=7,
            block_dim=3,
            n_sites=4,
            d_model=11,
            k=2,
            decoder_constraint="qr",
            selection_score="code_norm",
        )
    )
    conventional.load_state_dict(decoded.state_dict())
    z = torch.randn(13, 7, 3, generator=torch.Generator().manual_seed(482))
    assert torch.allclose(
        decoded.scores(z), conventional.scores(z), rtol=2e-5, atol=2e-6
    )


def test_isolated_loss_decrease_matches_explicit_candidate_reconstructions():
    model = BlockCrosscoder(
        BSCConfig(
            n_blocks=5,
            block_dim=3,
            n_sites=2,
            d_model=7,
            k=2,
            decoder_constraint="free",
            decoder_bias=False,
            reconstruction_loss="squared_l2",
            selection_score="isolated_loss_decrease",
        )
    )
    generator = torch.Generator().manual_seed(483)
    x = torch.randn(11, 2, 7, generator=generator)
    z = torch.randn(11, 5, 3, generator=generator)
    contribution = torch.einsum("ngb,sgbd->nsgd", z, model.decoder_tensor())
    expected = 2.0 * torch.einsum(
        "nsd,nsgd->ng", x, contribution
    ) - contribution.square().sum(dim=(1, 3))
    assert torch.allclose(model.scores(z, x=x), expected, rtol=1e-5, atol=1e-5)


def test_isolated_loss_decrease_inplace_accumulators_preserve_exact_gradients(
    device,
):
    config = BSCConfig(
        n_blocks=5,
        block_dim=3,
        n_sites=3,
        d_model=7,
        k=2,
        decoder_constraint="free",
        decoder_bias=False,
        reconstruction_loss="squared_l2",
        selection_score="isolated_loss_decrease",
    )
    actual_model = BlockCrosscoder(config).to(device)
    reference_model = copy.deepcopy(actual_model)
    generator = torch.Generator().manual_seed(983)
    x_actual = torch.randn(11, 3, 7, generator=generator).to(device).requires_grad_()
    z_actual = torch.randn(11, 5, 3, generator=generator).to(device).requires_grad_()
    x_reference = x_actual.detach().clone().requires_grad_()
    z_reference = z_actual.detach().clone().requires_grad_()

    actual = actual_model.scores(z_actual, x=x_actual)
    assert actual_model.D is not None
    actual_grads = torch.autograd.grad(
        actual.sum(), (x_actual, z_actual, actual_model.D)
    )

    assert reference_model.D is not None
    decoder = reference_model.D.float()
    code = z_reference.float()
    projected = torch.zeros_like(code)
    energy_sq = torch.zeros(
        z_reference.shape[:2], dtype=torch.float32, device=device
    )
    for site in range(config.n_sites):
        projected = projected + torch.einsum(
            "nd,gbd->ngb", x_reference[:, site].float(), decoder[site]
        )
        site_gram = torch.einsum(
            "gbd,gcd->gbc", decoder[site], decoder[site]
        )
        site_energy = torch.einsum(
            "ngb,gbc,ngc->ng", code, site_gram, code
        )
        energy_sq = energy_sq + site_energy
    expected = 2.0 * (projected * code).sum(dim=-1) - energy_sq
    expected_grads = torch.autograd.grad(
        expected.sum(), (x_reference, z_reference, reference_model.D)
    )

    assert torch.equal(actual, expected)
    assert all(
        torch.equal(actual_grad, expected_grad)
        for actual_grad, expected_grad in zip(
            actual_grads, expected_grads, strict=True
        )
    )


def test_isolated_loss_decrease_excludes_hidden_clean_targets():
    model = BlockCrosscoder(
        BSCConfig(
            n_blocks=4,
            block_dim=2,
            n_sites=3,
            d_model=5,
            k=2,
            decoder_constraint="free",
            decoder_bias=False,
            reconstruction_loss="squared_l2",
            selection_score="isolated_loss_decrease",
        )
    )
    x = torch.randn(9, 3, 5, generator=torch.Generator().manual_seed(484))
    observed = torch.tensor([[True, False, True]]).expand(len(x), -1)
    z = model.encode(x, observed=observed)
    baseline = model.scores(z, x=x, observed=observed)
    changed = x.clone()
    changed[:, 1] += 10_000.0
    assert torch.equal(model.encode(changed, observed=observed), z)
    assert torch.equal(model.scores(z, x=changed, observed=observed), baseline)


def test_isolated_loss_decrease_reduces_to_squared_code_norm_for_tied_stiefel():
    model = BlockCrosscoder(
        BSCConfig(
            n_blocks=6,
            block_dim=2,
            n_sites=3,
            d_model=7,
            k=2,
            encoder_mode="tied",
            decoder_constraint="qr",
            decoder_bias=False,
            reconstruction_loss="squared_l2",
            selection_score="isolated_loss_decrease",
        )
    )
    x = torch.randn(13, 3, 7, generator=torch.Generator().manual_seed(485))
    z = model.encode(x)
    assert torch.allclose(
        model.scores(z, x=x), z.square().sum(dim=-1), rtol=2e-5, atol=2e-5
    )


def test_isolated_loss_decrease_is_reciprocal_block_gauge_invariant():
    config = BSCConfig(
        n_blocks=3,
        block_dim=2,
        n_sites=2,
        d_model=4,
        k=2,
        decoder_constraint="free",
        decoder_bias=False,
        reconstruction_loss="squared_l2",
        selection_score="isolated_loss_decrease",
    )
    original = BlockCrosscoder(config)
    transformed = BlockCrosscoder(config)
    transformed.load_state_dict(original.state_dict())
    gauge = torch.tensor([[2.0, 0.25], [0.0, 0.5]])
    inverse_transpose = torch.linalg.inv(gauge).T
    with torch.no_grad():
        assert transformed.E is not None and transformed.D is not None
        transformed.E.copy_(torch.einsum("bc,sgcd->sgbd", gauge, transformed.E))
        transformed.D.copy_(
            torch.einsum("bc,sgcd->sgbd", inverse_transpose, transformed.D)
        )
    x = torch.randn(17, 2, 4, generator=torch.Generator().manual_seed(487))
    original_code = original.encode(x)
    transformed_code = transformed.encode(x)
    assert torch.allclose(
        transformed.scores(transformed_code, x=x),
        original.scores(original_code, x=x),
        rtol=1e-5,
        atol=1e-5,
    )


def test_isolated_loss_decrease_preserves_harmful_negative_scores():
    model = BlockCrosscoder(
        BSCConfig(
            n_blocks=1,
            block_dim=1,
            n_sites=1,
            d_model=1,
            k=1,
            decoder_constraint="free",
            decoder_bias=False,
            reconstruction_loss="squared_l2",
            selection_score="isolated_loss_decrease",
        )
    )
    with torch.no_grad():
        assert model.D is not None
        model.D.fill_(1.0)
    score = model.scores(torch.tensor([[[-1.0]]]), x=torch.tensor([[[1.0]]]))
    assert score.item() == -3.0


@pytest.mark.parametrize(
    "overrides",
    (
        {"decoder_bias": True, "reconstruction_loss": "squared_l2"},
        {"decoder_bias": False, "reconstruction_loss": "mean_l2"},
    ),
)
def test_isolated_loss_decrease_rejects_nonquadratic_or_biased_carriers(
    overrides,
):
    with pytest.raises(
        ValueError, match="bias-free quadratic reconstruction objective"
    ):
        BSCConfig(
            n_blocks=2,
            block_dim=2,
            n_sites=2,
            d_model=3,
            k=1,
            selection_score="isolated_loss_decrease",
            **overrides,
        )


def test_signed_streaming_quantile_matches_exact_and_is_order_independent(device):
    generator = torch.Generator().manual_seed(486)
    scores = torch.randn(200_000, generator=generator).pow(3)
    chunks = list(scores.split(4096))
    forward = SignedStreamingScoreQuantile(device=device)
    reverse = SignedStreamingScoreQuantile(device=device)
    for chunk in chunks:
        forward.update(chunk.to(device))
    for chunk in reversed(chunks):
        reverse.update(chunk.to(device))
    assert torch.equal(forward.counts, reverse.counts)
    for quantile in (0.01, 0.5, 0.99):
        index = min(max(int(round(quantile * scores.numel())), 1), scores.numel())
        exact = float(scores.kthvalue(index).values)
        approximate = forward.quantile(quantile)
        assert abs(approximate - exact) / max(abs(exact), 1e-9) < 1e-3


def test_isolated_loss_decrease_streaming_threshold_matches_exact(device):
    model = BlockCrosscoder(
        BSCConfig(
            n_blocks=16,
            block_dim=2,
            n_sites=3,
            d_model=8,
            k=3,
            decoder_constraint="free",
            decoder_bias=False,
            reconstruction_loss="squared_l2",
            selection_score="isolated_loss_decrease",
        )
    ).to(device)
    generator = torch.Generator().manual_seed(488)
    calibration = [
        torch.randn(128, 3, 8, generator=generator).to(device) for _ in range(8)
    ]
    exact = model.fit_threshold_(calibration, 3.0, method="exact")
    exact_counts = torch.cat(
        [model(batch, mode="threshold").mask.sum(dim=1) for batch in calibration]
    ).float()
    streaming = model.fit_threshold_(calibration, 3.0, method="streaming")
    streaming_counts = torch.cat(
        [model(batch, mode="threshold").mask.sum(dim=1) for batch in calibration]
    ).float()
    assert abs(streaming - exact) / max(abs(exact), 1e-9) < 1e-3
    assert abs(float(streaming_counts.mean() - exact_counts.mean())) <= 0.1
