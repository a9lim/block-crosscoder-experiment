"""Model-level checks: the selection-score identity, BatchTopK exactness
and gradient masking, init conventions, and a smoke test of the full
step -> retract loop on tiny synthetic data."""

import copy

import pytest
import torch

import block_crosscoder_experiment.model as model_module
from block_crosscoder_experiment.gram import gram_residual, retract_
from block_crosscoder_experiment.model import (
    BlockCrosscoder,
    BSCConfig,
    BSCOutput,
    ISOLATED_LOSS_EXACT_IMPLEMENTATION,
    ISOLATED_LOSS_MAPPED_IMPLEMENTATION,
    SignedStreamingScoreQuantile,
    batch_topk_mask,
    bsc_loss,
    token_topk_mask,
)
from block_crosscoder_experiment.runtime_limits import (
    DECODER_RETRACTION_CHOLESKY_QR_IMPLEMENTATION,
    DECODER_RETRACTION_HOUSEHOLDER_QR_IMPLEMENTATION,
    DECODER_RETRACTION_NOT_APPLICABLE,
    DECODER_RETRACTION_SYMMETRIC_POLAR_IMPLEMENTATION,
    DECODED_ENERGY_EXACT_IMPLEMENTATION,
    DECODED_ENERGY_STIEFEL_CODE_NORM_IMPLEMENTATION,
)

CFG = BSCConfig(n_blocks=16, block_dim=4, n_sites=4, d_model=32, k=3, seed=0)


def make_model(device, **overrides):
    if (
        "decoder_constraint" in overrides
        and "decoder_retraction_implementation" not in overrides
    ):
        overrides["decoder_retraction_implementation"] = None
    cfg = BSCConfig(**{**CFG.__dict__, **overrides})
    return BlockCrosscoder(cfg).to(device)


def whitened_batch(device, n=512, seed=1):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(n, CFG.n_sites, CFG.d_model, generator=gen).to(device)


def _legacy_per_site_forward(
    model: BlockCrosscoder,
    x: torch.Tensor,
    observed: torch.Tensor | None,
) -> tuple[BSCOutput, torch.Tensor, torch.Tensor]:
    """Test oracle for the superseded per-site BMM training kernel."""
    cfg = model.cfg
    decoder = model.decoder_tensor()
    if cfg.encoder_mode == "tied":
        assert model.log_gamma is not None
        encoder = decoder * model.log_gamma.exp()
    else:
        encoder = model.encoder_tensor()
    prepared = model._prepare_encoder_input(x)
    if observed is None and cfg.encoder_fusion == "sum":
        keep = None
    else:
        keep = model._site_observation_mask(prepared, observed)
        prepared = prepared * keep
    weights = encoder.reshape(cfg.n_sites, cfg.n_latents, cfg.d_model)
    per_site = torch.bmm(
        prepared.transpose(0, 1),
        weights.transpose(1, 2),
    )
    z = model._finish_encoded_sum(per_site.sum(dim=0), keep)
    scores = model.scores(
        z,
        x=x,
        observed=observed,
        _decoder=decoder,
        _observation_keep=keep,
    )
    mask = model._select_scores(scores, mode="topk", z=z)
    z_selected = z * mask.unsqueeze(-1)
    xhat = model.decode(z_selected, _decoder=decoder)
    return BSCOutput(xhat, z, z_selected, scores, mask), decoder, encoder


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    numerator = (actual.float() - expected.float()).norm()
    denominator = expected.float().norm().clamp_min(1e-12)
    return float((numerator / denominator).detach())


@pytest.mark.parametrize(
    ("constraint", "expected"),
    (
        ("qr", DECODER_RETRACTION_CHOLESKY_QR_IMPLEMENTATION),
        ("gram", DECODER_RETRACTION_SYMMETRIC_POLAR_IMPLEMENTATION),
        ("free", DECODER_RETRACTION_NOT_APPLICABLE),
        ("frobenius", DECODER_RETRACTION_NOT_APPLICABLE),
    ),
)
def test_decoder_retraction_identity_resolves_explicitly(constraint, expected):
    cfg = BSCConfig(
        n_blocks=4,
        block_dim=2,
        n_sites=2,
        d_model=5,
        k=2,
        decoder_constraint=constraint,
    )
    assert cfg.decoder_retraction_implementation == expected


@pytest.mark.parametrize(
    ("constraint", "implementation", "message"),
    (
        ("qr", "ambient_cuda_default", "unknown"),
        (
            "qr",
            DECODER_RETRACTION_SYMMETRIC_POLAR_IMPLEMENTATION,
            "requires a QR",
        ),
        (
            "gram",
            DECODER_RETRACTION_CHOLESKY_QR_IMPLEMENTATION,
            "requires symmetric-polar",
        ),
        (
            "free",
            DECODER_RETRACTION_HOUSEHOLDER_QR_IMPLEMENTATION,
            "not-applicable",
        ),
    ),
)
def test_decoder_retraction_identity_pairs_fail_closed(
    constraint,
    implementation,
    message,
):
    with pytest.raises(ValueError, match=message):
        BSCConfig(
            n_blocks=4,
            block_dim=2,
            n_sites=2,
            d_model=5,
            k=2,
            decoder_constraint=constraint,
            decoder_retraction_implementation=implementation,
        )


def test_stiefel_decoder_refuses_insufficient_active_coordinates():
    with pytest.raises(ValueError, match="at least block_dim active coordinates"):
        BSCConfig(
            n_blocks=4,
            block_dim=4,
            n_sites=2,
            d_model=3,
            site_dims=(1, 2),
            k=2,
            decoder_constraint="qr",
        )


@pytest.mark.parametrize(
    ("implementation", "function_name"),
    (
        (
            DECODER_RETRACTION_CHOLESKY_QR_IMPLEMENTATION,
            "_cholesky_qr_retract_count_tensor_",
        ),
        (
            DECODER_RETRACTION_HOUSEHOLDER_QR_IMPLEMENTATION,
            "_qr_retract_count_tensor_",
        ),
    ),
)
def test_model_qr_retraction_dispatch_is_identity_bound(
    monkeypatch,
    implementation,
    function_name,
):
    calls = 0
    original = getattr(model_module, function_name)

    def tracked(value):
        nonlocal calls
        calls += 1
        return original(value)

    monkeypatch.setattr(model_module, function_name, tracked)
    model = BlockCrosscoder(
        BSCConfig(
            n_blocks=8,
            block_dim=3,
            n_sites=2,
            d_model=7,
            k=2,
            decoder_constraint="qr",
            decoder_retraction_implementation=implementation,
        )
    )
    assert calls == 1
    with torch.no_grad():
        assert model.D is not None
        model.D.add_(0.01 * torch.randn_like(model.D))
    model.project_decoder_()
    assert calls == 2
    assert float(gram_residual(model.decoder_tensor()).detach().max()) <= 1e-4


def _loss_output(prediction: torch.Tensor, n_blocks: int, block_dim: int) -> BSCOutput:
    batch = prediction.shape[0]
    z = torch.zeros(
        batch,
        n_blocks,
        block_dim,
        device=prediction.device,
        dtype=prediction.dtype,
    )
    scores = torch.zeros(
        batch,
        n_blocks,
        device=prediction.device,
        dtype=prediction.dtype,
    )
    mask = torch.zeros(
        batch,
        n_blocks,
        device=prediction.device,
        dtype=torch.bool,
    )
    return BSCOutput(prediction, z, z, scores, mask)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA Inductor")
@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
@pytest.mark.parametrize("reconstruction_loss", ("mean_squared", "squared_l2"))
def test_compiled_cuda_quadratic_reduction_bounds_loss_and_input_gradient_drift(
    dtype, reconstruction_loss
):
    device = torch.device("cuda")
    batch, n_sites, d_model = 256, 4, 1024
    assert batch * n_sites * d_model == model_module._CUDA_QUADRATIC_FUSION_MIN_ELEMENTS
    cfg = BSCConfig(
        n_blocks=2,
        block_dim=2,
        n_sites=n_sites,
        d_model=d_model,
        k=1,
        seed=997,
        decoder_constraint="free",
        reconstruction_loss=reconstruction_loss,
    )
    model = BlockCrosscoder(cfg).to(device=device, dtype=dtype)
    finite_values = torch.tensor(
        (-4096.0, -17.0, -1.0, -(2.0**-10), 0.0, 2.0**-10, 1.0, 17.0, 4096.0),
        device=device,
        dtype=dtype,
    )
    repeats = (batch * n_sites * d_model + len(finite_values) - 1) // len(finite_values)
    prediction = finite_values.repeat(repeats)[: batch * n_sites * d_model]
    prediction = prediction.reshape(batch, n_sites, d_model)
    prediction = prediction.transpose(0, 1).contiguous().transpose(0, 1)
    prediction.requires_grad_()
    assert prediction.stride() == (d_model, batch * d_model, 1)
    target = finite_values.flip(0).repeat(repeats)[: batch * n_sites * d_model]
    target = target.reshape(batch, n_sites, d_model).clone().requires_grad_()
    reference_prediction = prediction.detach().clone().requires_grad_()
    reference_target = target.detach().clone().requires_grad_()

    denominator = batch if reconstruction_loss == "squared_l2" else target.numel()
    actual_loss = model_module._fp32_squared_error_reduction(
        prediction,
        target,
        denominator,
    )
    expected_loss = model_module._eager_fp32_squared_error_reduction(
        reference_prediction,
        reference_target,
        denominator,
    )
    direct_loss_relative_error = abs(
        float((actual_loss - expected_loss).detach())
    ) / max(abs(float(expected_loss.detach())), 1e-30)
    assert direct_loss_relative_error <= 2e-6

    bsc_actual_loss = bsc_loss(
        _loss_output(prediction, cfg.n_blocks, cfg.block_dim),
        target,
        model,
    )["rec"]
    loss_relative_error = abs(float((bsc_actual_loss - expected_loss).detach())) / max(
        abs(float(expected_loss.detach())), 1e-30
    )
    assert loss_relative_error <= 2e-6
    actual_prediction_gradient, actual_target_gradient = torch.autograd.grad(
        bsc_actual_loss,
        (prediction, target),
    )
    expected_prediction_gradient, expected_target_gradient = torch.autograd.grad(
        expected_loss,
        (reference_prediction, reference_target),
    )
    assert (
        _relative_l2(actual_prediction_gradient, expected_prediction_gradient) <= 2e-6
    )
    assert _relative_l2(actual_target_gradient, expected_target_gradient) <= 2e-6


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA Inductor")
@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_compiled_cuda_quadratic_preserves_nonfinite_refusal_surface(dtype):
    device = torch.device("cuda")
    n = model_module._CUDA_QUADRATIC_FUSION_MIN_ELEMENTS
    prediction = torch.zeros(n, device=device, dtype=dtype, requires_grad=True)
    target = torch.zeros_like(prediction, requires_grad=True)
    with torch.no_grad():
        prediction[:3] = torch.tensor(
            (float("nan"), float("inf"), float("-inf")),
            device=device,
            dtype=dtype,
        )
    reference_prediction = prediction.detach().clone().requires_grad_()
    reference_target = target.detach().clone().requires_grad_()
    actual_loss = model_module._fp32_squared_error_reduction(
        prediction,
        target,
        n,
    )
    expected_loss = model_module._eager_fp32_squared_error_reduction(
        reference_prediction,
        reference_target,
        n,
    )

    assert torch.isnan(actual_loss) and torch.isnan(expected_loss)
    actual_gradients = torch.autograd.grad(actual_loss, (prediction, target))
    expected_gradients = torch.autograd.grad(
        expected_loss,
        (reference_prediction, reference_target),
    )
    for actual_gradient, expected_gradient in zip(
        actual_gradients,
        expected_gradients,
        strict=True,
    ):
        for predicate in (torch.isnan, torch.isposinf, torch.isneginf):
            assert torch.equal(
                predicate(actual_gradient),
                predicate(expected_gradient),
            )
        finite_gradient = torch.isfinite(expected_gradient)
        assert torch.equal(
            actual_gradient[finite_gradient],
            expected_gradient[finite_gradient],
        )


def test_cuda_quadratic_compile_wrapper_is_lazy_cached_and_size_gated(
    device, monkeypatch
):
    compile_calls: list[dict[str, object]] = []
    compiled_calls = 0

    def fake_compile(function, **options):
        compile_calls.append(options)

        def compiled(*args):
            nonlocal compiled_calls
            compiled_calls += 1
            return function(*args)

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    model_module._compiled_cuda_fp32_squared_error_reduction.cache_clear()
    try:
        n = model_module._CUDA_QUADRATIC_FUSION_MIN_ELEMENTS
        large = torch.linspace(-2.0, 2.0, n, device=device)
        target = large.flip(0)
        for _ in range(2):
            actual = model_module._fp32_squared_error_reduction(large, target, n)
            expected = model_module._eager_fp32_squared_error_reduction(
                large,
                target,
                n,
            )
            assert torch.allclose(actual, expected, rtol=2e-6, atol=1e-6)
        small = large[: n - 1]
        assert torch.equal(
            model_module._fp32_squared_error_reduction(
                small,
                target[: n - 1],
                n - 1,
            ),
            model_module._eager_fp32_squared_error_reduction(
                small,
                target[: n - 1],
                n - 1,
            ),
        )
        if device.type == "cuda":
            assert compile_calls == [
                {
                    "backend": "inductor",
                    "fullgraph": True,
                    "dynamic": True,
                }
            ]
            assert compiled_calls == 2
        else:
            assert compile_calls == []
            assert compiled_calls == 0
    finally:
        model_module._compiled_cuda_fp32_squared_error_reduction.cache_clear()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA Inductor")
def test_compiled_cuda_quadratic_accepts_more_than_dynamo_static_shape_limit():
    model_module._compiled_cuda_fp32_squared_error_reduction.cache_clear()
    base = model_module._CUDA_QUADRATIC_FUSION_MIN_ELEMENTS
    try:
        for offset in range(9):
            n = base + offset
            prediction = torch.linspace(-1.0, 1.0, n, device="cuda")
            target = prediction.flip(0)
            actual = model_module._fp32_squared_error_reduction(
                prediction,
                target,
                n,
            )
            expected = model_module._eager_fp32_squared_error_reduction(
                prediction,
                target,
                n,
            )
            assert torch.allclose(actual, expected, rtol=2e-6, atol=1e-6)
    finally:
        model_module._compiled_cuda_fp32_squared_error_reduction.cache_clear()


@pytest.mark.parametrize(
    "case",
    ("observation_mask", "padding", "mean_l1", "mean_l2"),
)
def test_quadratic_compilation_excludes_non_dominant_loss_paths(
    device, monkeypatch, case
):
    overrides: dict[str, object] = {}
    observation_mask = None
    if case == "padding":
        overrides["site_dims"] = (8, 7, 6, 5)
    elif case in {"mean_l1", "mean_l2"}:
        overrides["reconstruction_loss"] = case
    cfg = BSCConfig(
        n_blocks=2,
        block_dim=2,
        n_sites=4,
        d_model=8,
        k=1,
        decoder_constraint="free",
        **overrides,
    )
    model = BlockCrosscoder(cfg).to(device)
    prediction = torch.randn(5, 4, 8, device=device, requires_grad=True)
    target = torch.randn_like(prediction)
    if case == "observation_mask":
        observation_mask = torch.ones(5, 4, dtype=torch.bool, device=device)
        observation_mask[::2, -1] = False

    def forbidden(*_args, **_kwargs):
        raise AssertionError("non-dominant reconstruction entered compiled helper")

    monkeypatch.setattr(model_module, "_fp32_squared_error_reduction", forbidden)
    loss = bsc_loss(
        _loss_output(prediction, cfg.n_blocks, cfg.block_dim),
        target,
        model,
        observation_mask=observation_mask,
    )["rec"]
    loss.backward()
    assert torch.isfinite(loss)
    assert prediction.grad is not None


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


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
@pytest.mark.parametrize(
    "case",
    (
        "untied_sum_all",
        "untied_sum_missing",
        "untied_mean_missing",
        "untied_rescaled_missing",
        "untied_source_missing",
        "untied_padded_missing",
        "tied_rescaled_missing",
        "factorized_mean_missing",
    ),
)
def test_flattened_training_encoder_tracks_legacy_kernel_end_to_end(
    device, dtype, case
):
    """Bound numerical drift from the authorized bf16 reduction-order change.

    The reference deliberately retains the previous ``[S,B,G*b]`` BMM
    materialization.  Besides encoder output, this checks every public forward
    tensor, reconstruction loss, hard support, and all parameter gradients.
    """
    common = dict(
        n_blocks=8,
        block_dim=2,
        n_sites=4,
        d_model=16,
        k=3,
        seed=731,
        selection="token_topk",
        decoder_constraint="free",
        encoder_bias=True,
        decoder_bias=True,
    )
    batch_tokens = 128
    overrides: dict[str, object] = {}
    observed: torch.Tensor | None
    if case == "untied_sum_all":
        observed = None
    else:
        observed = torch.ones(batch_tokens, 4, dtype=torch.bool, device=device)
        observed[::2, 0] = False
        observed[1::3, 2] = False
    if case == "untied_mean_missing":
        overrides["encoder_fusion"] = "mean"
    elif case == "untied_rescaled_missing":
        overrides["encoder_fusion"] = "availability_rescaled_sum"
    elif case == "untied_source_missing":
        overrides.update(encoder_fusion="source", source_site=1)
    elif case == "untied_padded_missing":
        overrides.update(
            site_dims=(16, 13, 10, 7),
            apply_decoder_bias_to_input=True,
        )
    elif case == "tied_rescaled_missing":
        overrides.update(
            encoder_mode="tied",
            encoder_fusion="availability_rescaled_sum",
        )
    elif case == "factorized_mean_missing":
        overrides.update(site_rank=2, encoder_fusion="mean")

    actual_model = BlockCrosscoder(BSCConfig(**{**common, **overrides})).to(
        device=device,
        dtype=dtype,
    )
    with torch.no_grad():
        actual_model.c.copy_(
            torch.linspace(
                -0.15,
                0.25,
                actual_model.c.numel(),
                device=device,
                dtype=dtype,
            ).reshape_as(actual_model.c)
        )
        assert actual_model.a is not None
        actual_model.a.copy_(
            torch.linspace(
                -0.05,
                0.10,
                actual_model.a.numel(),
                device=device,
                dtype=dtype,
            ).reshape_as(actual_model.a)
        )
    legacy_model = copy.deepcopy(actual_model)
    generator = torch.Generator(device="cpu").manual_seed(919)
    x = torch.randn(batch_tokens, 4, 16, generator=generator).to(
        device=device,
        dtype=dtype,
    )

    actual, actual_decoder, actual_encoder = actual_model.forward_with_materialized(
        x,
        observed=observed,
    )
    expected, expected_decoder, expected_encoder = _legacy_per_site_forward(
        legacy_model,
        x,
        observed,
    )
    actual_loss = bsc_loss(
        actual,
        x,
        actual_model,
        observation_mask=observed,
        decoder=actual_decoder,
        encoder=actual_encoder,
    )["total"]
    expected_loss = bsc_loss(
        expected,
        x,
        legacy_model,
        observation_mask=observed,
        decoder=expected_decoder,
        encoder=expected_encoder,
    )["total"]
    actual_loss.backward()
    expected_loss.backward()

    output_drifts = {
        "code": _relative_l2(actual.z, expected.z),
        "score": _relative_l2(actual.scores, expected.scores),
        "selected_code": _relative_l2(actual.z_selected, expected.z_selected),
        "reconstruction": _relative_l2(actual.xhat, expected.xhat),
        "loss": abs(float((actual_loss - expected_loss).detach()))
        / max(abs(float(expected_loss.detach())), 1e-12),
    }
    support_disagreement = float((actual.mask != expected.mask).sum()) / max(
        int(actual.mask.sum() + expected.mask.sum()),
        1,
    )
    actual_parameters = dict(actual_model.named_parameters())
    expected_parameters = dict(legacy_model.named_parameters())
    assert actual_parameters.keys() == expected_parameters.keys()
    gradient_drifts = {}
    for name in actual_parameters:
        actual_gradient = actual_parameters[name].grad
        expected_gradient = expected_parameters[name].grad
        assert actual_gradient is not None, name
        assert expected_gradient is not None, name
        gradient_drifts[name] = _relative_l2(actual_gradient, expected_gradient)

    if dtype == torch.float32:
        assert max(output_drifts.values()) < 2e-6, output_drifts
        assert support_disagreement == 0.0
        assert max(gradient_drifts.values()) < 5e-6, gradient_drifts
    else:
        # The old kernel rounded each site's GEMM before summing; the flattened
        # kernel rounds once after contracting sites and coordinates together.
        # These bounds make that authorized difference explicit without
        # pretending bf16 support is bitwise invariant.
        assert output_drifts["code"] < 6e-3, output_drifts
        assert output_drifts["score"] < 6e-3, output_drifts
        assert output_drifts["selected_code"] < 0.2, output_drifts
        assert output_drifts["reconstruction"] < 0.2, output_drifts
        assert output_drifts["loss"] < 3e-3, output_drifts
        assert support_disagreement < 0.02, support_disagreement
        assert max(gradient_drifts.values()) < 0.25, gradient_drifts


@pytest.mark.parametrize("selection", ("token_topk", "batch_topk"))
@pytest.mark.parametrize(
    "selection_score",
    ("code_norm", "decoder_weighted", "decoded_energy", "isolated_loss_decrease"),
)
def test_flattened_bf16_encoder_bounds_selector_and_gradient_drift(
    device, selection, selection_score
):
    """Quantify drift through every score geometry and hard TopK selector."""
    config = dict(
        n_blocks=128,
        block_dim=4,
        n_sites=4,
        d_model=64,
        k=8,
        seed=811,
        selection=selection,
        decoder_constraint="free",
        encoder_bias=True,
        decoder_bias=True,
        selection_score=selection_score,
    )
    if selection_score == "decoder_weighted":
        config["code_activation"] = "relu"
    elif selection_score == "isolated_loss_decrease":
        config["decoder_bias"] = False
    cfg = BSCConfig(**config)
    actual_model = BlockCrosscoder(cfg).to(device=device, dtype=torch.bfloat16)
    legacy_model = copy.deepcopy(actual_model)
    x = torch.randn(
        256,
        cfg.n_sites,
        cfg.d_model,
        generator=torch.Generator(device="cpu").manual_seed(812),
    ).to(device=device, dtype=torch.bfloat16)
    actual, actual_decoder, actual_encoder = actual_model.forward_with_materialized(x)
    expected, expected_decoder, expected_encoder = _legacy_per_site_forward(
        legacy_model,
        x,
        None,
    )
    actual_loss = bsc_loss(
        actual,
        x,
        actual_model,
        decoder=actual_decoder,
        encoder=actual_encoder,
    )["total"]
    expected_loss = bsc_loss(
        expected,
        x,
        legacy_model,
        decoder=expected_decoder,
        encoder=expected_encoder,
    )["total"]
    actual_loss.backward()
    expected_loss.backward()

    changed_event_fraction = float((actual.mask != expected.mask).sum()) / float(
        actual.mask.sum() + expected.mask.sum()
    )
    expected_parameters = dict(legacy_model.named_parameters())
    gradient_drifts = {}
    for name, parameter in actual_model.named_parameters():
        actual_gradient = parameter.grad
        expected_gradient = expected_parameters[name].grad
        if actual_gradient is None or expected_gradient is None:
            assert actual_gradient is None and expected_gradient is None, name
            continue
        gradient_drifts[name] = _relative_l2(actual_gradient, expected_gradient)
    drift = {
        "code": _relative_l2(actual.z, expected.z),
        "score": _relative_l2(actual.scores, expected.scores),
        "changed_event_fraction": changed_event_fraction,
        "selected_code": _relative_l2(actual.z_selected, expected.z_selected),
        "reconstruction": _relative_l2(actual.xhat, expected.xhat),
        "loss": abs(float((actual_loss - expected_loss).detach()))
        / abs(float(expected_loss.detach())),
        "gradient_max": max(gradient_drifts.values()),
    }
    assert drift["code"] < 6e-3, drift
    assert drift["score"] < 6e-3, drift
    assert drift["changed_event_fraction"] < 0.02, drift
    assert drift["selected_code"] < 0.2, drift
    assert drift["reconstruction"] < 0.2, drift
    assert drift["loss"] < 3e-3, drift
    assert drift["gradient_max"] < 0.25, {**drift, "gradients": gradient_drifts}


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
    for expected_tensor, selected_tensor in zip(actual[1:], selected, strict=True):
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
def test_frozen_encoder_sites_track_flattened_training_views(
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
                if observed is None or actual.dtype == torch.bool:
                    assert torch.equal(actual, expected)
                else:
                    # Frozen multi-view evaluation intentionally retains the
                    # per-site contractions it must cache, while training now
                    # fuses sites into one GEMM.  Their fp32 reduction orders
                    # differ without changing the bound view semantics.
                    torch.testing.assert_close(
                        actual,
                        expected,
                        rtol=2e-5,
                        atol=2e-6,
                    )


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA Inductor")
@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_compiled_cuda_selectors_match_eager_over_adversarial_surfaces(dtype):
    device = torch.device("cuda")
    batch, groups = 512, 2048
    assert batch * groups == model_module._CUDA_SELECTOR_FUSION_MIN_ELEMENTS
    generator = torch.Generator(device="cpu").manual_seed(1881)
    random_scores = torch.randn(batch, groups, generator=generator).to(
        device=device,
        dtype=dtype,
    )
    repeated_signed = torch.arange(batch * groups, device=device) % 17 - 8
    repeated_signed = repeated_signed.reshape(batch, groups).to(dtype)
    nonfinite = random_scores.clone()
    nonfinite[0, :3] = torch.tensor(
        (float("nan"), float("inf"), float("-inf")),
        device=device,
        dtype=dtype,
    )
    noncontiguous = (
        torch.randn(groups, batch, generator=generator)
        .to(
            device=device,
            dtype=dtype,
        )
        .transpose(0, 1)
    )
    assert not noncontiguous.is_contiguous()
    patterns = {
        "random": random_scores,
        "all_equal": torch.ones_like(random_scores),
        "repeated_signed": repeated_signed,
        "nonfinite": nonfinite,
        "noncontiguous": noncontiguous,
    }

    for name, scores in patterns.items():
        for k in (7, 7.4):
            n_keep = min(max(round(k), 0), groups)
            expected = model_module._eager_token_topk_interior(scores, n_keep)
            actual = token_topk_mask(scores, k)
            assert torch.equal(actual, expected), (name, k)
        for k in (3.0, 0.5):
            n_keep = min(round(k * batch), batch * groups)
            expected = model_module._eager_batch_topk_interior(scores, n_keep)
            actual = batch_topk_mask(scores, k)
            assert torch.equal(actual, expected), (name, k)

    # Bind the secondary ordering independently of the eager implementation.
    tied = patterns["all_equal"]
    token_mask = token_topk_mask(tied, 7)
    assert token_mask[:, :7].all() and not token_mask[:, 7:].any()
    batch_mask = batch_topk_mask(tied, 0.5).reshape(-1)
    assert batch_mask[: batch // 2].all() and not batch_mask[batch // 2 :].any()

    # Torch 2.8 specializes Python integer arguments under static compilation
    # and refuses after eight graph variants. Bind enough distinct budgets to
    # prove the dynamic n_keep graph remains exact beyond that failure mode.
    for n_keep in range(1, 37):
        assert torch.equal(
            token_topk_mask(random_scores, n_keep),
            model_module._eager_token_topk_interior(random_scores, n_keep),
        )
        batch_k = n_keep / batch
        assert round(batch_k * batch) == n_keep
        assert torch.equal(
            batch_topk_mask(random_scores, batch_k),
            model_module._eager_batch_topk_interior(random_scores, n_keep),
        )


def test_cuda_selector_compile_wrappers_are_lazy_cached_and_boundary_gated(
    device, monkeypatch
):
    compile_calls: list[tuple[str, dict[str, object]]] = []
    compiled_calls = {"batch": 0, "token": 0}

    def fake_compile(function, **options):
        name = "batch" if "batch" in function.__name__ else "token"
        compile_calls.append((name, options))

        def compiled(*args):
            compiled_calls[name] += 1
            return function(*args)

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    model_module._compiled_cuda_batch_topk_interior.cache_clear()
    model_module._compiled_cuda_token_topk_interior.cache_clear()
    try:
        batch, groups = 512, 2048
        large = torch.arange(batch * groups, device=device).reshape(batch, groups)
        large = large.to(torch.float32)
        for _ in range(2):
            token_topk_mask(large, 7)
            batch_topk_mask(large, 3.0)

        # Small tensors and public zero/all boundary cases never enter either
        # compiled interior.
        small = large[:16, :16]
        token_topk_mask(small, 3)
        batch_topk_mask(small, 3.0)
        token_topk_mask(large, 0)
        token_topk_mask(large, groups)
        batch_topk_mask(large, 0)
        batch_topk_mask(large, groups)

        if device.type == "cuda":
            options = {
                "backend": "inductor",
                "fullgraph": True,
                "dynamic": True,
            }
            assert compile_calls == [("token", options), ("batch", options)]
            assert compiled_calls == {"batch": 2, "token": 2}
        else:
            assert compile_calls == []
            assert compiled_calls == {"batch": 0, "token": 0}
    finally:
        model_module._compiled_cuda_batch_topk_interior.cache_clear()
        model_module._compiled_cuda_token_topk_interior.cache_clear()


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
            if actual.dtype == torch.bool:
                assert torch.equal(actual, expected)
            else:
                torch.testing.assert_close(
                    actual,
                    expected,
                    rtol=2e-5,
                    atol=2e-6,
                )
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


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_stiefel_code_norm_score_specialization_preserves_partial_view_support(
    device,
    dtype,
):
    if dtype == torch.bfloat16 and device.type != "cuda":
        pytest.skip("the bounded bf16 specialization is a CUDA forward path")
    cfg_values = {
        "n_blocks": 32,
        "block_dim": 3,
        "n_sites": 3,
        "d_model": 32,
        "site_dims": (32, 28, 24),
        "k": 4,
        "seed": 493,
        "selection": "token_topk",
        "encoder_mode": "tied",
        "encoder_fusion": "availability_rescaled_sum",
        "decoder_constraint": "gram",
        "selection_score": "decoded_energy",
    }
    exact = BlockCrosscoder(
        BSCConfig(
            **cfg_values,
            decoded_energy_implementation=DECODED_ENERGY_EXACT_IMPLEMENTATION,
        )
    ).to(device=device, dtype=dtype)
    fast = BlockCrosscoder(
        BSCConfig(
            **cfg_values,
            decoded_energy_implementation=(
                DECODED_ENERGY_STIEFEL_CODE_NORM_IMPLEMENTATION
            ),
        )
    ).to(device=device, dtype=dtype)
    fast.load_state_dict(exact.state_dict())
    fast.validate_decoded_energy_implementation()

    generator = torch.Generator().manual_seed(494)
    calibration = torch.randn(96, 3, 32, generator=generator).to(
        device=device,
        dtype=dtype,
    )
    exact.fit_threshold_([calibration], 4.0, method="exact")
    fast.fit_threshold_([calibration], 4.0, method="exact")
    threshold_scale = max(abs(float(exact.theta)), 1e-12)
    threshold_rtol = 2e-5 if dtype == torch.float32 else 5e-3
    assert abs(float(fast.theta - exact.theta)) / threshold_scale <= threshold_rtol

    observations = (
        None,
        torch.tensor([[True, False, False]], device=device).expand(96, -1),
        torch.tensor([[False, True, True]], device=device).expand(96, -1),
    )
    for observed in observations:
        exact_z = exact.encode(calibration, observed=observed)
        fast_z = fast.encode(calibration, observed=observed)
        torch.testing.assert_close(fast_z, exact_z, rtol=0, atol=0)
        exact_scores = exact.scores(exact_z)
        fast_scores = fast.scores(fast_z)
        score_drift = _relative_l2(fast_scores, exact_scores)
        assert score_drift <= (2e-6 if dtype == torch.float32 else 2e-3)
        for mode in ("topk", "threshold"):
            exact_mask = exact._select_scores(exact_scores, mode=mode, z=exact_z)
            fast_mask = fast._select_scores(fast_scores, mode=mode, z=fast_z)
            if dtype == torch.float32:
                assert torch.equal(fast_mask, exact_mask)
            else:
                disagreement = float((fast_mask != exact_mask).float().mean())
                intersection = int((fast_mask & exact_mask).sum())
                union = int((fast_mask | exact_mask).sum())
                assert disagreement <= 1e-3
                assert intersection / max(union, 1) >= 0.99


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("selection", ("token_topk", "batch_topk"))
@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_cuda_stiefel_score_specialization_bounds_forward_loss_and_gradients(
    selection,
    dtype,
):
    cfg_values = {
        "n_blocks": 64,
        "block_dim": 4,
        "n_sites": 4,
        "d_model": 128,
        "site_dims": (128, 120, 112, 104),
        "k": 8,
        "seed": 495,
        "selection": selection,
        "encoder_mode": "tied",
        "encoder_fusion": "availability_rescaled_sum",
        "decoder_constraint": "gram",
        "selection_score": "decoded_energy",
    }
    exact = BlockCrosscoder(
        BSCConfig(
            **cfg_values,
            decoded_energy_implementation=DECODED_ENERGY_EXACT_IMPLEMENTATION,
        ),
        device="cuda",
    ).to(dtype=dtype)
    fast = BlockCrosscoder(
        BSCConfig(
            **cfg_values,
            decoded_energy_implementation=(
                DECODED_ENERGY_STIEFEL_CODE_NORM_IMPLEMENTATION
            ),
        ),
        device="cuda",
    ).to(dtype=dtype)
    fast.load_state_dict(exact.state_dict())
    fast.validate_decoded_energy_implementation()
    x = torch.randn(
        128,
        4,
        128,
        generator=torch.Generator().manual_seed(496),
        device="cpu",
    ).to(device="cuda", dtype=dtype)

    exact_out, exact_decoder, exact_encoder = exact.forward_with_materialized(x)
    fast_out, fast_decoder, fast_encoder = fast.forward_with_materialized(x)
    exact_loss = bsc_loss(
        exact_out,
        x,
        exact,
        decoder=exact_decoder,
        encoder=exact_encoder,
    )["total"]
    fast_loss = bsc_loss(
        fast_out,
        x,
        fast,
        decoder=fast_decoder,
        encoder=fast_encoder,
    )["total"]
    exact_loss.backward()
    fast_loss.backward()

    score_drift = _relative_l2(fast_out.scores, exact_out.scores)
    output_drift = _relative_l2(fast_out.xhat, exact_out.xhat)
    loss_drift = abs(float((fast_loss - exact_loss).detach())) / max(
        abs(float(exact_loss.detach())), 1e-12
    )
    mask_disagreement = float((fast_out.mask != exact_out.mask).float().mean())
    intersection = int((fast_out.mask & exact_out.mask).sum())
    union = int((fast_out.mask | exact_out.mask).sum())
    support_iou = intersection / max(union, 1)
    exact_parameters = dict(exact.named_parameters())
    gradient_drifts = {}
    for name, parameter in fast.named_parameters():
        actual_gradient = parameter.grad
        expected_gradient = exact_parameters[name].grad
        if actual_gradient is None or expected_gradient is None:
            assert actual_gradient is None and expected_gradient is None, name
            continue
        gradient_drifts[name] = _relative_l2(actual_gradient, expected_gradient)

    if dtype == torch.float32:
        assert score_drift <= 2e-6
        assert mask_disagreement == 0.0
        assert output_drift <= 2e-6
        assert loss_drift <= 2e-6
        assert max(gradient_drifts.values()) <= 2e-6
    else:
        assert score_drift <= 2e-3
        assert mask_disagreement <= 1e-3
        assert support_iou >= 0.99
        assert output_drift <= 0.05
        assert loss_drift <= 1e-4
        assert max(gradient_drifts.values()) <= 0.06


def test_stiefel_code_norm_frozen_geometry_omits_decoder_gram_and_refuses_drift():
    model = BlockCrosscoder(
        BSCConfig(
            n_blocks=16,
            block_dim=2,
            n_sites=2,
            d_model=6,
            site_dims=(6, 4),
            k=3,
            selection="batch_topk",
            encoder_mode="tied",
            decoder_constraint="qr",
            selection_score="decoded_energy",
            decoded_energy_implementation=(
                DECODED_ENERGY_STIEFEL_CODE_NORM_IMPLEMENTATION
            ),
        )
    )
    with torch.no_grad():
        decoder = model.decoder_tensor()
        geometry = model._frozen_score_geometry(decoder)
        assert geometry.decoder_gram is None
        assert geometry.decoder_weight is None
        assert geometry.site_decoder_gram is None
        model.D.mul_(1.1)
    with pytest.raises(RuntimeError, match="invariant failed"):
        model.validate_decoded_energy_implementation()


@pytest.mark.parametrize(
    "overrides",
    (
        {"selection_score": "code_norm"},
        {"decoder_constraint": "free"},
        {"selection": "threshold"},
        {"site_rank": 1, "encoder_mode": "untied", "decoder_constraint": "free"},
    ),
)
def test_stiefel_code_norm_config_fails_closed_outside_complete_carrier(overrides):
    values = {
        "n_blocks": 8,
        "block_dim": 2,
        "n_sites": 2,
        "d_model": 5,
        "k": 2,
        "selection": "token_topk",
        "decoder_constraint": "gram",
        "selection_score": "decoded_energy",
        "decoded_energy_implementation": (
            DECODED_ENERGY_STIEFEL_CODE_NORM_IMPLEMENTATION
        ),
        **overrides,
    }
    with pytest.raises(ValueError, match="stiefel code-norm decoded energy"):
        BSCConfig(**values)


@pytest.mark.parametrize(
    "implementation",
    (ISOLATED_LOSS_EXACT_IMPLEMENTATION, ISOLATED_LOSS_MAPPED_IMPLEMENTATION),
)
def test_isolated_loss_decrease_matches_explicit_candidate_reconstructions(
    implementation,
):
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
            isolated_loss_decrease_implementation=implementation,
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
    energy_sq = torch.zeros(z_reference.shape[:2], dtype=torch.float32, device=device)
    for site in range(config.n_sites):
        projected = projected + torch.einsum(
            "nd,gbd->ngb", x_reference[:, site].float(), decoder[site]
        )
        site_gram = torch.einsum("gbd,gcd->gbc", decoder[site], decoder[site])
        site_energy = torch.einsum("ngb,gbc,ngc->ng", code, site_gram, code)
        energy_sq = energy_sq + site_energy
    expected = 2.0 * (projected * code).sum(dim=-1) - energy_sq
    expected_grads = torch.autograd.grad(
        expected.sum(), (x_reference, z_reference, reference_model.D)
    )

    assert torch.equal(actual, expected)
    assert all(
        torch.equal(actual_grad, expected_grad)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True)
    )


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
@pytest.mark.parametrize("view", ("all", "partial", "source"))
def test_mapped_isolated_loss_decrease_bounds_score_and_gradient_drift(
    device,
    dtype,
    view,
):
    if dtype == torch.bfloat16 and device.type != "cuda":
        pytest.skip("the bounded bf16 specialization is a CUDA forward path")
    cfg_values = {
        "n_blocks": 32,
        "block_dim": 4,
        "n_sites": 4,
        "d_model": 32,
        "site_dims": (32, 29, 26, 23),
        "k": 4,
        "seed": 984,
        "selection": "token_topk",
        "encoder_mode": "untied",
        "encoder_fusion": "source" if view == "source" else "availability_rescaled_sum",
        "source_site": 1,
        "decoder_constraint": "free",
        "decoder_bias": False,
        "reconstruction_loss": "squared_l2",
        "selection_score": "isolated_loss_decrease",
    }
    exact = BlockCrosscoder(
        BSCConfig(
            **cfg_values,
            isolated_loss_decrease_implementation=(ISOLATED_LOSS_EXACT_IMPLEMENTATION),
        )
    )
    with torch.no_grad():
        assert exact.D is not None
        scale = torch.linspace(0.65, 1.35, exact.cfg.n_blocks).view(1, -1, 1, 1)
        exact.D.mul_(scale)
    mapped = BlockCrosscoder(
        BSCConfig(
            **cfg_values,
            isolated_loss_decrease_implementation=(ISOLATED_LOSS_MAPPED_IMPLEMENTATION),
        )
    )
    mapped.load_state_dict(exact.state_dict())
    exact = exact.to(device=device, dtype=dtype)
    mapped = mapped.to(device=device, dtype=dtype)
    generator = torch.Generator().manual_seed(985)
    x_exact = (
        torch.randn(96, 4, 32, generator=generator)
        .to(
            device=device,
            dtype=dtype,
        )
        .requires_grad_()
    )
    z_exact = (
        torch.randn(96, 32, 4, generator=generator)
        .to(
            device=device,
            dtype=dtype,
        )
        .requires_grad_()
    )
    x_mapped = x_exact.detach().clone().requires_grad_()
    z_mapped = z_exact.detach().clone().requires_grad_()
    observed = None
    if view == "partial":
        observed = torch.ones(96, 4, dtype=torch.bool, device=device)
        observed[::2, 0] = False
        observed[1::3, 2] = False

    exact_scores = exact.scores(z_exact, x=x_exact, observed=observed)
    mapped_scores = mapped.scores(z_mapped, x=x_mapped, observed=observed)
    assert exact.D is not None and mapped.D is not None
    exact_grads = torch.autograd.grad(
        exact_scores.sum(),
        (x_exact, z_exact, exact.D),
    )
    mapped_grads = torch.autograd.grad(
        mapped_scores.sum(),
        (x_mapped, z_mapped, mapped.D),
    )
    score_drift = _relative_l2(mapped_scores, exact_scores)
    gradient_drifts = [
        _relative_l2(actual, expected)
        for actual, expected in zip(mapped_grads, exact_grads, strict=True)
    ]
    exact_mask = exact._select_scores(exact_scores, mode="topk", z=z_exact)
    mapped_mask = mapped._select_scores(mapped_scores, mode="topk", z=z_mapped)
    disagreement = float((mapped_mask != exact_mask).float().mean())
    intersection = int((mapped_mask & exact_mask).sum())
    union = int((mapped_mask | exact_mask).sum())
    if dtype == torch.float32:
        assert score_drift <= 2e-6
        assert disagreement == 0.0
        assert max(gradient_drifts) <= 5e-6
    else:
        assert score_drift <= 2e-3
        assert disagreement <= 1e-3
        assert intersection / max(union, 1) >= 0.99
        assert max(gradient_drifts) <= 0.02


@pytest.mark.parametrize("view", ("all", "partial"))
def test_mapped_isolated_loss_frozen_geometry_binds_cached_map_and_grams(view):
    model = BlockCrosscoder(
        BSCConfig(
            n_blocks=12,
            block_dim=3,
            n_sites=3,
            d_model=9,
            site_dims=(9, 7, 5),
            k=3,
            decoder_constraint="free",
            decoder_bias=False,
            reconstruction_loss="squared_l2",
            selection_score="isolated_loss_decrease",
            isolated_loss_decrease_implementation=(ISOLATED_LOSS_MAPPED_IMPLEMENTATION),
        )
    )
    x = torch.randn(31, 3, 9, generator=torch.Generator().manual_seed(986))
    z = torch.randn(31, 12, 3, generator=torch.Generator().manual_seed(987))
    observed = None
    if view == "partial":
        observed = torch.ones(31, 3, dtype=torch.bool)
        observed[::2, 0] = False
    with torch.no_grad():
        decoder = model.decoder_tensor()
        geometry = model._frozen_score_geometry(decoder)
        assert geometry.isolated_loss_decoder_map is not None
        assert geometry.site_decoder_gram is not None
        assert geometry.isolated_loss_all_site_gram is not None
        direct = model.scores(z, x=x, observed=observed, _decoder=decoder)
        cached = model.scores(
            z,
            x=x,
            observed=observed,
            _decoder=decoder,
            _score_geometry=geometry,
        )
        assert torch.equal(cached, direct)
        with pytest.raises(ValueError, match="not bound"):
            model.scores(
                z,
                x=x,
                observed=observed,
                _decoder=decoder.clone(),
                _score_geometry=geometry,
            )


def test_mapped_isolated_loss_preserves_factorized_free_decoder_gradients():
    cfg_values = {
        "n_blocks": 10,
        "block_dim": 3,
        "n_sites": 4,
        "d_model": 12,
        "k": 3,
        "seed": 990,
        "selection": "batch_topk",
        "encoder_mode": "untied",
        "decoder_constraint": "free",
        "decoder_bias": False,
        "reconstruction_loss": "squared_l2",
        "selection_score": "isolated_loss_decrease",
        "site_rank": 2,
    }
    exact = BlockCrosscoder(
        BSCConfig(
            **cfg_values,
            isolated_loss_decrease_implementation=(ISOLATED_LOSS_EXACT_IMPLEMENTATION),
        )
    )
    mapped = BlockCrosscoder(
        BSCConfig(
            **cfg_values,
            isolated_loss_decrease_implementation=(ISOLATED_LOSS_MAPPED_IMPLEMENTATION),
        )
    )
    mapped.load_state_dict(exact.state_dict())
    generator = torch.Generator().manual_seed(991)
    x_exact = torch.randn(41, 4, 12, generator=generator).requires_grad_()
    z_exact = torch.randn(41, 10, 3, generator=generator).requires_grad_()
    x_mapped = x_exact.detach().clone().requires_grad_()
    z_mapped = z_exact.detach().clone().requires_grad_()
    observed = torch.ones(41, 4, dtype=torch.bool)
    observed[::2, 1] = False

    exact_scores = exact.scores(z_exact, x=x_exact, observed=observed)
    mapped_scores = mapped.scores(z_mapped, x=x_mapped, observed=observed)
    assert exact.D_site is not None and exact.D_core is not None
    assert mapped.D_site is not None and mapped.D_core is not None
    exact_grads = torch.autograd.grad(
        exact_scores.sum(),
        (x_exact, z_exact, exact.D_site, exact.D_core),
    )
    mapped_grads = torch.autograd.grad(
        mapped_scores.sum(),
        (x_mapped, z_mapped, mapped.D_site, mapped.D_core),
    )
    assert _relative_l2(mapped_scores, exact_scores) <= 2e-6
    assert (
        max(
            _relative_l2(actual, expected)
            for actual, expected in zip(mapped_grads, exact_grads, strict=True)
        )
        <= 5e-6
    )


@pytest.mark.parametrize(
    "overrides,match",
    (
        (
            {"isolated_loss_decrease_implementation": "runtime_default"},
            "isolated_loss_decrease_implementation",
        ),
        (
            {
                "selection_score": "code_norm",
                "isolated_loss_decrease_implementation": (
                    ISOLATED_LOSS_MAPPED_IMPLEMENTATION
                ),
            },
            "mapped isolated-loss decrease",
        ),
        (
            {
                "decoder_constraint": "gram",
                "isolated_loss_decrease_implementation": (
                    ISOLATED_LOSS_MAPPED_IMPLEMENTATION
                ),
            },
            "mapped isolated-loss decrease",
        ),
    ),
)
def test_mapped_isolated_loss_config_fails_closed(overrides, match):
    values = {
        "n_blocks": 8,
        "block_dim": 2,
        "n_sites": 2,
        "d_model": 5,
        "k": 2,
        "selection_score": "isolated_loss_decrease",
        "decoder_constraint": "free",
        "decoder_bias": False,
        "reconstruction_loss": "squared_l2",
        **overrides,
    }
    with pytest.raises(ValueError, match=match):
        BSCConfig(**values)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
@pytest.mark.parametrize("selection", ("token_topk", "batch_topk"))
def test_cuda_mapped_isolated_loss_campaign_shape_bounds_full_step_drift(
    dtype,
    selection,
):
    cfg_values = {
        "n_blocks": 256,
        "block_dim": 4,
        "n_sites": 4,
        "d_model": 128,
        "k": 4,
        "seed": 988,
        "selection": selection,
        "encoder_mode": "untied",
        "encoder_fusion": "availability_rescaled_sum",
        "decoder_constraint": "free",
        "decoder_bias": False,
        "reconstruction_loss": "squared_l2",
        "selection_score": "isolated_loss_decrease",
    }
    exact = BlockCrosscoder(
        BSCConfig(
            **cfg_values,
            isolated_loss_decrease_implementation=(ISOLATED_LOSS_EXACT_IMPLEMENTATION),
        ),
        device="cuda",
    )
    with torch.no_grad():
        assert exact.D is not None
        scale = torch.linspace(0.6, 1.4, exact.cfg.n_blocks, device="cuda")
        exact.D.mul_(scale.view(1, -1, 1, 1))
    mapped = BlockCrosscoder(
        BSCConfig(
            **cfg_values,
            isolated_loss_decrease_implementation=(ISOLATED_LOSS_MAPPED_IMPLEMENTATION),
        ),
        device="cuda",
    )
    mapped.load_state_dict(exact.state_dict())
    exact = exact.to(dtype=dtype)
    mapped = mapped.to(dtype=dtype)
    x = torch.randn(
        8192,
        4,
        128,
        generator=torch.Generator().manual_seed(989),
    ).to(device="cuda", dtype=dtype)

    exact_out, exact_decoder, exact_encoder = exact.forward_with_materialized(x)
    mapped_out, mapped_decoder, mapped_encoder = mapped.forward_with_materialized(x)
    exact_loss = bsc_loss(
        exact_out,
        x,
        exact,
        decoder=exact_decoder,
        encoder=exact_encoder,
    )["total"]
    mapped_loss = bsc_loss(
        mapped_out,
        x,
        mapped,
        decoder=mapped_decoder,
        encoder=mapped_encoder,
    )["total"]
    exact_loss.backward()
    mapped_loss.backward()

    score_drift = _relative_l2(mapped_out.scores, exact_out.scores)
    output_drift = _relative_l2(mapped_out.xhat, exact_out.xhat)
    loss_drift = abs(float((mapped_loss - exact_loss).detach())) / max(
        abs(float(exact_loss.detach())), 1e-12
    )
    mask_disagreement = float((mapped_out.mask != exact_out.mask).float().mean())
    intersection = int((mapped_out.mask & exact_out.mask).sum())
    union = int((mapped_out.mask | exact_out.mask).sum())
    exact_parameters = dict(exact.named_parameters())
    gradient_drifts = {}
    for name, parameter in mapped.named_parameters():
        actual_gradient = parameter.grad
        expected_gradient = exact_parameters[name].grad
        if actual_gradient is None or expected_gradient is None:
            assert actual_gradient is None and expected_gradient is None, name
            continue
        gradient_drifts[name] = _relative_l2(actual_gradient, expected_gradient)

    if dtype == torch.float32:
        assert score_drift <= 2e-6
        assert mask_disagreement <= 1e-6
        assert output_drift <= 2e-4
        assert loss_drift <= 2e-6
        assert max(gradient_drifts.values()) <= 2e-4
    else:
        assert score_drift <= 2e-3
        assert mask_disagreement <= 1e-3
        assert intersection / max(union, 1) >= 0.99
        assert output_drift <= 0.05
        assert loss_drift <= 1e-4
        assert max(gradient_drifts.values()) <= 0.06


@pytest.mark.parametrize(
    "implementation",
    (ISOLATED_LOSS_EXACT_IMPLEMENTATION, ISOLATED_LOSS_MAPPED_IMPLEMENTATION),
)
def test_isolated_loss_decrease_excludes_hidden_clean_targets(implementation):
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
            isolated_loss_decrease_implementation=implementation,
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


@pytest.mark.parametrize(
    "implementation",
    (ISOLATED_LOSS_EXACT_IMPLEMENTATION, ISOLATED_LOSS_MAPPED_IMPLEMENTATION),
)
def test_isolated_loss_decrease_preserves_harmful_negative_scores(implementation):
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
            isolated_loss_decrease_implementation=implementation,
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


@pytest.mark.parametrize(
    "implementation",
    (ISOLATED_LOSS_EXACT_IMPLEMENTATION, ISOLATED_LOSS_MAPPED_IMPLEMENTATION),
)
def test_isolated_loss_decrease_streaming_threshold_matches_exact(
    device,
    implementation,
):
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
            isolated_loss_decrease_implementation=implementation,
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
