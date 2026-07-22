"""Numeric checks that the Gram constraint has its four required properties:
scale-gauge death, O(b)-invariant spectra, exact selection scores, and
free per-site Frobenius shares."""

import pytest
import torch

import block_crosscoder_experiment.gram as gram_module

from block_crosscoder_experiment.gram import (
    block_gram,
    cholesky_qr_retract_,
    decoder_nuclear_penalty,
    factorized_decoder_nuclear_penalty,
    factorized_map_nuclear_penalty,
    gram_residual,
    init_decoder_stack,
    map_nuclear_penalty,
    project_block_frobenius_,
    retract_,
    site_frobenius_shares,
    site_singular_values,
)
from block_crosscoder_experiment.runtime_limits import (
    CHOLESKY_QR_GRAM_CONDITION_MAX,
    MAP_NUCLEAR_EINSUM_REFERENCE_IMPLEMENTATION,
    MAP_NUCLEAR_GUARDED_MATMUL_IMPLEMENTATION,
)

S, G, B_DIM, D_MODEL = 4, 16, 4, 32


def random_stack(device, seed=0, scale=1.0):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    D = torch.randn(S, G, B_DIM, D_MODEL, generator=gen) * scale
    return D.to(device)


def random_orthogonal(n, device, seed=0):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(n, n, generator=gen))
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
    return q.to(device)


def test_retraction_satisfies_constraint(device):
    D = random_stack(device, scale=3.0)
    retract_(D)
    assert gram_residual(D).max().item() < 1e-5


def test_retraction_idempotent(device):
    D = random_stack(device)
    retract_(D)
    before = D.clone()
    retract_(D)
    assert (D - before).abs().max().item() < 1e-5


def test_retraction_kills_scale_gauge(device):
    """D and c*D retract to the same point — the z->cz gauge is dead."""
    D1 = random_stack(device)
    D2 = D1 * 7.3
    retract_(D1)
    retract_(D2)
    assert (D1 - D2).abs().max().item() < 1e-4


def test_retraction_requires_fp32(device):
    D = random_stack(device).to(torch.bfloat16)
    with pytest.raises(TypeError):
        retract_(D)


def test_retraction_floor_hits_on_deficient_block(device):
    D = random_stack(device)
    D[:, 0] = 0.0
    D[0, 0, 0, 0] = 1.0  # block 0: rank 1 across all sites
    floor_hits = retract_(D)
    assert floor_hits >= B_DIM - 1
    assert torch.isfinite(D).all()
    # Healthy blocks still land on the constraint.
    assert gram_residual(D)[1:].max().item() < 1e-5


def test_cholesky_qr_matches_positive_diagonal_householder(device):
    source = random_stack(device, seed=120, scale=0.7)
    householder = source.clone()
    cholesky = source.clone()

    gram_module.qr_retract_(householder)
    cholesky_qr_retract_(cholesky)

    torch.testing.assert_close(cholesky, householder, rtol=2e-6, atol=2e-6)
    assert float(gram_residual(cholesky).max()) <= 1e-4

    # Recover the canonical R from the returned Q and original input.  Its
    # diagonal is strictly positive by contract on every full-rank block.
    groups = source.shape[1]
    source_columns = source.permute(1, 0, 3, 2).reshape(groups, S * D_MODEL, B_DIM)
    q = householder.permute(1, 0, 3, 2).reshape(groups, S * D_MODEL, B_DIM)
    r = q.transpose(-1, -2) @ source_columns
    assert bool((torch.diagonal(r, dim1=-2, dim2=-1) > 0).all())
    torch.testing.assert_close(r, torch.triu(r), rtol=0, atol=2e-5)


def test_cholesky_qr_is_idempotent_and_positive_scale_invariant(device):
    source = random_stack(device, seed=121)
    scaled = source * 9.25
    cholesky_qr_retract_(source)
    cholesky_qr_retract_(scaled)
    torch.testing.assert_close(source, scaled, rtol=2e-6, atol=2e-6)
    before = source.clone()
    cholesky_qr_retract_(source)
    torch.testing.assert_close(source, before, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize(
    "retraction",
    (
        gram_module._cholesky_qr_retract_count_tensor_,
        gram_module._qr_retract_count_tensor_,
    ),
)
def test_qr_prevalidated_input_flag_preserves_exact_candidate(device, retraction):
    checked = random_stack(device, seed=1211)
    prevalidated = checked.clone()
    expected_count = retraction(checked)
    actual_count = retraction(prevalidated, input_finite=True)
    assert torch.equal(actual_count, expected_count)
    assert torch.equal(prevalidated, checked)


@pytest.mark.parametrize(
    "retraction",
    (
        gram_module._cholesky_qr_retract_count_tensor_,
        gram_module._qr_retract_count_tensor_,
    ),
)
def test_qr_prevalidated_input_still_refuses_nonfinite_candidate(
    device,
    retraction,
):
    source = random_stack(device, seed=1212)
    source[0, 0, 0, 0] = float("nan")
    before = source.clone()
    with pytest.raises(
        (ValueError, gram_module.CholeskyQRRetractionError),
        match="QR|candidate|Gram",
    ):
        retraction(source, input_finite=True)
    torch.testing.assert_close(source, before, equal_nan=True)


@pytest.mark.parametrize(
    "failure",
    (
        "nonfinite",
        "rank_deficient",
        "condition",
        "reconstruction",
        "candidate_nonfinite",
        "post_gram",
    ),
)
def test_cholesky_qr_failures_are_transactional_and_have_no_fallback(
    device,
    monkeypatch,
    failure,
):
    if failure == "condition":
        D = torch.zeros(1, 1, 4, 4, dtype=torch.float32, device=device)
        D[0, 0].copy_(torch.diag(torch.tensor([1.0, 0.1, 0.1, 0.1], device=device)))
        expected = "conditioning/reconstruction"
        gram = block_gram(D)
        assert float(torch.linalg.cond(gram, p=float("inf"))) > (
            CHOLESKY_QR_GRAM_CONDITION_MAX
        )
    else:
        D = random_stack(device, seed=122)
        if failure == "nonfinite":
            D[0, 0, 0, 0] = float("nan")
            expected = "finite"
        elif failure == "rank_deficient":
            D[:, 0].zero_()
            D[0, 0, 0, 0] = 1.0
            expected = "positive-definite"
        elif failure == "reconstruction":
            monkeypatch.setattr(
                gram_module,
                "CHOLESKY_QR_RECONSTRUCTION_RELATIVE_RESIDUAL_MAX",
                -1.0,
            )
            expected = "conditioning/reconstruction"
        elif failure == "candidate_nonfinite":
            original_bmm = torch.bmm

            def poisoned_bmm(input, mat2, *, out=None):
                result = original_bmm(input, mat2, out=out)
                assert out is not None
                if out.shape[-1] != out.shape[-2]:
                    out.fill_(float("inf"))
                return result

            monkeypatch.setattr(torch, "bmm", poisoned_bmm)
            expected = "post-Gram"
        else:
            assert failure == "post_gram"
            monkeypatch.setattr(
                gram_module,
                "CHOLESKY_QR_POST_GRAM_RESIDUAL_MAX",
                -1.0,
            )
            expected = "post-Gram"
    before = D.clone()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("fallback retraction was called")

    monkeypatch.setattr(torch.linalg, "qr", forbidden)
    monkeypatch.setattr(torch.linalg, "eigh", forbidden)
    with pytest.raises(RuntimeError, match=expected):
        cholesky_qr_retract_(D)
    assert torch.equal(torch.isnan(D), torch.isnan(before))
    assert torch.equal(torch.nan_to_num(D), torch.nan_to_num(before))


@pytest.mark.parametrize(
    ("failure", "expected"),
    (
        ("factor", "non-finite factors"),
        ("rank", "full-column-rank"),
        ("post_gram", "Gram bound"),
    ),
)
def test_householder_qr_combined_guard_failures_remain_transactional(
    device,
    monkeypatch,
    failure,
    expected,
):
    D = random_stack(device, seed=1221)
    if failure == "factor":
        original_qr = torch.linalg.qr

        def poisoned_qr(*args, **kwargs):
            q, r = original_qr(*args, **kwargs)
            q[(0,) * q.ndim] = float("inf")
            return q, r

        monkeypatch.setattr(torch.linalg, "qr", poisoned_qr)
    elif failure == "rank":
        D[:, 0].zero_()
    else:
        assert failure == "post_gram"
        monkeypatch.setattr(
            gram_module,
            "CHOLESKY_QR_POST_GRAM_RESIDUAL_MAX",
            -1.0,
        )
    before = D.clone()
    with pytest.raises(ValueError, match=expected):
        gram_module.qr_retract_(D)
    assert torch.equal(D, before)


def test_cholesky_qr_requires_fp32_and_sufficient_geometry(device):
    D = random_stack(device, seed=123).to(torch.bfloat16)
    with pytest.raises(TypeError, match="fp32"):
        cholesky_qr_retract_(D)
    too_narrow = torch.randn(1, 2, 4, 3, device=device)
    before = too_narrow.clone()
    with pytest.raises(ValueError, match="block_dim"):
        cholesky_qr_retract_(too_narrow)
    assert torch.equal(too_narrow, before)


def test_cuda_finite_guard_is_dynamic_cached_and_size_gated(device, monkeypatch):
    compile_calls = []
    compiled_calls = 0

    def fake_compile(function, **options):
        compile_calls.append(options)

        def compiled(*args):
            nonlocal compiled_calls
            compiled_calls += 1
            return function(*args)

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    gram_module._compiled_cuda_all_finite.cache_clear()
    try:
        n = gram_module._CUDA_FINITE_FUSION_MIN_ELEMENTS
        large = torch.ones(n, device=device)
        assert bool(gram_module._all_finite(large))
        assert bool(gram_module._all_finite(large))
        assert bool(gram_module._all_finite(large[:-1]))
        if device.type == "cuda":
            assert compile_calls == [
                {"backend": "inductor", "fullgraph": True, "dynamic": True}
            ]
            assert compiled_calls == 2
        else:
            assert compile_calls == []
            assert compiled_calls == 0
    finally:
        gram_module._compiled_cuda_all_finite.cache_clear()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA Inductor")
def test_compiled_cuda_finite_guard_handles_more_than_static_shape_limit():
    gram_module._compiled_cuda_all_finite.cache_clear()
    base = gram_module._CUDA_FINITE_FUSION_MIN_ELEMENTS
    try:
        for offset in range(9):
            value = torch.ones(base + offset, device="cuda")
            if offset % 3 == 1:
                value[-1] = float("nan")
            elif offset % 3 == 2:
                value[-1] = float("inf")
            assert bool(gram_module._all_finite(value)) is (offset % 3 == 0)
    finally:
        gram_module._compiled_cuda_all_finite.cache_clear()


def test_site_shares_sum_to_one_and_start_equal(device):
    D = init_decoder_stack(S, G, B_DIM, D_MODEL, device=device)
    shares = site_frobenius_shares(D)  # [S, G]
    assert torch.allclose(shares.sum(dim=0), torch.ones(G, device=device), atol=1e-5)
    # Gaussian init + one retraction: approximately equal shares (1/S).
    assert (shares.mean(dim=1) - 1 / S).abs().max().item() < 0.05


def site_exclusive_stack(device):
    """Constraint-satisfying stack with each code direction on one site:
    directions 0,1 -> site 0; directions 2,3 -> site 1 (b=4)."""
    gen = torch.Generator(device="cpu").manual_seed(3)
    D = torch.zeros(S, G, B_DIM, D_MODEL)
    for g in range(G):
        q, _ = torch.linalg.qr(torch.randn(D_MODEL, B_DIM, generator=gen))
        rows = q.T  # [b, d] orthonormal rows
        D[0, g, 0] = rows[0]
        D[0, g, 1] = rows[1]
        D[1, g, 2] = rows[2]
        D[1, g, 3] = rows[3]
    return D.to(device)


def test_unequal_shares_preserved(device):
    """The constraint fixes only the total; the depth profile is free."""
    D = site_exclusive_stack(device)
    assert gram_residual(D).max().item() < 1e-5
    before = D.clone()
    retract_(D)
    assert (D - before).abs().max().item() < 1e-5
    shares = site_frobenius_shares(D)
    expected = torch.tensor([0.5, 0.5, 0.0, 0.0], device=device)
    assert torch.allclose(shares[:, 0], expected, atol=1e-5)


def test_map_nuclear_matches_explicit_end_to_end_map(device):
    D = init_decoder_stack(S, G, B_DIM, D_MODEL, device=device)
    E = random_stack(device, seed=21)
    actual = map_nuclear_penalty(D, E, eps=0.0)
    explicit = []
    for g in range(G):
        dbar = D[:, g].permute(1, 0, 2).reshape(B_DIM, S * D_MODEL)
        ebar = E[:, g].permute(1, 0, 2).reshape(B_DIM, S * D_MODEL)
        explicit.append(torch.linalg.svdvals(dbar.T @ ebar).sum() / B_DIM)
    assert torch.allclose(actual, torch.stack(explicit).mean(), atol=1e-4)


def test_map_nuclear_matches_explicit_for_unconstrained_decoder(device):
    D = random_stack(device, seed=24, scale=0.3)
    E = random_stack(device, seed=25, scale=0.2)
    actual = map_nuclear_penalty(D, E, eps=0.0)
    explicit = []
    for g in range(G):
        dbar = D[:, g].permute(1, 0, 2).reshape(B_DIM, S * D_MODEL)
        ebar = E[:, g].permute(1, 0, 2).reshape(B_DIM, S * D_MODEL)
        explicit.append(torch.linalg.svdvals(dbar.T @ ebar).sum() / B_DIM)
    assert torch.allclose(actual, torch.stack(explicit).mean(), atol=2e-4)


def test_map_nuclear_exact_zero_smoothing_has_finite_grassmann_gradient(device):
    # The concatenated Gram constraint repeats every decoder-Gram eigenvalue
    # at one.  The exact SASA objective must therefore avoid eigendecomposition
    # eigenvector gradients, which are undefined at this intentional
    # degeneracy.
    D = init_decoder_stack(S, G, B_DIM, D_MODEL, device=device).requires_grad_()
    E = random_stack(device, seed=29).requires_grad_()
    loss = map_nuclear_penalty(D, E, eps=0.0)
    loss.backward()
    assert D.grad is not None and torch.isfinite(D.grad).all()
    assert E.grad is not None and torch.isfinite(E.grad).all()


def test_map_nuclear_accepts_rank_deficient_encoder(device):
    D = init_decoder_stack(S, G, B_DIM, D_MODEL, device=device)
    E = random_stack(device, seed=31)
    E[:, :, 1:] = 0.0
    actual = map_nuclear_penalty(D, E, eps=0.0)
    explicit = []
    for g in range(G):
        dbar = D[:, g].permute(1, 0, 2).reshape(B_DIM, S * D_MODEL)
        ebar = E[:, g].permute(1, 0, 2).reshape(B_DIM, S * D_MODEL)
        explicit.append(torch.linalg.svdvals(dbar.T @ ebar).sum() / B_DIM)
    assert torch.allclose(actual, torch.stack(explicit).mean(), atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
@pytest.mark.parametrize("sites", (1, 4, 6))
def test_guarded_map_nuclear_fast_path_has_bounded_reference_value_and_gradients(
    dtype,
    sites,
):
    generator = torch.Generator(device="cuda").manual_seed(5250 + sites)
    values = [
        torch.randn(
            sites,
            127,
            4,
            96,
            generator=generator,
            device="cuda",
            dtype=dtype,
        )
        .mul_(0.1)
        .requires_grad_()
        for _ in range(2)
    ]
    oracle_values = [value.detach().clone().requires_grad_() for value in values]
    actual = map_nuclear_penalty(
        *values,
        implementation=MAP_NUCLEAR_GUARDED_MATMUL_IMPLEMENTATION,
    )
    expected = map_nuclear_penalty(
        *oracle_values,
        implementation=MAP_NUCLEAR_EINSUM_REFERENCE_IMPLEMENTATION,
    )
    value_drift = (
        (actual - expected).detach().float().abs()
        / expected.detach().float().abs()
    )
    assert float(value_drift) <= 2e-6
    actual_gradients = torch.autograd.grad(actual, values)
    expected_gradients = torch.autograd.grad(expected, oracle_values)
    for actual_gradient, expected_gradient in zip(
        actual_gradients,
        expected_gradients,
        strict=True,
    ):
        difference = (actual_gradient.float() - expected_gradient.float()).norm()
        scale = expected_gradient.float().norm().clamp_min(1e-30)
        gradient_limit = 1e-4 if dtype == torch.float32 else 5e-4
        assert float(difference / scale) <= gradient_limit


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_guarded_map_nuclear_low_rank_fallback_is_reference_exact(dtype):
    generator = torch.Generator(device="cuda").manual_seed(5258)
    decoder = torch.randn(
        4,
        31,
        4,
        32,
        generator=generator,
        device="cuda",
        dtype=dtype,
    ).mul_(0.1)
    encoder = torch.zeros_like(decoder)
    encoder[:, :, 0] = torch.randn(
        4,
        31,
        32,
        generator=generator,
        device="cuda",
        dtype=dtype,
    ).mul_(0.1)
    actual_inputs = [value.requires_grad_() for value in (decoder, encoder)]
    reference_inputs = [
        value.detach().clone().requires_grad_() for value in actual_inputs
    ]
    actual = map_nuclear_penalty(
        *actual_inputs,
        implementation=MAP_NUCLEAR_GUARDED_MATMUL_IMPLEMENTATION,
    )
    expected = map_nuclear_penalty(
        *reference_inputs,
        implementation=MAP_NUCLEAR_EINSUM_REFERENCE_IMPLEMENTATION,
    )
    actual_gradients = torch.autograd.grad(actual, actual_inputs)
    expected_gradients = torch.autograd.grad(expected, reference_inputs)
    assert torch.equal(actual, expected)
    for actual_gradient, expected_gradient in zip(
        actual_gradients,
        expected_gradients,
        strict=True,
    ):
        assert torch.equal(actual_gradient, expected_gradient)


def test_guarded_map_nuclear_healthy_carrier_does_not_call_reference(
    monkeypatch,
    device,
):
    decoder = init_decoder_stack(S, G, B_DIM, D_MODEL, device=device)
    encoder = random_stack(device, seed=5259, scale=0.1)

    def refuse_reference(*args, **kwargs):
        raise AssertionError("healthy guarded map-nuclear carrier used reference")

    monkeypatch.setattr(gram_module.torch, "einsum", refuse_reference)
    result = map_nuclear_penalty(
        decoder,
        encoder,
        implementation=MAP_NUCLEAR_GUARDED_MATMUL_IMPLEMENTATION,
    )
    assert torch.isfinite(result)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_guarded_map_nuclear_near_singular_decoder_matches_reference_outcome(dtype):
    generator = torch.Generator(device="cuda").manual_seed(5262)
    decoder = torch.randn(
        4,
        31,
        4,
        32,
        generator=generator,
        device="cuda",
        dtype=dtype,
    ).mul_(0.1)
    decoder[:, :, 3] = decoder[:, :, 0] + torch.randn(
        4,
        31,
        32,
        generator=generator,
        device="cuda",
        dtype=dtype,
    ).mul_(1e-5)
    encoder = torch.randn_like(decoder)

    def evaluate(implementation):
        inputs = [
            value.detach().clone().requires_grad_() for value in (decoder, encoder)
        ]
        try:
            value = map_nuclear_penalty(*inputs, implementation=implementation)
        except ValueError as exc:
            return exc, None, None
        return None, value, torch.autograd.grad(value, inputs)

    actual_error, actual, actual_gradients = evaluate(
        MAP_NUCLEAR_GUARDED_MATMUL_IMPLEMENTATION
    )
    expected_error, expected, expected_gradients = evaluate(
        MAP_NUCLEAR_EINSUM_REFERENCE_IMPLEMENTATION
    )
    assert (actual_error is None) == (expected_error is None)
    if expected_error is not None:
        assert str(actual_error) == str(expected_error)
    else:
        assert torch.equal(actual, expected)
        for actual_gradient, expected_gradient in zip(
            actual_gradients,
            expected_gradients,
            strict=True,
        ):
            assert torch.equal(actual_gradient, expected_gradient)


@pytest.mark.parametrize(
    "implementation",
    (
        MAP_NUCLEAR_GUARDED_MATMUL_IMPLEMENTATION,
        MAP_NUCLEAR_EINSUM_REFERENCE_IMPLEMENTATION,
    ),
)
def test_map_nuclear_implementation_retains_rank_deficient_decoder_refusal(
    device,
    implementation,
):
    decoder = random_stack(device, seed=5260)
    decoder[:, :, 1:] = 0.0
    encoder = random_stack(device, seed=5261)
    with pytest.raises(
        ValueError,
        match="requires full-row-rank decoder blocks",
    ):
        map_nuclear_penalty(
            decoder,
            encoder,
            implementation=implementation,
        )


@pytest.mark.parametrize("site_rank", (1, 2))
@pytest.mark.parametrize("penalty", ("map", "decoder"))
def test_factorized_nuclear_penalties_match_materialized_value_and_gradients(
    device, site_rank, penalty
):
    generator = torch.Generator(device="cpu").manual_seed(5101 + site_rank)
    d_site = torch.randn(S, site_rank, generator=generator).to(device).requires_grad_()
    d_core = torch.randn(
        site_rank,
        G,
        B_DIM,
        D_MODEL,
        generator=generator,
    ).to(device).requires_grad_()
    e_site = torch.randn(S, site_rank, generator=generator).to(device).requires_grad_()
    e_core = torch.randn(
        site_rank,
        G,
        B_DIM,
        D_MODEL,
        generator=generator,
    ).to(device).requires_grad_()
    oracle_inputs = [
        value.detach().clone().requires_grad_()
        for value in (d_site, d_core, e_site, e_core)
    ]

    if penalty == "map":
        actual = factorized_map_nuclear_penalty(
            d_site,
            d_core,
            e_site,
            e_core,
            eps=1e-8,
        )
        od_site, od_core, oe_site, oe_core = oracle_inputs
        expected = map_nuclear_penalty(
            torch.einsum("sr,rgbd->sgbd", od_site, od_core),
            torch.einsum("sr,rgbd->sgbd", oe_site, oe_core),
            eps=1e-8,
        )
        actual_inputs = (d_site, d_core, e_site, e_core)
        expected_inputs = tuple(oracle_inputs)
    else:
        actual = factorized_decoder_nuclear_penalty(d_site, d_core, eps=1e-8)
        od_site, od_core = oracle_inputs[:2]
        expected = decoder_nuclear_penalty(
            torch.einsum("sr,rgbd->sgbd", od_site, od_core),
            eps=1e-8,
        )
        actual_inputs = (d_site, d_core)
        expected_inputs = (od_site, od_core)

    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
    actual_gradients = torch.autograd.grad(actual, actual_inputs)
    expected_gradients = torch.autograd.grad(expected, expected_inputs)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        difference = (actual_gradient - expected_gradient).float().norm()
        scale = expected_gradient.float().norm().clamp_min(1e-30)
        assert float(difference / scale) <= 1e-5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("site_rank", (1, 2))
@pytest.mark.parametrize("penalty", ("map", "decoder"))
def test_factorized_bf16_nuclear_penalties_have_bounded_materialization_drift(
    site_rank, penalty
):
    generator = torch.Generator(device="cuda").manual_seed(5151 + site_rank)
    tensors = [
        torch.randn(*shape, generator=generator, device="cuda", dtype=torch.bfloat16)
        .mul_(0.2)
        .requires_grad_()
        for shape in (
            (S, site_rank),
            (site_rank, G, B_DIM, D_MODEL),
            (S, site_rank),
            (site_rank, G, B_DIM, D_MODEL),
        )
    ]
    oracle_inputs = [value.detach().clone().requires_grad_() for value in tensors]
    d_site, d_core, e_site, e_core = tensors
    if penalty == "map":
        actual = factorized_map_nuclear_penalty(
            d_site, d_core, e_site, e_core, eps=1e-8
        )
        od_site, od_core, oe_site, oe_core = oracle_inputs
        expected = map_nuclear_penalty(
            torch.einsum("sr,rgbd->sgbd", od_site, od_core),
            torch.einsum("sr,rgbd->sgbd", oe_site, oe_core),
            eps=1e-8,
        )
        actual_inputs = tuple(tensors)
        expected_inputs = tuple(oracle_inputs)
    else:
        actual = factorized_decoder_nuclear_penalty(d_site, d_core, eps=1e-8)
        od_site, od_core = oracle_inputs[:2]
        expected = decoder_nuclear_penalty(
            torch.einsum("sr,rgbd->sgbd", od_site, od_core),
            eps=1e-8,
        )
        actual_inputs = (d_site, d_core)
        expected_inputs = (od_site, od_core)

    value_scale = expected.detach().float().abs().clamp_min(1e-30)
    assert float((actual - expected).detach().float().abs() / value_scale) <= 0.003
    actual_gradients = torch.autograd.grad(actual, actual_inputs)
    expected_gradients = torch.autograd.grad(expected, expected_inputs)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        actual32, expected32 = actual_gradient.float(), expected_gradient.float()
        difference = (actual32 - expected32).norm()
        scale = expected32.norm().clamp_min(1e-30)
        assert float(difference / scale) <= 0.01
        cosine = torch.nn.functional.cosine_similarity(
            actual32.flatten(), expected32.flatten(), dim=0
        )
        assert float(cosine) >= 0.9999


@pytest.mark.parametrize("site_rank", (1, 2))
def test_factorized_map_exact_zero_has_finite_gradients_and_accepts_rank_deficiency(
    device, site_rank
):
    generator = torch.Generator(device="cpu").manual_seed(5190 + site_rank)
    values = [
        torch.randn(*shape, generator=generator).to(device).requires_grad_()
        for shape in (
            (S, site_rank),
            (site_rank, G, B_DIM, D_MODEL),
            (S, site_rank),
            (site_rank, G, B_DIM, D_MODEL),
        )
    ]
    loss = factorized_map_nuclear_penalty(*values, eps=0.0)
    gradients = torch.autograd.grad(loss, values)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)

    d_site, d_core, e_site, e_core = [value.detach().clone() for value in values]
    e_core[:, :, 1:] = 0.0
    actual = factorized_map_nuclear_penalty(
        d_site,
        d_core,
        e_site,
        e_core,
        eps=0.0,
    )
    expected = map_nuclear_penalty(
        torch.einsum("sr,rgbd->sgbd", d_site, d_core),
        torch.einsum("sr,rgbd->sgbd", e_site, e_core),
        eps=0.0,
    )
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)


def test_frobenius_projection(device):
    D = random_stack(device, scale=3.0)
    hits = project_block_frobenius_(D)
    norms = D.float().pow(2).sum(dim=(0, 2, 3)).sqrt()
    assert hits == G
    assert norms.max() <= 1.0 + 1e-5


def test_private_projection_counts_remain_device_resident_and_exact(device):
    for private, public in (
        (gram_module._retract_count_tensor_, retract_),
        (gram_module._qr_retract_count_tensor_, gram_module.qr_retract_),
        (
            gram_module._cholesky_qr_retract_count_tensor_,
            cholesky_qr_retract_,
        ),
        (
            gram_module._project_block_frobenius_count_tensor_,
            project_block_frobenius_,
        ),
        (
            gram_module._normalize_block_frobenius_count_tensor_,
            gram_module.normalize_block_frobenius_,
        ),
        (
            gram_module._project_latent_rows_count_tensor_,
            gram_module.project_latent_rows_,
        ),
    ):
        private_input = random_stack(device, seed=119, scale=3.0)
        public_input = private_input.clone()
        count = private(private_input)
        public_count = public(public_input)
        assert count.shape == ()
        assert count.dtype == torch.int64
        assert count.device == private_input.device
        assert int(count.cpu()) == public_count
        assert torch.equal(private_input, public_input)


def test_site_singular_values_casts_before_gram(device):
    D = random_stack(device, seed=22)
    expected = site_singular_values(D)
    actual = site_singular_values(D.to(torch.bfloat16))
    # The only difference is the input parameter cast, not bf16 accumulation.
    reference = site_singular_values(D.to(torch.bfloat16).float())
    assert torch.equal(actual, reference)
    assert torch.allclose(actual, expected, atol=2e-2)


def test_o_b_invariance(device):
    """A per-block O(b) rotation leaves constraint and spectra unchanged."""
    D = random_stack(device, seed=6)
    retract_(D)
    R = random_orthogonal(B_DIM, device, seed=7)
    D_rot = torch.einsum("bc,sgcd->sgbd", R, D)
    assert gram_residual(D_rot).max().item() < 1e-4
    sv, sv_rot = site_singular_values(D), site_singular_values(D_rot)
    assert (sv - sv_rot).abs().max().item() < 1e-4


def test_block_gram_matches_naive(device):
    D = random_stack(device, seed=8)
    M = block_gram(D)
    g = 3
    naive = torch.stack([D[s, g] @ D[s, g].T for s in range(S)]).sum(dim=0)
    assert torch.allclose(M[g], naive, atol=1e-5)


def test_chunked_gram_and_retraction_match(monkeypatch, device):
    D = random_stack(device, seed=9)
    expected = torch.einsum("sgbd,sgcd->gbc", D, D)
    monkeypatch.setattr(gram_module, "_GRAM_BLOCK_CHUNK", 3)
    monkeypatch.setattr(gram_module, "_RETRACT_UNCHUNKED_MAX", 0)
    assert torch.allclose(block_gram(D), expected, atol=1e-5)
    retract_(D)
    assert gram_residual(D).max().item() < 1e-5


@pytest.mark.parametrize("sites", (1, 4, 6))
def test_sitewise_polar_application_is_bitwise_einsum_exact(
    monkeypatch,
    device,
    sites,
):
    generator = torch.Generator(device="cpu").manual_seed(5280 + sites)
    source = torch.randn(sites, 17, 4, 33, generator=generator).to(device)
    expected = source.clone()
    actual = source.clone()
    gram = torch.einsum("sgbd,sgcd->gbc", expected, expected)
    evals, evecs = torch.linalg.eigh(gram)
    expected_hits = int((evals < 1e-6).sum())
    inv_sqrt = (
        evecs
        @ torch.diag_embed(evals.clamp_min(1e-6).rsqrt())
        @ evecs.transpose(-1, -2)
    )
    expected.copy_(torch.einsum("gbc,sgcd->sgbd", inv_sqrt, expected))
    monkeypatch.setattr(gram_module, "_GRAM_BLOCK_CHUNK", 7)
    monkeypatch.setattr(gram_module, "_RETRACT_UNCHUNKED_MAX", 0)
    actual_hits = retract_(actual)
    assert actual_hits == expected_hits
    assert torch.equal(actual, expected)


def test_chunked_site_spectrum_matches_and_has_grad(monkeypatch, device):
    D = random_stack(device, seed=10)
    expected = site_singular_values(D)
    monkeypatch.setattr(gram_module, "_SPECTRUM_BLOCK_CHUNK", 3)
    monkeypatch.setattr(gram_module, "_SPECTRUM_CUDA_BLOCK_CHUNK", 3)
    monkeypatch.setattr(gram_module, "_SPECTRUM_UNCHUNKED_MAX", 0)
    D.requires_grad_(True)
    actual = site_singular_values(D)
    assert torch.allclose(actual, expected, atol=1e-5)
    actual.sum().backward()
    assert D.grad is not None and torch.isfinite(D.grad).all()
