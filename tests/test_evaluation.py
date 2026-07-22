from __future__ import annotations

import copy
from dataclasses import asdict
import math
import weakref

import pytest
import torch

import block_crosscoder_experiment.evaluation as evaluation_module
from block_crosscoder_experiment.evaluation import (
    centered_fvu,
    evaluate_selector_and_shared_code_modes,
    evaluate_shared_code,
    evaluate_shared_code_modes,
    load_trained_model,
)
from block_crosscoder_experiment.model import BSCConfig, BlockCrosscoder
from block_crosscoder_experiment.runtime_limits import (
    DECODED_ENERGY_EXACT_IMPLEMENTATION,
    DECODED_ENERGY_STIEFEL_CODE_NORM_IMPLEMENTATION,
    EVALUATION_SPARSE_DECODE_DENSITY_DENOMINATOR,
)


def test_centered_fvu_excludes_padding() -> None:
    x = torch.randn(32, 2, 5)
    y = x.clone()
    y[:, 1, 3:] = 1e6
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.bool)
    assert torch.equal(centered_fvu(x, y, mask), torch.zeros(2, dtype=torch.float64))


def test_centered_fvu_centers_each_coordinate_not_one_site_scalar() -> None:
    noise = torch.linspace(-1, 1, 32).view(-1, 1, 1)
    offsets = torch.tensor([[[1_000.0, -2_000.0, 4_000.0]]])
    target = offsets + noise * torch.tensor([[[1.0, 2.0, 3.0]]])
    prediction = offsets.expand_as(target)
    expected = (target - prediction).double().square().sum() / (
        target - target.mean(dim=0)
    ).double().square().sum()
    actual = centered_fvu(target, prediction)
    assert actual.shape == (1,)
    assert torch.allclose(actual[0], expected)


def test_stiefel_score_specialization_preserves_exact_shared_view_payloads() -> None:
    cfg_values = {
        "n_blocks": 24,
        "block_dim": 2,
        "n_sites": 3,
        "d_model": 7,
        "site_dims": (7, 5, 3),
        "k": 3,
        "seed": 2501,
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
    ).eval()
    fast = BlockCrosscoder(
        BSCConfig(
            **cfg_values,
            decoded_energy_implementation=(
                DECODED_ENERGY_STIEFEL_CODE_NORM_IMPLEMENTATION
            ),
        )
    ).eval()
    fast.load_state_dict(exact.state_dict())
    x = torch.randn(96, 3, 7, generator=torch.Generator().manual_seed(2502))
    exact.fit_threshold_([x], target_avg_blocks=3, method="exact")
    fast.fit_threshold_([x], target_avg_blocks=3, method="exact")

    expected = evaluate_shared_code_modes(exact, [x])
    actual = evaluate_shared_code_modes(fast, [x])
    normalized_actual = copy.deepcopy(actual)
    for mode in ("topk", "threshold"):
        assert actual[mode]["model_cfg"]["decoded_energy_implementation"] == (
            DECODED_ENERGY_STIEFEL_CODE_NORM_IMPLEMENTATION
        )
        assert expected[mode]["model_cfg"]["decoded_energy_implementation"] == (
            DECODED_ENERGY_EXACT_IMPLEMENTATION
        )
        normalized_actual[mode]["model_cfg"]["decoded_energy_implementation"] = (
            DECODED_ENERGY_EXACT_IMPLEMENTATION
        )

    def assert_nested_close(left, right) -> None:
        if isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                assert_nested_close(left[key], right[key])
        elif isinstance(left, list):
            assert len(left) == len(right)
            for left_item, right_item in zip(left, right, strict=True):
                assert_nested_close(left_item, right_item)
        elif isinstance(left, float):
            if math.isnan(left):
                assert math.isnan(right)
            else:
                assert left == pytest.approx(right, rel=1e-10, abs=1e-12)
        else:
            assert left == right

    assert_nested_close(normalized_actual, expected)


def test_load_trained_model_refuses_specialization_binding_and_residual_mutation(
    tmp_path,
) -> None:
    cfg = BSCConfig(
        n_blocks=16,
        block_dim=2,
        n_sites=2,
        d_model=6,
        site_dims=(6, 4),
        k=3,
        selection="token_topk",
        encoder_mode="tied",
        decoder_constraint="gram",
        selection_score="decoded_energy",
        decoded_energy_implementation=(DECODED_ENERGY_STIEFEL_CODE_NORM_IMPLEMENTATION),
    )
    model = BlockCrosscoder(cfg)
    model_cfg = asdict(cfg)
    train_cfg = {"retract_every": 1}
    payload = {
        "model_cfg": model_cfg,
        "train_cfg": train_cfg,
        "model": model.state_dict(),
        "run_binding": {
            "model_cfg": copy.deepcopy(model_cfg),
            "train_cfg": copy.deepcopy(train_cfg),
        },
    }
    valid_path = tmp_path / "valid.pt"
    torch.save(payload, valid_path)
    restored, metadata = load_trained_model(valid_path)
    assert restored.uses_stiefel_code_norm_decoded_energy
    assert metadata["model_cfg"]["decoded_energy_implementation"] == (
        DECODED_ENERGY_STIEFEL_CODE_NORM_IMPLEMENTATION
    )

    rebound = copy.deepcopy(payload)
    rebound["model_cfg"]["decoded_energy_implementation"] = (
        DECODED_ENERGY_EXACT_IMPLEMENTATION
    )
    rebound_path = tmp_path / "rebound.pt"
    torch.save(rebound, rebound_path)
    with pytest.raises(ValueError, match="run binding mismatch"):
        load_trained_model(rebound_path)

    missing = copy.deepcopy(payload)
    del missing["model_cfg"]["decoded_energy_implementation"]
    missing_path = tmp_path / "missing.pt"
    torch.save(missing, missing_path)
    with pytest.raises(ValueError, match="lacks decoded_energy_implementation"):
        load_trained_model(missing_path)

    missing_isolated = copy.deepcopy(payload)
    del missing_isolated["model_cfg"]["isolated_loss_decrease_implementation"]
    missing_isolated_path = tmp_path / "missing-isolated-loss.pt"
    torch.save(missing_isolated, missing_isolated_path)
    with pytest.raises(
        ValueError,
        match="lacks isolated_loss_decrease_implementation",
    ):
        load_trained_model(missing_isolated_path)

    drifted = copy.deepcopy(payload)
    drifted["model"]["D"].mul_(1.1)
    drifted_path = tmp_path / "drifted.pt"
    torch.save(drifted, drifted_path)
    with pytest.raises(RuntimeError, match="invariant failed"):
        load_trained_model(drifted_path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_sparse_evaluation_decode_dispatches_at_exact_density_cap(
    monkeypatch,
) -> None:
    cfg = BSCConfig(
        n_blocks=64,
        block_dim=2,
        n_sites=2,
        d_model=5,
        site_dims=(5, 3),
        k=2,
        encoder_mode="tied",
        decoder_constraint="free",
        decoder_bias=True,
    )
    model = BlockCrosscoder(cfg, device="cuda").eval()
    with torch.no_grad():
        model.c.uniform_(-0.25, 0.25)
    decoder = model.decoder_tensor()
    mask = torch.zeros(4, cfg.n_blocks, dtype=torch.bool, device="cuda")
    max_events = mask.numel() // EVALUATION_SPARSE_DECODE_DENSITY_DENOMINATOR
    assert max_events == 8
    mask[:, :2] = True
    selected = torch.zeros(
        4,
        cfg.n_blocks,
        cfg.block_dim,
        device="cuda",
    )
    selected[mask] = torch.randn(max_events, cfg.block_dim, device="cuda")
    dense_at_cap = model.decode(selected, _decoder=decoder)

    sparse_mm = torch.sparse.mm
    sparse_csr_tensor = torch.sparse_csr_tensor
    tensor_nonzero = torch.Tensor.nonzero
    native_decode = model.decode
    sparse_calls = 0
    dense_calls = 0
    nonzero_calls = 0
    csr_terminal_offsets = []
    previous_site_result = None

    def counted_sparse_mm(*args, **kwargs):
        nonlocal sparse_calls, previous_site_result
        assert previous_site_result is None or previous_site_result() is None
        sparse_calls += 1
        result = sparse_mm(*args, **kwargs)
        previous_site_result = weakref.ref(result)
        return result

    def counted_sparse_csr(crow, columns, values, *args, **kwargs):
        csr_terminal_offsets.append(int(crow[-1].item()))
        assert len(columns) == len(values) == csr_terminal_offsets[-1]
        return sparse_csr_tensor(crow, columns, values, *args, **kwargs)

    def counted_nonzero(tensor, *args, **kwargs):
        nonlocal nonzero_calls
        nonzero_calls += 1
        return tensor_nonzero(tensor, *args, **kwargs)

    def counted_dense(*args, **kwargs):
        nonlocal dense_calls
        dense_calls += 1
        return native_decode(*args, **kwargs)

    monkeypatch.setattr(torch.sparse, "mm", counted_sparse_mm)
    monkeypatch.setattr(torch, "sparse_csr_tensor", counted_sparse_csr)
    monkeypatch.setattr(torch.Tensor, "nonzero", counted_nonzero)
    monkeypatch.setattr(model, "decode", counted_dense)
    sparse_at_cap = evaluation_module._decode_selected_for_evaluation(
        model,
        selected,
        mask,
        decoder,
    )
    torch.testing.assert_close(sparse_at_cap, dense_at_cap, rtol=3e-7, atol=5e-7)
    assert sparse_calls == cfg.n_sites
    assert dense_calls == 0
    assert nonzero_calls == 1
    assert csr_terminal_offsets == [max_events * cfg.block_dim]
    assert not sparse_at_cap.requires_grad
    assert previous_site_result is not None and previous_site_result() is None
    assert torch.equal(
        sparse_at_cap[:, 1, 3:],
        torch.zeros_like(sparse_at_cap[:, 1, 3:]),
    )

    static_calls = 0
    nonzero_static = torch.nonzero_static

    def counted_nonzero_static(tensor, *, size, fill_value=-1):
        nonlocal static_calls
        static_calls += 1
        return nonzero_static(tensor, size=size, fill_value=fill_value)

    monkeypatch.setattr(torch, "nonzero_static", counted_nonzero_static)
    static_at_cap = evaluation_module._decode_selected_for_evaluation(
        model,
        selected,
        mask,
        decoder,
        selected_count=max_events,
    )
    torch.testing.assert_close(static_at_cap, sparse_at_cap, rtol=0, atol=0)
    assert static_calls == 1
    assert nonzero_calls == 1
    sparse_calls -= cfg.n_sites

    static_zero = evaluation_module._decode_selected_for_evaluation(
        model,
        torch.zeros_like(selected),
        torch.zeros_like(mask),
        decoder,
        selected_count=0,
    )
    assert torch.equal(
        static_zero,
        native_decode(torch.zeros_like(selected), _decoder=decoder),
    )
    assert static_calls == 1
    with pytest.raises(ValueError, match="selected_count"):
        evaluation_module._decode_selected_for_evaluation(
            model,
            selected,
            mask,
            decoder,
            selected_count=-1,
        )

    active_zero_dense = native_decode(torch.zeros_like(selected), _decoder=decoder)
    active_zero_sparse = evaluation_module._decode_selected_for_evaluation(
        model,
        torch.zeros_like(selected),
        mask,
        decoder,
    )
    torch.testing.assert_close(
        active_zero_sparse,
        active_zero_dense,
        rtol=0,
        atol=0,
    )
    assert sparse_calls == 2 * cfg.n_sites
    assert nonzero_calls == 2
    assert csr_terminal_offsets[-1] == max_events * cfg.block_dim
    assert not active_zero_sparse.requires_grad

    zero_mask = torch.zeros_like(mask)
    zero_selected = torch.zeros_like(selected)
    dense_zero = native_decode(zero_selected, _decoder=decoder)
    sparse_zero = evaluation_module._decode_selected_for_evaluation(
        model,
        zero_selected,
        zero_mask,
        decoder,
    )
    assert torch.equal(sparse_zero, dense_zero)
    assert sparse_calls == 2 * cfg.n_sites
    assert dense_calls == 0
    assert nonzero_calls == 2

    above_mask = mask.clone()
    above_mask[0, 2] = True
    above_selected = selected.clone()
    above_selected[0, 2] = torch.randn(cfg.block_dim, device="cuda")
    dense_above = native_decode(above_selected, _decoder=decoder)
    actual_above = evaluation_module._decode_selected_for_evaluation(
        model,
        above_selected,
        above_mask,
        decoder,
    )
    assert torch.equal(actual_above, dense_above)
    assert sparse_calls == 2 * cfg.n_sites
    assert dense_calls == 1
    assert nonzero_calls == 2

    bf16_selected = selected.bfloat16()
    bf16_decoder = decoder.bfloat16()
    dense_bf16 = native_decode(bf16_selected, _decoder=bf16_decoder)
    actual_bf16 = evaluation_module._decode_selected_for_evaluation(
        model,
        bf16_selected,
        mask,
        bf16_decoder,
    )
    assert torch.equal(actual_bf16, dense_bf16)
    assert sparse_calls == 2 * cfg.n_sites
    assert dense_calls == 2
    assert nonzero_calls == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_sparse_evaluation_preserves_full_site_and_loo_mode_payloads(
    monkeypatch,
) -> None:
    cfg = BSCConfig(
        n_blocks=128,
        block_dim=2,
        n_sites=2,
        d_model=8,
        k=1,
        encoder_fusion="availability_rescaled_sum",
    )
    model = BlockCrosscoder(cfg, device="cuda").eval()
    with torch.no_grad():
        assert model.E is not None
        model.E.zero_()
        encoder = model._encoder_full_tensor()
        encoder[:, 0, 0, 0] = 4.0
        encoder[:, 1, 0, 0] = 3.0
        encoder[:, 2, 0, 0] = 1.0
        model.theta.fill_(4.0)
    scale = torch.linspace(0.9, 1.1, 64, device="cuda").view(-1, 1, 1)
    x = scale.expand(-1, cfg.n_sites, cfg.d_model).clone()
    monkeypatch.setattr(
        evaluation_module,
        "EVALUATION_SPARSE_DECODE_DENSITY_DENOMINATOR",
        mask_denominator := x.shape[0] * cfg.n_blocks + 1,
    )
    assert mask_denominator > x.shape[0] * cfg.n_blocks
    dense = evaluate_shared_code_modes(
        model,
        [x],
        device="cuda",
        selection_modes=("topk", "threshold"),
    )

    sparse_mm = torch.sparse.mm
    native_decode = model.decode
    sparse_calls = 0
    dense_calls = 0

    def counted_sparse_mm(*args, **kwargs):
        nonlocal sparse_calls
        sparse_calls += 1
        return sparse_mm(*args, **kwargs)

    def counted_dense(*args, **kwargs):
        nonlocal dense_calls
        dense_calls += 1
        return native_decode(*args, **kwargs)

    monkeypatch.setattr(torch.sparse, "mm", counted_sparse_mm)
    monkeypatch.setattr(model, "decode", counted_dense)
    monkeypatch.setattr(
        evaluation_module,
        "EVALUATION_SPARSE_DECODE_DENSITY_DENOMINATOR",
        EVALUATION_SPARSE_DECODE_DENSITY_DENOMINATOR,
    )
    actual = evaluate_shared_code_modes(
        model,
        [x],
        device="cuda",
        selection_modes=("topk", "threshold"),
    )

    def assert_nested_close(left, right) -> None:
        if isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                assert_nested_close(left[key], right[key])
        elif isinstance(left, list):
            assert len(left) == len(right)
            for left_item, right_item in zip(left, right, strict=True):
                assert_nested_close(left_item, right_item)
        elif isinstance(left, float):
            assert left == pytest.approx(right, rel=2e-6, abs=1e-8)
        else:
            assert left == right

    assert_nested_close(actual, dense)
    assert sparse_calls == 2 * cfg.n_sites * (1 + 2 * cfg.n_sites)
    assert dense_calls == 0
    assert actual["topk"]["fire_count"] != actual["threshold"]["fire_count"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("batch_tokens", "n_sites", "d_model", "n_blocks", "active_blocks"),
    (
        (8192, 4, 768, 2048, 8),
        (2048, 4, 2560, 4096, 32),
    ),
)
def test_cuda_sparse_evaluation_decode_campaign_release_gate(
    batch_tokens,
    n_sites,
    d_model,
    n_blocks,
    active_blocks,
) -> None:
    cfg = BSCConfig(
        n_blocks=n_blocks,
        block_dim=4,
        n_sites=n_sites,
        d_model=d_model,
        k=active_blocks,
        encoder_mode="tied",
        decoder_constraint="unit_latent",
    )
    model = BlockCrosscoder(cfg, device="cuda").eval()
    decoder = model.decoder_tensor()
    token = torch.arange(batch_tokens, device="cuda").unsqueeze(1)
    within = torch.arange(active_blocks, device="cuda").unsqueeze(0)

    for seed in (2511, 2521):
        generator = torch.Generator(device="cuda").manual_seed(seed)
        ids = (token * 131 + within + seed) % n_blocks
        mask = torch.zeros(
            batch_tokens,
            n_blocks,
            dtype=torch.bool,
            device="cuda",
        )
        mask.scatter_(1, ids, True)
        selected = torch.zeros(
            batch_tokens,
            n_blocks,
            cfg.block_dim,
            device="cuda",
        )
        selected.scatter_(
            1,
            ids.unsqueeze(-1).expand(-1, -1, cfg.block_dim),
            torch.randn(
                batch_tokens,
                active_blocks,
                cfg.block_dim,
                device="cuda",
                generator=generator,
            ),
        )
        target = torch.randn(
            batch_tokens,
            n_sites,
            d_model,
            device="cuda",
            generator=generator,
        )
        dense = model.decode(selected, _decoder=decoder)
        sparse = evaluation_module._decode_selected_for_evaluation(
            model,
            selected,
            mask,
            decoder,
        )
        difference = sparse - dense
        dense_sse = (target - dense).double().square().sum(dim=(0, 2))
        sparse_sse = (target - sparse).double().square().sum(dim=(0, 2))
        assert difference.abs().max().item() <= 1e-6
        assert (difference.norm() / dense.norm().clamp_min(1e-30)).item() <= 3e-7
        assert (
            (sparse_sse - dense_sse).abs() / dense_sse.clamp_min(1e-30)
        ).max().item() <= 1e-9
        repeat_max_abs = 0.0
        for _ in range(8):
            repeated = evaluation_module._decode_selected_for_evaluation(
                model,
                selected,
                mask,
                decoder,
            )
            repeat_max_abs = max(
                repeat_max_abs,
                (repeated - sparse).abs().max().item(),
            )
            del repeated
        assert repeat_max_abs <= 1e-6


def test_shared_code_endpoints_work_for_tied_and_untied_encoders() -> None:
    for cfg in (
        BSCConfig(8, 2, 2, 6, 2, encoder_mode="tied"),
        BSCConfig(
            10,
            1,
            2,
            6,
            2,
            encoder_mode="untied",
            code_activation="relu",
            selection_score="decoder_weighted",
            decoder_constraint="free",
        ),
    ):
        model = BlockCrosscoder(cfg)
        x = torch.randn(40, 2, 6)
        model.fit_threshold_([x], target_avg_blocks=2)
        result = evaluate_shared_code(model, [x[:20], x[20:]])
        assert result["n_tokens"] == 40
        assert len(result["site_only_fvu"]) == 2
        assert len(result["used_contribution_eigenvalues"]) == 2


def test_shared_code_builds_frozen_score_geometry_once(monkeypatch) -> None:
    model = BlockCrosscoder(BSCConfig(8, 2, 2, 6, 2, selection_score="decoded_energy"))
    original = model._frozen_score_geometry
    calls = 0

    def counted(decoder):
        nonlocal calls
        calls += 1
        return original(decoder)

    monkeypatch.setattr(model, "_frozen_score_geometry", counted)
    x = torch.randn(40, 2, 6)
    result = evaluate_shared_code(
        model,
        [x[:20], x[20:]],
        selection_mode="topk",
    )
    assert result["n_tokens"] == 40
    assert calls == 1


@pytest.mark.parametrize("block_dim", (1, 3))
def test_batched_mode_quadratics_exactly_match_per_mode_reference(
    block_dim,
    monkeypatch,
) -> None:
    generator = torch.Generator().manual_seed(805)
    n_tokens, groups, modes = 19, 11, 2
    full = torch.randn(n_tokens, groups, block_dim, generator=generator)
    partial = torch.randn(n_tokens, groups, block_dim, generator=generator)
    decoder = torch.randn(groups, block_dim, block_dim, generator=generator).double()
    gram = decoder @ decoder.transpose(-1, -2)
    full_masks = torch.rand(modes, n_tokens, groups, generator=generator) > 0.45
    partial_masks = torch.rand(modes, n_tokens, groups, generator=generator) > 0.55

    references = []
    for mode in range(modes):
        intersection = full_masks[mode] & partial_masks[mode]
        intersection_f = intersection.unsqueeze(-1).double()
        full_selected = full.double() * intersection_f
        partial_selected = partial.double() * intersection_f
        if block_dim == 1:
            full_scalar = full_selected[..., 0]
            partial_scalar = partial_selected[..., 0]
            weight = gram[:, 0, 0].unsqueeze(0)
            full_q = full_scalar.square() * weight
            partial_q = partial_scalar.square() * weight
            cross = full_scalar * partial_scalar * weight
        else:
            full_mapped = torch.einsum(
                "ngb,gbc->ngc",
                full_selected,
                gram,
            )
            full_q = (full_mapped * full_selected).sum(dim=-1)
            partial_q = torch.einsum(
                "ngb,gbc,ngc->ng",
                partial_selected,
                gram,
                partial_selected,
            )
            cross = (full_mapped * partial_selected).sum(dim=-1)
        references.append(
            (
                intersection.sum(dim=0).double(),
                2.0 * cross.sum(dim=0),
                full_q.sum(dim=0) + partial_q.sum(dim=0),
                full_selected.sum(dim=0),
                partial_selected.sum(dim=0),
                full_q.sum(dim=0),
            )
        )

    precomputed = evaluation_module._batched_mode_concordance(
        full,
        partial,
        gram,
        intersection=full_masks & partial_masks,
    )
    equations: list[str] = []
    original_einsum = torch.einsum

    def counted_einsum(equation, *operands):
        equations.append(equation)
        return original_einsum(equation, *operands)

    monkeypatch.setattr(torch, "einsum", counted_einsum)
    actual = evaluation_module._batched_mode_concordance(
        full,
        partial,
        gram,
        full_masks,
        partial_masks,
    )
    actual_fields = tuple(actual)
    for actual_field, precomputed_field in zip(
        actual,
        precomputed,
        strict=True,
    ):
        assert torch.equal(actual_field, precomputed_field)
    for mode, reference in enumerate(references):
        for field, expected in zip(actual_fields, reference, strict=True):
            assert torch.equal(field[mode], expected)
    assert equations == ([] if block_dim == 1 else ["ngb,gbc->ngc", "ngb,gbc,ngc->ng"])


def test_batched_mode_concordance_intersection_interface_refuses_ambiguity() -> None:
    code = torch.zeros(2, 3, 1)
    gram = torch.ones(3, 1, 1, dtype=torch.float64)
    masks = torch.zeros(1, 2, 3, dtype=torch.bool)
    with pytest.raises(ValueError, match="need masks or a precomputed intersection"):
        evaluation_module._batched_mode_concordance(code, code, gram)
    with pytest.raises(ValueError, match="not both"):
        evaluation_module._batched_mode_concordance(
            code,
            code,
            gram,
            masks,
            masks,
            intersection=masks,
        )


@pytest.mark.parametrize("groups", (1, 511, 512, 513, 2048))
@pytest.mark.parametrize("tokens", (1, 63))
def test_blockwise_intersection_and_derived_union_counts_match_set_oracle(
    groups,
    tokens,
) -> None:
    generator = torch.Generator().manual_seed(groups * 100 + tokens)
    full = torch.rand(2, tokens, groups, generator=generator) > 0.47
    partial = torch.rand(2, tokens, groups, generator=generator) > 0.53
    expected_intersection = (full & partial).sum(dim=(1, 2)).double()
    expected_union = (full | partial).sum(dim=(1, 2)).double()
    actual_intersection = torch.zeros(2, dtype=torch.float64)
    for start in range(0, groups, 512):
        actual_intersection += (
            full[:, :, start : start + 512]
            & partial[:, :, start : start + 512]
        ).sum(dim=(1, 2)).double()
    actual_union = (
        full.sum(dim=(1, 2)).double()
        + partial.sum(dim=(1, 2)).double()
        - actual_intersection
    )
    assert torch.equal(actual_intersection, expected_intersection)
    assert torch.equal(actual_union, expected_union)


def test_batched_mode_geometry_suppresses_unselected_nonfinite_values() -> None:
    generator = torch.Generator().manual_seed(1337)
    full = torch.randn(4, 3, 2, generator=generator)
    partial = torch.randn(4, 3, 2, generator=generator)
    full[0, 0, 0] = torch.nan
    partial[2, 1, 1] = torch.inf
    gram = torch.eye(2, dtype=torch.float64).expand(3, 2, 2).clone()
    full_masks = torch.ones(2, 4, 3, dtype=torch.bool)
    partial_masks = torch.ones_like(full_masks)
    full_masks[:, 0, 0] = False
    partial_masks[:, 2, 1] = False

    concordance = evaluation_module._batched_mode_concordance(
        full,
        partial,
        gram,
        full_masks,
        partial_masks,
    )
    moments = evaluation_module._batched_mode_selected_moments(
        full,
        gram,
        full_masks,
    )
    delta = evaluation_module._batched_mode_selected_delta_sq(
        full,
        partial,
        full_masks,
        partial_masks,
    )

    for tensor in (*concordance, *moments, delta):
        assert bool(torch.isfinite(tensor).all())


@pytest.mark.parametrize("block_dim", (1, 3))
def test_batched_selected_moments_and_delta_preserve_exact_reductions(
    block_dim,
) -> None:
    generator = torch.Generator().manual_seed(911)
    n_tokens, groups, modes = 23, 9, 2
    full = torch.randn(n_tokens, groups, block_dim, generator=generator)
    partial = torch.randn(n_tokens, groups, block_dim, generator=generator)
    decoder = torch.randn(groups, block_dim, block_dim, generator=generator).double()
    gram = decoder @ decoder.transpose(-1, -2)
    full_masks = torch.rand(modes, n_tokens, groups, generator=generator) > 0.4
    partial_masks = torch.rand(modes, n_tokens, groups, generator=generator) > 0.6

    moments = evaluation_module._batched_mode_selected_moments(
        full,
        gram,
        full_masks,
    )
    delta = evaluation_module._batched_mode_selected_delta_sq(
        full,
        partial,
        full_masks,
        partial_masks,
    )
    for mode in range(modes):
        full_selected = (full * full_masks[mode].unsqueeze(-1)).double()
        partial_selected = (partial * partial_masks[mode].unsqueeze(-1)).double()
        if block_dim == 1:
            expected_energy = (
                full_selected[..., 0].square() * gram[:, 0, 0].unsqueeze(0)
            ).sum(dim=0)
        else:
            expected_energy = torch.einsum(
                "ngb,gbc,ngc->ng",
                full_selected,
                gram,
                full_selected,
            ).sum(dim=0)
        assert torch.equal(moments.decoded_energy[mode], expected_energy)
        assert torch.equal(moments.code_sum[mode], full_selected.sum(dim=0))
        assert torch.equal(
            moments.code_outer[mode],
            torch.einsum("ngb,ngc->gbc", full_selected, full_selected),
        )
        assert torch.equal(
            delta[mode],
            (partial_selected - full_selected).square().sum(dim=(0, 2)),
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA exactness regression",
)
@pytest.mark.parametrize("block_dim", (2, 4, 6, 8))
def test_batched_selected_mode_reductions_are_cuda_bit_exact_for_short_batches(
    block_dim,
) -> None:
    # These M=2 short-batch cases differ bitwise under both rejected batched
    # CUDA schedules: ``mngb,mngc->mgbc`` for the outer product and a
    # ``sum(dim=(1, 3))`` for the selected-code delta.
    generator = torch.Generator().manual_seed(
        100_000 * block_dim + 63_150 + int(block_dim == 2)
    )
    n_tokens, groups, modes = 63, 15, 2
    full = torch.randn(
        n_tokens,
        groups,
        block_dim,
        generator=generator,
    ).cuda()
    partial = torch.randn(
        n_tokens,
        groups,
        block_dim,
        generator=generator,
    ).cuda()
    full_masks = (torch.rand(modes, n_tokens, groups, generator=generator) > 0.4).cuda()
    partial_masks = (
        torch.rand(modes, n_tokens, groups, generator=generator) > 0.6
    ).cuda()
    decoder = (
        torch.randn(
            groups,
            block_dim,
            block_dim,
            generator=generator,
        )
        .double()
        .cuda()
    )
    gram = decoder @ decoder.transpose(-1, -2)

    moments = evaluation_module._batched_mode_selected_moments(
        full,
        gram,
        full_masks,
    )
    delta = evaluation_module._batched_mode_selected_delta_sq(
        full,
        partial,
        full_masks,
        partial_masks,
    )
    for mode in range(modes):
        zero = torch.zeros((), dtype=torch.float64, device="cuda")
        full_selected = torch.where(
            full_masks[mode].unsqueeze(-1),
            full.double(),
            zero,
        )
        partial_selected = torch.where(
            partial_masks[mode].unsqueeze(-1),
            partial.double(),
            zero,
        )
        assert torch.equal(
            moments.code_outer[mode],
            torch.einsum("ngb,ngc->gbc", full_selected, full_selected),
        )
        assert torch.equal(
            delta[mode],
            (partial_selected - full_selected).square().sum(dim=(0, 2)),
        )


@pytest.mark.parametrize(
    "config",
    (
        BSCConfig(
            n_blocks=7,
            block_dim=2,
            n_sites=3,
            d_model=5,
            k=2,
            encoder_fusion="availability_rescaled_sum",
            selection_score="isolated_loss_decrease",
            decoder_constraint="free",
            decoder_bias=False,
        ),
        BSCConfig(
            n_blocks=6,
            block_dim=1,
            n_sites=3,
            d_model=4,
            k=2,
            encoder_fusion="source",
            source_site=1,
        ),
        BSCConfig(
            n_blocks=6,
            block_dim=2,
            n_sites=1,
            d_model=5,
            k=2,
            encoder_mode="untied",
            decoder_constraint="free",
        ),
        BSCConfig(
            n_blocks=6,
            block_dim=2,
            n_sites=3,
            d_model=6,
            k=2,
            site_dims=(6, 4, 2),
            encoder_mode="untied",
            decoder_constraint="free",
        ),
    ),
)
def test_shared_code_modes_exactly_matches_independent_mode_evaluations(
    config,
) -> None:
    from block_crosscoder_experiment.cli.run_cell import _evaluate_native_selector

    model = BlockCrosscoder(config)
    x = torch.randn(
        29,
        config.n_sites,
        config.d_model,
        generator=torch.Generator().manual_seed(991),
    )
    model.fit_threshold_([x], target_avg_blocks=2)
    # Make the selectors observably different without changing the threshold
    # endpoint contract: top-k retains k events while the deployed threshold
    # intentionally retains none.
    model.theta.fill_(1e30)
    batches = [x[:11], x[11:]]
    topk = evaluate_shared_code(model, batches, selection_mode="topk")
    threshold = evaluate_shared_code(model, batches, selection_mode="threshold")
    native_topk = _evaluate_native_selector(
        model,
        batches,
        device=torch.device("cpu"),
        selection_mode="topk",
    )
    native_threshold = _evaluate_native_selector(
        model,
        batches,
        device=torch.device("cpu"),
        selection_mode="threshold",
    )
    fused = evaluate_shared_code_modes(
        model,
        batches,
        selection_modes=("topk", "threshold"),
    )
    joint = evaluate_selector_and_shared_code_modes(
        model,
        batches,
        selection_modes=("topk", "threshold"),
    )

    assert fused == {"topk": topk, "threshold": threshold}
    assert joint.shared_code == fused
    assert joint.selector == {
        "topk": native_topk,
        "threshold": native_threshold,
    }
    assert (
        fused["topk"]["functional_dependence"]["pre_selection"]
        == fused["threshold"]["functional_dependence"]["pre_selection"]
    )
    assert fused["topk"]["fire_count"] != fused["threshold"]["fire_count"]


def test_shared_code_modes_shares_view_encoding_but_not_mode_decoding(
    monkeypatch,
) -> None:
    config = BSCConfig(
        n_blocks=8,
        block_dim=2,
        n_sites=3,
        d_model=6,
        k=2,
        encoder_fusion="availability_rescaled_sum",
    )
    model = BlockCrosscoder(config)
    x = torch.randn(17, config.n_sites, config.d_model)
    model.fit_threshold_([x], target_avg_blocks=2)
    encoder_calls = 0
    score_calls = 0
    decode_calls = 0
    original_encoder = model._frozen_encoder_sites
    original_scores = model.scores
    original_decode = model.decode

    def counted_encoder(*args, **kwargs):
        nonlocal encoder_calls
        encoder_calls += 1
        return original_encoder(*args, **kwargs)

    def counted_scores(*args, **kwargs):
        nonlocal score_calls
        score_calls += 1
        return original_scores(*args, **kwargs)

    def counted_decode(*args, **kwargs):
        nonlocal decode_calls
        decode_calls += 1
        return original_decode(*args, **kwargs)

    monkeypatch.setattr(model, "_frozen_encoder_sites", counted_encoder)
    monkeypatch.setattr(model, "scores", counted_scores)
    monkeypatch.setattr(model, "decode", counted_decode)
    result = evaluate_shared_code_modes(model, [x])

    assert result.keys() == {"topk", "threshold"}
    # One full view plus S site-only and S leave-one-out views. Both selectors
    # consume each shared score tensor, while reconstruction stays per-mode.
    assert encoder_calls == 1
    assert score_calls == 1 + 2 * config.n_sites
    assert decode_calls == 2 * (1 + 2 * config.n_sites)


def test_joint_mode_endpoints_consume_one_stream_and_reuse_full_outputs(
    monkeypatch,
) -> None:
    config = BSCConfig(
        n_blocks=8,
        block_dim=2,
        n_sites=3,
        d_model=6,
        k=2,
        encoder_fusion="availability_rescaled_sum",
        selection_score="isolated_loss_decrease",
        decoder_constraint="free",
        decoder_bias=False,
    )
    model = BlockCrosscoder(config)
    x = torch.randn(17, config.n_sites, config.d_model)
    model.fit_threshold_([x], target_avg_blocks=2)
    iterator_count = 0
    encoder_calls = 0
    score_calls = 0
    decode_calls = 0
    observed_views = []
    original_encoder = model._frozen_encoder_sites
    original_scores = model.scores
    original_decode = model.decode
    original_select = model.select_with_materialized

    class SingleUseBatches:
        def __iter__(self):
            nonlocal iterator_count
            iterator_count += 1
            if iterator_count != 1:
                raise AssertionError("evaluation stream was traversed more than once")
            yield x

    def counted_encoder(*args, **kwargs):
        nonlocal encoder_calls
        encoder_calls += 1
        return original_encoder(*args, **kwargs)

    def counted_scores(*args, **kwargs):
        nonlocal score_calls
        score_calls += 1
        return original_scores(*args, **kwargs)

    def counted_decode(*args, **kwargs):
        nonlocal decode_calls
        decode_calls += 1
        return original_decode(*args, **kwargs)

    def counted_select(*args, **kwargs):
        observed_views.append(kwargs.get("observed"))
        return original_select(*args, **kwargs)

    monkeypatch.setattr(model, "_frozen_encoder_sites", counted_encoder)
    monkeypatch.setattr(model, "scores", counted_scores)
    monkeypatch.setattr(model, "decode", counted_decode)
    monkeypatch.setattr(model, "select_with_materialized", counted_select)
    result = evaluate_selector_and_shared_code_modes(model, SingleUseBatches())

    assert result.selector.keys() == {"topk", "threshold"}
    assert result.shared_code.keys() == {"topk", "threshold"}
    assert iterator_count == 1
    assert encoder_calls == 1
    assert score_calls == 1 + 2 * config.n_sites
    assert decode_calls == 2 * (1 + 2 * config.n_sites)
    assert observed_views[0] is not None
    assert bool(observed_views[0].all())


def test_joint_mode_endpoints_apply_max_tokens_before_every_reduction() -> None:
    from block_crosscoder_experiment.cli.run_cell import _evaluate_native_selector

    config = BSCConfig(
        n_blocks=8,
        block_dim=2,
        n_sites=3,
        d_model=6,
        k=2,
        encoder_fusion="availability_rescaled_sum",
        selection="batch_topk",
    )
    model = BlockCrosscoder(config)
    x = torch.randn(29, config.n_sites, config.d_model)
    model.fit_threshold_([x], target_avg_blocks=2)
    batches = [x[:10], x[10:20], x[20:]]
    truncated = [x[:10], x[10:13]]

    joint = evaluate_selector_and_shared_code_modes(
        model,
        batches,
        max_tokens=13,
    )
    assert joint.shared_code == evaluate_shared_code_modes(model, truncated)
    assert joint.selector == {
        mode: _evaluate_native_selector(
            model,
            truncated,
            device=torch.device("cpu"),
            selection_mode=mode,
        )
        for mode in ("topk", "threshold")
    }
    assert all(payload["n_tokens"] == 13 for payload in joint.selector.values())
    assert all(payload["n_tokens"] == 13 for payload in joint.shared_code.values())


@pytest.mark.parametrize("include_selector_payloads", (False, True))
def test_shared_code_modes_releases_partial_outputs_between_views(
    monkeypatch,
    include_selector_payloads,
) -> None:
    config = BSCConfig(
        n_blocks=8,
        block_dim=2,
        n_sites=3,
        d_model=6,
        k=2,
        encoder_fusion="availability_rescaled_sum",
    )
    model = BlockCrosscoder(config)
    x = torch.randn(34, config.n_sites, config.d_model)
    model.fit_threshold_([x], target_avg_blocks=2)
    live: list[weakref.ReferenceType[torch.Tensor]] = []
    peak = 0
    original_decode = model.decode

    def tracked_decode(*args, **kwargs):
        nonlocal live, peak
        prediction = original_decode(*args, **kwargs)
        live = [reference for reference in live if reference() is not None]
        live.append(weakref.ref(prediction))
        peak = max(peak, len(live))
        return prediction

    monkeypatch.setattr(model, "decode", tracked_decode)
    evaluator = (
        evaluate_selector_and_shared_code_modes
        if include_selector_payloads
        else evaluate_shared_code_modes
    )
    evaluator(model, [x[:17], x[17:]])

    # Only the two selector predictions for the current view may coexist.
    # Full-view predictions are reduced and released before partial views.
    assert peak == 2


def test_shared_code_modes_releases_selected_codes_after_each_view(
    monkeypatch,
) -> None:
    config = BSCConfig(
        n_blocks=8,
        block_dim=2,
        n_sites=3,
        d_model=6,
        k=2,
        encoder_fusion="availability_rescaled_sum",
    )
    model = BlockCrosscoder(config)
    x = torch.randn(17, config.n_sites, config.d_model)
    model.fit_threshold_([x], target_avg_blocks=2)
    references: list[weakref.ReferenceType[torch.Tensor]] = []
    call_count = 0
    peak = 0
    original = evaluation_module._decode_selected_for_evaluation

    def tracked_decode(model, selected, mask, decoder, **kwargs):
        nonlocal call_count, peak
        references[:] = [ref for ref in references if ref() is not None]
        # Each view has two selector decodes. No selected code from the prior
        # view may survive into the first decode of the next one.
        if call_count % 2 == 0:
            assert not references
        prediction = original(model, selected, mask, decoder, **kwargs)
        references.append(weakref.ref(selected))
        peak = max(peak, len(references))
        call_count += 1
        return prediction

    monkeypatch.setattr(
        evaluation_module,
        "_decode_selected_for_evaluation",
        tracked_decode,
    )
    evaluate_shared_code_modes(model, [x])

    assert call_count == 2 * (1 + 2 * config.n_sites)
    assert peak == 2
    assert not any(reference() is not None for reference in references)


@pytest.mark.parametrize(
    ("config", "expected_bmm_calls"),
    (
        (BSCConfig(8, 2, 3, 6, 2, encoder_fusion="sum"), 1),
        (BSCConfig(8, 2, 3, 6, 2, encoder_fusion="source"), 0),
        (BSCConfig(8, 2, 1, 6, 2, encoder_fusion="sum"), 0),
    ),
)
def test_shared_code_reuses_one_encoder_contraction_per_batch(
    monkeypatch, config, expected_bmm_calls
) -> None:
    model = BlockCrosscoder(config)
    x = torch.randn(17, config.n_sites, config.d_model)
    original = torch.bmm
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "bmm", counted)
    result = evaluate_shared_code(model, [x], selection_mode="topk")
    assert result["n_tokens"] == len(x)
    assert calls == expected_bmm_calls


def _assert_nested_close(actual, expected) -> None:
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_close(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_nested_close(actual_item, expected_item)
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=2e-5, abs=2e-8)
    else:
        assert actual == expected


@pytest.mark.parametrize("selection_mode", ("topk", "threshold"))
def test_shared_code_cached_partial_views_track_direct_reencoding(
    monkeypatch, selection_mode
) -> None:
    model = BlockCrosscoder(
        BSCConfig(
            n_blocks=7,
            block_dim=2,
            n_sites=3,
            d_model=5,
            k=2,
            encoder_fusion="availability_rescaled_sum",
            selection_score="isolated_loss_decrease",
            decoder_constraint="free",
            decoder_bias=False,
        )
    )
    x = torch.randn(23, 3, 5, generator=torch.Generator().manual_seed(844))
    if selection_mode == "threshold":
        model.fit_threshold_([x], target_avg_blocks=2)
    original = model.select_with_materialized

    def uncached(*args, **kwargs):
        kwargs.pop("_encoder_sites", None)
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "select_with_materialized", uncached)
    reference = evaluate_shared_code(model, [x], selection_mode=selection_mode)
    monkeypatch.setattr(model, "select_with_materialized", original)
    cached = evaluate_shared_code(model, [x], selection_mode=selection_mode)
    _assert_nested_close(cached, reference)


def test_shared_code_block_chunking_preserves_complete_payload(monkeypatch) -> None:
    torch.manual_seed(123)
    model = BlockCrosscoder(
        BSCConfig(9, 2, 3, 5, 3, encoder_mode="untied", decoder_constraint="free")
    )
    x = torch.randn(37, 3, 5)
    monkeypatch.setattr(evaluation_module, "EVALUATION_CONCORDANCE_BLOCK_CHUNK", 100)
    reference = evaluate_shared_code(model, [x], selection_mode="topk")
    monkeypatch.setattr(evaluation_module, "EVALUATION_CONCORDANCE_BLOCK_CHUNK", 2)
    chunked = evaluate_shared_code(model, [x], selection_mode="topk")

    def assert_payload_close(left, right) -> None:
        if isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                assert_payload_close(left[key], right[key])
        elif isinstance(left, list):
            assert len(left) == len(right)
            for left_item, right_item in zip(left, right, strict=True):
                assert_payload_close(left_item, right_item)
        elif isinstance(left, float):
            assert left == pytest.approx(right, rel=1e-12, abs=1e-12)
        else:
            assert left == right

    assert_payload_close(chunked, reference)


def test_shared_code_native_mode_is_explicit_and_needs_no_deployed_threshold() -> None:
    model = BlockCrosscoder(BSCConfig(6, 1, 2, 5, 2))
    x = torch.randn(24, 2, 5)
    native = evaluate_shared_code(model, [x], selection_mode="topk")
    assert native["selection_mode"] == "topk"
    try:
        evaluate_shared_code(model, [x], selection_mode="threshold")
    except RuntimeError as exc:
        assert "threshold not calibrated" in str(exc)
    else:  # pragma: no cover - protects the native/deployed boundary
        raise AssertionError("deployed evaluation accepted an uncalibrated threshold")


def test_used_contribution_centers_constant_active_codes() -> None:
    cfg = BSCConfig(
        n_blocks=3,
        block_dim=2,
        n_sites=2,
        d_model=4,
        k=3,
        encoder_bias=True,
        selection="dense",
        code_activation="relu",
        decoder_constraint="free",
    )
    model = BlockCrosscoder(cfg)
    with torch.no_grad():
        assert model.E is not None and model.a is not None
        model.E.zero_()
        model.a.fill_(1.0)
        model.theta.fill_(0.0)
    result = evaluate_shared_code(model, [torch.randn(64, 2, 4)])
    used = torch.tensor(result["used_contribution_eigenvalues"])
    assert torch.allclose(used, torch.zeros_like(used), atol=1e-10)


def test_functional_dependence_profiles_pre_and_post_selected_code() -> None:
    cfg = BSCConfig(
        n_blocks=1,
        block_dim=1,
        n_sites=2,
        d_model=1,
        k=1,
        selection="dense",
        code_activation="relu",
        decoder_constraint="free",
        decoder_bias=False,
    )
    model = BlockCrosscoder(cfg)
    with torch.no_grad():
        assert model.E is not None
        model.E.fill_(1.0)
    scale = torch.tensor([1.0, 2.0, 3.0, 4.0]).view(-1, 1, 1)
    x = torch.cat((2.0 * scale, scale), dim=1)
    result = evaluate_shared_code(model, [x], selection_mode="topk")
    functional = result["functional_dependence"]

    assert functional["interpretation"].startswith("descriptive_only")
    for stage in ("pre_selection", "post_selection"):
        delta = torch.tensor(functional[stage]["delta_by_site_block"])
        profile = torch.tensor(functional[stage]["normalized_profile_by_site_block"])
        coherence = torch.tensor(functional[stage]["coherence_per_block"])
        assert delta.shape == (2, 1)
        assert delta[0, 0] == pytest.approx(2.0 * delta[1, 0])
        assert torch.allclose(profile[:, 0], torch.tensor([1.0, 0.5]))
        assert coherence[0] == pytest.approx(1.5)
        assert functional[stage]["defined_per_block"] == [True]


def test_shared_code_evaluation_accepts_factorized_models() -> None:
    model = BlockCrosscoder(
        BSCConfig(
            n_blocks=5,
            block_dim=2,
            n_sites=3,
            d_model=4,
            k=2,
            site_rank=1,
            decoder_constraint="free",
        )
    )
    result = evaluate_shared_code(model, [torch.randn(19, 3, 4)], selection_mode="topk")
    assert result["n_tokens"] == 19
    assert (
        len(result["functional_dependence"]["pre_selection"]["coherence_per_block"])
        == 5
    )


def test_partial_view_coordinate_concordance_is_exact_for_redundant_codes() -> None:
    model = BlockCrosscoder(
        BSCConfig(
            n_blocks=1,
            block_dim=1,
            n_sites=2,
            d_model=1,
            k=1,
            encoder_fusion="availability_rescaled_sum",
            decoder_constraint="free",
            decoder_bias=False,
        )
    )
    with torch.no_grad():
        assert model.E is not None and model.D is not None
        model.E.fill_(0.5)
        model.D.fill_(1.0)
    values = torch.linspace(0.25, 2.0, 32).view(-1, 1, 1)
    x = values.expand(-1, 2, -1).clone()
    result = evaluate_shared_code(model, [x], selection_mode="topk")
    coordinate = result["partial_view_coordinate_concordance"]
    assert result["site_only_support_iou"] == pytest.approx([1.0, 1.0])
    assert result["leave_one_site_out_support_iou"] == pytest.approx([1.0, 1.0])
    for view in ("site_only", "leave_one_site_out"):
        assert coordinate[view]["concordance"] == pytest.approx([1.0, 1.0])
        assert coordinate[view]["support_intersection_recall"] == pytest.approx(
            [1.0, 1.0]
        )
        assert coordinate[view]["decoded_energy_coverage"] == pytest.approx([1.0, 1.0])


def test_per_block_support_and_energy_diagnostics_do_not_hide_low_intersection_counts() -> (
    None
):
    model = BlockCrosscoder(
        BSCConfig(
            n_blocks=1,
            block_dim=1,
            n_sites=2,
            d_model=1,
            k=1,
            encoder_fusion="availability_rescaled_sum",
            decoder_constraint="free",
            decoder_bias=False,
        )
    )
    with torch.no_grad():
        assert model.E is not None and model.D is not None
        model.E.fill_(0.5)
        model.D.fill_(1.0)
    values = torch.linspace(0.25, 2.0, 16).view(-1, 1, 1)
    result = evaluate_shared_code(
        model,
        [values.expand(-1, 2, -1).clone()],
        selection_mode="topk",
    )
    distribution = result["partial_view_coordinate_concordance"]["site_only"][
        "per_block_distribution"
    ]
    assert distribution["minimum_intersection_events_for_concordance"] == 32
    assert distribution["concordance"]["eligible_block_pattern_count"] == 0
    assert (
        distribution["support_intersection_recall"]["eligible_block_pattern_count"] == 2
    )
    assert distribution["decoded_energy_coverage"]["eligible_block_pattern_count"] == 2


def test_partial_view_concordance_detects_coordinate_drift_at_fixed_support() -> None:
    model = BlockCrosscoder(
        BSCConfig(
            n_blocks=1,
            block_dim=1,
            n_sites=2,
            d_model=1,
            k=1,
            encoder_fusion="availability_rescaled_sum",
            decoder_constraint="free",
            decoder_bias=False,
        )
    )
    with torch.no_grad():
        assert model.E is not None and model.D is not None
        encoder = model._encoder_full_tensor()
        encoder[0].fill_(1.0)
        encoder[1].fill_(-2.0)
        model.D.fill_(1.0)
    values = torch.linspace(0.25, 2.0, 32).view(-1, 1, 1)
    x = values.expand(-1, 2, -1).clone()
    result = evaluate_shared_code(model, [x], selection_mode="topk")
    coordinate = result["partial_view_coordinate_concordance"]
    assert result["site_only_support_iou"] == pytest.approx([1.0, 1.0])
    assert coordinate["site_only"]["decoded_energy_coverage"] == pytest.approx(
        [1.0, 1.0]
    )
    assert sum(coordinate["site_only"]["concordance"]) / 2 < 0.5


def test_partial_view_concordance_penalizes_additive_coordinate_offset() -> None:
    model = BlockCrosscoder(
        BSCConfig(
            n_blocks=1,
            block_dim=1,
            n_sites=2,
            d_model=1,
            k=1,
            encoder_fusion="availability_rescaled_sum",
            decoder_constraint="free",
            decoder_bias=False,
        )
    )
    with torch.no_grad():
        assert model.E is not None and model.D is not None
        model.E.fill_(0.5)
        model.D.fill_(1.0)
    varying = torch.linspace(-1.0, 1.0, 64).view(-1, 1, 1)
    offset = 2.0
    x = torch.cat((varying + offset, varying - offset), dim=1)
    result = evaluate_shared_code(model, [x], selection_mode="topk")
    coordinate = result["partial_view_coordinate_concordance"]
    assert result["site_only_support_iou"] == pytest.approx([1.0, 1.0])
    assert coordinate["site_only"]["decoded_energy_coverage"] == pytest.approx(
        [1.0, 1.0]
    )
    # A centered correlation would be one here.  Concordance also prices the
    # non-null decoded mean offset and must therefore reject equality.
    assert max(coordinate["site_only"]["concordance"]) < 0.2


def test_partial_view_coordinate_concordance_is_block_gauge_invariant() -> None:
    cfg = BSCConfig(
        n_blocks=1,
        block_dim=2,
        n_sites=2,
        d_model=3,
        k=1,
        encoder_fusion="sum",
        decoder_constraint="free",
        decoder_bias=False,
    )
    original = BlockCrosscoder(cfg)
    x = torch.randn(64, 2, 3, generator=torch.Generator().manual_seed(908))
    first = evaluate_shared_code(original, [x], selection_mode="topk")
    first_coordinate = first["partial_view_coordinate_concordance"]
    for gauge in (
        torch.tensor([[2.0, 0.25], [0.0, 0.5]]),
        torch.tensor([[-1.0, 0.0], [0.0, 1.0]]),
    ):
        transformed = BlockCrosscoder(cfg)
        transformed.load_state_dict(original.state_dict())
        inverse_transpose = torch.linalg.inv(gauge).T
        with torch.no_grad():
            assert transformed.E is not None and transformed.D is not None
            encoder = transformed._encoder_full_tensor()
            encoder.copy_(torch.einsum("bc,sgcd->sgbd", gauge, encoder))
            transformed.D.copy_(
                torch.einsum("bc,sgcd->sgbd", inverse_transpose, transformed.D)
            )
        second = evaluate_shared_code(transformed, [x], selection_mode="topk")
        second_coordinate = second["partial_view_coordinate_concordance"]
        for view in ("site_only", "leave_one_site_out"):
            for metric in (
                "concordance",
                "support_intersection_recall",
                "decoded_energy_coverage",
            ):
                assert second_coordinate[view][metric] == pytest.approx(
                    first_coordinate[view][metric], abs=1e-7
                )
