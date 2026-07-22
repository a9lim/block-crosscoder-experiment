from __future__ import annotations

import weakref

import pytest
import torch

import block_crosscoder_experiment.evaluation as evaluation_module
from block_crosscoder_experiment.evaluation import (
    centered_fvu,
    evaluate_selector_and_shared_code_modes,
    evaluate_shared_code,
    evaluate_shared_code_modes,
)
from block_crosscoder_experiment.model import BSCConfig, BlockCrosscoder


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
    model = BlockCrosscoder(
        BSCConfig(8, 2, 2, 6, 2, selection_score="decoded_energy")
    )
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
    for mode, reference in enumerate(references):
        for field, expected in zip(actual_fields, reference, strict=True):
            assert torch.equal(field[mode], expected)
    assert equations == (
        []
        if block_dim == 1
        else ["ngb,gbc->ngc", "ngb,gbc,ngc->ng"]
    )


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
                full_selected[..., 0].square()
                * gram[:, 0, 0].unsqueeze(0)
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
    full_masks = (
        torch.rand(modes, n_tokens, groups, generator=generator) > 0.4
    ).cuda()
    partial_masks = (
        torch.rand(modes, n_tokens, groups, generator=generator) > 0.6
    ).cuda()
    decoder = torch.randn(
        groups,
        block_dim,
        block_dim,
        generator=generator,
    ).double().cuda()
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
    assert all(
        payload["n_tokens"] == 13 for payload in joint.shared_code.values()
    )


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

    # Two selector-specific full outputs and two outputs for the current
    # partial view coexist; prior views and batches must be gone.
    assert peak == 4


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
        model.E[0].fill_(1.0)
        model.E[1].fill_(-2.0)
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
            transformed.E.copy_(torch.einsum("bc,sgcd->sgbd", gauge, transformed.E))
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
