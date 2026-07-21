from __future__ import annotations

import pytest
import torch

import block_crosscoder_experiment.evaluation as evaluation_module
from block_crosscoder_experiment.evaluation import centered_fvu, evaluate_shared_code
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
