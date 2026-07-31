"""Focused tests for data allocation and resource estimates."""

from __future__ import annotations

import pytest

from block_crosscoder_experiment.cli.data import (
    estimate_capture_pipeline_residency_bytes,
    estimate_store_bytes,
    estimate_writer_residency_bytes,
    parse_capture_split_sizes,
    parse_split_sizes,
    transformer_lens_model_name,
    whole_sequence_split_plan,
)


def test_transformer_lens_loader_name_preserves_model_identity() -> None:
    assert transformer_lens_model_name("openai-community/gpt2") == "gpt2"
    assert (
        transformer_lens_model_name("google/gemma-3-1b-pt")
        == "google/gemma-3-1b-pt"
    )


def test_split_parser_requires_named_positive_roles() -> None:
    assert parse_split_sizes(
        ["normalization_fit=10", "calibration=20", "train=30"]
    ) == {"normalization_fit": 10, "calibration": 20, "train": 30}

    with pytest.raises(ValueError, match="missing required"):
        parse_split_sizes(["train=30"])
    with pytest.raises(ValueError, match="duplicate"):
        parse_split_sizes(
            ["normalization_fit=10", "calibration=20", "train=30", "train=40"]
        )
    with pytest.raises(ValueError, match="POSITIVE"):
        parse_split_sizes(["normalization_fit=10", "calibration=20", "train=0"])


def test_phase2_capture_roles_are_complete_and_canonical_order() -> None:
    parsed = parse_capture_split_sizes(
        [
            "train=50",
            "confirmation=40",
            "development=30",
            "calibration=20",
            "normalization_fit=10",
        ],
        profile="phase2",
    )
    assert list(parsed) == [
        "normalization_fit",
        "calibration",
        "development",
        "confirmation",
        "train",
    ]

    with pytest.raises(ValueError, match="missing"):
        parse_capture_split_sizes(
            ["normalization_fit=10", "calibration=20", "train=50"],
            profile="phase2",
        )


def test_whole_sequence_allocation_never_shares_sequences_between_roles() -> None:
    plan = whole_sequence_split_plan(
        {"normalization_fit": 5, "calibration": 9, "train": 17},
        tokens_per_sequence=4,
    )

    assert plan["normalization_fit"]["sequence_start"] == 0
    assert plan["normalization_fit"]["sequence_stop_exclusive"] == 2
    assert plan["calibration"]["sequence_start"] == 2
    assert plan["calibration"]["sequence_stop_exclusive"] == 5
    assert plan["train"]["sequence_start"] == 5
    assert plan["train"]["actual_tokens"] == 20


def test_store_estimate_prices_padded_bfloat16_acts_and_row_ids() -> None:
    small = estimate_store_bytes(
        {"train": 100},
        (4, 6),
        n_views=1,
        tokens_per_shard=50,
    )
    two_views = estimate_store_bytes(
        {"train": 100},
        (4, 6),
        n_views=2,
        tokens_per_shard=50,
    )
    assert small > 100 * (2 * 2 * 6 + 3 * 8)
    assert two_views == 2 * small


def test_writer_residency_is_one_pending_plus_one_staging_shard() -> None:
    estimate = estimate_writer_residency_bytes(
        (4, 6), tokens_per_shard=100, row_id_width=3
    )
    assert estimate["bytes_per_token"] == 48
    assert estimate["writer_residency_bytes"] == 2 * estimate["shard_payload_bytes"]


def test_capture_pipeline_estimate_accounts_for_overlap_buffers() -> None:
    writer = estimate_writer_residency_bytes((4, 6), tokens_per_shard=100)
    synchronous = estimate_capture_pipeline_residency_bytes(
        writer,
        (4, 6),
        batch_rows=2,
        context=8,
        drop_positions=1,
        cuda_overlap=False,
    )
    overlapped = estimate_capture_pipeline_residency_bytes(
        writer,
        (4, 6),
        batch_rows=2,
        context=8,
        drop_positions=1,
        cuda_overlap=True,
    )
    assert overlapped["peak_host_pipeline_bytes"] > synchronous[
        "peak_host_pipeline_bytes"
    ]
    assert overlapped["peak_cuda_capture_lookahead_bytes"] > 0
