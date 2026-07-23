"""Determinism and sensitivity checks for durable tensor digests."""

from __future__ import annotations

from collections import OrderedDict

import pytest
import torch

import block_crosscoder_experiment.serialization as serialization
from block_crosscoder_experiment.serialization import (
    model_state_digest,
    typed_payload_digest,
)


def test_typed_payload_digest_separates_previously_colliding_frames() -> None:
    assert typed_payload_digest([1, 2]) != typed_payload_digest([12])
    assert typed_payload_digest([1, 2]) != typed_payload_digest((1, 2))
    assert typed_payload_digest({"1": "value"}) != typed_payload_digest(["1", "value"])
    assert typed_payload_digest(1) != typed_payload_digest(True)
    assert typed_payload_digest(1) != typed_payload_digest(1.0)


def test_typed_payload_digest_is_mapping_order_independent_and_tensor_sensitive() -> (
    None
):
    tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    left = {"tensor": tensor, "meta": {"label": "x", "count": 12}}
    right = {"meta": {"count": 12, "label": "x"}, "tensor": tensor}
    assert typed_payload_digest(left) == typed_payload_digest(right)
    assert typed_payload_digest(left) != typed_payload_digest(
        {**left, "tensor": tensor.view(torch.int32)}
    )
    assert typed_payload_digest(left) != typed_payload_digest(
        {**left, "tensor": tensor.reshape(2, 6)}
    )


def test_typed_payload_digest_refuses_ambiguous_or_unsupported_values() -> None:
    with pytest.raises(TypeError, match="string keys"):
        typed_payload_digest({1: "value"})
    with pytest.raises(TypeError, match="does not support"):
        typed_payload_digest(object())
    with pytest.raises(ValueError, match="non-finite"):
        typed_payload_digest(float("nan"))


def test_model_state_digest_is_worker_independent_and_binds_mapping_order() -> None:
    chunk = serialization._MODEL_STATE_DIGEST_CHUNK_BYTES
    state = OrderedDict(
        (
            ("z", torch.arange(chunk + 1, dtype=torch.uint8)),
            ("a", torch.arange(257, dtype=torch.int64)),
        )
    )
    reverse_order = OrderedDict(reversed(tuple(state.items())))

    serial = model_state_digest(state, max_workers=1)
    assert model_state_digest(state, max_workers=16) == serial
    assert model_state_digest(reverse_order, max_workers=16) != serial


def test_model_state_digest_binds_every_chunk_boundary() -> None:
    chunk = serialization._MODEL_STATE_DIGEST_CHUNK_BYTES
    original = torch.zeros(chunk + 1, dtype=torch.uint8)
    baseline = model_state_digest({"weight": original}, max_workers=16)

    for index in (0, chunk - 1, chunk):
        mutated = original.clone()
        mutated[index] = 1
        assert model_state_digest({"weight": mutated}, max_workers=16) != baseline


def test_model_state_digest_binds_field_metadata_and_values() -> None:
    values = torch.arange(8, dtype=torch.float32)
    baseline = model_state_digest({"weight": values})

    assert model_state_digest({"renamed": values}) != baseline
    assert model_state_digest({"weight": values.reshape(2, 4)}) != baseline
    assert model_state_digest({"weight": values.view(torch.int32)}) != baseline
    assert model_state_digest({"weight": values, "extra": values}) != baseline
    mutated = values.clone()
    mutated[3] += 1
    assert model_state_digest({"weight": mutated}) != baseline


@pytest.mark.parametrize("max_workers", [0, -1, 1.5, True])
def test_model_state_digest_rejects_invalid_worker_count(max_workers: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        model_state_digest({"weight": torch.zeros(1)}, max_workers=max_workers)  # type: ignore[arg-type]
