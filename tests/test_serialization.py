"""Determinism and sensitivity checks for durable tensor digests."""

from __future__ import annotations

from collections import OrderedDict

import pytest
import torch

import block_crosscoder_experiment.serialization as serialization
from block_crosscoder_experiment.serialization import model_state_digest


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
