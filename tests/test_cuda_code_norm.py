"""CUDA release gates for the exact BF16 four-coordinate norm."""

import pytest
import torch

pytest.importorskip("triton")

import block_crosscoder_experiment.cuda_code_norm as cuda_code_norm_module
from block_crosscoder_experiment.cuda_code_norm import cuda_code_norm4
from block_crosscoder_experiment.model import BSCConfig, BlockCrosscoder
from block_crosscoder_experiment.runtime_limits import (
    CODE_NORM_NATIVE_IMPLEMENTATION,
    CUDA_CODE_NORM_LARGE_OUTPUTS,
    CUDA_CODE_NORM_MIN_OUTPUTS,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _equal_including_nan_payload_agnostic(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> bool:
    return bool(((actual == expected) | (actual.isnan() & expected.isnan())).all())


def test_cuda_code_norm4_is_bitwise_native_on_finite_and_pathological_values():
    generator = torch.Generator(device="cuda").manual_seed(2801)
    cases = [
        torch.randn(
            257,
            33,
            4,
            generator=generator,
            dtype=torch.bfloat16,
            device="cuda",
        )
        * scale
        for scale in (2.0**-40, 2.0**-10, 1.0, 2.0**10, 2.0**40)
    ]
    cases.append(
        torch.tensor(
            [
                [0.0, -0.0, 0.0, -0.0],
                [float("inf"), 1.0, 2.0, 3.0],
                [-float("inf"), 1.0, 2.0, 3.0],
                [float("nan"), 1.0, 2.0, 3.0],
                [torch.finfo(torch.bfloat16).max, 0.0, 0.0, 0.0],
                [torch.finfo(torch.bfloat16).tiny, 0.0, 0.0, 0.0],
            ],
            dtype=torch.bfloat16,
            device="cuda",
        )
    )
    for code in cases:
        with torch.no_grad():
            actual = cuda_code_norm4(code)
            expected = code.norm(dim=-1)
        assert _equal_including_nan_payload_agnostic(actual, expected)


@pytest.mark.parametrize(
    "outputs",
    (CUDA_CODE_NORM_MIN_OUTPUTS, CUDA_CODE_NORM_LARGE_OUTPUTS),
)
def test_cuda_code_norm4_is_bitwise_native_at_both_bound_tile_schedules(outputs):
    code = torch.randn(
        outputs,
        4,
        generator=torch.Generator(device="cuda").manual_seed(2807 + outputs),
        dtype=torch.bfloat16,
        device="cuda",
    )
    with torch.no_grad():
        actual = cuda_code_norm4(code)
        expected = code.norm(dim=-1)
    assert torch.equal(actual, expected)


def test_large_no_grad_model_scores_dispatch_to_cuda_code_norm(monkeypatch):
    model = BlockCrosscoder(
        BSCConfig(
            n_blocks=1,
            block_dim=4,
            n_sites=1,
            d_model=4,
            k=1,
            decoder_constraint="free",
            decoder_bias=False,
        )
    )
    code = torch.randn(
        CUDA_CODE_NORM_MIN_OUTPUTS,
        4,
        generator=torch.Generator(device="cuda").manual_seed(2813),
        dtype=torch.bfloat16,
        device="cuda",
    )
    expected = code.norm(dim=-1)
    calls = 0
    original = cuda_code_norm_module.cuda_code_norm4

    def observed(value):
        nonlocal calls
        calls += 1
        return original(value)

    monkeypatch.setattr(cuda_code_norm_module, "cuda_code_norm4", observed)
    with torch.no_grad():
        actual = model.scores(code)
    assert calls == 1
    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    "case",
    ("grad_enabled", "fp32", "block_dim", "noncontiguous", "small", "native_id"),
)
def test_model_code_norm_falls_back_outside_complete_cuda_carrier(monkeypatch, case):
    identity = (
        {"code_norm_implementation": CODE_NORM_NATIVE_IMPLEMENTATION}
        if case == "native_id"
        else {}
    )
    model = BlockCrosscoder(
        BSCConfig(
            n_blocks=1,
            block_dim=4,
            n_sites=1,
            d_model=4,
            k=1,
            decoder_constraint="free",
            decoder_bias=False,
            **identity,
        )
    )
    outputs = CUDA_CODE_NORM_MIN_OUTPUTS
    dtype = torch.float32 if case == "fp32" else torch.bfloat16
    width = 3 if case == "block_dim" else 4
    rows = 1024 if case == "small" else outputs
    if case == "noncontiguous":
        code = torch.randn(
            4,
            rows,
            dtype=dtype,
            device="cuda",
        ).transpose(0, 1)
    else:
        code = torch.randn(rows, width, dtype=dtype, device="cuda")

    def refused(_value):
        raise AssertionError("CUDA specialization crossed a fallback gate")

    monkeypatch.setattr(cuda_code_norm_module, "cuda_code_norm4", refused)
    if case == "grad_enabled":
        actual = model.scores(code)
    else:
        with torch.no_grad():
            actual = model.scores(code)
    assert torch.equal(actual, code.norm(dim=-1))


def test_cuda_code_norm4_refuses_invalid_direct_calls():
    with pytest.raises(TypeError, match="BF16 CUDA"):
        with torch.no_grad():
            cuda_code_norm4(torch.ones(4, 4, device="cuda"))
    with pytest.raises(ValueError, match="contiguous final dimension four"):
        with torch.no_grad():
            cuda_code_norm4(
                torch.ones(4, 5, 4, dtype=torch.bfloat16, device="cuda").transpose(
                    0, 1
                )
            )
    with pytest.raises(RuntimeError, match="no-grad only"):
        cuda_code_norm4(torch.ones(4, 4, dtype=torch.bfloat16, device="cuda"))
