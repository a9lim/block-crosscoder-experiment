"""CUDA gates for certified bf16 gradient transfer."""

import math

import pytest
import torch

pytest.importorskip("triton")

from block_crosscoder_experiment.cuda_gradient_copy import (
    cuda_copy_bf16_gradient_and_flag_,
)
from block_crosscoder_experiment.trainer import _finite_gradients_with_l2_guard

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _run_certificate(
    source: torch.Tensor,
    *,
    rec: float = 1.0,
    total: float = 1.0,
) -> tuple[torch.Tensor, bool, bool, bool]:
    destination = torch.empty_like(source, dtype=torch.float32)
    unsafe = torch.zeros((), dtype=torch.int32, device="cuda")
    rec_tensor = torch.tensor(rec, dtype=torch.float32, device="cuda")
    total_tensor = torch.tensor(total, dtype=torch.float32, device="cuda")
    safe_l2_limit = math.sqrt(
        torch.finfo(torch.float32).max / source.numel()
    )
    cuda_copy_bf16_gradient_and_flag_(
        source,
        destination,
        unsafe,
        rec_tensor,
        total_tensor,
        safe_l2_limit=safe_l2_limit,
    )
    certificate = not bool(unsafe)
    fallback = _finite_gradients_with_l2_guard(
        rec_tensor,
        total_tensor,
        [destination],
    )
    admitted = certificate or fallback
    return destination, certificate, fallback, admitted


@pytest.mark.parametrize(
    ("case", "certificate", "fallback"),
    (
        ("ordinary", True, True),
        ("nan", False, False),
        ("positive_inf", False, False),
        ("negative_inf", False, False),
        # Above the dimension-aware sufficient bound, but with a finite exact
        # historical L2 norm: this must enter and pass the fallback.
        ("large_sparse", False, True),
        # Both the sufficient bound and historical fp32 L2 norm fail.
        ("large_dense", False, False),
    ),
)
def test_cuda_gradient_copy_matches_historical_fallback(
    case,
    certificate,
    fallback,
):
    generator = torch.Generator(device="cuda").manual_seed(2601)
    source = torch.randn(
        4096,
        generator=generator,
        dtype=torch.bfloat16,
        device="cuda",
    )
    if case == "nan":
        source[17] = float("nan")
    elif case == "positive_inf":
        source[17] = float("inf")
    elif case == "negative_inf":
        source[17] = -float("inf")
    elif case == "large_sparse":
        source.zero_()
        source[17] = 1e18
    elif case == "large_dense":
        source.fill_(1e30)

    destination, actual_certificate, actual_fallback, admitted = _run_certificate(
        source
    )
    torch.testing.assert_close(
        destination,
        source.float(),
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )
    historical = _finite_gradients_with_l2_guard(
        torch.tensor(1.0, device="cuda"),
        torch.tensor(1.0, device="cuda"),
        [source.float()],
    )
    assert actual_certificate is certificate
    assert actual_fallback is fallback
    assert admitted is historical


@pytest.mark.parametrize(("rec", "total"), ((float("nan"), 1.0), (1.0, float("inf"))))
def test_cuda_gradient_copy_flags_nonfinite_loss_scalars(rec, total):
    source = torch.ones(4096, dtype=torch.bfloat16, device="cuda")
    destination, certificate, fallback, admitted = _run_certificate(
        source,
        rec=rec,
        total=total,
    )
    assert torch.equal(destination, source.float())
    assert certificate is False
    assert fallback is False
    assert admitted is False


def test_cuda_gradient_copy_repeated_safe_execution_resets_exactly():
    generator = torch.Generator(device="cuda").manual_seed(2602)
    first = torch.randn(
        1 << 20,
        generator=generator,
        dtype=torch.bfloat16,
        device="cuda",
    )
    second = torch.randn(
        1 << 20,
        generator=generator,
        dtype=torch.bfloat16,
        device="cuda",
    )
    destination = torch.empty_like(first, dtype=torch.float32)
    unsafe = torch.ones((), dtype=torch.int32, device="cuda")
    rec = torch.ones((), device="cuda")
    total = torch.ones((), device="cuda")
    safe_l2_limit = math.sqrt(torch.finfo(torch.float32).max / first.numel())
    for source in (first, second):
        unsafe.zero_()
        cuda_copy_bf16_gradient_and_flag_(
            source,
            destination,
            unsafe,
            rec,
            total,
            safe_l2_limit=safe_l2_limit,
        )
        assert not bool(unsafe)
        assert torch.equal(destination, source.float())


def test_cuda_gradient_copy_preserves_transposed_dense_storage():
    source = torch.randn(
        2,
        4,
        dtype=torch.bfloat16,
        device="cuda",
    ).transpose(0, 1)
    destination = torch.empty_like(source, dtype=torch.float32)
    unsafe = torch.zeros((), dtype=torch.int32, device="cuda")
    rec = torch.ones((), device="cuda")
    total = torch.ones((), device="cuda")
    assert not source.is_contiguous()
    assert source.stride() == destination.stride() == (1, 4)
    cuda_copy_bf16_gradient_and_flag_(
        source,
        destination,
        unsafe,
        rec,
        total,
        safe_l2_limit=1e10,
    )
    assert not bool(unsafe)
    assert torch.equal(destination, source.float())
