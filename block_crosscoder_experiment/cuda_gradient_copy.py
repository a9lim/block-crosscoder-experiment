"""Certified bf16-to-fp32 CUDA gradient transfer.

Imported lazily by :mod:`block_crosscoder_experiment.trainer`, so CPU-only
installations do not need a Triton package.  The copy kernel carries the
ordinary nonlogging finite/L2-safe certificate while the source gradient is
already in flight, avoiding a second complete read of the fp32 master buffer.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

__all__ = ["cuda_copy_bf16_gradient_and_flag_"]


def _has_dense_base_storage(tensor: torch.Tensor) -> bool:
    if tensor.layout != torch.strided or tensor.storage_offset() != 0:
        return False
    expected_stride = 1
    for size, stride in sorted(
        (
            (size, stride)
            for size, stride in zip(tensor.shape, tensor.stride(), strict=True)
            if size > 1
        ),
        key=lambda item: item[1],
    ):
        if stride != expected_stride:
            return False
        expected_stride *= size
    return expected_stride == tensor.numel()


@triton.jit
def _copy_bf16_gradient_and_flag(
    source,
    destination,
    unsafe,
    rec,
    total,
    n_elements,
    safe_l2_limit,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < n_elements
    values = tl.load(source + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(destination + offsets, values, mask=mask)

    unsafe_values = mask & (
        (values != values) | (tl.abs(values) >= safe_l2_limit)
    )
    block_unsafe = tl.max(unsafe_values.to(tl.int32), axis=0)
    if tl.program_id(0) == 0:
        rec_value = tl.load(rec)
        total_value = tl.load(total)
        scalar_unsafe = (
            (rec_value != rec_value)
            | (tl.abs(rec_value) == float("inf"))
            | (total_value != total_value)
            | (tl.abs(total_value) == float("inf"))
        )
        block_unsafe |= scalar_unsafe.to(tl.int32)
    # The admitted path performs no atomics.  Only an unsafe block contends
    # for the one persistent flag, and any nonzero value has the same meaning.
    tl.atomic_or(unsafe, 1, mask=block_unsafe != 0)


def cuda_copy_bf16_gradient_and_flag_(
    source: torch.Tensor,
    destination: torch.Tensor,
    unsafe: torch.Tensor,
    rec: torch.Tensor,
    total: torch.Tensor,
    *,
    safe_l2_limit: float,
) -> None:
    """Copy one gradient exactly and accumulate an unsafe-certificate flag.

    A zero flag after every current gradient has passed through this function
    proves that ``rec`` and ``total`` are finite, every copied gradient is
    finite, and the historical global fp32 L2 reduction cannot overflow.
    Callers must fall back to their exact historical check when the flag is
    nonzero.
    """

    if (
        not source.is_cuda
        or not destination.is_cuda
        or not unsafe.is_cuda
        or not rec.is_cuda
        or not total.is_cuda
    ):
        raise ValueError("certified gradient copy requires CUDA tensors")
    if source.device != destination.device or any(
        tensor.device != source.device for tensor in (unsafe, rec, total)
    ):
        raise ValueError("certified gradient copy tensors must share one device")
    if source.dtype != torch.bfloat16 or destination.dtype != torch.float32:
        raise TypeError("certified gradient copy requires bf16 source and fp32 output")
    if unsafe.dtype != torch.int32 or unsafe.numel() != 1:
        raise TypeError("certified gradient copy flag must be one int32 scalar")
    if rec.numel() != 1 or total.numel() != 1:
        raise ValueError("certified gradient copy losses must be scalar tensors")
    if source.shape != destination.shape:
        raise ValueError("certified gradient source and destination shapes differ")
    if (
        source.stride() != destination.stride()
        or not _has_dense_base_storage(source)
        or not _has_dense_base_storage(destination)
    ):
        raise ValueError(
            "certified gradient copy tensors must share one dense storage layout"
        )
    if source.numel() == 0:
        raise ValueError("certified gradient copy requires a nonempty tensor")
    if not math.isfinite(safe_l2_limit) or safe_l2_limit <= 0.0:
        raise ValueError("certified gradient copy requires a finite positive bound")

    block_size = 8192
    _copy_bf16_gradient_and_flag[(triton.cdiv(source.numel(), block_size),)](
        source,
        destination,
        unsafe,
        rec,
        total,
        source.numel(),
        safe_l2_limit,
        block_size=block_size,
        num_warps=4,
    )
