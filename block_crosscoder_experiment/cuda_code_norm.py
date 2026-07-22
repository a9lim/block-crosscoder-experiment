"""Bitwise-native BF16 four-coordinate norms on CUDA.

Imported lazily by :mod:`block_crosscoder_experiment.model`, so CPU-only
installations do not need a Triton package.  The kernel is a no-grad
specialization for the Phase-2 block-dimension-four selector pools; every
other carrier retains the ordinary PyTorch norm.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from .runtime_limits import CUDA_CODE_NORM_LARGE_OUTPUTS

__all__ = ["cuda_code_norm4"]


@triton.jit
def _code_norm4(
    code,
    output,
    n_outputs,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < n_outputs
    base = offsets * 4
    x0 = tl.load(code + base, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(code + base + 1, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(code + base + 2, mask=mask, other=0.0).to(tl.float32)
    x3 = tl.load(code + base + 3, mask=mask, other=0.0).to(tl.float32)
    squared_norm = (x0 * x0 + x1 * x1) + (x2 * x2 + x3 * x3)
    # sqrt.rn is required for bitwise parity with torch.linalg.vector_norm;
    # Triton's generic sqrt may select a lower-precision approximation.
    norm = libdevice.sqrt_rn(squared_norm)
    tl.store(output + offsets, norm, mask=mask)


def cuda_code_norm4(code: torch.Tensor) -> torch.Tensor:
    """Return native-equivalent ``code.norm(dim=-1)`` on the CUDA carrier."""

    if not code.is_cuda or code.dtype != torch.bfloat16:
        raise TypeError("CUDA code-norm specialization requires a BF16 CUDA tensor")
    if code.ndim < 1 or code.shape[-1] != 4 or not code.is_contiguous():
        raise ValueError(
            "CUDA code-norm specialization requires contiguous final dimension four"
        )
    if torch.is_grad_enabled():
        raise RuntimeError("CUDA code-norm specialization is no-grad only")
    output = torch.empty(code.shape[:-1], device=code.device, dtype=code.dtype)
    if output.numel() == 0:
        return output
    block_size = 128 if output.numel() >= CUDA_CODE_NORM_LARGE_OUTPUTS else 256
    _code_norm4[(triton.cdiv(output.numel(), block_size),)](
        code,
        output,
        output.numel(),
        block_size=block_size,
        num_warps=4,
    )
    return output
