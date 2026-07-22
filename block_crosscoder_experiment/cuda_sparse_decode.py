"""Low-density CUDA decode for hard-TopK factorized training.

Imported lazily by :mod:`block_crosscoder_experiment.model`, so CPU-only
installations do not need a Triton package.  The kernel consumes the selected
block mask directly and never materializes the dense zero-filled code.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

__all__ = ["cuda_sparse_topk_decode"]


@triton.jit
def _sparse_decode_forward(
    code,
    weight,
    selected_groups,
    row_ptr,
    output,
    output_width: tl.constexpr,
    groups: tl.constexpr,
    block_dim: tl.constexpr,
    output_tile: tl.constexpr,
):
    row = tl.program_id(0)
    output_offsets = tl.program_id(1) * output_tile + tl.arange(0, output_tile)
    start = tl.load(row_ptr + row)
    stop = tl.load(row_ptr + row + 1)
    accumulator = tl.zeros((output_tile,), tl.float32)
    for event in range(start, stop):
        group = tl.load(selected_groups + event)
        for coordinate in tl.static_range(block_dim):
            value = tl.load(
                code + row * (groups * block_dim) + group * block_dim + coordinate
            )
            decoder = tl.load(
                weight
                + (group * block_dim + coordinate) * output_width
                + output_offsets,
                mask=output_offsets < output_width,
                other=0.0,
            )
            accumulator += value * decoder
    tl.store(
        output + row * output_width + output_offsets,
        accumulator,
        mask=output_offsets < output_width,
    )


@triton.jit
def _sparse_decode_backward_code(
    grad_output,
    weight,
    rows,
    selected_groups,
    grad_code,
    output_width: tl.constexpr,
    groups: tl.constexpr,
    block_dim: tl.constexpr,
    output_tile: tl.constexpr,
):
    event = tl.program_id(0)
    coordinate = tl.program_id(1)
    row = tl.load(rows + event)
    group = tl.load(selected_groups + event)
    accumulator = tl.zeros((output_tile,), tl.float32)
    for output_start in tl.static_range(0, output_width, output_tile):
        output_offsets = output_start + tl.arange(0, output_tile)
        grad = tl.load(
            grad_output + row * output_width + output_offsets,
            mask=output_offsets < output_width,
            other=0.0,
        )
        decoder = tl.load(
            weight + (group * block_dim + coordinate) * output_width + output_offsets,
            mask=output_offsets < output_width,
            other=0.0,
        )
        accumulator += grad * decoder
    tl.store(
        grad_code + row * (groups * block_dim) + group * block_dim + coordinate,
        tl.sum(accumulator, axis=0),
    )


@triton.jit
def _sparse_decode_backward_weight(
    grad_output,
    code,
    group_rows,
    group_ptr,
    grad_weight,
    output_width: tl.constexpr,
    groups: tl.constexpr,
    block_dim: tl.constexpr,
    output_tile: tl.constexpr,
):
    group = tl.program_id(0)
    coordinate = tl.program_id(1)
    output_offsets = tl.program_id(2) * output_tile + tl.arange(0, output_tile)
    start = tl.load(group_ptr + group)
    stop = tl.load(group_ptr + group + 1)
    accumulator = tl.zeros((output_tile,), tl.float32)
    for event in range(start, stop):
        row = tl.load(group_rows + event)
        value = tl.load(
            code + row * (groups * block_dim) + group * block_dim + coordinate
        )
        grad = tl.load(
            grad_output + row * output_width + output_offsets,
            mask=output_offsets < output_width,
            other=0.0,
        )
        accumulator += value * grad
    tl.store(
        grad_weight + (group * block_dim + coordinate) * output_width + output_offsets,
        accumulator,
        mask=output_offsets < output_width,
    )


class _SparseTopKDecode(torch.autograd.Function):
    """Autograd carrier with deterministic row/group reduction orders."""

    @staticmethod
    def forward(
        ctx,
        code: torch.Tensor,
        mask: torch.Tensor,
        weight: torch.Tensor,
        selected_count: int,
    ) -> torch.Tensor:
        batch, groups, block_dim = code.shape
        output_width = weight.shape[1]
        events = torch.nonzero_static(mask, size=selected_count)
        rows = events[:, 0].to(torch.int32)
        selected_groups = events[:, 1].to(torch.int32)

        row_counts = torch.bincount(rows, minlength=batch)
        row_ptr = torch.empty(batch + 1, dtype=torch.int64, device=code.device)
        row_ptr[0] = 0
        torch.cumsum(row_counts, dim=0, out=row_ptr[1:])

        group_order = torch.argsort(selected_groups, stable=True)
        group_rows = rows[group_order]
        group_counts = torch.bincount(selected_groups, minlength=groups)
        group_ptr = torch.empty(groups + 1, dtype=torch.int64, device=code.device)
        group_ptr[0] = 0
        torch.cumsum(group_counts, dim=0, out=group_ptr[1:])

        output = torch.empty(
            batch,
            output_width,
            dtype=code.dtype,
            device=code.device,
        )
        output_tile = 128
        _sparse_decode_forward[(batch, triton.cdiv(output_width, output_tile))](
            code,
            weight,
            selected_groups,
            row_ptr,
            output,
            output_width=output_width,
            groups=groups,
            block_dim=block_dim,
            output_tile=output_tile,
            num_warps=4,
        )
        ctx.save_for_backward(
            code,
            weight,
            rows,
            selected_groups,
            group_rows,
            group_ptr,
        )
        ctx.shape = (batch, groups, block_dim, output_width)
        return output

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None, torch.Tensor, None]:
        (
            code,
            weight,
            rows,
            selected_groups,
            group_rows,
            group_ptr,
        ) = ctx.saved_tensors
        batch, groups, block_dim, output_width = ctx.shape
        grad_output = grad_output.contiguous()
        grad_code = torch.zeros_like(code)
        grad_weight = torch.empty_like(weight)
        output_tile = 128
        if len(rows):
            _sparse_decode_backward_code[(len(rows), block_dim)](
                grad_output,
                weight,
                rows,
                selected_groups,
                grad_code,
                output_width=output_width,
                groups=groups,
                block_dim=block_dim,
                output_tile=output_tile,
                num_warps=4,
            )
        _sparse_decode_backward_weight[
            (
                groups,
                block_dim,
                triton.cdiv(output_width, output_tile),
            )
        ](
            grad_output,
            code,
            group_rows,
            group_ptr,
            grad_weight,
            output_width=output_width,
            groups=groups,
            block_dim=block_dim,
            output_tile=output_tile,
            num_warps=4,
        )
        return grad_code, None, grad_weight, None


def cuda_sparse_topk_decode(
    code: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
    *,
    selected_count: int,
) -> torch.Tensor:
    """Decode a low-density hard-TopK code through a packed rank-space map."""

    if not code.is_cuda or not mask.is_cuda or not weight.is_cuda:
        raise ValueError("sparse TopK decode requires CUDA tensors")
    if code.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise TypeError("sparse TopK decode requires bf16 code and weight tensors")
    if mask.dtype != torch.bool:
        raise TypeError("sparse TopK decode mask must be boolean")
    if code.ndim != 3 or mask.shape != code.shape[:2]:
        raise ValueError("sparse TopK decode mask must match [batch, groups]")
    if weight.ndim != 2 or weight.shape[0] != code.shape[1] * code.shape[2]:
        raise ValueError("sparse TopK decode weight must have shape [groups*block, d]")
    if (
        isinstance(selected_count, bool)
        or int(selected_count) != selected_count
        or not 0 < selected_count <= mask.numel()
    ):
        raise ValueError("selected_count must be an integer in [1, mask.numel()]")
    if (
        not code.is_contiguous()
        or not mask.is_contiguous()
        or not weight.is_contiguous()
    ):
        raise ValueError("sparse TopK decode tensors must be contiguous")
    return _SparseTopKDecode.apply(code, mask, weight, int(selected_count))
