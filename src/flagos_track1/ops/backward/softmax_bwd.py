"""Backward pass for softmax along the last dim.

Math: ``grad_x = (grad_y - sum(grad_y * y)) * y`` where ``y = softmax(x)``.
"""

from __future__ import annotations

import torch

from ...utils import HAS_TRITON, has_cuda, next_power_of_2

if HAS_TRITON:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore

    @triton.jit
    def _softmax_bwd_kernel(
        gy_ptr, y_ptr, gx_ptr,
        n_rows, n_cols,
        stride_g, stride_y, stride_gx,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        if row >= n_rows:
            return
        col_offs = tl.arange(0, BLOCK_N)
        mask = col_offs < n_cols
        gy = tl.load(gy_ptr + row * stride_g + col_offs, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(y_ptr + row * stride_y + col_offs, mask=mask, other=0.0).to(tl.float32)
        s = tl.sum(gy * y, axis=0)
        gx = (gy - s) * y
        tl.store(gx_ptr + row * stride_gx + col_offs, gx, mask=mask)


def softmax_backward_op(grad_y: torch.Tensor, y: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Returns grad_x given grad_y and the saved forward output y."""
    fallback = (grad_y - (grad_y * y).sum(dim=dim, keepdim=True)) * y
    if not (HAS_TRITON and has_cuda() and grad_y.is_cuda):
        return fallback.to(grad_y.dtype)
    if dim != -1 and dim != grad_y.ndim - 1:
        return fallback.to(grad_y.dtype)
    n_cols = y.shape[-1]
    block_n = next_power_of_2(n_cols)
    if block_n > 16384:
        return fallback.to(grad_y.dtype)
    gy2 = grad_y.contiguous().view(-1, n_cols)
    y2 = y.contiguous().view(-1, n_cols)
    gx = torch.empty_like(gy2)
    n_rows = gy2.shape[0]
    num_warps = 8 if block_n >= 4096 else 4 if block_n >= 1024 else 2
    _softmax_bwd_kernel[(n_rows,)](
        gy2, y2, gx,
        n_rows, n_cols,
        gy2.stride(0), y2.stride(0), gx.stride(0),
        BLOCK_N=block_n, num_warps=num_warps,
    )
    return gx.view_as(grad_y).to(grad_y.dtype)
