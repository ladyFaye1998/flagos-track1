"""Online (one-pass) softmax along the last dim, with fp32 accumulation."""

from __future__ import annotations

import torch

from ...utils import HAS_TRITON, has_cuda, next_power_of_2

if HAS_TRITON:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore

    @triton.jit
    def _softmax_kernel(
        x_ptr, y_ptr,
        n_rows, n_cols,
        stride_xr, stride_xc, stride_yr, stride_yc,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        if row >= n_rows:
            return
        col_offs = tl.arange(0, BLOCK_N)
        mask = col_offs < n_cols
        row_ptr = x_ptr + row * stride_xr
        x = tl.load(row_ptr + col_offs * stride_xc, mask=mask, other=-float("inf")).to(tl.float32)
        x = x - tl.max(x, axis=0)
        ex = tl.exp(x)
        denom = tl.sum(ex, axis=0)
        y = ex / denom
        out_ptr = y_ptr + row * stride_yr
        tl.store(out_ptr + col_offs * stride_yc, y, mask=mask)


def softmax_op(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    if not (HAS_TRITON and has_cuda() and x.is_cuda):
        return torch.softmax(x, dim=dim)
    if dim != -1 and dim != x.ndim - 1:
        # Reduce along last dim — move target dim to the end then back.
        perm = list(range(x.ndim))
        perm[dim], perm[-1] = perm[-1], perm[dim]
        y = softmax_op(x.permute(perm).contiguous(), dim=-1)
        inv = [0] * x.ndim
        for i, p in enumerate(perm):
            inv[p] = i
        return y.permute(inv).contiguous()

    x2 = x.contiguous().view(-1, x.shape[-1])
    n_rows, n_cols = x2.shape
    block_n = next_power_of_2(n_cols)
    if block_n > 16384:
        # Very wide rows: defer to PyTorch (handled by FlagGems' tiled path).
        return torch.softmax(x, dim=-1)
    y = torch.empty_like(x2)
    num_warps = 8 if block_n >= 4096 else 4 if block_n >= 1024 else 2
    _softmax_kernel[(n_rows,)](
        x2, y, n_rows, n_cols,
        x2.stride(0), x2.stride(1), y.stride(0), y.stride(1),
        BLOCK_N=block_n, num_warps=num_warps,
    )
    return y.view_as(x).to(x.dtype)
