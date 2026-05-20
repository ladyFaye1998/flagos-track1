"""Argmax reduction along the last dim (single-tile path)."""

from __future__ import annotations

import torch

from ...utils import HAS_TRITON, has_cuda, next_power_of_2

if HAS_TRITON:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore

    @triton.jit
    def _argmax_kernel(
        x_ptr, out_ptr,
        n_rows, n_cols,
        stride_xr,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        if row >= n_rows:
            return
        col_offs = tl.arange(0, BLOCK_N)
        mask = col_offs < n_cols
        x = tl.load(x_ptr + row * stride_xr + col_offs, mask=mask, other=-float("inf")).to(tl.float32)
        idx = tl.argmax(x, axis=0)
        tl.store(out_ptr + row, idx.to(tl.int64))


def argmax_op(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    if not (HAS_TRITON and has_cuda() and x.is_cuda):
        return torch.argmax(x, dim=dim)
    if dim != -1 and dim != x.ndim - 1:
        perm = list(range(x.ndim))
        perm[dim], perm[-1] = perm[-1], perm[dim]
        return argmax_op(x.permute(perm).contiguous(), dim=-1)

    x2 = x.contiguous().view(-1, x.shape[-1])
    n_rows, n_cols = x2.shape
    block_n = next_power_of_2(n_cols)
    if block_n > 16384:
        return torch.argmax(x, dim=dim)
    out = torch.empty(n_rows, device=x.device, dtype=torch.int64)
    num_warps = 8 if block_n >= 4096 else 4 if block_n >= 1024 else 2
    _argmax_kernel[(n_rows,)](
        x2, out, n_rows, n_cols,
        x2.stride(0),
        BLOCK_N=block_n, num_warps=num_warps,
    )
    return out.view(x.shape[:-1])
