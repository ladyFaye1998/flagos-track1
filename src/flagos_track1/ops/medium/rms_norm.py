"""Forward RMSNorm (Llama-style): y = x * rsqrt(mean(x^2) + eps) * w."""

from __future__ import annotations

import torch

from ...utils import HAS_TRITON, has_cuda, next_power_of_2

if HAS_TRITON:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore

    @triton.jit
    def _rmsnorm_kernel(
        x_ptr, y_ptr, w_ptr,
        n_rows, n_cols, eps,
        stride_xr, stride_yr,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        if row >= n_rows:
            return
        col_offs = tl.arange(0, BLOCK_N)
        mask = col_offs < n_cols
        x_row = tl.load(x_ptr + row * stride_xr + col_offs, mask=mask, other=0.0).to(tl.float32)
        var = tl.sum(x_row * x_row, axis=0) / n_cols
        rrms = 1.0 / tl.sqrt(var + eps)
        w = tl.load(w_ptr + col_offs, mask=mask, other=1.0).to(tl.float32)
        y = x_row * rrms * w
        tl.store(y_ptr + row * stride_yr + col_offs, y, mask=mask)


def rms_norm_op(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    if not (HAS_TRITON and has_cuda() and x.is_cuda):
        var = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        return ((x * torch.rsqrt(var + eps).to(x.dtype)) * weight).to(x.dtype)

    n_cols = x.shape[-1]
    block_n = next_power_of_2(n_cols)
    if block_n > 16384:
        var = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        return ((x * torch.rsqrt(var + eps).to(x.dtype)) * weight).to(x.dtype)
    x2 = x.contiguous().view(-1, n_cols)
    n_rows = x2.shape[0]
    y = torch.empty_like(x2)
    num_warps = 8 if block_n >= 4096 else 4 if block_n >= 1024 else 2
    _rmsnorm_kernel[(n_rows,)](
        x2, y, weight.contiguous(),
        n_rows, n_cols, float(eps),
        x2.stride(0), y.stride(0),
        BLOCK_N=block_n, num_warps=num_warps,
    )
    return y.view_as(x).to(x.dtype)
