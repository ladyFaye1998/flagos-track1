"""Forward LayerNorm (forward only) with optional affine weight + bias."""

from __future__ import annotations

import torch

from ...utils import HAS_TRITON, has_cuda, next_power_of_2

if HAS_TRITON:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore

    @triton.jit
    def _ln_kernel(
        x_ptr, y_ptr, w_ptr, b_ptr,
        n_rows, n_cols, eps,
        stride_xr, stride_yr,
        HAS_W: tl.constexpr, HAS_B: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        if row >= n_rows:
            return
        col_offs = tl.arange(0, BLOCK_N)
        mask = col_offs < n_cols
        x_row = tl.load(x_ptr + row * stride_xr + col_offs, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(x_row, axis=0) / n_cols
        diff = tl.where(mask, x_row - mean, 0.0)
        var = tl.sum(diff * diff, axis=0) / n_cols
        rstd = 1.0 / tl.sqrt(var + eps)
        y = diff * rstd
        if HAS_W:
            w = tl.load(w_ptr + col_offs, mask=mask, other=1.0).to(tl.float32)
            y = y * w
        if HAS_B:
            b = tl.load(b_ptr + col_offs, mask=mask, other=0.0).to(tl.float32)
            y = y + b
        tl.store(y_ptr + row * stride_yr + col_offs, y, mask=mask)


def layer_norm_op(
    x: torch.Tensor,
    normalized_shape,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    fallback = torch.nn.functional.layer_norm(x, normalized_shape, weight=weight, bias=bias, eps=eps)
    if not (HAS_TRITON and has_cuda() and x.is_cuda):
        return fallback
    if not isinstance(normalized_shape, (tuple, list)):
        normalized_shape = (normalized_shape,)
    if tuple(normalized_shape) != tuple(x.shape[-len(normalized_shape):]):
        return fallback
    n_cols = 1
    for d in normalized_shape:
        n_cols *= d
    block_n = next_power_of_2(n_cols)
    if block_n > 16384:
        return fallback
    x2 = x.contiguous().view(-1, n_cols)
    n_rows = x2.shape[0]
    y = torch.empty_like(x2)
    w = weight.contiguous().view(-1) if weight is not None else x2
    b = bias.contiguous().view(-1) if bias is not None else x2
    num_warps = 8 if block_n >= 4096 else 4 if block_n >= 1024 else 2
    _ln_kernel[(n_rows,)](
        x2, y, w, b,
        n_rows, n_cols, float(eps),
        x2.stride(0), y.stride(0),
        HAS_W=weight is not None, HAS_B=bias is not None,
        BLOCK_N=block_n, num_warps=num_warps,
    )
    return y.view_as(x).to(x.dtype)
