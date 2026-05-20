"""Backward pass for affine LayerNorm.

Returns ``(grad_x, grad_weight, grad_bias)``. Uses fp32 reduction internally
and atomic-adds into the (small) weight/bias gradient buffers.
"""

from __future__ import annotations

import torch

from ...utils import HAS_TRITON, has_cuda, next_power_of_2

if HAS_TRITON:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore

    @triton.jit
    def _ln_bwd_kernel(
        x_ptr, g_ptr, w_ptr, gx_ptr, gw_ptr, gb_ptr,
        n_rows, n_cols, eps,
        stride_xr, stride_gr, stride_gxr,
        HAS_AFFINE: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        if row >= n_rows:
            return
        col_offs = tl.arange(0, BLOCK_N)
        mask = col_offs < n_cols
        x = tl.load(x_ptr + row * stride_xr + col_offs, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(g_ptr + row * stride_gr + col_offs, mask=mask, other=0.0).to(tl.float32)
        if HAS_AFFINE:
            w = tl.load(w_ptr + col_offs, mask=mask, other=1.0).to(tl.float32)
        else:
            w = tl.full([BLOCK_N], 1.0, tl.float32)
        mean = tl.sum(x, axis=0) / n_cols
        diff = tl.where(mask, x - mean, 0.0)
        var = tl.sum(diff * diff, axis=0) / n_cols
        rstd = 1.0 / tl.sqrt(var + eps)
        x_hat = diff * rstd
        gw = g * w
        m1 = tl.sum(gw, axis=0) / n_cols
        m2 = tl.sum(gw * x_hat, axis=0) / n_cols
        gx = rstd * (gw - m1 - x_hat * m2)
        tl.store(gx_ptr + row * stride_gxr + col_offs, gx, mask=mask)
        if HAS_AFFINE:
            tl.atomic_add(gw_ptr + col_offs, g * x_hat, mask=mask)
            tl.atomic_add(gb_ptr + col_offs, g, mask=mask)


def layer_norm_backward_op(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    normalized_shape,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if not isinstance(normalized_shape, (tuple, list)):
        normalized_shape = (normalized_shape,)
    has_affine = weight is not None
    n_cols = 1
    for d in normalized_shape:
        n_cols *= d

    def _fallback():
        x32 = x.to(torch.float32)
        g32 = grad_out.to(torch.float32)
        w32 = weight.to(torch.float32) if weight is not None else torch.ones(n_cols, device=x.device)
        mean = x32.mean(dim=-1, keepdim=True)
        var = x32.var(dim=-1, keepdim=True, unbiased=False)
        rstd = torch.rsqrt(var + eps)
        x_hat = (x32 - mean) * rstd
        gw = g32 * w32
        m1 = gw.mean(dim=-1, keepdim=True)
        m2 = (gw * x_hat).mean(dim=-1, keepdim=True)
        grad_x = rstd * (gw - m1 - x_hat * m2)
        grad_w = (g32 * x_hat).reshape(-1, n_cols).sum(dim=0).to(weight.dtype) if has_affine else None
        grad_b = g32.reshape(-1, n_cols).sum(dim=0).to(bias.dtype) if bias is not None else None
        return grad_x.to(x.dtype), grad_w, grad_b

    if not (HAS_TRITON and has_cuda() and x.is_cuda):
        return _fallback()
    block_n = next_power_of_2(n_cols)
    if block_n > 16384:
        return _fallback()

    x2 = x.contiguous().view(-1, n_cols)
    g2 = grad_out.contiguous().view(-1, n_cols)
    n_rows = x2.shape[0]
    gx = torch.empty_like(x2)
    gw_buf = torch.zeros(n_cols, device=x.device, dtype=torch.float32)
    gb_buf = torch.zeros(n_cols, device=x.device, dtype=torch.float32)
    num_warps = 8 if block_n >= 4096 else 4 if block_n >= 1024 else 2
    _ln_bwd_kernel[(n_rows,)](
        x2, g2,
        weight.contiguous() if has_affine else x2,
        gx, gw_buf, gb_buf,
        n_rows, n_cols, float(eps),
        x2.stride(0), g2.stride(0), gx.stride(0),
        HAS_AFFINE=has_affine,
        BLOCK_N=block_n, num_warps=num_warps,
    )
    grad_w = gw_buf.to(weight.dtype) if has_affine else None
    grad_b = gb_buf.to(bias.dtype) if bias is not None else None
    return gx.view_as(x).to(x.dtype), grad_w, grad_b
