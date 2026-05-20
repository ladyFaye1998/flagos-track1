"""Backward pass for RMSNorm: produces (grad_x, grad_weight).

Math (for a single feature row of length N):
    rrms          = 1 / sqrt(mean(x^2) + eps)
    x_hat         = x * rrms
    y             = x_hat * w               -> grad_w  = sum_rows(grad_out * x_hat)
                                              grad_xhat = grad_out * w
    Let m         = mean(grad_xhat * x_hat)
    grad_x        = rrms * (grad_xhat - x_hat * m)
"""

from __future__ import annotations

import torch

from ...utils import HAS_TRITON, has_cuda, next_power_of_2

if HAS_TRITON:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore

    @triton.jit
    def _rmsnorm_bwd_kernel(
        x_ptr, g_ptr, w_ptr, gx_ptr, gw_ptr,
        n_rows, n_cols, eps,
        stride_xr, stride_gr, stride_gxr,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        if row >= n_rows:
            return
        col_offs = tl.arange(0, BLOCK_N)
        mask = col_offs < n_cols
        x = tl.load(x_ptr + row * stride_xr + col_offs, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(g_ptr + row * stride_gr + col_offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + col_offs, mask=mask, other=0.0).to(tl.float32)

        var = tl.sum(x * x, axis=0) / n_cols
        rrms = 1.0 / tl.sqrt(var + eps)
        x_hat = x * rrms
        g_xhat = g * w
        m = tl.sum(g_xhat * x_hat, axis=0) / n_cols
        gx = rrms * (g_xhat - x_hat * m)
        tl.store(gx_ptr + row * stride_gxr + col_offs, gx, mask=mask)

        # grad_w contribution from this row: atomic_add into a fp32 buffer.
        contrib = g * x_hat
        tl.atomic_add(gw_ptr + col_offs, contrib, mask=mask)


def rms_norm_backward_op(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not (HAS_TRITON and has_cuda() and x.is_cuda):
        x32 = x.to(torch.float32)
        w32 = weight.to(torch.float32)
        g32 = grad_out.to(torch.float32)
        n = x.shape[-1]
        var = x32.pow(2).mean(dim=-1, keepdim=True)
        rrms = torch.rsqrt(var + eps)
        x_hat = x32 * rrms
        grad_w = (g32 * x_hat).reshape(-1, n).sum(dim=0).to(weight.dtype)
        g_xhat = g32 * w32
        mean_term = (g_xhat * x_hat).mean(dim=-1, keepdim=True)
        grad_x = rrms * (g_xhat - x_hat * mean_term)
        return grad_x.to(x.dtype), grad_w

    n_cols = x.shape[-1]
    block_n = next_power_of_2(n_cols)
    if block_n > 16384:
        return rms_norm_backward_op.__wrapped__(grad_out, x, weight, eps)  # type: ignore[attr-defined]

    x2 = x.contiguous().view(-1, n_cols)
    g2 = grad_out.contiguous().view(-1, n_cols)
    n_rows = x2.shape[0]
    grad_x = torch.empty_like(x2)
    grad_w = torch.zeros(n_cols, device=x.device, dtype=torch.float32)
    num_warps = 8 if block_n >= 4096 else 4 if block_n >= 1024 else 2
    _rmsnorm_bwd_kernel[(n_rows,)](
        x2, g2, weight.contiguous(), grad_x, grad_w,
        n_rows, n_cols, float(eps),
        x2.stride(0), g2.stride(0), grad_x.stride(0),
        BLOCK_N=block_n, num_warps=num_warps,
    )
    return grad_x.view_as(x).to(x.dtype), grad_w.to(weight.dtype)
