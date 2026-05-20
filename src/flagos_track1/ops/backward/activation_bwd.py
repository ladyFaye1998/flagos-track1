"""Backward kernels for SiLU and GELU (exact + tanh approx)."""

from __future__ import annotations

import torch

from ...utils import HAS_TRITON, has_cuda

if HAS_TRITON:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore

    @triton.jit
    def _silu_bwd_kernel(x_ptr, g_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        s = 1.0 / (1.0 + tl.exp(-x))
        out = g * s * (1.0 + x * (1.0 - s))  # d/dx (x * sigmoid(x))
        tl.store(out_ptr + offs, out, mask=mask)

    @triton.jit
    def _gelu_exact_bwd_kernel(x_ptr, g_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        inv_sqrt2 = 0.7071067811865475
        inv_sqrt_2pi = 0.3989422804014327  # 1 / sqrt(2*pi)
        cdf = 0.5 * (1.0 + tl.erf(x * inv_sqrt2))
        pdf = inv_sqrt_2pi * tl.exp(-0.5 * x * x)
        out = g * (cdf + x * pdf)
        tl.store(out_ptr + offs, out, mask=mask)

    @triton.jit
    def _gelu_tanh_bwd_kernel(x_ptr, g_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        k0 = 0.7978845608028654  # sqrt(2/pi)
        k1 = 0.044715
        x2 = x * x
        x3 = x2 * x
        inner = k0 * (x + k1 * x3)
        e2 = tl.exp(2.0 * inner)
        t = (e2 - 1.0) / (e2 + 1.0)
        d_inner = k0 * (1.0 + 3.0 * k1 * x2)
        d_t = (1.0 - t * t) * d_inner
        out = g * (0.5 * (1.0 + t) + 0.5 * x * d_t)
        tl.store(out_ptr + offs, out, mask=mask)


def _launch(kernel, x, g):
    x_c = x.contiguous()
    g_c = g.contiguous()
    out = torch.empty_like(x_c)
    n = x_c.numel()
    if n == 0:
        return out.view_as(x)
    BLOCK = 1024
    grid = ((n + BLOCK - 1) // BLOCK,)
    kernel[grid](x_c, g_c, out, n, BLOCK=BLOCK, num_warps=4)
    return out.view_as(x).to(x.dtype)


def silu_backward_op(grad_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if not (HAS_TRITON and has_cuda() and x.is_cuda):
        s = torch.sigmoid(x.float())
        return (grad_out.float() * s * (1.0 + x.float() * (1.0 - s))).to(x.dtype)
    return _launch(_silu_bwd_kernel, x, grad_out)


def gelu_backward_op(
    grad_out: torch.Tensor, x: torch.Tensor, approximate: str = "none"
) -> torch.Tensor:
    if not (HAS_TRITON and has_cuda() and x.is_cuda):
        import math
        xf = x.float()
        gf = grad_out.float()
        if approximate == "tanh":
            k0 = math.sqrt(2.0 / math.pi)
            k1 = 0.044715
            inner = k0 * (xf + k1 * xf.pow(3))
            t = torch.tanh(inner)
            d_inner = k0 * (1.0 + 3.0 * k1 * xf.pow(2))
            d_t = (1.0 - t.pow(2)) * d_inner
            grad = gf * (0.5 * (1.0 + t) + 0.5 * xf * d_t)
        else:
            cdf = 0.5 * (1.0 + torch.erf(xf / math.sqrt(2.0)))
            pdf = torch.exp(-0.5 * xf.pow(2)) / math.sqrt(2.0 * math.pi)
            grad = gf * (cdf + xf * pdf)
        return grad.to(x.dtype)
    kernel = _gelu_tanh_bwd_kernel if approximate == "tanh" else _gelu_exact_bwd_kernel
    return _launch(kernel, x, grad_out)
