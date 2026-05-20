"""Rotary Position Embedding (RoPE) — interleaved (even/odd) layout.

Apply per-position rotation matrices to the last dim of ``x``:
``y[..., 0::2] = x[..., 0::2] * cos - x[..., 1::2] * sin``
``y[..., 1::2] = x[..., 0::2] * sin + x[..., 1::2] * cos``
"""

from __future__ import annotations

import torch

from ...utils import HAS_TRITON, has_cuda

if HAS_TRITON:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore

    @triton.jit
    def _rope_kernel(
        X, Out, Cos, Sin,
        N_ROWS, HALF_D,
        stride_xr, stride_or,
        stride_cs,
        BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0)
        if row >= N_ROWS:
            return
        offs = tl.arange(0, BLOCK_D)
        mask = offs < HALF_D
        cos = tl.load(Cos + row * stride_cs + offs, mask=mask, other=0.0).to(tl.float32)
        sin = tl.load(Sin + row * stride_cs + offs, mask=mask, other=0.0).to(tl.float32)
        x1 = tl.load(X + row * stride_xr + 2 * offs, mask=mask, other=0.0).to(tl.float32)
        x2 = tl.load(X + row * stride_xr + 2 * offs + 1, mask=mask, other=0.0).to(tl.float32)
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        tl.store(Out + row * stride_or + 2 * offs, y1, mask=mask)
        tl.store(Out + row * stride_or + 2 * offs + 1, y2, mask=mask)


def rope_op(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """``x`` shape: (..., D). ``cos`` / ``sin`` broadcast to (N_rows, D/2)."""
    if not (HAS_TRITON and has_cuda() and x.is_cuda):
        x1, x2 = x[..., 0::2], x[..., 1::2]
        rot1 = x1 * cos - x2 * sin
        rot2 = x1 * sin + x2 * cos
        out = torch.empty_like(x)
        out[..., 0::2] = rot1
        out[..., 1::2] = rot2
        return out

    D = x.shape[-1]
    half_d = D // 2
    flat = x.contiguous().view(-1, D)
    n_rows = flat.shape[0]
    out = torch.empty_like(flat)

    # cos/sin broadcast to (n_rows, half_d) — usually they're (seq, half_d).
    cs = cos.contiguous().view(-1, half_d)
    sn = sin.contiguous().view(-1, half_d)
    if cs.shape[0] != n_rows:
        cs = cs.expand(n_rows, half_d).contiguous()
        sn = sn.expand(n_rows, half_d).contiguous()

    BLOCK_D = 1
    while BLOCK_D < half_d and BLOCK_D < 4096:
        BLOCK_D *= 2

    _rope_kernel[(n_rows,)](
        flat, out, cs, sn,
        n_rows, half_d,
        flat.stride(0), out.stride(0),
        cs.stride(0),
        BLOCK_D=BLOCK_D, num_warps=4,
    )
    return out.view_as(x).to(x.dtype)
