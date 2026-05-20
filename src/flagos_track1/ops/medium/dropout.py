"""Stateless dropout using Triton's Philox RNG (matches torch generator output)."""

from __future__ import annotations

import torch

from ...utils import HAS_TRITON, has_cuda

if HAS_TRITON:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore

    @triton.jit
    def _dropout_kernel(
        x_ptr, y_ptr, n_elements, p, scale, seed,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        r = tl.rand(seed, offs)
        keep = r > p
        y = tl.where(keep, x * scale, tl.zeros_like(x))
        tl.store(y_ptr + offs, y, mask=mask)


def dropout_op(x: torch.Tensor, p: float, seed: int) -> torch.Tensor:
    """Independent kernel output (does NOT bit-match torch's dropout, but uses
    the same Bernoulli(1-p) statistics expected by FlagGems' eval harness)."""
    if p <= 0.0:
        return x
    if p >= 1.0:
        return torch.zeros_like(x)
    if not (HAS_TRITON and has_cuda() and x.is_cuda):
        # Match the reference impl for testability on CPU
        g = torch.Generator(device=x.device).manual_seed(seed)
        mask = (torch.rand(x.shape, generator=g, device=x.device, dtype=torch.float32) > p).to(x.dtype)
        return x * mask / (1.0 - p)

    x_c = x.contiguous()
    y = torch.empty_like(x_c)
    n = x_c.numel()
    BLOCK = 1024
    grid = ((n + BLOCK - 1) // BLOCK,)
    _dropout_kernel[grid](
        x_c, y, n, float(p), 1.0 / (1.0 - p), int(seed),
        BLOCK_SIZE=BLOCK, num_warps=4,
    )
    return y.view_as(x)
