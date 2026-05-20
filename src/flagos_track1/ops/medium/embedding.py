"""Forward embedding lookup with optional ``padding_idx`` zeroing."""

from __future__ import annotations

import torch

from ...utils import HAS_TRITON, has_cuda

if HAS_TRITON:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore

    @triton.jit
    def _embedding_kernel(
        out_ptr, idx_ptr, w_ptr,
        n_indices, dim, padding_idx,
        stride_wn, stride_on,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        if pid >= n_indices:
            return
        idx = tl.load(idx_ptr + pid)
        is_pad = idx == padding_idx
        col_offs = tl.arange(0, BLOCK_D)
        mask = col_offs < dim
        w_row = tl.load(w_ptr + idx * stride_wn + col_offs, mask=mask, other=0.0)
        w_row = tl.where(is_pad, tl.zeros_like(w_row), w_row)
        tl.store(out_ptr + pid * stride_on + col_offs, w_row, mask=mask)


def embedding_op(
    indices: torch.Tensor, weight: torch.Tensor, padding_idx: int | None = None
) -> torch.Tensor:
    if not (HAS_TRITON and has_cuda() and weight.is_cuda):
        return torch.nn.functional.embedding(indices, weight, padding_idx=padding_idx)

    dim = weight.shape[-1]
    flat_idx = indices.reshape(-1).to(torch.int64).contiguous()
    out = torch.empty(flat_idx.shape[0], dim, device=weight.device, dtype=weight.dtype)

    BLOCK_D = 1
    while BLOCK_D < dim and BLOCK_D < 4096:
        BLOCK_D *= 2

    pad = -1 if padding_idx is None else int(padding_idx)
    _embedding_kernel[(flat_idx.shape[0],)](
        out, flat_idx, weight.contiguous(),
        flat_idx.shape[0], dim, pad,
        weight.stride(0), out.stride(0),
        BLOCK_D=BLOCK_D, num_warps=4,
    )
    return out.view(*indices.shape, dim)
