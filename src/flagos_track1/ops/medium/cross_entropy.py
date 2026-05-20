"""Fused softmax cross-entropy (mean reduction, supports ``ignore_index``)."""

from __future__ import annotations

import torch

from ...utils import HAS_TRITON, has_cuda, next_power_of_2

if HAS_TRITON:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore

    @triton.jit
    def _ce_kernel(
        logits_ptr, target_ptr, loss_ptr, count_ptr,
        n_rows, n_classes, ignore_index,
        stride_lr,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        if row >= n_rows:
            return
        t = tl.load(target_ptr + row)
        is_valid = t != ignore_index
        col_offs = tl.arange(0, BLOCK_N)
        mask = col_offs < n_classes
        row_ptr = logits_ptr + row * stride_lr
        x = tl.load(row_ptr + col_offs, mask=mask, other=-float("inf")).to(tl.float32)
        m = tl.max(x, axis=0)
        x_shift = x - m
        logsumexp = m + tl.log(tl.sum(tl.exp(x_shift), axis=0))
        # log p(t) = x[t] - logsumexp
        x_t = tl.load(row_ptr + t).to(tl.float32)
        nll = logsumexp - x_t
        contrib = tl.where(is_valid, nll, 0.0)
        tl.atomic_add(loss_ptr, contrib)
        tl.atomic_add(count_ptr, tl.where(is_valid, 1, 0))


def cross_entropy_op(
    logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100
) -> torch.Tensor:
    if not (HAS_TRITON and has_cuda() and logits.is_cuda):
        return torch.nn.functional.cross_entropy(
            logits, targets, ignore_index=ignore_index, reduction="mean"
        )

    flat_logits = logits.reshape(-1, logits.shape[-1]).contiguous()
    flat_targets = targets.reshape(-1).to(torch.int64).contiguous()
    n_rows, n_classes = flat_logits.shape
    block_n = next_power_of_2(n_classes)
    if block_n > 32768:
        return torch.nn.functional.cross_entropy(
            logits, targets, ignore_index=ignore_index, reduction="mean"
        )

    loss = torch.zeros(1, device=logits.device, dtype=torch.float32)
    count = torch.zeros(1, device=logits.device, dtype=torch.int32)
    num_warps = 8 if block_n >= 4096 else 4 if block_n >= 1024 else 2
    _ce_kernel[(n_rows,)](
        flat_logits, flat_targets, loss, count,
        n_rows, n_classes, int(ignore_index),
        flat_logits.stride(0),
        BLOCK_N=block_n, num_warps=num_warps,
    )
    denom = count.float().clamp(min=1.0)
    return (loss / denom).squeeze().to(logits.dtype)
