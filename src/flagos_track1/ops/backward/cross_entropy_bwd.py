"""Backward pass for mean-reduced softmax cross-entropy.

Given saved logits and integer targets, returns ``grad_logits`` of the same
shape as ``logits``. ``ignore_index`` rows contribute zero gradient.
"""

from __future__ import annotations

import torch

from ...utils import HAS_TRITON, has_cuda, next_power_of_2

if HAS_TRITON:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore

    @triton.jit
    def _ce_bwd_kernel(
        logits_ptr, target_ptr, grad_loss_ptr, grad_logits_ptr,
        valid_count, n_rows, n_classes, ignore_index,
        stride_lr, stride_gr,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        if row >= n_rows:
            return
        t = tl.load(target_ptr + row)
        is_valid = t != ignore_index
        col_offs = tl.arange(0, BLOCK_N)
        mask = col_offs < n_classes
        x_row = tl.load(logits_ptr + row * stride_lr + col_offs, mask=mask, other=-float("inf")).to(tl.float32)
        m = tl.max(x_row, axis=0)
        ex = tl.exp(x_row - m)
        probs = ex / tl.sum(ex, axis=0)
        # grad = (softmax - onehot(t)) * grad_loss / valid_count
        onehot = tl.where(col_offs == t, 1.0, 0.0)
        grad_loss = tl.load(grad_loss_ptr).to(tl.float32)
        scale = grad_loss / valid_count
        grad = (probs - onehot) * scale
        grad = tl.where(is_valid, grad, 0.0)
        tl.store(grad_logits_ptr + row * stride_gr + col_offs, grad, mask=mask)


def cross_entropy_backward_op(
    grad_loss: torch.Tensor,
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1).to(torch.int64)
    valid_count = (flat_targets != ignore_index).sum().clamp(min=1).item()

    def _fallback():
        x32 = flat_logits.float()
        probs = torch.softmax(x32, dim=-1)
        onehot = torch.zeros_like(probs)
        valid = flat_targets != ignore_index
        idx = flat_targets.clone()
        idx[~valid] = 0
        onehot.scatter_(1, idx.unsqueeze(1), 1.0)
        grad = (probs - onehot) * (grad_loss.item() / valid_count)
        grad[~valid] = 0.0
        return grad.view_as(logits).to(logits.dtype)

    if not (HAS_TRITON and has_cuda() and logits.is_cuda):
        return _fallback()
    n_rows, n_classes = flat_logits.shape
    block_n = next_power_of_2(n_classes)
    if block_n > 32768:
        return _fallback()

    grad_logits = torch.empty_like(flat_logits, dtype=torch.float32)
    grad_loss_t = grad_loss.detach().to(torch.float32).reshape(-1)[:1].contiguous()
    num_warps = 8 if block_n >= 4096 else 4 if block_n >= 1024 else 2
    _ce_bwd_kernel[(n_rows,)](
        flat_logits.contiguous(), flat_targets.contiguous(),
        grad_loss_t, grad_logits,
        int(valid_count), n_rows, n_classes, int(ignore_index),
        flat_logits.stride(0), grad_logits.stride(0),
        BLOCK_N=block_n, num_warps=num_warps,
    )
    return grad_logits.view_as(logits).to(logits.dtype)
