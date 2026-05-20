"""Fused MoE router: hidden -> router_logits -> softmax -> top-k (+ renorm).

This is the *front* of every modern MoE block (Mixtral, Qwen-MoE, DeepSeek-MoE).
Inputs:
    hidden:        (B, H)
    router_weight: (E, H)  -- E = num_experts
    topk:          int
Outputs:
    weights: (B, K)  -- normalised expert weights
    indices: (B, K)  -- chosen expert indices, int32
"""

from __future__ import annotations

import torch

from ...utils import HAS_TRITON, has_cuda, next_power_of_2

if HAS_TRITON:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore

    @triton.jit
    def _softmax_topk_kernel(
        logits_ptr, w_ptr, idx_ptr,
        n_rows, n_experts, topk,
        stride_lr, stride_wr, stride_ir,
        renorm: tl.constexpr,
        BLOCK_E: tl.constexpr,
    ):
        row = tl.program_id(0)
        if row >= n_rows:
            return
        col_offs = tl.arange(0, BLOCK_E)
        mask = col_offs < n_experts
        x = tl.load(logits_ptr + row * stride_lr + col_offs, mask=mask, other=-float("inf")).to(tl.float32)
        x = x - tl.max(x, axis=0)
        ex = tl.exp(x)
        denom = tl.sum(ex, axis=0)
        probs = ex / denom

        # Top-k via repeated argmax + mask (k is small, so this is fine).
        running = probs
        wsum = 0.0
        for k in range(0, topk):
            best_idx = tl.argmax(running, axis=0)
            best_val = tl.max(running, axis=0)
            tl.store(idx_ptr + row * stride_ir + k, best_idx.to(tl.int32))
            tl.store(w_ptr + row * stride_wr + k, best_val)
            wsum = wsum + best_val
            running = tl.where(col_offs == best_idx, -float("inf"), running)

        if renorm:
            for k in range(0, topk):
                cur = tl.load(w_ptr + row * stride_wr + k)
                tl.store(w_ptr + row * stride_wr + k, cur / wsum)


def fused_moe_topk_op(
    hidden: torch.Tensor,
    router_weight: torch.Tensor,
    topk: int,
    renormalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Step 1: router logits in fp32 — standard MoE practice; matches reference.
    logits = torch.matmul(hidden.float(), router_weight.float().t())  # (B, E)
    n_rows, n_experts = logits.shape

    if not (HAS_TRITON and has_cuda() and hidden.is_cuda):
        probs = torch.softmax(logits.float(), dim=-1)
        w, idx = torch.topk(probs, k=topk, dim=-1)
        if renormalize:
            w = w / w.sum(dim=-1, keepdim=True)
        return w.to(hidden.dtype), idx.to(torch.int32)

    block_e = next_power_of_2(n_experts)
    if block_e > 4096:
        probs = torch.softmax(logits.float(), dim=-1)
        w, idx = torch.topk(probs, k=topk, dim=-1)
        if renormalize:
            w = w / w.sum(dim=-1, keepdim=True)
        return w.to(hidden.dtype), idx.to(torch.int32)

    weights = torch.empty(n_rows, topk, device=hidden.device, dtype=torch.float32)
    indices = torch.empty(n_rows, topk, device=hidden.device, dtype=torch.int32)
    num_warps = 4 if block_e >= 1024 else 2
    _softmax_topk_kernel[(n_rows,)](
        logits.contiguous(), weights, indices,
        n_rows, n_experts, topk,
        logits.stride(0), weights.stride(0), indices.stride(0),
        renorm=renormalize,
        BLOCK_E=block_e, num_warps=num_warps,
    )
    return weights.to(hidden.dtype), indices
