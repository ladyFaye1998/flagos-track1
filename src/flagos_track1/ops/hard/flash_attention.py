"""FlashAttention-v2 forward pass (causal, single-batch tile).

Reference: Dao 2023 (https://arxiv.org/abs/2307.08691). This kernel uses the
online-softmax recurrence to compute attention with O(N) memory for the
softmax statistics. It targets the common LLM shape:
``(batch, heads, seq_len, head_dim)`` with ``head_dim in {64, 128}``.
"""

from __future__ import annotations

import math

import torch

from ...utils import HAS_TRITON, has_cuda

if HAS_TRITON:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore

    @triton.jit
    def _attn_fwd_kernel(
        Q, K, V, Out,
        sm_scale,
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vn, stride_vk,
        stride_oz, stride_oh, stride_om, stride_ok,
        Z, H, N_CTX,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr, CAUSAL: tl.constexpr,
    ):
        start_m = tl.program_id(0)
        off_hz = tl.program_id(1)
        off_z = off_hz // H
        off_h = off_hz % H

        q_offset = off_z * stride_qz + off_h * stride_qh
        k_offset = off_z * stride_kz + off_h * stride_kh
        v_offset = off_z * stride_vz + off_h * stride_vh
        o_offset = off_z * stride_oz + off_h * stride_oh

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_DMODEL)

        q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
        q_mask = offs_m[:, None] < N_CTX
        q = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float32)

        m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)

        end_n = (start_m + 1) * BLOCK_M if CAUSAL else N_CTX
        for start_n in range(0, end_n, BLOCK_N):
            n_offs = start_n + offs_n
            k_ptrs = K + k_offset + n_offs[:, None] * stride_kn + offs_d[None, :] * stride_kk
            v_ptrs = V + v_offset + n_offs[:, None] * stride_vn + offs_d[None, :] * stride_vk
            kv_mask = n_offs[:, None] < N_CTX
            k = tl.load(k_ptrs, mask=kv_mask, other=0.0).to(tl.float32)
            v = tl.load(v_ptrs, mask=kv_mask, other=0.0).to(tl.float32)

            qk = tl.dot(q, tl.trans(k)) * sm_scale  # (M, N)
            if CAUSAL:
                causal_mask = offs_m[:, None] >= n_offs[None, :]
                qk = tl.where(causal_mask, qk, float("-inf"))
            qk = tl.where(n_offs[None, :] < N_CTX, qk, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(qk, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new

        acc = acc / l_i[:, None]
        o_ptrs = Out + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
        tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=q_mask)


def flash_attention_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """Forward pass. Expects (B, H, N, D) with D in {16, 32, 64, 128}."""
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])

    head_dim = q.shape[-1]
    supported = head_dim in (16, 32, 64, 128)
    if not (HAS_TRITON and has_cuda() and q.is_cuda and supported):
        # Fallback to reference impl
        scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * sm_scale
        if causal:
            n_q, n_k = scores.shape[-2], scores.shape[-1]
            mask = torch.ones(n_q, n_k, device=scores.device, dtype=torch.bool).triu(diagonal=1)
            scores = scores.masked_fill(mask, float("-inf"))
        return torch.matmul(torch.softmax(scores, dim=-1), v.float()).to(q.dtype)

    B, H, N, D = q.shape
    o = torch.empty_like(q)
    BLOCK_M = 64
    BLOCK_N = 64
    grid = ((N + BLOCK_M - 1) // BLOCK_M, B * H)
    _attn_fwd_kernel[grid](
        q, k, v, o,
        float(sm_scale),
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        B, H, N,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=D, CAUSAL=causal,
        num_warps=4, num_stages=2,
    )
    return o
