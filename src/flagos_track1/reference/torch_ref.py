"""Pure-PyTorch implementations used as the golden reference for every kernel.

These are deliberately the *simplest* possible expressions of each op so the
correctness tests are unambiguous. Production-fast paths live under
``flagos_track1.ops.*``.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Easy tier (8 element-wise)
# ---------------------------------------------------------------------------
def ref_abs(x: torch.Tensor) -> torch.Tensor:
    return torch.abs(x)


def ref_exp(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(x)


def ref_log(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x)


def ref_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


def ref_relu(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x)


def ref_tanh(x: torch.Tensor) -> torch.Tensor:
    return torch.tanh(x)


def ref_gelu(x: torch.Tensor, approximate: str = "none") -> torch.Tensor:
    return F.gelu(x, approximate=approximate)


def ref_silu(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x)


# ---------------------------------------------------------------------------
# Medium tier (8 classic DL)
# ---------------------------------------------------------------------------
def ref_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.softmax(x, dim=dim)


def ref_layer_norm(
    x: torch.Tensor,
    normalized_shape,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    return F.layer_norm(x, normalized_shape, weight=weight, bias=bias, eps=eps)


def ref_rms_norm(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    var = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    return (x * torch.rsqrt(var + eps).to(x.dtype)) * weight


def ref_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100
) -> torch.Tensor:
    return F.cross_entropy(logits, targets, ignore_index=ignore_index, reduction="mean")


def ref_embedding(
    indices: torch.Tensor, weight: torch.Tensor, padding_idx: int | None = None
) -> torch.Tensor:
    return F.embedding(indices, weight, padding_idx=padding_idx)


def ref_dropout(x: torch.Tensor, p: float, seed: int) -> torch.Tensor:
    g = torch.Generator(device=x.device).manual_seed(seed)
    mask = (torch.rand(x.shape, generator=g, device=x.device, dtype=torch.float32) > p).to(x.dtype)
    return x * mask / (1.0 - p)


def ref_argmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.argmax(x, dim=dim)


def ref_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.matmul(a, b)


# ---------------------------------------------------------------------------
# Hard tier (4 ops)
# ---------------------------------------------------------------------------
def ref_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """Naive O(N^2) reference matching FlashAttention-v2 semantics."""
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * sm_scale
    if causal:
        n_q, n_k = scores.shape[-2], scores.shape[-1]
        mask = torch.ones(n_q, n_k, device=scores.device, dtype=torch.bool).triu(diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v.float()).to(q.dtype)


def ref_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """RoPE applied along last dim, splitting head_dim into (even, odd)."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    rot1 = x1 * cos - x2 * sin
    rot2 = x1 * sin + x2 * cos
    out = torch.empty_like(x)
    out[..., 0::2] = rot1
    out[..., 1::2] = rot2
    return out


def ref_fused_moe_topk(
    hidden: torch.Tensor,
    router_weight: torch.Tensor,
    topk: int,
    renormalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (topk_weights[B, K], topk_indices[B, K])."""
    logits = torch.matmul(hidden.float(), router_weight.float().t())
    probs = torch.softmax(logits, dim=-1)
    weights, indices = torch.topk(probs, k=topk, dim=-1)
    if renormalize:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights.to(hidden.dtype), indices.to(torch.int32)


def ref_rms_norm_backward(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (grad_x, grad_weight) for the RMSNorm op."""
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
