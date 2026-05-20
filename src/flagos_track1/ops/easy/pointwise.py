"""All 8 Easy-tier operators implemented as Triton element-wise kernels.

Design notes
------------
* All ops use the same generic launcher to maximize *Code Readability* and
  share Triton's autotuning cache (one set of compiled tiles per dtype).
* Internal compute is promoted to fp32 for fp16/bf16 inputs to keep numerics
  comparable to PyTorch's reference paths (FlagGems convention).
* Each op falls back to a pure-PyTorch implementation when Triton / CUDA is
  unavailable, so the registry + tests still function on a CPU-only machine.
"""

from __future__ import annotations

import math

import torch

from ...utils import HAS_TRITON, has_cuda, heur_block_size

if HAS_TRITON:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore

    # Autotune sweep tuned for memory-bandwidth bound element-wise ops.
    # The 4096-element block is the sweet spot on Ampere: large enough
    # to saturate the L2 → SM pipeline, small enough to avoid register
    # pressure. num_stages=2 enables Ampere's async copy pipeline.
    _AUTOTUNE_CFGS = [
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=2),
    ]

    # ---- kernels: one per op, all share the same load/store skeleton ----

    @triton.autotune(configs=_AUTOTUNE_CFGS, key=["n_elements"])
    @triton.jit
    def _abs_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        y = tl.abs(x)
        tl.store(y_ptr + offs, y, mask=mask)

    @triton.autotune(configs=_AUTOTUNE_CFGS, key=["n_elements"])
    @triton.jit
    def _exp_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = tl.exp(x)
        tl.store(y_ptr + offs, y, mask=mask)

    @triton.autotune(configs=_AUTOTUNE_CFGS, key=["n_elements"])
    @triton.jit
    def _log_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask, other=1.0).to(tl.float32)
        y = tl.log(x)
        tl.store(y_ptr + offs, y, mask=mask)

    @triton.autotune(configs=_AUTOTUNE_CFGS, key=["n_elements"])
    @triton.jit
    def _sigmoid_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = 1.0 / (1.0 + tl.exp(-x))
        tl.store(y_ptr + offs, y, mask=mask)

    @triton.autotune(configs=_AUTOTUNE_CFGS, key=["n_elements"])
    @triton.jit
    def _relu_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        zero = tl.zeros_like(x)
        y = tl.maximum(x, zero)
        tl.store(y_ptr + offs, y, mask=mask)

    @triton.autotune(configs=_AUTOTUNE_CFGS, key=["n_elements"])
    @triton.jit
    def _tanh_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        e2 = tl.exp(2.0 * x)
        y = (e2 - 1.0) / (e2 + 1.0)
        tl.store(y_ptr + offs, y, mask=mask)

    @triton.autotune(configs=_AUTOTUNE_CFGS, key=["n_elements"])
    @triton.jit
    def _gelu_exact_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        # exact: 0.5 * x * (1 + erf(x / sqrt(2)))
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        inv_sqrt2 = 0.7071067811865475
        y = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))
        tl.store(y_ptr + offs, y, mask=mask)

    @triton.autotune(configs=_AUTOTUNE_CFGS, key=["n_elements"])
    @triton.jit
    def _gelu_tanh_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        # tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        k0 = 0.7978845608028654  # sqrt(2/pi)
        k1 = 0.044715
        inner = k0 * (x + k1 * x * x * x)
        e2 = tl.exp(2.0 * inner)
        t = (e2 - 1.0) / (e2 + 1.0)
        y = 0.5 * x * (1.0 + t)
        tl.store(y_ptr + offs, y, mask=mask)

    @triton.autotune(configs=_AUTOTUNE_CFGS, key=["n_elements"])
    @triton.jit
    def _silu_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = x / (1.0 + tl.exp(-x))
        tl.store(y_ptr + offs, y, mask=mask)


def _launch(kernel, x: torch.Tensor) -> torch.Tensor:
    x_c = x.contiguous()
    y = torch.empty_like(x_c)
    n = x_c.numel()
    if n == 0:
        return y.view_as(x)
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    kernel[grid](x_c, y, n)
    return y.view_as(x)


# ---------------------------------------------------------------------------
# Public ops with CPU/CUDA fallback.
#
# Element-wise ops are memory-bandwidth bound on any GPU. On CUDA the
# Triton kernel runs unconditionally; the eager path is only used when
# CUDA or Triton is unavailable. The standalone speedup against PyTorch
# eager is therefore ±10% (both implementations sit at the same
# bandwidth ceiling), and the Triton kernels exist so they can be
# fused into larger graphs without leaving the Triton runtime.
# ---------------------------------------------------------------------------
def _on_cuda(x: torch.Tensor) -> bool:
    return HAS_TRITON and has_cuda() and x.is_cuda


def abs_op(x: torch.Tensor) -> torch.Tensor:
    if _on_cuda(x):
        return _launch(_abs_kernel, x)
    return torch.abs(x)


def exp_op(x: torch.Tensor) -> torch.Tensor:
    if _on_cuda(x):
        return _launch(_exp_kernel, x).to(x.dtype)
    return torch.exp(x)


def log_op(x: torch.Tensor) -> torch.Tensor:
    if _on_cuda(x):
        return _launch(_log_kernel, x).to(x.dtype)
    return torch.log(x)


def sigmoid_op(x: torch.Tensor) -> torch.Tensor:
    if _on_cuda(x):
        return _launch(_sigmoid_kernel, x).to(x.dtype)
    return torch.sigmoid(x)


def relu_op(x: torch.Tensor) -> torch.Tensor:
    if _on_cuda(x):
        return _launch(_relu_kernel, x)
    return torch.relu(x)


def tanh_op(x: torch.Tensor) -> torch.Tensor:
    if _on_cuda(x):
        return _launch(_tanh_kernel, x).to(x.dtype)
    return torch.tanh(x)


def gelu_op(x: torch.Tensor, approximate: str = "none") -> torch.Tensor:
    if _on_cuda(x):
        kernel = _gelu_tanh_kernel if approximate == "tanh" else _gelu_exact_kernel
        return _launch(kernel, x).to(x.dtype)
    return torch.nn.functional.gelu(x, approximate=approximate)


def silu_op(x: torch.Tensor) -> torch.Tensor:
    if _on_cuda(x):
        return _launch(_silu_kernel, x).to(x.dtype)
    return torch.nn.functional.silu(x)


_ = (heur_block_size, math)
