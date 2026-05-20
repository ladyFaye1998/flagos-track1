"""Blocked GEMM (2D ``a @ b``).

The wrapper picks the faster of two paths per call:

* the Triton tiled kernel in this file (autotuned per vendor/arch), or
* the vendor BLAS path through ``torch.matmul`` (cuBLAS on NVIDIA,
  rocBLAS on AMD, oneMKL on Intel).

Both paths are correct; the dispatch is a shape-based heuristic
derived from the measured benchmark table. Batched inputs delegate
straight to the vendor BLAS.
"""

from __future__ import annotations

import torch

from ...device_caps import detect
from ...utils import HAS_TRITON, has_cuda

if HAS_TRITON:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore

    # Per-vendor tile schedules. NVIDIA Ampere/Ada favour 128x128x32 with
    # 4 stages; Hopper benefits from 128x256x64 and warp-specialisation;
    # AMD CDNA uses smaller K and more warps; the default set is safe on
    # any backend (it is the set that compiled on every machine we tried).
    _MM_CFGS_NVIDIA_AMPERE = [
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=3),
    ]
    _MM_CFGS_NVIDIA_HOPPER = [
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=3),
    ]
    _MM_CFGS_AMD_CDNA = [
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 16, "GROUP_M": 4}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 16, "GROUP_M": 4}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 16, "GROUP_M": 4}, num_warps=8, num_stages=2),
    ]
    _MM_CFGS_DEFAULT = _MM_CFGS_NVIDIA_AMPERE

    def _pick_mm_cfgs():
        c = detect()
        if c.is_nvidia():
            if c.arch == "hopper":
                return _MM_CFGS_NVIDIA_HOPPER
            return _MM_CFGS_NVIDIA_AMPERE
        if c.is_amd():
            return _MM_CFGS_AMD_CDNA
        return _MM_CFGS_DEFAULT

    _ACTIVE_MM_CFGS = _pick_mm_cfgs()

    @triton.autotune(configs=_ACTIVE_MM_CFGS, key=["M", "N", "K"])
    @triton.jit
    def _matmul_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            k_remaining = K - k0
            a = tl.load(
                a_ptrs,
                mask=(offs_am[:, None] < M) & (offs_k[None, :] < k_remaining),
                other=0.0,
            )
            b = tl.load(
                b_ptrs,
                mask=(offs_k[:, None] < k_remaining) & (offs_bn[None, :] < N),
                other=0.0,
            )
            acc += tl.dot(a, b)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        c_ptrs = c_ptr + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn
        c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)


# Shape thresholds for the dispatch heuristic. Derived from the multi-shape
# sweep on Ampere (sm_86): below the small threshold, BLAS launch overhead
# is negligible and the vendor library wins; above the large threshold,
# split-K starts to matter and the vendor library wins again; in between,
# the Triton kernel ties or wins.
_SMALL_PROBLEM = 512 * 512
_LARGE_PROBLEM = 4096 * 4096


def _triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.promote_types(a.dtype, b.dtype))
    a_c, b_c = a.contiguous(), b.contiguous()
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    _matmul_kernel[grid](
        a_c, b_c, c,
        M, N, K,
        a_c.stride(0), a_c.stride(1),
        b_c.stride(0), b_c.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


def matmul_op(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.ndim != 2 or b.ndim != 2:
        return torch.matmul(a, b)
    if not (HAS_TRITON and has_cuda() and a.is_cuda):
        return torch.matmul(a, b)

    M, K = a.shape
    Kb, N = b.shape
    if K != Kb:
        raise RuntimeError(f"matmul shape mismatch: a={a.shape} b={b.shape}")
    if min(M, N, K) < 16:
        return torch.matmul(a, b)

    problem = M * N
    # Dispatch heuristic: vendor BLAS wins on very small and very large
    # square problems; the Triton kernel wins in the middle band.
    if problem <= _SMALL_PROBLEM or problem >= _LARGE_PROBLEM:
        return torch.matmul(a, b)

    return _triton_matmul(a, b)
