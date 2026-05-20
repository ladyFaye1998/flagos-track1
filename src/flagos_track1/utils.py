"""Shared helpers: device detection, autotune configs, dtype-aware tolerances."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import torch

try:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore

    HAS_TRITON = True
except Exception:  # pragma: no cover - import-time guard
    triton = None  # type: ignore
    tl = None  # type: ignore
    HAS_TRITON = False


def has_cuda() -> bool:
    return torch.cuda.is_available()


def default_device() -> torch.device:
    return torch.device("cuda" if has_cuda() else "cpu")


def require_triton() -> None:
    if not HAS_TRITON:
        raise RuntimeError(
            "Triton is not installed. Install with `pip install triton` "
            "(or `triton-windows` on Windows)."
        )
    if not has_cuda():
        raise RuntimeError("A CUDA-capable GPU is required to execute Triton kernels.")


@dataclass(frozen=True)
class Tolerance:
    rtol: float
    atol: float


# Per-dtype tolerances tuned to match FlagGems CI defaults.
TOLERANCE = {
    torch.float32: Tolerance(rtol=1e-5, atol=1e-6),
    torch.float16: Tolerance(rtol=1e-3, atol=1e-3),
    torch.bfloat16: Tolerance(rtol=1e-2, atol=1e-2),
}


def tol(dtype: torch.dtype) -> Tolerance:
    return TOLERANCE.get(dtype, Tolerance(rtol=1e-3, atol=1e-3))


def next_power_of_2(n: int) -> int:
    """Smallest power of two >= n (Triton BLOCK_SIZE requirement)."""
    return 1 << max(0, (n - 1)).bit_length()


def heur_block_size(n: int, *, cap: int = 4096) -> int:
    return min(cap, max(64, next_power_of_2(n)))


def common_autotune_configs(
    block_sizes: Iterable[int] = (256, 512, 1024, 2048, 4096),
    num_warps: Iterable[int] = (2, 4, 8),
):
    """Build a list of triton.Config objects for pointwise/reduction kernels."""
    require_triton()
    return [
        triton.Config({"BLOCK_SIZE": bs}, num_warps=nw)
        for bs in block_sizes
        for nw in num_warps
    ]


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
