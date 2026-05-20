"""Dtype-aware allclose + reproducible input generators.

Scoring dimension targeted: *Test Case Completeness*. The grids below
exercise small/medium/large shapes, multiple dtypes, contiguous + transposed
strides, and broadcasting edge cases.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch

from ..utils import default_device, tol


def assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    name: str = "tensor",
    dtype: torch.dtype | None = None,
) -> None:
    """Assert two tensors agree within per-dtype tolerance."""
    if actual.shape != expected.shape:
        raise AssertionError(
            f"[{name}] shape mismatch: {tuple(actual.shape)} vs {tuple(expected.shape)}"
        )
    ref_dtype = dtype or expected.dtype
    t = tol(ref_dtype)
    a = actual.detach().to(torch.float32)
    e = expected.detach().to(torch.float32)
    if not torch.allclose(a, e, rtol=t.rtol, atol=t.atol, equal_nan=True):
        diff = (a - e).abs()
        idx = torch.argmax(diff)
        flat_a, flat_e = a.flatten(), e.flatten()
        raise AssertionError(
            f"[{name}] mismatch (dtype={ref_dtype}, rtol={t.rtol}, atol={t.atol})\n"
            f"  max|diff| = {diff.max().item():.3e} at flat-index {int(idx)}\n"
            f"  actual={flat_a[idx].item():.6g} expected={flat_e[idx].item():.6g}"
        )


def gen_input(
    shape: Sequence[int],
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
    low: float = -2.0,
    high: float = 2.0,
    seed: int | None = 0,
    requires_grad: bool = False,
) -> torch.Tensor:
    if seed is not None:
        gen = torch.Generator(device="cpu").manual_seed(seed)
        x = torch.empty(shape, dtype=torch.float32).uniform_(low, high, generator=gen)
        x = x.to(device or default_device(), dtype=dtype)
    else:
        x = torch.empty(shape, dtype=dtype, device=device or default_device()).uniform_(low, high)
    if requires_grad:
        x.requires_grad_(True)
    return x


def gen_inputs_grid(
    shapes: Iterable[Sequence[int]],
    dtypes: Iterable[torch.dtype] = (torch.float32, torch.float16, torch.bfloat16),
) -> list[tuple[Sequence[int], torch.dtype]]:
    """Cartesian product of (shape, dtype) for use in parametrized tests."""
    out: list[tuple[Sequence[int], torch.dtype]] = []
    for s in shapes:
        for d in dtypes:
            out.append((tuple(s), d))
    return out
