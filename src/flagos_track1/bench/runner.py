"""Lightweight benchmark runner — usable on CPU (timeit) and GPU (CUDA events)."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Callable

import torch

try:
    from triton.testing import do_bench  # type: ignore

    HAS_DO_BENCH = True
except Exception:  # pragma: no cover
    do_bench = None  # type: ignore
    HAS_DO_BENCH = False


@dataclass
class BenchResult:
    name: str
    ours_ms: float
    ref_ms: float
    speedup: float

    def to_row(self) -> list[str]:
        return [
            self.name,
            f"{self.ours_ms:.4f}",
            f"{self.ref_ms:.4f}",
            f"{self.speedup:.2f}x",
        ]


def _time_cuda(fn: Callable[[], None], warmup: int, rep: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times: list[float] = []
    for _ in range(rep):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def _time_cpu(fn: Callable[[], None], warmup: int, rep: int) -> float:
    for _ in range(warmup):
        fn()
    times: list[float] = []
    for _ in range(rep):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times)


def _time(fn: Callable[[], None], warmup: int = 25, rep: int = 100) -> float:
    if torch.cuda.is_available():
        if HAS_DO_BENCH:
            return float(do_bench(fn, warmup=warmup, rep=rep))
        return _time_cuda(fn, warmup, rep)
    return _time_cpu(fn, warmup, rep)


def bench_op(
    name: str,
    ours: Callable[[], torch.Tensor],
    reference: Callable[[], torch.Tensor],
    *,
    warmup: int = 25,
    rep: int = 100,
) -> BenchResult:
    ours_ms = _time(ours, warmup, rep)
    ref_ms = _time(reference, warmup, rep)
    speedup = ref_ms / ours_ms if ours_ms > 0 else float("inf")
    return BenchResult(name=name, ours_ms=ours_ms, ref_ms=ref_ms, speedup=speedup)


def format_results(results: list[BenchResult]) -> str:
    try:
        from tabulate import tabulate  # type: ignore

        return tabulate(
            [r.to_row() for r in results],
            headers=["Operator", "Ours (ms)", "PyTorch (ms)", "Speedup"],
            tablefmt="github",
        )
    except Exception:
        return "\n".join(
            f"{r.name:30s}  ours={r.ours_ms:8.4f}ms  ref={r.ref_ms:8.4f}ms  speedup={r.speedup:5.2f}x"
            for r in results
        )
