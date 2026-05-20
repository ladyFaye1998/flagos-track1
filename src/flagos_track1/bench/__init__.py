"""Benchmark utilities (wraps :func:`triton.testing.do_bench` when available)."""

from .runner import BenchResult, bench_op, format_results  # noqa: F401
