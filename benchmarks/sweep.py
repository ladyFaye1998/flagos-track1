"""Multi-shape benchmark sweep across small/medium/large per operator.

Produces a per-operator table with one row per shape and one summary table
with geometric-mean speedup per operator across shapes. Output is appended
to BENCHMARKS.md by `benchmarks/save_results.py`.
"""

from __future__ import annotations

import argparse
import math
from typing import Callable

import torch

from flagos_track1 import OP_REGISTRY, list_ops
from flagos_track1.bench import bench_op, format_results
from flagos_track1.testing import gen_input


SHAPE_SWEEPS: dict[str, list[tuple]] = {
    # ── easy element-wise: pure memory-bound, scale rows × cols ──
    "abs":     [(1024, 1024), (4096, 4096), (8192, 4096)],
    "exp":     [(1024, 1024), (4096, 4096), (8192, 4096)],
    "log":     [(1024, 1024), (4096, 4096), (8192, 4096)],
    "sigmoid": [(1024, 1024), (4096, 4096), (8192, 4096)],
    "relu":    [(1024, 1024), (4096, 4096), (8192, 4096)],
    "tanh":    [(1024, 1024), (4096, 4096), (8192, 4096)],
    "gelu":    [(1024, 1024), (4096, 4096), (8192, 4096)],
    "silu":    [(1024, 1024), (4096, 4096), (8192, 4096)],
    # ── medium: row-reductions / normalizations / matmul ──
    "softmax":       [(512, 1024), (4096, 4096), (4096, 16384)],
    "argmax":        [(512, 1024), (4096, 4096), (4096, 16384)],
    "layer_norm":    [(512, 1024), (4096, 4096), (4096, 8192)],
    "rms_norm":      [(512, 1024), (4096, 4096), (4096, 8192)],
    "cross_entropy": [(512, 1024), (4096, 32000), (8192, 32000)],
    "embedding":     [(1024, 1024), (4096, 4096), (32000, 4096)],
    "dropout":       [(1024, 1024), (4096, 4096), (8192, 4096)],
    "matmul":        [(256, 256), (1024, 1024), (2048, 2048)],
    # ── hard: attention, rope, moe, backward ──
    "flash_attention":  [(1, 8, 256, 64), (1, 8, 1024, 64), (2, 8, 2048, 64)],
    "rope":             [(256, 1, 64), (1024, 1, 128), (4096, 1, 128)],
    "fused_moe_topk":   [(512, 1024), (4096, 4096), (8192, 4096)],
    "rms_norm_backward":[(512, 1024), (4096, 4096), (4096, 8192)],
}


def _make_inputs(name: str, sh: tuple):
    if name == "log":
        return (gen_input(sh, dtype=torch.float16, low=0.1, high=4.0),)
    if name == "cross_entropy":
        logits = gen_input(sh, dtype=torch.float16)
        targets = torch.randint(0, sh[-1], (sh[0],), device=logits.device)
        return logits, targets
    if name == "embedding":
        w = gen_input(sh, dtype=torch.float16)
        idx = torch.randint(0, sh[0], (min(sh[0], 4096),), device=w.device)
        return idx, w
    if name == "dropout":
        return (gen_input(sh, dtype=torch.float16), 0.1, 42)
    if name == "matmul":
        a = gen_input(sh, dtype=torch.float16)
        b = gen_input(sh, dtype=torch.float16, seed=1)
        return a, b
    if name == "flash_attention":
        q = gen_input(sh, dtype=torch.float16)
        k = gen_input(sh, dtype=torch.float16, seed=1)
        v = gen_input(sh, dtype=torch.float16, seed=2)
        return q, k, v
    if name == "rope":
        B, N, D = sh
        x = gen_input(sh, dtype=torch.float16)
        cos = gen_input((B * N, D // 2), dtype=torch.float16, seed=1).view(B, N, D // 2)
        sin = gen_input((B * N, D // 2), dtype=torch.float16, seed=2).view(B, N, D // 2)
        return x, cos, sin
    if name == "fused_moe_topk":
        B, H = sh
        hidden = gen_input((B, H), dtype=torch.float16)
        router = gen_input((64, H), dtype=torch.float16, seed=1)
        return hidden, router, 4
    if name == "layer_norm":
        x = gen_input(sh, dtype=torch.float16)
        w = torch.ones(sh[-1], device=x.device, dtype=x.dtype)
        b = torch.zeros(sh[-1], device=x.device, dtype=x.dtype)
        return x, (sh[-1],), w, b
    if name == "rms_norm":
        x = gen_input(sh, dtype=torch.float16)
        w = torch.ones(sh[-1], device=x.device, dtype=x.dtype)
        return x, w
    if name == "rms_norm_backward":
        x = gen_input(sh, dtype=torch.float32)
        w = torch.ones(sh[-1], device=x.device, dtype=x.dtype)
        g = gen_input(sh, dtype=torch.float32, seed=1)
        return g, x, w
    return (gen_input(sh, dtype=torch.float16),)


def _geomean(xs: list[float]) -> float:
    return math.exp(sum(math.log(max(x, 1e-9)) for x in xs) / len(xs)) if xs else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["easy", "medium", "hard", "all"], default="all")
    ap.add_argument("--markdown", action="store_true", help="emit a github-flavored markdown block")
    args = ap.parse_args()

    tier = None if args.tier == "all" else args.tier
    entries = list_ops(tier)

    per_op: dict[str, list[tuple[tuple, float, float, float]]] = {}
    for entry in entries:
        shapes = SHAPE_SWEEPS.get(entry.name)
        if not shapes:
            continue
        per_op[entry.name] = []
        for sh in shapes:
            try:
                inputs = _make_inputs(entry.name, sh)
                ours = lambda fn=entry.op, ins=inputs: fn(*ins)
                ref = lambda fn=entry.reference, ins=inputs: fn(*ins)
                r = bench_op(entry.name, ours, ref)
                per_op[entry.name].append((sh, r.ours_ms, r.ref_ms, r.speedup))
            except Exception as exc:  # pragma: no cover
                print(f"  [skip] {entry.name} @ {sh}: {exc}")

    if args.markdown:
        rows = []
        for name, runs in per_op.items():
            gm = _geomean([s for _, _, _, s in runs])
            for sh, ours_ms, ref_ms, sp in runs:
                rows.append([name, str(sh), f"{ours_ms:.4f}", f"{ref_ms:.4f}", f"{sp:.2f}x"])
            rows.append([f"**{name} (geomean)**", "-", "-", "-", f"**{gm:.2f}x**"])
        try:
            from tabulate import tabulate
            print(tabulate(rows, headers=["Operator", "Shape", "Ours (ms)", "PyTorch (ms)", "Speedup"], tablefmt="github"))
        except Exception:
            for r in rows:
                print("  ".join(r))
    else:
        for name, runs in per_op.items():
            gm = _geomean([s for _, _, _, s in runs])
            print(f"\n{name}  (geomean {gm:.2f}x)")
            for sh, ours_ms, ref_ms, sp in runs:
                print(f"  {sh!s:24s}  ours={ours_ms:8.4f}ms  ref={ref_ms:8.4f}ms  speedup={sp:5.2f}x")


if __name__ == "__main__":
    main()
