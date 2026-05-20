"""Benchmark every registered operator and print a comparison table.

Usage::

    python benchmarks/run_all.py
    python benchmarks/run_all.py --tier hard
"""

from __future__ import annotations

import argparse

import torch

from flagos_track1 import OP_REGISTRY, list_ops
from flagos_track1.bench import bench_op, format_results
from flagos_track1.testing import gen_input


DEFAULT_SHAPES = {
    "abs": (4096, 4096),
    "exp": (4096, 4096),
    "log": (4096, 4096),
    "sigmoid": (4096, 4096),
    "relu": (4096, 4096),
    "tanh": (4096, 4096),
    "gelu": (4096, 4096),
    "silu": (4096, 4096),
    "softmax": (4096, 4096),
    "argmax": (4096, 4096),
    "layer_norm": (4096, 4096),
    "rms_norm": (4096, 4096),
    "cross_entropy": (4096, 32000),
    "embedding": (4096, 4096),
    "dropout": (4096, 4096),
    "matmul": (1024, 1024),
    "flash_attention": (1, 8, 1024, 64),
    "rope": (1024, 1, 128),
    "fused_moe_topk": (4096, 4096),
    "rms_norm_backward": (4096, 4096),
}


def _make_inputs(name: str):
    sh = DEFAULT_SHAPES[name]
    if name == "log":
        return (gen_input(sh, dtype=torch.float16, low=0.1, high=4.0),)
    if name == "cross_entropy":
        logits = gen_input(sh, dtype=torch.float16)
        targets = torch.randint(0, sh[-1], (sh[0],), device=logits.device)
        return logits, targets
    if name == "embedding":
        w = gen_input(sh, dtype=torch.float16)
        idx = torch.randint(0, sh[0], (4096,), device=w.device)
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["easy", "medium", "hard", "all"], default="all")
    args = ap.parse_args()
    tier = None if args.tier == "all" else args.tier
    entries = list_ops(tier)

    results = []
    for entry in entries:
        try:
            inputs = _make_inputs(entry.name)
            ours = lambda fn=entry.op, ins=inputs: fn(*ins)
            ref_fn = lambda fn=entry.reference, ins=inputs: fn(*ins)
            results.append(bench_op(entry.name, ours, ref_fn))
        except Exception as exc:  # pragma: no cover
            print(f"  [error] {entry.name}: {exc}")

    print(format_results(results))


if __name__ == "__main__":
    main()
