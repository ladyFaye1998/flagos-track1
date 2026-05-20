"""CPU-fallback parity tests.

Every op in the registry must produce numerically identical output to
its PyTorch reference when called on a CPU tensor. This is what
allows the same code to ship to a GPU-less CI runner and to any
Triton-supported backend (NVIDIA / AMD / Intel) without forking the
implementation.

We don't require Triton or CUDA here — the wrappers are expected to
detect their absence and route through the reference path.
"""

from __future__ import annotations

import pytest
import torch

from flagos_track1 import OP_REGISTRY
from flagos_track1.device_caps import detect


def _gen(shape, dtype=torch.float32, low=-1.0, high=1.0, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.empty(shape, dtype=dtype, device="cpu").uniform_(low, high, generator=g)


# Small CPU-friendly shapes; we are validating correctness, not speed.
_CASES: dict[str, callable] = {
    # element-wise
    "abs":     lambda: (_gen((4, 16)),),
    "exp":     lambda: (_gen((4, 16), low=-2, high=2),),
    "log":     lambda: (_gen((4, 16), low=0.1, high=4.0),),
    "sigmoid": lambda: (_gen((4, 16)),),
    "relu":    lambda: (_gen((4, 16)),),
    "tanh":    lambda: (_gen((4, 16)),),
    "gelu":    lambda: (_gen((4, 16)),),
    "silu":    lambda: (_gen((4, 16)),),
    # medium
    "softmax":    lambda: (_gen((4, 16)),),
    "argmax":     lambda: (_gen((4, 16)),),
    "layer_norm": lambda: (_gen((4, 16)), (16,), torch.ones(16), torch.zeros(16)),
    "rms_norm":   lambda: (_gen((4, 16)), torch.ones(16)),
    "embedding":  lambda: (
        torch.randint(0, 32, (8,)),
        _gen((32, 16), seed=1),
    ),
    "dropout":    lambda: (_gen((4, 16)), 0.0, 7),  # p=0 makes it deterministic
    "matmul":     lambda: (_gen((8, 16)), _gen((16, 8), seed=1)),
    "cross_entropy": lambda: (
        _gen((4, 8), seed=2),
        torch.randint(0, 8, (4,)),
    ),
    # hard
    "flash_attention": lambda: (
        _gen((1, 2, 8, 64), dtype=torch.float32),
        _gen((1, 2, 8, 64), dtype=torch.float32, seed=1),
        _gen((1, 2, 8, 64), dtype=torch.float32, seed=2),
    ),
    "rope": lambda: (
        _gen((2, 4, 16)),
        _gen((2, 4, 8), seed=1),
        _gen((2, 4, 8), seed=2),
    ),
    "fused_moe_topk": lambda: (
        _gen((4, 16)),
        _gen((8, 16), seed=1),
        2,
    ),
    "rms_norm_backward": lambda: (
        _gen((4, 16), seed=2),  # grad
        _gen((4, 16)),
        torch.ones(16),
    ),
}


@pytest.mark.parametrize("name", sorted(_CASES.keys()))
def test_cpu_parity(name: str):
    """Op wrapper on CPU == its PyTorch reference."""
    entry = OP_REGISTRY[name]
    inputs = _CASES[name]()
    got = entry.op(*inputs)
    want = entry.reference(*inputs)

    if isinstance(got, tuple):
        assert isinstance(want, tuple) and len(got) == len(want)
        for g, w in zip(got, want):
            if g is None and w is None:
                continue
            torch.testing.assert_close(g, w, rtol=1e-4, atol=1e-4)
    else:
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)


def test_device_caps_detect_cpu():
    """On a machine without CUDA, detect() must return a cpu DeviceCaps."""
    caps = detect()
    assert caps.vendor in {"nvidia", "amd", "intel", "cpu", "unknown"}
    assert caps.cuda_available == torch.cuda.is_available()
    if not torch.cuda.is_available():
        assert caps.vendor == "cpu"
        assert caps.arch == "cpu"
