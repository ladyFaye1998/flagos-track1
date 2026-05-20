"""Correctness tests for the 8 easy element-wise operators."""

from __future__ import annotations

import pytest
import torch

from flagos_track1 import OP_REGISTRY, list_ops
from flagos_track1.testing import assert_close, gen_input

EASY = [e.name for e in list_ops("easy")]
SHAPES = [(7,), (33,), (1024,), (4, 1024), (3, 5, 257)]
DTYPES = [torch.float32, torch.float16]


@pytest.mark.parametrize("name", EASY)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_easy_forward(name: str, shape, dtype):
    entry = OP_REGISTRY[name]
    if name == "log":
        x = gen_input(shape, dtype=dtype, low=0.1, high=4.0)
    else:
        x = gen_input(shape, dtype=dtype, low=-3.0, high=3.0)
    got = entry.op(x)
    want = entry.reference(x)
    assert_close(got, want, name=name, dtype=dtype)


def test_relu_zero_boundary():
    x = torch.tensor([-1.0, 0.0, 1e-6, 1.0])
    if torch.cuda.is_available():
        x = x.cuda()
    got = OP_REGISTRY["relu"].op(x)
    want = OP_REGISTRY["relu"].reference(x)
    assert_close(got, want, name="relu_boundary")


def test_gelu_tanh_approx():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required to exercise the tanh-approx kernel")
    x = gen_input((1024,), dtype=torch.float16)
    got = OP_REGISTRY["gelu"].op(x, approximate="tanh")
    want = OP_REGISTRY["gelu"].reference(x, approximate="tanh")
    assert_close(got, want, name="gelu_tanh", dtype=torch.float16)
