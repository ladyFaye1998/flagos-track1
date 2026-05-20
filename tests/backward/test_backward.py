"""Correctness tests for the 5 backward kernels.

Covers softmax_backward, layer_norm_backward, cross_entropy_backward,
silu_backward and gelu_backward (exact + tanh approx). The reference
gradient comes from torch.autograd, which gives end-to-end coverage
of both the math and the autodiff plumbing.
"""

from __future__ import annotations

import pytest
import torch

from flagos_track1.ops.backward import (
    cross_entropy_backward_op,
    gelu_backward_op,
    layer_norm_backward_op,
    silu_backward_op,
    softmax_backward_op,
)
from flagos_track1.testing import assert_close, gen_input


SHAPES_2D = [(8, 64), (4, 257), (16, 1024)]
DTYPES = [torch.float32, torch.float16]


@pytest.mark.parametrize("shape", SHAPES_2D)
@pytest.mark.parametrize("dtype", DTYPES)
def test_softmax_backward(shape, dtype):
    x = gen_input(shape, dtype=dtype).requires_grad_(True)
    y = torch.softmax(x, dim=-1)
    grad_y = gen_input(shape, dtype=dtype, seed=1)
    y.backward(grad_y)
    ref_grad_x = x.grad.detach()
    got = softmax_backward_op(grad_y, y.detach())
    assert_close(got, ref_grad_x, name=f"softmax_bwd[{shape},{dtype}]", dtype=dtype)


@pytest.mark.parametrize("shape", SHAPES_2D)
@pytest.mark.parametrize("affine", [True, False])
def test_layer_norm_backward(shape, affine):
    n_cols = shape[-1]
    x = gen_input(shape, dtype=torch.float32).requires_grad_(True)
    w = torch.ones(n_cols, device=x.device, dtype=x.dtype, requires_grad=True) if affine else None
    b = torch.zeros(n_cols, device=x.device, dtype=x.dtype, requires_grad=True) if affine else None
    y = torch.nn.functional.layer_norm(x, (n_cols,), weight=w, bias=b, eps=1e-5)
    grad_y = gen_input(shape, dtype=torch.float32, seed=2)
    y.backward(grad_y)
    ref_gx = x.grad.detach()
    ref_gw = w.grad.detach() if affine else None
    ref_gb = b.grad.detach() if affine else None
    gx, gw, gb = layer_norm_backward_op(grad_y, x.detach(), (n_cols,), w, b)
    assert_close(gx, ref_gx, name=f"ln_bwd_x[{shape}]", dtype=torch.float32)
    if affine:
        # Atomic-add accumulation over N rows introduces order-dependent
        # float32 rounding (~ulp * N magnitude). Math is identical.
        torch.testing.assert_close(gw, ref_gw, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(gb, ref_gb, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("n_rows,n_classes", [(8, 16), (32, 1024), (4, 257)])
def test_cross_entropy_backward(n_rows, n_classes):
    logits = gen_input((n_rows, n_classes), dtype=torch.float32).requires_grad_(True)
    targets = torch.randint(0, n_classes, (n_rows,), device=logits.device)
    loss = torch.nn.functional.cross_entropy(logits, targets, reduction="mean")
    grad_loss = torch.tensor(1.0, device=logits.device, dtype=torch.float32)
    loss.backward(grad_loss)
    ref = logits.grad.detach()
    got = cross_entropy_backward_op(grad_loss, logits.detach(), targets)
    assert_close(got, ref, name=f"ce_bwd[{n_rows},{n_classes}]", dtype=torch.float32)


@pytest.mark.parametrize("shape", [(1024,), (4, 1024), (3, 17, 5)])
@pytest.mark.parametrize("dtype", DTYPES)
def test_silu_backward(shape, dtype):
    x = gen_input(shape, dtype=dtype).requires_grad_(True)
    y = torch.nn.functional.silu(x)
    grad_y = gen_input(shape, dtype=dtype, seed=1)
    y.backward(grad_y)
    ref = x.grad.detach()
    got = silu_backward_op(grad_y, x.detach())
    assert_close(got, ref, name=f"silu_bwd[{shape},{dtype}]", dtype=dtype)


@pytest.mark.parametrize("shape", [(1024,), (4, 1024)])
@pytest.mark.parametrize("approximate", ["none", "tanh"])
@pytest.mark.parametrize("dtype", DTYPES)
def test_gelu_backward(shape, approximate, dtype):
    x = gen_input(shape, dtype=dtype).requires_grad_(True)
    y = torch.nn.functional.gelu(x, approximate=approximate)
    grad_y = gen_input(shape, dtype=dtype, seed=1)
    y.backward(grad_y)
    ref = x.grad.detach()
    got = gelu_backward_op(grad_y, x.detach(), approximate=approximate)
    assert_close(got, ref, name=f"gelu_bwd[{approximate},{shape},{dtype}]", dtype=dtype)
