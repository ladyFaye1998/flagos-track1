"""Correctness tests for the 8 medium-tier kernels."""

from __future__ import annotations

import pytest
import torch

from flagos_track1 import OP_REGISTRY
from flagos_track1.testing import assert_close, gen_input

DTYPES = [torch.float32, torch.float16]


# ----- softmax / argmax (last-dim reductions) -----
@pytest.mark.parametrize("shape", [(16, 32), (4, 1, 257), (1024,)])
@pytest.mark.parametrize("dtype", DTYPES)
def test_softmax(shape, dtype):
    x = gen_input(shape, dtype=dtype)
    got = OP_REGISTRY["softmax"].op(x)
    want = OP_REGISTRY["softmax"].reference(x)
    assert_close(got, want, name="softmax", dtype=dtype)


@pytest.mark.parametrize("shape", [(16, 32), (1024,), (3, 5, 17)])
@pytest.mark.parametrize("dtype", DTYPES)
def test_argmax(shape, dtype):
    x = gen_input(shape, dtype=dtype)
    got = OP_REGISTRY["argmax"].op(x)
    want = OP_REGISTRY["argmax"].reference(x)
    assert torch.equal(got, want), f"argmax mismatch on shape={shape}"


# ----- LayerNorm + RMSNorm -----
@pytest.mark.parametrize("shape", [(8, 64), (4, 1, 128), (2, 5, 257)])
@pytest.mark.parametrize("dtype", DTYPES)
def test_layer_norm(shape, dtype):
    x = gen_input(shape, dtype=dtype)
    w = torch.ones(shape[-1], device=x.device, dtype=dtype)
    b = torch.zeros(shape[-1], device=x.device, dtype=dtype)
    got = OP_REGISTRY["layer_norm"].op(x, (shape[-1],), w, b)
    want = OP_REGISTRY["layer_norm"].reference(x, (shape[-1],), w, b)
    assert_close(got, want, name="layer_norm", dtype=dtype)


@pytest.mark.parametrize("shape", [(8, 64), (4, 1, 128), (2, 5, 257)])
@pytest.mark.parametrize("dtype", DTYPES)
def test_rms_norm(shape, dtype):
    x = gen_input(shape, dtype=dtype)
    w = gen_input((shape[-1],), dtype=dtype, low=0.5, high=1.5, seed=11)
    got = OP_REGISTRY["rms_norm"].op(x, w)
    want = OP_REGISTRY["rms_norm"].reference(x, w)
    assert_close(got, want, name="rms_norm", dtype=dtype)


# ----- Cross-entropy + embedding + dropout -----
@pytest.mark.parametrize("shape", [(16, 32), (4, 257)])
@pytest.mark.parametrize("dtype", DTYPES)
def test_cross_entropy(shape, dtype):
    logits = gen_input(shape, dtype=dtype)
    targets = torch.randint(0, shape[-1], (shape[0],), device=logits.device)
    got = OP_REGISTRY["cross_entropy"].op(logits, targets)
    want = OP_REGISTRY["cross_entropy"].reference(logits, targets)
    assert_close(got, want, name="cross_entropy", dtype=dtype)


@pytest.mark.parametrize("vocab,dim", [(128, 64), (1024, 256)])
@pytest.mark.parametrize("dtype", DTYPES)
def test_embedding(vocab, dim, dtype):
    w = gen_input((vocab, dim), dtype=dtype)
    idx = torch.randint(0, vocab, (16,), device=w.device)
    got = OP_REGISTRY["embedding"].op(idx, w)
    want = OP_REGISTRY["embedding"].reference(idx, w)
    assert_close(got, want, name="embedding", dtype=dtype)


def test_dropout_stats():
    p = 0.3
    x = torch.ones(8, 4096, device="cuda" if torch.cuda.is_available() else "cpu")
    y = OP_REGISTRY["dropout"].op(x, p=p, seed=7)
    drop_rate = (y == 0).float().mean().item()
    assert abs(drop_rate - p) < 0.05, f"observed drop rate {drop_rate} far from {p}"
    # The non-zero entries should equal 1 / (1 - p)
    kept = y[y != 0]
    if kept.numel():
        expected_scale = 1.0 / (1.0 - p)
        assert torch.allclose(kept, torch.full_like(kept, expected_scale), atol=1e-3)


# ----- matmul -----
@pytest.mark.parametrize("M,N,K", [(64, 64, 64), (128, 64, 96), (33, 17, 65)])
def test_matmul(M, N, K):
    a = gen_input((M, K), dtype=torch.float16)
    b = gen_input((K, N), dtype=torch.float16, seed=2)
    got = OP_REGISTRY["matmul"].op(a, b)
    want = OP_REGISTRY["matmul"].reference(a, b)
    assert_close(got, want, name="matmul", dtype=torch.float16)
