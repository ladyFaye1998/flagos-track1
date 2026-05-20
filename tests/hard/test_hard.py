"""Correctness tests for the 4 hard-tier kernels."""

from __future__ import annotations

import pytest
import torch

from flagos_track1 import OP_REGISTRY
from flagos_track1.testing import assert_close, gen_input


@pytest.mark.parametrize("B,H,N,D", [(1, 4, 128, 64), (2, 8, 256, 128)])
@pytest.mark.parametrize("causal", [True, False])
def test_flash_attention(B, H, N, D, causal):
    q = gen_input((B, H, N, D), dtype=torch.float16)
    k = gen_input((B, H, N, D), dtype=torch.float16, seed=1)
    v = gen_input((B, H, N, D), dtype=torch.float16, seed=2)
    got = OP_REGISTRY["flash_attention"].op(q, k, v, causal=causal)
    want = OP_REGISTRY["flash_attention"].reference(q, k, v, causal=causal)
    assert_close(got, want, name=f"flash_attention[causal={causal}]", dtype=torch.float16)


@pytest.mark.parametrize("B,N,D", [(1, 1, 64), (2, 4, 128)])
def test_rope(B, N, D):
    x = gen_input((B, N, D), dtype=torch.float16)
    cos = gen_input((B * N, D // 2), dtype=torch.float16, seed=1).view(B, N, D // 2)
    sin = gen_input((B * N, D // 2), dtype=torch.float16, seed=2).view(B, N, D // 2)
    got = OP_REGISTRY["rope"].op(x, cos, sin)
    want = OP_REGISTRY["rope"].reference(x, cos, sin)
    assert_close(got, want, name="rope", dtype=torch.float16)


@pytest.mark.parametrize("B,H,E,K", [(8, 64, 16, 2), (32, 256, 64, 4)])
def test_fused_moe_topk(B, H, E, K):
    hidden = gen_input((B, H), dtype=torch.float16)
    router = gen_input((E, H), dtype=torch.float16, seed=1)
    w_got, i_got = OP_REGISTRY["fused_moe_topk"].op(hidden, router, K)
    w_ref, i_ref = OP_REGISTRY["fused_moe_topk"].reference(hidden, router, K)
    # Indices must match exactly (or up to tie-breaking; our refs use the same path).
    assert torch.equal(i_got.sort(dim=-1).values, i_ref.sort(dim=-1).values), (
        f"top-k indices differ: {i_got} vs {i_ref}"
    )
    assert_close(w_got.sort(dim=-1).values, w_ref.sort(dim=-1).values,
                 name="moe_weights", dtype=torch.float16)


@pytest.mark.parametrize("shape", [(8, 64), (4, 257)])
def test_rms_norm_backward(shape):
    x = gen_input(shape, dtype=torch.float32, requires_grad=False)
    w = gen_input((shape[-1],), dtype=torch.float32, low=0.5, high=1.5, seed=11)
    g = gen_input(shape, dtype=torch.float32, seed=2)
    gx_got, gw_got = OP_REGISTRY["rms_norm_backward"].op(g, x, w)
    gx_ref, gw_ref = OP_REGISTRY["rms_norm_backward"].reference(g, x, w)
    assert_close(gx_got, gx_ref, name="rms_norm_bwd[grad_x]", dtype=torch.float32)
    assert_close(gw_got, gw_ref, name="rms_norm_bwd[grad_w]", dtype=torch.float32)
