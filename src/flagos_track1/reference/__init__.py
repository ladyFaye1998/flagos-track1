"""PyTorch reference implementations used as ground truth in tests + benches."""

from .torch_ref import (  # noqa: F401
    ref_abs,
    ref_argmax,
    ref_cross_entropy,
    ref_dropout,
    ref_embedding,
    ref_exp,
    ref_flash_attention,
    ref_fused_moe_topk,
    ref_gelu,
    ref_layer_norm,
    ref_log,
    ref_matmul,
    ref_relu,
    ref_rms_norm,
    ref_rms_norm_backward,
    ref_rope,
    ref_sigmoid,
    ref_silu,
    ref_softmax,
    ref_tanh,
)
