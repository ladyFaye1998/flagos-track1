"""Easy tier (8 element-wise) — all unary, all 1D-flattened pointwise kernels.

These share a single fused launcher (``_unary_launch``) which auto-tunes
``BLOCK_SIZE`` / ``num_warps`` and handles any input dtype + non-contiguous
strides by ``.contiguous()`` + ``.view(-1)``.
"""

from .pointwise import (  # noqa: F401
    abs_op,
    exp_op,
    gelu_op,
    log_op,
    relu_op,
    sigmoid_op,
    silu_op,
    tanh_op,
)
