"""Central registry for all 20 operators.

Each operator is registered with:
- ``name``: canonical short name (matches FlagGems convention where possible)
- ``tier``: 'easy' | 'medium' | 'hard'
- ``op``: callable (forward)
- ``reference``: callable producing the golden output for the same inputs

The registry powers the CLI (`flagos test`, `flagos bench`, `flagos list`) and
the pytest parametrization in `tests/`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

from . import easy, hard, medium
from .. import reference as ref


@dataclass(frozen=True)
class OpEntry:
    name: str
    tier: str
    op: Callable
    reference: Callable

    @property
    def prize_rmb(self) -> int:
        return {"easy": 1000, "medium": 2000, "hard": 3000}[self.tier]


def _e(name: str, op: Callable, reference: Callable) -> OpEntry:
    return OpEntry(name=name, tier="easy", op=op, reference=reference)


def _m(name: str, op: Callable, reference: Callable) -> OpEntry:
    return OpEntry(name=name, tier="medium", op=op, reference=reference)


def _h(name: str, op: Callable, reference: Callable) -> OpEntry:
    return OpEntry(name=name, tier="hard", op=op, reference=reference)


OP_REGISTRY: Dict[str, OpEntry] = {
    # ---- Easy (8) ----
    "abs": _e("abs", easy.abs_op, ref.ref_abs),
    "exp": _e("exp", easy.exp_op, ref.ref_exp),
    "log": _e("log", easy.log_op, ref.ref_log),
    "sigmoid": _e("sigmoid", easy.sigmoid_op, ref.ref_sigmoid),
    "relu": _e("relu", easy.relu_op, ref.ref_relu),
    "tanh": _e("tanh", easy.tanh_op, ref.ref_tanh),
    "gelu": _e("gelu", easy.gelu_op, ref.ref_gelu),
    "silu": _e("silu", easy.silu_op, ref.ref_silu),
    # ---- Medium (8) ----
    "softmax": _m("softmax", medium.softmax_op, ref.ref_softmax),
    "layer_norm": _m("layer_norm", medium.layer_norm_op, ref.ref_layer_norm),
    "rms_norm": _m("rms_norm", medium.rms_norm_op, ref.ref_rms_norm),
    "cross_entropy": _m("cross_entropy", medium.cross_entropy_op, ref.ref_cross_entropy),
    "embedding": _m("embedding", medium.embedding_op, ref.ref_embedding),
    "dropout": _m("dropout", medium.dropout_op, ref.ref_dropout),
    "argmax": _m("argmax", medium.argmax_op, ref.ref_argmax),
    "matmul": _m("matmul", medium.matmul_op, ref.ref_matmul),
    # ---- Hard (4) ----
    "flash_attention": _h("flash_attention", hard.flash_attention_op, ref.ref_flash_attention),
    "rope": _h("rope", hard.rope_op, ref.ref_rope),
    "fused_moe_topk": _h("fused_moe_topk", hard.fused_moe_topk_op, ref.ref_fused_moe_topk),
    "rms_norm_backward": _h(
        "rms_norm_backward", hard.rms_norm_backward_op, ref.ref_rms_norm_backward
    ),
}


def get_op(name: str) -> OpEntry:
    if name not in OP_REGISTRY:
        raise KeyError(
            f"Unknown operator '{name}'. Known: {sorted(OP_REGISTRY.keys())}"
        )
    return OP_REGISTRY[name]


def list_ops(tier: str | None = None) -> List[OpEntry]:
    items = list(OP_REGISTRY.values())
    if tier is not None:
        items = [e for e in items if e.tier == tier]
    return items
