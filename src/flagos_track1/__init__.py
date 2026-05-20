"""FlagOS Track 1 — 20 Triton operators (Easy / Medium / Hard).

Public API: import :mod:`flagos_track1.ops` to get the operator registry,
or call individual ops directly, e.g. ``from flagos_track1.ops.easy import gelu``.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .device_caps import DeviceCaps, detect as detect_device  # noqa: F401
from .ops import OP_REGISTRY, get_op, list_ops  # noqa: F401
