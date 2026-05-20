"""Device capability detection used by per-vendor kernel tuning.

The same Triton source emits different machine code per backend; this
module gives the kernels a stable handle on which family they are
running on so they can pick an appropriate tile schedule, warp count
and stage count.

Returned values are pure data — no side effects, safe to call at
import time. Detection is cached.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Literal

import torch


Vendor = Literal["nvidia", "amd", "intel", "cpu", "unknown"]
Arch = Literal[
    "ampere", "hopper", "ada", "turing", "volta", "pascal",  # NVIDIA
    "cdna2", "cdna3", "rdna3",                               # AMD
    "xe-hpc", "xe-hpg",                                      # Intel
    "cpu", "unknown",
]


@dataclass(frozen=True)
class DeviceCaps:
    vendor: Vendor
    arch: Arch
    sm: int | None              # NVIDIA SM (e.g. 80, 89, 90) or None
    name: str
    triton_available: bool
    cuda_available: bool

    def is_nvidia(self) -> bool:
        return self.vendor == "nvidia"

    def is_amd(self) -> bool:
        return self.vendor == "amd"

    def has_tensor_cores(self) -> bool:
        # NVIDIA Volta+ and recent AMD MI series expose matrix units.
        if self.is_nvidia() and self.sm is not None:
            return self.sm >= 70
        if self.is_amd() and self.arch in ("cdna2", "cdna3"):
            return True
        return False


def _nvidia_arch_for_sm(sm: int) -> Arch:
    # Mapping per NVIDIA's compute-capability table.
    if sm >= 90:
        return "hopper"
    if sm >= 89:
        return "ada"
    if sm >= 80:
        return "ampere"
    if sm >= 75:
        return "turing"
    if sm >= 70:
        return "volta"
    return "pascal"


@functools.lru_cache(maxsize=1)
def detect() -> DeviceCaps:
    try:
        import triton  # type: ignore  # noqa: F401
        triton_avail = True
    except Exception:
        triton_avail = False

    if not torch.cuda.is_available():
        return DeviceCaps(
            vendor="cpu", arch="cpu", sm=None,
            name="cpu", triton_available=triton_avail, cuda_available=False,
        )

    name = torch.cuda.get_device_name(0)
    name_l = name.lower()

    if any(k in name_l for k in ("nvidia", "geforce", "rtx", "gtx", "tesla", "quadro",
                                  "h100", "h200", "a100", "a40", "a10")):
        cap = torch.cuda.get_device_capability(0)
        sm = cap[0] * 10 + cap[1]
        return DeviceCaps(
            vendor="nvidia", arch=_nvidia_arch_for_sm(sm), sm=sm, name=name,
            triton_available=triton_avail, cuda_available=True,
        )

    if any(k in name_l for k in ("amd", "radeon", "mi100", "mi200", "mi210", "mi250",
                                  "mi300", "mi325", "instinct")):
        if "mi300" in name_l or "mi325" in name_l:
            arch: Arch = "cdna3"
        elif any(k in name_l for k in ("mi100", "mi200", "mi210", "mi250")):
            arch = "cdna2"
        elif "radeon" in name_l:
            arch = "rdna3"
        else:
            arch = "unknown"
        return DeviceCaps(
            vendor="amd", arch=arch, sm=None, name=name,
            triton_available=triton_avail, cuda_available=True,
        )

    if any(k in name_l for k in ("intel", "arc", "xe")):
        arch = "xe-hpc" if "max" in name_l or "ponte" in name_l else "xe-hpg"
        return DeviceCaps(
            vendor="intel", arch=arch, sm=None, name=name,
            triton_available=triton_avail, cuda_available=True,
        )

    return DeviceCaps(
        vendor="unknown", arch="unknown", sm=None, name=name,
        triton_available=triton_avail, cuda_available=True,
    )


def describe() -> str:
    c = detect()
    if c.vendor == "cpu":
        return "cpu (PyTorch reference path)"
    sm = f"sm{c.sm}" if c.sm is not None else "n/a"
    return f"{c.vendor}/{c.arch} ({sm}) - {c.name}"
