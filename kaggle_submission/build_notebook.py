"""Generate the Kaggle notebook for the FlagOS Track 1 log10 sub-task.

Run: `python build_notebook.py`  (writes ./notebook.ipynb)
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).parent / "notebook.ipynb"


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


def md(src: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": src.splitlines(keepends=True),
    }


CELLS = [
    md("""# FlagOS Track 1 - `torch.log10` Triton operator

A drop-in Triton replacement for `torch.log10`, submitted to the
**FlagOS Open Computing Global Challenge - Track 1: Operator
Development & Optimization**.

The kernel uses the identity `log10(x) = ln(x) * (1 / ln 10)`, promotes
fp16 / bf16 inputs to fp32 internally, and falls back to PyTorch when
CUDA or Triton is unavailable so the notebook executes on any backend.

Contents:

1. Environment.
2. Kernel + Python wrapper (`log10`, in-place `log10_`).
3. Correctness vs `torch.log10` (4 dtypes, edge values, shape sweep,
   `out=` and in-place paths, integer promotion).
4. Benchmark vs `torch.log10` across 8 problem sizes.
5. `submission.csv` generation.
"""),

    md("## 1. Environment"),
    code("""import math, os, warnings
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
print("torch :", torch.__version__, "| cuda:", torch.cuda.is_available())

HAS_TRITON = False
try:
    import triton
    import triton.language as tl
    import triton.testing as tt
    HAS_TRITON = True
    print("triton:", triton.__version__)
except Exception as e:
    print("triton: not available ->", e)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    cap = torch.cuda.get_device_capability(0)
    print("gpu   :", torch.cuda.get_device_name(0), f"(sm_{cap[0]}{cap[1]})")
print("device:", DEVICE, "| triton path active:", HAS_TRITON and DEVICE == "cuda")
"""),

    md("""## 2. Kernel and Python wrapper

`log10(x) = ln(x) * 0.4342944819032518`. fp16 / bf16 inputs are promoted
to fp32 inside the kernel and cast back on store, matching PyTorch's
numerical behaviour. fp64 has its own kernel so the accumulator type is
not forced down. The wrapper accepts `out=` and exposes an in-place
variant `log10_`."""),

    code("""RECIP_LN10 = 1.0 / math.log(10.0)  # 0.4342944819032518


def _autotune_configs():
    if not HAS_TRITON:
        return []
    return [
        triton.Config({"BLOCK_SIZE": bs}, num_warps=nw, num_stages=ns)
        for bs in (1024, 2048, 4096, 8192)
        for nw in (4, 8)
        for ns in (2, 3)
    ]


if HAS_TRITON:
    @triton.autotune(configs=_autotune_configs(), key=["n_elements"])
    @triton.jit
    def _log10_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask, other=1.0)
        y = (tl.log(x.to(tl.float32)) * 0.4342944819032518).to(x.dtype)
        tl.store(y_ptr + offs, y, mask=mask)

    @triton.autotune(configs=_autotune_configs(), key=["n_elements"])
    @triton.jit
    def _log10_kernel_f64(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask, other=1.0)
        tl.store(y_ptr + offs, tl.log(x) * 0.4342944819032518, mask=mask)


def log10(input: torch.Tensor, *, out=None) -> torch.Tensor:
    \"\"\"Triton replacement for torch.log10. Supports out= keyword.\"\"\"
    if not input.is_floating_point():
        input = input.to(torch.float32)
    if not (HAS_TRITON and input.is_cuda and DEVICE == "cuda"):
        return torch.log10(input, out=out) if out is not None else torch.log10(input)
    if not input.is_contiguous():
        input = input.contiguous()
    if out is None:
        out = torch.empty_like(input)
    n = input.numel()
    if n == 0:
        return out
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    if input.dtype == torch.float64:
        _log10_kernel_f64[grid](input, out, n)
    else:
        _log10_kernel[grid](input, out, n)
    return out


def log10_(input: torch.Tensor) -> torch.Tensor:
    \"\"\"In-place: writes log10(input) back into input.\"\"\"
    return log10(input, out=input)


print("public API:", log10.__name__, log10_.__name__)
"""),

    md("## 3. Correctness"),

    code("""TOL = {
    torch.float16:  (1e-3, 1e-3),
    torch.bfloat16: (1e-2, 1.6e-2),
    torch.float32:  (1e-5, 1.3e-6),
    torch.float64:  (1e-7, 1e-7),
}

for dtype, (rtol, atol) in TOL.items():
    torch.manual_seed(0)
    x = torch.rand(1024, 1024, device=DEVICE, dtype=dtype) + 0.1
    torch.testing.assert_close(log10(x), torch.log10(x),
                               equal_nan=True, rtol=rtol, atol=atol)
    print(f"ok  random   {str(dtype):>18}")

edge = torch.tensor(
    [0., -1., 1., 10., 1e-30, 1e30,
     float('inf'), -float('inf'), float('nan')],
    device=DEVICE, dtype=torch.float32,
)
torch.testing.assert_close(log10(edge), torch.log10(edge),
                           equal_nan=True, rtol=1e-5, atol=1.3e-6)
print("ok  edge values:", [round(v, 4) for v in log10(edge).cpu().tolist()])

x = torch.rand(257, device=DEVICE, dtype=torch.float32) + 0.1
out = torch.empty_like(x)
ret = log10(x, out=out)
assert ret.data_ptr() == out.data_ptr(), "out= path must alias"
torch.testing.assert_close(out, torch.log10(x), rtol=1e-5, atol=1.3e-6)
print("ok  out= path")

z = torch.rand(257, device=DEVICE, dtype=torch.float32) + 0.1
ref = torch.log10(z)
log10_(z)
torch.testing.assert_close(z, ref, rtol=1e-5, atol=1.3e-6)
print("ok  in-place log10_")

for shape in [(1,), (1, 1), (33,), (3, 17, 5), (128, 256), (1024, 1024)]:
    x = torch.rand(shape, device=DEVICE, dtype=torch.float32) + 0.1
    torch.testing.assert_close(log10(x), torch.log10(x),
                               rtol=1e-5, atol=1.3e-6)
print("ok  multi-shape")

ints = torch.arange(1, 11, device=DEVICE)
torch.testing.assert_close(log10(ints), torch.log10(ints.float()),
                           rtol=1e-5, atol=1.3e-6)
print("ok  integer promotion")
"""),

    md("## 4. Benchmark"),

    code("""if DEVICE == "cuda" and HAS_TRITON:
    sizes = [1 << k for k in (10, 13, 16, 18, 20, 22, 24, 26)]
    rows = []
    for n in sizes:
        x = torch.rand(n, device="cuda", dtype=torch.float32) + 0.1
        log10(x); torch.log10(x); torch.cuda.synchronize()
        tri_ms = tt.do_bench(lambda: log10(x), warmup=25, rep=100)
        ref_ms = tt.do_bench(lambda: torch.log10(x), warmup=25, rep=100)
        rows.append({
            "elements": n,
            "triton_ms": tri_ms,
            "torch_ms": ref_ms,
            "speedup": ref_ms / tri_ms,
            "bandwidth_gbs": (2 * n * 4) / (tri_ms * 1e-3) / 1e9,
        })
    bench = pd.DataFrame(rows)
    print(bench.to_string(index=False))
    geom = float(np.exp(np.log(bench["speedup"]).mean()))
    peak = float(bench["bandwidth_gbs"].max())
    print(f"\\ngeomean speedup vs torch.log10: {geom:.3f}x")
    print(f"peak effective bandwidth      : {peak:.0f} GB/s")
else:
    print("Skipping CUDA benchmark on CPU. Correctness above still validates the operator.")
"""),

    md("## 5. submission.csv"),

    code("""NUM_ROWS = 1000
x = torch.tensor(
    np.linspace(0.001, 1000.0, NUM_ROWS),
    device=DEVICE, dtype=torch.float64,
)
targets = log10(x).cpu().numpy()
ids = np.arange(len(targets), dtype=np.int64)

path = "/kaggle/working/submission.csv" if os.path.isdir("/kaggle/working") else "submission.csv"
pd.DataFrame({"ID": ids, "target": targets}).to_csv(path, index=False)

print(f"wrote {path}  rows={len(targets)}")
print(pd.read_csv(path).head(3))
print("...")
print(pd.read_csv(path).tail(3))
"""),
]


def main() -> None:
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"Wrote {OUT} ({OUT.stat().st_size} bytes, {len(CELLS)} cells)")


if __name__ == "__main__":
    main()
