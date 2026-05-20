<p align="center">
  <img src="docs/banner.png" alt="FlagOS Track 1 - 20 Triton GPU Operators" width="100%" />
</p>

# FlagOS Track 1 - Operator Development & Optimization

Submission for the **FlagOS Open Computing Global Challenge (Season 1, Track 1)**.
Twenty operators implemented as autotuned Triton kernels with a PyTorch
reference path, dtype-aware correctness tests, and a benchmark suite.

## Repository layout

```
src/flagos_track1/
  ops/easy/        abs exp log sigmoid relu tanh gelu silu
  ops/medium/      softmax layer_norm rms_norm cross_entropy
                   embedding dropout argmax matmul
  ops/hard/        flash_attention rope fused_moe_topk rms_norm_backward
  ops/backward/    softmax / layer_norm / cross_entropy / activation grads
  reference/       PyTorch reference for every op
  testing/         dtype-aware assert_close + reproducible input generators
  bench/           CUDA-event / triton.do_bench wrapper
  cli.py           `flagos` command (info / list / test / bench / package)
tests/             pytest suites, mirrored per tier
benchmarks/        run_all.py (headline) + sweep.py (multi-shape) + save_results.py
docs/              submission guide, technical notes, banner
demo/              narrated MP4 walkthrough
.github/workflows/ CI: pytest + CLI smoke on Py 3.10/3.11/3.12
```

## Install

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1     # PowerShell on Windows
source .venv/bin/activate        # Linux / macOS
pip install -e .
pip install -r requirements.txt
```

On Windows, install `triton-windows` in place of `triton`.

## CLI

```bash
flagos info                            # torch / triton / cuda versions
flagos list                            # the 20 operators grouped by tier
flagos test --tier easy                # run pytest for one tier
flagos test --op softmax               # run pytest for a single op
flagos bench --tier hard               # measured ms vs PyTorch reference
flagos package --out submission.zip    # archive the submission bundle
```

The same commands are available without installing the package via
`python scripts/flagos_cli.py …`.

## Correctness

`pytest` ships with **178 parametrised cases** across 25 forward + backward
kernels, covering fp16 / bf16 / fp32, shape sweeps from `(7,)` to
`(4096, 4096)`, edge values (NaN, ±Inf, zero boundary), backward
validation against `torch.autograd`, and a CPU-fallback parity suite
that runs in GitHub Actions on every push.

```bash
python -m pytest tests/ -q
# 178 passed
```

Tolerances are per dtype (`src/flagos_track1/utils.py::TOLERANCE`) and
match the FlagGems CI defaults.

## Benchmarks

Headline numbers below; the full **multi-shape sweep** (small / medium /
large per op, with geomean speedups across shapes) lives in
[`BENCHMARKS.md`](BENCHMARKS.md). Measured on RTX 3060 Laptop GPU, CUDA
12.4, fp16 unless noted, median of 100 reps after 25 warmups via
`triton.testing.do_bench`.

| Tier | Op | Mine (ms) | Torch (ms) | Speedup |
|---|---|---:|---:|---:|
| Hard | flash_attention | 0.176 | 1.702 | **9.7x** |
| Medium | rms_norm | 0.220 | 1.581 | **7.2x** |
| Hard | rope | 0.013 | 0.047 | **3.8x** |
| Medium | dropout | 0.346 | 1.333 | **3.9x** |
| Hard | rms_norm_backward | 1.405 | 5.008 | **3.6x** |
| Medium | embedding | 0.224 | 0.616 | **2.8x** |
| Medium | cross_entropy | 1.308 | 3.283 | **2.5x** |
| Medium | softmax | 0.224 | 0.409 | **1.8x** |
| Medium | layer_norm | 0.221 | 0.325 | **1.5x** |
| Medium | argmax | 0.123 | 0.177 | **1.4x** |
| Medium | matmul | 0.189 | 0.265 | **1.4x** |
| Hard | fused_moe_topk | 0.979 | 1.432 | **1.5x** |

Every Medium and Hard kernel beats the PyTorch reference. Easy-tier
element-wise ops are memory-bandwidth-bound — both Triton and the
native CUDA kernels sit at the bandwidth ceiling, so the geomean
speedups land between **0.99x and 1.76x** depending on op (full table in
`BENCHMARKS.md`). The dispatch logic for matmul (Triton in the band
where it wins, vendor BLAS at the extremes) and the per-vendor tile
configs are described in [`docs/TECHNICAL_NOTES.md`](docs/TECHNICAL_NOTES.md)
and [`docs/BACKENDS.md`](docs/BACKENDS.md).

## The 20 operators

| Tier | Op | Notes |
|---|---|---|
| Easy | abs | element-wise pointwise |
| Easy | exp | fp32 internal compute |
| Easy | log | fp32 internal compute |
| Easy | sigmoid | fp32 internal compute |
| Easy | relu | `tl.maximum` |
| Easy | tanh | derived from `exp` |
| Easy | gelu | exact + tanh-approx kernels |
| Easy | silu | `x * sigmoid(x)` |
| Medium | softmax | online softmax along the last axis |
| Medium | layer_norm | single-pass mean + variance, no diff materialisation |
| Medium | rms_norm | Llama-style RMSNorm with fp32 accumulator |
| Medium | cross_entropy | fused log-softmax + NLL with `ignore_index` |
| Medium | embedding | gather with optional `padding_idx` |
| Medium | dropout | Philox-based mask, scale-aware |
| Medium | argmax | last-dim tile reduction |
| Medium | matmul | blocked GEMM, autotuned BLOCK_M/N/K |
| Hard | flash_attention | FA-v2 forward, causal mask, head_dim in {16, 32, 64, 128} |
| Hard | rope | interleaved RoPE, fp32 cos/sin |
| Hard | fused_moe_topk | router softmax + top-k + renormalisation |
| Hard | rms_norm_backward | analytic `grad_x`, atomic `grad_w` |

## Scoring dimensions

| Dimension | Evidence |
|---|---|
| Functional Correctness | 178/178 `pytest` cases pass; backward kernels validated against `torch.autograd`; CPU-fallback parity tested on every push |
| Performance Competitiveness | [`BENCHMARKS.md`](BENCHMARKS.md) — every Medium and Hard kernel beats PyTorch, every Easy kernel matches or beats parity geomean; multi-shape sweep with reproducible methodology |
| Open-Source Adaptability | Apache-2.0, `pyproject.toml`, FlagGems-style module layout, one op per file, GitHub Actions CI |
| Cross-Platform Compatibility | `device_caps.detect()` exposes vendor + arch; per-vendor autotune configs for matmul and flash_attention; PyTorch fallback on every op; CI matrix on Py 3.10/3.11/3.12 verifies CPU parity; see [`docs/BACKENDS.md`](docs/BACKENDS.md) for the supported backend matrix |
| Test Case Completeness | parametrised shape × dtype grids, edge-value battery, 5 backward kernels exercised end-to-end, CPU-parity coverage of all 20 ops |
| Code Readability | one op per module, type hints throughout, docstrings on every public entry point, per-op rationale in [`docs/TECHNICAL_NOTES.md`](docs/TECHNICAL_NOTES.md) |

See [`docs/SUBMISSION_GUIDE.md`](docs/SUBMISSION_GUIDE.md) for the
submission checklist, [`docs/TECHNICAL_NOTES.md`](docs/TECHNICAL_NOTES.md)
for per-op design notes and supported envelopes,
[`docs/BACKENDS.md`](docs/BACKENDS.md) for the backend matrix and
detection logic, and [`demo/demo.mp4`](demo/demo.mp4) for the
narrated walkthrough.

## License

Apache-2.0, see [`LICENSE`](LICENSE).
