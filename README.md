<p align="center">
  <img src="docs/banner.png" alt="FlagOS Track 1 — 20 Triton GPU Operators" width="100%" />
</p>

# FlagOS Track 1 — LLM Operator Development & Optimization

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-128%2F128%20pass-brightgreen.svg)](#run-tests)
[![Operators](https://img.shields.io/badge/operators-20-58A6FF.svg)](#repository-layout)
[![Stack](https://img.shields.io/badge/stack-PyTorch%20%2B%20Triton-orange.svg)](#install)

A submission scaffold for the **FlagOS Open Computing Global Challenge (Season 1, Track 1)**.
Implements **20 Triton operators** (8 Easy + 8 Medium + 4 Hard), each with:

- a pure-PyTorch reference (golden)
- a Triton kernel (autotuned where useful)
- a CPU/CUDA fallback so tests + the CLI run anywhere
- correctness + benchmark coverage via `pytest` and a `flagos` CLI


---

## Repository layout

```
src/flagos_track1/
  ops/
    easy/      # 8 element-wise kernels (abs, exp, log, sigmoid, relu, tanh, gelu, silu)
    medium/    # 8 classic-DL kernels (softmax, layer_norm, rms_norm, cross_entropy,
               #                       embedding, dropout, argmax, matmul)
    hard/      # 4 cutting-edge kernels (flash_attention, rope, fused_moe_topk,
               #                         rms_norm_backward)
  reference/   # PyTorch golden references
  testing/     # dtype-aware assert_close + reproducible input generators
  bench/       # CUDA-event / do_bench wrapper
  cli.py       # `flagos` command (list / test / bench / package / info)
tests/         # pytest suites mirroring src/ tiers
benchmarks/    # standalone runner: `python benchmarks/run_all.py`
scripts/       # `python scripts/flagos_cli.py` without installing
docs/          # submission guide
```

## Install

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1    # on Windows
pip install -e .
pip install -r requirements.txt
```

On Windows you may need `pip install triton-windows` instead of `triton`.

## CLI

```bash
flagos info                  # show torch / triton / cuda versions
flagos list                  # list all 20 ops + per-tier prize
flagos test --tier easy      # run pytest for one tier
flagos test --op softmax     # run pytest for a single op
flagos bench --tier hard     # micro-benchmark vs PyTorch reference
flagos package --out submission.zip   # build a submission archive
```

Or without installing the package:

```bash
python scripts\flagos_cli.py list
```

## The 20 operators

| Tier | Op | Prize | Notes |
|---|---|---|---|
| Easy | abs | 1k RMB | autotuned 1-D pointwise |
| Easy | exp | 1k | fp32 internal |
| Easy | log | 1k | fp32 internal |
| Easy | sigmoid | 1k | fp32 internal |
| Easy | relu | 1k | `tl.maximum` |
| Easy | tanh | 1k | derived from `exp` |
| Easy | gelu | 1k | exact + tanh-approx kernels |
| Easy | silu | 1k | x * sigmoid(x) |
| Medium | softmax | 2k | online softmax, last-dim |
| Medium | layer_norm | 2k | forward, optional weight + bias |
| Medium | rms_norm | 2k | Llama-style RMSNorm |
| Medium | cross_entropy | 2k | fused log-softmax + NLL, ignore_index |
| Medium | embedding | 2k | gather + padding_idx |
| Medium | dropout | 2k | Triton Philox-based, scale-aware |
| Medium | argmax | 2k | tile-reduction last dim |
| Medium | matmul | 2k | blocked GEMM with autotune |
| Hard | flash_attention | 3k | FA-v2 forward, causal, D ∈ {16, 32, 64, 128} |
| Hard | rope | 3k | interleaved RoPE |
| Hard | fused_moe_topk | 3k | router softmax + top-k + renorm |
| Hard | rms_norm_backward | 3k | analytic grad_x + atomic grad_w |

## What still needs your attention

This scaffold targets four of the six scoring dimensions out of the box:

| Dimension | Status |
|---|---|
| Functional Correctness | ✅ covered by `tests/` (dtype-aware tolerances) |
| Open-Source Adaptability | ✅ FlagGems-style module layout + naming |
| Test Case Completeness | ✅ parametrized grids of shapes × dtypes |
| Code Readability | ✅ small focused files, type-hinted, fully documented |
| Performance Competitiveness | ⚠️ baselines are correct but not yet tuned to beat cuBLAS/cuDNN on every shape |
| Cross-hardware Compatibility | ⚠️ tested on CUDA; FlagOS expects 10+ backends — re-run autotune sweeps per chip |

To climb the leaderboard:

1. Re-run `flagos bench --tier <t>` on every target hardware backend and commit the chosen tile sizes.
2. For `matmul` and `flash_attention`, sweep `num_stages` / `BLOCK_*` and consider a split-K variant.
3. Add backward kernels for `softmax`, `layer_norm`, `cross_entropy` if the official task list requires them.
4. Replace the fallback paths once you confirm the official test harness only calls the Triton kernels.

See [`docs/SUBMISSION_GUIDE.md`](docs/SUBMISSION_GUIDE.md) for the full submission checklist.
