# Submission Guide — FlagOS Track 1

## 1. Pre-flight checks

```bash
flagos info        # torch / triton / cuda versions
flagos list        # confirm all 20 ops registered
flagos test        # full pytest suite (128/128 pass)
flagos bench       # micro-benchmarks vs PyTorch reference
```

## 2. Package the archive

```bash
flagos package --out submission.zip
```

The archive bundles:

- `src/flagos_track1/**` — the 20 Triton kernels + PyTorch references + utilities
- `tests/**` — the 128-case parametrized pytest suite ("Test Case Completeness")
- `benchmarks/**` — the reproducible perf runner
- `docs/**` — this guide + technical notes
- `README.md`, `pyproject.toml`, `requirements.txt`, `LICENSE`

Build artifacts (`__pycache__`, `*.egg-info`, etc.) are skipped automatically.

## 3. Per-chip tile configs

Triton autotune keys are wired on every kernel, so the same source compiles
optimal tile sizes on each backend FlagGems targets (NVIDIA H100/A100, AMD
MI300, Cambricon, Ascend, Iluvatar, …). Cached configs live next to each op
(see `_AUTOTUNE_CFGS` in `ops/easy/pointwise.py` and `_MM_CFGS` in
`ops/medium/matmul.py`).

## 4. Operator coverage

The 20 ops registered in `src/flagos_track1/ops/__init__.py` match the
official Track-1 task list across all three tiers:

- 8 Easy pointwise: abs, exp, log, sigmoid, relu, tanh, gelu, silu
- 8 Medium: softmax, layer_norm, rms_norm, cross_entropy, embedding,
  dropout, argmax, matmul
- 4 Hard: flash_attention, rope, fused_moe_topk, rms_norm_backward

## 5. Scoring dimensions cheat sheet

| Dimension | Where it lives in the repo |
|---|---|
| Functional Correctness | `tests/` — dtype-aware `assert_close`, 128 cases, all green |
| Performance Competitiveness | `benchmarks/run_all.py` + per-op `triton.autotune` |
| Open-Source Adaptability | Apache-2.0 + FlagGems-style layout under `src/flagos_track1/ops/{easy,medium,hard}/` |
| Cross-Platform Compatibility | PyTorch fallback path on every op so imports work on any device |
| Test Case Completeness | parametrized shape × dtype grids + edge-value batteries |
| Code Readability | one op per file, full type hints, docstrings, no dead code |

## 6. Submit

Per the official rules ([DoraHacks page](https://dorahacks.io/hackathon/flagos-open-computing/detail)):

1. Submit through the official channel before **2026-05-20 15:59 (Beijing time)**.
2. The contact info matches the registration form.
3. Third-party code is cited with source + license in the relevant file header.
4. The same work is submitted to a single track only.
