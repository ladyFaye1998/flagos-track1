# Submission Guide - FlagOS Track 1

## 1. Pre-flight

```bash
flagos info        # torch / triton / cuda versions
flagos list        # confirm all 20 operators are registered
flagos test        # 157/157 pytest cases on the Triton + CUDA path
flagos bench       # measured ms vs the PyTorch reference
```

On CPU-only machines (no CUDA), every op still imports cleanly and runs
through the PyTorch fallback — useful as a smoke test before pushing to
a GPU runner.

## 2. Package the archive

```bash
flagos package --out submission.zip
```

The bundle contains:

- `src/flagos_track1/**` — 20 forward kernels, 5 backward kernels, PyTorch references, utilities
- `tests/**` — parametrised pytest suite (157 cases, including `tests/backward/`)
- `benchmarks/**` — `run_all.py` (headline), `sweep.py` (multi-shape), `save_results.py`
- `docs/**` — this guide, `TECHNICAL_NOTES.md`, banner
- `BENCHMARKS.md`, `README.md`, `pyproject.toml`, `requirements.txt`, `LICENSE`

Build artefacts (`__pycache__/`, `*.egg-info/`, `demo/audio/`, `demo/frames/`)
are excluded by the packager. Verified output: 51 files, ~135 KB.

## 3. Per-chip tile configs

Every kernel that benefits from tuning is wrapped in `triton.autotune` with
an explicit key list, so the same source compiles a fresh tile schedule on
each backend (NVIDIA Ampere / Hopper, AMD MI300, Cambricon, Ascend,
Iluvatar, ...). Active config sweeps live next to the ops:

- `src/flagos_track1/ops/easy/pointwise.py::_AUTOTUNE_CFGS`
- `src/flagos_track1/ops/medium/matmul.py::_MM_CFGS`

The numerical contract and supported shape/dtype envelope per op is
documented in [`TECHNICAL_NOTES.md`](TECHNICAL_NOTES.md).

## 4. Operator coverage

**20 forward operators** registered in `src/flagos_track1/ops/__init__.py`:

- **Easy (8):** abs, exp, log, sigmoid, relu, tanh, gelu, silu
- **Medium (8):** softmax, layer_norm, rms_norm, cross_entropy,
  embedding, dropout, argmax, matmul
- **Hard (4):** flash_attention, rope, fused_moe_topk, rms_norm_backward

**5 backward kernels** in `src/flagos_track1/ops/backward/`:

- `softmax_backward`, `layer_norm_backward`, `cross_entropy_backward`,
  `silu_backward`, `gelu_backward` (exact + tanh approx)

All backward kernels are validated against `torch.autograd` in
`tests/backward/test_backward.py`.

## 5. Scoring dimensions cheat sheet

| Dimension | Where it lives in the repo |
|---|---|
| Functional Correctness | `tests/` — 157 cases, dtype-aware `assert_close`, backward kernels validated against `torch.autograd` |
| Performance Competitiveness | `BENCHMARKS.md` — headline table + multi-shape sweep with geomean speedup per op |
| Open-Source Adaptability | Apache-2.0, `pyproject.toml`, FlagGems-style layout, `.github/workflows/ci.yml` matrix |
| Cross-Platform Compatibility | PyTorch fallback on every op, autotune keys per kernel, CI on Py 3.10/3.11/3.12 |
| Test Case Completeness | parametrised shape x dtype grids, edge-value batteries, forward + backward coverage |
| Code Readability | one op per file, full type hints, docstrings, per-op rationale in `docs/TECHNICAL_NOTES.md` |

## 6. Submit

Per the official rules ([DoraHacks](https://dorahacks.io/hackathon/flagos-open-computing/detail)):

1. Submit through the official channel before the published deadline.
2. The contact information matches the registration form.
3. Third-party code, where used, is cited with source + license in the file header.
4. The same work is submitted to a single track only.
