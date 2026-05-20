# Submission Guide — FlagOS Track 1

## 1. Pre-flight checks

```bash
flagos info        # torch / triton / cuda OK?
flagos list        # 20 ops registered?
flagos test        # all pytest green?
flagos bench       # speed sanity check vs PyTorch ref
```

## 2. Package the archive

```bash
flagos package --out submission.zip
```

This bundles:

- `src/flagos_track1/**` — your kernels + reference + utilities
- `tests/**` — full pytest suite (= "Test Case Completeness" score)
- `benchmarks/**` — reproducible perf script
- `docs/**` — this guide + any technical write-up
- `README.md`, `pyproject.toml`, `requirements.txt`, `LICENSE`

`__pycache__` and other artifacts are skipped automatically.

## 3. Tune per chip before submitting

For every hardware backend you target (NVIDIA H100, A100, AMD MI300, Cambricon,
Ascend, Iluvatar, etc.):

1. Activate the matching Triton fork (FlagGems supports 10+ backends).
2. Run `flagos bench --tier all`.
3. Commit the winning `triton.Config` per op (see `_AUTOTUNE_CFGS` in
   `ops/easy/pointwise.py` and `_MM_CFGS` in `ops/medium/matmul.py`).

## 4. Match the official task list

Before final submission, double-check that the 20 ops you registered match the
official Track-1 task list (released to registered participants). If a task
asks for an op we don't have, add a module under the matching tier and register
it in `src/flagos_track1/ops/__init__.py`.

## 5. Scoring dimensions cheat sheet

| Dimension | Where to look in this repo |
|---|---|
| Functional Correctness | `tests/` (dtype-aware `assert_close`) |
| Performance Competitiveness | `benchmarks/run_all.py` + per-chip `triton.Config` tuning |
| Open-Source Adaptability | FlagGems-style layout under `src/flagos_track1/ops/{easy,medium,hard}/` |
| Cross-hardware Compatibility | each kernel falls back to PyTorch when Triton/CUDA missing |
| Test Case Completeness | parametrized shape × dtype grids in `tests/` |
| Code Readability | one op per file, docstrings, no dead code |

## 6. Submit

Per the official rules ([DoraHacks page](https://dorahacks.io/hackathon/flagos-open-computing/detail)):

1. Submit through the officially designated channel before **2026-05-20 15:59 (Beijing time)**.
2. Include the contact info that matches your registration form.
3. If you used any third-party / open-source code, cite the source and license
   in the relevant file header.
4. Do not submit the same work to multiple tracks.
