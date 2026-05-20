# Technical notes

Per-operator design notes for `src/flagos_track1/ops/`. Each section
covers the algorithmic choice, the numerical precision contract, the
tile sizing strategy and the supported shape/dtype envelope. Measured
speedups come from `BENCHMARKS.md` (RTX 3060, CUDA 12.x).

## Cross-platform design

A single Triton source compiles to every backend Triton supports.
Per-vendor variation is confined to two layers:

1. **`device_caps.detect()`** — returns a frozen `DeviceCaps` dataclass
   with `vendor ∈ {nvidia, amd, intel, cpu, unknown}` and `arch ∈
   {ampere, hopper, ada, turing, cdna2, cdna3, rdna3, xe-hpc, xe-hpg,
   cpu}`. Cached at import time, zero per-call overhead.
2. **Tile schedules** — kernels that benefit from per-vendor tuning
   (matmul, flash_attention) select their `triton.Config` set or
   `num_warps / num_stages` from the detected arch. Other kernels
   use a single config that is safe on every backend.

Every op also has a PyTorch eager fallback, so the wrapper runs
end-to-end on CPU. `tests/test_cpu_fallback.py` exercises that path
on every push through GitHub Actions; the full backend matrix is in
[`docs/BACKENDS.md`](BACKENDS.md).

## Conventions

- All kernels live behind a thin Python wrapper that
  (a) accepts the same signature as the PyTorch reference,
  (b) falls back to PyTorch on CPU or when the input is outside the
      supported tile envelope, and
  (c) does not allocate scratch outside the kernel launch.
- Reductions and intermediate products run in **fp32** regardless of the
  input dtype, then cast back at store time. This matches PyTorch's
  native softmax/layernorm precision and is the source of the close
  numerical agreement with the reference (`assert_close` at rtol=1e-3,
  atol=1e-3 for fp16; 1e-5, 1e-6 for fp32).
- Block-size constants follow the rule "next power of two of the inner
  reduction dimension, capped at 16384", with `num_warps` ramped from
  2 → 4 → 8 across `BLOCK_N` thresholds. This is small enough to fit
  registers on Ampere and avoids spilling.

## Easy tier — element-wise ops (8 kernels)

`abs, exp, log, sigmoid, relu, tanh, gelu, silu`

- Purely memory-bound; runtime is dominated by the load + store, not
  the arithmetic. The Triton kernel uses a 1-D grid over `BLOCK_SIZE
  ∈ {1024, 2048, 4096}` with `num_warps = 4 or 8` and `num_stages = 2`
  (Ampere async copy pipeline), selected by `triton.autotune` on
  the `n_elements` key.
- Geomean speedups across the sweep: `silu` **1.76x**, `tanh` **1.58x**,
  `gelu` **1.31x**, `log` **1.19x**, `sigmoid` **1.10x**, `relu`
  **1.03x**, `exp` **1.02x**, `abs` **0.99x** (parity). The ops where
  the math is non-trivial (sigmoid, gelu, silu, tanh) benefit most
  from Triton's single-pass fused exp/erf; pure-bandwidth ops (abs,
  relu) sit at the bandwidth ceiling that PyTorch's native kernels
  also reach.
- `log` clamps inputs to `[0.1, ∞)` in the input generator to avoid
  `-inf` in the reference, mirroring how callers use it in LLM
  training (no `log(0)`).
- `gelu` uses the **exact** erf-based formulation by default; the
  backward kernel additionally exposes the tanh approximation
  selected by `approximate="tanh"`.

## Medium tier — reductions, normalizations, matmul (8 kernels)

### `softmax`
- One block per row, `BLOCK_N = next_pow2(n_cols)` up to 16384.
- Online stabilisation: subtract row-max before `exp`. Reductions
  in fp32; final divide casts back to input dtype.
- Geomean 1.59x speedup, peaks at 2.00x for 4096×4096.

### `argmax`
- Same row-per-block layout. Uses `tl.argmax` over the fp32-promoted
  row. The wrapper short-circuits to `torch.argmax` when the row is
  wider than the largest supported block (>16384).
- Geomean 1.72x.

### `layer_norm`
- Single-pass fused mean/var/normalize/affine. We compute `mean` and
  `mean_of_x2` in the same reduction, then `var = mean_x2 - mean*mean`,
  which halves the number of loads vs the two-pass formulation.
- `num_warps = 8 for BLOCK_N≥4096 else 4 for ≥1024 else 2`, plus
  `num_stages=2` for pipelined loads. Geomean 1.47x.
- Previous two-pass implementation was 0.7x; the rewrite is documented
  in the git history.

### `rms_norm`
- Fused single-pass: `rstd = rsqrt(mean(x^2) + eps)`, then
  `y = x * rstd * w`. No mean subtraction, no bias.
- **Biggest medium-tier win: geomean 6.26x** over the eager
  `x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w` path,
  which materialises three intermediate tensors.

### `cross_entropy`
- Fused log-softmax + NLL in one row-major kernel: compute
  `log_sum_exp` and the target logit in one pass, return
  `lse - x[t]`. Handles `ignore_index` by masking the per-row loss
  and the divisor.
- Geomean 1.83x against `torch.nn.functional.cross_entropy(reduction="mean")`.
- 1D-grid over rows; each program loads its row in `BLOCK_N` chunks
  (`BLOCK_N = next_pow2(n_classes)`, supports up to 32k classes —
  enough for Llama-class vocabularies).

### `embedding`
- Indirect gather via `tl.load(W + indices[:, None] * D + offs[None])`.
  Vectorised over indices and embedding dim.
- Geomean 2.84x against `F.embedding`, with the largest gain
  (3.89x) on `(32000, 4096)` — the case where PyTorch's gather
  serialises most.

### `dropout`
- Philox PRNG state derived from `(seed, offset)`, runs in fp32 then
  casts back. No `randperm` materialisation.
- Geomean 2.87x vs `F.dropout` because the reference path constructs
  a full Bernoulli mask tensor.

### `matmul`
- Per-vendor autotune configs picked at import time via
  `device_caps.detect()`: Ampere / Ada use 64-128 × 32 tiles with
  3-stage pipelines, Hopper uses 128-256 × 64 tiles with 3-4 stages,
  AMD CDNA uses smaller K and 2-stage pipelines.
- FP32 accumulator, fp16 inputs.
- The wrapper picks between the Triton kernel and the vendor BLAS
  per call: very small problems (≤ 512²) hit BLAS because launch
  overhead dominates; very large problems (≥ 4096²) hit BLAS because
  split-K matters; the middle band uses the Triton kernel. Geomean
  speedup across the sweep: **1.20x** (was 0.85x before dispatch).
- The thresholds are device-derived (Ampere on RTX 3060 Laptop GPU);
  rerunning `benchmarks/sweep.py` on a different chip and adjusting
  the two constants in `matmul.py` is a 2-line change.

## Hard tier — fused attention, RoPE, MoE, backward (4 kernels)

### `flash_attention` — FlashAttention-v2 forward
- Online-softmax recurrence: keep `(m_i, l_i)` per query block,
  rescale `acc` by `alpha = exp(m_old - m_new)` on each KV tile.
- Tiling: `BLOCK_M = BLOCK_N = 64`, `BLOCK_DMODEL = head_dim`.
- Supports `head_dim ∈ {16, 32, 64, 128}`, causal + non-causal.
- **Geomean 9.41x**, up to **15.7x at (2, 8, 2048, 64)**, because
  the PyTorch reference materialises the full `N×N` score matrix.

### `rope`
- Applies rotary position embedding in-place per `(D/2)`-pair:
  `[x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]`.
- 1D grid over `(B*N)`, one block per token. fp32 trig, fp16 store.
- Geomean 4.10x.

### `fused_moe_topk`
- Router GEMM `(hidden @ router.T)` runs in **fp32** internally
  (precision was off-by-one in fp16; this was the fix that brought
  correctness back). Then per-row top-K + softmax over the k
  selected experts.
- Geomean 1.14x. The bottleneck is the router matmul against
  cuBLAS; the top-K + softmax fusion saves a launch but not much
  bandwidth.

### `rms_norm_backward`
- Returns `(grad_x, grad_w)`. Same fused-reduction strategy as the
  forward; `grad_w` is reduced via `atomic_add` into a single
  fp32 buffer, then cast back to weight dtype.
- Geomean 4.62x. The PyTorch reference path runs the full autograd
  graph of three intermediate tensors.

## Backward kernels (5 additional, exercised by tests)

`softmax_bwd, layer_norm_bwd, cross_entropy_bwd, silu_bwd, gelu_bwd`

- Not in the 20-op headline grid; shipped because real training
  pipelines need them and the same reduction/fusion patterns apply.
- All five are validated against `torch.autograd` in
  `tests/backward/test_backward.py`.
- `layer_norm_bwd` and `cross_entropy_bwd` use `atomic_add` for
  the parameter gradients (`grad_weight`, `grad_bias`). The
  resulting non-determinism is below 1 ULP × N magnitude (atol
  1e-4 in the test); the per-element gradients (`grad_x`,
  `grad_logits`) are bit-comparable.

## Numerical contract summary

| Op family            | Reduction dtype | Accumulator dtype | Tolerance vs reference |
|----------------------|-----------------|-------------------|------------------------|
| element-wise         | n/a             | fp32              | fp32: 1e-6, fp16: 1e-3 |
| softmax / layer_norm | fp32            | fp32              | fp32: 1e-5, fp16: 1e-3 |
| matmul / attention   | fp32            | fp32              | fp16: 1e-3             |
| backward (parameter) | fp32            | fp32 atomic_add   | 1e-4 (atomic ordering) |
| backward (input)     | fp32            | fp32              | matches forward family |

## Shape / dtype envelope per op

| Operator            | Dtypes               | Inner-dim cap | Notes                          |
|---------------------|----------------------|---------------|--------------------------------|
| element-wise (8)    | fp16, bf16, fp32     | unrestricted  | 1D grid, BLOCK=1024            |
| softmax / argmax    | fp16, bf16, fp32     | 16384         | falls back beyond              |
| layer_norm / rms    | fp16, bf16, fp32     | 16384         | requires contiguous last dim   |
| cross_entropy       | fp16, bf16, fp32     | 32768         | up to Llama-class vocab        |
| embedding           | fp16, bf16, fp32     | unrestricted  | gather over arbitrary indices  |
| dropout             | fp16, bf16, fp32     | unrestricted  | Philox seed=42 reproducible    |
| matmul              | fp16, bf16 → fp32 acc| 4096²         | autotuned 4 configs            |
| flash_attention     | fp16, bf16           | seq≤8192, D∈{16,32,64,128} | causal + non-causal |
| rope                | fp16, bf16, fp32     | D≤128 even    |                                |
| fused_moe_topk      | fp16, bf16           | hidden≤4096   | router GEMM in fp32            |
| rms_norm_backward   | fp16, bf16, fp32     | 16384         |                                |
