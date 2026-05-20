# Backends

The same Triton source is intended to compile on any backend Triton
supports. Per-vendor variation is confined to two places:

1. **Tile schedules** — autotune configs and `num_warps` / `num_stages`,
   selected at import time from `src/flagos_track1/device_caps.py::detect()`.
2. **Dispatch heuristics** — the matmul and easy-tier wrappers pick
   between the Triton kernel and the vendor BLAS / eager kernel based
   on problem size, because launch overhead and split-K crossover
   points differ per chip.

Nothing else is vendor-specific.

## Supported matrix

| Backend                                | Triton 2.3+ | Tested by author | Per-vendor configs |
|----------------------------------------|:-----------:|:----------------:|:------------------:|
| NVIDIA Ampere (sm_80, sm_86, sm_87)    | yes         | RTX 3060 Laptop  | yes (matmul + flash_attention) |
| NVIDIA Ada (sm_89)                     | yes         | architecturally  | yes (same Ampere path) |
| NVIDIA Hopper (sm_90)                  | yes         | architecturally  | yes (larger tiles + 3-stage pipeline) |
| NVIDIA Turing (sm_75)                  | yes         | architecturally  | falls back to Ampere configs |
| AMD CDNA2 / CDNA3 (MI200 / MI300)      | yes         | architecturally  | yes (smaller K, 2-stage pipelines) |
| AMD RDNA3 (Radeon RX 7000)             | yes         | architecturally  | uses CDNA configs |
| Intel Xe-HPC / Xe-HPG                  | yes         | architecturally  | uses default configs |
| CPU (no Triton)                        | n/a         | yes (CI)         | PyTorch eager fallback |

"Architecturally" means: the source compiles via Triton's standard
backend selection and uses the per-vendor `DeviceCaps`-driven tile
schedule, but I do not have physical access to the chip to publish a
benchmark number. The CPU-fallback path covers the import + signature
contract on every push through GitHub Actions.

## How detection works

`device_caps.detect()` returns a frozen dataclass:

```python
DeviceCaps(
    vendor="nvidia",          # nvidia / amd / intel / cpu / unknown
    arch="ampere",            # ampere / hopper / ada / turing / cdna2 / cdna3 / ...
    sm=86,                    # NVIDIA SM number, None elsewhere
    name="NVIDIA GeForce RTX 3060 Laptop GPU",
    triton_available=True,
    cuda_available=True,
)
```

The function is `@functools.lru_cache(maxsize=1)`-decorated, so it
runs once per process. Kernels read the result at import time and
pick configs accordingly — no per-call overhead.

## Per-op behaviour

| Op family            | NVIDIA               | AMD                  | Intel               | CPU                |
|----------------------|----------------------|----------------------|---------------------|--------------------|
| 8 easy element-wise  | Triton (≥4M elem) / torch (smaller) | same | same | torch eager     |
| softmax / layer_norm | Triton autotune (BLOCK_N = next_pow2) | same | same | F.softmax / F.layer_norm |
| rms_norm             | Triton single-pass fused             | same | same | reference path     |
| cross_entropy        | Triton fused log-softmax + NLL       | same | same | F.cross_entropy    |
| embedding            | Triton gather                        | same | same | F.embedding        |
| dropout              | Triton + Philox PRNG                 | same | same | F.dropout (deterministic seed) |
| argmax               | Triton row-tile reduction            | same | same | torch.argmax       |
| matmul               | Triton if 512²≤problem≤4096²; else cuBLAS | rocBLAS thresholds same | oneMKL same | torch.matmul |
| flash_attention      | Hopper: 128×128 / Ampere: 64×64      | CDNA: 64×32          | default 64×64       | reference (materialised softmax) |
| rope / fused_moe_topk / rms_norm_backward | Triton with shape gates | same | same | reference paths |

## Reproducing on another backend

```bash
pip install -e .
python -c "from flagos_track1.device_caps import describe; print(describe())"
python -m pytest tests/ -q                # 157 cases, every op
python benchmarks/sweep.py --tier all --markdown   # multi-shape numbers
```

If your backend isn't detected correctly, the kernels still run — they
just use the default config. Open an issue with the device name and I
can add an entry to `device_caps.py`.
