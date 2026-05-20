# Benchmarks

Measured speedups of the Triton kernels in `src/flagos_track1/ops/` vs the PyTorch reference paths in `src/flagos_track1/reference/`.

- Device: **NVIDIA GeForce RTX 3060 Laptop GPU (CUDA 12.4)**
- Last run: **2026-05-20 12:37 UTC**
- Methodology: median of 100 reps after 25 warmups via `triton.testing.do_bench`

Re-run with:

```bash
python benchmarks/run_all.py --tier {easy,medium,hard,all}     # single-shape headline
python benchmarks/sweep.py    --tier {easy,medium,hard,all} --markdown  # multi-shape
python benchmarks/save_results.py                              # regenerates this file
```

## Headline (representative shape per op)

### Easy tier

| Operator   |   Ours (ms) |   PyTorch (ms) | Speedup   |
|------------|-------------|----------------|-----------|
| abs        |      0.2296 |         0.2413 | 1.05x     |
| exp        |      0.2231 |         0.2301 | 1.03x     |
| log        |      0.2245 |         0.225  | 1.00x     |
| sigmoid    |      0.2299 |         0.2304 | 1.00x     |
| relu       |      0.2203 |         0.2193 | 1.00x     |
| tanh       |      0.2217 |         0.2363 | 1.07x     |
| gelu       |      0.2265 |         0.2577 | 1.14x     |
| silu       |      0.2195 |         0.2249 | 1.02x     |

### Medium tier

| Operator      |   Ours (ms) |   PyTorch (ms) | Speedup   |
|---------------|-------------|----------------|-----------|
| softmax       |      0.2238 |         0.4092 | 1.83x     |
| layer_norm    |      0.2205 |         0.3248 | 1.47x     |
| rms_norm      |      0.2204 |         1.581  | 7.17x     |
| cross_entropy |      1.3079 |         3.2829 | 2.51x     |
| embedding     |      0.2237 |         0.6156 | 2.75x     |
| dropout       |      0.3461 |         1.3331 | 3.85x     |
| argmax        |      0.1225 |         0.1768 | 1.44x     |
| matmul        |      0.1893 |         0.2651 | 1.40x     |

### Hard tier

| Operator          |   Ours (ms) |   PyTorch (ms) | Speedup   |
|-------------------|-------------|----------------|-----------|
| flash_attention   |      0.1758 |         1.702  | 9.68x     |
| rope              |      0.0125 |         0.0474 | 3.80x     |
| fused_moe_topk    |      0.9787 |         1.4318 | 1.46x     |
| rms_norm_backward |      1.4047 |         5.0084 | 3.57x     |

## Multi-shape sweep (small / medium / large per op)

### Easy tier sweep

| Operator              | Shape        | Ours (ms)   | PyTorch (ms)   | Speedup   |
|-----------------------|--------------|-------------|----------------|-----------|
| abs                   | (1024, 1024) | 0.0203      | 0.0207         | 1.02x     |
| abs                   | (4096, 4096) | 0.2336      | 0.2254         | 0.96x     |
| abs                   | (8192, 4096) | 0.4530      | 0.4513         | 1.00x     |
| **abs (geomean)**     | -            | -           | -              | **0.99x** |
| exp                   | (1024, 1024) | 0.0218      | 0.0215         | 0.98x     |
| exp                   | (4096, 4096) | 0.2291      | 0.2395         | 1.05x     |
| exp                   | (8192, 4096) | 0.4515      | 0.4624         | 1.02x     |
| **exp (geomean)**     | -            | -           | -              | **1.02x** |
| log                   | (1024, 1024) | 0.0251      | 0.0265         | 1.06x     |
| log                   | (4096, 4096) | 0.2379      | 0.2470         | 1.04x     |
| log                   | (8192, 4096) | 0.4569      | 0.6976         | 1.53x     |
| **log (geomean)**     | -            | -           | -              | **1.19x** |
| sigmoid               | (1024, 1024) | 0.0241      | 0.0217         | 0.90x     |
| sigmoid               | (4096, 4096) | 0.2201      | 0.2433         | 1.11x     |
| sigmoid               | (8192, 4096) | 0.4564      | 0.6138         | 1.34x     |
| **sigmoid (geomean)** | -            | -           | -              | **1.10x** |
| relu                  | (1024, 1024) | 0.0206      | 0.0210         | 1.02x     |
| relu                  | (4096, 4096) | 0.2275      | 0.2347         | 1.03x     |
| relu                  | (8192, 4096) | 0.4444      | 0.4647         | 1.05x     |
| **relu (geomean)**    | -            | -           | -              | **1.03x** |
| tanh                  | (1024, 1024) | 0.0232      | 0.0288         | 1.24x     |
| tanh                  | (4096, 4096) | 0.2363      | 0.3682         | 1.56x     |
| tanh                  | (8192, 4096) | 0.4409      | 0.8957         | 2.03x     |
| **tanh (geomean)**    | -            | -           | -              | **1.58x** |
| gelu                  | (1024, 1024) | 0.0314      | 0.0392         | 1.25x     |
| gelu                  | (4096, 4096) | 0.4099      | 0.5686         | 1.39x     |
| gelu                  | (8192, 4096) | 1.0209      | 1.3223         | 1.30x     |
| **gelu (geomean)**    | -            | -           | -              | **1.31x** |
| silu                  | (1024, 1024) | 0.0214      | 0.0461         | 2.15x     |
| silu                  | (4096, 4096) | 0.2291      | 0.3584         | 1.56x     |
| silu                  | (8192, 4096) | 0.4908      | 0.7941         | 1.62x     |
| **silu (geomean)**    | -            | -           | -              | **1.76x** |

### Medium tier sweep

| Operator                    | Shape         | Ours (ms)   | PyTorch (ms)   | Speedup   |
|-----------------------------|---------------|-------------|----------------|-----------|
| softmax                     | (512, 1024)   | 0.0127      | 0.0185         | 1.45x     |
| softmax                     | (4096, 4096)  | 0.2298      | 0.4809         | 2.09x     |
| softmax                     | (4096, 16384) | 0.8819      | 1.5285         | 1.73x     |
| **softmax (geomean)**       | -             | -           | -              | **1.74x** |
| layer_norm                  | (512, 1024)   | 0.0157      | 0.0249         | 1.58x     |
| layer_norm                  | (4096, 4096)  | 0.2350      | 0.3431         | 1.46x     |
| layer_norm                  | (4096, 8192)  | 0.4525      | 0.6843         | 1.51x     |
| **layer_norm (geomean)**    | -             | -           | -              | **1.52x** |
| rms_norm                    | (512, 1024)   | 0.0146      | 0.0970         | 6.63x     |
| rms_norm                    | (4096, 4096)  | 0.2207      | 1.7764         | 8.05x     |
| rms_norm                    | (4096, 8192)  | 0.4564      | 3.5841         | 7.85x     |
| **rms_norm (geomean)**      | -             | -           | -              | **7.48x** |
| cross_entropy               | (512, 1024)   | 0.0727      | 0.0797         | 1.10x     |
| cross_entropy               | (4096, 32000) | 1.5559      | 3.3021         | 2.12x     |
| cross_entropy               | (8192, 32000) | 2.8865      | 6.7011         | 2.32x     |
| **cross_entropy (geomean)** | -             | -           | -              | **1.75x** |
| embedding                   | (1024, 1024)  | 0.0205      | 0.0744         | 3.63x     |
| embedding                   | (4096, 4096)  | 0.2223      | 1.0900         | 4.90x     |
| embedding                   | (32000, 4096) | 0.2617      | 1.1325         | 4.33x     |
| **embedding (geomean)**     | -             | -           | -              | **4.25x** |
| dropout                     | (1024, 1024)  | 0.0497      | 0.1210         | 2.43x     |
| dropout                     | (4096, 4096)  | 0.6427      | 1.7152         | 2.67x     |
| dropout                     | (8192, 4096)  | 1.2227      | 3.7175         | 3.04x     |
| **dropout (geomean)**       | -             | -           | -              | **2.70x** |
| argmax                      | (512, 1024)   | 0.0171      | 0.0314         | 1.84x     |
| argmax                      | (4096, 4096)  | 0.1730      | 0.3285         | 1.90x     |
| argmax                      | (4096, 16384) | 0.4635      | 1.0463         | 2.26x     |
| **argmax (geomean)**        | -             | -           | -              | **1.99x** |
| matmul                      | (256, 256)    | 0.0248      | 0.0245         | 0.99x     |
| matmul                      | (1024, 1024)  | 0.4080      | 0.6506         | 1.59x     |
| matmul                      | (2048, 2048)  | 3.5677      | 3.8936         | 1.09x     |
| **matmul (geomean)**        | -             | -           | -              | **1.20x** |

### Hard tier sweep

| Operator                        | Shape            | Ours (ms)   | PyTorch (ms)   | Speedup    |
|---------------------------------|------------------|-------------|----------------|------------|
| flash_attention                 | (1, 8, 256, 64)  | 0.0310      | 0.1731         | 5.59x      |
| flash_attention                 | (1, 8, 1024, 64) | 0.1910      | 2.2701         | 11.88x     |
| flash_attention                 | (2, 8, 2048, 64) | 1.1777      | 21.2961        | 18.08x     |
| **flash_attention (geomean)**   | -                | -           | -              | **10.63x** |
| rope                            | (256, 1, 64)     | 0.0079      | 0.0535         | 6.77x      |
| rope                            | (1024, 1, 128)   | 0.0128      | 0.0650         | 5.08x      |
| rope                            | (4096, 1, 128)   | 0.0258      | 0.1052         | 4.07x      |
| **rope (geomean)**              | -                | -           | -              | **5.19x**  |
| fused_moe_topk                  | (512, 1024)      | 0.1417      | 0.2029         | 1.43x      |
| fused_moe_topk                  | (4096, 4096)     | 2.5449      | 3.4008         | 1.34x      |
| fused_moe_topk                  | (8192, 4096)     | 5.6961      | 6.3338         | 1.11x      |
| **fused_moe_topk (geomean)**    | -                | -           | -              | **1.29x**  |
| rms_norm_backward               | (512, 1024)      | 0.0944      | 0.2337         | 2.47x      |
| rms_norm_backward               | (4096, 4096)     | 1.6013      | 6.6106         | 4.13x      |
| rms_norm_backward               | (4096, 8192)     | 4.0237      | 13.5923        | 3.38x      |
| **rms_norm_backward (geomean)** | -                | -           | -              | **3.26x**  |
