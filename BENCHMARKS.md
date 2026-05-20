# Benchmarks

Measured speedups of the Triton kernels in `src/flagos_track1/ops/` vs the PyTorch reference paths in `src/flagos_track1/reference/`.

- Device: **NVIDIA GeForce RTX 3060 Laptop GPU (CUDA 12.4)**
- Last run: **2026-05-20 12:21 UTC**
- Methodology: median of 20 reps after 5 warmups via `triton.testing.do_bench`

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
| abs        |      0.2701 |         0.2734 | 1.01x     |
| exp        |      0.2184 |         0.2449 | 1.12x     |
| log        |      0.2567 |         0.2955 | 1.15x     |
| sigmoid    |      0.2273 |         0.2667 | 1.17x     |
| relu       |      0.2612 |         0.2411 | 0.92x     |
| tanh       |      0.2743 |         0.254  | 0.93x     |
| gelu       |      0.2188 |         0.3081 | 1.41x     |
| silu       |      0.2593 |         0.2226 | 0.86x     |

### Medium tier

| Operator      |   Ours (ms) |   PyTorch (ms) | Speedup   |
|---------------|-------------|----------------|-----------|
| softmax       |      0.2138 |         0.4884 | 2.28x     |
| layer_norm    |      0.2203 |         0.3549 | 1.61x     |
| rms_norm      |      0.2221 |         1.9886 | 8.95x     |
| cross_entropy |      0.9193 |         2.8559 | 3.11x     |
| embedding     |      0.2484 |         0.5149 | 2.07x     |
| dropout       |      0.4649 |         1.5961 | 3.43x     |
| argmax        |      0.1473 |         0.2289 | 1.55x     |
| matmul        |      0.1889 |         0.2197 | 1.16x     |

### Hard tier

| Operator          |   Ours (ms) |   PyTorch (ms) | Speedup   |
|-------------------|-------------|----------------|-----------|
| flash_attention   |      0.1226 |         2.3207 | 18.93x    |
| rope              |      0.0144 |         0.0871 | 6.04x     |
| fused_moe_topk    |      1.372  |         1.3486 | 0.98x     |
| rms_norm_backward |      1.1012 |         4.7971 | 4.36x     |

## Multi-shape sweep (small / medium / large per op)

### Easy tier sweep

| Operator              | Shape        | Ours (ms)   | PyTorch (ms)   | Speedup   |
|-----------------------|--------------|-------------|----------------|-----------|
| abs                   | (1024, 1024) | 0.0286      | 0.0292         | 1.02x     |
| abs                   | (4096, 4096) | 0.2274      | 0.2564         | 1.13x     |
| abs                   | (8192, 4096) | 0.5374      | 0.5102         | 0.95x     |
| **abs (geomean)**     | -            | -           | -              | **1.03x** |
| exp                   | (1024, 1024) | 0.0404      | 0.0255         | 0.63x     |
| exp                   | (4096, 4096) | 0.2542      | 0.2369         | 0.93x     |
| exp                   | (8192, 4096) | 0.4577      | 0.4515         | 0.99x     |
| **exp (geomean)**     | -            | -           | -              | **0.83x** |
| log                   | (1024, 1024) | 0.0289      | 0.0287         | 0.99x     |
| log                   | (4096, 4096) | 0.2540      | 0.2742         | 1.08x     |
| log                   | (8192, 4096) | 0.4942      | 0.6023         | 1.22x     |
| **log (geomean)**     | -            | -           | -              | **1.09x** |
| sigmoid               | (1024, 1024) | 0.0202      | 0.0321         | 1.59x     |
| sigmoid               | (4096, 4096) | 0.2482      | 0.2574         | 1.04x     |
| sigmoid               | (8192, 4096) | 0.5446      | 0.6375         | 1.17x     |
| **sigmoid (geomean)** | -            | -           | -              | **1.24x** |
| relu                  | (1024, 1024) | 0.0405      | 0.0379         | 0.94x     |
| relu                  | (4096, 4096) | 0.2510      | 0.2778         | 1.11x     |
| relu                  | (8192, 4096) | 0.4999      | 0.4279         | 0.86x     |
| **relu (geomean)**    | -            | -           | -              | **0.96x** |
| tanh                  | (1024, 1024) | 0.0379      | 0.0264         | 0.69x     |
| tanh                  | (4096, 4096) | 0.2220      | 0.2952         | 1.33x     |
| tanh                  | (8192, 4096) | 0.5273      | 0.7297         | 1.38x     |
| **tanh (geomean)**    | -            | -           | -              | **1.09x** |
| gelu                  | (1024, 1024) | 0.0269      | 0.0569         | 2.11x     |
| gelu                  | (4096, 4096) | 0.2628      | 0.2915         | 1.11x     |
| gelu                  | (8192, 4096) | 0.6242      | 1.0074         | 1.61x     |
| **gelu (geomean)**    | -            | -           | -              | **1.56x** |
| silu                  | (1024, 1024) | 0.0282      | 0.0321         | 1.14x     |
| silu                  | (4096, 4096) | 0.2851      | 0.2805         | 0.98x     |
| silu                  | (8192, 4096) | 0.4618      | 0.5419         | 1.17x     |
| **silu (geomean)**    | -            | -           | -              | **1.10x** |

### Medium tier sweep

| Operator                    | Shape         | Ours (ms)   | PyTorch (ms)   | Speedup   |
|-----------------------------|---------------|-------------|----------------|-----------|
| softmax                     | (512, 1024)   | 0.0183      | 0.0235         | 1.29x     |
| softmax                     | (4096, 4096)  | 0.2436      | 0.7638         | 3.14x     |
| softmax                     | (4096, 16384) | 0.9040      | 1.7847         | 1.97x     |
| **softmax (geomean)**       | -             | -           | -              | **2.00x** |
| layer_norm                  | (512, 1024)   | 0.0228      | 0.0298         | 1.31x     |
| layer_norm                  | (4096, 4096)  | 0.2201      | 0.3656         | 1.66x     |
| layer_norm                  | (4096, 8192)  | 0.4922      | 0.7918         | 1.61x     |
| **layer_norm (geomean)**    | -             | -           | -              | **1.52x** |
| rms_norm                    | (512, 1024)   | 0.0336      | 0.0810         | 2.41x     |
| rms_norm                    | (4096, 4096)  | 0.2358      | 1.6699         | 7.08x     |
| rms_norm                    | (4096, 8192)  | 0.5155      | 3.7257         | 7.23x     |
| **rms_norm (geomean)**      | -             | -           | -              | **4.98x** |
| cross_entropy               | (512, 1024)   | 0.0569      | 0.0383         | 0.67x     |
| cross_entropy               | (4096, 32000) | 0.8851      | 3.4263         | 3.87x     |
| cross_entropy               | (8192, 32000) | 2.1924      | 5.4467         | 2.48x     |
| **cross_entropy (geomean)** | -             | -           | -              | **1.86x** |
| embedding                   | (1024, 1024)  | 0.0195      | 0.0680         | 3.49x     |
| embedding                   | (4096, 4096)  | 0.2154      | 0.9433         | 4.38x     |
| embedding                   | (32000, 4096) | 0.2570      | 0.9815         | 3.82x     |
| **embedding (geomean)**     | -             | -           | -              | **3.88x** |
| dropout                     | (1024, 1024)  | 0.0348      | 0.1042         | 3.00x     |
| dropout                     | (4096, 4096)  | 0.2666      | 1.4476         | 5.43x     |
| dropout                     | (8192, 4096)  | 0.7812      | 3.2453         | 4.15x     |
| **dropout (geomean)**       | -             | -           | -              | **4.07x** |
| argmax                      | (512, 1024)   | 0.0128      | 0.0175         | 1.37x     |
| argmax                      | (4096, 4096)  | 0.1295      | 0.1860         | 1.44x     |
| argmax                      | (4096, 16384) | 0.4479      | 0.6896         | 1.54x     |
| **argmax (geomean)**        | -             | -           | -              | **1.45x** |
| matmul                      | (256, 256)    | 0.0237      | 0.0203         | 0.86x     |
| matmul                      | (1024, 1024)  | 0.2893      | 0.5279         | 1.82x     |
| matmul                      | (2048, 2048)  | 3.4944      | 3.9854         | 1.14x     |
| **matmul (geomean)**        | -             | -           | -              | **1.21x** |

### Hard tier sweep

| Operator                        | Shape            | Ours (ms)   | PyTorch (ms)   | Speedup   |
|---------------------------------|------------------|-------------|----------------|-----------|
| flash_attention                 | (1, 8, 256, 64)  | 0.0351      | 0.1784         | 5.08x     |
| flash_attention                 | (1, 8, 1024, 64) | 0.1694      | 1.7863         | 10.54x    |
| flash_attention                 | (2, 8, 2048, 64) | 1.1130      | 17.8954        | 16.08x    |
| **flash_attention (geomean)**   | -                | -           | -              | **9.52x** |
| rope                            | (256, 1, 64)     | 0.0113      | 0.0706         | 6.25x     |
| rope                            | (1024, 1, 128)   | 0.0134      | 0.0840         | 6.29x     |
| rope                            | (4096, 1, 128)   | 0.0428      | 0.0831         | 1.94x     |
| **rope (geomean)**              | -                | -           | -              | **4.24x** |
| fused_moe_topk                  | (512, 1024)      | 0.1040      | 0.1796         | 1.73x     |
| fused_moe_topk                  | (4096, 4096)     | 1.3851      | 1.5808         | 1.14x     |
| fused_moe_topk                  | (8192, 4096)     | 2.6863      | 2.7202         | 1.01x     |
| **fused_moe_topk (geomean)**    | -                | -           | -              | **1.26x** |
| rms_norm_backward               | (512, 1024)      | 0.0922      | 0.2250         | 2.44x     |
| rms_norm_backward               | (4096, 4096)     | 1.1721      | 5.2538         | 4.48x     |
| rms_norm_backward               | (4096, 8192)     | 4.1318      | 10.5011        | 2.54x     |
| **rms_norm_backward (geomean)** | -                | -           | -              | **3.03x** |
