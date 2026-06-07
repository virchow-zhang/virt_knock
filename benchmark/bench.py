#!/usr/bin/env python
"""Benchmark GPU vs CPU performance for the core matmul in PC network construction.

Usage:  python benchmark/bench.py
"""

import time
import sys
import numpy as np
import torch

N_RUNS_GPU = 30
N_RUNS_CPU = 5


def bench_matmul(n_cells: int, n_genes: int, batch: int = 64, r: int = 13):
    """Benchmark the key matmul: (batch*r, n_cells) @ (n_cells, n_genes)."""
    gflops = 2 * (batch * r) * n_cells * n_genes / 1e9

    # ── GPU ──
    if torch.cuda.is_available():
        device = "cuda"
        prop = torch.cuda.get_device_properties(0)
        X = torch.randn(n_cells, n_genes, device=device, dtype=torch.float32)
        QT = torch.randn(batch * r, n_cells, device=device, dtype=torch.float32)

        # warmup
        for _ in range(5):
            _ = QT @ X
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(N_RUNS_GPU):
            _ = QT @ X
        torch.cuda.synchronize()
        gpu_ms = (time.perf_counter() - t0) / N_RUNS_GPU * 1000
        gpu_tflops = gflops / (gpu_ms / 1000) / 1000
        del X, QT
    else:
        device = prop = gpu_ms = gpu_tflops = None

    # ── CPU ──
    X_c = torch.randn(n_cells, n_genes, dtype=torch.float32)
    QT_c = torch.randn(batch * r, n_cells, dtype=torch.float32)

    for _ in range(3):
        _ = QT_c @ X_c

    t0 = time.perf_counter()
    for _ in range(N_RUNS_CPU):
        _ = QT_c @ X_c
    cpu_ms = (time.perf_counter() - t0) / N_RUNS_CPU * 1000
    cpu_gflops = gflops / (cpu_ms / 1000)
    del X_c, QT_c

    # ── Report ──
    print(f"\n{'='*65}")
    print(f"  Matmul: ({batch*r},{n_cells}) @ ({n_cells},{n_genes})  =  {gflops:.1f} GFLOPs")
    print(f"{'='*65}")
    if device:
        print(f"  GPU:  {prop.name}  |  {gpu_ms:6.1f} ms  |  {gpu_tflops:.2f} TFLOPS")
    print(f"  CPU:  {cpu_ms:6.0f} ms  |  {cpu_gflops:.0f} GFLOPS")
    if device:
        print(f"  {'Speedup:':>6}  {cpu_ms/gpu_ms:.0f}x")
    return cpu_ms / gpu_ms if device else 1.0


if __name__ == "__main__":
    print("virt_knock — GPU vs CPU Benchmark")
    print(f"  PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"  CUDA:    {torch.version.cuda}  |  GPU: {p.name} ({p.total_memory/1e9:.1f} GB)")
    else:
        print("  CUDA:    not available")

    # Small matrix (500 cells, typical subsample)
    bench_matmul(500, 7900)

    # Full matrix (all cells)
    su = bench_matmul(42507, 7901)

    print(f"\n{'='*65}")
    print(f"  Estimated PC network speedup (7901 genes): ~{su:.0f}x on matmul")
    print(f"  Estimated end-to-end speedup: ~5-8x")
    print(f"{'='*65}")
