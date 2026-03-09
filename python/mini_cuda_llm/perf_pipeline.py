import argparse
import csv
import json
import os
import time
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .api import (
    cuda_gemm_numpy,
    cuda_gemm_numpy_advanced,
    cuda_gemm_numpy_cublas,
    cuda_softmax_numpy,
    cuda_softmax_numpy_advanced,
    cuda_vector_add_numpy,
    cuda_vector_add_numpy_advanced,
)


def _timeit(fn: Callable[[], np.ndarray], rounds: int) -> float:
    start = time.perf_counter()
    for _ in range(rounds):
        _ = fn()
    end = time.perf_counter()
    return (end - start) / rounds


def _softmax_numpy(x: np.ndarray) -> np.ndarray:
    x_max = np.max(x, axis=1, keepdims=True)
    ex = np.exp(x - x_max)
    return ex / np.sum(ex, axis=1, keepdims=True)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def run_vector_suite(sizes: List[int], rounds: int, warmup: int) -> List[Dict[str, float]]:
    rows = []
    for size in sizes:
        a = np.random.rand(size).astype(np.float32)
        b = np.random.rand(size).astype(np.float32)

        for _ in range(warmup):
            _ = cuda_vector_add_numpy(a, b)
            _ = cuda_vector_add_numpy_advanced(a, b)
            _ = a + b

        t_basic = _timeit(lambda: cuda_vector_add_numpy(a, b), rounds)
        t_adv = _timeit(lambda: cuda_vector_add_numpy_advanced(a, b), rounds)
        t_np = _timeit(lambda: a + b, rounds)

        out_basic = cuda_vector_add_numpy(a, b)
        out_adv = cuda_vector_add_numpy_advanced(a, b)
        out_np = a + b

        max_diff_basic = float(np.max(np.abs(out_basic - out_np)))
        max_diff_adv = float(np.max(np.abs(out_adv - out_np)))

        # Effective bytes for vector add: read a,b and write c (3 * n * 4 bytes)
        gb_basic = (size * 3 * 4) / t_basic / 1e9
        gb_adv = (size * 3 * 4) / t_adv / 1e9

        rows.append(
            {
                "size": float(size),
                "latency_ms_basic": t_basic * 1e3,
                "latency_ms_advanced": t_adv * 1e3,
                "latency_ms_numpy": t_np * 1e3,
                "speedup_basic_over_adv": t_basic / t_adv,
                "speedup_numpy_over_adv": t_np / t_adv,
                "diff_basic": max_diff_basic,
                "diff_advanced": max_diff_adv,
                "gbps_basic": gb_basic,
                "gbps_advanced": gb_adv,
            }
        )
    return rows


def run_dl_suite(shapes: List[Tuple[int, int, int]], rounds: int, warmup: int) -> List[Dict[str, float]]:
    rows = []
    for m, n, k in shapes:
        x = np.random.randn(m, n).astype(np.float32)
        a = np.random.randn(m, k).astype(np.float32)
        b = np.random.randn(k, n).astype(np.float32)

        for _ in range(warmup):
            _ = cuda_softmax_numpy(x)
            _ = cuda_softmax_numpy_advanced(x)
            _ = _softmax_numpy(x)
            _ = cuda_gemm_numpy(a, b)
            _ = cuda_gemm_numpy_advanced(a, b)
            _ = cuda_gemm_numpy_cublas(a, b)
            _ = a @ b

        t_soft_basic = _timeit(lambda: cuda_softmax_numpy(x), rounds)
        t_soft_adv = _timeit(lambda: cuda_softmax_numpy_advanced(x), rounds)
        t_soft_np = _timeit(lambda: _softmax_numpy(x), rounds)

        t_gemm_basic = _timeit(lambda: cuda_gemm_numpy(a, b), rounds)
        t_gemm_adv = _timeit(lambda: cuda_gemm_numpy_advanced(a, b), rounds)
        t_gemm_cublas = _timeit(lambda: cuda_gemm_numpy_cublas(a, b), rounds)
        t_gemm_np = _timeit(lambda: a @ b, rounds)

        ref_soft = _softmax_numpy(x)
        diff_soft_basic = float(np.max(np.abs(cuda_softmax_numpy(x) - ref_soft)))
        diff_soft_adv = float(np.max(np.abs(cuda_softmax_numpy_advanced(x) - ref_soft)))

        ref_gemm = a @ b
        diff_gemm_basic = float(np.max(np.abs(cuda_gemm_numpy(a, b) - ref_gemm)))
        diff_gemm_adv = float(np.max(np.abs(cuda_gemm_numpy_advanced(a, b) - ref_gemm)))
        diff_gemm_cublas = float(np.max(np.abs(cuda_gemm_numpy_cublas(a, b) - ref_gemm)))

        flops = 2.0 * m * n * k
        gflops_basic = flops / t_gemm_basic / 1e9
        gflops_adv = flops / t_gemm_adv / 1e9
        gflops_cublas = flops / t_gemm_cublas / 1e9

        rows.append(
            {
                "m": float(m),
                "n": float(n),
                "k": float(k),
                "softmax_ms_basic": t_soft_basic * 1e3,
                "softmax_ms_advanced": t_soft_adv * 1e3,
                "softmax_ms_numpy": t_soft_np * 1e3,
                "softmax_speedup_basic_over_adv": t_soft_basic / t_soft_adv,
                "softmax_diff_basic": diff_soft_basic,
                "softmax_diff_advanced": diff_soft_adv,
                "gemm_ms_basic": t_gemm_basic * 1e3,
                "gemm_ms_advanced": t_gemm_adv * 1e3,
                "gemm_ms_cublas": t_gemm_cublas * 1e3,
                "gemm_ms_numpy": t_gemm_np * 1e3,
                "gemm_speedup_basic_over_cublas": t_gemm_basic / t_gemm_cublas,
                "gemm_speedup_advanced_over_cublas": t_gemm_adv / t_gemm_cublas,
                "gemm_diff_basic": diff_gemm_basic,
                "gemm_diff_advanced": diff_gemm_adv,
                "gemm_diff_cublas": diff_gemm_cublas,
                "gemm_gflops_basic": gflops_basic,
                "gemm_gflops_advanced": gflops_adv,
                "gemm_gflops_cublas": gflops_cublas,
            }
        )
    return rows


def _write_csv(path: str, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(path: str, vector_rows: List[Dict[str, float]], dl_rows: List[Dict[str, float]]) -> None:
    best_vec = max(vector_rows, key=lambda r: r["speedup_basic_over_adv"])
    best_gemm = max(dl_rows, key=lambda r: r["gemm_speedup_basic_over_cublas"])

    lines = [
        "# Performance Summary",
        "",
        "## VectorAdd",
        f"- Best advanced gain over basic: {best_vec['speedup_basic_over_adv']:.3f}x at size={int(best_vec['size'])}",
        f"- Max advanced bandwidth: {max(r['gbps_advanced'] for r in vector_rows):.3f} GB/s",
        "",
        "## GEMM",
        f"- Best cuBLAS gain over basic: {best_gemm['gemm_speedup_basic_over_cublas']:.3f}x at shape=({int(best_gemm['m'])},{int(best_gemm['n'])},{int(best_gemm['k'])})",
        f"- Max cuBLAS throughput: {max(r['gemm_gflops_cublas'] for r in dl_rows):.3f} GFLOPS",
        "",
        "## Accuracy",
        f"- VectorAdd max abs diff (advanced): {max(r['diff_advanced'] for r in vector_rows):.8f}",
        f"- Softmax max abs diff (advanced): {max(r['softmax_diff_advanced'] for r in dl_rows):.8f}",
        f"- GEMM max abs diff (cuBLAS): {max(r['gemm_diff_cublas'] for r in dl_rows):.8f}",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _plot(vector_rows: List[Dict[str, float]], dl_rows: List[Dict[str, float]], out_dir: str) -> None:
    sizes = [int(r["size"]) for r in vector_rows]
    vb = [r["latency_ms_basic"] for r in vector_rows]
    va = [r["latency_ms_advanced"] for r in vector_rows]
    vn = [r["latency_ms_numpy"] for r in vector_rows]

    shapes = [f"{int(r['m'])}" for r in dl_rows]
    gb = [r["gemm_ms_basic"] for r in dl_rows]
    ga = [r["gemm_ms_advanced"] for r in dl_rows]
    gc = [r["gemm_ms_cublas"] for r in dl_rows]

    sg = [r["gemm_speedup_basic_over_cublas"] for r in dl_rows]

    plt.figure(figsize=(14, 4))

    plt.subplot(1, 3, 1)
    plt.plot(sizes, vb, marker="o", label="Vector basic")
    plt.plot(sizes, va, marker="o", label="Vector advanced")
    plt.plot(sizes, vn, marker="o", label="Vector NumPy")
    plt.xscale("log")
    plt.xlabel("Vector Size")
    plt.ylabel("Latency (ms)")
    plt.title("VectorAdd Latency")
    plt.legend()

    plt.subplot(1, 3, 2)
    x = np.arange(len(shapes))
    width = 0.25
    plt.bar(x - width, gb, width=width, label="GEMM basic")
    plt.bar(x, ga, width=width, label="GEMM advanced")
    plt.bar(x + width, gc, width=width, label="GEMM cuBLAS")
    plt.xticks(x, shapes)
    plt.xlabel("Matrix Size (m=n=k)")
    plt.ylabel("Latency (ms)")
    plt.title("GEMM Latency")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.bar(shapes, sg)
    plt.xlabel("Matrix Size (m=n=k)")
    plt.ylabel("Speedup (basic/cuBLAS)")
    plt.title("GEMM Speedup")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "performance_overview.png"), dpi=160)
    plt.close()


def run_pipeline(out_dir: str, rounds: int, warmup: int) -> None:
    _ensure_dir(out_dir)

    vector_sizes = [100_000, 500_000, 1_000_000, 5_000_000]
    dl_shapes = [(256, 256, 256), (512, 512, 512), (1024, 1024, 1024)]

    vector_rows = run_vector_suite(vector_sizes, rounds=rounds, warmup=warmup)
    dl_rows = run_dl_suite(dl_shapes, rounds=rounds, warmup=warmup)

    _write_csv(os.path.join(out_dir, "vector_add.csv"), vector_rows)
    _write_csv(os.path.join(out_dir, "dl_ops.csv"), dl_rows)

    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"vector": vector_rows, "dl_ops": dl_rows}, f, indent=2)

    _write_summary(os.path.join(out_dir, "summary.md"), vector_rows, dl_rows)
    _plot(vector_rows, dl_rows, out_dir)

    print(f"Pipeline finished. Results saved to: {out_dir}")
    print("Generated files: vector_add.csv, dl_ops.csv, results.json, summary.md, performance_overview.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full benchmark pipeline with visualization")
    parser.add_argument("--out", type=str, default="/root/mini-cuda-llm/reports/latest")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    run_pipeline(out_dir=args.out, rounds=args.rounds, warmup=args.warmup)
