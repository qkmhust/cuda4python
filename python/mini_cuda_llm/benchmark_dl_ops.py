import argparse
import time

import numpy as np

from .api import (
    cuda_gemm_numpy,
    cuda_gemm_numpy_advanced,
    cuda_gemm_numpy_cublas,
    cuda_relu_numpy,
    cuda_softmax_numpy,
    cuda_softmax_numpy_advanced,
)


def _timeit(fn, rounds: int) -> float:
    start = time.perf_counter()
    for _ in range(rounds):
        fn()
    end = time.perf_counter()
    return (end - start) / rounds


def _softmax_numpy(x: np.ndarray) -> np.ndarray:
    x_max = np.max(x, axis=1, keepdims=True)
    ex = np.exp(x - x_max)
    return ex / np.sum(ex, axis=1, keepdims=True)


def run_benchmark(rows: int, cols: int, rounds: int, warmup: int) -> None:
    x2d = np.random.randn(rows, cols).astype(np.float32)
    x1d = x2d.reshape(-1)
    a = np.random.randn(rows, cols).astype(np.float32)
    b = np.random.randn(cols, rows).astype(np.float32)

    for _ in range(warmup):
        _ = cuda_relu_numpy(x1d)
        _ = np.maximum(x1d, 0.0)
        _ = cuda_softmax_numpy(x2d)
        _ = cuda_softmax_numpy_advanced(x2d)
        _ = _softmax_numpy(x2d)
        _ = cuda_gemm_numpy(a, b)
        _ = cuda_gemm_numpy_advanced(a, b)
        _ = cuda_gemm_numpy_cublas(a, b)
        _ = a @ b

    relu_cuda_t = _timeit(lambda: cuda_relu_numpy(x1d), rounds)
    relu_np_t = _timeit(lambda: np.maximum(x1d, 0.0), rounds)

    softmax_basic_t = _timeit(lambda: cuda_softmax_numpy(x2d), rounds)
    softmax_adv_t = _timeit(lambda: cuda_softmax_numpy_advanced(x2d), rounds)
    softmax_np_t = _timeit(lambda: _softmax_numpy(x2d), rounds)

    gemm_basic_t = _timeit(lambda: cuda_gemm_numpy(a, b), rounds)
    gemm_adv_t = _timeit(lambda: cuda_gemm_numpy_advanced(a, b), rounds)
    gemm_cublas_t = _timeit(lambda: cuda_gemm_numpy_cublas(a, b), rounds)
    gemm_np_t = _timeit(lambda: a @ b, rounds)

    relu_diff = float(np.max(np.abs(cuda_relu_numpy(x1d) - np.maximum(x1d, 0.0))))
    softmax_basic_diff = float(np.max(np.abs(cuda_softmax_numpy(x2d) - _softmax_numpy(x2d))))
    softmax_adv_diff = float(np.max(np.abs(cuda_softmax_numpy_advanced(x2d) - _softmax_numpy(x2d))))
    gemm_basic_diff = float(np.max(np.abs(cuda_gemm_numpy(a, b) - (a @ b))))
    gemm_adv_diff = float(np.max(np.abs(cuda_gemm_numpy_advanced(a, b) - (a @ b))))
    gemm_cublas_diff = float(np.max(np.abs(cuda_gemm_numpy_cublas(a, b) - (a @ b))))

    print(f"shape=[{rows}, {cols}], rounds={rounds}, warmup={warmup}")
    print("[ReLU]")
    print(f"CUDA:  {relu_cuda_t * 1e3:.3f} ms")
    print(f"NumPy: {relu_np_t * 1e3:.3f} ms")
    print(f"max abs diff: {relu_diff:.8f}")
    print("[Softmax]")
    print(f"CUDA basic:    {softmax_basic_t * 1e3:.3f} ms")
    print(f"CUDA advanced: {softmax_adv_t * 1e3:.3f} ms")
    print(f"NumPy:         {softmax_np_t * 1e3:.3f} ms")
    print(f"speedup (basic/advanced): {softmax_basic_t / softmax_adv_t:.3f}x")
    print(f"max abs diff basic: {softmax_basic_diff:.8f}")
    print(f"max abs diff advanced: {softmax_adv_diff:.8f}")
    print("[GEMM]")
    print(f"CUDA basic:    {gemm_basic_t * 1e3:.3f} ms")
    print(f"CUDA advanced: {gemm_adv_t * 1e3:.3f} ms")
    print(f"CUDA cuBLAS:   {gemm_cublas_t * 1e3:.3f} ms")
    print(f"NumPy:         {gemm_np_t * 1e3:.3f} ms")
    print(f"speedup (basic/advanced): {gemm_basic_t / gemm_adv_t:.3f}x")
    print(f"speedup (advanced/cuBLAS): {gemm_adv_t / gemm_cublas_t:.3f}x")
    print(f"speedup (basic/cuBLAS): {gemm_basic_t / gemm_cublas_t:.3f}x")
    print(f"max abs diff basic: {gemm_basic_diff:.8f}")
    print(f"max abs diff advanced: {gemm_adv_diff:.8f}")
    print(f"max abs diff cuBLAS: {gemm_cublas_diff:.8f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark CUDA DL operators (ReLU/Softmax)")
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--cols", type=int, default=1024)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    run_benchmark(rows=args.rows, cols=args.cols, rounds=args.rounds, warmup=args.warmup)
