import argparse
import time

import numpy as np

from .api import cuda_vector_add_numpy_advanced
from .triton_intro import triton_vector_add_numpy


def _timeit(fn, rounds: int) -> float:
    start = time.perf_counter()
    for _ in range(rounds):
        _ = fn()
    end = time.perf_counter()
    return (end - start) / rounds


def _run_one(size: int, rounds: int, warmup: int) -> None:
    a = np.random.rand(size).astype(np.float32)
    b = np.random.rand(size).astype(np.float32)

    for _ in range(warmup):
        _ = a + b
        _ = cuda_vector_add_numpy_advanced(a, b)
        try:
            _ = triton_vector_add_numpy(a, b)
        except Exception:
            pass

    numpy_t = _timeit(lambda: a + b, rounds)
    cuda_t = _timeit(lambda: cuda_vector_add_numpy_advanced(a, b), rounds)

    print(f"size={size}")
    print(f"  NumPy:         {numpy_t * 1e3:.3f} ms")
    print(f"  CUDA advanced: {cuda_t * 1e3:.3f} ms")

    try:
        triton_t = _timeit(lambda: triton_vector_add_numpy(a, b), rounds)
        cuda_out = cuda_vector_add_numpy_advanced(a, b)
        triton_out = triton_vector_add_numpy(a, b)
        diff = float(np.max(np.abs(cuda_out - triton_out)))

        print(f"  Triton:        {triton_t * 1e3:.3f} ms")
        print(f"  speedup (CUDA/Triton): {cuda_t / triton_t:.3f}x")
        print(f"  max abs diff: {diff:.8f}")
    except Exception as exc:
        print("  Triton: skipped")
        print(f"    reason: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark NumPy vs CUDA vs Triton (VectorAdd)")
    parser.add_argument("--sizes", type=str, default="100000,500000,1000000")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    sizes = [int(x.strip()) for x in args.sizes.split(",") if x.strip()]
    for s in sizes:
        _run_one(size=s, rounds=args.rounds, warmup=args.warmup)
