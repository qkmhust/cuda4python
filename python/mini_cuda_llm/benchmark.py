import argparse
import time

import numpy as np

from .api import cuda_vector_add_numpy


def _timeit(fn, rounds: int) -> float:
    start = time.perf_counter()
    for _ in range(rounds):
        fn()
    end = time.perf_counter()
    return (end - start) / rounds


def run_benchmark(size: int, rounds: int, warmup: int) -> None:
    a = np.random.rand(size).astype(np.float32)
    b = np.random.rand(size).astype(np.float32)

    for _ in range(warmup):
        cuda_vector_add_numpy(a, b)
        _ = a + b

    cuda_time = _timeit(lambda: cuda_vector_add_numpy(a, b), rounds)
    numpy_time = _timeit(lambda: a + b, rounds)

    cuda_out = cuda_vector_add_numpy(a, b)
    np_out = a + b
    max_abs_diff = float(np.max(np.abs(cuda_out - np_out)))

    print(f"size={size}, rounds={rounds}, warmup={warmup}")
    print(f"CUDA avg time:  {cuda_time * 1e3:.3f} ms")
    print(f"NumPy avg time: {numpy_time * 1e3:.3f} ms")
    print(f"speedup (NumPy/CUDA): {numpy_time / cuda_time:.3f}x")
    print(f"max abs diff: {max_abs_diff:.8f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark CUDA vs NumPy vector add")
    parser.add_argument("--size", type=int, default=1_000_000, help="Vector size")
    parser.add_argument("--rounds", type=int, default=50, help="Measured rounds")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup rounds")
    args = parser.parse_args()

    run_benchmark(size=args.size, rounds=args.rounds, warmup=args.warmup)
