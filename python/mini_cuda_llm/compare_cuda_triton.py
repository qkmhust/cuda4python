import time

import numpy as np

from .api import cuda_vector_add_numpy_advanced
from .triton_intro import triton_vector_add_numpy


def _timeit(fn, rounds: int = 20) -> float:
    start = time.perf_counter()
    for _ in range(rounds):
        _ = fn()
    end = time.perf_counter()
    return (end - start) / rounds


def run_compare(size: int = 1_000_000) -> None:
    a = np.random.rand(size).astype(np.float32)
    b = np.random.rand(size).astype(np.float32)

    cuda_out = cuda_vector_add_numpy_advanced(a, b)
    cuda_ms = _timeit(lambda: cuda_vector_add_numpy_advanced(a, b)) * 1e3

    print(f"size={size}")
    print(f"CUDA advanced: {cuda_ms:.3f} ms")

    try:
        triton_out = triton_vector_add_numpy(a, b)
        triton_ms = _timeit(lambda: triton_vector_add_numpy(a, b)) * 1e3
        max_diff = float(np.max(np.abs(cuda_out - triton_out)))

        print(f"Triton:        {triton_ms:.3f} ms")
        print(f"max abs diff (CUDA vs Triton): {max_diff:.8f}")
    except Exception as exc:
        print("Triton compare skipped:")
        print(f"  {exc}")
        print("Install optional deps: pip install triton torch")


if __name__ == "__main__":
    run_compare()
