import numpy as np

from .api import cuda_vector_add, cuda_vector_add_numpy, validate_cuda_vector_add


def run_validation() -> None:
    result = cuda_vector_add([1.0, 2.0, 3.0], [2.0, 3.0, 4.0])
    print("vector add result:", result)

    np_result = cuda_vector_add_numpy(
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
        np.array([2.0, 3.0, 4.0], dtype=np.float32),
    )
    print("numpy vector add result:", np_result.tolist())

    ok = validate_cuda_vector_add(1024, 3.0, 1e-5)
    print("bulk validation:", "passed" if ok else "failed")


if __name__ == "__main__":
    run_validation()
