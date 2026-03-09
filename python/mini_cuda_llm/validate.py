import numpy as np

from .api import (
    cuda_gemm_numpy,
    cuda_gemm_numpy_advanced,
    cuda_gemm_numpy_cublas,
    cuda_relu_numpy,
    cuda_softmax_numpy,
    cuda_softmax_numpy_advanced,
    cuda_vector_add,
    cuda_vector_add_numpy,
    validate_cuda_vector_add,
)


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

    relu_in = np.array([-2.0, -0.5, 0.0, 1.5], dtype=np.float32)
    relu_out = cuda_relu_numpy(relu_in)
    print("relu result:", relu_out.tolist())

    softmax_in = np.array([[1.0, 2.0, 3.0], [0.1, 0.2, 0.3]], dtype=np.float32)
    softmax_basic = cuda_softmax_numpy(softmax_in)
    softmax_adv = cuda_softmax_numpy_advanced(softmax_in)
    print("softmax basic row sums:", np.sum(softmax_basic, axis=1).tolist())
    print("softmax advanced row sums:", np.sum(softmax_adv, axis=1).tolist())

    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    gemm_basic = cuda_gemm_numpy(a, b)
    gemm_adv = cuda_gemm_numpy_advanced(a, b)
    gemm_cublas = cuda_gemm_numpy_cublas(a, b)
    gemm_np = a @ b
    print("gemm basic max diff:", float(np.max(np.abs(gemm_basic - gemm_np))))
    print("gemm advanced max diff:", float(np.max(np.abs(gemm_adv - gemm_np))))
    print("gemm cublas max diff:", float(np.max(np.abs(gemm_cublas - gemm_np))))


if __name__ == "__main__":
    run_validation()
