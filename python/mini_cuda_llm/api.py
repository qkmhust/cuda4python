import ctypes
import atexit
import os
from typing import Iterable, List

import numpy as np


_LIB_NAME = "libmini_cuda_llm_cuda.so"
_THIS_DIR = os.path.dirname(__file__)
_LIB_PATH = os.path.join(_THIS_DIR, _LIB_NAME)


class _CudaLib:
    # 统一管理 ctypes 函数签名，避免每次调用重复设置。
    def __init__(self) -> None:
        if not os.path.exists(_LIB_PATH):
            raise FileNotFoundError(
                f"CUDA library not found at {_LIB_PATH}. Run CMake build and install first."
            )
        self.lib = ctypes.CDLL(_LIB_PATH)
        self.lib.vectorAddHost.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
        ]
        self.lib.vectorAddHost.restype = ctypes.c_int

        self.lib.validateVectorAdd.argtypes = [ctypes.c_int, ctypes.c_float, ctypes.c_float]
        self.lib.validateVectorAdd.restype = ctypes.c_int

        self.lib.vectorAddHostAdvanced.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
        ]
        self.lib.vectorAddHostAdvanced.restype = ctypes.c_int

        self.lib.releaseAdvancedResources.argtypes = []
        self.lib.releaseAdvancedResources.restype = ctypes.c_int

        self.lib.reluHost.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
        ]
        self.lib.reluHost.restype = ctypes.c_int

        self.lib.softmaxHost.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.lib.softmaxHost.restype = ctypes.c_int

        self.lib.softmaxHostAdvanced.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.lib.softmaxHostAdvanced.restype = ctypes.c_int

        self.lib.gemmHost.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.lib.gemmHost.restype = ctypes.c_int

        self.lib.gemmHostAdvanced.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.lib.gemmHostAdvanced.restype = ctypes.c_int

        self.lib.gemmHostCublas.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.lib.gemmHostCublas.restype = ctypes.c_int

        self.lib.releaseGemmResources.argtypes = []
        self.lib.releaseGemmResources.restype = ctypes.c_int


_cuda = _CudaLib()


def _cleanup() -> None:
    # 释放 advanced/cublas 路径缓存资源，防止进程退出时遗留。
    _cuda.lib.releaseAdvancedResources()
    _cuda.lib.releaseGemmResources()


atexit.register(_cleanup)


def _ensure_1d_pair(a: np.ndarray, b: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    if a.shape != b.shape:
        raise ValueError(f"{name}: input arrays must have the same shape")
    if a.ndim != 1:
        raise ValueError(f"{name}: only 1D arrays are supported")
    a_arr = np.ascontiguousarray(a, dtype=np.float32)
    b_arr = np.ascontiguousarray(b, dtype=np.float32)
    return a_arr, b_arr


def _ensure_2d(x: np.ndarray, name: str) -> tuple[np.ndarray, int, int]:
    if x.ndim != 2:
        raise ValueError(f"{name}: input must be a 2D array [rows, cols]")
    rows, cols = x.shape
    x_arr = np.ascontiguousarray(x, dtype=np.float32)
    return x_arr, int(rows), int(cols)


def _ensure_gemm_inputs(a: np.ndarray, b: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"{name}: GEMM inputs must be 2D arrays")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"{name}: GEMM shape mismatch, a.shape[1] must equal b.shape[0]")
    a_arr = np.ascontiguousarray(a, dtype=np.float32)
    b_arr = np.ascontiguousarray(b, dtype=np.float32)
    m, k = a_arr.shape
    _, n = b_arr.shape
    return a_arr, b_arr, int(m), int(n), int(k)


def cuda_vector_add(a: Iterable[float], b: Iterable[float]) -> List[float]:
    a_list = list(a)
    b_list = list(b)
    if len(a_list) != len(b_list):
        raise ValueError("Input vectors must have the same length")
    if not a_list:
        return []

    n = len(a_list)
    arr_t = ctypes.c_float * n
    a_arr = arr_t(*a_list)
    b_arr = arr_t(*b_list)
    c_arr = arr_t(*([0.0] * n))

    code = _cuda.lib.vectorAddHost(a_arr, b_arr, c_arr, n)
    if code != 0:
        raise RuntimeError(f"vectorAddHost failed, cuda code={code}")

    return [c_arr[i] for i in range(n)]


def validate_cuda_vector_add(n: int = 1024, expected: float = 3.0, tolerance: float = 1e-5) -> bool:
    code = _cuda.lib.validateVectorAdd(n, expected, tolerance)
    if code == 0:
        return True
    if code == -1:
        return False
    raise RuntimeError(f"validateVectorAdd failed, cuda code={code}")


def cuda_vector_add_numpy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # NumPy 入口：做输入检查 + 连续化，再交给底层 CUDA C API。
    a_arr, b_arr = _ensure_1d_pair(a, b, "cuda_vector_add_numpy")
    if a_arr.size == 0:
        return np.array([], dtype=np.float32)
    c_arr = np.empty_like(a_arr)

    n = int(a_arr.size)
    code = _cuda.lib.vectorAddHost(
        a_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        b_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        c_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        n,
    )
    if code != 0:
        raise RuntimeError(f"vectorAddHost failed, cuda code={code}")

    return c_arr


def cuda_vector_add_numpy_advanced(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_arr, b_arr = _ensure_1d_pair(a, b, "cuda_vector_add_numpy_advanced")
    if a_arr.size == 0:
        return np.array([], dtype=np.float32)
    c_arr = np.empty_like(a_arr)

    n = int(a_arr.size)
    code = _cuda.lib.vectorAddHostAdvanced(
        a_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        b_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        c_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        n,
    )
    if code != 0:
        raise RuntimeError(f"vectorAddHostAdvanced failed, cuda code={code}")

    return c_arr


def cuda_relu_numpy(x: np.ndarray) -> np.ndarray:
    if x.ndim != 1:
        raise ValueError("ReLU input must be a 1D array")
    if x.size == 0:
        return np.array([], dtype=np.float32)

    x_arr = np.ascontiguousarray(x, dtype=np.float32)
    y_arr = np.empty_like(x_arr)

    n = int(x_arr.size)
    code = _cuda.lib.reluHost(
        x_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        y_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        n,
    )
    if code != 0:
        raise RuntimeError(f"reluHost failed, cuda code={code}")
    return y_arr


def cuda_softmax_numpy(x: np.ndarray) -> np.ndarray:
    x_arr, rows, cols = _ensure_2d(x, "cuda_softmax_numpy")
    if rows == 0 or cols == 0:
        return np.empty((rows, cols), dtype=np.float32)
    y_arr = np.empty_like(x_arr)

    code = _cuda.lib.softmaxHost(
        x_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        y_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        int(rows),
        int(cols),
    )
    if code != 0:
        raise RuntimeError(f"softmaxHost failed, cuda code={code}")
    return y_arr


def cuda_softmax_numpy_advanced(x: np.ndarray) -> np.ndarray:
    # advanced softmax 仅替换底层实现，保持与 basic 相同接口。
    x_arr, rows, cols = _ensure_2d(x, "cuda_softmax_numpy_advanced")
    if rows == 0 or cols == 0:
        return np.empty((rows, cols), dtype=np.float32)
    y_arr = np.empty_like(x_arr)

    code = _cuda.lib.softmaxHostAdvanced(
        x_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        y_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        int(rows),
        int(cols),
    )
    if code != 0:
        raise RuntimeError(f"softmaxHostAdvanced failed, cuda code={code}")
    return y_arr


def cuda_gemm_numpy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_arr, b_arr, m, n, k = _ensure_gemm_inputs(a, b, "cuda_gemm_numpy")
    if m == 0 or n == 0 or k == 0:
        return np.empty((m, n), dtype=np.float32)
    c_arr = np.empty((m, n), dtype=np.float32)

    code = _cuda.lib.gemmHost(
        a_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        b_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        c_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        int(m),
        int(n),
        int(k),
    )
    if code != 0:
        raise RuntimeError(f"gemmHost failed, cuda code={code}")
    return c_arr


def cuda_gemm_numpy_advanced(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_arr, b_arr, m, n, k = _ensure_gemm_inputs(a, b, "cuda_gemm_numpy_advanced")
    if m == 0 or n == 0 or k == 0:
        return np.empty((m, n), dtype=np.float32)
    c_arr = np.empty((m, n), dtype=np.float32)

    code = _cuda.lib.gemmHostAdvanced(
        a_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        b_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        c_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        int(m),
        int(n),
        int(k),
    )
    if code != 0:
        raise RuntimeError(f"gemmHostAdvanced failed, cuda code={code}")
    return c_arr


def cuda_gemm_numpy_cublas(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # 工程路径：通过 cuBLAS 调用高性能 SGEMM。
    a_arr, b_arr, m, n, k = _ensure_gemm_inputs(a, b, "cuda_gemm_numpy_cublas")
    if m == 0 or n == 0 or k == 0:
        return np.empty((m, n), dtype=np.float32)
    c_arr = np.empty((m, n), dtype=np.float32)

    code = _cuda.lib.gemmHostCublas(
        a_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        b_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        c_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        int(m),
        int(n),
        int(k),
    )
    if code != 0:
        raise RuntimeError(f"gemmHostCublas failed, cuda code={code}")
    return c_arr
