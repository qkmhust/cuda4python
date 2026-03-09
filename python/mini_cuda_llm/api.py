import ctypes
import atexit
import os
from typing import Iterable, List

import numpy as np


_LIB_NAME = "libmini_cuda_llm_cuda.so"
_THIS_DIR = os.path.dirname(__file__)
_LIB_PATH = os.path.join(_THIS_DIR, _LIB_NAME)


class _CudaLib:
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


_cuda = _CudaLib()


def _cleanup() -> None:
    _cuda.lib.releaseAdvancedResources()


atexit.register(_cleanup)


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
    if a.shape != b.shape:
        raise ValueError("Input arrays must have the same shape")
    if a.ndim != 1:
        raise ValueError("Only 1D arrays are supported")
    if a.size == 0:
        return np.array([], dtype=np.float32)

    a_arr = np.ascontiguousarray(a, dtype=np.float32)
    b_arr = np.ascontiguousarray(b, dtype=np.float32)
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
    if a.shape != b.shape:
        raise ValueError("Input arrays must have the same shape")
    if a.ndim != 1:
        raise ValueError("Only 1D arrays are supported")
    if a.size == 0:
        return np.array([], dtype=np.float32)

    a_arr = np.ascontiguousarray(a, dtype=np.float32)
    b_arr = np.ascontiguousarray(b, dtype=np.float32)
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
