"""Triton 入门示例：1D 向量加法。

这个模块是可选能力：
- 如果环境安装了 Triton，会调用 Triton kernel。
- 如果未安装，会抛出清晰错误，提示如何安装。
"""

from __future__ import annotations

import numpy as np

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover - optional dependency
    triton = None
    tl = None
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _vec_add_kernel(
        x_ptr,
        y_ptr,
        out_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        out = x + y
        tl.store(out_ptr + offsets, out, mask=mask)


def triton_vector_add_numpy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """使用 Triton 对两个 1D float32 NumPy 向量做加法。"""
    if not _HAS_TRITON:
        raise RuntimeError(
            "Triton is not installed. Install with: pip install triton"
        )

    if a.shape != b.shape:
        raise ValueError("triton_vector_add_numpy: input arrays must have the same shape")
    if a.ndim != 1:
        raise ValueError("triton_vector_add_numpy: only 1D arrays are supported")

    a_arr = np.ascontiguousarray(a, dtype=np.float32)
    b_arr = np.ascontiguousarray(b, dtype=np.float32)
    n = a_arr.size

    if n == 0:
        return np.array([], dtype=np.float32)

    # Triton 直接操作 GPU tensor，使用 torch 作为最小桥接层。
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is not available for Triton demo")

    x = torch.from_numpy(a_arr).to(device="cuda")
    y = torch.from_numpy(b_arr).to(device="cuda")
    out = torch.empty_like(x)

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _vec_add_kernel[grid](x, y, out, n, BLOCK_SIZE=1024)

    return out.cpu().numpy()
