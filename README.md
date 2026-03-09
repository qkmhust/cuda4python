# mini-cuda-llm

一个面向初学者的 CUDA 学习项目，用最小但规范的代码演示：

- CUDA Kernel 基础写法
- Host 与 Device 内存传输
- 从 `basic` 到 `advanced` 的性能优化路径
- C++ + Python（NumPy）双入口验证与基准测试

如果你想把它当作 CUDA 入门资料，这个仓库可以从第一个 Kernel 一直学到可观测的性能优化。

## 1. 学习目标

完成本项目后，你将掌握：

- 如何编写和启动一个基础 CUDA Kernel
- 如何进行 Host/Device 数据拷贝与错误处理
- 为什么基础版通常慢于 NumPy（传输开销）
- 如何通过缓存显存、异步拷贝和 Stream 提升吞吐
- 如何使用 `nvidia-smi` / `watch` 观察 GPU 运行状态

## 2. 项目亮点

- 入门友好：保留 `basic` 版本作为参考实现
- 逐步优化：新增 `advanced` 版本用于对照学习
- 结果严谨：提供验证脚本，自动做数值一致性检查
- 工程规范：CMake 构建、头文件接口、Python 包装齐全

## 3. 项目结构

```text
mini-cuda-llm/
├── CMakeLists.txt
├── include/
│   └── kernels.h
├── src/
│   ├── main.cpp
│   ├── vector_add.cu             # basic: 入门实现（清晰优先）
│   └── vector_add_advanced.cu    # advanced: 性能优化实现
├── python/
│   ├── setup.py
│   └── mini_cuda_llm/
│       ├── __init__.py
│       ├── api.py
│       ├── validate.py
│       └── benchmark.py
└── scripts/
    └── monitor_gpu.sh
```

## 4. 环境要求

- Linux
- NVIDIA GPU（已正确安装驱动）
- CUDA Toolkit（含 `nvcc`）
- CMake >= 3.18
- Python 3.8+

推荐先检查：

```bash
cmake --version
nvcc --version
nvidia-smi
python3 --version
```

## 5. 快速开始

### 5.1 构建并运行 C++ Demo

```bash
cd /root/mini-cuda-llm
cmake -S . -B build
cmake --build build -j
./build/mini_cuda_llm
```

预期看到：

- `h_c[0] = 3`
- `validation passed`

### 5.2 安装 Python 包

```bash
cd /root/mini-cuda-llm
cmake --install build --prefix /root/mini-cuda-llm
python3 -m pip install -e python
```

### 5.3 运行功能验证

```bash
python3 -m mini_cuda_llm.validate
```

### 5.4 运行性能对比

```bash
python3 -m mini_cuda_llm.benchmark --size 1000000 --rounds 30 --warmup 5
```

输出包含三组耗时：

- `CUDA basic avg time`
- `CUDA advanced avg time`
- `NumPy avg time`

并给出：

- `speedup (basic/advanced)`
- `speedup (NumPy/advanced)`
- 数值误差 `max abs diff`

## 6. Basic vs Advanced

### Basic（`src/vector_add.cu`）

设计原则：可读性优先、便于初学者理解。

特点：

- 每次调用都 `cudaMalloc/cudaFree`
- 同步 `cudaMemcpy`
- 显式 `cudaDeviceSynchronize`
- 错误处理路径清晰

适合：

- 学习 API 与执行流程
- 做第一版正确实现

### Advanced（`src/vector_add_advanced.cu`）

设计原则：在保持接口不变前提下提升性能。

特点：

- 缓存 Device 内存，减少重复分配
- 使用 `cudaStream` + `cudaMemcpyAsync`
- 提供 `releaseAdvancedResources()` 进行资源回收

适合：

- 学习基础优化手段
- 观察吞吐提升与瓶颈迁移

## 7. Python API 说明

文件：`python/mini_cuda_llm/api.py`

主要接口：

- `cuda_vector_add(a, b) -> List[float]`
- `cuda_vector_add_numpy(a, b) -> np.ndarray`（basic 路径）
- `cuda_vector_add_numpy_advanced(a, b) -> np.ndarray`（advanced 路径）
- `validate_cuda_vector_add(n=1024, expected=3.0, tolerance=1e-5) -> bool`

输入约束：

- NumPy 接口要求 1D、同形状
- 自动转为连续 `float32`

## 8. C/CUDA 接口契约

文件：`include/kernels.h`

- `int vectorAddHost(const float* a, const float* b, float* c, int n)`
- `int vectorAddHostAdvanced(const float* a, const float* b, float* c, int n)`
- `int validateVectorAdd(int n, float expected, float tolerance)`
- `int releaseAdvancedResources()`

返回值规范：

- `0`：成功
- `-1`：验证失败（仅 `validateVectorAdd`）
- 其他：CUDA 错误码（可对照 `cudaError_t`）

## 9. GPU 监控（watch / nvidia-smi）

### 实时监控（交互）

```bash
cd /root/mini-cuda-llm
./scripts/monitor_gpu.sh
```

该脚本等价于：

```bash
watch -n 1 nvidia-smi
```

### 日志采样（非交互）

```bash
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv -l 1
```

建议用法：

1. 开一个终端运行监控。
2. 另一个终端跑 benchmark。
3. 对比 `basic` 和 `advanced` 的 GPU 利用率与显存占用变化。

## 10. 常见问题排查

### 1) `cuda_runtime.h: No such file or directory`

通常是 CUDA include/link 未正确配置。确认：

- CMake 中使用 `find_package(CUDAToolkit REQUIRED)`
- 目标链接 `CUDA::cudart`

### 2) Python 报找不到 `.so`

先执行：

```bash
cmake --install build --prefix /root/mini-cuda-llm
python3 -m pip install -e python
```

### 3) `advanced` 不一定比 NumPy 快

这很常见。小规模数据时，Host/Device 拷贝成本可能大于 GPU 计算收益。

## 11. 学习路线建议

1. 先跑通 `main.cpp`，理解端到端执行。
2. 阅读 `vector_add.cu`，掌握基础 API。
3. 跑 `validate.py`，建立正确性意识。
4. 跑 `benchmark.py`，观察性能数据。
5. 对照 `vector_add_advanced.cu`，理解每个优化点。
6. 打开 GPU 监控，形成“代码-数据-硬件指标”闭环。

## 12. 后续可扩展方向

- 使用 pinned memory（页锁定内存）降低传输开销
- 使用多 stream 做拷贝与计算重叠
- 引入 batched 接口，减少 Python 调用开销
- 增加更多算子（axpy、matmul）形成系统化练习

## 13. 深度学习算子进阶路径

本仓库现在已覆盖以下梯度：

1. `VectorAdd`：理解 CUDA 基础执行模型。
2. `ReLU`：最常见激活函数，学习一元逐元素算子写法。
3. `Softmax basic`：一行一个线程，逻辑直观，便于理解数值稳定性。
4. `Softmax advanced`：一行一个 block，使用 shared memory 做并行归约。

对应 benchmark：

```bash
python3 -m mini_cuda_llm.benchmark_dl_ops --rows 1024 --cols 1024 --rounds 30 --warmup 5
```

下一步建议按这个顺序继续深入：

1. LayerNorm（先单样本，再 batched）
2. GEMM（先 naive，再 tiled shared memory）
3. Attention 的 `QK^T` 和 Softmax 融合

---

如果你正在教学或自学 CUDA，这个仓库推荐作为“第一份可跑、可测、可优化、可观测”的模板。
