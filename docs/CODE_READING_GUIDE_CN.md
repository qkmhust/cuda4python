# 代码阅读指南（中文）

这份指南帮助你按难度逐步阅读项目，实现“看得懂 -> 跑得通 -> 会分析 -> 能扩展”。

## 阶段 0：先跑通项目

1. 编译与安装
```bash
cd /root/mini-cuda-llm
cmake -S . -B build
cmake --build build -j
cmake --install build --prefix /root/mini-cuda-llm
python3 -m pip install -e python
```
2. 功能验证
```bash
python3 -m mini_cuda_llm.validate
```
3. 性能测试
```bash
python3 -m mini_cuda_llm.benchmark_dl_ops --rows 1024 --cols 1024 --rounds 20 --warmup 5
```

## 阶段 1：理解接口总览（先看这个）

阅读文件：`include/kernels.h`

目标：先知道项目暴露了哪些 API，再进入实现细节。

- VectorAdd：`vectorAddHost` / `vectorAddHostAdvanced`
- Softmax：`softmaxHost` / `softmaxHostAdvanced`
- GEMM：`gemmHost` / `gemmHostAdvanced` / `gemmHostCublas`

## 阶段 2：读基础版数据流

阅读顺序：

1. `src/vector_add.cu`
2. `python/mini_cuda_llm/api.py` 中 `cuda_vector_add_numpy`

你应该重点看：

- Host 输入检查
- `cudaMalloc` / `cudaMemcpy` / kernel 启动 / `cudaDeviceSynchronize`
- 错误码如何向上返回到 Python

## 阶段 3：读 first optimization（advanced）

阅读顺序：

1. `src/vector_add_advanced.cu`
2. `api.py` 中 `cuda_vector_add_numpy_advanced`

你应该重点看：

- 为什么要缓存 `g_d_a/g_d_b/g_d_c`
- 为什么引入 `cudaStream`
- `releaseAdvancedResources` 在生命周期中的作用

## 阶段 4：进入 DL 算子（ReLU / Softmax）

阅读顺序：

1. `src/dl_ops.cu`
2. `api.py` 中 `cuda_relu_numpy` / `cuda_softmax_numpy(_advanced)`

你应该重点看：

- ReLU 的逐元素并行模式
- Softmax 数值稳定性（先减 max）
- basic 与 advanced 在并行粒度上的区别（线程级 vs block 级）

## 阶段 5：进入 GEMM 核心

阅读顺序：

1. `src/gemm.cu` 中 `gemmKernelBasic`
2. `src/gemm.cu` 中 `gemmKernelTiled`
3. `src/gemm.cu` 中 `gemmHostCublas`

你应该重点看：

- 朴素 GEMM 的访存模式
- tiled GEMM 如何用 shared memory 降低全局访存
- cuBLAS 为什么显著更快（工程级优化）

## 阶段 6：学会分析结果

阅读顺序：

1. `python/mini_cuda_llm/benchmark.py`
2. `python/mini_cuda_llm/benchmark_dl_ops.py`
3. `python/mini_cuda_llm/perf_pipeline.py`

你应该重点看：

- 延迟、加速比、误差如何计算
- GB/s 与 GFLOPS 的含义
- 为什么 advanced 不一定总是更快（端到端开销）

## 阶段 7：配合监控闭环

运行：
```bash
./scripts/monitor_gpu.sh
```

结合 benchmark 观察：

- GPU 利用率变化
- 显存占用变化
- 不同算子、不同规模下的行为差异

## 建议学习节奏

1. 每读完一个阶段，先运行对应脚本验证。
2. 每次只改一个点（例如线程块大小），再复测。
3. 用 `reports/latest/summary.md` 记录你的结论。

这样你会真正建立从“代码到性能”的因果理解。
