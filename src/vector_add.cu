#include <cuda_runtime.h>

#include <cmath>
#include <vector>

#include "kernels.h"

// 基础 CUDA kernel：每个线程处理一个元素。
__global__ void vectorAddKernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// 仅负责 kernel 启动配置，便于和 host 逻辑解耦。
void launchVectorAdd(const float* a, const float* b, float* c, int n) {
    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    vectorAddKernel<<<blocks, threadsPerBlock>>>(a, b, c, n);
}

// 基础路径（教学版）：单次调用内完整执行分配、拷贝、计算、回传。
int vectorAddHost(const float* a, const float* b, float* c, int n) {
    if (a == nullptr || b == nullptr || c == nullptr || n <= 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;

    cudaError_t err = cudaMalloc(&d_a, bytes);
    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    err = cudaMalloc(&d_b, bytes);
    if (err != cudaSuccess) {
        cudaFree(d_a);
        return static_cast<int>(err);
    }

    err = cudaMalloc(&d_c, bytes);
    if (err != cudaSuccess) {
        cudaFree(d_a);
        cudaFree(d_b);
        return static_cast<int>(err);
    }

    // 使用同步拷贝，流程更直观：H2D -> Kernel -> D2H。
    err = cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        return static_cast<int>(err);
    }
    err = cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        return static_cast<int>(err);
    }

    launchVectorAdd(d_a, d_b, d_c, n);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        return static_cast<int>(err);
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        return static_cast<int>(err);
    }

    err = cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return static_cast<int>(err);
}

// 功能自检：构造固定输入并逐元素验证输出是否在容差内。
int validateVectorAdd(int n, float expected, float tolerance) {
    if (n <= 0 || tolerance < 0.0f) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    std::vector<float> a(n, 1.0f);
    std::vector<float> b(n, 2.0f);
    std::vector<float> c(n, 0.0f);

    int code = vectorAddHost(a.data(), b.data(), c.data(), n);
    if (code != static_cast<int>(cudaSuccess)) {
        return code;
    }

    for (int i = 0; i < n; ++i) {
        if (std::fabs(c[i] - expected) > tolerance) {
            return -1;
        }
    }

    return 0;
}
