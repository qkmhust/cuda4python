#include <cuda_runtime.h>

#include "kernels.h"

namespace {
float* g_d_a = nullptr;
float* g_d_b = nullptr;
float* g_d_c = nullptr;
int g_capacity = 0;
cudaStream_t g_stream = nullptr;

__global__ void vectorAddKernelAdvanced(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int ensureCapacityAdvanced(int n) {
    if (n <= g_capacity) {
        return 0;
    }

    if (g_d_a) cudaFree(g_d_a);
    if (g_d_b) cudaFree(g_d_b);
    if (g_d_c) cudaFree(g_d_c);
    g_d_a = nullptr;
    g_d_b = nullptr;
    g_d_c = nullptr;
    g_capacity = 0;

    size_t bytes = static_cast<size_t>(n) * sizeof(float);
    cudaError_t err = cudaMalloc(&g_d_a, bytes);
    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }
    err = cudaMalloc(&g_d_b, bytes);
    if (err != cudaSuccess) {
        cudaFree(g_d_a);
        g_d_a = nullptr;
        return static_cast<int>(err);
    }
    err = cudaMalloc(&g_d_c, bytes);
    if (err != cudaSuccess) {
        cudaFree(g_d_a);
        cudaFree(g_d_b);
        g_d_a = nullptr;
        g_d_b = nullptr;
        return static_cast<int>(err);
    }

    g_capacity = n;
    return 0;
}
}  // namespace

int vectorAddHostAdvanced(const float* a, const float* b, float* c, int n) {
    if (a == nullptr || b == nullptr || c == nullptr || n <= 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    if (g_stream == nullptr) {
        cudaError_t streamErr = cudaStreamCreate(&g_stream);
        if (streamErr != cudaSuccess) {
            return static_cast<int>(streamErr);
        }
    }

    int capCode = ensureCapacityAdvanced(n);
    if (capCode != 0) {
        return capCode;
    }

    size_t bytes = static_cast<size_t>(n) * sizeof(float);
    cudaError_t err = cudaMemcpyAsync(g_d_a, a, bytes, cudaMemcpyHostToDevice, g_stream);
    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }
    err = cudaMemcpyAsync(g_d_b, b, bytes, cudaMemcpyHostToDevice, g_stream);
    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    vectorAddKernelAdvanced<<<blocks, threadsPerBlock, 0, g_stream>>>(g_d_a, g_d_b, g_d_c, n);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    err = cudaMemcpyAsync(c, g_d_c, bytes, cudaMemcpyDeviceToHost, g_stream);
    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    err = cudaStreamSynchronize(g_stream);
    return static_cast<int>(err);
}

int releaseAdvancedResources() {
    if (g_d_a) cudaFree(g_d_a);
    if (g_d_b) cudaFree(g_d_b);
    if (g_d_c) cudaFree(g_d_c);
    g_d_a = nullptr;
    g_d_b = nullptr;
    g_d_c = nullptr;
    g_capacity = 0;

    if (g_stream) {
        cudaError_t err = cudaStreamDestroy(g_stream);
        g_stream = nullptr;
        return static_cast<int>(err);
    }

    return 0;
}
