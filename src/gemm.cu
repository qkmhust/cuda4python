#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "kernels.h"

namespace {
constexpr int TILE = 16;

// cuBLAS 路径缓存资源：用于减少重复创建 handle/stream 与显存分配。
float* g_gemm_a = nullptr;
float* g_gemm_b = nullptr;
float* g_gemm_c = nullptr;
size_t g_cap_a = 0;
size_t g_cap_b = 0;
size_t g_cap_c = 0;
cudaStream_t g_gemm_stream = nullptr;
cublasHandle_t g_cublas = nullptr;

// GEMM 基础版：每线程计算 C 的一个元素，易理解但访存效率一般。
__global__ void gemmKernelBasic(const float* a, const float* b, float* c, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || col >= n) {
        return;
    }

    float acc = 0.0f;
    for (int i = 0; i < k; ++i) {
        acc += a[row * k + i] * b[i * n + col];
    }
    c[row * n + col] = acc;
}

// GEMM 进阶版：分块加载到 shared memory，减少全局内存访问。
__global__ void gemmKernelTiled(const float* a, const float* b, float* c, int m, int n, int k) {
    __shared__ float tileA[TILE][TILE];
    __shared__ float tileB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;
    int numTiles = (k + TILE - 1) / TILE;

    for (int t = 0; t < numTiles; ++t) {
        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;

        tileA[threadIdx.y][threadIdx.x] = (row < m && aCol < k) ? a[row * k + aCol] : 0.0f;
        tileB[threadIdx.y][threadIdx.x] = (bRow < k && col < n) ? b[bRow * n + col] : 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE; ++i) {
            acc += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < m && col < n) {
        c[row * n + col] = acc;
    }
}

int gemmImpl(
    const float* a,
    const float* b,
    float* c,
    int m,
    int n,
    int k,
    bool useTiled
) {
    // 教学路径：每次调用独立申请/释放显存，便于理解完整数据流。
    if (a == nullptr || b == nullptr || c == nullptr || m <= 0 || n <= 0 || k <= 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    size_t bytesA = static_cast<size_t>(m) * static_cast<size_t>(k) * sizeof(float);
    size_t bytesB = static_cast<size_t>(k) * static_cast<size_t>(n) * sizeof(float);
    size_t bytesC = static_cast<size_t>(m) * static_cast<size_t>(n) * sizeof(float);

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;

    cudaError_t err = cudaMalloc(&d_a, bytesA);
    if (err != cudaSuccess) return static_cast<int>(err);
    err = cudaMalloc(&d_b, bytesB);
    if (err != cudaSuccess) {
        cudaFree(d_a);
        return static_cast<int>(err);
    }
    err = cudaMalloc(&d_c, bytesC);
    if (err != cudaSuccess) {
        cudaFree(d_a);
        cudaFree(d_b);
        return static_cast<int>(err);
    }

    err = cudaMemcpy(d_a, a, bytesA, cudaMemcpyHostToDevice);
    if (err == cudaSuccess) err = cudaMemcpy(d_b, b, bytesB, cudaMemcpyHostToDevice);

    if (err == cudaSuccess) {
        dim3 block(TILE, TILE);
        dim3 grid((n + TILE - 1) / TILE, (m + TILE - 1) / TILE);
        if (useTiled) {
            gemmKernelTiled<<<grid, block>>>(d_a, d_b, d_c, m, n, k);
        } else {
            gemmKernelBasic<<<grid, block>>>(d_a, d_b, d_c, m, n, k);
        }
        err = cudaGetLastError();
    }

    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    if (err == cudaSuccess) err = cudaMemcpy(c, d_c, bytesC, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return static_cast<int>(err);
}

// cuBLAS 路径的显存扩容函数：只在需要时增长容量。
int ensureGemmCapacity(size_t bytesA, size_t bytesB, size_t bytesC) {
    if (bytesA > g_cap_a) {
        if (g_gemm_a) cudaFree(g_gemm_a);
        cudaError_t err = cudaMalloc(&g_gemm_a, bytesA);
        if (err != cudaSuccess) return static_cast<int>(err);
        g_cap_a = bytesA;
    }
    if (bytesB > g_cap_b) {
        if (g_gemm_b) cudaFree(g_gemm_b);
        cudaError_t err = cudaMalloc(&g_gemm_b, bytesB);
        if (err != cudaSuccess) return static_cast<int>(err);
        g_cap_b = bytesB;
    }
    if (bytesC > g_cap_c) {
        if (g_gemm_c) cudaFree(g_gemm_c);
        cudaError_t err = cudaMalloc(&g_gemm_c, bytesC);
        if (err != cudaSuccess) return static_cast<int>(err);
        g_cap_c = bytesC;
    }
    return 0;
}
}  // namespace

int gemmHost(const float* a, const float* b, float* c, int m, int n, int k) {
    return gemmImpl(a, b, c, m, n, k, false);
}

int gemmHostAdvanced(const float* a, const float* b, float* c, int m, int n, int k) {
    return gemmImpl(a, b, c, m, n, k, true);
}

int gemmHostCublas(const float* a, const float* b, float* c, int m, int n, int k) {
    if (a == nullptr || b == nullptr || c == nullptr || m <= 0 || n <= 0 || k <= 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    if (g_gemm_stream == nullptr) {
        cudaError_t err = cudaStreamCreate(&g_gemm_stream);
        if (err != cudaSuccess) return static_cast<int>(err);
    }
    if (g_cublas == nullptr) {
        cublasStatus_t st = cublasCreate(&g_cublas);
        if (st != CUBLAS_STATUS_SUCCESS) return static_cast<int>(cudaErrorUnknown);
        st = cublasSetStream(g_cublas, g_gemm_stream);
        if (st != CUBLAS_STATUS_SUCCESS) return static_cast<int>(cudaErrorUnknown);
    }

    size_t bytesA = static_cast<size_t>(m) * static_cast<size_t>(k) * sizeof(float);
    size_t bytesB = static_cast<size_t>(k) * static_cast<size_t>(n) * sizeof(float);
    size_t bytesC = static_cast<size_t>(m) * static_cast<size_t>(n) * sizeof(float);

    int capCode = ensureGemmCapacity(bytesA, bytesB, bytesC);
    if (capCode != 0) return capCode;

    cudaError_t err = cudaMemcpyAsync(g_gemm_a, a, bytesA, cudaMemcpyHostToDevice, g_gemm_stream);
    if (err != cudaSuccess) return static_cast<int>(err);
    err = cudaMemcpyAsync(g_gemm_b, b, bytesB, cudaMemcpyHostToDevice, g_gemm_stream);
    if (err != cudaSuccess) return static_cast<int>(err);

    const float alpha = 1.0f;
    const float beta = 0.0f;
    // 行主序映射到 cuBLAS 列主序：C^T = B^T * A^T。
    cublasStatus_t st = cublasSgemm(
        g_cublas,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        n,
        m,
        k,
        &alpha,
        g_gemm_b,
        n,
        g_gemm_a,
        k,
        &beta,
        g_gemm_c,
        n
    );
    if (st != CUBLAS_STATUS_SUCCESS) return static_cast<int>(cudaErrorUnknown);

    err = cudaMemcpyAsync(c, g_gemm_c, bytesC, cudaMemcpyDeviceToHost, g_gemm_stream);
    if (err != cudaSuccess) return static_cast<int>(err);

    err = cudaStreamSynchronize(g_gemm_stream);
    return static_cast<int>(err);
}

// 释放 cuBLAS 相关资源，防止长时间实验中的资源泄漏。
int releaseGemmResources() {
    if (g_gemm_a) cudaFree(g_gemm_a);
    if (g_gemm_b) cudaFree(g_gemm_b);
    if (g_gemm_c) cudaFree(g_gemm_c);
    g_gemm_a = nullptr;
    g_gemm_b = nullptr;
    g_gemm_c = nullptr;
    g_cap_a = 0;
    g_cap_b = 0;
    g_cap_c = 0;

    if (g_cublas) {
        cublasDestroy(g_cublas);
        g_cublas = nullptr;
    }
    if (g_gemm_stream) {
        cudaError_t err = cudaStreamDestroy(g_gemm_stream);
        g_gemm_stream = nullptr;
        return static_cast<int>(err);
    }
    return 0;
}
