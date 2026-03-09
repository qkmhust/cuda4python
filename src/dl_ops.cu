#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>

#include "kernels.h"

__global__ void reluKernel(const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = x[idx];
        y[idx] = v > 0.0f ? v : 0.0f;
    }
}

// Basic softmax: one thread processes one row for readability.
__global__ void softmaxRowKernelBasic(const float* x, float* y, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }

    int offset = row * cols;
    float maxVal = -FLT_MAX;
    for (int c = 0; c < cols; ++c) {
        float v = x[offset + c];
        if (v > maxVal) maxVal = v;
    }

    float sumExp = 0.0f;
    for (int c = 0; c < cols; ++c) {
        sumExp += expf(x[offset + c] - maxVal);
    }

    for (int c = 0; c < cols; ++c) {
        y[offset + c] = expf(x[offset + c] - maxVal) / sumExp;
    }
}

// Advanced softmax: one block per row with shared-memory reductions.
__global__ void softmaxRowKernelAdvanced(const float* x, float* y, int cols) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int offset = row * cols;

    __shared__ float sharedMax[256];
    __shared__ float sharedSum[256];

    float localMax = -FLT_MAX;
    for (int c = tid; c < cols; c += blockDim.x) {
        float v = x[offset + c];
        if (v > localMax) localMax = v;
    }
    sharedMax[tid] = localMax;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float other = sharedMax[tid + stride];
            if (other > sharedMax[tid]) sharedMax[tid] = other;
        }
        __syncthreads();
    }
    float maxVal = sharedMax[0];

    float localSum = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x) {
        localSum += expf(x[offset + c] - maxVal);
    }
    sharedSum[tid] = localSum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedSum[tid] += sharedSum[tid + stride];
        }
        __syncthreads();
    }
    float sumExp = sharedSum[0];

    for (int c = tid; c < cols; c += blockDim.x) {
        y[offset + c] = expf(x[offset + c] - maxVal) / sumExp;
    }
}

int reluHost(const float* x, float* y, int n) {
    if (x == nullptr || y == nullptr || n <= 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    size_t bytes = static_cast<size_t>(n) * sizeof(float);
    float *d_x = nullptr, *d_y = nullptr;

    cudaError_t err = cudaMalloc(&d_x, bytes);
    if (err != cudaSuccess) return static_cast<int>(err);
    err = cudaMalloc(&d_y, bytes);
    if (err != cudaSuccess) {
        cudaFree(d_x);
        return static_cast<int>(err);
    }

    err = cudaMemcpy(d_x, x, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_x);
        cudaFree(d_y);
        return static_cast<int>(err);
    }

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    reluKernel<<<blocks, threads>>>(d_x, d_y, n);

    err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    if (err == cudaSuccess) err = cudaMemcpy(y, d_y, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);
    return static_cast<int>(err);
}

int softmaxHost(const float* x, float* y, int rows, int cols) {
    if (x == nullptr || y == nullptr || rows <= 0 || cols <= 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    size_t bytes = static_cast<size_t>(rows) * static_cast<size_t>(cols) * sizeof(float);
    float *d_x = nullptr, *d_y = nullptr;

    cudaError_t err = cudaMalloc(&d_x, bytes);
    if (err != cudaSuccess) return static_cast<int>(err);
    err = cudaMalloc(&d_y, bytes);
    if (err != cudaSuccess) {
        cudaFree(d_x);
        return static_cast<int>(err);
    }

    err = cudaMemcpy(d_x, x, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_x);
        cudaFree(d_y);
        return static_cast<int>(err);
    }

    int threads = 128;
    int blocks = (rows + threads - 1) / threads;
    softmaxRowKernelBasic<<<blocks, threads>>>(d_x, d_y, rows, cols);

    err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    if (err == cudaSuccess) err = cudaMemcpy(y, d_y, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);
    return static_cast<int>(err);
}

int softmaxHostAdvanced(const float* x, float* y, int rows, int cols) {
    if (x == nullptr || y == nullptr || rows <= 0 || cols <= 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    size_t bytes = static_cast<size_t>(rows) * static_cast<size_t>(cols) * sizeof(float);
    float *d_x = nullptr, *d_y = nullptr;

    cudaError_t err = cudaMalloc(&d_x, bytes);
    if (err != cudaSuccess) return static_cast<int>(err);
    err = cudaMalloc(&d_y, bytes);
    if (err != cudaSuccess) {
        cudaFree(d_x);
        return static_cast<int>(err);
    }

    err = cudaMemcpy(d_x, x, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_x);
        cudaFree(d_y);
        return static_cast<int>(err);
    }

    const int threads = 256;
    softmaxRowKernelAdvanced<<<rows, threads>>>(d_x, d_y, cols);

    err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    if (err == cudaSuccess) err = cudaMemcpy(y, d_y, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);
    return static_cast<int>(err);
}
