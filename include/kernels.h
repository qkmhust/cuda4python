#ifndef KERNELS_H
#define KERNELS_H

#ifdef __cplusplus
extern "C" {
#endif

// VectorAdd 基础路径（教学友好）。
void launchVectorAdd(const float* a, const float* b, float* c, int n);
int vectorAddHost(const float* a, const float* b, float* c, int n);
int validateVectorAdd(int n, float expected, float tolerance);
// VectorAdd 进阶路径（缓存显存 + 异步流）。
int vectorAddHostAdvanced(const float* a, const float* b, float* c, int n);
int releaseAdvancedResources();

// DL 常见算子：ReLU / Softmax。
int reluHost(const float* x, float* y, int n);
int softmaxHost(const float* x, float* y, int rows, int cols);
int softmaxHostAdvanced(const float* x, float* y, int rows, int cols);
// GEMM 三层路径：basic -> tiled -> cuBLAS。
int gemmHost(const float* a, const float* b, float* c, int m, int n, int k);
int gemmHostAdvanced(const float* a, const float* b, float* c, int m, int n, int k);
int gemmHostCublas(const float* a, const float* b, float* c, int m, int n, int k);
int releaseGemmResources();

#ifdef __cplusplus
}
#endif

#endif  // KERNELS_H
