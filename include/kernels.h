#ifndef KERNELS_H
#define KERNELS_H

#ifdef __cplusplus
extern "C" {
#endif

void launchVectorAdd(const float* a, const float* b, float* c, int n);
int vectorAddHost(const float* a, const float* b, float* c, int n);
int validateVectorAdd(int n, float expected, float tolerance);
int vectorAddHostAdvanced(const float* a, const float* b, float* c, int n);
int releaseAdvancedResources();

int reluHost(const float* x, float* y, int n);
int softmaxHost(const float* x, float* y, int rows, int cols);
int softmaxHostAdvanced(const float* x, float* y, int rows, int cols);

#ifdef __cplusplus
}
#endif

#endif  // KERNELS_H
