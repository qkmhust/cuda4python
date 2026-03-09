#include <cuda_runtime.h>

#include <iostream>
#include <vector>

#include "kernels.h"

int main() {
    const int n = 1024;

    std::vector<float> h_a(n, 1.0f);
    std::vector<float> h_b(n, 2.0f);
    std::vector<float> h_c(n, 0.0f);

    int code = vectorAddHost(h_a.data(), h_b.data(), h_c.data(), n);
    if (code != static_cast<int>(cudaSuccess)) {
        std::cerr << "vectorAddHost failed, cuda code=" << code << std::endl;
        return 1;
    }

    std::cout << "h_c[0] = " << h_c[0] << std::endl;

    int validateCode = validateVectorAdd(n, 3.0f, 1e-5f);
    if (validateCode != 0) {
        std::cerr << "validateVectorAdd failed, code=" << validateCode << std::endl;
        return 2;
    }

    std::cout << "validation passed" << std::endl;

    return 0;
}
