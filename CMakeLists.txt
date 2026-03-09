cmake_minimum_required(VERSION 3.18)
project(mini_cuda_llm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_library(mini_cuda_llm_cuda SHARED
    src/vector_add.cu
    src/vector_add_advanced.cu
)

target_include_directories(mini_cuda_llm_cuda PUBLIC include)

target_link_libraries(mini_cuda_llm_cuda PUBLIC CUDA::cudart)

set_target_properties(mini_cuda_llm_cuda PROPERTIES
    OUTPUT_NAME mini_cuda_llm_cuda
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

add_executable(mini_cuda_llm src/main.cpp)
target_include_directories(mini_cuda_llm PRIVATE include)
target_link_libraries(mini_cuda_llm PRIVATE mini_cuda_llm_cuda)

install(TARGETS mini_cuda_llm_cuda
    LIBRARY DESTINATION python/mini_cuda_llm
)
