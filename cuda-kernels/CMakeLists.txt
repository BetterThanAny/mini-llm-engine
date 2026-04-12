cmake_minimum_required(VERSION 3.18)
project(cuda_kernels LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# RTX 3080 = sm_86
set(CMAKE_CUDA_ARCHITECTURES 86)

# Common compile options
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-O2>)

find_package(CUDAToolkit REQUIRED)

add_subdirectory(vector_add)
add_subdirectory(reduction)
add_subdirectory(transpose)
add_subdirectory(gemm)
add_subdirectory(rmsnorm)
add_subdirectory(softmax)
