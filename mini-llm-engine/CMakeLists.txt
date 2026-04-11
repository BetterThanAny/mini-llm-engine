cmake_minimum_required(VERSION 3.18)
project(mini_llm_engine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86)

find_package(CUDAToolkit REQUIRED)

# Third-party
add_subdirectory(third_party/sentencepiece EXCLUDE_FROM_ALL)

include_directories(src)

add_executable(llm_engine
    src/main.cpp
    src/model.cpp
    src/tensor.cpp
    src/ops_cpu.cpp
    src/kv_cache.cpp
    src/sampler.cpp
    src/ops_cuda.cu
)

target_link_libraries(llm_engine
    CUDA::cudart
    CUDA::cublas
    sentencepiece
)

# Tests
enable_testing()
add_subdirectory(tests)

# Benchmarks
add_subdirectory(benchmarks)
