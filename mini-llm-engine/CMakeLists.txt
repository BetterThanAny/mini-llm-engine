cmake_minimum_required(VERSION 3.18)
project(mini_llm_engine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86)

# Use shared cudart consistently — avoids dual-linking static+shared cudart
# which creates two runtime instances with separate state.
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

# WSL2: the real GPU driver (libcuda.so.1) lives in /usr/lib/wsl/lib/.
# The apt nvidia-cuda-toolkit installs a non-functional stub libcuda.so in
# /usr/lib/x86_64-linux-gnu/ that shadows the real driver when dlopen'd.
# Use DT_RPATH (not RUNPATH) so the path propagates to dlopen inside cudart.
if(EXISTS "/usr/lib/wsl/lib/libcuda.so.1")
    set(CMAKE_BUILD_RPATH "/usr/lib/wsl/lib")
    # --disable-new-dtags forces DT_RPATH instead of DT_RUNPATH.
    add_link_options("LINKER:--disable-new-dtags")
    # Force libcuda.so.1 into DT_NEEDED so RPATH resolves it correctly.
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--no-as-needed -L/usr/lib/wsl/lib -lcuda -Wl,--as-needed")
endif()

find_package(CUDAToolkit REQUIRED)

# Third-party: sentencepiece via FetchContent
include(FetchContent)
FetchContent_Declare(
    sentencepiece
    GIT_REPOSITORY http://github.com/google/sentencepiece.git
    GIT_TAG        v0.2.0
    GIT_SHALLOW    TRUE
)
set(SPM_ENABLE_SHARED   OFF CACHE BOOL "" FORCE)
set(SPM_ENABLE_TCMALLOC OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(sentencepiece)

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

target_include_directories(llm_engine PRIVATE
    ${sentencepiece_SOURCE_DIR}/src
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
