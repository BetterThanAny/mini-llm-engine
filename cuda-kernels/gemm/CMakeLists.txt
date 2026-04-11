add_executable(gemm gemm.cu)
target_link_libraries(gemm CUDA::cudart CUDA::cublas)
