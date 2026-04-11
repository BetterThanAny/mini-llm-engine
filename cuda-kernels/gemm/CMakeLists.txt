add_executable(gemm gemm.cu)
target_link_libraries(gemm CUDA::cudart CUDA::cublas)
set_target_properties(gemm PROPERTIES CUDA_ARCHITECTURES "86")
