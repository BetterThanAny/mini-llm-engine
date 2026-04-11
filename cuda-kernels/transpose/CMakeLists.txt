add_executable(transpose transpose.cu)
target_link_libraries(transpose CUDA::cudart)
set_target_properties(transpose PROPERTIES CUDA_ARCHITECTURES "86")
