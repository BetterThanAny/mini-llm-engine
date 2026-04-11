add_executable(transpose transpose.cu)
target_link_libraries(transpose CUDA::cudart)
