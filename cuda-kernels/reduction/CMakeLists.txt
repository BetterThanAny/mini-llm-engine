add_executable(reduction reduction.cu)
target_link_libraries(reduction CUDA::cudart)
