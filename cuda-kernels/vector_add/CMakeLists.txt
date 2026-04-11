add_executable(vector_add vector_add.cu)
target_link_libraries(vector_add CUDA::cudart)
