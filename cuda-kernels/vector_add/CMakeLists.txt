add_executable(vector_add vector_add.cu)

target_link_libraries(vector_add PRIVATE CUDA::cudart)

# Explicitly target RTX 3080 (sm_86); inherited from top-level but stated here
# for clarity when building this subdir standalone.
set_target_properties(vector_add PROPERTIES
    CUDA_ARCHITECTURES "86"
    OUTPUT_NAME "vector_add"
)
