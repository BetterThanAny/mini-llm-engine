add_executable(reduction reduction.cu)

target_link_libraries(reduction PRIVATE CUDA::cudart)

# RTX 3080 = sm_86
set_target_properties(reduction PROPERTIES
    CUDA_ARCHITECTURES "86"
    OUTPUT_NAME "reduction"
)
