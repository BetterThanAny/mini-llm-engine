add_executable(softmax softmax.cu)

target_link_libraries(softmax PRIVATE CUDA::cudart)

set_target_properties(softmax PROPERTIES
    CUDA_ARCHITECTURES "86"
    OUTPUT_NAME "softmax"
)
