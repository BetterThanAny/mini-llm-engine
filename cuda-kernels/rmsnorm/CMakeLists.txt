add_executable(rmsnorm rmsnorm.cu)
target_link_libraries(rmsnorm PRIVATE CUDA::cudart)
set_target_properties(rmsnorm PROPERTIES
    CUDA_ARCHITECTURES "86"
    OUTPUT_NAME "rmsnorm"
)
