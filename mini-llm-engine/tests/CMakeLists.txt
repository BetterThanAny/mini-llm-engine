find_package(Python3 COMPONENTS Interpreter REQUIRED)

add_test(
    NAME flash_attn_python
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_flash_attn.py
)
set_tests_properties(flash_attn_python PROPERTIES
    SKIP_RETURN_CODE 77
)

add_test(
    NAME gpu_cpu_match_smoke
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_gpu_cpu_match.py
)
set_tests_properties(gpu_cpu_match_smoke PROPERTIES
    ENVIRONMENT "ENGINE_BIN=$<TARGET_FILE:llm_engine>"
    SKIP_RETURN_CODE 77
)
