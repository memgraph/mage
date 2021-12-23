set(ccugraph_alg_module_src
    main.cu)

# TODO(gitbuda): Implement add_cuda_query_module function.
add_query_module(ccugraph_alg 1 "${ccugraph_alg_module_src}")

# Link external libraries
target_link_libraries(ccugraph_alg PRIVATE mg_utility)
target_include_directories(ccugraph_alg PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(ccugraph_alg PROPERTIES
  CUDA_STANDARD          14
  CUDA_STANDARD_REQUIRED ON
  CUDA_ARCHITECTURES     "35;50;72")
