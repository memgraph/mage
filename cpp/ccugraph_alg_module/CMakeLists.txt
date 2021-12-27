set(ccugraph_alg_module_src
    main.cu)

# TODO(gitbuda): Implement add_cuda_query_module function.
add_query_module(ccugraph_alg 1 "${ccugraph_alg_module_src}")

# Link external libraries
target_link_libraries(ccugraph_alg PRIVATE mg_utility)
target_include_directories(ccugraph_alg PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
# TODO(gitubda): CUDA_* have to be highly configurable.
# CUDA_ARCHITECTURES -> https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/).
set_target_properties(ccugraph_alg PROPERTIES
  CUDA_STANDARD          17
  CUDA_STANDARD_REQUIRED ON
  CUDA_ARCHITECTURES     "75"
)
