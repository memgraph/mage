set(ccugraph_alg_module_src
    main.cu)

# TODO(gitbuda): Implement add_cuda_query_module function.
add_query_module(ccugraph_alg 1 "${ccugraph_alg_module_src}")

# Link external libraries
target_link_libraries(ccugraph_alg PRIVATE mg_utility mage_cugraph)
target_include_directories(ccugraph_alg PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
# TODO(gitubda): CUDA_* have to be highly configurable.
# CUDA_ARCHITECTURES -> https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/).
list(APPEND MAGE_CUDA_FLAGS --expt-extended-lambda)
target_compile_options(ccugraph_alg
  PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${MAGE_CXX_FLAGS}>"
          "$<$<COMPILE_LANGUAGE:CUDA>:${MAGE_CUDA_FLAGS}>"
)
# TODO(gitbuda): Consider adding PIC stuff to the cuda targets.
# TODO(gitbuda): Figure out why cmake fails without following lines.
set_target_properties(ccugraph_alg PROPERTIES
  CXX_STANDARD                        17
  CXX_STANDARD_REQUIRED               ON
  CUDA_STANDARD                       17
  CUDA_STANDARD_REQUIRED              ON
  CUDA_ARCHITECTURES                  "75"
  POSITION_INDEPENDENT_CODE           ON
  INTERFACE_POSITION_INDEPENDENT_CODE ON
)
