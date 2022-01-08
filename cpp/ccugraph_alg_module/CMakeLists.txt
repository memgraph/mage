set(ccugraph_alg_module_src
    main.cu)

add_query_module(ccugraph_alg 1 "${ccugraph_alg_module_src}")

# TODO(gitbuda): Add macro for cuda set_target_properties + deal with CUDA_ARCHITECTURES somehow.

# Link external libraries
target_link_libraries(ccugraph_alg PRIVATE mg_utility mage_cugraph)
target_include_directories(ccugraph_alg PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
list(APPEND MAGE_CUDA_FLAGS --expt-extended-lambda)
if(CMAKE_BUILD_TYPE MATCHES Debug)
    message(STATUS "Building ccugraph_alg cuda module with debugging flags")
    list(APPEND MAGE_CUDA_FLAGS -g -G -Xcompiler=-rdynamic)
endif()
target_compile_options(ccugraph_alg
  PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${MAGE_CXX_FLAGS}>"
          "$<$<COMPILE_LANGUAGE:CUDA>:${MAGE_CUDA_FLAGS}>"
)
set_target_properties(ccugraph_alg PROPERTIES
  CXX_STANDARD                        17
  CXX_STANDARD_REQUIRED               ON
  CUDA_STANDARD                       17
  CUDA_STANDARD_REQUIRED              ON
  CUDA_ARCHITECTURES                  "75"
  POSITION_INDEPENDENT_CODE           ON
  INTERFACE_POSITION_INDEPENDENT_CODE ON
)
