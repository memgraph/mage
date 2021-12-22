set(ccugraph_alg_module_src
    main.cu)

add_query_module(ccugraph_alg 1 "${ccugraph_alg_module_src}")

# Link external libraries
target_link_libraries(ccugraph_alg PRIVATE mg_utility)
target_include_directories(ccugraph_alg PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
