add_query_module(ccugraph_alg 1 main.cu)
target_mage_cugraph(ccugraph_alg)
target_link_libraries(ccugraph_alg PRIVATE mg_utility)
target_include_directories(ccugraph_alg PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
