add_query_module(cugraph.pagerank 1 pagerank.cu)
target_mage_cugraph(cugraph.pagerank)
target_link_libraries(cugraph.pagerank PRIVATE mg_utility)
target_include_directories(cugraph.pagerank PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_query_module(cugraph.louvain 1 louvain.cu)
target_mage_cugraph(cugraph.louvain)
target_link_libraries(cugraph.louvain PRIVATE mg_utility)
target_include_directories(cugraph.louvain PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_query_module(cugraph.katz_centrality 1 katz_centrality.cu)
target_mage_cugraph(cugraph.katz_centrality)
target_link_libraries(cugraph.katz_centrality PRIVATE mg_utility)
target_include_directories(cugraph.katz_centrality PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_query_module(cugraph.leiden 1 leiden.cu)
target_mage_cugraph(cugraph.leiden)
target_link_libraries(cugraph.leiden PRIVATE mg_utility)
target_include_directories(cugraph.leiden PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
