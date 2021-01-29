set(pagerank_module_src
    pagerank_module.cpp
    pagerank/pagerank.cpp)

add_query_module(pagerank_module 1 "${pagerank_module_src}")
target_link_libraries(pagerank_module PUBLIC utils)
target_include_directories(pagerank_module PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
