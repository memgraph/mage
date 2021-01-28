set(pagerank_module_src
        pagerank_module.cpp
        algorithm/pagerank_algorithm.cpp
        memgraph_utilities/mg_utility.cpp)

add_query_module(pagerank_module 1 "${pagerank_module_src}")
target_include_directories(pagerank_module PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})