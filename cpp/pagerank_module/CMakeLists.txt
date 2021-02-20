set(pagerank_module_src
    pagerank_module.cpp
    pagerank/pagerank.cpp)

add_query_module(pagerank_module 1 "${pagerank_module_src}")
target_link_libraries(pagerank_module PUBLIC utils)
target_include_directories(pagerank_module PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

find_package(Threads REQUIRED)
add_executable(pagerank_test pagerank/pagerank.cpp pagerank/pagerank_test.cpp)
target_link_libraries(pagerank_test PRIVATE utils Threads::Threads)
target_include_directories(pagerank_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
