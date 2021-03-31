set(bipartite_matching_src 
    bipartite_matching_module.cpp
    algorithm/bipartite_matching.cpp)

add_query_module(bipartite_matching 1 "${bipartite_matching_src}")

# Link external libraries
target_link_libraries(bipartite_matching PRIVATE mg_utility)
target_include_directories(bipartite_matching PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Module tests
set(bipartite_matching_test_src 
    bipartite_matching_test.cpp
    algorithm/bipartite_matching.cpp)

add_executable(bipartite_matching_test  "${bipartite_matching_test_src}")
target_link_libraries(bipartite_matching_test PRIVATE mg_utility gtest)