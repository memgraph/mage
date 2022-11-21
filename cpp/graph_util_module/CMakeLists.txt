# Add all module files related to graph util module
set(graph_util_src
    graph_util_module.cpp)

add_query_module(graph_util 1 "${graph_util_src}")

# Link external libraries
target_link_libraries(graph_util PRIVATE mg_utility)
target_include_directories(graph_util PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Module tests
# set(katz_test_src
# katz_centrality_test.cpp
# algorithm/katz.cpp)

# add_executable(katz_test "${katz_test_src}")
# target_link_libraries(katz_test PRIVATE mg_utility mage_gtest)

# # Module tests
# set(katz_online_test_src
# katz_centrality_online_test.cpp
# algorithm/katz.cpp)

# add_executable(katz_online_test "${katz_online_test_src}")
# target_link_libraries(katz_online_test PRIVATE mg_utility mage_gtest)
