set(katz_src
    katz_centrality_module.cpp
    algorithm/katz.cpp)

add_query_module(katz 1 "${katz_src}")

# Link external libraries
target_link_libraries(katz PRIVATE mg_utility)
target_include_directories(katz PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Module tests
set(katz_test_src
    katz_centrality_test.cpp
    algorithm/katz.cpp)

add_executable(katz_test "${katz_test_src}")
target_link_libraries(katz_test PRIVATE mg_utility gtest)
