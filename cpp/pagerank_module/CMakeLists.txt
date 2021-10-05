set(pagerank_module_src
    pagerank_module.cpp
    algorithm/pagerank.cpp)

add_query_module(pagerank 1 "${pagerank_module_src}")

# Link external libraries
target_link_libraries(pagerank PRIVATE mg_utility)
target_include_directories(pagerank PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

################################################################

# Module tests
include(GoogleTest)
set(pagerank_test_src
    pagerank_test.cpp
    algorithm/pagerank.cpp)

add_executable(pagerank_test "${pagerank_test_src}")
target_link_libraries(pagerank_test PRIVATE mg_utility gtest)
gtest_add_tests(TARGET pagerank_test)

################################################################

set(pagerank_approx_module_src
    pagerank_approx_module.cpp
    algorithm_approx/pagerank.cpp
    algorithm_approx/pagerank.cpp)

add_query_module(pagerank_approx 1 "${pagerank_approx_module_src}")

# Link external libraries
target_link_libraries(pagerank_approx PRIVATE mg_utility)
target_include_directories(pagerank_approx PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

################################################################

# Module tests
include(GoogleTest)
set(pagerank_approx_test_src
    pagerank_approx_test.cpp
    algorithm_approx/pagerank.cpp)

add_executable(pagerank_approx_test "${pagerank_approx_test_src}")
target_link_libraries(pagerank_approx_test PRIVATE mg_utility gtest)
gtest_add_tests(TARGET pagerank_approx_test)
