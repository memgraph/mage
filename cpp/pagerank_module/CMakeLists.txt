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
target_link_libraries(pagerank_test PRIVATE mg_utility mage_gtest)
gtest_add_tests(TARGET pagerank_test)

################################################################

set(pagerank_online_module_src
    pagerank_online_module.cpp
    algorithm_online/pagerank.cpp
    algorithm_online/pagerank.cpp)

add_query_module(pagerank_online 1 "${pagerank_online_module_src}")

# Link external libraries
target_link_libraries(pagerank_online PRIVATE mg_utility)
target_include_directories(pagerank_online PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

################################################################

# Module tests
include(GoogleTest)
set(pagerank_online_test_src
    pagerank_online_test.cpp
    algorithm_online/pagerank.cpp)

add_executable(pagerank_online_test "${pagerank_online_test_src}")
target_link_libraries(pagerank_online_test PRIVATE mg_utility mage_gtest)
gtest_add_tests(TARGET pagerank_online_test)
