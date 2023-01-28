set(node_similarity_module_src
    node_similarity_module.cpp
    algorithm/node_similarity.cpp)

add_query_module(node_similarity 1 "${node_similarity_module_src}")

# Link external libraries
target_link_libraries(node_similarity PRIVATE mg_utility)
target_include_directories(node_similarity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Module tests
if (NOT MAGE_CUGRAPH_ENABLE)
    include(GoogleTest)
    set(node_similarity_test_src
        node_similarity_test.cpp
        algorithm/node_similarity.cpp)

    add_executable(node_similarity_test "${node_similarity_test_src}")
    target_link_libraries(node_similarity_test PRIVATE mg_utility mage_gtest)
    gtest_add_tests(TARGET node_similarity_test)
endif()
