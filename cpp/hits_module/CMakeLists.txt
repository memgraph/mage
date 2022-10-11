set(hits_module_src
        hits_module.cpp
        algorithm/hits.cpp)

add_query_module(hits 1 "${hits_module_src}")

# Link external libraries
target_link_libraries(hits PRIVATE mg_utility)
target_include_directories(hits PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

################################################################

# Module tests
if (NOT MAGE_CUGRAPH_ENABLE)
    include(GoogleTest)
    set(hits_test_src
            hits_test.cpp
            algorithm/hits.cpp)

    add_executable(hits_test "${hits_test_src}")
    target_link_libraries(hits_test PRIVATE mg_utility mage_gtest)
    gtest_add_tests(TARGET hits_test)
endif()

################################################################