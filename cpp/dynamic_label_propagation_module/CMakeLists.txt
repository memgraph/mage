set(dynamic_label_propagation_module_src
    dynamic_label_propagation_module.cpp
    algorithm/dynamic_label_propagation.cpp)

add_query_module(dynamic_label_propagation 1 "${dynamic_label_propagation_module_src}")

# Link external libraries
target_link_libraries(dynamic_label_propagation PRIVATE mg_utility)
target_include_directories(dynamic_label_propagation PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Module tests
include(GoogleTest)
set(dynamic_label_propagation_test_src
    dynamic_label_propagation_test.cpp
    algorithm/dynamic_label_propagation.cpp)

add_executable(dynamic_label_propagation_test "${dynamic_label_propagation_test_src}")
target_link_libraries(dynamic_label_propagation_test PRIVATE mg_utility gtest)
gtest_add_tests(TARGET dynamic_label_propagation_test)
