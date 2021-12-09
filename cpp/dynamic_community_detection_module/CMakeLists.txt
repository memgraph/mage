set(dynamic_community_detection_module_src
    dynamic_community_detection_module.cpp
    algorithm/dynamic_community_detection.cpp)

add_query_module(dynamic_community_detection 1 "${dynamic_community_detection_module_src}")

# Link external libraries
target_link_libraries(dynamic_community_detection PRIVATE mg_utility)
target_include_directories(dynamic_community_detection PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Module tests
include(GoogleTest)
set(dynamic_community_detection_test_src
    dynamic_community_detection_test.cpp
    algorithm/dynamic_community_detection.cpp)

add_executable(dynamic_community_detection_test "${dynamic_community_detection_test_src}")
target_link_libraries(dynamic_community_detection_test PRIVATE mg_utility gtest)
gtest_add_tests(TARGET dynamic_community_detection_test)
