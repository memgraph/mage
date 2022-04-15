################# Louvain

# Community detection module

set(community_detection_module_src
    community_detection_module.cpp
    algorithm/louvain.cpp)

add_subdirectory(grappolo)
add_query_module(community_detection 1 "${community_detection_module_src}")

target_include_directories(community_detection PRIVATE ${GRAPPOLO_HEADERS} ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(community_detection PRIVATE mg_utility basic_cd full_syn_opt coloring util OpenMP::OpenMP_CXX)

# Module tests
include(GoogleTest)
set(community_detection_test_src
    community_detection_test.cpp
    algorithm/louvain.cpp)

add_executable(community_detection_test "${community_detection_test_src}")
target_include_directories(community_detection_test PRIVATE ${GRAPPOLO_HEADERS} ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(community_detection_test PRIVATE mg_utility basic_cd full_syn_opt coloring util OpenMP::OpenMP_CXX gtest)
gtest_add_tests(TARGET community_detection_test)


################# LabelRankT

# Community detection online module

set(community_detection_online_module_src
    community_detection_online_module.cpp
    algorithm_online/community_detection.cpp)

add_query_module(community_detection_online 1 "${community_detection_online_module_src}")

# Link external libraries
target_link_libraries(community_detection_online PRIVATE mg_utility)
target_include_directories(community_detection_online PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Module tests
include(GoogleTest)
set(community_detection_online_test_src
    community_detection_online_test.cpp
    algorithm_online/community_detection.cpp)

add_executable(community_detection_online_test "${community_detection_online_test_src}")
target_link_libraries(community_detection_online_test PRIVATE mg_utility gtest)
gtest_add_tests(TARGET community_detection_online_test)
