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
if (NOT MAGE_CUGRAPH_ENABLE)
    include(GoogleTest)
    set(community_detection_test_src
        community_detection_test.cpp
        algorithm/louvain.cpp)

    add_executable(community_detection_test "${community_detection_test_src}")
    target_include_directories(community_detection_test PRIVATE ${GRAPPOLO_HEADERS} ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(community_detection_test PRIVATE mg_utility basic_cd full_syn_opt coloring util OpenMP::OpenMP_CXX mage_gtest)
    gtest_add_tests(TARGET community_detection_test)
endif()


################# LabelRankT

# Community detection online module

set(community_detection_online_module_src
    community_detection_online_module.cpp)

add_query_module(community_detection_online 1 "${community_detection_online_module_src}")
