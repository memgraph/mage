set(betweenness_centrality_src 
    betweenness_centrality_module.cpp
    algorithm/betweenness_centrality.cpp)

add_query_module(betweenness_centrality 1 "${betweenness_centrality_src}")

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Link external libraries
target_link_libraries(betweenness_centrality PRIVATE mg_utility)
target_include_directories(betweenness_centrality PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Module tests
include(GoogleTest)
set(betweenness_centrality_src 
    betweenness_centrality_test.cpp
    algorithm/betweenness_centrality.cpp)

add_executable(betweenness_centrality_test  "${betweenness_centrality_src}")
target_link_libraries(betweenness_centrality_test PRIVATE mg_utility gtest)
gtest_add_tests(TARGET betweenness_centrality_test)