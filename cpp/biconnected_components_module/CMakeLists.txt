set(biconnected_components_module_src 
    biconnected_components_module.cpp
    algorithm/biconnected_components.cpp)

add_query_module(biconnected_components_module 1 "${biconnected_components_module_src}")

# Link external libraries
target_link_libraries(biconnected_components_module PRIVATE mg_utility)
target_include_directories(biconnected_components_module PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Module tests
set(biconnected_components_test_src 
    algorithm/biconnected_components.cpp
    biconnected_components_test.cpp)

add_executable(biconnected_components_test  "${biconnected_components_test_src}")
target_link_libraries(biconnected_components_test PRIVATE mg_graph gtest)