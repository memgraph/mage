set(shortest_path_module_src
    shortest_path_module.cpp
    algorithm/shortest_path.cpp)

add_query_module(k_shortest_path 1 "${shortest_path_module_src}")

# Link external libraries
target_link_libraries(k_shortest_path PRIVATE mg_utility)
target_include_directories(k_shortest_path PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
