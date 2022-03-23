set(sp_module_src
    shortest_path.cpp)

add_query_module(all_shortest_paths 1 ${sp_module_src})

# Link external libraries
target_link_libraries(all_shortest_paths PRIVATE mg_utility)
