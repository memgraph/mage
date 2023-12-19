set(shortest_path_module_src
    shortest_path_module.cpp
    algorithm/k_weighted_shortest_path.cpp)

add_query_module(shortest_path 1 "${shortest_path_module_src}")
