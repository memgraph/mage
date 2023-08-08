set(neighbors_module_src
    neighbors_module.cpp
    algorithm/neighbors.cpp)

add_query_module(neighbors 1 "${neighbors_module_src}")
