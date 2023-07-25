set(map_module_src
    map_module.cpp
    algorithm/algorithm.cpp)

add_query_module(map 1 "${map_module_src}")
