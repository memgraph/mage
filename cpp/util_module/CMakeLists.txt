set(util_module_src
    util_module.cpp
    algorithm/util.cpp)

add_query_module(create 1 "${util_module_src}")