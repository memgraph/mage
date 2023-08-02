set(date_module_src
    date_module.cpp
    algorithm/date.cpp)

add_query_module(date 1 "${date_module_src}")
