set(collections_src
    collections_module.cpp
    algorithm/collections.cpp)

add_query_module(collections_module 1 "${collections_src}")
