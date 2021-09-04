set(dummy_src
    module.cpp)

add_query_module(module 1 ${dummy_src})

# Link external libraries
target_link_libraries(module PRIVATE mg_utility)
