set(connectivity_module_src
    connectivity_module.cpp)

add_query_module(connectivity_module 1 ${connectivity_module_src})

# Link external libraries
target_link_libraries(connectivity_module PRIVATE mg_utility)
