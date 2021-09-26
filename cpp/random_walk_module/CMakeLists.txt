set(random_walk_module_src
    random_walk_module.cpp)

add_query_module(random_walk 1 ${random_walk_module_src})

# Link external libraries
target_link_libraries(random_walk PRIVATE mg_utility)
