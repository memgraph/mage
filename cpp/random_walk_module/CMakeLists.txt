set(random_walk_module_src
    random_walk_module.cpp)

add_query_module(random_walk_generator 1 ${random_walk_module_src})

# Link external libraries
target_link_libraries(random_walk_generator PRIVATE mg_utility)