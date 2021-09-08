set(uuid_module_src
    uuid_module.cpp)

add_query_module(uuid 1 ${uuid_module_src})

# Link external libraries
target_link_libraries(uuid PRIVATE mg_utility)
