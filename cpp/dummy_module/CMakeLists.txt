set(dummy_src
    dummy.cpp)

add_query_module(dummy 1 ${dummy_src})

# Link external libraries
target_link_libraries(dummy PRIVATE mg_procedure mg_utility)
