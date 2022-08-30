set(api_test_module_src
    api_test_module.cpp)

add_query_module(api_test_module 1 ${api_test_module_src})

# Link external libraries
target_link_libraries(api_test_module PRIVATE mg_utility)
