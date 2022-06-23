set(simple_math_module_src
    simple_math_module.cpp)

add_query_module(simple_math 1 "${simple_math_module_src}")

# Link external libraries
target_link_libraries(simple_math PRIVATE mg_utility)
target_include_directories(simple_math PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
