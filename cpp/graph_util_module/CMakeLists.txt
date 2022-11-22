# Add all module files related to graph util module
set(graph_util_src
    graph_util_module.cpp)

add_query_module(graph_util 1 "${graph_util_src}")

# Link external libraries
target_link_libraries(graph_util PRIVATE mg_utility)
target_include_directories(graph_util PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
