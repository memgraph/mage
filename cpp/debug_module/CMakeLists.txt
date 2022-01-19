add_query_module(debug 1 debug_module.cpp)
target_link_libraries(debug PRIVATE mg_utility)
target_include_directories(debug PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
