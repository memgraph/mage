set(randomwalk_module_src
    algorithm/random_walk.cpp)

add_query_module(randomwalk 1 "${randomwalk_module_src}")

# Link external libraries
target_link_libraries(randomwalk PRIVATE mg_utility)
target_include_directories(randomwalk PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
