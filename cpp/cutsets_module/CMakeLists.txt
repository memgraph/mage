set(cutsets_src 
    cutsets_module.cpp
    algorithm/cutsets.cpp)

add_query_module(cutsets 1 "${cutsets_src}")

# Link external libraries
target_link_libraries(cutsets PRIVATE mg_utility)
target_include_directories(cutsets PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Module tests
set(cutsets_test_src 
    cutsets_test.cpp
    algorithm/cutsets.cpp)

add_executable(cutsets_test  "${cutsets_test_src}")
target_link_libraries(cutsets_test PRIVATE mg_utility gtest)