set(testing_module_src
    text_module.cpp
    algorithm/text.cpp)

add_query_module(testing 1 "${testing_module_src}")
