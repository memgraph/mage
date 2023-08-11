set(text_module_src
    text_module.cpp
    algorithm/text.cpp)

add_query_module(text 1 "${text_module_src}")
