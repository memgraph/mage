set(text_utils_module_src
    text_utils_module.cpp
    algorithm/text.cpp)

add_query_module(text_utils 1 "${text_utils_module_src}")
target_link_libraries(text_utils PRIVATE fmt::fmt)
