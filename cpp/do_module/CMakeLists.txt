# Add all module files related to graph util module
set(do_module_src
    do_module.cpp)

add_query_module(do_cpp 1 "${do_module_src}")

target_include_directories(do_cpp PRIVATE ${MGCLIENT_INCLUDE_DIRS})
target_link_libraries(do_cpp PRIVATE mgclient fmt::fmt)
