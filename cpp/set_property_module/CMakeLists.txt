include(GNUInstallDirs)

set(set_property_module_src
    set_property_module.cpp)

add_query_module(set_property 1 "${set_property_module_src}")

target_compile_definitions(set_property PRIVATE MGCLIENT_STATIC_DEFINE)
target_include_directories(set_property PRIVATE ${MGCLIENT_INCLUDE_DIRS})
target_link_libraries(set_property PRIVATE mgclient fmt::fmt)
