include(GNUInstallDirs)

# Add all module files related to graph util module
set(periodic_src
    periodic.cpp)

add_query_module(periodic 1 "${periodic_src}")

target_compile_definitions(periodic PRIVATE MGCLIENT_STATIC_DEFINE)
target_include_directories(periodic PRIVATE ${MGCLIENT_INCLUDE_DIRS})
target_link_libraries(periodic PRIVATE mgclient fmt::fmt)
