include(GNUInstallDirs)

# Add all module files related to graph util module
set(execute_src
    execute.cpp)

add_query_module(execute 1 "${execute_src}")

target_link_libraries(execute PRIVATE fmt::fmt)
