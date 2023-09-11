set(schema_module_src
    schema_module.cpp
    algorithm/schema.cpp)

add_query_module(schema 1 "${schema_module_src}")
