set(csv_import_src 
    csv_import_module.cpp
    algorithm/csv_import.cpp)

add_query_module(csv_import 1 "${csv_import_src}")
