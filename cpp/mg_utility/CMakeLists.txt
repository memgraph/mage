add_library(mg_utility INTERFACE)
target_include_directories(mg_utility INTERFACE ${CMAKE_SOURCE_DIR}/mg_utility ${CMAKE_SOURCE_DIR}/memgraph/include)
# target_include_directories(mg_procedure INTERFACE )
# target_link_libraries(mg_utility INTERFACE mg_procedure)