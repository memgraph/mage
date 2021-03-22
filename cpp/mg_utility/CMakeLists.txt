set(mg_utility_src mg_utils.cpp data_structures/graph.cpp)

add_library(mg_utility SHARED ${mg_utility_src})
target_include_directories(mg_utility PUBLIC ${CMAKE_SOURCE_DIR}/mg_utility)
target_include_directories(mg_utility PUBLIC ${CMAKE_SOURCE_DIR}/mg_utility/data_structures)
target_link_libraries(mg_utility PRIVATE mg_procedure)