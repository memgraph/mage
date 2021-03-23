set(mg_utility_src mg_utils.cpp mg_graph.cpp)

add_library(mg_utility SHARED ${mg_utility_src})
target_include_directories(mg_utility PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mg_utility PRIVATE mg_procedure)


set(mg_graph_src mg_graph.cpp)

add_library(mg_graph SHARED ${mg_graph_src})
target_include_directories(mg_graph PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})