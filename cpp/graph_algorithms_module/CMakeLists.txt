set(CYCLES_MODULE cycles_module.cpp)

set(SOURCES algorithms/biconnected_components.cpp
            algorithms/bipartite_matching.cpp
            algorithms/bridges.cpp
            algorithms/cutsets.cpp
            algorithms/cutsets_bf.cpp
            algorithms/cycles.cpp
            data_structures/graph.cpp
            interface/mg_graph_interface.cpp
            )

include_directories(.)

add_library(graph_util STATIC ${SOURCES})
target_include_directories(graph_util PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_query_module(cycles_module 1 ${CYCLES_MODULE})
target_link_libraries(cycles_module graph_util)

#TODO: Setup this in the mage repo, not finished yet
