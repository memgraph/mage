set(louvain_module_src 
    louvain_module.cpp
    algorithm/louvain.cpp)

add_subdirectory(grappolo)
add_query_module(louvain 1 "${louvain_module_src}")

target_include_directories(louvain PRIVATE ${GRAPPOLO_HEADERS} ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(louvain PRIVATE mg_utility  basic_cd util full_syn_opt coloring OpenMP::OpenMP_CXX)