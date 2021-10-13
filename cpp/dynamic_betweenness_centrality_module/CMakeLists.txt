
set(dynamic_betweenness_centrality_src
        dynamic_betweenness_centrality_test.cpp
        algorithm/dynamic_betweenness_centrality.cpp algorithm/dynamic_betweenness_centrality.hpp algorithm/bcc_utility.hpp algorithm/bcc_utiity.cpp)

add_executable(dynamic_betweenness_centrality "${dynamic_betweenness_centrality_src}")
target_link_libraries(dynamic_betweenness_centrality PRIVATE mg_utility)