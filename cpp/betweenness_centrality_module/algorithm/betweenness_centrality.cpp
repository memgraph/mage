#include <unordered_map>
#include <unordered_multiset>
#include <stack>

#include "betweenness_centrality.hpp"

std::unordered_map<std::uint64_t, double> betweenness_centrality::BetweennessCentralityUnweighted(const mg_graph::GraphView<> &graph){
    std::unordered_map<std::uint64_t, double> betweenness_centrality;
    for (auto node : graph.Nodes()) {
        betweenness_centrality[node.id] = 0;
    }

    for (auto node : graph.Nodes()) {
        std::unordered_map<std::uint64_t, std::unordered_multiset<std::unit64_t>> predecessors;
        std::unordered_map<std::uint64_t, std::uint64_t> shortest_paths_counter;
        std::unordered_map<std::uint64_t, std::uint64_t> distance;
        std::unordered_map<std::uint64_t, double> dependency;
        for (auto n : graph.Nodes()) {
            predecessors[n.id] = std::unordered_multiset<std::uint64_t>();
            shortest_paths_counter[n.id] = 0;
            distance[n.id] = -1;
            dependency[n.id] = 0;
        }
        shortest_paths_counter[node.id] = 1;
        distance[node.id] = 0;
        std::uint64_t phase = 0;
        std::vector<std::vector<std::uint64_t>> stacks;
        stacks.push_back(std::vector<std::uint64_t>());
        stacks[phase].push_back(node.id);
        std::uint64_t counter = 1;

        while (counter > 0)
        {
            counter = 0;
            for (auto node_id : stacks[phase]){
                for(auto neighbor : graph.Neighbours(node_id)){
                    auto neighbor_id = neihgbor.node_id;
                    if (distance[neighbor_id] < 0) {
                        if (stacks.size() < phase + 1) stacks.push_back(std::vector<std::uint64_t>());
                        stacks[phase+1].push_back(neighbor_id);
                        counter++;
                        distance[neighbor_id] = distance[node_id] + 1;
                    }
                    if (distance[neighbor_id] == distance[node_id] + 1) {
                        shortest_paths_counter[neighbor_id] += shortest_paths_counter[node_id];
                        predecessors[neighbor_id].insert(node_id); 
                    }
                }
            }
            phase++;
        }
        phase--;

        while (phase > 0)
        {
            for (auto node_id : stacks[phase]) {
                for (auto p : predecessors[node_id]) {
                    dependency[p] += ((double)shortest_paths_counter[p] / shortest_paths_counter[node_id]) * (1 + dependency[node_id]);
                }
                betweenness_centrality[node_id] += dependency[node_id];
            }
            phase--;
        }
    
    }

    return betweenness_centrality;

}