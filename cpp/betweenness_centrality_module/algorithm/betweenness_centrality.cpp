#include <unordered_map>
#include <stack>
#include <queue>
#include <unordered_set>

#include "betweenness_centrality.hpp"

namespace betweenness_centrality_util {

void BFS(const std::uint64_t source_node, const std::vector<std::vector<std::uint64_t>> &adj_list,
        std::stack<std::uint64_t> &visited, std::vector<std::vector<std::uint64_t>> &predecessors,
        std::vector<std::uint64_t> &shortest_paths_counter){
    
    auto graph_size = adj_list.size();

    // -1 to indicate that node is not visited
    std::vector<int> distance (graph_size, -1);

    shortest_paths_counter[source_node] = 1;
    distance[source_node] = 0;

    std::queue<std::uint64_t> BFS_queue; 
    BFS_queue.push(source_node);

    while(!BFS_queue.empty()){
        auto current_node = BFS_queue.front();
        BFS_queue.pop();
        visited.push(current_node);

        for(auto neighbor : adj_list[current_node]) {

            //node found for the first time 
            if (distance[neighbor] < 0) {
                BFS_queue.push(neighbor);
                distance[neighbor] = distance[current_node] + 1;
            }

            //shortest path from node to neighbor_id goes through current_node 
            if (distance[neighbor] = distance[current_node] + 1) {
                shortest_paths_counter[neighbor] += shortest_paths_counter[current_node];
                predecessors[neighbor].emplace_back(current_node);
            }
        }
    }
}

void mapNodes(const mg_graph::GraphView<> &graph,
            std::unordered_map<std::uint64_t, std::uint64_t> &node_to_index,
            std::vector<std::uint64_t> &index_to_node){
    std::uint64_t id = 0;
    for (auto node : graph.Nodes()) {
        node_to_index[node.id] = id;
        index_to_node[id]=node.id;
        id++;
    }
}

void UnweightedDirectedGraph(const mg_graph::GraphView<> &graph, 
                            std::vector<std::vector<std::uint64_t>> &adj_list,
                            std::vector<std::uint64_t> &index_to_node) {

    std::unordered_map<std::uint64_t, std::uint64_t> node_to_index;
    betweenness_centrality_util::mapNodes(graph, node_to_index, index_to_node);

    for (const auto &edge : graph.Edges()) {
        auto from_index = node_to_index[edge.from];
        auto to_index = node_to_index[edge.to];
        adj_list[from_index].emplace_back(to_index);
    }
}

void UnweightedUndirectedGraph(const mg_graph::GraphView<> &graph,
                                std::vector<std::vector<std::uint64_t>> &adj_list,
                                std::vector<std::uint64_t> &index_to_node) {

    std::unordered_map<std::uint64_t, std::uint64_t> node_to_index;
    betweenness_centrality_util::mapNodes(graph, node_to_index, index_to_node);

    // data structure used to efficiently discard multiple edges between nodes
    // direction of an edge is ignored
    std::vector<std::unordered_set<std::uint64_t>> adj_helper (adj_list.size(), std::unordered_set<std::uint64_t>());
    for (const auto &edge : graph.Edges()) {
        auto from_index = node_to_index[edge.from];
        auto to_index = node_to_index[edge.to];
        adj_helper[from_index].insert(to_index);
        adj_helper[to_index].insert(from_index); 
    }

    for (std::uint64_t id = 0; id < adj_list.size(); id++) {
        std::copy(adj_helper[id].begin(), adj_helper[id].end(), std::back_inserter(adj_list[id]));
    }
}

std::vector<double> calculateBetweennessUnweightedDirected(const std::vector<std::vector<std::uint64_t>> &adj_list){
    auto graph_size = adj_list.size();
    std::vector<double> betweenness_centrality (graph_size, 0);

    // perform bfs for every node in the graph 
    for (std::uint64_t node = 0; node < graph_size; node++) {

        // data structures used in BFS
        std::stack<std::uint64_t> visited;
        std::vector<std::vector<std::uint64_t>> predecessors (graph_size, std::vector<std::uint64_t>());
        std::vector<std::uint64_t> shortest_paths_counter (graph_size, 0);
        betweenness_centrality_util::BFS(node, adj_list, visited, predecessors, shortest_paths_counter);


        std::vector<double> dependency (graph_size, 0);

        while(!visited.empty()){
            auto current_node = visited.top();
            visited.pop();

            for (auto p : predecessors[current_node]) {
                double fraction = (double)shortest_paths_counter[p] / shortest_paths_counter[current_node];
                dependency[p] += fraction * (1 + dependency[current_node]);
            }

            if (current_node != node){
                betweenness_centrality[current_node] += dependency[current_node];
            }
        }
    }

    return betweenness_centrality;
}

std::vector<double> calculateBetweennessUnweightedUndirected(const std::vector<std::vector<std::uint64_t>> &adj_list){
    auto graph_size = adj_list.size();
    std::vector<double> betweenness_centrality (graph_size, 0);

    // Perform bfs for every node in the graph 
    for (std::uint64_t node = 0; node < graph_size; node++) {
        // data structures used in BFS
        std::stack<std::uint64_t> visited;
        std::vector<std::vector<std::uint64_t>> predecessors (graph_size, std::vector<std::uint64_t>());
        std::vector<std::uint64_t> shortest_paths_counter (graph_size, 0);
        betweenness_centrality_util::BFS(node, adj_list, visited, predecessors, shortest_paths_counter);

        std::vector<double> dependency (graph_size, 0);

        while(!visited.empty()){
            auto current_node = visited.top();
            visited.pop();

            for (auto p : predecessors[current_node]) {
                double fraction = (double)shortest_paths_counter[p] / shortest_paths_counter[current_node];
                dependency[p] += fraction * (1 + dependency[current_node]);
            }

            if (current_node != node){
                //centrality scores need to be divided by two since all shortest paths are considered twice
                betweenness_centrality[current_node] += dependency[current_node] / 2;
            }
        }
    }

    return betweenness_centrality;
}

}  // namespace betweenness_centrality_util


namespace betweenness_centrality_alg {
std::unordered_map<std::uint64_t, double> BetweennessCentralityUnweightedDirected(const mg_graph::GraphView<> &graph){

    std::vector<std::vector<std::uint64_t>> adj_list (graph.Nodes().size(), std::vector<std::uint64_t>());
    std::vector<std::uint64_t> index_to_node;
    betweenness_centrality_util::UnweightedDirectedGraph(graph, adj_list, index_to_node);

    std::vector<double> betweenness = betweenness_centrality_util::calculateBetweennessUnweightedDirected(adj_list);

    std::unordered_map<std::uint64_t, double> betweenness_centrality;
    for(std::uint64_t id = 0; id < betweenness.size(); id++){
        betweenness_centrality[index_to_node[id]] = betweenness[id];
    }
    return betweenness_centrality;
}

std::unordered_map<std::uint64_t, double> BetweennessCentralityUnweightedUndirected(const mg_graph::GraphView<> &graph){

    std::vector<std::vector<std::uint64_t>> adj_list (graph.Nodes().size(), std::vector<std::uint64_t>());
    std::vector<std::uint64_t> index_to_node;
    betweenness_centrality_util::UnweightedUndirectedGraph(graph, adj_list, index_to_node);

    std::vector<double> betweenness = betweenness_centrality_util::calculateBetweennessUnweightedDirected(adj_list);

    std::unordered_map<std::uint64_t, double> betweenness_centrality;
    for(std::uint64_t id = 0; id < betweenness.size(); id++){
        betweenness_centrality[index_to_node[id]] = betweenness[id];
    }
    return betweenness_centrality;
}

}  // namespace betweenness_centrality_alg