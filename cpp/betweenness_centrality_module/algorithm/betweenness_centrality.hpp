#pragma once

#include <mg_graph.hpp>

namespace betweenness_centrality_util {

void BFS (const std::uint64_t source_node, const std::vector<std::vector<std::uint64_t>> &adj_list,
            std::stack<std::uint64_t> &visited, std::vector<std::vector<std::uint64_t>> &predecessors,
            std::vector<std::uint64_t> &shortest_paths_counter);

void mapNodes(const mg_graph::GraphView<> &graph,
            std::unordered_map<std::uint64_t, std::uint64_t> &node_to_index,
            std::vector<std::uint64_t> &index_to_node);

void UnweightedDirectedGraph(const mg_graph::GraphView<> &graph,
                            std::vector<std::vector<std::uint64_t>> &adj_list,
                            std::vector<std::uint64_t> &index_to_node);

void UnweightedUndirectedGraph(const mg_graph::GraphView<> &graph,
                            std::vector<std::vector<std::uint64_t>> &adj_list,
                            std::vector<std::uint64_t> &index_to_node);

std::vector<double> calculateBetweennessUnweightedDirected(const std::vector<std::vector<std::uint64_t>> &adj_list);
std::vector<double> calculateBetweennessUnweightedUndirected(const std::vector<std::vector<std::uint64_t>> &adj_list);
}  // namespace betweenness_centrality_util


namespace betweenness_centrality_alg {

std::unordered_map<std::uint64_t, double> BetweennessCentralityUnweightedDirected(const mg_graph::GraphView<> &graph);
std::unordered_map<std::uint64_t, double> BetweennessCentralityUnweightedUndirected(const mg_graph::GraphView<> &graph);

}  // namespace betweenness_centrality_alg