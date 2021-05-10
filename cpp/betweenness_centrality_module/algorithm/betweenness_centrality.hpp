#pragma once

#include <stack>
#include <vector>

#include <mg_graph.hpp>

namespace betweenness_centrality_util {

void BFS(const std::uint64_t source_node, const mg_graph::GraphView<> &graph,
        std::stack<std::uint64_t> &visited, std::vector<std::vector<std::uint64_t>> &predecessors,
        std::vector<std::uint64_t> &shortest_paths_counter);

}  // namespace betweenness_centrality_util


namespace betweenness_centrality_alg {

std::vector<double> BetweennessCentrality(const mg_graph::GraphView<> &graph, bool directed=true, bool normalized=true);

}  // namespace betweenness_centrality_alg

