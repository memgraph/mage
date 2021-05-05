#pragma once

#include <mg_graph.hpp>


namespace betweenness_centrality {

std::unordered_map<std::uint64_t, double> BetweennessCentralityUnweighted(const mg_graph::GraphView<> &graph);

}  // namespace betweenness_centrality