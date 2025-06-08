#pragma once

#include <mg_graph.hpp>

namespace degree_centrality_alg {

enum class AlgorithmType { kUndirected = 0, kOut = 1, kIn = 2 };

std::vector<double> GetDegreeCentrality(const mg_graph::GraphView<> &graph, const AlgorithmType algorithm_type);

}  // namespace degree_centrality_alg
