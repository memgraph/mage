#pragma once

#include <stdint.h>
#include <queue>

#include <mg_graph.hpp>

namespace bipartite_matching_util {
bool BipartiteMatchinDFS(uint64_t node, const std::vector<std::vector<uint64_t>> &adj_list, std::vector<bool> &visited,
                         std::vector<uint32_t> &matched);
}  // namespace bipartite_matching_util

namespace bipartite_matching_alg {
bool IsGraphBipartite(const mg_graph::GraphView<> *G);

bool IsSubgraphBipartite(const mg_graph::GraphView<> *G, std::vector<int64_t> colors, uint64_t node_index);

uint64_t BipartiteMatching(const std::vector<std::pair<uint64_t, uint64_t>> &edges);

uint64_t BipartiteMatching(const mg_graph::GraphView<> *G);
}  // namespace bipartite_matching_alg
