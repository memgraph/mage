#pragma once

#include <stack>

#include <mg_graph.hpp>

namespace bridges_util {

/// Simple struct that keeps the state of nodes in algorithms that rely on
/// DFS traversal.
struct NodeState {
  std::vector<bool> visited;
  std::vector<int32_t> discovery, low_link, depth;
  std::vector<uint32_t> parent;
  uint64_t counter;
};

void BridgeDfs(uint64_t node_id, uint64_t parent_id, bridges_util::NodeState *state,
               std::vector<mg_graph::Edge<>> *bridges, const mg_graph::GraphView<> *G);
}  // namespace bridges_util

namespace bridges_alg {

std::vector<mg_graph::Edge<>> GetBridges(const mg_graph::GraphView<> *G);

}  // namespace bridges_alg