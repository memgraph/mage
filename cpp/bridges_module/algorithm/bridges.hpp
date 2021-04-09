#pragma once

#include <mg_graph.hpp>

namespace bridges_util {

/// Simple struct that keeps the state of nodes in algorithms that rely on
/// DFS traversal.
struct NodeState {
  std::vector<bool> visited;
  std::vector<std::uint64_t> discovery, low_link, depth;
  std::vector<std::uint64_t> parent;
  std::uint64_t counter;

  explicit NodeState(std::uint64_t number_of_nodes) {
    visited.resize(number_of_nodes, false);
    discovery.resize(number_of_nodes, 0);
    low_link.resize(number_of_nodes, 0);
    counter = 0;
  }

  void Update(std::uint64_t node_id) {
    counter++;
    visited[node_id] = true;
    discovery[node_id] = counter;
    low_link[node_id] = counter;
  }
};

void BridgeDfs(uint64_t node_id, uint64_t parent_id, bridges_util::NodeState *state,
               std::vector<mg_graph::Edge<>> *bridges, const mg_graph::GraphView<> &graph);
}  // namespace bridges_util

namespace bridges_alg {

std::vector<mg_graph::Edge<>> GetBridges(const mg_graph::GraphView<> &graph);

}  // namespace bridges_alg