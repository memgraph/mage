#pragma once

#include <mg_graph.hpp>

namespace bridges_util {

/// Simple struct that keeps the state of nodes in algorithms that rely on
/// DFS traversal.
struct NodeState {
  std::vector<bool> visited;
  std::vector<std::uint64_t> discovery, low_link;
  std::uint64_t counter;

  explicit NodeState(std::size_t number_of_nodes);

  void Update(std::uint64_t node_id);
};

void BridgeDfs(std::uint64_t node_id, std::uint64_t parent_id, bridges_util::NodeState *state,
               std::vector<mg_graph::Edge<>> *bridges, const mg_graph::GraphView<> &graph);
}  // namespace bridges_util

namespace bridges_alg {

std::vector<mg_graph::Edge<>> GetBridges(const mg_graph::GraphView<> &graph);

}  // namespace bridges_alg