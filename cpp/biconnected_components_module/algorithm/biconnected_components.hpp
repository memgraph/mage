#pragma once

#include <mg_graph.hpp>
#include <stack>

namespace bcc_utility {

/// Simple struct that keeps the state of nodes in algorithms that rely on
/// DFS traversal.
struct NodeState {
  std::vector<bool> visited;
  std::vector<int32_t> discovery, low_link, depth;
  std::vector<uint32_t> parent;
};

void BccDfs(uint32_t node_id, uint32_t parent_id, bcc_utility::NodeState *state,
            std::stack<mg_graph::Edge> *edge_stack,
            std::vector<std::vector<mg_graph::Edge>> *bcc,
            const mg_graph::GraphView *graph);

} // namespace bcc_utility

namespace bcc_algorithm {

std::vector<std::vector<mg_graph::Edge>>
GetBiconnectedComponents(const mg_graph::GraphView *graph);

} // namespace bcc_algorithm
