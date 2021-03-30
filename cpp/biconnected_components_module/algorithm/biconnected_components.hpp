#pragma once

#include <stack>

#include <mg_graph.hpp>

namespace bcc_utility {

/// Simple struct that keeps the state of nodes in algorithms that rely on
/// DFS traversal.
struct NodeState {
  std::vector<bool> visited;
  std::vector<int64_t> discovery, low_link, depth;
  std::vector<uint64_t> parent;
  uint32_t counter;
};

///
///@brief DFS Algorithm for obtaining the biconnected components within the graph
///
///@param node_id Starting Node ID
///@param parent_id Parental Node ID
///@param state Current node state
///@param edge_stack Current state of edge stack
///@param bcc Current biconnected components
///@param graph Graph to work on
///
void BccDfs(uint64_t node_id, uint64_t parent_id, bcc_utility::NodeState *state,
            std::stack<mg_graph::Edge<>> *edge_stack, std::vector<std::vector<mg_graph::Edge<>>> *bcc,
            const mg_graph::GraphView<> *graph);

}  // namespace bcc_utility

namespace bcc_algorithm {

///
///@brief Method for getting all of the biconnected components inside of a GraphView
///
///@param graph GraphView object
///@return std::vector<std::vector<mg_graph::Edge<>>>
///
std::vector<std::vector<mg_graph::Edge<>>> GetBiconnectedComponents(const mg_graph::GraphView<> *graph);

}  // namespace bcc_algorithm
