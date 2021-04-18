#pragma once

#include <stack>

#include <mg_graph.hpp>

namespace bcc_utility {

/// Simple struct that keeps the state of nodes in algorithms that rely on
/// DFS traversal.
struct NodeState {
  std::vector<bool> visited;
  std::vector<std::uint64_t> discovery, low_link, depth;
  std::vector<std::uint64_t> parent;
  uint64_t counter;

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
void BccDFS(std::uint64_t node_id, std::uint64_t parent_id, bcc_utility::NodeState *state,
            std::stack<mg_graph::Edge<>> *edge_stack, std::vector<std::vector<mg_graph::Edge<>>> *bcc,
            const mg_graph::GraphView<> &graph);

}  // namespace bcc_utility

namespace bcc_algorithm {

///
///@brief Method for getting all of the biconnected components inside of a GraphView
///
///@param graph GraphView object
///@return std::vector<std::vector<mg_graph::Edge<>>>
///
std::vector<std::vector<mg_graph::Edge<>>> GetBiconnectedComponents(const mg_graph::GraphView<> &graph);

}  // namespace bcc_algorithm
