#pragma once

#include <mg_graph.hpp>

#include <cstdint>
#include <stack>
#include <unordered_map>
#include <unordered_set>

namespace bcc_utility {

/// Simple struct that keeps the state of nodes in algorithms that rely on
/// DFS traversal.
struct NodeState {
  std::unordered_map<uint64_t, bool> visited;
  std::unordered_map<uint64_t, uint64_t> discovery, low_link;
  std::uint64_t counter;

  explicit NodeState(std::uint64_t number_of_nodes);
  void Update(std::uint64_t node_id);
};

///
///@brief DFS Algorithm for obtaining the biconnected components within the graph
///
///@param node_id Starting Node ID
///@param parent_id Parental Node ID
///@param state Current node state
///@param edge_stack Current state of edge stack
///@param bcc_edges Current biconnected components
///@param graph Graph to work on
///
void BccDFS(std::uint64_t node_id, std::uint64_t parent_id, bcc_utility::NodeState *state,
            std::stack<mg_graph::Edge<>> *edge_stack, std::vector<std::vector<mg_graph::Edge<>>> *bcc_edges,
            std::vector<std::unordered_set<std::uint64_t>> *bcc_nodes, const mg_graph::GraphView<> &graph,
            std::unordered_set<uint64_t> &articulationPoints);

}  // namespace bcc_utility

namespace bcc_algorithm {

///
///@brief Method for getting all of the biconnected components inside of a GraphView
///
///@param graph GraphView object
///@return std::vector<std::vector<mg_graph::Edge<>>>
///
std::vector<std::vector<mg_graph::Edge<>>> GetBiconnectedComponents(
    const mg_graph::GraphView<> &graph, std::unordered_set<uint64_t> &articulationPoints,
    std::vector<std::unordered_set<std::uint64_t>> &bcc_nodes);

}  // namespace bcc_algorithm
