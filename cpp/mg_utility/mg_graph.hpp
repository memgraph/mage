/// @file graph.hpp
/// This file contains the graph definitions.

#pragma once

#include <algorithm>
#include <map>
#include <unordered_map>

#include "data_structures/graph_data.hpp"
#include "data_structures/graph_view.hpp"
#include "mg_exceptions.hpp"

namespace mg_graph {

/// Graph representation.
template <typename TSize = std::uint64_t>
class Graph : public GraphView<TSize> {

  using typename GraphView<TSize>::TNode;
  using typename GraphView<TSize>::TEdge;
  using typename GraphView<TSize>::TNeighbour;

  static_assert(
      std::is_unsigned_v<TSize>,
      "mg_graph::Graph expects the type to be an unsigned integer type\n"
      "only (uint8_t, uint16_t, uint32_t, or uint64_t).");

public:
  /// Create object Graph
  explicit Graph() = default;

  /// Destroys the object.
  ~Graph() = default;

  /// Gets all graph nodes.
  ///
  /// @return Vector of nodes
  const std::vector<TNode> &Nodes() const override { return nodes_; }

  /// Gets all graph edges.
  ///
  /// This method always returns a complete list of edges. If some edges are
  /// deleted with EraseEdge method, this method will return also deleted edges.
  /// Deleted edges will have invalid edge id. Method CheckIfEdgeValid is used
  /// for checking the edge. Recommendation is to use EraseEdge method only in
  /// test cases.
  /// @return Vector of edges.
  const std::vector<TEdge> &Edges() const override { return edges_; }

  /// Gets the edges between two neighbour nodes.
  ///
  /// @param[in] first node id
  /// @param[in] second node id
  ///
  /// @return     Iterator range
  std::vector<TSize> GetEdgesBetweenNodes(TSize first,
                                          TSize second) const override {
    std::vector<TSize> edge_ids;
    const auto [range_start, range_end] =
        nodes_to_edge_.equal_range(std::minmax(first, second));
    edge_ids.reserve(std::distance(range_start, range_end));
    for (auto it = range_start; it != range_end; ++it) {
      if (IsEdgeValid(it->second)) {
        edge_ids.push_back(it->second);
      }
    }
    return edge_ids;
  }

  /// Gets all incident edges ids.
  ///
  /// @return all incident edges
  const std::vector<TSize> &IncidentEdges(TSize node_id) const override {
    if (node_id < 0 && node_id >= nodes_.size()) {
      throw mg_exception::InvalidIDException();
    }

    return adj_list_[node_id];
  }

  /// Gets neighbour nodes.
  ///
  /// @param[in] node_id target node id
  ///
  /// @return vector of neighbours
  const std::vector<TNeighbour> &Neighbours(TSize node_id) const override {
    if (node_id < 0 && node_id >= nodes_.size()) {
      throw mg_exception::InvalidIDException();
    }

    return neighbours_[node_id];
  }

  /// Gets node with node id.
  ///
  /// @param[in] node_id node id
  ///
  /// @return target Node struct
  const TNode &GetNode(TSize node_id) const override {
    if (node_id < 0 && node_id >= nodes_.size()) {
      throw mg_exception::InvalidIDException();
    }

    return nodes_[node_id];
  }

  /// Gets Edge with edge id.
  ///
  /// @param[in] edge_id edge id
  ///
  /// @return Edge struct
  const TEdge &GetEdge(TSize edge_id) const override {
    if (edge_id < 0 && edge_id >= edges_.size()) {
      throw mg_exception::InvalidIDException();
    }
    return edges_[edge_id];
  }

  /// Creates a node.
  ///
  /// @return     Created node id
  TSize CreateNode(std::uint64_t memgraph_id) {
    auto id = nodes_.size();
    nodes_.push_back({id});
    adj_list_.emplace_back();
    neighbours_.emplace_back();

    inner_to_memgraph_id_.emplace(id, memgraph_id);
    memgraph_to_inner_id_.emplace(memgraph_id, id);
    return id;
  }

  /// Creates an edge.
  ///
  /// Creates an undirected edge in the graph, but edge will contain information
  /// about the original directed property.
  ///
  /// @param[in]  from  The from node identifier
  /// @param[in]  to    The to node identifier
  ///
  /// @return     Created edge id
  TSize CreateEdge(std::uint64_t memgraph_id_from,
                   std::uint64_t memgraph_id_to) {
    auto from = memgraph_to_inner_id_[memgraph_id_from];
    auto to = memgraph_to_inner_id_[memgraph_id_to];

    if (from < 0 || to < 0 || from >= nodes_.size() || to >= nodes_.size()) {
      throw mg_exception::InvalidIDException();
    }
    auto id = edges_.size();
    edges_.push_back({id, from, to});
    adj_list_[from].push_back(id);
    adj_list_[to].push_back(id);
    neighbours_[from].emplace_back(to, id);
    neighbours_[to].emplace_back(from, id);
    nodes_to_edge_.insert({std::minmax(from, to), id});
    return id;
  }

  /// Gets all valid edges.
  ///
  /// Edge is valid if is not deleted with EraseEdge method.
  ///
  /// @return Vector of valid edges
  std::vector<TEdge> ExistingEdges() const {
    std::vector<TEdge> edges_out;
    edges_out.reserve(edges_.size());
    for (const auto &edge : edges_) {
      if (edge.id == Graph::k_deleted_edge_id_)
        continue;
      edges_out.push_back(edge);
    }
    return edges_out;
  }

  /// Checks if edge is valid.
  ///
  /// Edge is valid if is created and if is not deleted.
  ///
  /// @return true if edge is valid, otherwise returns false
  bool IsEdgeValid(TSize edge_id) const {
    if (edge_id < 0 || edge_id >= edges_.size()) {
      return false;
    }
    if (edges_[edge_id].id == k_deleted_edge_id_) {
      return false;
    }
    return true;
  }

  /// Removes edge from graph.
  ///
  /// Recommendation is to use this method only in the tests.
  ///
  /// @param[in] node_from node id of node on same edge
  /// @param[in] node_to node id of node on same edge
  void EraseEdge(TSize node_from, TSize node_to) {
    if (node_from < 0 && node_from >= nodes_.size()) {
      throw mg_exception::InvalidIDException();
    }
    if (node_to < 0 && node_to >= nodes_.size()) {
      throw mg_exception::InvalidIDException();
    }

    auto it = nodes_to_edge_.find(std::minmax(node_from, node_to));
    if (it == nodes_to_edge_.end()) {
      return;
    }

    auto edge_id = it->second;

    for (auto it = adj_list_[node_from].begin();
         it != adj_list_[node_from].end(); ++it) {

      if (edges_[*it].to == node_to || edges_[*it].from == node_to) {
        edges_[*it].id = Graph::k_deleted_edge_id_;
        adj_list_[node_from].erase(it);
        break;
      }
    }
    for (auto it = adj_list_[node_to].begin(); it != adj_list_[node_to].end();
         ++it) {

      if (edges_[*it].to == node_from || edges_[*it].from == node_from) {
        edges_[*it].id = Graph::k_deleted_edge_id_;
        adj_list_[node_to].erase(it);
        break;
      }
    }

    for (auto it = neighbours_[node_from].begin();
         it != neighbours_[node_from].end(); ++it) {

      if (it->edge_id == edge_id) {
        neighbours_[node_from].erase(it);
        break;
      }
    }
    for (auto it = neighbours_[node_to].begin();
         it != neighbours_[node_to].end(); ++it) {

      if (it->edge_id == edge_id) {
        neighbours_[node_to].erase(it);
        break;
      }
    }
  }

  ///
  /// Returns the Memgraph database ID from graph view
  ///
  /// @param node_id view's inner ID
  ///
  TSize GetMemgraphNodeId(std::uint64_t node_id) {
    if (inner_to_memgraph_id_.find(node_id) == inner_to_memgraph_id_.end()) {
      throw mg_exception::InvalidIDException();
    }
    return inner_to_memgraph_id_[node_id];
  }

  /// Removes all edges and nodes from graph.
  void Clear() {
    adj_list_.clear();
    nodes_.clear();
    edges_.clear();
    nodes_to_edge_.clear();
    neighbours_.clear();

    memgraph_to_inner_id_.clear();
    inner_to_memgraph_id_.clear();
  }

private:
  // Constant is used for marking deleted edges.
  // If edge id is equal to constant, edge is deleted.
  static constexpr inline TSize k_deleted_edge_id_ =
      std::numeric_limits<TSize>::max();

  std::vector<std::vector<TSize>> adj_list_;
  std::vector<std::vector<TNeighbour>> neighbours_;

  std::vector<TNode> nodes_;
  std::vector<TEdge> edges_;
  std::unordered_map<TSize, std::uint64_t> inner_to_memgraph_id_;
  std::unordered_map<std::uint64_t, TSize> memgraph_to_inner_id_;

  std::multimap<std::pair<TSize, TSize>, TSize> nodes_to_edge_;
};
} // namespace mg_graph
