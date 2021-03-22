/// @file graph.hpp
/// This file contains the graph definitions.

#pragma once

#include "graph_data.hpp"
#include "graph_view.hpp"
#include <map>
#include <unordered_map>

namespace mg_graph {

/// Graph representation.
class Graph : public GraphView {
public:
  /// Create object Graph
  explicit Graph();

  /// Destroys the object.
  ~Graph();

  /// Gets all graph nodes.
  ///
  /// @return Vector of nodes
  const std::vector<Node> &Nodes() const override;

  /// Gets all graph edges.
  ///
  /// This method always returns a complete list of edges. If some edges are
  /// deleted with EraseEdge method, this method will return also deleted edges.
  /// Deleted edges will have invalid edge id. Method CheckIfEdgeValid is used
  /// for checking the edge. Recommendation is to use EraseEdge method only in
  /// test cases.
  /// @return Vector of edges.
  const std::vector<Edge> &Edges() const override;

  /// Gets the edges between two neighbour nodes.
  ///
  /// @param[in] first node id
  /// @param[in] second node id
  ///
  /// @return     Iterator range
  std::vector<uint32_t> GetEdgesBetweenNodes(uint32_t first,
                                             uint32_t second) const override;

  /// Gets all incident edges ids.
  ///
  /// @return all incident edges
  const std::vector<uint32_t> &IncidentEdges(uint32_t node_id) const override;

  /// Gets neighbour nodes.
  ///
  /// @param[in] node_id target node id
  ///
  /// @return vector of neighbours
  const std::vector<Neighbour> &Neighbours(uint32_t node_id) const override;

  /// Gets node with node id.
  ///
  /// @param[in] node_id node id
  ///
  /// @return target Node struct
  const Node &GetNode(uint32_t node_id) const override;

  /// Gets Edge with edge id.
  ///
  /// @param[in] edge_id edge id
  ///
  /// @return Edge struct
  const Edge &GetEdge(uint32_t edge_id) const override;

  /// Creates a node.
  ///
  /// @return     Created node id
  uint32_t CreateNode(uint32_t memgraph_id);

  /// Creates an edge.
  ///
  /// Creates an undirected edge in the graph, but edge will contain information
  /// about the original directed property.
  ///
  /// @param[in]  from  The from node identifier
  /// @param[in]  to    The to node identifier
  ///
  /// @return     Created edge id
  uint32_t CreateEdge(uint32_t from, uint32_t to);

  /// Gets all valid edges.
  ///
  /// Edge is valid if is not deleted with EraseEdge method.
  ///
  /// @return Vector of valid edges
  std::vector<Edge> ExistingEdges() const;

  /// Checks if edge is valid.
  ///
  /// Edge is valid if is created and if is not deleted.
  ///
  /// @return true if edge is valid, otherwise returns false
  bool IsEdgeValid(uint32_t edge_id) const;

  /// Removes edge from graph.
  ///
  /// Recommendation is to use this method only in the tests.
  ///
  /// @param[in] u node id of node on same edge
  /// @param[in] v node id of node on same edge
  void EraseEdge(uint32_t u, uint32_t v);

  ///
  /// Returns the Memgraph database ID from graph view
  ///
  /// @param node_id view's inner ID
  ///
  uint32_t GetMemgraphNodeId(uint32_t node_id);

  /// Removes all edges and nodes from graph.
  void Clear();

private:
  // Constant is used for marking deleted edges.
  // If edge id is equal to constant, edge is deleted.
  static const uint32_t kDeletedEdgeId = std::numeric_limits<uint32_t>::max();

  std::vector<std::vector<uint32_t>> adj_list_;
  std::vector<std::vector<Neighbour>> neighbours_;

  std::vector<Node> nodes_;
  std::vector<Edge> edges_;
  std::unordered_map<uint32_t, uint32_t> inner_to_memgraph_id_;
  std::unordered_map<uint32_t, uint32_t> memgraph_to_inner_id_;

  std::multimap<std::pair<uint32_t, uint32_t>, uint32_t> nodes_to_edge_;
};
} // namespace mg_graph
