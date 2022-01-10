#pragma once

#include <vector>

#include "graph_data.hpp"

namespace mg_graph {

/// Graph view interface.
///
/// Interface provides methods for fetching graph data.
/// There are two methods for changing variables on the edges:
/// SetVariableState and SetVariableValue
template <typename TSize = std::uint64_t>
class GraphView {
  static_assert(std::is_unsigned_v<TSize>,
                "mg_graph::GraphView expects the type to be an unsigned integer type\n"
                "only (uint8_t, uint16_t, uint32_t, or uint64_t).");

 public:
  using TNode = Node<TSize>;
  using TEdge = Edge<TSize>;
  using TNeighbour = Neighbour<TSize>;

  /// Destroys the object.
  virtual ~GraphView() = 0;

  /// Gets all graph nodes.
  ///
  /// @return Vector of nodes
  virtual const std::vector<TNode> &Nodes() const = 0;

  /// Gets all graph edges.
  ///
  /// @return Vector of edges.
  virtual const std::vector<TEdge> &Edges() const = 0;

  /// Gets the edges between two neighbour nodes.
  ///
  /// @param[in] first node id
  /// @param[in] second node id
  ///
  /// @return     Iterator range
  virtual std::vector<TSize> GetEdgesBetweenNodes(TSize first, TSize second) const = 0;

  /// Gets all incident edges ids.
  ///
  /// @return all incident edges
  virtual const std::vector<TSize> &IncidentEdges(TSize node_id) const = 0;

  /// Gets neighbour nodes.
  ///
  /// @param[in] node_id target node id
  ///
  /// @return vector of neighbours
  virtual const std::vector<TNeighbour> &Neighbours(TSize node_id) const = 0;

  /// Gets out-neighbour nodes.
  ///
  /// @param[in] node_id target node id
  ///
  /// @return vector of neighbours
  virtual const std::vector<TNeighbour> &OutNeighbours(TSize node_id) const = 0;

  /// Gets in-neighbour nodes.
  ///
  /// @param[in] node_id target node id
  ///
  /// @return vector of neighbours
  virtual const std::vector<TNeighbour> &InNeighbours(TSize node_id) const = 0;

  /// Gets node with node id.
  ///
  /// @param[in] node_id node id
  ///
  /// @return target Node struct
  virtual const TNode &GetNode(TSize node_id) const = 0;

  /// Gets Edge with edge id.
  ///
  /// @param[in] edge_id edge id
  ///
  /// @return Edge struct
  virtual const TEdge &GetEdge(TSize edge_id) const = 0;

  virtual std::uint64_t GetMemgraphNodeId(TSize node_id) const = 0;

  virtual TSize GetInnerNodeId(std::uint64_t memgraph_id) const = 0;

  virtual bool NodeExists(std::uint64_t memgraph_id) const = 0;
};

template <typename TSize>
inline GraphView<TSize>::~GraphView() {}

}  // namespace mg_graph
