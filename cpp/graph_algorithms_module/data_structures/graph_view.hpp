#pragma once

#include <set>
#include <vector>

#include "graph_data.hpp"

namespace graphdata {

/// Graph view interface.
///
/// Interface provides methods for fetching graph data.
/// There are two methods for changing variables on the edges:
/// SetVariableState and SetVariableValue
class GraphView {
 public:
  /// Destroys the object.
  virtual ~GraphView() = 0;

  /// Gets all graph nodes.
  ///
  /// @return Vector of nodes
  virtual const std::vector<Node> &Nodes() const = 0;

  /// Gets all graph edges.
  ///
  /// @return Vector of edges.
  virtual const std::vector<Edge> &Edges() const = 0;

  /// Gets the edges between two neighbour nodes.
  ///
  /// @param[in] first node id
  /// @param[in] second node id
  ///
  /// @return     Iterator range
  virtual std::vector<uint32_t> GetEdgesBetweenNodes(uint32_t first,
                                                     uint32_t second) const = 0;

  /// Gets all incident edges ids.
  ///
  /// @return all incident edges
  virtual const std::vector<uint32_t> &IncidentEdges(
      uint32_t node_id) const = 0;

  /// Gets neighbour nodes.
  ///
  /// @param[in] node_id target node id
  ///
  /// @return vector of neighbours
  virtual const std::vector<Neighbour> &Neighbours(uint32_t node_id) const = 0;

  /// Gets node with node id.
  ///
  /// @param[in] node_id node id
  ///
  /// @return target Node struct
  virtual const Node &GetNode(uint32_t node_id) const = 0;

  /// Gets Edge with edge id.
  ///
  /// @param[in] edge_id edge id
  ///
  /// @return Edge struct
  virtual const Edge &GetEdge(uint32_t edge_id) const = 0;
};

inline GraphView::~GraphView() {}

}  // namespace graphdata
