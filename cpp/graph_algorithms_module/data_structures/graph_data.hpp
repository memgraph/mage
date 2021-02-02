/// @file graph_data.hpp
/// This file contains the graph data - nodes, edges, variables.

#pragma once

#include <stdint.h>

#include <map>
#include <string>

namespace graphdata {

/// Node representation.
///
/// Node has id and type. Id must be unique in the graph.
struct Node {
  uint32_t id;
};

/// Edge representation.
///
/// @var id edge id
/// @var from node
/// @var to node
/// @var type edge type
/// @var variables set of variables on edge with the state
struct Edge {
  uint32_t id;
  uint32_t from;
  uint32_t to;
};

/// Neighbour representation.
///
/// Helper structure for storing node and edge id.
struct Neighbour {
  uint32_t node_id;
  uint32_t edge_id;
  Neighbour(uint32_t n_id, uint32_t e_id) : node_id(n_id), edge_id(e_id) {}
};
}  // namespace graphdata
