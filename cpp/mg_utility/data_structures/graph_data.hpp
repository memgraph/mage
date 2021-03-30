/// @file graph_data.hpp
/// This file contains the graph data - nodes, edges, variables.

#pragma once

namespace mg_graph {

/// Node representation.
///
/// Node has id and type. Id must be unique in the graph.
template <typename TSize = uint64_t> struct Node { TSize id; };

/// Edge representation.
///
/// @var id edge id
/// @var from node
/// @var to node
/// @var type edge type
/// @var variables set of variables on edge with the state
template <typename TSize = uint64_t> struct Edge {
  TSize id;
  TSize from;
  TSize to;
};

/// Neighbour representation.
///
/// Helper structure for storing node and edge id.
template <typename TSize = uint64_t> struct Neighbour {
  TSize node_id;
  TSize edge_id;
  Neighbour(TSize n_id, TSize e_id) : node_id(n_id), edge_id(e_id) {}
};
} // namespace mg_graph
