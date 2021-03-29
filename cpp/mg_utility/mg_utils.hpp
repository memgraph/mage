/// @file mg_utils.hpp
///
/// The file contains methods that connect mg procedures and the outside code
/// Methods like mapping a graph into memory or assigning new mg results or
/// their properties are implemented.
#pragma once

#include <functional>

#include "mg_graph.hpp"
#include "mg_procedure.h"

namespace mg_graph {

template <typename TSize>
void CreateGraphNode(mg_graph::Graph<TSize> *graph, const mgp_vertex *vertex) {
  // Get Memgraph internal ID property
  auto id_val = mgp_vertex_get_id(vertex);
  TSize memgraph_id = id_val.as_int;

  graph->CreateNode(memgraph_id);
}

template <typename TSize>
void CreateGraphEdge(mg_graph::Graph<TSize> *graph,
                     const mgp_vertex *vertex_from,
                     const mgp_vertex *vertex_to) {
  // Get Memgraph internal ID property
  TSize memgraph_id_from = mgp_vertex_get_id(vertex_from).as_int;
  TSize memgraph_id_to = mgp_vertex_get_id(vertex_to).as_int;
  graph->CreateEdge(memgraph_id_from, memgraph_id_to);
}
} // namespace mg_graph

namespace mg_utility {

/// Calls a function in it's destructor (on scope exit).
///
/// Example usage:
///
/// void long_function() {
///   resource.enable();
///   // long block of code, might throw an exception
///   resource.disable(); // we want this to happen for sure, and function end
/// }
///
/// Can be nicer and safer:
///
/// void long_function() {
///   resource.enable();
///   OnScopeExit on_exit([&resource] { resource.disable(); });
///   // long block of code, might trow an exception
/// }
class OnScopeExit {
public:
  explicit OnScopeExit(const std::function<void()> &function)
      : function_(function) {}
  ~OnScopeExit() { function_(); }

private:
  std::function<void()> function_;
};

///@brief Method for mapping the Memgraph in-memory graph into user-based graph
/// with iterative node indices. Graph object stores information about node and
/// edge mappings, alongside with connection information
///
///@param graph Memgraph graph
///@param result Memgraph result object
///@param memory Memgraph storage object
///@return mg_graph::Graph
///
template <typename TSize = uint64_t>
mg_graph::Graph<TSize> *GetGraphView(const mgp_graph *memgraph_graph,
                                     mgp_result *result, mgp_memory *memory) {
  mg_graph::Graph<TSize> *graph = new mg_graph::Graph<TSize>();

  ///
  /// Mapping Memgraph in-memory vertices into graph view
  ///

  // Safe creation of vertices iterator
  auto *vertices_it = mgp_graph_iter_vertices(memgraph_graph, memory);
  if (vertices_it == nullptr) {
    throw mg_exception::NotEnoughMemoryException();
  }
  mg_utility::OnScopeExit delete_vertices_it([&vertices_it] {
    if (vertices_it != nullptr) {
      mgp_vertices_iterator_destroy(vertices_it);
    }
  });

  // Iterate trough Memgraph vertices and map them to GraphView
  for (const auto *vertex = mgp_vertices_iterator_get(vertices_it); vertex;
       vertex = mgp_vertices_iterator_next(vertices_it)) {
    mg_graph::CreateGraphNode(graph, vertex);
  }
  mgp_vertices_iterator_destroy(vertices_it);

  ///
  /// Mapping Memgraph in-memory edges into graph view
  ///

  // Safe creation of vertices iterator
  vertices_it = mgp_graph_iter_vertices(memgraph_graph, memory);
  if (vertices_it == nullptr) {
    throw mg_exception::NotEnoughMemoryException();
  }

  for (const auto *vertex_from = mgp_vertices_iterator_get(vertices_it);
       vertex_from; vertex_from = mgp_vertices_iterator_next(vertices_it)) {

    // Safe creation of edges iterator
    auto *edges_it = mgp_vertex_iter_out_edges(vertex_from, memory);

    if (edges_it == nullptr) {
      throw mg_exception::NotEnoughMemoryException();
    }
    mg_utility::OnScopeExit delete_edges_it([&edges_it] {
      if (edges_it != nullptr) {
        mgp_edges_iterator_destroy(edges_it);
      }
    });

    for (const auto *out_edge = mgp_edges_iterator_get(edges_it); out_edge;
         out_edge = mgp_edges_iterator_next(edges_it)) {
      auto vertex_to = mgp_edge_get_to(out_edge);
      mg_graph::CreateGraphEdge(graph, vertex_from, vertex_to);
    }
  }

  return graph;
}

/// Inserts a string of value string_value to the field field_name of
/// the record mgp_result_record record.
void InsertStringValueResult(mgp_result_record *record, const char *field_name,
                             const char *string_value, mgp_memory *memory) {
  mgp_value *value = mgp_value_make_string(string_value, memory);
  if (value == nullptr) {
    throw mg_exception::NotEnoughMemoryException();
  }

  int result_inserted = mgp_result_record_insert(record, field_name, value);

  mgp_value_destroy(value);
  if (!result_inserted) {
    throw mg_exception::NotEnoughMemoryException();
  }
}

/// Inserts an integer of value int_value to the field field_name of
/// the record mgp_result_record record.
void InsertIntValueResult(mgp_result_record *record, const char *field_name,
                          const int int_value, mgp_memory *memory) {
  mgp_value *value = mgp_value_make_int(int_value, memory);
  if (value == nullptr) {
    throw mg_exception::NotEnoughMemoryException();
  }

  int result_inserted = mgp_result_record_insert(record, field_name, value);

  mgp_value_destroy(value);
  if (!result_inserted) {
    throw mg_exception::NotEnoughMemoryException();
  }
}

/// Inserts a node of value vertex_value to the field field_name of
/// the record mgp_result_record record.
void InsertNodeValueResult(mgp_result_record *record, const char *field_name,
                           mgp_vertex *vertex_value, mgp_memory *memory) {
  mgp_value *value = mgp_value_make_vertex(vertex_value);
  if (value == nullptr) {
    throw mg_exception::NotEnoughMemoryException();
  }
  int result_inserted = mgp_result_record_insert(record, field_name, value);

  mgp_value_destroy(value);
  if (!result_inserted) {
    throw mg_exception::NotEnoughMemoryException();
  }
}

/// Inserts a node with its ID node_id to create a vertex and insert
/// the node to the field field_name of the record mgp_result_record record.
void InsertNodeValueResult(const mgp_graph *graph, mgp_result_record *record,
                           const char *field_name, const int node_id,
                           mgp_memory *memory) {
  mgp_vertex *vertex = mgp_graph_get_vertex_by_id(
      graph, mgp_vertex_id{.as_int = node_id}, memory);

  InsertNodeValueResult(record, field_name, vertex, memory);
}

/// Inserts a relationship of value edge_value to the field field_name of
/// the record mgp_result_record record.
void InsertRelationshipValueResult(mgp_result_record *record,
                                   const char *field_name, mgp_edge *edge_value,
                                   mgp_memory *memory);

/// Inserts a relationship with its ID edge_id to create a relationship and
/// insert the edge to the field field_name of the record mgp_result_record
/// record.
void InsertRelationshipValueResult(const mgp_graph *graph,
                                   mgp_result_record *record,
                                   const char *field_name, const int edge_id,
                                   mgp_memory *memory);
} // namespace mg_utility
