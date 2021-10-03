/// @file mg_utils.hpp
///
/// The file contains methods that connect mg procedures and the outside code
/// Methods like mapping a graph into memory or assigning new mg results or
/// their properties are implemented.
#pragma once

#include <functional>
#include <memory>

#include "mg_graph.hpp"
#include "mgp.hpp"

namespace mg_graph {

///
///@brief Creates vertex inside GraphView. First step is getting Memgraph's UID for identifying vertex inside Memgraph
/// platform.
///
///@tparam TSize Parameter for storing vertex identifiers
///@param graph Memgraph's graph instance
///@param vertex Memgraph's vertex instance
///
template <typename TSize>
void CreateGraphNode(mg_graph::Graph<TSize> *graph, mgp_vertex *vertex) {
  // Get Memgraph internal ID property
  auto id_val = mgp::vertex_get_id(vertex);
  auto memgraph_id = id_val.as_int;

  graph->CreateNode(memgraph_id);
}

///
///@brief Creates edge within the GraphView. First step is getting Memgraph's UID for indetifying starting and ending
/// vertex in the edge.
///
///@tparam TSize Parameter for storing vertex identifiers
///@param graph Memgraph's graph instance
///@param vertex_from Memgraph's starting vertex instance
///@param vertex_to Memgraph's ending vertex instance
///@param graph_type Type of stored graph: Directed/Undirected
///
template <typename TSize>
void CreateGraphEdge(mg_graph::Graph<TSize> *graph, mgp_vertex *vertex_from, mgp_vertex *vertex_to,
                     const mg_graph::GraphType graph_type) {
  // Get Memgraph internal ID property
  auto memgraph_id_from = mgp::vertex_get_id(vertex_from).as_int;
  auto memgraph_id_to = mgp::vertex_get_id(vertex_to).as_int;

  graph->CreateEdge(memgraph_id_from, memgraph_id_to, graph_type);
}
}  // namespace mg_graph

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
  explicit OnScopeExit(const std::function<void()> &function) : function_(function) {}
  ~OnScopeExit() { function_(); }

 private:
  std::function<void()> function_;
};

///@brief Method for mapping the Memgraph in-memory graph into user-based graph
/// with iterative node indices. Graph object stores information about node and
/// edge mappings, alongside with connection information. Created ghraph is
/// zero-indexed, meaning that indices start with index 0.
///
///@param graph Memgraph graph
///@param result Memgraph result object
///@param memory Memgraph storage object
///@return mg_graph::Graph
///
template <typename TSize = std::uint64_t>
std::unique_ptr<mg_graph::Graph<TSize>> GetGraphView(mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory,
                                                     const mg_graph::GraphType graph_type) {
  auto graph = std::make_unique<mg_graph::Graph<TSize>>();

  ///
  /// Mapping Memgraph in-memory vertices into graph view
  ///

  // Safe creation of vertices iterator

  auto *vertices_it = mgp::graph_iter_vertices(memgraph_graph, memory);
  if (vertices_it == nullptr) {
    throw mg_exception::NotEnoughMemoryException();
  }
  mg_utility::OnScopeExit delete_vertices_it([&vertices_it] { mgp::vertices_iterator_destroy(vertices_it); });

  // Iterate trough Memgraph vertices and map them to GraphView
  for (auto *vertex = mgp::vertices_iterator_get(vertices_it); vertex;
       vertex = mgp::vertices_iterator_next(vertices_it)) {
    mg_graph::CreateGraphNode(graph.get(), vertex);
  }

  ///
  /// Mapping Memgraph in-memory edges into graph view
  ///

  // Safe creation of vertices iterator
  vertices_it = mgp::graph_iter_vertices(memgraph_graph, memory);

  for (auto *vertex_from = mgp::vertices_iterator_get(vertices_it); vertex_from;
       vertex_from = mgp::vertices_iterator_next(vertices_it)) {
    // Safe creation of edges iterator
    auto *edges_it = mgp::vertex_iter_out_edges(vertex_from, memory);

    mg_utility::OnScopeExit delete_edges_it([&edges_it] { mgp::edges_iterator_destroy(edges_it); });

    for (auto *out_edge = mgp::edges_iterator_get(edges_it); out_edge; out_edge = mgp::edges_iterator_next(edges_it)) {
      auto vertex_to = mgp::edge_get_to(out_edge);
      mg_graph::CreateGraphEdge(graph.get(), vertex_from, vertex_to, graph_type);
    }
  }

  return graph;
}

namespace {
void InsertRecord(mgp_result_record *record, const char *field_name, mgp_value *value) {
  mg_utility::OnScopeExit delete_value([&value] { mgp::value_destroy(value); });
  mgp::result_record_insert(record, field_name, value);
}
}  // namespace

/// Inserts a string of value string_value to the field field_name of
/// the record mgp_result_record record.
void InsertStringValueResult(mgp_result_record *record, const char *field_name, const char *string_value,
                             mgp_memory *memory) {
  auto value = mgp::value_make_string(string_value, memory);
  InsertRecord(record, field_name, value);
}

/// Inserts an integer of value int_value to the field field_name of
/// the record mgp_result_record record.
void InsertIntValueResult(mgp_result_record *record, const char *field_name, const int int_value, mgp_memory *memory) {
  auto value = mgp::value_make_int(int_value, memory);
  InsertRecord(record, field_name, value);
}

/// Inserts a double of value double_value to the field field_name of
/// the record mgp_result_record record.
void InsertDoubleValue(mgp_result_record *record, const char *field_name, const double double_value,
                       mgp_memory *memory) {
  auto value = mgp::value_make_double(double_value, memory);
  InsertRecord(record, field_name, value);
}

/// Inserts a node of value vertex_value to the field field_name of
/// the record mgp_result_record record.
void InsertNodeValueResult(mgp_result_record *record, const char *field_name, mgp_vertex *vertex_value,
                           mgp_memory *memory) {
  auto value = mgp::value_make_vertex(vertex_value);
  InsertRecord(record, field_name, value);
}

/// Inserts a node with its ID node_id to create a vertex and insert
/// the node to the field field_name of the record mgp_result_record record.
void InsertNodeValueResult(mgp_graph *graph, mgp_result_record *record, const char *field_name, const int node_id,
                           mgp_memory *memory) {
  auto *vertex = mgp::graph_get_vertex_by_id(graph, mgp_vertex_id{.as_int = node_id}, memory);
  InsertNodeValueResult(record, field_name, vertex, memory);
}

/// Inserts a relationship of value edge_value to the field field_name of
/// the record mgp_result_record record.
void InsertRelationshipValueResult(mgp_result_record *record, const char *field_name, mgp_edge *edge_value,
                                   mgp_memory *memory) {
  auto value = mgp::value_make_edge(edge_value);
  InsertRecord(record, field_name, value);
}

/// Inserts a relationship with its ID edge_id to create a relationship and
/// insert the edge to the field field_name of the record mgp_result_record
/// record.
void InsertRelationshipValueResult(mgp_graph *graph, mgp_result_record *record, const char *field_name,
                                   const int edge_id, mgp_memory *memory);
}  // namespace mg_utility
