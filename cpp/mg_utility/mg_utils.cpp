#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <unordered_set>

#include "mg_exceptions.hpp"
#include "mg_graph.hpp"
#include "mg_procedure.h"
#include "mg_utils.hpp"

namespace mg_graph {

void CreateGraphNode(mg_graph::Graph *graph, const mgp_vertex *vertex) {
  // Get Memgraph internal ID property
  auto id_val = mgp_vertex_get_id(vertex);
  uint32_t memgraph_id = id_val.as_int;

  graph->CreateNode(memgraph_id);
}

void CreateGraphEdge(mg_graph::Graph *graph, const mgp_vertex *vertex_from,
                     const mgp_vertex *vertex_to) {
  // Get Memgraph internal ID property
  uint32_t memgraph_id_from = mgp_vertex_get_id(vertex_from).as_int;
  uint32_t memgraph_id_to = mgp_vertex_get_id(vertex_to).as_int;
  graph->CreateEdge(memgraph_id_from, memgraph_id_to);
}
} // namespace mg_graph

namespace mg_utility {

mg_graph::Graph *GetGraphView(const mgp_graph *memgraph_graph,
                              mgp_result *result, mgp_memory *memory) {

  mg_graph::Graph *graph = new mg_graph::Graph();

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

void InsertNodeValueResult(const mgp_graph *graph, mgp_result_record *record,
                           const char *field_name, const int node_id,
                           mgp_memory *memory) {

  mgp_vertex *vertex = mgp_graph_get_vertex_by_id(
      graph, mgp_vertex_id{.as_int = node_id}, memory);

  InsertNodeValueResult(record, field_name, vertex, memory);
}

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

void InsertRelationshipValue(mgp_result_record *record, const char *field_name,
                             mgp_edge *edge_value, mgp_memory *memory) {
  mgp_value *value = mgp_value_make_edge(edge_value);
  if (value == nullptr) {
    throw mg_exception::NotEnoughMemoryException();
  }

  int result_inserted = mgp_result_record_insert(record, field_name, value);

  mgp_value_destroy(value);
  if (!result_inserted) {
    throw mg_exception::NotEnoughMemoryException();
  }
}

void InsertRelationshipValue(const mgp_graph *graph, mgp_result_record *record,
                             const char *field_name, const int edge_id,
                             mgp_memory *memory) {
  // TODO: Needs implementation of edge fetching by ID in Memgraph to work.
}

} // namespace mg_utility