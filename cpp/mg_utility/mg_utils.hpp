/// @file mg_utils.hpp
///
/// The file contains methods that connect mg procedures and the outside code
/// Methods like mapping a graph into memory or assigning new mg results or
/// their properties are implemented.
#pragma once

#include "mg_graph.hpp"
#include <functional>
#include <mg_procedure.h>

namespace mg_utility {

///@brief Method for mapping the Memgraph in-memory graph into user-based graph
/// with iterative node indices. Graph object stores information about node and
/// edge mappings, alongside with connection information
///
///@param graph Memgraph graph
///@param result Memgraph result object
///@param memory Memgraph storage object
///@return mg_graph::Graph
///
mg_graph::Graph *GetGraphView(const mgp_graph *graph, mgp_result *result,
                              mgp_memory *memory);

/// Inserts a string of value string_value to the field field_name of
/// the record mgp_result_record record.
void InsertStringValueResult(mgp_result_record *record, const char *field_name,
                             const char *string_value, mgp_memory *memory);

/// Inserts an integer of value int_value to the field field_name of
/// the record mgp_result_record record.
void InsertIntValueResult(mgp_result_record *record, const char *field_name,
                          const int int_value, mgp_memory *memory);

/// Inserts a node of value vertex_value to the field field_name of
/// the record mgp_result_record record.
void InsertNodeValueResult(mgp_result_record *record, const char *field_name,
                           mgp_vertex *vertex_value, mgp_memory *memory);

/// Inserts a node with its ID node_id to create a vertex and insert
/// the node to the field field_name of the record mgp_result_record record.
void InsertNodeValueResult(const mgp_graph *graph, mgp_result_record *record,
                           const char *field_name, const int node_id,
                           mgp_memory *memory);

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
} // namespace mg_utility
