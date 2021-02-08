/// @file
///
/// The file contains methods that connect mg procedures and the outside code
/// Methods like mapping a graph into memory or assigning new mg results or
/// their properties are implemented.

#pragma once

#include "data_structures/graph.hpp"
#include "mg_procedure.h"

namespace mg_interface {
/// The method maps the mg graph into memory and assigns vertexes and edges
/// to the graphdata::Graph *g variable. Node mapping and edge mapping are
/// 2 maps that are used with keys of the newly created node and edge IDs
/// while the values represent inner mg IDs of the same nodes and edges.
void GetGraphView(graphdata::Graph *g,
                  std::map<uint32_t, uint32_t> &node_mapping,
                  std::map<uint32_t, uint32_t> &edge_mapping,
                  const mgp_graph *graph, mgp_result *result,
                  mgp_memory *memory);

/// Inserts a string of value string_value to the field field_name of
/// the record mgp_result_record record.
bool InsertStringValue(mgp_result_record *record, const char *field_name,
                       const char *string_value, mgp_memory *memory);

/// Inserts an integer of value int_value to the field field_name of
/// the record mgp_result_record record.
bool InsertIntValue(mgp_result_record *record, const char *field_name,
                    const int int_value, mgp_memory *memory);

/// Inserts a node of value vertex_value to the field field_name of
/// the record mgp_result_record record.
bool InsertNodeValue(mgp_result_record *record, const char *field_name,
                     mgp_vertex *vertex_value, mgp_memory *memory);

/// Inserts a node with its ID node_id to create a vertex and insert
/// the node to the field field_name of the record mgp_result_record record.
bool InsertNodeValue(const mgp_graph *graph, mgp_result_record *record,
                     const char *field_name, const int node_id,
                     mgp_memory *memory);

/// Inserts a relationship of value edge_value to the field field_name of
/// the record mgp_result_record record.
bool InsertRelationshipValue(mgp_result_record *record, const char *field_name,
                             mgp_edge *edge_value, mgp_memory *memory);

/// Inserts a relationship with its ID edge_id to create a relationship and
/// insert the edge to the field field_name of the record mgp_result_record
/// record.
bool InsertRelationshipValue(const mgp_graph *graph, mgp_result_record *record,
                             const char *field_name, const int edge_id,
                             mgp_memory *memory);

/// Work done if not enough memory to assign the error message to result.
void NotEnoughMemory(mgp_result *result);
}  // namespace mg_interface
