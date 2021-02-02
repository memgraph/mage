/// @file
///
/// The file contains function declarations of general graph algorithms. In our
/// case, these are the algorithms that don't rely on any domain knowledge
/// for solving the observability problem.

#pragma once

#include "data_structures/graph.hpp"
#include "mg_procedure.h"

namespace mg_interface {
void GetGraphView(graphdata::Graph *g, std::map<uint32_t, uint32_t> &node_mapping,
                  std::map<uint32_t, uint32_t> &edge_mapping,
                  const mgp_graph *graph, mgp_result *result,
                  mgp_memory *memory);

bool InsertStringValue(mgp_result_record *record, const char *field_name,
                       const char *string_value, mgp_memory *memory);

bool InsertIntValue(mgp_result_record *record, const char *field_name,
                    const int int_value, mgp_memory *memory);

bool InsertNodeValue(mgp_result_record *record, const char *field_name,
                     mgp_vertex *vertex_value, mgp_memory *memory);

bool InsertNodeValue(const mgp_graph *graph, mgp_result_record *record,
                     const char *field_name, const int node_id,
                     mgp_memory *memory);

bool InsertRelationshipValue(mgp_result_record *record, const char *field_name,
                             mgp_edge *edge_value, mgp_memory *memory);

bool InsertRelationshipValue(const mgp_graph *graph, mgp_result_record *record,
                             const char *field_name, const int edge_id,
                             mgp_memory *memory);

void NotEnoughMemory(mgp_result *result);
}  // namespace mg_interface
