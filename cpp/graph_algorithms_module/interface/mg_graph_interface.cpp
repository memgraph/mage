#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <unordered_set>

#include "data_structures/graph.hpp"
#include "mg_interface.hpp"
#include "mg_procedure.h"

namespace {
template <typename T>
using PairOptional = std::optional<std::pair<bool, T>>;

const char *fieldID = "id";

PairOptional<uint32_t> ApplyNodeID(const mgp_vertex *vertex,
                                   mgp_memory *memory) {
  auto *id_val = mgp_vertex_get_property(vertex, fieldID, memory);
  if (id_val == nullptr) return std::nullopt;

  bool ret_value = true;
  uint32_t id;
  if (!mgp_value_is_int(id_val)) {
    // std::cout << "Node id is wrong, must be int: ";
    // PrintValue(id_val);
    // std::cout << std::endl;
    ret_value = false;
  } else {
    id = mgp_value_get_int(id_val);
  }
  mgp_value_destroy(id_val);
  return std::make_pair(ret_value, id);
}

// @throw std::bad_alloc
bool ApplyEdgeID(uint32_t edge_id, const mgp_edge *edge,
                 std::map<uint32_t, uint32_t> *mapping, mgp_memory *memory) {
  auto *var = mgp_edge_get_property(edge, fieldID, memory);
  if (var == nullptr) return false;
  if (mgp_value_is_int(var)) {
    auto value_int = mgp_value_get_int(var);
    (*mapping)[edge_id] = value_int;
  }
  mgp_value_destroy(var);
  return true;
}

bool CreateNode(graphdata::Graph *graph, const mgp_vertex *vertex,
                std::map<uint32_t, uint32_t> *mapping, mgp_memory *memory) {
  auto is_created_id = ApplyNodeID(vertex, memory);
  if (!is_created_id) return false;
  if (!is_created_id->first) return true;
  uint32_t node_id = is_created_id->second;

  uint32_t inner_id = graph->CreateNode();
  if (mapping->find(node_id) != mapping->end()) {
    // std::cout << "Duplicate node id: " << node_id << std::endl;
    return true;
  }

  (*mapping)[node_id] = inner_id;
  return true;
}

std::optional<bool> CreateEdge(uint32_t *edge_id, graphdata::Graph *graph,
                               const mgp_vertex *from, const mgp_vertex *to,
                               const mgp_edge *edge,
                               std::map<uint32_t, uint32_t> *mapping,
                               mgp_memory *memory) {
  auto is_created = ApplyNodeID(from, memory);
  if (!is_created) return std::nullopt;
  if (!is_created->first) return false;
  uint32_t from_id = is_created->second;

  is_created = ApplyNodeID(to, memory);
  if (!is_created) return std::nullopt;
  if (!is_created->first) return false;
  uint32_t to_id = is_created->second;

  if (mapping->find(from_id) == mapping->end()) return false;
  if (mapping->find(to_id) == mapping->end()) return false;

  uint32_t from_inner = (*mapping)[from_id];
  uint32_t to_inner = (*mapping)[to_id];

  *edge_id = graph->CreateEdge(from_inner, to_inner);

  return true;
}
}  // namespace

namespace mg_interface {

void GetGraphView(graphdata::Graph *g, std::map<uint32_t, uint32_t> &node_mapping,
                  std::map<uint32_t, uint32_t> &edge_mapping,
                  const mgp_graph *graph, mgp_result *result,
                  mgp_memory *memory) {
  auto *vertices_it = mgp_graph_iter_vertices(graph, memory);
  if (vertices_it == nullptr) {
    NotEnoughMemory(result);
    return;
  }

  for (const auto *vertex = mgp_vertices_iterator_get(vertices_it); vertex;
       vertex = mgp_vertices_iterator_next(vertices_it)) {
    bool memory_ok = CreateNode(g, vertex, &node_mapping, memory);
    if (!memory_ok) {
      mgp_vertices_iterator_destroy(vertices_it);
      NotEnoughMemory(result);
      return;
    }
  }
  mgp_vertices_iterator_destroy(vertices_it);

  vertices_it = mgp_graph_iter_vertices(graph, memory);
  if (vertices_it == nullptr) {
    NotEnoughMemory(result);
    return;
  }

  for (const auto *vertex = mgp_vertices_iterator_get(vertices_it); vertex;
       vertex = mgp_vertices_iterator_next(vertices_it)) {
    auto *edges_it = mgp_vertex_iter_out_edges(vertex, memory);
    if (edges_it == nullptr) {
      mgp_vertices_iterator_destroy(vertices_it);
      NotEnoughMemory(result);
      return;
    }

    for (const auto *out_edge = mgp_edges_iterator_get(edges_it); out_edge;
         out_edge = mgp_edges_iterator_next(edges_it)) {
      uint32_t created_edge;
      auto check =
          CreateEdge(&created_edge, g, vertex, mgp_edge_get_to(out_edge),
                     out_edge, &node_mapping, memory);

      if (!check) {
        mgp_edges_iterator_destroy(edges_it);
        mgp_vertices_iterator_destroy(vertices_it);
        NotEnoughMemory(result);
        return;
      }

      if (*check) {
        bool is_created = false;
        try {
          is_created =
              ApplyEdgeID(created_edge, out_edge, &edge_mapping, memory);
        } catch (const std::exception &e) {
          mgp_edges_iterator_destroy(edges_it);
          mgp_vertices_iterator_destroy(vertices_it);
          mgp_result_set_error_msg(result, e.what());
          return;
        }
        if (!is_created) {
          mgp_edges_iterator_destroy(edges_it);
          mgp_vertices_iterator_destroy(vertices_it);
          NotEnoughMemory(result);
          return;
        }
      } else {
        std::cout << "Edge is not created!" << std::endl;
      }
    }

    mgp_edges_iterator_destroy(edges_it);
  }

  mgp_vertices_iterator_destroy(vertices_it);
}

bool InsertStringValue(mgp_result_record *record, const char *field_name,
                       const char *string_value, mgp_memory *memory) {
  mgp_value *value = mgp_value_make_string(string_value, memory);
  if (value == nullptr) return false;

  int result_inserted = mgp_result_record_insert(record, field_name, value);

  mgp_value_destroy(value);
  if (!result_inserted) return result_inserted;

  return true;
}

bool InsertIntValue(mgp_result_record *record, const char *field_name,
                    const int int_value, mgp_memory *memory) {
  mgp_value *value = mgp_value_make_int(int_value, memory);
  if (value == nullptr) return false;

  int result_inserted = mgp_result_record_insert(record, field_name, value);

  mgp_value_destroy(value);
  if (!result_inserted) return result_inserted;

  return true;
}

bool InsertNodeValue(const mgp_graph *graph, mgp_result_record *record,
                     const char *field_name, const int node_id,
                     mgp_memory *memory) {
  mgp_vertex *vertex = mgp_graph_get_vertex_by_id(
      graph, mgp_vertex_id{.as_int = node_id}, memory);

  return InsertNodeValue(record, field_name, vertex, memory);
}

bool InsertNodeValue(mgp_result_record *record, const char *field_name,
                     mgp_vertex *vertex_value, mgp_memory *memory) {
  mgp_value *value = mgp_value_make_vertex(vertex_value);
  if (value == nullptr) return false;

  int result_inserted = mgp_result_record_insert(record, field_name, value);

  mgp_value_destroy(value);
  if (!result_inserted) return result_inserted;

  return true;
}

bool InsertRelationshipValue(mgp_result_record *record, const char *field_name,
                             mgp_edge *edge_value, mgp_memory *memory) {
  mgp_value *value = mgp_value_make_edge(edge_value);
  if (value == nullptr) return false;

  int result_inserted = mgp_result_record_insert(record, field_name, value);

  mgp_value_destroy(value);
  if (!result_inserted) return result_inserted;

  return true;
}

bool InsertRelationshipValue(const mgp_graph *graph, mgp_result_record *record,
                             const char *field_name, const int edge_id,
                             mgp_memory *memory) {
  //TODO: Needs implementation of edge fetching by ID in Memgraph to work.
  return true;
}

void NotEnoughMemory(mgp_result *result) {
  mgp_result_set_error_msg(result, "Not enough memory!");
}

}  // namespace mg_interface