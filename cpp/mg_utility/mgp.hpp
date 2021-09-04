/// @file mg_utils.hpp
///
/// The file contains methods that connect mg procedures and the outside code
/// Methods like mapping a graph into memory or assigning new mg results or
/// their properties are implemented.
#pragma once

#include "mg_exceptions.hpp"
#include "mg_procedure.h"

namespace mgp {

namespace {
void MgExceptionHandle(mgp_error_code result_code) {
  switch (result_code) {
    case MGP_ERROR_UNKNOWN_ERROR:
      throw mg_exception::UnknownException();
    case MGP_ERROR_UNABLE_TO_ALLOCATE:
      throw mg_exception::AllocationException();
    case MGP_ERROR_INSUFFICIENT_BUFFER:
      throw mg_exception::InsufficientBufferException();
    case MGP_ERROR_OUT_OF_RANGE:
      throw mg_exception::OutOfRangeException();
    case MGP_ERROR_LOGIC_ERROR:
      throw mg_exception::LogicException();
    case MGP_ERROR_NON_EXISTENT_OBJECT:
      throw mg_exception::NonExistendObjectException();
    case MGP_ERROR_INVALID_ARGUMENT:
      throw mg_exception::InvalidArgumentException();
    default:
      return;
  }
}
template <typename TResult, typename TFunc, typename... TArgs>
TResult MgInvoke(TFunc func, TArgs... args) {
  TResult result{};

  auto result_code = func(args..., &result);
  MgExceptionHandle(result_code);

  return result;
}

template <typename TFunc, typename... TArgs>
void MgInvokeVoid(TFunc func, TArgs... args) {
  auto result_code = func(args...);
  MgExceptionHandle(result_code);
}
}  // namespace

void value_destroy(struct mgp_value *val) { mgp_value_destroy(val); }

struct mgp_value *value_make_null(struct mgp_memory *memory) {
  return MgInvoke<mgp_value *>(mgp_value_make_null, memory);
}

struct mgp_value *value_make_bool(int val, struct mgp_memory *memory) {
  return MgInvoke<mgp_value *>(mgp_value_make_bool, val, memory);
}

struct mgp_value *value_make_int(int64_t val, struct mgp_memory *memory) {
  return MgInvoke<mgp_value *>(mgp_value_make_int, val, memory);
}

struct mgp_value *value_make_double(double val, struct mgp_memory *memory) {
  return MgInvoke<mgp_value *>(mgp_value_make_double, val, memory);
}

struct mgp_value *value_make_string(const char *val, struct mgp_memory *memory) {
  return MgInvoke<mgp_value *>(mgp_value_make_string, val, memory);
}

struct mgp_value *value_make_list(struct mgp_list *val) {
  return MgInvoke<mgp_value *>(mgp_value_make_list, val);
}

struct mgp_value *value_make_map(struct mgp_map *val) {
  return MgInvoke<mgp_value *>(mgp_value_make_map, val);
}

struct mgp_value *value_make_vertex(struct mgp_vertex *val) {
  return MgInvoke<mgp_value *>(mgp_value_make_vertex, val);
}

struct mgp_value *value_make_edge(struct mgp_edge *val) {
  return MgInvoke<mgp_value *>(mgp_value_make_edge, val);
}

struct mgp_value *value_make_path(struct mgp_path *val) {
  return MgInvoke<mgp_value *>(mgp_value_make_path, val);
}

enum mgp_value_type value_get_type(const struct mgp_value *val) {
  return MgInvoke<enum mgp_value_type>(mgp_value_get_type, val);
}

bool value_is_null(const struct mgp_value *val) { return MgInvoke<int>(mgp_value_is_null, val); }

bool value_is_bool(const struct mgp_value *val) { return MgInvoke<int>(mgp_value_is_bool, val); }

bool value_is_int(const struct mgp_value *val) { return MgInvoke<int>(mgp_value_is_int, val); }

bool value_is_double(const struct mgp_value *val) { return MgInvoke<int>(mgp_value_is_double, val); }

bool value_is_string(const struct mgp_value *val) { return MgInvoke<int>(mgp_value_is_string, val); }

bool value_is_list(const struct mgp_value *val) { return MgInvoke<int>(mgp_value_is_list, val); }

bool value_is_map(const struct mgp_value *val) { return MgInvoke<int>(mgp_value_is_map, val); }

bool value_is_vertex(const struct mgp_value *val) { return MgInvoke<int>(mgp_value_is_vertex, val); }

bool value_is_edge(const struct mgp_value *val) { return MgInvoke<int>(mgp_value_is_edge, val); }

bool value_is_path(const struct mgp_value *val) { return MgInvoke<int>(mgp_value_is_path, val); }

bool value_get_bool(const struct mgp_value *val) { return MgInvoke<int>(mgp_value_get_bool, val); }

int64_t value_get_int(const struct mgp_value *val) { return MgInvoke<int64_t>(mgp_value_get_int, val); }

double value_get_double(const struct mgp_value *val) { return MgInvoke<double>(mgp_value_get_double, val); }

const char *value_get_string(const struct mgp_value *val) { return MgInvoke<const char *>(mgp_value_get_string, val); }

struct mgp_list *value_get_list(const struct mgp_value *val) {
  return MgInvoke<struct mgp_list *>(mgp_value_get_list, val);
}

struct mgp_map *value_get_map(const struct mgp_value *val) {
  return MgInvoke<struct mgp_map *>(mgp_value_get_map, val);
}

struct mgp_vertex *value_get_vertex(const struct mgp_value *val) {
  return MgInvoke<struct mgp_vertex *>(mgp_value_get_vertex, val);
}

struct mgp_edge *value_get_edge(const struct mgp_value *val) {
  return MgInvoke<struct mgp_edge *>(mgp_value_get_edge, val);
}

struct mgp_path *value_get_path(const struct mgp_value *val) {
  return MgInvoke<struct mgp_path *>(mgp_value_get_path, val);
}

struct mgp_list *list_make_empty(size_t capacity, struct mgp_memory *memory) {
  return MgInvoke<struct mgp_list *>(mgp_list_make_empty, capacity, memory);
}

void list_destroy(struct mgp_list *list) { mgp_list_destroy(list); }

void list_append(struct mgp_list *list, const struct mgp_value *val) { MgInvokeVoid(mgp_list_append, list, val); }

void list_append_extend(struct mgp_list *list, const struct mgp_value *val) {
  MgInvokeVoid(mgp_list_append_extend, list, val);
}

size_t list_size(const struct mgp_list *list) { return MgInvoke<size_t>(mgp_list_size, list); }

size_t list_capacity(const struct mgp_list *list) { return MgInvoke<size_t>(mgp_list_capacity, list); }

struct mgp_value *list_at(struct mgp_list *list, size_t index) {
  return MgInvoke<struct mgp_value *>(mgp_list_at, list, index);
}

struct mgp_map *map_make_empty(struct mgp_memory *memory) {
  return MgInvoke<struct mgp_map *>(mgp_map_make_empty, memory);
}

void map_destroy(struct mgp_map *map) { mgp_map_destroy(map); }

void map_insert(struct mgp_map *map, const char *key, const struct mgp_value *value) {
  MgInvokeVoid(mgp_map_insert, map, key, value);
}

size_t map_size(const struct mgp_map *map) { return MgInvoke<size_t>(mgp_map_size, map); }
struct mgp_value *map_at(struct mgp_map *map, const char *key) {
  return MgInvoke<struct mgp_value *>(mgp_map_at, map, key);
}

// const char *map_item_key(const struct mgp_map_item *item) { return MgInvoke<const char *>(mgp_map_item_key, item); }

struct mgp_value *map_item_value(struct mgp_map_item *item) {
  return MgInvoke<struct mgp_value *>(mgp_map_item_value, item);
}

// struct mgp_map_items_iterator *map_iter_items(const struct mgp_map *map, struct mgp_memory *memory) {
//   return MgInvoke<struct mgp_map_items_iterator *>(mgp_map_iter_items, map, memory);
// }

void map_items_iterator_destroy(struct mgp_map_items_iterator *it) { mgp_map_items_iterator_destroy(it); }

// struct mgp_map_item *map_items_iterator_get(const struct mgp_map_items_iterator *it) {
//   return MgInvoke<struct mgp_map_item *>(mgp_map_items_iterator_get, it);
// }

struct mgp_map_item *map_items_iterator_next(struct mgp_map_items_iterator *it) {
  return MgInvoke<struct mgp_map_item *>(mgp_map_items_iterator_next, it);
}

struct mgp_path *path_make_with_start(const struct mgp_vertex *vertex, struct mgp_memory *memory) {
  return MgInvoke<struct mgp_path *>(mgp_path_make_with_start, vertex, memory);
}

struct mgp_path *path_copy(const struct mgp_path *path, struct mgp_memory *memory) {
  return MgInvoke<struct mgp_path *>(mgp_path_copy, path, memory);
}

void path_destroy(struct mgp_path *path) { mgp_path_destroy(path); }

void path_expand(struct mgp_path *path, struct mgp_edge *edge) { MgInvokeVoid(mgp_path_expand, path, edge); }

size_t path_size(const struct mgp_path *path) { return MgInvoke<size_t>(mgp_path_size, path); }

struct mgp_vertex *path_vertex_at(struct mgp_path *path, size_t index) {
  return MgInvoke<struct mgp_vertex *>(mgp_path_vertex_at, path, index);
}

struct mgp_edge *path_edge_at(struct mgp_path *path, size_t index) {
  return MgInvoke<struct mgp_edge *>(mgp_path_edge_at, path, index);
}

bool path_equal(struct mgp_path *p1, struct mgp_path *p2) { return MgInvoke<int>(mgp_path_equal, p1, p2); }

void result_set_error_msg(struct mgp_result *res, const char *error_msg) {
  MgInvokeVoid(mgp_result_set_error_msg, res, error_msg);
}

struct mgp_result_record *result_new_record(struct mgp_result *res) {
  return MgInvoke<struct mgp_result_record *>(mgp_result_new_record, res);
}

void result_record_insert(struct mgp_result_record *record, const char *field_name, const struct mgp_value *val) {
  MgInvokeVoid(mgp_result_record_insert, record, field_name, val);
}
void properties_iterator_destroy(struct mgp_properties_iterator *it) { mgp_properties_iterator_destroy(it); }

struct mgp_property *properties_iterator_get(struct mgp_properties_iterator *it) {
  return MgInvoke<struct mgp_property *>(mgp_properties_iterator_get, it);
}

struct mgp_property *properties_iterator_next(struct mgp_properties_iterator *it) {
  return MgInvoke<struct mgp_property *>(mgp_properties_iterator_next, it);
}

void edges_iterator_destroy(struct mgp_edges_iterator *it) { mgp_edges_iterator_destroy(it); }

struct mgp_vertex_id vertex_get_id(const struct mgp_vertex *v) {
  return MgInvoke<struct mgp_vertex_id>(mgp_vertex_get_id, v);
}

struct mgp_vertex *vertex_copy(const struct mgp_vertex *v, struct mgp_memory *memory) {
  return MgInvoke<struct mgp_vertex *>(mgp_vertex_copy, v, memory);
}

void vertex_destroy(struct mgp_vertex *v) { mgp_vertex_destroy(v); }

bool vertex_equal(const struct mgp_vertex *v1, const struct mgp_vertex *v2) {
  return MgInvoke<int>(mgp_vertex_equal, v1, v2);
}

size_t vertex_labels_count(const struct mgp_vertex *v) { return MgInvoke<size_t>(mgp_vertex_labels_count, v); }

struct mgp_label vertex_label_at(const struct mgp_vertex *v, size_t index) {
  return MgInvoke<struct mgp_label>(mgp_vertex_label_at, v, index);
}

bool vertex_has_label(const struct mgp_vertex *v, struct mgp_label label) {
  return MgInvoke<int>(mgp_vertex_has_label, v, label);
}

bool vertex_has_label_named(const struct mgp_vertex *v, const char *label_name) {
  return MgInvoke<int>(mgp_vertex_has_label_named, v, label_name);
}

struct mgp_value *vertex_get_property(const struct mgp_vertex *v, const char *property_name,
                                      struct mgp_memory *memory) {
  return MgInvoke<struct mgp_value *>(mgp_vertex_get_property, v, property_name, memory);
}

struct mgp_properties_iterator *vertex_iter_properties(struct mgp_vertex *v, struct mgp_memory *memory) {
  return MgInvoke<struct mgp_properties_iterator *>(mgp_vertex_iter_properties, v, memory);
}

struct mgp_edges_iterator *vertex_iter_in_edges(struct mgp_vertex *v, struct mgp_memory *memory) {
  return MgInvoke<struct mgp_edges_iterator *>(mgp_vertex_iter_in_edges, v, memory);
}

struct mgp_edges_iterator *vertex_iter_out_edges(const struct mgp_vertex *v, struct mgp_memory *memory) {
  return MgInvoke<struct mgp_edges_iterator *>(mgp_vertex_iter_out_edges, v, memory);
}

struct mgp_edge *edges_iterator_get(struct mgp_edges_iterator *it) {
  return MgInvoke<struct mgp_edge *>(mgp_edges_iterator_get, it);
}

struct mgp_edge *edges_iterator_next(struct mgp_edges_iterator *it) {
  return MgInvoke<struct mgp_edge *>(mgp_edges_iterator_next, it);
}

struct mgp_edge_id edge_get_id(const struct mgp_edge *e) {
  return MgInvoke<struct mgp_edge_id>(mgp_edge_get_id, e);
}

struct mgp_edge *edge_copy(const struct mgp_edge *e, struct mgp_memory *memory) {
  return MgInvoke<struct mgp_edge *>(mgp_edge_copy, e, memory);
}

void edge_destroy(struct mgp_edge *e) { mgp_edge_destroy(e); }

bool edge_equal(const struct mgp_edge *e1, const struct mgp_edge *e2) { return MgInvoke<int>(mgp_edge_equal, e1, e2); }

struct mgp_edge_type edge_get_type(const struct mgp_edge *e) {
  return MgInvoke<struct mgp_edge_type>(mgp_edge_get_type, e);
}

struct mgp_vertex *edge_get_from(struct mgp_edge *e) {
  return MgInvoke<struct mgp_vertex *>(mgp_edge_get_from, e);
}

struct mgp_vertex *edge_get_to(struct mgp_edge *e) {
  return MgInvoke<struct mgp_vertex *>(mgp_edge_get_to, e);
}

struct mgp_value *edge_get_property(const struct mgp_edge *e, const char *property_name, struct mgp_memory *memory) {
  return MgInvoke<struct mgp_value *>(mgp_edge_get_property, e, property_name, memory);
}

struct mgp_properties_iterator *edge_iter_properties(const struct mgp_edge *e, struct mgp_memory *memory) {
  return MgInvoke<struct mgp_properties_iterator *>(mgp_edge_iter_properties, e, memory);
}

struct mgp_vertex *graph_get_vertex_by_id(struct mgp_graph *g, struct mgp_vertex_id id, struct mgp_memory *memory) {
  return MgInvoke<struct mgp_vertex *>(mgp_graph_get_vertex_by_id, g, id, memory);
}

void vertices_iterator_destroy(struct mgp_vertices_iterator *it) { mgp_vertices_iterator_destroy(it); }

struct mgp_vertices_iterator *graph_iter_vertices(const struct mgp_graph *g, struct mgp_memory *memory) {
  return MgInvoke<struct mgp_vertices_iterator *>(mgp_graph_iter_vertices, g, memory);
}

struct mgp_vertex *vertices_iterator_get(struct mgp_vertices_iterator *it) {
  return MgInvoke<struct mgp_vertex *>(mgp_vertices_iterator_get, it);
}

struct mgp_vertex *vertices_iterator_next(struct mgp_vertices_iterator *it) {
  return MgInvoke<struct mgp_vertex *>(mgp_vertices_iterator_get, it);
}

const struct mgp_type *type_any() { return MgInvoke<const struct mgp_type *>(mgp_type_any); }

const struct mgp_type *type_bool() { return MgInvoke<const struct mgp_type *>(mgp_type_bool); }

const struct mgp_type *type_string() { return MgInvoke<const struct mgp_type *>(mgp_type_string); }

const struct mgp_type *type_int() { return MgInvoke<const struct mgp_type *>(mgp_type_int); }

const struct mgp_type *type_float() { return MgInvoke<const struct mgp_type *>(mgp_type_float); }

const struct mgp_type *type_number() { return MgInvoke<const struct mgp_type *>(mgp_type_number); }

const struct mgp_type *type_map() { return MgInvoke<const struct mgp_type *>(mgp_type_map); }

const struct mgp_type *type_node() { return MgInvoke<const struct mgp_type *>(mgp_type_node); }

const struct mgp_type *type_relationship() { return MgInvoke<const struct mgp_type *>(mgp_type_relationship); }

const struct mgp_type *type_path() { return MgInvoke<const struct mgp_type *>(mgp_type_path); }

const struct mgp_type *type_list(const struct mgp_type *element_type) {
  return MgInvoke<const struct mgp_type *>(mgp_type_list, element_type);
}

const struct mgp_type *type_nullable(const struct mgp_type *type) {
  return MgInvoke<const struct mgp_type *>(mgp_type_nullable, type);
}
struct mgp_proc *module_add_read_procedure(struct mgp_module *module, const char *name, mgp_proc_cb cb) {
  return MgInvoke<struct mgp_proc *>(mgp_module_add_read_procedure, module, name, cb);
}

void proc_add_arg(struct mgp_proc *proc, const char *name, const struct mgp_type *type) {
  MgInvokeVoid(mgp_proc_add_arg, proc, name, type);
}

void proc_add_opt_arg(struct mgp_proc *proc, const char *name, const struct mgp_type *type,
                      const struct mgp_value *default_value) {
  MgInvokeVoid(mgp_proc_add_opt_arg, proc, name, type, default_value);
}

void proc_add_result(struct mgp_proc *proc, const char *name, const struct mgp_type *type) {
  MgInvokeVoid(mgp_proc_add_result, proc, name, type);
}

void proc_add_deprecated_result(struct mgp_proc *proc, const char *name, const struct mgp_type *type) {
  MgInvokeVoid(mgp_proc_add_deprecated_result, proc, name, type);
}

bool must_abort(const struct mgp_graph *graph) { return MgInvoke<int>(mgp_must_abort, graph); }

}  // namespace mgp