/// @file mgp.hpp
///
/// The file contains methods that connect mg procedures and the outside code
/// Methods like mapping a graph into memory or assigning new mg results or
/// their properties are implemented.
#pragma once

#include "mg_exceptions.hpp"
#include "mg_procedure.h"
#include <fmt/format.h>

namespace mgp {

namespace {
void MgExceptionHandle(mgp_error result_code) {
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
    case MGP_ERROR_DELETED_OBJECT:
      throw mg_exception::DeletedObjectException();
    case MGP_ERROR_INVALID_ARGUMENT:
      throw mg_exception::InvalidArgumentException();
    case MGP_ERROR_KEY_ALREADY_EXISTS:
      throw mg_exception::KeyAlreadyExistsException();
    case MGP_ERROR_IMMUTABLE_OBJECT:
      throw mg_exception::ImmutableObjectException();
    case MGP_ERROR_VALUE_CONVERSION:
      throw mg_exception::ValueConversionException();
    case MGP_ERROR_SERIALIZATION_ERROR:
      throw mg_exception::SerializationException();
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

enum mgp_value_type value_get_type(struct mgp_value *val) {
  return MgInvoke<enum mgp_value_type>(mgp_value_get_type, val);
}

bool value_is_null(struct mgp_value *val) { return MgInvoke<int>(mgp_value_is_null, val); }

bool value_is_bool(struct mgp_value *val) { return MgInvoke<int>(mgp_value_is_bool, val); }

bool value_is_int(struct mgp_value *val) { return MgInvoke<int>(mgp_value_is_int, val); }

bool value_is_double(struct mgp_value *val) { return MgInvoke<int>(mgp_value_is_double, val); }

bool value_is_string(struct mgp_value *val) { return MgInvoke<int>(mgp_value_is_string, val); }

bool value_is_list(struct mgp_value *val) { return MgInvoke<int>(mgp_value_is_list, val); }

bool value_is_map(struct mgp_value *val) { return MgInvoke<int>(mgp_value_is_map, val); }

bool value_is_vertex(struct mgp_value *val) { return MgInvoke<int>(mgp_value_is_vertex, val); }

bool value_is_edge(struct mgp_value *val) { return MgInvoke<int>(mgp_value_is_edge, val); }

bool value_is_path(struct mgp_value *val) { return MgInvoke<int>(mgp_value_is_path, val); }

bool value_get_bool(struct mgp_value *val) { return MgInvoke<int>(mgp_value_get_bool, val); }

int64_t value_get_int(struct mgp_value *val) { return MgInvoke<int64_t>(mgp_value_get_int, val); }

double value_get_double(struct mgp_value *val) { return MgInvoke<double>(mgp_value_get_double, val); }

const char *value_get_string(struct mgp_value *val) { return MgInvoke<const char *>(mgp_value_get_string, val); }

struct mgp_list *value_get_list(struct mgp_value *val) {
  return MgInvoke<struct mgp_list *>(mgp_value_get_list, val);
}

struct mgp_map *value_get_map(struct mgp_value *val) {
  return MgInvoke<struct mgp_map *>(mgp_value_get_map, val);
}

struct mgp_vertex *value_get_vertex(struct mgp_value *val) {
  return MgInvoke<struct mgp_vertex *>(mgp_value_get_vertex, val);
}

struct mgp_edge *value_get_edge(struct mgp_value *val) {
  return MgInvoke<struct mgp_edge *>(mgp_value_get_edge, val);
}

struct mgp_path *value_get_path(struct mgp_value *val) {
  return MgInvoke<struct mgp_path *>(mgp_value_get_path, val);
}

struct mgp_list *list_make_empty(size_t capacity, struct mgp_memory *memory) {
  return MgInvoke<struct mgp_list *>(mgp_list_make_empty, capacity, memory);
}

void list_destroy(struct mgp_list *list) { mgp_list_destroy(list); }

void list_append(struct mgp_list *list, struct mgp_value *val) { MgInvokeVoid(mgp_list_append, list, val); }

void list_append_extend(struct mgp_list *list, struct mgp_value *val) {
  MgInvokeVoid(mgp_list_append_extend, list, val);
}

size_t list_size(struct mgp_list *list) { return MgInvoke<size_t>(mgp_list_size, list); }

size_t list_capacity(struct mgp_list *list) { return MgInvoke<size_t>(mgp_list_capacity, list); }

struct mgp_value *list_at(struct mgp_list *list, size_t index) {
  return MgInvoke<struct mgp_value *>(mgp_list_at, list, index);
}

struct mgp_map *map_make_empty(struct mgp_memory *memory) {
  return MgInvoke<struct mgp_map *>(mgp_map_make_empty, memory);
}

void map_destroy(struct mgp_map *map) { mgp_map_destroy(map); }

void map_insert(struct mgp_map *map, const char *key, struct mgp_value *value) {
  MgInvokeVoid(mgp_map_insert, map, key, value);
}

size_t map_size(struct mgp_map *map) { return MgInvoke<size_t>(mgp_map_size, map); }
struct mgp_value *map_at(struct mgp_map *map, const char *key) {
  return MgInvoke<struct mgp_value *>(mgp_map_at, map, key);
}

// const char *map_item_key(struct mgp_map_item *item) { return MgInvoke<const char *>(mgp_map_item_key, item); }

struct mgp_value *map_item_value(struct mgp_map_item *item) {
  return MgInvoke<struct mgp_value *>(mgp_map_item_value, item);
}

// struct mgp_map_items_iterator *map_iter_items(struct mgp_map *map, struct mgp_memory *memory) {
//   return MgInvoke<struct mgp_map_items_iterator *>(mgp_map_iter_items, map, memory);
// }

void map_items_iterator_destroy(struct mgp_map_items_iterator *it) { mgp_map_items_iterator_destroy(it); }

// struct mgp_map_item *map_items_iterator_get(struct mgp_map_items_iterator *it) {
//   return MgInvoke<struct mgp_map_item *>(mgp_map_items_iterator_get, it);
// }

struct mgp_map_item *map_items_iterator_next(struct mgp_map_items_iterator *it) {
  return MgInvoke<struct mgp_map_item *>(mgp_map_items_iterator_next, it);
}

struct mgp_path *path_make_with_start(struct mgp_vertex *vertex, struct mgp_memory *memory) {
  return MgInvoke<struct mgp_path *>(mgp_path_make_with_start, vertex, memory);
}

struct mgp_path *path_copy(struct mgp_path *path, struct mgp_memory *memory) {
  return MgInvoke<struct mgp_path *>(mgp_path_copy, path, memory);
}

void path_destroy(struct mgp_path *path) { mgp_path_destroy(path); }

void path_expand(struct mgp_path *path, struct mgp_edge *edge) { MgInvokeVoid(mgp_path_expand, path, edge); }

size_t path_size(struct mgp_path *path) { return MgInvoke<size_t>(mgp_path_size, path); }

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

void result_record_insert(struct mgp_result_record *record, const char *field_name, struct mgp_value *val) {
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

struct mgp_vertex_id vertex_get_id(struct mgp_vertex *v) {
  return MgInvoke<struct mgp_vertex_id>(mgp_vertex_get_id, v);
}

struct mgp_vertex *vertex_copy(struct mgp_vertex *v, struct mgp_memory *memory) {
  return MgInvoke<struct mgp_vertex *>(mgp_vertex_copy, v, memory);
}

void vertex_destroy(struct mgp_vertex *v) { mgp_vertex_destroy(v); }

bool vertex_equal(struct mgp_vertex *v1, struct mgp_vertex *v2) { return MgInvoke<int>(mgp_vertex_equal, v1, v2); }

size_t vertex_labels_count(struct mgp_vertex *v) { return MgInvoke<size_t>(mgp_vertex_labels_count, v); }

struct mgp_label vertex_label_at(struct mgp_vertex *v, size_t index) {
  return MgInvoke<struct mgp_label>(mgp_vertex_label_at, v, index);
}

bool vertex_has_label(struct mgp_vertex *v, struct mgp_label label) {
  return MgInvoke<int>(mgp_vertex_has_label, v, label);
}

bool vertex_has_label_named(struct mgp_vertex *v, const char *label_name) {
  return MgInvoke<int>(mgp_vertex_has_label_named, v, label_name);
}

struct mgp_value *vertex_get_property(struct mgp_vertex *v, const char *property_name, struct mgp_memory *memory) {
  return MgInvoke<struct mgp_value *>(mgp_vertex_get_property, v, property_name, memory);
}

struct mgp_properties_iterator *vertex_iter_properties(struct mgp_vertex *v, struct mgp_memory *memory) {
  return MgInvoke<struct mgp_properties_iterator *>(mgp_vertex_iter_properties, v, memory);
}

struct mgp_edges_iterator *vertex_iter_in_edges(struct mgp_vertex *v, struct mgp_memory *memory) {
  return MgInvoke<struct mgp_edges_iterator *>(mgp_vertex_iter_in_edges, v, memory);
}

struct mgp_edges_iterator *vertex_iter_out_edges(struct mgp_vertex *v, struct mgp_memory *memory) {
  return MgInvoke<struct mgp_edges_iterator *>(mgp_vertex_iter_out_edges, v, memory);
}

struct mgp_edge *edges_iterator_get(struct mgp_edges_iterator *it) {
  return MgInvoke<struct mgp_edge *>(mgp_edges_iterator_get, it);
}

struct mgp_edge *edges_iterator_next(struct mgp_edges_iterator *it) {
  return MgInvoke<struct mgp_edge *>(mgp_edges_iterator_next, it);
}

struct mgp_edge_id edge_get_id(struct mgp_edge *e) {
  return MgInvoke<struct mgp_edge_id>(mgp_edge_get_id, e);
}

struct mgp_edge *edge_copy(struct mgp_edge *e, struct mgp_memory *memory) {
  return MgInvoke<struct mgp_edge *>(mgp_edge_copy, e, memory);
}

void edge_destroy(struct mgp_edge *e) { mgp_edge_destroy(e); }

bool edge_equal(struct mgp_edge *e1, struct mgp_edge *e2) { return MgInvoke<int>(mgp_edge_equal, e1, e2); }

struct mgp_edge_type edge_get_type(struct mgp_edge *e) {
  return MgInvoke<struct mgp_edge_type>(mgp_edge_get_type, e);
}

struct mgp_vertex *edge_get_from(struct mgp_edge *e) {
  return MgInvoke<struct mgp_vertex *>(mgp_edge_get_from, e);
}

struct mgp_vertex *edge_get_to(struct mgp_edge *e) {
  return MgInvoke<struct mgp_vertex *>(mgp_edge_get_to, e);
}

struct mgp_value *edge_get_property(struct mgp_edge *e, const char *property_name, struct mgp_memory *memory) {
  return MgInvoke<struct mgp_value *>(mgp_edge_get_property, e, property_name, memory);
}

struct mgp_properties_iterator *edge_iter_properties(struct mgp_edge *e, struct mgp_memory *memory) {
  return MgInvoke<struct mgp_properties_iterator *>(mgp_edge_iter_properties, e, memory);
}

struct mgp_vertex *graph_get_vertex_by_id(struct mgp_graph *g, struct mgp_vertex_id id, struct mgp_memory *memory) {
  return MgInvoke<struct mgp_vertex *>(mgp_graph_get_vertex_by_id, g, id, memory);
}

void vertices_iterator_destroy(struct mgp_vertices_iterator *it) { mgp_vertices_iterator_destroy(it); }

struct mgp_vertices_iterator *graph_iter_vertices(struct mgp_graph *g, struct mgp_memory *memory) {
  return MgInvoke<struct mgp_vertices_iterator *>(mgp_graph_iter_vertices, g, memory);
}

struct mgp_vertex *vertices_iterator_get(struct mgp_vertices_iterator *it) {
  return MgInvoke<struct mgp_vertex *>(mgp_vertices_iterator_get, it);
}

struct mgp_vertex *vertices_iterator_next(struct mgp_vertices_iterator *it) {
  return MgInvoke<struct mgp_vertex *>(mgp_vertices_iterator_next, it);
}

struct mgp_type *type_any() {
  return MgInvoke<struct mgp_type *>(mgp_type_any);
}

struct mgp_type *type_bool() {
  return MgInvoke<struct mgp_type *>(mgp_type_bool);
}

struct mgp_type *type_string() {
  return MgInvoke<struct mgp_type *>(mgp_type_string);
}

struct mgp_type *type_int() {
  return MgInvoke<struct mgp_type *>(mgp_type_int);
}

struct mgp_type *type_float() {
  return MgInvoke<struct mgp_type *>(mgp_type_float);
}

struct mgp_type *type_number() {
  return MgInvoke<struct mgp_type *>(mgp_type_number);
}

struct mgp_type *type_map() {
  return MgInvoke<struct mgp_type *>(mgp_type_map);
}

struct mgp_type *type_node() {
  return MgInvoke<struct mgp_type *>(mgp_type_node);
}

struct mgp_type *type_relationship() {
  return MgInvoke<struct mgp_type *>(mgp_type_relationship);
}

struct mgp_type *type_path() {
  return MgInvoke<struct mgp_type *>(mgp_type_path);
}

struct mgp_type *type_list(struct mgp_type *element_type) {
  return MgInvoke<struct mgp_type *>(mgp_type_list, element_type);
}

struct mgp_type *type_nullable(struct mgp_type *type) {
  return MgInvoke<struct mgp_type *>(mgp_type_nullable, type);
}
struct mgp_proc *module_add_read_procedure(struct mgp_module *module, const char *name, mgp_proc_cb cb) {
  return MgInvoke<struct mgp_proc *>(mgp_module_add_read_procedure, module, name, cb);
}

struct mgp_proc *module_add_write_procedure(struct mgp_module *module, const char *name, mgp_proc_cb cb) {
  return MgInvoke<struct mgp_proc *>(mgp_module_add_write_procedure, module, name, cb);
}

void proc_add_arg(struct mgp_proc *proc, const char *name, struct mgp_type *type) {
  MgInvokeVoid(mgp_proc_add_arg, proc, name, type);
}

void proc_add_opt_arg(struct mgp_proc *proc, const char *name, struct mgp_type *type, struct mgp_value *default_value) {
  MgInvokeVoid(mgp_proc_add_opt_arg, proc, name, type, default_value);
}

void proc_add_result(struct mgp_proc *proc, const char *name, struct mgp_type *type) {
  MgInvokeVoid(mgp_proc_add_result, proc, name, type);
}

void proc_add_deprecated_result(struct mgp_proc *proc, const char *name, struct mgp_type *type) {
  MgInvokeVoid(mgp_proc_add_deprecated_result, proc, name, type);
}

bool must_abort(struct mgp_graph *graph) { return mgp_must_abort(graph); }

// Generation
struct mgp_vertex *graph_create_vertex(struct mgp_graph *graph, struct mgp_memory *memory) {
  return MgInvoke<struct mgp_vertex *>(mgp_graph_create_vertex, graph, memory);
}

struct mgp_edge *graph_create_edge(struct mgp_graph *graph, struct mgp_vertex *from, struct mgp_vertex *to,
                                   struct mgp_edge_type type, struct mgp_memory *memory) {
  return MgInvoke<struct mgp_edge *>(mgp_graph_create_edge, graph, from, to, type, memory);
}

void vertex_add_label(struct mgp_vertex *vertex, struct mgp_label label) {
  MgInvokeVoid(mgp_vertex_add_label, vertex, label);
}

int date_get_day_add1(struct mgp_date *date) {
  return MgInvoke<int>(mgp_date_get_day_add1, date);
}

int add_10(int number) {
  return MgInvoke<int>(mgp_add_10, number);
}

template <typename... Args>
void log(const enum mgp_log_level log_level, const char *output) {
  return MgInvokeVoid(mgp_log, log_level, output);
}



}  // namespace mgp
