#include <mg_procedure.h>
#include <ctime>
#include <mg_exceptions.hpp>
#include <mg_utils.hpp>
#include <random>
#include <vector>

namespace {

constexpr char const *kProcedureGet = "get";
constexpr char const *kFieldPath = "path";

constexpr char const *kArgumentStart = "start";
constexpr char const *kArgumentLength = "length";

/// Memgraph query module implementation of a random walk algorithm.
/// Random walk is an algorithm that provides random paths in a graph.
///
/// @param args Memgraph module arguments
/// @param graph Memgraph graph instance
/// @param result Memgraph result storage
/// @param memory Memgraph memory storage
void Generate(const mgp_list *args, const mgp_graph *graph, mgp_result *result, mgp_memory *memory) {
  try {
    auto start = mgp_value_get_vertex(mgp_list_at(args, 0));
    auto length = mgp_value_get_int(mgp_list_at(args, 1));

    auto path = mgp_path_make_with_start(start, memory);
    auto vertex = start;
    srand(time(NULL));

    for (int i = 0; i < length; i++) {
      auto iter = mgp_vertex_iter_out_edges(vertex, memory);
      if (!iter) {
        throw mg_exception::NotEnoughMemoryException();
      }
      const mgp_edge *edge = mgp_edges_iterator_get(iter);
      std::vector<const mgp_edge *> edges_vector;
      while (edge) {
        edges_vector.push_back(mgp_edge_copy(edge, memory));
        edge = mgp_edges_iterator_next(iter);
      }
      int size = edges_vector.size();
      if (size == 0) {
        break;
      }
      const mgp_edge *random_edge = edges_vector[rand() % edges_vector.size()];
      mgp_path_expand(path, random_edge);
      vertex = mgp_edge_get_to(random_edge);
      mgp_edges_iterator_destroy(iter);
    }
    mgp_result_record *record = mgp_result_new_record(result);
    if (record == nullptr) {
      throw mg_exception::NotEnoughMemoryException();
    }
    mg_utility::InsertPathValueResult(record, kFieldPath, path, memory);
  } catch (const std::exception &e) {
    // We must not let any exceptions out of our module.
    mgp_result_set_error_msg(result, e.what());
    return;
  }
}
}  // namespace

// Each module needs to define mgp_init_module function.
// Here you can register multiple procedures your module supports.
extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  struct mgp_proc *random_walk_proc = mgp_module_add_read_procedure(module, kProcedureGet, Generate);
  if (!random_walk_proc) return 1;

  // Query module arguments
  if (!mgp_proc_add_arg(random_walk_proc, kArgumentStart, mgp_type_node())) return 1;
  if (!mgp_proc_add_arg(random_walk_proc, kArgumentLength, mgp_type_int())) return 1;

  // Query module output record
  if (!mgp_proc_add_result(random_walk_proc, kFieldPath, mgp_type_path())) return 1;

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
