#include <iostream>
#include <queue>
#include <unordered_map>

#include <mg_exceptions.hpp>
#include <mg_utils.hpp>
#include <mgp.hpp>

namespace {

const char *field_vertex = "node";
const char *field_vertex_id = "node_id";

/// Finds weakly connected components of a graph.
///
/// Time complexity: O(|V|+|E|)
static void DummyFunc(const mgp::Graph &graph, const mgp::RecordFactory &record_factory, const mgp::Vertex &vertex) {
  auto vertices = graph.Vertices();
  for (auto v : vertices) {
    std::cout << std::to_string(v.Id()) << '\n';
  }

  std::cout << std::to_string(vertex.Id()) << '\n';
  auto *record = record_factory.NewRecord();
  record->Insert(field_vertex, vertex);
  record->Insert(field_vertex_id, vertex.Id());
}

static void CWrapper(const mgp_list *args, const mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  try {
    auto graph = mgp::Graph(memgraph_graph, memory);
    auto record_factory = mgp::RecordFactory::GetInstance(result, memory);
    auto vertex = mgp::Vertex::FromList(args, 0);
    DummyFunc(graph, record_factory, *vertex);
  } catch (const std::exception &e) {
    // We must not let any exceptions out of our module.
    mgp_result_set_error_msg(result, e.what());
    return;
  }
}
}  // namespace

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  struct mgp_proc *dum_proc = mgp_module_add_read_procedure(module, "get", CWrapper);
  if (!dum_proc) return 1;
  if (!mgp_proc_add_arg(dum_proc, field_vertex, mgp_type_node())) return 1;
  if (!mgp_proc_add_result(dum_proc, field_vertex, mgp_type_node())) return 1;
  if (!mgp_proc_add_result(dum_proc, field_vertex_id, mgp_type_int())) return 1;
  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
