#include <iostream>
#include <queue>

#include <mg_exceptions.hpp>
#include <mg_utils.hpp>
#include <mgp.hpp>

namespace {

const char *field_vertex = "node";
const char *field_vertex_id = "node_id";
const char *field_vertex_name = "name";

/// Dummy function for testing the functionality of C++ Memgraph query modules wrapper
static void DummyFunc(const mgp::ImmutableList &arg_list, mgp::Graph &graph, const mgp::RecordFactory &record_factory) {
  std::cout << "Function called" << std::endl;
  auto vertex = arg_list[0].ValueVertex();

  std::cout << "Get vertex" << std::endl;
  for (int i = 0; i < 1; i++) {
    auto record = record_factory.NewRecord();
    record.Insert(field_vertex, vertex);
    record.Insert(field_vertex_id, vertex.id().AsInt());
  }

  for (auto v : graph.vertices()) {
    auto record = record_factory.NewRecord();
    record.Insert(field_vertex, v);
    record.Insert(field_vertex_id, v.id().AsInt());
    record.Insert(field_vertex_name, v.properties()["name"].ValueString());
  }
}

static void CWrapper(const mgp_list *args, const mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  try {
    auto graph = mgp::Graph(memgraph_graph, memory);
    auto record_factory = mgp::RecordFactory(result, memory);
    auto list = mgp::ImmutableList(args, memory);

    DummyFunc(list, graph, record_factory);
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
  if (!mgp_proc_add_result(dum_proc, field_vertex_name, mgp_type_string())) return 1;
  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
