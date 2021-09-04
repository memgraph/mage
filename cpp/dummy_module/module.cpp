#include <uuid/uuid.h>

#include <mg_exceptions.hpp>
#include <mgp.hpp>

namespace {

constexpr char const *kFieldId = "id";
constexpr char const *kFieldVertex = "vertex";

void Generate(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  try {
    auto vertex = mgp::value_get_vertex(mgp::list_at(args, 0));
    auto *record = mgp::result_new_record(result);

    auto vertex_id = mgp::vertex_get_id(vertex).as_int;
    mgp::InsertIntValueResult(record, kFieldId, vertex_id, memory);
  } catch (const std::exception &e) {
    // We must not let any exceptions out of our module.
    mgp::result_set_error_msg(result, e.what());
    return;
  }
}
}  // namespace

extern "C" int mgp_init_module(mgp_module *module, mgp_memory *memory) {
  try {
    auto *uuid_proc = mgp::module_add_read_procedure(module, "get", Generate);
    mgp::proc_add_arg(uuid_proc, kFieldVertex, mgp::type_node());
    mgp::proc_add_result(uuid_proc, kFieldId, mgp::type_int());
    return 0;
  } catch (const std::exception &e) {
    return 1;
  }
}

extern "C" int mgp_shutdown_module() { return 0; }
