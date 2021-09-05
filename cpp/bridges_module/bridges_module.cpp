#include <mg_exceptions.hpp>
#include <mg_utils.hpp>

#include "algorithm/bridges.hpp"

namespace {

const char *kProcedureGet = "get";

// const char *fieldEdgeID = "edge_id";
const char *k_field_node_from = "node_from";
const char *k_field_node_to = "node_to";

void InsertBridgeRecord(mgp_graph *graph, mgp_result *result, mgp_memory *memory, const std::uint64_t node_from_id,
                        const std::uint64_t node_to_id) {
  auto *record = mgp::result_new_record(result);

  mg_utility::InsertNodeValueResult(graph, record, k_field_node_from, node_from_id, memory);
  mg_utility::InsertNodeValueResult(graph, record, k_field_node_to, node_to_id, memory);
}

void GetBridges(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  try {
    auto graph = mg_utility::GetGraphView(memgraph_graph, result, memory, mg_graph::GraphType::kUndirectedGraph);
    auto bridges = bridges_alg::GetBridges(*graph);

    for (const auto &bridge_edge : bridges) {
      InsertBridgeRecord(memgraph_graph, result, memory, graph->GetMemgraphNodeId(bridge_edge.from),
                         graph->GetMemgraphNodeId(bridge_edge.to));
    }
  } catch (const std::exception &e) {
    // We must not let any exceptions out of our module.
    mgp::result_set_error_msg(result, e.what());
    return;
  }
}
}  // namespace

// Each module needs to define mgp_init_module function.
// Here you can register multiple procedures your module supports.
extern "C" int mgp_init_module(mgp_module *module, mgp_memory *memory) {
  try {
    auto *proc = mgp::module_add_read_procedure(module, kProcedureGet, GetBridges);
    mgp::proc_add_result(proc, k_field_node_from, mgp::type_node());
    mgp::proc_add_result(proc, k_field_node_to, mgp::type_node());
  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

// This is an optional function if you need to release any resources before the
// module is unloaded. You will probably need this if you acquired some
// resources in mgp_init_module.
extern "C" int mgp_shutdown_module() {
  // Return 0 to indicate success.
  return 0;
}