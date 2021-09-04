#include <mg_exceptions.hpp>
#include <mg_utils.hpp>

#include "algorithm/biconnected_components.hpp"

namespace {
const char *kProcedureGet = "get";

const char *fieldBiconnectedComponentID = "bcc_id";
// const char *fieldEdgeID = "edge_id";
const char *fieldNodeFrom = "node_from";
const char *fieldNodeTo = "node_to";

void InsertBiconnectedComponentRecord(mgp_graph *graph, mgp_result *result, mgp_memory *memory, const int bcc_id,
                                      const int edge_id, const int node_from_id, const int node_to_id) {
  auto *record = mgp::result_new_record(result);

  mg_utility::InsertIntValueResult(record, fieldBiconnectedComponentID, bcc_id, memory);
  // TODO: Implement edge getting function
  // mg_utility::InsertIntValueResult(record, fieldEdgeID, edge_id, memory);
  mg_utility::InsertNodeValueResult(graph, record, fieldNodeFrom, node_from_id, memory);
  mg_utility::InsertNodeValueResult(graph, record, fieldNodeTo, node_to_id, memory);
}

void GetBiconnectedComponents(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  try {
    auto graph = mg_utility::GetGraphView(memgraph_graph, result, memory, mg_graph::GraphType::kUndirectedGraph);

    auto bccs = bcc_algorithm::GetBiconnectedComponents(*graph);

    for (std::uint64_t bcc_id = 0; bcc_id < bccs.size(); bcc_id++) {
      for (const auto &edge : bccs[bcc_id]) {
        InsertBiconnectedComponentRecord(memgraph_graph, result, memory, bcc_id, edge.id,
                                         graph->GetMemgraphNodeId(edge.from), graph->GetMemgraphNodeId(edge.to));
      }
    }
  } catch (const std::exception &e) {
    // We must not let any exceptions out of our module.
    mgp_result_set_error_msg(result, e.what());
    return;
  }
}
}  // namespace

// Each module needs to define mgp_init_module function.
// Here you can register multiple procedures your module supports.
extern "C" int mgp_init_module(mgp_module *module, mgp_memory *memory) {
  try {
    auto *proc = mgp::module_add_read_procedure(module, kProcedureGet, GetBiconnectedComponents);

    mgp_proc_add_result(proc, fieldBiconnectedComponentID, mgp::type_int());
    mgp_proc_add_result(proc, fieldNodeFrom, mgp::type_node());
    mgp_proc_add_result(proc, fieldNodeTo, mgp::type_node());
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