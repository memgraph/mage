#include "algorithm/biconnected_components.hpp"

#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <unordered_set>

#include <mg_exceptions.hpp>
#include <mg_procedure.h>
#include <mg_utils.hpp>

namespace {

const char *fieldBiconnectedComponentID = "bcc_id";
const char *fieldEdgeID = "edge_id";
const char *fieldNodeFrom = "node_from";
const char *fieldNodeTo = "node_to";

void InsertBiconnectedComponentRecord(const mgp_graph *graph,
                                      mgp_result *result, mgp_memory *memory,
                                      const int bcc_id, const int edge_id,
                                      const int node_from_id,
                                      const int node_to_id) {
  mgp_result_record *record = mgp_result_new_record(result);
  if (record == nullptr) {
    throw mg_exception::NotEnoughMemoryException();
  }

  mg_utility::InsertIntValueResult(record, fieldBiconnectedComponentID, bcc_id,
                                   memory);

  // TODO: Implement edge getting function
  // mg_utility::InsertIntValueResult(record, fieldEdgeID, edge_id, memory);

  mg_utility::InsertNodeValueResult(graph, record, fieldNodeFrom, node_from_id,
                                    memory);

  mg_utility::InsertNodeValueResult(graph, record, fieldNodeTo, node_to_id,
                                    memory);
}

void GetBiconnectedComponents(const mgp_list *args,
                              const mgp_graph *memgraph_graph,
                              mgp_result *result, mgp_memory *memory) {
  try {
    mg_graph::Graph *graph =
        mg_utility::GetGraphView(memgraph_graph, result, memory);

    std::vector<std::vector<mg_graph::Edge>> bccs =
        bcc_algorithm::GetBiconnectedComponents(graph);

    for (uint32_t bcc_id = 0; bcc_id < bccs.size(); bcc_id++) {
      for (const auto &edge : bccs[bcc_id]) {
        InsertBiconnectedComponentRecord(memgraph_graph, result, memory, bcc_id,
                                         edge.id,
                                         graph->GetMemgraphNodeId(edge.from),
                                         graph->GetMemgraphNodeId(edge.to));
      }
    }
  } catch (const std::exception &e) {
    // We must not let any exceptions out of our module.
    mgp_result_set_error_msg(result, e.what());
    return;
  }
}
} // namespace

// Each module needs to define mgp_init_module function.
// Here you can register multiple procedures your module supports.
extern "C" int mgp_init_module(mgp_module *module, mgp_memory *memory) {
  mgp_proc *proc = mgp_module_add_read_procedure(
      module, "biconnected_components", GetBiconnectedComponents);

  if (!proc)
    return 1;

  if (!mgp_proc_add_result(proc, fieldBiconnectedComponentID, mgp_type_int()))
    return 1;
  // if (!mgp_proc_add_result(proc, fieldEdgeID, mgp_type_int()))
  // return 1;
  if (!mgp_proc_add_result(proc, fieldNodeFrom, mgp_type_node()))
    return 1;
  if (!mgp_proc_add_result(proc, fieldNodeTo, mgp_type_node()))
    return 1;

  return 0;
}

// This is an optional function if you need to release any resources before the
// module is unloaded. You will probably need this if you acquired some
// resources in mgp_init_module.
extern "C" int mgp_shutdown_module() {
  // Return 0 to indicate success.
  return 0;
}