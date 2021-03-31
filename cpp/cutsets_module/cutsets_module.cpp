#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <unordered_set>

#include <mg_procedure.h>
#include <mg_exceptions.hpp>
#include <mg_utils.hpp>

#include "algorithm/cutsets.hpp"

namespace {

// const char *fieldEdgeID = "edge_id";
const char *fieldCutsetID = "cutset_id";
const char *fieldNodeFrom = "node_from";
const char *fieldNodeTo = "node_to";

void InsertCutsetRecord(const mgp_graph *graph, mgp_result *result, mgp_memory *memory, const int cutset_id,
                        const int node_from_id, const int node_to_id) {
  mgp_result_record *record = mgp_result_new_record(result);
  if (record == nullptr) {
    throw mg_exception::NotEnoughMemoryException();
  }

  mg_utility::InsertIntValueResult(record, fieldCutsetID, cutset_id, memory);

  mg_utility::InsertNodeValueResult(graph, record, fieldNodeFrom, node_from_id, memory);

  mg_utility::InsertNodeValueResult(graph, record, fieldNodeTo, node_to_id, memory);
}

void GetCutSets(const mgp_list *args, const mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  try {
    auto *graph = mg_utility::GetGraphView(memgraph_graph, result, memory);
    auto cutsets = cutsets_alg::GetCutsets(graph);

    for (uint64_t cutset_id = 0; cutset_id < cutsets.size(); cutset_id++) {
      for (const auto &edge : cutsets[cutset_id]) {
        InsertCutsetRecord(memgraph_graph, result, memory, cutset_id, graph->GetMemgraphNodeId(edge.from),
                           graph->GetMemgraphNodeId(edge.to));
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
  mgp_proc *proc = mgp_module_add_read_procedure(module, "get", GetCutSets);

  if (!proc) return 1;

  // if (!mgp_proc_add_result(proc, fieldEdgeID, mgp_type_int())) return 1;
  if (!mgp_proc_add_result(proc, fieldCutsetID, mgp_type_int())) return 1;
  if (!mgp_proc_add_result(proc, fieldNodeFrom, mgp_type_node())) return 1;
  if (!mgp_proc_add_result(proc, fieldNodeTo, mgp_type_node())) return 1;

  return 0;
}

// This is an optional function if you need to release any resources before the
// module is unloaded. You will probably need this if you acquired some
// resources in mgp_init_module.
extern "C" int mgp_shutdown_module() {
  // Return 0 to indicate success.
  return 0;
}