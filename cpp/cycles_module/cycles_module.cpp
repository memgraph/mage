#include <mg_procedure.h>
#include <mg_utils.hpp>

#include "algorithm/cycles.hpp"

namespace {

constexpr char const *kFieldCycleId = "cycle_id";
constexpr char const *kFieldNode = "node";

void InsertCycleRecord(const mgp_graph *graph, mgp_result *result, mgp_memory *memory, const int cycle_id,
                       const int node_id) {
  auto *record = mgp_result_new_record(result);
  if (record == nullptr) {
    throw mg_exception::NotEnoughMemoryException();
  }

  mg_utility::InsertIntValueResult(record, kFieldCycleId, cycle_id, memory);
  mg_utility::InsertNodeValueResult(graph, record, kFieldNode, node_id, memory);
}

void GetCycles(const mgp_list *args, const mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  try {
    auto graph = mg_utility::GetGraphView(memgraph_graph, result, memory, mg_graph::GraphType::kUndirectedGraph);
    auto cycles = cycles_alg::GetCycles(*graph);

    for (std::size_t cycle_id = 0; cycle_id < cycles.size(); cycle_id++) {
      // Insert each node on the cycle
      for (const auto &node : cycles[cycle_id]) {
        InsertCycleRecord(memgraph_graph, result, memory, cycle_id, graph->GetMemgraphNodeId(node.id));
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
  mgp_proc *proc = mgp_module_add_read_procedure(module, "get", GetCycles);

  if (!proc) return 1;

  if (!mgp_proc_add_result(proc, kFieldCycleId, mgp_type_int())) return 1;
  if (!mgp_proc_add_result(proc, kFieldNode, mgp_type_node())) return 1;

  return 0;
}

// This is an optional function if you need to release any resources before the
// module is unloaded. You will probably need this if you acquired some
// resources in mgp_init_module.
extern "C" int mgp_shutdown_module() {
  // Return 0 to indicate success.
  return 0;
}