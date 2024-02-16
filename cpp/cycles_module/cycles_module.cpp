#include <mg_utils.hpp>

#include "algorithm/cycles.hpp"

namespace {

constexpr char const *kProcedureGet = "get";

constexpr char const *kFieldCycleId = "cycle_id";
constexpr char const *kFieldNode = "node";

void InsertCycleRecord(mgp_graph *graph, mgp_result *result, mgp_memory *memory, const int cycle_id,
                       const int node_id) {
  auto *node = mg_utility::GetNodeForInsertion(node_id, graph, memory);
  if (!node) return;

  auto *record = mgp::result_new_record(result);
  if (record == nullptr) throw mg_exception::NotEnoughMemoryException();

  mg_utility::InsertIntValueResult(record, kFieldCycleId, cycle_id, memory);
  mg_utility::InsertNodeValueResult(record, kFieldNode, node, memory);
}

void GetCycles(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
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
    mgp::result_set_error_msg(result, e.what());
    return;
  }
}
}  // namespace

// Each module needs to define mgp_init_module function.
// Here you can register multiple procedures your module supports.
extern "C" int mgp_init_module(mgp_module *module, mgp_memory *memory) {
  try {
    auto *proc = mgp::module_add_read_procedure(module, kProcedureGet, GetCycles);

    mgp::proc_add_result(proc, kFieldCycleId, mgp::type_int());
    mgp::proc_add_result(proc, kFieldNode, mgp::type_node());
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