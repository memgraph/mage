#include <mg_procedure.h>
#include <mg_utils.hpp>

#include "algorithm/betweenness_centrality.hpp"

namespace {

constexpr char const *kFieldBCScore = "betweeenness_centrality";
constexpr char const *kFieldNode = "node";

void InsertBCRecord(const mgp_graph *graph, mgp_result *result, mgp_memory *memory, const double betweeenness_centrality,
                       const int node_id) {
  auto *record = mgp_result_new_record(result);
  if (record == nullptr) {
    throw mg_exception::NotEnoughMemoryException();
  }

  mg_utility::InsertDoubleValue(record, kFieldBCScore, betweeenness_centrality, memory);
  mg_utility::InsertNodeValueResult(graph, record, kFieldNode, node_id, memory);
}

void BetweennessCentrality(const mgp_list *args, const mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  try {
    auto graph = mg_utility::GetGraphView(memgraph_graph, result, memory);
    auto BC = betweenness_centrality_alg::BetweennessCentralityUnweightedDirected(*graph);

    auto number_of_nodes = graph->Nodes().size();
    for (std::uint64_t node_id = 0; node_id < number_of_nodes; ++node_id) {
      InsertBCRecord(memgraph_graph, result, memory, BC[node_id], graph->GetMemgraphNodeId(node_id));
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
  mgp_proc *proc = mgp_module_add_read_procedure(module, "get", BetweennessCentrality);

  if (!proc) return 1;

  if (!mgp_proc_add_result(proc, "node", mgp_type_node())) return 1;
  if (!mgp_proc_add_result(proc, "betweenness", mgp_type_float())) return 1;

  return 0;
}

// This is an optional function if you need to release any resources before the
// module is unloaded. You will probably need this if you acquired some
// resources in mgp_init_module.
extern "C" int mgp_shutdown_module() {
  // Return 0 to indicate success.
  return 0;
}
