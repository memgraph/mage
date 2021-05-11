#include <mg_procedure.h>
#include <mg_utils.hpp>

#include "algorithm/betweenness_centrality.hpp"

namespace {

constexpr char const *kFieldBCScore = "betweeenness_centrality";
constexpr char const *kFieldNode = "node";

constexpr char const *kArgumentDirected = "directed";
constexpr char const *kArgumentNormalized = "normalized";

void InsertBCRecord(const mgp_graph *graph, mgp_result *result, mgp_memory *memory, const double betweeenness_centrality,
                       const int node_id) {
  auto *record = mgp_result_new_record(result);
  if (record == nullptr) {
    throw mg_exception::NotEnoughMemoryException();
  }
  mg_utility::InsertNodeValueResult(graph, record, kFieldNode, node_id, memory);
  mg_utility::InsertDoubleValue(record, kFieldBCScore, betweeenness_centrality, memory);
}

void GetBetweennessCentrality(const mgp_list *args, const mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  try {
    auto directed_value = mgp_list_at(args, 0);
    auto normalized_value = mgp_list_at(args, 1);

    auto directed = mgp_value_get_bool(directed_value);
    auto normalized = mgp_value_get_int(normalized_value);
  
    auto graph_type = mg_graph::GraphType::kUndirectedGraph;
    if (directed) graph_type = mg_graph::GraphType::kDirectedGraph;

    auto graph = mg_utility::GetGraphView(memgraph_graph, result, memory, graph_type);
    auto BC = betweenness_centrality_alg::BetweennessCentrality(*graph, directed, normalized);

    auto number_of_nodes = graph->Nodes().size();
    for (std::uint64_t node_id = 0; node_id < number_of_nodes; ++node_id)
      InsertBCRecord(memgraph_graph, result, memory, BC[node_id], graph->GetMemgraphNodeId(node_id));
  
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
  mgp_proc *proc = mgp_module_add_read_procedure(module, "get", GetBetweennessCentrality);
  if (!proc) return 1;

  auto bool_value_directed = mgp_value_make_bool(true, memory);
  auto bool_value_normalized =  mgp_value_make_bool(true, memory);
  // Query module arguments
  if (!mgp_proc_add_opt_arg(proc, kArgumentDirected, mgp_type_bool(), bool_value_directed))
    return 1;
  if (!mgp_proc_add_opt_arg(proc, kArgumentNormalized, mgp_type_bool(), bool_value_normalized))
    return 1;
  
  mgp_value_destroy(bool_value_directed);
  mgp_value_destroy(bool_value_normalized);

  // Query module output record
  if (!mgp_proc_add_result(proc, kFieldNode, mgp_type_node())) return 1;
  if (!mgp_proc_add_result(proc, kFieldBCScore, mgp_type_float())) return 1;

  return 0;
}

// This is an optional function if you need to release any resources before the
// module is unloaded. You will probably need this if you acquired some
// resources in mgp_init_module.
extern "C" int mgp_shutdown_module() {
  // Return 0 to indicate success.
  return 0;
}
