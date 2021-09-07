#include <thread>

#include <mg_procedure.h>
#include <mg_utils.hpp>

#include "algorithm/betweenness_centrality.hpp"

namespace {

constexpr char const *kFieldBCScore = "betweenness_centrality";
constexpr char const *kFieldNode = "node";

constexpr char const *kArgumentDirected = "directed";
constexpr char const *kArgumentNormalized = "normalized";
constexpr char const *kArgumentThreads = "threads";

void InsertBCRecord(const mgp_graph *graph, mgp_result *result, mgp_memory *memory,
                    const double betweenness_centrality, const int node_id) {
  auto *record = mgp_result_new_record(result);
  if (record == nullptr) {
    throw mg_exception::NotEnoughMemoryException();
  }
  mg_utility::InsertNodeValueResult(graph, record, kFieldNode, node_id, memory);
  mg_utility::InsertDoubleValue(record, kFieldBCScore, betweenness_centrality, memory);
}

void GetBetweennessCentrality(const mgp_list *args, const mgp_graph *memgraph_graph, mgp_result *result,
                              mgp_memory *memory) {
  try {
    auto directed = mgp_value_get_bool(mgp_list_at(args, 0));
    auto normalize = mgp_value_get_bool(mgp_list_at(args, 1));
    auto threads = mgp_value_get_int(mgp_list_at(args, 2));

    if (threads <= 0) threads = std::thread::hardware_concurrency();

    auto graph_type = directed ? mg_graph::GraphType::kDirectedGraph : mg_graph::GraphType::kUndirectedGraph;

    auto graph = mg_utility::GetGraphView(memgraph_graph, result, memory, graph_type);
    auto BC = betweenness_centrality_alg::BetweennessCentrality(*graph, directed, normalize, threads);

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

  // Query module arguments
  auto bool_value_directed = mgp_value_make_bool(true, memory);
  auto bool_value_normalized = mgp_value_make_bool(true, memory);
  auto int_value_threads = mgp_value_make_int(std::thread::hardware_concurrency(), memory);

  if (!mgp_proc_add_opt_arg(proc, kArgumentDirected, mgp_type_bool(), bool_value_directed)) return 1;
  if (!mgp_proc_add_opt_arg(proc, kArgumentNormalized, mgp_type_bool(), bool_value_normalized)) return 1;
  if (!mgp_proc_add_opt_arg(proc, kArgumentThreads, mgp_type_int(), int_value_threads)) return 1;

  mgp_value_destroy(bool_value_directed);
  mgp_value_destroy(bool_value_normalized);
  mgp_value_destroy(int_value_threads);

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
