#include <thread>

#include <mg_utils.hpp>
#include <chrono>
#include "algorithm/betweenness_centrality.hpp"
using namespace std::chrono;
 
namespace {

constexpr char const *kProcedureGet = "get";

constexpr char const *kFieldBCScore = "betweenness_centrality";
constexpr char const *kFieldNode = "node";

constexpr char const *kArgumentDirected = "directed";
constexpr char const *kArgumentNormalized = "normalized";
constexpr char const *kArgumentThreads = "threads";

void InsertBCRecord(mgp_graph *graph, mgp_result *result, mgp_memory *memory, const double betweenness_centrality,
                    const int node_id) {
  auto *record = mgp::result_new_record(result);

  mg_utility::InsertNodeValueResult(graph, record, kFieldNode, node_id, memory);
  mg_utility::InsertDoubleValueResult(record, kFieldBCScore, betweenness_centrality, memory);
}

void GetBetweennessCentrality(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  try {
    auto directed = mgp::value_get_bool(mgp::list_at(args, 0));
    auto normalize = mgp::value_get_bool(mgp::list_at(args, 1));
    auto threads = mgp::value_get_int(mgp::list_at(args, 2));

    if (threads <= 0) threads = std::thread::hardware_concurrency();

    // auto graph_type = directed ? mg_graph::GraphType::kDirectedGraph : mg_graph::GraphType::kUndirectedGraph;
    auto start =steady_clock::now();

    auto adj_matrix = mg_utility::GetAdjacencyMatrix(memgraph_graph,memory,false,nullptr,1.0,directed);
    auto end = duration_cast<milliseconds>(steady_clock::now() - start).count();

    auto BC = betweenness_centrality_alg::BetweennessCentrality(adj_matrix, directed, normalize, threads);

    auto number_of_nodes = adj_matrix.size();
    for(auto& entries : BC){
      InsertBCRecord(memgraph_graph, result, memory, entries.first, entries.second);
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
  mgp_value *bool_value_directed;
  mgp_value *bool_value_normalized;
  mgp_value *int_value_threads;

  try {
    auto *proc = mgp::module_add_read_procedure(module, kProcedureGet, GetBetweennessCentrality);

    // Query module arguments
    bool_value_directed = mgp::value_make_bool(true, memory);
    bool_value_normalized = mgp::value_make_bool(true, memory);
    int_value_threads = mgp::value_make_int(std::thread::hardware_concurrency(), memory);

    mgp::proc_add_opt_arg(proc, kArgumentDirected, mgp::type_bool(), bool_value_directed);
    mgp::proc_add_opt_arg(proc, kArgumentNormalized, mgp::type_bool(), bool_value_normalized);
    mgp::proc_add_opt_arg(proc, kArgumentThreads, mgp::type_int(), int_value_threads);

    // Query module output record
    mgp::proc_add_result(proc, kFieldNode, mgp::type_node());
    mgp::proc_add_result(proc, kFieldBCScore, mgp::type_float());

  } catch (const std::exception &e) {
    // Destroy the values if exception occurs
    mgp_value_destroy(bool_value_directed);
    mgp_value_destroy(bool_value_normalized);
    mgp_value_destroy(int_value_threads);
    return 1;
  }

  mgp_value_destroy(bool_value_directed);
  mgp_value_destroy(bool_value_normalized);
  mgp_value_destroy(int_value_threads);

  return 0;
}

// This is an optional function if you need to release any resources before the
// module is unloaded. You will probably need this if you acquired some
// resources in mgp_init_module.
extern "C" int mgp_shutdown_module() {
  // Return 0 to indicate success.
  return 0;
}
