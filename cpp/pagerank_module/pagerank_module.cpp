#include <mg_utils.hpp>

#include "algorithm/pagerank.hpp"

constexpr char const *kFieldNode = "node";
constexpr char const *kFieldRank = "rank";

constexpr char const *kArgumentMaxIterations = "max_iterations";
constexpr char const *kArgumentDampingFactor = "damping_factor";
constexpr char const *kArgumentStopEpsilon = "stop_epsilon";

void InsertPagerankRecord(const mgp_graph *graph, mgp_result *result, mgp_memory *memory, const std::uint64_t node_id,
                          double rank) {
  auto *record = mgp_result_new_record(result);
  if (record == nullptr) {
    throw mg_exception::NotEnoughMemoryException();
  }

  mg_utility::InsertNodeValueResult(graph, record, kFieldNode, node_id, memory);
  mg_utility::InsertDoubleValue(record, kFieldRank, rank, memory);
}

/// Memgraph query module implementation of parallel pagerank_module algorithm.
/// PageRank is the algorithm for measuring influence of connected nodes.
///
/// @param args Memgraph module arguments
/// @param memgraphGraph Memgraph graph instance
/// @param result Memgraph result storage
/// @param memory Memgraph memory storage
void PagerankWrapper(const mgp_list *args, const mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  try {
    auto max_iterations = mgp_value_get_int(mgp_list_at(args, 0));
    auto damping_factor = mgp_value_get_double(mgp_list_at(args, 1));
    auto stop_epsilon = mgp_value_get_double(mgp_list_at(args, 2));

    auto graph = mg_utility::GetGraphView(memgraph_graph, result, memory);

    auto graph_edges = graph->Edges();
    std::vector<pagerank_alg::EdgePair> pagerank_edges;
    std::transform(graph_edges.begin(), graph_edges.end(), std::back_inserter(pagerank_edges),
                   [](const mg_graph::Edge<uint64_t> &edge) -> pagerank_alg::EdgePair {
                     return {edge.from, edge.to};
                   });

    auto number_of_nodes = graph->Nodes().size();

    auto pagerank_graph = pagerank_alg::PageRankGraph(number_of_nodes, pagerank_edges.size(), pagerank_edges);
    auto pageranks =
        pagerank_alg::ParallelIterativePageRank(pagerank_graph, max_iterations, damping_factor, stop_epsilon);

    for (std::uint64_t node_id = 0; node_id < number_of_nodes; ++node_id) {
      InsertPagerankRecord(memgraph_graph, result, memory, graph->GetMemgraphNodeId(node_id), pageranks[node_id]);
    }
  } catch (const std::exception &e) {
    // We must not let any exceptions out of our module.
    mgp_result_set_error_msg(result, e.what());
    return;
  }
}

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  struct mgp_proc *pagerank_proc = mgp_module_add_read_procedure(module, "get", PagerankWrapper);

  if (!pagerank_proc) return 1;

  auto default_max_iterations = mgp_value_make_int(100, memory);
  auto default_damping_factor = mgp_value_make_double(0.85, memory);
  auto default_stop_epsilon = mgp_value_make_double(1e-5, memory);

  if (!mgp_proc_add_opt_arg(pagerank_proc, kArgumentMaxIterations, mgp_type_int(), default_max_iterations)) return 1;
  if (!mgp_proc_add_opt_arg(pagerank_proc, kArgumentDampingFactor, mgp_type_float(), default_damping_factor)) return 1;
  if (!mgp_proc_add_opt_arg(pagerank_proc, kArgumentStopEpsilon, mgp_type_float(), default_stop_epsilon)) return 1;

  mgp_value_destroy(default_max_iterations);
  mgp_value_destroy(default_damping_factor);
  mgp_value_destroy(default_stop_epsilon);

  // Query module output record
  if (!mgp_proc_add_result(pagerank_proc, kFieldNode, mgp_type_node())) return 1;
  if (!mgp_proc_add_result(pagerank_proc, kFieldRank, mgp_type_float())) return 1;

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
