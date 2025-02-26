#include "algorithm/pagerank.hpp"

#include <mg_utils.hpp>

namespace {
constexpr char const *kProcedureGet = "get";

constexpr char const *kFieldNode = "node";
constexpr char const *kFieldRank = "rank";

constexpr char const *kArgumentMaxIterations = "max_iterations";
constexpr char const *kArgumentDampingFactor = "damping_factor";
constexpr char const *kArgumentStopEpsilon = "stop_epsilon";

void InsertPagerankRecord(mgp_graph *graph, mgp_result *result, mgp_memory *memory, const std::uint64_t node_id,
                          double rank) {
  auto *vertex = mgp::graph_get_vertex_by_id(graph, mgp_vertex_id{.as_int = static_cast<int64_t>(node_id)}, memory);
  if (!vertex) {
    if (mgp::graph_is_transactional(graph)) {
      throw mg_exception::InvalidIDException();
    }
    return;
  }

  auto *record = mgp::result_new_record(result);
  if (record == nullptr) throw mg_exception::NotEnoughMemoryException();

  mg_utility::InsertNodeValueResult(record, kFieldNode, vertex, memory);
  mg_utility::InsertDoubleValueResult(record, kFieldRank, rank, memory);
}

/// Memgraph query module implementation of parallel pagerank_module algorithm.
/// PageRank is the algorithm for measuring influence of connected nodes.
///
/// @param args Memgraph module arguments
/// @param memgraphGraph Memgraph graph instance
/// @param result Memgraph result storage
/// @param memory Memgraph memory storage
void PagerankWrapper(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  try {
    auto max_iterations = mgp::value_get_int(mgp::list_at(args, 0));
    auto damping_factor = mgp::value_get_double(mgp::list_at(args, 1));
    auto stop_epsilon = mgp::value_get_double(mgp::list_at(args, 2));

    auto pagerank_graph = pagerank_alg::PageRankGraph(memgraph_graph, memory);
    auto pageranks =
        pagerank_alg::ParallelIterativePageRank(pagerank_graph, max_iterations, damping_factor, stop_epsilon);

    for (std::uint64_t node_id = 0; node_id < pagerank_graph.GetNodeCount(); ++node_id) {
      InsertPagerankRecord(memgraph_graph, result, memory, pagerank_graph.GetMemgraphNodeId(node_id), pageranks[node_id]);
    }
  } catch (const std::exception &e) {
    // We must not let any exceptions out of our module.
    mgp::result_set_error_msg(result, e.what());
    return;
  }
}
}  // namespace

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  mgp_value *default_max_iterations;
  mgp_value *default_damping_factor;
  mgp_value *default_stop_epsilon;
  try {
    auto *pagerank_proc = mgp::module_add_read_procedure(module, kProcedureGet, PagerankWrapper);

    default_max_iterations = mgp::value_make_int(100, memory);
    default_damping_factor = mgp::value_make_double(0.85, memory);
    default_stop_epsilon = mgp::value_make_double(1e-5, memory);

    mgp::proc_add_opt_arg(pagerank_proc, kArgumentMaxIterations, mgp::type_int(), default_max_iterations);
    mgp::proc_add_opt_arg(pagerank_proc, kArgumentDampingFactor, mgp::type_float(), default_damping_factor);
    mgp::proc_add_opt_arg(pagerank_proc, kArgumentStopEpsilon, mgp::type_float(), default_stop_epsilon);

    // Query module output record
    mgp::proc_add_result(pagerank_proc, kFieldNode, mgp::type_node());
    mgp::proc_add_result(pagerank_proc, kFieldRank, mgp::type_float());

  } catch (const std::exception &e) {
    // Destroy values if exception occurs earlier
    mgp_value_destroy(default_max_iterations);
    mgp_value_destroy(default_damping_factor);
    mgp_value_destroy(default_stop_epsilon);
    return 1;
  }

  mgp_value_destroy(default_max_iterations);
  mgp_value_destroy(default_damping_factor);
  mgp_value_destroy(default_stop_epsilon);

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
