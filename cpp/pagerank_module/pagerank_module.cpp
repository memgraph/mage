#include <mg_utils.hpp>

#include "algorithm/pagerank.hpp"

const char *k_field_node = "node";
const char *k_field_rank = "rank";

void InsertPagerankRecord(const mgp_graph *graph, mgp_result *result, mgp_memory *memory, const std::uint64_t node_id,
                          double rank) {
  auto *record = mgp_result_new_record(result);
  if (record == nullptr) {
    throw mg_exception::NotEnoughMemoryException();
  }

  mg_utility::InsertNodeValueResult(graph, record, k_field_node, node_id, memory);
  mg_utility::InsertDoubleValue(record, k_field_rank, rank, memory);
}

// TODO(gitbuda): Add pagerank e2e module test.

/// Memgraph query module implementation of parallel pagerank_module algorithm.
/// PageRank is the algorithm for measuring influence of connected nodes.
///
/// @param args Memgraph module arguments
/// @param memgraphGraph Memgraph graph instance
/// @param result Memgraph result storage
/// @param memory Memgraph memory storage
void PagerankWrapper(const mgp_list *args, const mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  try {
    auto graph = mg_utility::GetGraphView(memgraph_graph, result, memory);

    auto graph_edges = graph->Edges();
    std::vector<std::pair<std::uint64_t, std::uint64_t>> pagerank_edges;
    std::transform(graph_edges.begin(), graph_edges.end(), std::back_inserter(pagerank_edges),
                   [](const mg_graph::Edge<uint64_t> &edge) -> std::pair<std::uint64_t, std::uint64_t> {
                     return {edge.from, edge.to};
                   });

    auto number_of_nodes = graph->Nodes().size();
    auto graph_nodes = graph->Nodes();

    auto pagerank_graph = pagerank_alg::PageRankGraph(number_of_nodes, pagerank_edges.size(), pagerank_edges);
    auto pageranks = pagerank_alg::ParallelIterativePageRank(pagerank_graph);

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

  if (!mgp_proc_add_result(pagerank_proc, "node", mgp_type_node())) return 1;
  if (!mgp_proc_add_result(pagerank_proc, "rank", mgp_type_float())) return 1;

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
