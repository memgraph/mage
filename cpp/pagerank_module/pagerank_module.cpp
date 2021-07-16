#include <mg_utils.hpp>

#include "algorithm/pagerank.hpp"
#include "algorithm_approx/pagerank.hpp"

constexpr char const *kFieldNode = "node";
constexpr char const *kFieldRank = "rank";

constexpr char const *kArgumentMaxIterations = "max_iterations";
constexpr char const *kArgumentDampingFactor = "damping_factor";
constexpr char const *kArgumentStopEpsilon = "stop_epsilon";

constexpr char const *kArgumentWalksPerNode = "walks_per_node";
constexpr char const *kArgumentWalkStopEpsilon = "walk_stop_epsilon";

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
void ExactPageRankWrapper(const mgp_list *args, const mgp_graph *memgraph_graph, mgp_result *result,
                          mgp_memory *memory) {
  try {
    auto max_iterations = mgp_value_get_int(mgp_list_at(args, 0));
    auto damping_factor = mgp_value_get_double(mgp_list_at(args, 1));
    auto stop_epsilon = mgp_value_get_double(mgp_list_at(args, 2));

    auto graph = mg_utility::GetGraphView(memgraph_graph, result, memory, mg_graph::GraphType::kDirectedGraph);

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

void ApproxPageRankWrapper(const mgp_list *args, const mgp_graph *memgraph_graph, mgp_result *result,
                           mgp_memory *memory) {
  try {
    auto walks_per_node = mgp_value_get_int(mgp_list_at(args, 0));
    auto walk_stop_epsilon = mgp_value_get_double(mgp_list_at(args, 1));

    auto graph = mg_utility::GetGraphView(memgraph_graph, result, memory, mg_graph::GraphType::kDirectedGraph);

    auto pageranks = pagerank_approx_alg::PageRankApprox(*graph, walks_per_node, walk_stop_epsilon);

    auto number_of_nodes = graph->Nodes().size();
    for (std::uint64_t node_id = 0; node_id < number_of_nodes; ++node_id) {
      InsertPagerankRecord(memgraph_graph, result, memory, graph->GetMemgraphNodeId(node_id), pageranks[node_id]);
    }
  } catch (const std::exception &e) {
    // We must not let any exceptions out of our module.
    mgp_result_set_error_msg(result, e.what());
    return;
  }
}

void ApproxPageRankAddEdge(const mgp_list *args, const mgp_graph *memgraph_graph, mgp_result *result,
                           mgp_memory *memory) {
  try {
    auto edge = mgp_value_get_edge(mgp_list_at(args, 0));
    auto from = mgp_vertex_get_id(mgp_edge_get_from(edge)).as_int;
    auto to = mgp_vertex_get_id(mgp_edge_get_to(edge)).as_int;

    auto graph = mg_utility::GetGraphView(memgraph_graph, result, memory, mg_graph::GraphType::kDirectedGraph);

    graph->CreateEdge(from, to, mg_graph::GraphType::kDirectedGraph);

    auto pageranks = pagerank_approx_alg::Update(*graph, {graph->GetInnerNodeId(from), graph->GetInnerNodeId(to)});

    auto number_of_nodes = graph->Nodes().size();
    for (std::uint64_t node_id = 0; node_id < number_of_nodes; ++node_id) {
      InsertPagerankRecord(memgraph_graph, result, memory, graph->GetMemgraphNodeId(node_id), pageranks[node_id]);
    }
  } catch (const std::exception &e) {
    // We must not let any exceptions out of our module.
    mgp_result_set_error_msg(result, e.what());
    return;
  }
}

void ApproxPageRankAddNode(const mgp_list *args, const mgp_graph *memgraph_graph, mgp_result *result,
                           mgp_memory *memory) {
  try {
    auto node_id = mgp_vertex_get_id(mgp_value_get_vertex(mgp_list_at(args, 0))).as_int;

    auto graph = mg_utility::GetGraphView(memgraph_graph, result, memory, mg_graph::GraphType::kDirectedGraph);
    graph->CreateNode(node_id);

    auto pageranks = pagerank_approx_alg::Update(*graph, graph->GetInnerNodeId(node_id));

    auto number_of_nodes = graph->Nodes().size();
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
  // Exact deterministic PageRank solution
  {
    struct mgp_proc *pagerank_proc = mgp_module_add_read_procedure(module, "get", ExactPageRankWrapper);

    if (!pagerank_proc) return 1;

    auto default_max_iterations = mgp_value_make_int(100, memory);
    auto default_damping_factor = mgp_value_make_double(0.85, memory);
    auto default_stop_epsilon = mgp_value_make_double(1e-5, memory);

    if (!mgp_proc_add_opt_arg(pagerank_proc, kArgumentMaxIterations, mgp_type_int(), default_max_iterations)) return 1;
    if (!mgp_proc_add_opt_arg(pagerank_proc, kArgumentDampingFactor, mgp_type_float(), default_damping_factor))
      return 1;
    if (!mgp_proc_add_opt_arg(pagerank_proc, kArgumentStopEpsilon, mgp_type_float(), default_stop_epsilon)) return 1;

    mgp_value_destroy(default_max_iterations);
    mgp_value_destroy(default_damping_factor);
    mgp_value_destroy(default_stop_epsilon);

    // Query module output record
    if (!mgp_proc_add_result(pagerank_proc, kFieldNode, mgp_type_node())) return 1;
    if (!mgp_proc_add_result(pagerank_proc, kFieldRank, mgp_type_float())) return 1;
  }

  // Approximate PageRank solution
  {
    struct mgp_proc *pagerank_proc = mgp_module_add_read_procedure(module, "approx", ApproxPageRankWrapper);

    if (!pagerank_proc) return 1;

    auto default_walks_per_node = mgp_value_make_int(10, memory);
    auto default_walk_stop_epsilon = mgp_value_make_double(0.1, memory);

    if (!mgp_proc_add_opt_arg(pagerank_proc, kArgumentWalksPerNode, mgp_type_int(), default_walks_per_node)) return 1;
    if (!mgp_proc_add_opt_arg(pagerank_proc, kArgumentStopEpsilon, mgp_type_float(), default_walk_stop_epsilon))
      return 1;

    mgp_value_destroy(default_walks_per_node);
    mgp_value_destroy(default_walk_stop_epsilon);

    // Query module output record
    if (!mgp_proc_add_result(pagerank_proc, kFieldNode, mgp_type_node())) return 1;
    if (!mgp_proc_add_result(pagerank_proc, kFieldRank, mgp_type_float())) return 1;
  }

  // Approximate PageRank Update edge
  {
    struct mgp_proc *pagerank_proc = mgp_module_add_read_procedure(module, "approx_add_edge", ApproxPageRankAddEdge);

    if (!pagerank_proc) return 1;

    if (!mgp_proc_add_arg(pagerank_proc, "edge", mgp_type_relationship())) return 1;

    // Query module output record
    if (!mgp_proc_add_result(pagerank_proc, kFieldNode, mgp_type_node())) return 1;
    if (!mgp_proc_add_result(pagerank_proc, kFieldRank, mgp_type_float())) return 1;
  }

  // Approximate PageRank Update node
  {
    struct mgp_proc *pagerank_proc = mgp_module_add_read_procedure(module, "approx_add_node", ApproxPageRankAddNode);

    if (!pagerank_proc) return 1;

    if (!mgp_proc_add_arg(pagerank_proc, "node", mgp_type_node())) return 1;

    // Query module output record
    if (!mgp_proc_add_result(pagerank_proc, kFieldNode, mgp_type_node())) return 1;
    if (!mgp_proc_add_result(pagerank_proc, kFieldRank, mgp_type_float())) return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
