#include <mg_utils.hpp>

#include <iostream>
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

    auto pageranks = pagerank_approx_alg::SetPagerank(*graph, walks_per_node, walk_stop_epsilon);

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

// void ApproxPageRankCreateUpdate(const mgp_list *args, const mgp_graph *memgraph_graph, mgp_result *result,
//                                 mgp_memory *memory) {
//   try {
//     std::vector<double> pageranks;

//     auto nodes_list = mgp_value_get_list(mgp_list_at(args, 0));
//     auto edges_list = mgp_value_get_list(mgp_list_at(args, 1));

//     auto graph = mg_utility::GetGraphView(memgraph_graph, result, memory, mg_graph::GraphType::kDirectedGraph);

//     auto nodes_size = mgp_list_size(nodes_list);
//     for (std::size_t i = 0; i < nodes_size; i++) {
//       auto vertex = mgp_value_get_vertex(mgp_list_at(nodes_list, i));
//       auto vertex_id = mgp_vertex_get_id(vertex).as_int;

//       pageranks = pagerank_approx_alg::UpdateCreate(*graph, graph->GetInnerNodeId(vertex_id));
//     }

//     auto edges_size = mgp_list_size(edges_list);
//     for (std::size_t i = 0; i < edges_size; i++) {
//       auto edge = mgp_value_get_edge(mgp_list_at(edges_list, i));
//       auto from_id = mgp_vertex_get_id(mgp_edge_get_from(edge)).as_int;
//       auto to_id = mgp_vertex_get_id(mgp_edge_get_to(edge)).as_int;

//       pageranks =
//           pagerank_approx_alg::UpdateCreate(*graph, {graph->GetInnerNodeId(from_id), graph->GetInnerNodeId(to_id)});
//     }

//     auto number_of_nodes = graph->Nodes().size();
//     for (std::uint64_t node_id = 0; node_id < number_of_nodes; ++node_id) {
//       InsertPagerankRecord(memgraph_graph, result, memory, graph->GetMemgraphNodeId(node_id), pageranks[node_id]);
//     }
//   } catch (const std::exception &e) {
//     // We must not let any exceptions out of our module.
//     mgp_result_set_error_msg(result, e.what());
//     return;
//   }
// }

// void ApproxPageRankDeleteUpdate(const mgp_list *args, const mgp_graph *memgraph_graph, mgp_result *result,
//                                 mgp_memory *memory) {
//   try {
//     std::vector<double> pageranks;

//     auto nodes_list = mgp_value_get_list(mgp_list_at(args, 0));
//     auto edges_list = mgp_value_get_list(mgp_list_at(args, 1));

//     auto graph = mg_utility::GetGraphView(memgraph_graph, result, memory, mg_graph::GraphType::kDirectedGraph);

//     auto edges_size = mgp_list_size(edges_list);
//     for (std::size_t i = 0; i < edges_size; i++) {
//       auto edge = mgp_value_get_edge(mgp_list_at(edges_list, i));
//       auto from_id = mgp_vertex_get_id(mgp_edge_get_from(edge)).as_int;
//       auto to_id = mgp_vertex_get_id(mgp_edge_get_to(edge)).as_int;

//       pageranks =
//           pagerank_approx_alg::UpdateDelete(*graph, {graph->GetInnerNodeId(from_id), graph->GetInnerNodeId(to_id)});
//     }

//     auto nodes_size = mgp_list_size(nodes_list);
//     for (std::size_t i = 0; i < nodes_size; i++) {
//       auto vertex = mgp_value_get_vertex(mgp_list_at(nodes_list, i));
//       auto vertex_id = mgp_vertex_get_id(vertex).as_int;

//       pageranks = pagerank_approx_alg::UpdateDelete(*graph, graph->GetInnerNodeId(vertex_id));
//     }

//     auto number_of_nodes = graph->Nodes().size();
//     for (std::uint64_t node_id = 0; node_id < number_of_nodes; ++node_id) {
//       InsertPagerankRecord(memgraph_graph, result, memory, graph->GetMemgraphNodeId(node_id), pageranks[node_id]);
//     }
//   } catch (const std::exception &e) {
//     // We must not let any exceptions out of our module.
//     mgp_result_set_error_msg(result, e.what());
//     return;
//   }
// }

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

  // // Approximate PageRank Create Edges/Nodes
  // {
  //   struct mgp_proc *pagerank_proc =
  //       mgp_module_add_read_procedure(module, "approx_create_update", ApproxPageRankCreateUpdate);

  //   if (!pagerank_proc) return 1;

  //   auto default_nodes = mgp_value_make_null(memory);
  //   auto default_edges = mgp_value_make_null(memory);

  //   if (!mgp_proc_add_opt_arg(pagerank_proc, "create_nodes", mgp_type_nullable(mgp_type_list(mgp_type_node())),
  //                             default_nodes))
  //     return 1;
  //   if (!mgp_proc_add_opt_arg(pagerank_proc, "created_edges",
  //   mgp_type_nullable(mgp_type_list(mgp_type_relationship())),
  //                             default_edges))
  //     return 1;

  //   mgp_value_destroy(default_nodes);
  //   mgp_value_destroy(default_edges);

  //   // Query module output record
  //   if (!mgp_proc_add_result(pagerank_proc, kFieldNode, mgp_type_node())) return 1;
  //   if (!mgp_proc_add_result(pagerank_proc, kFieldRank, mgp_type_float())) return 1;
  // }

  // {
  //   struct mgp_proc *pagerank_proc =
  //       mgp_module_add_read_procedure(module, "approx_delete_update", ApproxPageRankCreateUpdate);

  //   if (!pagerank_proc) return 1;

  //   auto default_nodes = mgp_value_make_null(memory);
  //   auto default_edges = mgp_value_make_null(memory);

  //   if (!mgp_proc_add_opt_arg(pagerank_proc, "delete_nodes", mgp_type_nullable(mgp_type_list(mgp_type_node())),
  //                             default_nodes))
  //     return 1;
  //   if (!mgp_proc_add_opt_arg(pagerank_proc, "delete_edges",
  //   mgp_type_nullable(mgp_type_list(mgp_type_relationship())),
  //                             default_edges))
  //     return 1;

  //   mgp_value_destroy(default_nodes);
  //   mgp_value_destroy(default_edges);

  //   // Query module output record
  //   if (!mgp_proc_add_result(pagerank_proc, kFieldNode, mgp_type_node())) return 1;
  //   if (!mgp_proc_add_result(pagerank_proc, kFieldRank, mgp_type_float())) return 1;
  // }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
