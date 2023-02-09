#include <mg_utils.hpp>

#include "algorithm_online/pagerank.hpp"

constexpr char const *kProcedureSet = "set";
constexpr char const *kProcedureGet = "get";
constexpr char const *kProcedureUpdate = "update";
constexpr char const *kProcedureReset = "reset";

constexpr char const *kFieldNode = "node";
constexpr char const *kFieldRank = "rank";
constexpr char const *kFieldMessage = "message";

constexpr char const *kArgumentWalksPerNode = "walks_per_node";
constexpr char const *kArgumentWalkStopEpsilon = "walk_stop_epsilon";

constexpr char const *kArgumentCreatedVertices = "created_vertices";
constexpr char const *kArgumentCreatedEdges = "created_edges";
constexpr char const *kArgumentDeletedVertices = "deleted_vertices";
constexpr char const *kArgumentDeletedEdges = "deleted_edges";

void InsertPageRankRecord(mgp_graph *graph, mgp_result *result, mgp_memory *memory, const std::uint64_t node_id,
                          const double rank) {
  auto *record = mgp::result_new_record(result);

  mg_utility::InsertNodeValueResult(graph, record, kFieldNode, node_id, memory);
  mg_utility::InsertDoubleValueResult(record, kFieldRank, rank, memory);
}

void InsertMessageRecord(mgp_result *result, mgp_memory *memory, const char *message) {
  auto *record = mgp::result_new_record(result);

  mg_utility::InsertStringValueResult(record, kFieldMessage, message, memory);
}

void OnlinePageRankGet(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  try {
    auto graph = mg_utility::GetGraphView(memgraph_graph, result, memory, mg_graph::GraphType::kDirectedGraph);

    auto pageranks = pagerank_online_alg::ContextEmpty() ? pagerank_online_alg::SetPageRank(*graph)
                                                         : pagerank_online_alg::GetPageRank(*graph);

    for (auto const &[node_id, rank] : pageranks) {
      InsertPageRankRecord(memgraph_graph, result, memory, node_id, rank);
    }
  } catch (const std::exception &e) {
    mgp::result_set_error_msg(result, e.what());
    return;
  }
}

void OnlinePageRankSet(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  try {
    auto walks_per_node = mgp::value_get_int(mgp::list_at(args, 0));
    auto walk_stop_epsilon = mgp::value_get_double(mgp::list_at(args, 1));

    auto graph = mg_utility::GetGraphView(memgraph_graph, result, memory, mg_graph::GraphType::kDirectedGraph);

    auto pageranks = pagerank_online_alg::SetPageRank(*graph, walks_per_node, walk_stop_epsilon);

    for (auto const &[node_id, rank] : pageranks) {
      InsertPageRankRecord(memgraph_graph, result, memory, node_id, rank);
    }
  } catch (const std::exception &e) {
    mgp::result_set_error_msg(result, e.what());
    return;
  }
}

void OnlinePageRankUpdate(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  try {
    mgp::memory = memory;

    const auto graph = mgp::Graph(memgraph_graph);
    const auto arguments = mgp::List(args);
    const auto record_factory = mgp::RecordFactory(result);

    std::vector<std::pair<uint64_t, double>> pageranks;

    if (pagerank_online_alg::ContextEmpty()) {
      auto legacy_graph = mg_utility::GetGraphView(memgraph_graph, result, memory, mg_graph::GraphType::kDirectedGraph);
      pageranks = pagerank_online_alg::SetPageRank(*legacy_graph);
    } else {
      const auto created_nodes_ = arguments[0].ValueList();
      const auto created_relationships_ = arguments[1].ValueList();
      const auto deleted_nodes_ = arguments[2].ValueList();
      const auto deleted_relationships_ = arguments[3].ValueList();

      std::vector<std::uint64_t> created_nodes;
      std::vector<std::pair<std::uint64_t, std::uint64_t>> created_relationships;
      std::vector<std::uint64_t> deleted_nodes;
      std::vector<std::pair<std::uint64_t, std::uint64_t>> deleted_relationships;

      for (const auto &node : created_nodes_) {
        created_nodes.push_back(node.ValueNode().Id().AsUint());
      }
      for (const auto &relationship : created_relationships_) {
        created_relationships.push_back({relationship.ValueRelationship().From().Id().AsUint(),
                                         relationship.ValueRelationship().To().Id().AsUint()});
      }
      for (const auto &node : deleted_nodes_) {
        deleted_nodes.push_back(node.ValueNode().Id().AsUint());
      }
      for (const auto &relationship : deleted_relationships_) {
        deleted_relationships.push_back({relationship.ValueRelationship().From().Id().AsUint(),
                                         relationship.ValueRelationship().To().Id().AsUint()});
      }

      pageranks = pagerank_online_alg::UpdatePageRank(graph, created_nodes, created_relationships, deleted_nodes,
                                                      deleted_relationships);
    }

    for (auto const &[node_id, rank] : pageranks) {
      auto record = record_factory.NewRecord();
      record.Insert(kFieldNode, graph.GetNodeById(mgp::Id::FromUint(node_id)));
      record.Insert(kFieldRank, rank);
    }
  } catch (const std::exception &e) {
    mgp::result_set_error_msg(result, e.what());
    return;
  }
}

void OnlinePageRankReset(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  try {
    pagerank_online_alg::Reset();
    InsertMessageRecord(result, memory, "The context has been reset!");
  } catch (const std::exception &e) {
    InsertMessageRecord(result, memory,
                        "Reset failed: An exception occurred, please check your `pagerank_online` module!");
  }
}

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  // Approximative PageRank computation
  {
    try {
      auto *pagerank_proc = mgp::module_add_read_procedure(module, kProcedureSet, OnlinePageRankSet);

      auto default_walks_per_node = mgp::value_make_int(10, memory);
      auto default_walk_stop_epsilon = mgp::value_make_double(0.1, memory);

      mgp::proc_add_opt_arg(pagerank_proc, kArgumentWalksPerNode, mgp::type_int(), default_walks_per_node);
      mgp::proc_add_opt_arg(pagerank_proc, kArgumentWalkStopEpsilon, mgp::type_float(), default_walk_stop_epsilon);

      mgp::value_destroy(default_walks_per_node);
      mgp::value_destroy(default_walk_stop_epsilon);

      mgp::proc_add_result(pagerank_proc, kFieldNode, mgp::type_node());
      mgp::proc_add_result(pagerank_proc, kFieldRank, mgp::type_float());
    } catch (const std::exception &e) {
      return 1;
    }
  }

  // Retrieval of previously calculated PageRank results
  {
    try {
      auto *pagerank_proc = mgp::module_add_read_procedure(module, kProcedureGet, OnlinePageRankGet);

      mgp::proc_add_result(pagerank_proc, kFieldNode, mgp::type_node());
      mgp::proc_add_result(pagerank_proc, kFieldRank, mgp::type_float());

    } catch (const std::exception &e) {
      return 1;
    }
  }

  // Dynamic PageRank update
  {
    try {
      auto *pagerank_proc = mgp::module_add_read_procedure(module, kProcedureUpdate, OnlinePageRankUpdate);

      auto default_created_vertices = mgp::value_make_list(mgp::list_make_empty(0, memory));
      auto default_created_edges = mgp::value_make_list(mgp::list_make_empty(0, memory));
      auto default_deleted_vertices = mgp::value_make_list(mgp::list_make_empty(0, memory));
      auto default_deleted_edges = mgp::value_make_list(mgp::list_make_empty(0, memory));

      mgp::proc_add_opt_arg(pagerank_proc, kArgumentCreatedVertices,
                            mgp::type_nullable(mgp::type_list(mgp::type_node())), default_created_vertices);
      mgp::proc_add_opt_arg(pagerank_proc, kArgumentCreatedEdges,
                            mgp::type_nullable(mgp::type_list(mgp::type_relationship())), default_created_edges);
      mgp::proc_add_opt_arg(pagerank_proc, kArgumentDeletedVertices,
                            mgp::type_nullable(mgp::type_list(mgp::type_node())), default_deleted_vertices);
      mgp::proc_add_opt_arg(pagerank_proc, kArgumentDeletedEdges,
                            mgp::type_nullable(mgp::type_list(mgp::type_relationship())), default_deleted_edges);

      mgp::value_destroy(default_created_vertices);
      mgp::value_destroy(default_created_edges);
      mgp::value_destroy(default_deleted_vertices);
      mgp::value_destroy(default_deleted_edges);

      mgp::proc_add_result(pagerank_proc, kFieldNode, mgp::type_node());
      mgp::proc_add_result(pagerank_proc, kFieldRank, mgp::type_float());

    } catch (const std::exception &e) {
      return 1;
    }
  }

  // Reset procedure
  {
    try {
      auto *pagerank_proc = mgp::module_add_read_procedure(module, kProcedureReset, OnlinePageRankReset);
      mgp::proc_add_result(pagerank_proc, kFieldMessage, mgp::type_string());

    } catch (const std::exception &e) {
      return 1;
    }
  }

  // Reset data structures upon reloading the module
  pagerank_online_alg::Reset();

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
