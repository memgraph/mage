#include <mg_utils.hpp>

#include "algorithm/katz.hpp"

namespace {

constexpr char const *kProcedureSet = "set";
constexpr char const *kProcedureGet = "get";
constexpr char const *kProcedureUpdate = "update";
constexpr char const *kProcedureReset = "reset";

constexpr char const *kFieldNode = "node";
constexpr char const *kFieldRank = "rank";
constexpr char const *kFieldMessage = "message";

constexpr char const *kArgumentAlpha = "alpha";
constexpr char const *kArgumentEpsilon = "epsilon";
constexpr char const *kArgumentCreatedVertices = "created_vertices";
constexpr char const *kArgumentCreatedEdges = "created_edges";
constexpr char const *kArgumentDeletedVertices = "deleted_vertices";
constexpr char const *kArgumentDeletedEdges = "deleted_edges";

void InsertKatzRecord(mgp_graph *graph, mgp_result *result, mgp_memory *memory, const double katz_centrality,
                      const int node_id) {
  auto *node = mgp::graph_get_vertex_by_id(graph, mgp_vertex_id{.as_int = static_cast<int64_t>(node_id)}, memory);
  if (!node) {
    if (mgp::graph_is_transactional(graph)) {
      throw mg_exception::InvalidIDException();
    }
    // In IN_MEMORY_ANALYTICAL mode, vertices/edges may be erased by parallel transactions.
    return;
  }

  auto *record = mgp::result_new_record(result);
  if (record == nullptr) throw mg_exception::NotEnoughMemoryException();

  mg_utility::InsertNodeValueResult(record, kFieldNode, node, memory);
  mg_utility::InsertDoubleValueResult(record, kFieldRank, katz_centrality, memory);
}

void InsertMessageRecord(mgp_result *result, mgp_memory *memory, const char *message) {
  auto *record = mgp::result_new_record(result);

  mg_utility::InsertStringValueResult(record, kFieldMessage, message, memory);
}

void GetKatzCentrality(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  try {
    auto graph = mg_utility::GetGraphView(memgraph_graph, result, memory, mg_graph::GraphType::kDirectedGraph);
    auto katz_centralities = katz_alg::GetKatz(*graph);

    for (auto &[vertex_id, centrality] : katz_centralities) {
      InsertKatzRecord(memgraph_graph, result, memory, centrality, vertex_id);
    }
  } catch (const std::exception &e) {
    mgp::result_set_error_msg(result, e.what());
    return;
  }
}

void SetKatzCentrality(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  try {
    auto alpha = mgp::value_get_double(mgp::list_at(args, 0));
    auto epsilon = mgp::value_get_double(mgp::list_at(args, 1));

    auto graph = mg_utility::GetGraphView(memgraph_graph, result, memory, mg_graph::GraphType::kDirectedGraph);
    auto katz_centralities = katz_alg::SetKatz(*graph, alpha, epsilon);

    for (auto &[vertex_id, centrality] : katz_centralities) {
      InsertKatzRecord(memgraph_graph, result, memory, centrality, vertex_id);
    }
  } catch (const std::exception &e) {
    mgp::result_set_error_msg(result, e.what());
    return;
  }
}

void UpdateKatzCentrality(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  try {
    mgp::MemoryDispatcherGuard guard{memory};

    const auto record_factory = mgp::RecordFactory(result);
    const auto graph = mgp::Graph(memgraph_graph);
    const auto arguments = mgp::List(args);

    std::vector<std::pair<uint64_t, double>> katz_centralities;

    if (katz_alg::NoPreviousData()) {
      auto legacy_graph = mg_utility::GetGraphView(memgraph_graph, result, memory, mg_graph::GraphType::kDirectedGraph);
      katz_centralities = katz_alg::SetKatz(*legacy_graph);

      for (auto &[node_id, centrality] : katz_centralities) {
        // As IN_MEMORY_ANALYTICAL doesn’t offer ACID guarantees, check if the graph elements in the result exist
        try {
          // If so, throw an exception:
          const auto node = graph.GetNodeById(mgp::Id::FromUint(node_id));

          // Otherwise:
          auto record = record_factory.NewRecord();
          record.Insert(kFieldNode, node);
          record.Insert(kFieldRank, centrality);
        } catch (const std::exception &e) {
          continue;
        }
      }

      return;
    }

    const auto created_nodes_ = arguments[0].ValueList();
    const auto created_relationships_ = arguments[1].ValueList();

    const auto deleted_nodes_ = arguments[2].ValueList();
    const auto deleted_relationships_ = arguments[3].ValueList();

    std::vector<std::uint64_t> created_nodes;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> created_relationships;
    std::vector<std::uint64_t> created_relationship_ids;

    std::vector<std::uint64_t> deleted_nodes;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> deleted_relationships;

    for (const auto &node : created_nodes_) {
      created_nodes.push_back(node.ValueNode().Id().AsUint());
    }
    for (const auto &relationship : created_relationships_) {
      created_relationships.push_back(
          {relationship.ValueRelationship().From().Id().AsUint(), relationship.ValueRelationship().To().Id().AsUint()});
    }
    for (const auto &relationship : created_relationships_) {
      created_relationship_ids.push_back(relationship.ValueRelationship().Id().AsUint());
    }

    for (const auto &node : deleted_nodes_) {
      deleted_nodes.push_back(node.ValueNode().Id().AsUint());
    }
    for (const auto &relationship : deleted_relationships_) {
      deleted_relationships.push_back(
          {relationship.ValueRelationship().From().Id().AsUint(), relationship.ValueRelationship().To().Id().AsUint()});
    }

    katz_centralities = katz_alg::UpdateKatz(graph, created_nodes, created_relationships, created_relationship_ids,
                                             deleted_nodes, deleted_relationships);

    for (auto &[node_id, centrality] : katz_centralities) {
      // As IN_MEMORY_ANALYTICAL doesn’t offer ACID guarantees, check if the graph elements in the result exist
      try {
        // If so, throw an exception:
        const auto node = graph.GetNodeById(mgp::Id::FromUint(node_id));

        // Otherwise:
        auto record = record_factory.NewRecord();
        record.Insert(kFieldNode, node);
        record.Insert(kFieldRank, centrality);
      } catch (const std::exception &e) {
        continue;
      }
    }
  } catch (const std::exception &e) {
    mgp::result_set_error_msg(result, e.what());
    return;
  }
}

void KatzCentralityReset(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  try {
    katz_alg::Reset();
    InsertMessageRecord(result, memory, "The algorithm has been successfully reset!");
  } catch (const std::exception &e) {
    InsertMessageRecord(result, memory, "Reset failed: An exception occurred, please check the module!");
  }
}
}  // namespace

extern "C" int mgp_init_module(mgp_module *module, mgp_memory *memory) {
  try {
    // Dynamic Katz centrality
    {
      auto default_alpha = mgp::value_make_double(0.2, memory);
      auto default_epsilon = mgp::value_make_double(1e-2, memory);

      auto *proc = mgp::module_add_read_procedure(module, kProcedureSet, SetKatzCentrality);

      mgp::proc_add_opt_arg(proc, kArgumentAlpha, mgp::type_float(), default_alpha);
      mgp::proc_add_opt_arg(proc, kArgumentEpsilon, mgp::type_float(), default_epsilon);

      mgp::proc_add_result(proc, kFieldNode, mgp::type_node());
      mgp::proc_add_result(proc, kFieldRank, mgp::type_float());

      mgp::value_destroy(default_alpha);
      mgp::value_destroy(default_alpha);
    }

    {
      auto *proc = mgp::module_add_read_procedure(module, kProcedureGet, GetKatzCentrality);

      mgp::proc_add_result(proc, kFieldNode, mgp::type_node());
      mgp::proc_add_result(proc, kFieldRank, mgp::type_float());
    }

    {
      auto *proc = mgp::module_add_read_procedure(module, kProcedureUpdate, UpdateKatzCentrality);

      auto default_created_vertices = mgp::value_make_list(mgp::list_make_empty(0, memory));
      auto default_created_edges = mgp::value_make_list(mgp::list_make_empty(0, memory));
      auto default_deleted_vertices = mgp::value_make_list(mgp::list_make_empty(0, memory));
      auto default_deleted_edges = mgp::value_make_list(mgp::list_make_empty(0, memory));

      mgp::proc_add_opt_arg(proc, kArgumentCreatedVertices, mgp::type_nullable(mgp::type_list(mgp::type_node())),
                            default_created_vertices);
      mgp::proc_add_opt_arg(proc, kArgumentCreatedEdges, mgp::type_nullable(mgp::type_list(mgp::type_relationship())),
                            default_created_edges);
      mgp::proc_add_opt_arg(proc, kArgumentDeletedVertices, mgp::type_nullable(mgp::type_list(mgp::type_node())),
                            default_deleted_vertices);
      mgp::proc_add_opt_arg(proc, kArgumentDeletedEdges, mgp::type_nullable(mgp::type_list(mgp::type_relationship())),
                            default_deleted_edges);

      mgp::value_destroy(default_created_vertices);
      mgp::value_destroy(default_created_edges);
      mgp::value_destroy(default_deleted_vertices);
      mgp::value_destroy(default_deleted_edges);

      mgp::proc_add_result(proc, kFieldNode, mgp::type_node());
      mgp::proc_add_result(proc, kFieldRank, mgp::type_float());
    }

    {
      auto *proc = mgp::module_add_read_procedure(module, kProcedureReset, KatzCentralityReset);
      mgp::proc_add_result(proc, kFieldMessage, mgp::type_string());
    }

  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
