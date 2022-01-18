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

constexpr char const *kArgumentCreatedVertices = "created_vertices";
constexpr char const *kArgumentCreatedEdges = "created_edges";
constexpr char const *kArgumentDeletedVertices = "deleted_vertices";
constexpr char const *kArgumentDeletedEdges = "deleted_edges";

void InsertKatzRecord(mgp_graph *graph, mgp_result *result, mgp_memory *memory, const double katz_centrality,
                      const int node_id) {
  auto *record = mgp::result_new_record(result);

  mg_utility::InsertNodeValueResult(graph, record, kFieldNode, node_id, memory);
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
      // Insert the Katz centrality record
      InsertKatzRecord(memgraph_graph, result, memory, centrality, vertex_id);
    }
  } catch (const std::exception &e) {
    // We must not let any exceptions out of our module.
    mgp::result_set_error_msg(result, e.what());
    return;
  }
}

void UpdateKatzCentrality(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  try {
    // Created vertices
    auto created_vertices = mg_utility::GetVerticesFromList(mg_utility::GetListFromArgument(args, 0));
    auto created_edges = mg_utility::GetEdgesFromList(mg_utility::GetListFromArgument(args, 1));
    auto created_edge_ids = mg_utility::GetEdgeIDsFromList(mg_utility::GetListFromArgument(args, 1));

    // Deleted entities
    auto deleted_vertices = mg_utility::GetVerticesFromList(mg_utility::GetListFromArgument(args, 2));
    auto deleted_edges = mg_utility::GetEdgesFromList(mg_utility::GetListFromArgument(args, 3));

    auto graph = mg_utility::GetGraphView(memgraph_graph, result, memory, mg_graph::GraphType::kDirectedGraph);
    std::transform(created_edge_ids.begin(), created_edge_ids.end(), created_edge_ids.begin(),
                   [&graph](std::uint64_t id) -> std::uint64_t { return graph.get()->GetInnerEdgeId(id); });

    auto katz_centralities = katz_alg::UpdateKatz(*graph, created_vertices, created_edges, created_edge_ids,
                                                  deleted_vertices, deleted_edges);

    for (auto &[vertex_id, centrality] : katz_centralities) {
      // Insert the Katz centrality record
      InsertKatzRecord(memgraph_graph, result, memory, centrality, vertex_id);
    }
  } catch (const std::exception &e) {
    // We must not let any exceptions out of our module.
    mgp::result_set_error_msg(result, e.what());
    return;
  }
}
void KatzCentralityReset(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  try {
    katz_alg::Reset();
    InsertMessageRecord(result, memory,
                        "Katz centrality context is reset! Before running again it will run initialization.");
  } catch (const std::exception &e) {
    // We must not let any exceptions out of our module.
    InsertMessageRecord(result, memory,
                        "Reset failed: An exception occurred, please check your `katz_centrality_online` module!");
  }
}
}  // namespace

// Each module needs to define mgp_init_module function.
// Here you can register multiple procedures your module supports.
extern "C" int mgp_init_module(mgp_module *module, mgp_memory *memory) {
  try {
    // Static Katz centrality
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

// This is an optional function if you need to release any resources before the
// module is unloaded. You will probably need this if you acquired some
// resources in mgp_init_module.
extern "C" int mgp_shutdown_module() {
  // Return 0 to indicate success.
  return 0;
}
