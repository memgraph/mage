#include <algorithm>
#include <thread>

#include <mg_generate.hpp>
#include <mg_graph.hpp>
#include <mg_utils.hpp>

#include "algorithm_online/betweenness_centrality_online.hpp"

namespace {
constexpr std::string_view kProcedureSet{"set"};
constexpr std::string_view kProcedureGet{"get"};
constexpr std::string_view kProcedureUpdate{"update"};
constexpr std::string_view kProcedureReset{"reset"};

constexpr std::string_view kArgumentCreatedVertices{"created_vertices"};
constexpr std::string_view kArgumentCreatedEdges{"created_edges"};
constexpr std::string_view kArgumentDeletedVertices{"deleted_vertices"};
constexpr std::string_view kArgumentDeletedEdges{"deleted_edges"};
constexpr std::string_view kArgumentNormalize{"normalize"};
constexpr std::string_view kArgumentThreads{"threads"};

constexpr std::string_view kFieldNode{"node"};
constexpr std::string_view kFieldBC{"betweenness_centrality"};
constexpr std::string_view kFieldMessage{"message"};

constexpr bool DEFAULT_DIRECTED = false;

online_bc::OnlineBC algorithm = online_bc::OnlineBC();
bool initialized = false;
}  // namespace

void InsertOnlineBCRecord(mgp_graph *graph, mgp_result *result, mgp_memory *memory, const std::uint64_t node_id,
                          double bc_score) {
  auto *record = mgp::result_new_record(result);

  mg_utility::InsertNodeValueResult(graph, record, kFieldNode.data(), node_id, memory);
  mg_utility::InsertDoubleValueResult(record, kFieldBC.data(), bc_score, memory);
}

void InsertMessageRecord(mgp_result *result, mgp_memory *memory, const char *message) {
  auto *record = mgp::result_new_record(result);

  mg_utility::InsertStringValueResult(record, kFieldMessage.data(), message, memory);
}

void Set(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  try {
    const auto normalize = mgp::value_get_bool(mgp::list_at(args, 0));
    auto threads = mgp::value_get_int(mgp::list_at(args, 1));

    if (threads <= 0) threads = std::thread::hardware_concurrency();

    auto graph = mg_utility::GetGraphView(memgraph_graph, result, memory, mg_graph::GraphType::kUndirectedGraph);
    const auto node_bc_scores = algorithm.Set(*graph, DEFAULT_DIRECTED, normalize, threads);
    ::initialized = true;

    for (const auto [node_id, bc_score] : node_bc_scores) {
      InsertOnlineBCRecord(memgraph_graph, result, memory, node_id, bc_score);
    }
  } catch (const std::exception &e) {
    mgp::result_set_error_msg(result, e.what());
    return;
  }
}

void Get(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  try {
    const auto normalize = mgp::value_get_bool(mgp::list_at(args, 0));

    auto graph = mg_utility::GetGraphView(memgraph_graph, result, memory, mg_graph::GraphType::kUndirectedGraph);
    const auto node_bc_scores = algorithm.Get(*graph, normalize);

    for (const auto [node_id, bc_score] : node_bc_scores) {
      InsertOnlineBCRecord(memgraph_graph, result, memory, node_id, bc_score);
    }
  } catch (const std::exception &e) {
    mgp::result_set_error_msg(result, e.what());
    return;
  }
}

void Update(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  try {
    const auto created_nodes = mgp::value_get_list(mgp::list_at(args, 0));
    const auto created_edges_list = mgp::value_get_list(mgp::list_at(args, 1));
    const auto deleted_nodes = mgp::value_get_list(mgp::list_at(args, 2));
    const auto deleted_edges_list = mgp::value_get_list(mgp::list_at(args, 3));
    const auto normalize = mgp::value_get_bool(mgp::list_at(args, 4));
    const auto threads = mgp::value_get_int(mgp::list_at(args, 5));

    auto graph = mg_utility::GetGraphView(memgraph_graph, result, memory, mg_graph::GraphType::kUndirectedGraph);
    std::unordered_map<uint64_t, double> node_bc_scores;

    if (!::initialized) {
      node_bc_scores = algorithm.Set(*graph, DEFAULT_DIRECTED, normalize, threads);
      ::initialized = true;
    } else {
      const auto created_node_ids = mg_utility::GetNodeIDs(created_nodes);
      const auto created_edges = mg_utility::GetEdgeEndpointIDs(created_edges_list);
      const auto deleted_node_ids = mg_utility::GetNodeIDs(deleted_nodes);
      const auto deleted_edges = mg_utility::GetEdgeEndpointIDs(deleted_edges_list);

      if (created_node_ids.size() == 0 && deleted_node_ids.size() == 0) {  // Use the online algorithm
        // Construct the graph as before the update
        std::vector<std::pair<std::uint64_t, std::uint64_t>> edges;
        edges.reserve(graph->Edges().size());
        for (const auto edge : graph->Edges()) {
          std::pair<uint64_t, uint64_t> edge_memgraph_ids = {graph->GetMemgraphNodeId(edge.from),
                                                             graph->GetMemgraphNodeId(edge.to)};
          // Newly created edges aren’t part of the prior graph
          if (std::find(created_edges.begin(), created_edges.end(), edge_memgraph_ids) == created_edges.end())
            edges.push_back({edge.from, edge.to});
        }
        for (const auto edge : deleted_edges) {
          edges.push_back({graph->GetInnerNodeId(edge.first), graph->GetInnerNodeId(edge.second)});
        }
        auto prior_graph = mg_generate::BuildGraph(graph->Nodes().size(), edges, mg_graph::GraphType::kUndirectedGraph);

        // Dynamically update betweenness centrality scores by each created edge
        for (const auto created_edge : created_edges) {
          edges.push_back({graph->GetInnerNodeId(created_edge.first), graph->GetInnerNodeId(created_edge.second)});
          graph = mg_generate::BuildGraph(graph->Nodes().size(), edges, mg_graph::GraphType::kUndirectedGraph);
          node_bc_scores = algorithm.Update(*prior_graph, *graph, online_bc::Operation::INSERT_EDGE, -1, created_edge,
                                            normalize, threads);
          prior_graph = std::move(graph);
        }
        // Dynamically update betweenness centrality scores by each deleted edge
        for (const auto deleted_edge : deleted_edges) {
          const std::pair<std::uint64_t, std::uint64_t> edge_to_delete{graph->GetInnerNodeId(deleted_edge.first),
                                                                       graph->GetInnerNodeId(deleted_edge.second)};
          edges.erase(std::remove(edges.begin(), edges.end(), edge_to_delete), edges.end());
          graph = mg_generate::BuildGraph(graph->Nodes().size(), edges, mg_graph::GraphType::kUndirectedGraph);
          node_bc_scores = algorithm.Update(*prior_graph, *graph, online_bc::Operation::DELETE_EDGE, -1, deleted_edge,
                                            normalize, threads);
          prior_graph = std::move(graph);
        }
      } else {  // Default to offline update
        node_bc_scores = algorithm.Set(*graph, DEFAULT_DIRECTED, normalize, threads);
      }
    }

    for (const auto [node_id, bc_score] : node_bc_scores) {
      InsertOnlineBCRecord(memgraph_graph, result, memory, node_id, bc_score);
    }
  } catch (const std::exception &e) {
    mgp::result_set_error_msg(result, e.what());
    return;
  }
}

void Reset(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  try {
    ::algorithm = online_bc::OnlineBC();
    ::initialized = false;

    InsertMessageRecord(result, memory, "The algorithm has been successfully reset!");
  } catch (const std::exception &e) {
    InsertMessageRecord(result, memory, "Reset failed: An exception occurred, please check your module!");
    return;
  }
}

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  // Add .set()
  {
    try {
      auto *set_proc = mgp::module_add_read_procedure(module, kProcedureSet.data(), Set);

      auto default_normalize = mgp::value_make_bool(true, memory);
      auto default_threads = mgp::value_make_int(std::thread::hardware_concurrency(), memory);

      mgp::proc_add_opt_arg(set_proc, kArgumentNormalize.data(), mgp::type_bool(), default_normalize);
      mgp::proc_add_opt_arg(set_proc, kArgumentThreads.data(), mgp::type_int(), default_threads);

      mgp::value_destroy(default_normalize);
      mgp::value_destroy(default_threads);

      mgp::proc_add_result(set_proc, kFieldNode.data(), mgp::type_node());
      mgp::proc_add_result(set_proc, kFieldBC.data(), mgp::type_float());
    } catch (const std::exception &e) {
      return 1;
    }
  }

  // Add .get()
  {
    try {
      auto *get_proc = mgp::module_add_read_procedure(module, kProcedureGet.data(), Get);

      auto default_normalize = mgp::value_make_bool(true, memory);
      mgp::proc_add_opt_arg(get_proc, kArgumentNormalize.data(), mgp::type_bool(), default_normalize);

      mgp::proc_add_result(get_proc, kFieldNode.data(), mgp::type_node());
      mgp::proc_add_result(get_proc, kFieldBC.data(), mgp::type_float());
    } catch (const std::exception &e) {
      return 1;
    }
  }

  // Add .update()
  {
    try {
      auto *update_proc = mgp::module_add_read_procedure(module, kProcedureUpdate.data(), Update);

      auto default_vertices = mgp::value_make_list(mgp::list_make_empty(0, memory));
      auto default_edges = mgp::value_make_list(mgp::list_make_empty(0, memory));
      auto default_normalize = mgp::value_make_bool(true, memory);
      auto default_threads = mgp::value_make_int(std::thread::hardware_concurrency(), memory);

      mgp::proc_add_opt_arg(update_proc, kArgumentCreatedVertices.data(), mgp::type_list(mgp::type_node()),
                            default_vertices);
      mgp::proc_add_opt_arg(update_proc, kArgumentCreatedEdges.data(), mgp::type_list(mgp::type_relationship()),
                            default_edges);
      mgp::proc_add_opt_arg(update_proc, kArgumentDeletedVertices.data(), mgp::type_list(mgp::type_node()),
                            default_vertices);
      mgp::proc_add_opt_arg(update_proc, kArgumentDeletedEdges.data(), mgp::type_list(mgp::type_relationship()),
                            default_edges);
      mgp::proc_add_opt_arg(update_proc, kArgumentNormalize.data(), mgp::type_bool(), default_normalize);
      mgp::proc_add_opt_arg(update_proc, kArgumentThreads.data(), mgp::type_int(), default_threads);

      mgp::value_destroy(default_vertices);
      mgp::value_destroy(default_edges);
      mgp::value_destroy(default_normalize);
      mgp::value_destroy(default_threads);

      mgp::proc_add_result(update_proc, kFieldNode.data(), mgp::type_node());
      mgp::proc_add_result(update_proc, kFieldBC.data(), mgp::type_float());
    } catch (const std::exception &e) {
      return 1;
    }
  }

  // Add .reset()
  {
    try {
      auto *reset_proc = mgp::module_add_read_procedure(module, kProcedureReset.data(), Reset);
      mgp::proc_add_result(reset_proc, kFieldMessage.data(), mgp::type_string());
    } catch (const std::exception &e) {
      return 1;
    }
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }

// int main(void) { return 0; }
