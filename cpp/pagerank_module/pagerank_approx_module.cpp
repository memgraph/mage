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

void InsertPagerankRecord(mgp_graph *graph, mgp_result *result, mgp_memory *memory, const std::uint64_t node_id,
                          double rank) {
  auto *record = mgp::result_new_record(result);
  if (record == nullptr) {
    throw mg_exception::NotEnoughMemoryException();
  }

  mg_utility::InsertNodeValueResult(graph, record, kFieldNode, node_id, memory);
  mg_utility::InsertDoubleValue(record, kFieldRank, rank, memory);
}

void ApproxPagerankGet(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  try {
    auto walks_per_node = mgp::value_get_int(mgp::list_at(args, 0));
    auto walk_stop_epsilon = mgp::value_get_double(mgp::list_at(args, 1));

    auto graph = mg_utility::GetGraphView(memgraph_graph, result, memory, mg_graph::GraphType::kDirectedGraph);

    auto pageranks = pagerank_approx_alg::SetPagerank(*graph, walks_per_node, walk_stop_epsilon);

    for (auto const [node_id, rank] : pageranks) {
      InsertPagerankRecord(memgraph_graph, result, memory, node_id, rank);
    }
  } catch (const std::exception &e) {
    // We must not let any exceptions out of our module.
    mgp::result_set_error_msg(result, e.what());
    return;
  }
}

void ApproxPagerankUpdate(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  try {
    // Created vertices
    auto created_vertices_list = mgp::value_get_list(mgp::list_at(args, 0));
    auto size = mgp::list_size(created_vertices_list);
    auto created_vertices = std::vector<std::uint64_t>(size);
    for (std::size_t i = 0; i < size; i++) {
      created_vertices[i] = mgp::vertex_get_id(mgp::value_get_vertex(mgp::list_at(created_vertices_list, i))).as_int;
    }

    auto created_edges_list = mgp::value_get_list(mgp::list_at(args, 1));
    size = mgp::list_size(created_edges_list);
    auto created_edges = std::vector<std::pair<std::uint64_t, std::uint64_t>>(size);
    for (std::size_t i = 0; i < size; i++) {
      auto edge = mgp::value_get_edge(mgp::list_at(created_edges_list, i));
      auto from = mgp::vertex_get_id(mgp::edge_get_from(edge)).as_int;
      auto to = mgp::vertex_get_id(mgp::edge_get_to(edge)).as_int;
      created_edges[i] = std::make_pair(from, to);
    }

    // Deleted vertices
    auto deleted_vertices_list = mgp::value_get_list(mgp::list_at(args, 2));
    size = mgp::list_size(deleted_vertices_list);
    auto deleted_vertices = std::vector<std::uint64_t>(size);
    for (std::size_t i = 0; i < size; i++) {
      deleted_vertices[i] = mgp::vertex_get_id(mgp::value_get_vertex(mgp::list_at(deleted_vertices_list, i))).as_int;
    }

    auto deleted_edges_list = mgp::value_get_list(mgp::list_at(args, 3));
    size = mgp::list_size(deleted_edges_list);
    auto deleted_edges = std::vector<std::pair<std::uint64_t, std::uint64_t>>(size);
    for (std::size_t i = 0; i < size; i++) {
      auto edge = mgp::value_get_edge(mgp::list_at(deleted_edges_list, i));
      auto from = mgp::vertex_get_id(mgp::edge_get_from(edge)).as_int;
      auto to = mgp::vertex_get_id(mgp::edge_get_to(edge)).as_int;
      deleted_edges[i] = std::make_pair(from, to);
    }

    auto graph = mg_utility::GetGraphView(memgraph_graph, result, memory, mg_graph::GraphType::kDirectedGraph);

    auto pageranks =
        pagerank_approx_alg::UpdatePagerank(*graph, created_vertices, created_edges, deleted_vertices, deleted_edges);

    auto number_of_nodes = graph->Nodes().size();
    for (auto const [node_id, rank] : pageranks) {
      InsertPagerankRecord(memgraph_graph, result, memory, node_id, rank);
    }
  } catch (const std::exception &e) {
    // We must not let any exceptions out of our module.
    mgp::result_set_error_msg(result, e.what());
    return;
  }
}

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  // Approximate PageRank solution
  {
    try {
      auto *pagerank_proc = mgp::module_add_read_procedure(module, "get", ApproxPagerankGet);

      auto default_walks_per_node = mgp::value_make_int(10, memory);
      auto default_walk_stop_epsilon = mgp::value_make_double(0.1, memory);

      mgp::proc_add_opt_arg(pagerank_proc, kArgumentWalksPerNode, mgp::type_int(), default_walks_per_node);
      mgp::proc_add_opt_arg(pagerank_proc, kArgumentStopEpsilon, mgp::type_float(), default_walk_stop_epsilon);

      mgp::value_destroy(default_walks_per_node);
      mgp::value_destroy(default_walk_stop_epsilon);

      // Query module output record
      mgp::proc_add_result(pagerank_proc, kFieldNode, mgp::type_node());
      mgp::proc_add_result(pagerank_proc, kFieldRank, mgp::type_float());

    } catch (const std::exception &e) {
      return 1;
    }
  }

  // Approximate PageRank Create Edges/Nodes
  {
    try {
      auto *pagerank_proc = mgp::module_add_read_procedure(module, "update", ApproxPagerankUpdate);

      auto default_created_vertices = mgp::value_make_null(memory);
      auto default_created_edges = mgp::value_make_null(memory);
      auto default_deleted_vertices = mgp::value_make_null(memory);
      auto default_deleted_edges = mgp::value_make_null(memory);

      mgp::proc_add_opt_arg(pagerank_proc, "created_vertices", mgp::type_nullable(mgp::type_list(mgp::type_node())),
                            default_created_vertices);
      mgp::proc_add_opt_arg(pagerank_proc, "created_edges",
                            mgp::type_nullable(mgp::type_list(mgp::type_relationship())), default_created_edges);
      mgp::proc_add_opt_arg(pagerank_proc, "deleted_vertices", mgp::type_nullable(mgp::type_list(mgp::type_node())),
                            default_deleted_vertices);
      mgp::proc_add_opt_arg(pagerank_proc, "deleted_edges",
                            mgp::type_nullable(mgp::type_list(mgp::type_relationship())), default_deleted_edges);

      mgp::value_destroy(default_created_vertices);
      mgp::value_destroy(default_created_edges);
      mgp::value_destroy(default_deleted_vertices);
      mgp::value_destroy(default_deleted_edges);

      // Query module output record
      mgp::proc_add_result(pagerank_proc, kFieldNode, mgp::type_node());
      mgp::proc_add_result(pagerank_proc, kFieldRank, mgp::type_float());

    } catch (const std::exception &e) {
      return 1;
    }
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
