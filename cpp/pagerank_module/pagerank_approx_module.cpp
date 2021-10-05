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

void ApproxPagerankGet(const mgp_list *args, const mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  try {
    auto walks_per_node = mgp_value_get_int(mgp_list_at(args, 0));
    auto walk_stop_epsilon = mgp_value_get_double(mgp_list_at(args, 1));

    auto graph = mg_utility::GetGraphView(memgraph_graph, result, memory, mg_graph::GraphType::kDirectedGraph);

    auto pageranks = pagerank_approx_alg::SetPagerank(*graph, walks_per_node, walk_stop_epsilon);

    for (auto const [node_id, rank] : pageranks) {
      InsertPagerankRecord(memgraph_graph, result, memory, node_id, rank);
    }
  } catch (const std::exception &e) {
    // We must not let any exceptions out of our module.
    mgp_result_set_error_msg(result, e.what());
    return;
  }
}

void ApproxPagerankUpdate(const mgp_list *args, const mgp_graph *memgraph_graph, mgp_result *result,
                          mgp_memory *memory) {
  try {
    // Created vertices
    auto created_vertices_list = mgp_value_get_list(mgp_list_at(args, 0));
    auto size = mgp_list_size(created_vertices_list);
    auto created_vertices = std::vector<std::uint64_t>(size);
    for (std::size_t i = 0; i < size; i++) {
      created_vertices[i] = mgp_vertex_get_id(mgp_value_get_vertex(mgp_list_at(created_vertices_list, i))).as_int;
    }

    auto created_edges_list = mgp_value_get_list(mgp_list_at(args, 1));
    size = mgp_list_size(created_edges_list);
    auto created_edges = std::vector<std::pair<std::uint64_t, std::uint64_t>>(size);
    for (std::size_t i = 0; i < size; i++) {
      auto edge = mgp_value_get_edge(mgp_list_at(created_edges_list, i));
      auto from = mgp_vertex_get_id(mgp_edge_get_from(edge)).as_int;
      auto to = mgp_vertex_get_id(mgp_edge_get_to(edge)).as_int;
      created_edges[i] = std::make_pair(from, to);
    }

    // Deleted vertices
    auto deleted_vertices_list = mgp_value_get_list(mgp_list_at(args, 2));
    size = mgp_list_size(deleted_vertices_list);
    auto deleted_vertices = std::vector<std::uint64_t>(size);
    for (std::size_t i = 0; i < size; i++) {
      deleted_vertices[i] = mgp_vertex_get_id(mgp_value_get_vertex(mgp_list_at(deleted_vertices_list, i))).as_int;
    }

    auto deleted_edges_list = mgp_value_get_list(mgp_list_at(args, 3));
    size = mgp_list_size(deleted_edges_list);
    auto deleted_edges = std::vector<std::pair<std::uint64_t, std::uint64_t>>(size);
    for (std::size_t i = 0; i < size; i++) {
      auto edge = mgp_value_get_edge(mgp_list_at(deleted_edges_list, i));
      auto from = mgp_vertex_get_id(mgp_edge_get_from(edge)).as_int;
      auto to = mgp_vertex_get_id(mgp_edge_get_to(edge)).as_int;
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
    mgp_result_set_error_msg(result, e.what());
    return;
  }
}

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  // Approximate PageRank solution
  {
    struct mgp_proc *pagerank_proc = mgp_module_add_read_procedure(module, "get", ApproxPagerankGet);

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

  // Approximate PageRank Create Edges/Nodes
  {
    struct mgp_proc *pagerank_proc = mgp_module_add_read_procedure(module, "update", ApproxPagerankUpdate);

    if (!pagerank_proc) return 1;

    auto default_created_vertices = mgp_value_make_null(memory);
    auto default_created_edges = mgp_value_make_null(memory);
    auto default_deleted_vertices = mgp_value_make_null(memory);
    auto default_deleted_edges = mgp_value_make_null(memory);

    if (!mgp_proc_add_opt_arg(pagerank_proc, "created_vertices", mgp_type_nullable(mgp_type_list(mgp_type_node())),
                              default_created_vertices))
      return 1;
    if (!mgp_proc_add_opt_arg(pagerank_proc, "created_edges", mgp_type_nullable(mgp_type_list(mgp_type_relationship())),
                              default_created_edges))
      return 1;
    if (!mgp_proc_add_opt_arg(pagerank_proc, "deleted_vertices", mgp_type_nullable(mgp_type_list(mgp_type_node())),
                              default_deleted_vertices))
      return 1;
    if (!mgp_proc_add_opt_arg(pagerank_proc, "deleted_edges", mgp_type_nullable(mgp_type_list(mgp_type_relationship())),
                              default_deleted_edges))
      return 1;

    mgp_value_destroy(default_created_vertices);
    mgp_value_destroy(default_created_edges);
    mgp_value_destroy(default_deleted_vertices);
    mgp_value_destroy(default_deleted_edges);

    // Query module output record
    if (!mgp_proc_add_result(pagerank_proc, kFieldNode, mgp_type_node())) return 1;
    if (!mgp_proc_add_result(pagerank_proc, kFieldRank, mgp_type_float())) return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
