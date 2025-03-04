#include <omp.h>
#include <cstdint>
#include <mg_exceptions.hpp>
#include <mg_utils.hpp>

#include "_mgp.hpp"
#include "algorithm/louvain.hpp"

namespace {

const char *kProcedureGet = "get";
const char *kProcedureGetSubgraph = "get_subgraph";

const char *kArgumentSubgraphNodes = "subgraph_nodes";
const char *kArgumentSubgraphRelationships = "subgraph_relationships";
const char *kArgumentWeightProperty = "weight";
const char *kArgumentColoring = "coloring";
const char *kArgumentMinGraphShrink = "min_graph_shrink";
const char *kArgumentCommunityAlgThreshold = "community_alg_threshold";
const char *kArgumentColoringAlgThreshold = "coloring_alg_threshold";
const char *kNumberOfThreads = "num_threads";

const char *kFieldNode = "node";
const char *kFieldCommunity = "community_id";

const char *kDefaultWeightProperty = "weight";
const double kDefaultWeight = 1.0;

void InsertLouvainRecords(mgp_graph *graph, mgp_result *result, mgp_memory *memory, const std::vector<std::int64_t> &communities, const louvain_alg::LouvainGraph &louvain_graph) {
  auto *vertices_it = mgp::graph_iter_vertices(graph, memory);  // Safe vertex iterator creation
  mg_utility::OnScopeExit delete_vertices_it([&vertices_it] { mgp::vertices_iterator_destroy(vertices_it); });
  for (auto *vertex = mgp::vertices_iterator_get(vertices_it); vertex;
        vertex = mgp::vertices_iterator_next(vertices_it)) {
    auto memgraph_vertex_id = mgp::vertex_get_id(vertex).as_int;
    auto vertex_id_it = louvain_graph.memgraph_id_to_id.find(memgraph_vertex_id);
    if (vertex_id_it == louvain_graph.memgraph_id_to_id.end()) {
      continue;
    }
    auto community = communities[vertex_id_it->second];
    mgp_result_record *record = mgp::result_new_record(result);
    if (record == nullptr) throw mg_exception::NotEnoughMemoryException();
    mg_utility::InsertNodeValueResult(record, kFieldNode, vertex, memory);
    mg_utility::InsertIntValueResult(record, kFieldCommunity, community, memory);
  }
}

void InsertLouvainRecordsForSubgraph(mgp_list *subgraph_nodes, mgp_result *result, mgp_memory *memory,
                                     const std::vector<std::int64_t> &communities, const louvain_alg::LouvainGraph &louvain_graph) {
  std::unordered_set<std::int64_t> inserted_vertices;
  for (std::size_t i = 0; i < mgp::list_size(subgraph_nodes); i++) {
    auto *vertex = mgp::value_get_vertex(mgp::list_at(subgraph_nodes, i));
    auto memgraph_vertex_id = mgp::vertex_get_id(vertex).as_int;
    auto vertex_id_it = louvain_graph.memgraph_id_to_id.find(memgraph_vertex_id);
    if (vertex_id_it == louvain_graph.memgraph_id_to_id.end()) {
      continue;
    }
    auto community = communities[vertex_id_it->second];
    auto inserted = inserted_vertices.insert(memgraph_vertex_id).second;
    if (inserted) {
      mgp_result_record *record = mgp::result_new_record(result);
      if (record == nullptr) throw mg_exception::NotEnoughMemoryException();
      mg_utility::InsertNodeValueResult(record, kFieldNode, vertex, memory);
      mg_utility::InsertIntValueResult(record, kFieldCommunity, community, memory);
    }
  }
}

void LouvainCommunityDetection(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory,
                               bool subgraph) {
  int i = 0;
  mgp_list *subgraph_nodes = nullptr;
  mgp_list *subgraph_relationships = nullptr;
  if (subgraph) {
    subgraph_nodes = mgp::value_get_list(mgp::list_at(args, i++));
    subgraph_relationships = mgp::value_get_list(mgp::list_at(args, i++));
  }
  const auto *weight_property = mgp::value_get_string(mgp::list_at(args, i++));
  auto coloring = mgp::value_get_bool(mgp::list_at(args, i++));
  auto min_graph_shrink = mgp::value_get_int(mgp::list_at(args, i++));
  auto community_alg_threshold = mgp::value_get_double(mgp::list_at(args, i++));
  auto coloring_alg_threshold = mgp::value_get_double(mgp::list_at(args, i++));
  auto num_threads = mgp::value_get_int(mgp::list_at(args, i++));
  num_threads = num_threads > omp_get_max_threads() ? omp_get_max_threads() : num_threads;
  auto louvain_graph = subgraph
                        ? louvain_alg::GetSubgraphEdgeList(memory, subgraph_nodes, subgraph_relationships,
                                                          weight_property, kDefaultWeight)
                        : louvain_alg::GetGraphEdgeList(memgraph_graph, memory, weight_property, kDefaultWeight);
  if (louvain_graph.edges.empty()) {
    return;
  }
  auto *grappolo_graph = (louvain_alg::GrappoloGraph *)malloc(sizeof(louvain_alg::GrappoloGraph));
  louvain_alg::GetGrappoloSuitableGraph(*grappolo_graph, num_threads, louvain_graph);
  const auto communities = louvain_alg::GrappoloCommunityDetection(*grappolo_graph, memgraph_graph, coloring, min_graph_shrink, community_alg_threshold, coloring_alg_threshold, num_threads);
  subgraph ? InsertLouvainRecordsForSubgraph(subgraph_nodes, result, memory, communities, louvain_graph)
           : InsertLouvainRecords(memgraph_graph, result, memory, communities, louvain_graph);
}

void OnGraph(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  try {
    LouvainCommunityDetection(args, memgraph_graph, result, memory, false);
  } catch (const std::exception &e) {
    mgp::result_set_error_msg(result, e.what());
    return;
  }
}

void OnSubgraph(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  try {
    LouvainCommunityDetection(args, memgraph_graph, result, memory, true);
  } catch (const std::exception &e) {
    mgp::result_set_error_msg(result, e.what());
    return;
  }
}
}  // namespace

extern "C" int mgp_init_module(mgp_module *module, mgp_memory *memory) {
  try {
    auto *default_weight_property = mgp::value_make_string(kDefaultWeightProperty, memory);
    auto *default_coloring = mgp::value_make_bool(false, memory);
    auto *default_min_graph_shrink = mgp::value_make_int(100000, memory);
    auto *default_community_alg_threshold = mgp::value_make_double(1e-6, memory);
    auto *default_coloring_alg_threshold = mgp::value_make_double(0.01, memory);
    auto *default_num_threads = mgp::value_make_int(omp_get_max_threads() / 2, memory);

    {
      auto *proc = mgp::module_add_read_procedure(module, kProcedureGet, OnGraph);

      mgp::proc_add_opt_arg(proc, kArgumentWeightProperty, mgp::type_string(), default_weight_property);
      mgp::proc_add_opt_arg(proc, kArgumentColoring, mgp::type_bool(), default_coloring);
      mgp::proc_add_opt_arg(proc, kArgumentMinGraphShrink, mgp::type_int(), default_min_graph_shrink);
      mgp::proc_add_opt_arg(proc, kArgumentCommunityAlgThreshold, mgp::type_float(), default_community_alg_threshold);
      mgp::proc_add_opt_arg(proc, kArgumentColoringAlgThreshold, mgp::type_float(), default_coloring_alg_threshold);
      mgp::proc_add_opt_arg(proc, kNumberOfThreads, mgp::type_int(), default_num_threads);

      mgp::proc_add_result(proc, kFieldNode, mgp::type_node());
      mgp::proc_add_result(proc, kFieldCommunity, mgp::type_int());
    }

    {
      auto *proc = mgp::module_add_read_procedure(module, kProcedureGetSubgraph, OnSubgraph);

      mgp::proc_add_arg(proc, kArgumentSubgraphNodes, mgp::type_list(mgp::type_node()));
      mgp::proc_add_arg(proc, kArgumentSubgraphRelationships, mgp::type_list(mgp::type_relationship()));
      mgp::proc_add_opt_arg(proc, kArgumentWeightProperty, mgp::type_string(), default_weight_property);
      mgp::proc_add_opt_arg(proc, kArgumentColoring, mgp::type_bool(), default_coloring);
      mgp::proc_add_opt_arg(proc, kArgumentMinGraphShrink, mgp::type_int(), default_min_graph_shrink);
      mgp::proc_add_opt_arg(proc, kArgumentCommunityAlgThreshold, mgp::type_float(), default_community_alg_threshold);
      mgp::proc_add_opt_arg(proc, kArgumentColoringAlgThreshold, mgp::type_float(), default_coloring_alg_threshold);
      mgp::proc_add_opt_arg(proc, kNumberOfThreads, mgp::type_int(), default_num_threads);

      mgp::proc_add_result(proc, kFieldNode, mgp::type_node());
      mgp::proc_add_result(proc, kFieldCommunity, mgp::type_int());
    }

    mgp::value_destroy(default_weight_property);
    mgp::value_destroy(default_coloring);
    mgp::value_destroy(default_min_graph_shrink);
    mgp::value_destroy(default_community_alg_threshold);
    mgp::value_destroy(default_coloring_alg_threshold);
  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
