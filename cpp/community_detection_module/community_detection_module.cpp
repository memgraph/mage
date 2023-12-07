#include <mg_exceptions.hpp>
#include <mg_utils.hpp>

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

const char *kFieldNode = "node";
const char *kFieldCommunity = "community_id";

const char *kDefaultWeightProperty = "weight";
const double kDefaultWeight = 1.0;

void InsertLouvainRecord(mgp_graph *graph, mgp_result *result, mgp_memory *memory, const std::uint64_t node_id,
                         const std::uint64_t community) {
  mgp_result_record *record = mgp::result_new_record(result);
  if (record == nullptr) throw mg_exception::NotEnoughMemoryException();

  mg_utility::InsertNodeValueResult(graph, record, kFieldNode, node_id, memory);
  mg_utility::InsertIntValueResult(record, kFieldCommunity, community, memory);
}

void LouvainCommunityDetection(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory,
                               bool subgraph) {
  int i = 0;

  mgp_list *subgraph_nodes, *subgraph_relationships;
  if (subgraph) {
    subgraph_nodes = mgp::value_get_list(mgp::list_at(args, i++));
    subgraph_relationships = mgp::value_get_list(mgp::list_at(args, i++));
  }
  auto weight_property = mgp::value_get_string(mgp::list_at(args, i++));
  auto coloring = mgp::value_get_bool(mgp::list_at(args, i++));
  auto min_graph_shrink = mgp::value_get_int(mgp::list_at(args, i++));
  auto community_alg_threshold = mgp::value_get_double(mgp::list_at(args, i++));
  auto coloring_alg_threshold = mgp::value_get_double(mgp::list_at(args, i++));

  auto graph =
      subgraph
          ? mg_utility::GetWeightedSubgraphView(memgraph_graph, result, memory, subgraph_nodes, subgraph_relationships,
                                                mg_graph::GraphType::kUndirectedGraph, weight_property, kDefaultWeight)
          : mg_utility::GetWeightedGraphView(memgraph_graph, result, memory, mg_graph::GraphType::kUndirectedGraph,
                                             weight_property, kDefaultWeight);

  auto communities =
      louvain_alg::GetCommunities(*graph, memgraph_graph, coloring, min_graph_shrink, community_alg_threshold, coloring_alg_threshold);

  for (std::uint64_t node_id = 0; node_id < graph->Nodes().size(); ++node_id) {
    InsertLouvainRecord(memgraph_graph, result, memory, graph->GetMemgraphNodeId(node_id), communities[node_id]);
  }
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
    auto default_weight_property = mgp::value_make_string(kDefaultWeightProperty, memory);
    auto default_coloring = mgp::value_make_bool(false, memory);
    auto default_min_graph_shrink = mgp::value_make_int(100000, memory);
    auto default_community_alg_threshold = mgp::value_make_double(1e-6, memory);
    auto default_coloring_alg_threshold = mgp::value_make_double(0.01, memory);

    {
      auto proc = mgp::module_add_read_procedure(module, kProcedureGet, OnGraph);

      mgp::proc_add_opt_arg(proc, kArgumentWeightProperty, mgp::type_string(), default_weight_property);
      mgp::proc_add_opt_arg(proc, kArgumentColoring, mgp::type_bool(), default_coloring);
      mgp::proc_add_opt_arg(proc, kArgumentMinGraphShrink, mgp::type_int(), default_min_graph_shrink);
      mgp::proc_add_opt_arg(proc, kArgumentCommunityAlgThreshold, mgp::type_float(), default_community_alg_threshold);
      mgp::proc_add_opt_arg(proc, kArgumentColoringAlgThreshold, mgp::type_float(), default_coloring_alg_threshold);

      mgp::proc_add_result(proc, kFieldNode, mgp::type_node());
      mgp::proc_add_result(proc, kFieldCommunity, mgp::type_int());
    }

    {
      auto proc = mgp::module_add_read_procedure(module, kProcedureGetSubgraph, OnSubgraph);

      mgp::proc_add_arg(proc, kArgumentSubgraphNodes, mgp::type_list(mgp::type_node()));
      mgp::proc_add_arg(proc, kArgumentSubgraphRelationships, mgp::type_list(mgp::type_relationship()));
      mgp::proc_add_opt_arg(proc, kArgumentWeightProperty, mgp::type_string(), default_weight_property);
      mgp::proc_add_opt_arg(proc, kArgumentColoring, mgp::type_bool(), default_coloring);
      mgp::proc_add_opt_arg(proc, kArgumentMinGraphShrink, mgp::type_int(), default_min_graph_shrink);
      mgp::proc_add_opt_arg(proc, kArgumentCommunityAlgThreshold, mgp::type_float(), default_community_alg_threshold);
      mgp::proc_add_opt_arg(proc, kArgumentColoringAlgThreshold, mgp::type_float(), default_coloring_alg_threshold);

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
