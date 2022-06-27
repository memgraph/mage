#include <mg_exceptions.hpp>
#include <mg_utils.hpp>

#include "algorithm/louvain.hpp"

namespace {

const char *kProcedureGet = "get";
const char *kProcedureGetSubgraph = "get_subgraph";

const char *kFieldNode = "node";
const char *kFieldCommunity = "community_id";

// const char *kArgumentWeightProperty = "weight";
const char *kArgumentColoring = "coloring";
const char *kArgumentMinGraphShrink = "min_graph_shrink";
const char *kArgumentCommunityAlgThreshold = "community_alg_threshold";
const char *kArgumentColoringAlgThreshold = "coloring_alg_threshold";

const char *kSubgraphNodes = "subgraph_nodes";
const char *kSubgraphRelationships = "subgraph_relationships";

const char *kDefaultWeightProperty = "weight";
const double kDefaultWeight = 1.0;

void InsertLouvainRecord(mgp_graph *graph, mgp_result *result,
                         mgp_memory *memory, const std::uint64_t node_id,
                         const std::uint64_t community) {
  mgp_result_record *record = mgp::result_new_record(result);
  if (record == nullptr) throw mg_exception::NotEnoughMemoryException();

  mg_utility::InsertNodeValueResult(graph, record, kFieldNode, node_id, memory);
  mg_utility::InsertIntValueResult(record, kFieldCommunity, community, memory);
}

void LouvainCommunityDetection(mgp_list *args, mgp_graph *memgraph_graph,
                               mgp_result *result, mgp_memory *memory) {
  try {
    auto weight_property = mgp::value_get_string(mgp::list_at(args, 0));
    auto coloring = mgp::value_get_bool(mgp::list_at(args, 1));
    auto min_graph_shrink = mgp::value_get_int(mgp::list_at(args, 2));
    auto community_alg_threshold = mgp::value_get_double(mgp::list_at(args, 3));
    auto coloring_alg_threshold = mgp::value_get_double(mgp::list_at(args, 4));

    auto graph = mg_utility::GetWeightedGraphView(
        memgraph_graph, result, memory, mg_graph::GraphType::kUndirectedGraph,
        weight_property, kDefaultWeight);
    auto communities = louvain_alg::GetCommunities(
        *graph, coloring, min_graph_shrink, community_alg_threshold,
        coloring_alg_threshold);

    for (std::uint64_t node_id = 0; node_id < graph->Nodes().size();
         ++node_id) {
      InsertLouvainRecord(memgraph_graph, result, memory,
                          graph->GetMemgraphNodeId(node_id),
                          communities[node_id]);
    }
  } catch (const std::exception &e) {
    mgp::result_set_error_msg(result, e.what());
    return;
  }
}

void LouvainCommunityDetectionSubgraph(mgp_list *args,
                                       mgp_graph *memgraph_graph,
                                       mgp_result *result, mgp_memory *memory) {
  try {
    auto subgraph_nodes = mgp::value_get_list(mgp::list_at(args, 0));
    auto subgraph_relationships = mgp::value_get_list(mgp::list_at(args, 1));
    auto weight_property = mgp::value_get_string(mgp::list_at(args, 2));
    auto coloring = mgp::value_get_bool(mgp::list_at(args, 3));
    auto min_graph_shrink = mgp::value_get_int(mgp::list_at(args, 4));
    auto community_alg_threshold = mgp::value_get_double(mgp::list_at(args, 5));
    auto coloring_alg_threshold = mgp::value_get_double(mgp::list_at(args, 6));

    auto graph = mg_utility::GetWeightedSubgraphView(
        memgraph_graph, result, memory, subgraph_nodes, subgraph_relationships,
        mg_graph::GraphType::kUndirectedGraph, weight_property, kDefaultWeight);

    auto communities = louvain_alg::GetCommunities(
        *graph, coloring, min_graph_shrink, community_alg_threshold,
        coloring_alg_threshold);

    for (std::uint64_t node_id = 0; node_id < graph->Nodes().size();
         ++node_id) {
      InsertLouvainRecord(memgraph_graph, result, memory,
                          graph->GetMemgraphNodeId(node_id),
                          communities[node_id]);
    }
  } catch (const std::exception &e) {
    // We must not let any exceptions out of our module.
    mgp::result_set_error_msg(result, e.what());
    return;
  }
}
}  // namespace

extern "C" int mgp_init_module(mgp_module *module, mgp_memory *memory) {
  {
    try {
      auto get_proc = mgp::module_add_read_procedure(module, kProcedureGet,
                                                     LouvainCommunityDetection);

      auto default_weight_property =
          mgp::value_make_string(kDefaultWeightProperty, memory);
      auto default_coloring = mgp::value_make_bool(false, memory);
      auto default_min_graph_shrink = mgp::value_make_int(100000, memory);
      auto default_community_alg_threshold =
          mgp::value_make_double(0.000001, memory);
      auto default_coloring_alg_threshold =
          mgp::value_make_double(0.01, memory);

      mgp::proc_add_opt_arg(get_proc, kDefaultWeightProperty,
                            mgp::type_string(), default_weight_property);
      mgp::proc_add_opt_arg(get_proc, kArgumentColoring, mgp::type_bool(),
                            default_coloring);
      mgp::proc_add_opt_arg(get_proc, kArgumentMinGraphShrink, mgp::type_int(),
                            default_min_graph_shrink);
      mgp::proc_add_opt_arg(get_proc, kArgumentCommunityAlgThreshold,
                            mgp::type_float(), default_community_alg_threshold);
      mgp::proc_add_opt_arg(get_proc, kArgumentColoringAlgThreshold,
                            mgp::type_float(), default_coloring_alg_threshold);

      mgp::proc_add_result(get_proc, kFieldNode, mgp::type_node());
      mgp::proc_add_result(get_proc, kFieldCommunity, mgp::type_int());

      mgp::value_destroy(default_weight_property);
      mgp::value_destroy(default_coloring);
      mgp::value_destroy(default_min_graph_shrink);
      mgp::value_destroy(default_community_alg_threshold);
      mgp::value_destroy(default_coloring_alg_threshold);
    } catch (const std::exception &e) {
      return 1;
    }
  }
  {
    try {
      auto get_subgraph_proc = mgp::module_add_read_procedure(
          module, kProcedureGetSubgraph, LouvainCommunityDetectionSubgraph);

      auto default_weight_property =
          mgp::value_make_string(kDefaultWeightProperty, memory);
      auto default_coloring = mgp::value_make_bool(false, memory);
      auto default_min_graph_shrink = mgp::value_make_int(100000, memory);
      auto default_community_alg_threshold =
          mgp::value_make_double(0.000001, memory);
      auto default_coloring_alg_threshold =
          mgp::value_make_double(0.01, memory);

      mgp::proc_add_arg(get_subgraph_proc, kSubgraphNodes,
                        mgp::type_list(mgp::type_node()));
      mgp::proc_add_arg(get_subgraph_proc, kSubgraphRelationships,
                        mgp::type_list(mgp::type_relationship()));
      mgp::proc_add_opt_arg(get_subgraph_proc, kDefaultWeightProperty,
                            mgp::type_string(), default_weight_property);
      mgp::proc_add_opt_arg(get_subgraph_proc, kArgumentColoring,
                            mgp::type_bool(), default_coloring);
      mgp::proc_add_opt_arg(get_subgraph_proc, kArgumentMinGraphShrink,
                            mgp::type_int(), default_min_graph_shrink);
      mgp::proc_add_opt_arg(get_subgraph_proc, kArgumentCommunityAlgThreshold,
                            mgp::type_float(), default_community_alg_threshold);
      mgp::proc_add_opt_arg(get_subgraph_proc, kArgumentColoringAlgThreshold,
                            mgp::type_float(), default_coloring_alg_threshold);

      mgp::proc_add_result(get_subgraph_proc, kFieldNode, mgp::type_node());
      mgp::proc_add_result(get_subgraph_proc, kFieldCommunity, mgp::type_int());

      mgp::value_destroy(default_weight_property);
      mgp::value_destroy(default_coloring);
      mgp::value_destroy(default_min_graph_shrink);
      mgp::value_destroy(default_community_alg_threshold);
      mgp::value_destroy(default_coloring_alg_threshold);
    } catch (const std::exception &e) {
      return 1;
    }
  }
  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
