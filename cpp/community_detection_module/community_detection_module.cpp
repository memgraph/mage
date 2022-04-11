#include <mg_exceptions.hpp>
#include <mg_utils.hpp>

#include "algorithm/louvain.hpp"

namespace {

const char *kProcedureGet = "get";

const char *kFieldNode = "node";
const char *kFieldCommunity = "community_id";

const char *kArgumentWeightProperty = "weight";
const char *kArgumentColoring = "coloring";
const char *kArgumentMinGraphShrink = "min_graph_shrink";
const char *kArgumentCommunityAlgThreshold = "community_alg_threshold";
const char *kArgumentColoringAlgThreshold = "coloring_alg_threshold";

const char *kDefaultWeightProperty = "weight";
const double kDefaultWeight = 1.0;

void InsertLouvainRecord(mgp_graph *graph, mgp_result *result, mgp_memory *memory, const std::uint64_t node_id,
                         const std::uint64_t community) {
  mgp_result_record *record = mgp::result_new_record(result);
  if (record == nullptr) {
    throw mg_exception::NotEnoughMemoryException();
  }

  mg_utility::InsertNodeValueResult(graph, record, kFieldNode, node_id, memory);
  mg_utility::InsertIntValueResult(record, kFieldCommunity, community, memory);
}

void LouvainCommunityDetection(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  try {
    // Get the query module arguments
    auto weight_property = mgp::value_get_string(mgp::list_at(args, 0));
    auto coloring = mgp::value_get_bool(mgp::list_at(args, 1));
    auto min_graph_shrink = mgp::value_get_int(mgp::list_at(args, 2));
    auto community_alg_threshold = mgp::value_get_double(mgp::list_at(args, 3));
    auto coloring_alg_threshold = mgp::value_get_double(mgp::list_at(args, 4));

    auto graph = mg_utility::GetWeightedGraphView(memgraph_graph, result, memory, mg_graph::GraphType::kUndirectedGraph,
                                                  weight_property, kDefaultWeight);
    auto communities = louvain_alg::GetCommunities(*graph, coloring, min_graph_shrink, community_alg_threshold,
                                                   coloring_alg_threshold);

    for (auto node_id = 0; node_id < graph->Nodes().size(); ++node_id) {
      InsertLouvainRecord(memgraph_graph, result, memory, graph->GetMemgraphNodeId(node_id), communities[node_id]);
    }
  } catch (const std::exception &e) {
    // We must not let any exceptions out of our module.
    mgp::result_set_error_msg(result, e.what());
    return;
  }
}
}  // namespace

// Each module needs to define mgp_init_module function.
// Here you can register multiple procedures your module supports.
extern "C" int mgp_init_module(mgp_module *module, mgp_memory *memory) {
  try {
    auto community_proc = mgp::module_add_read_procedure(module, kProcedureGet, LouvainCommunityDetection);

    auto default_weight_property = mgp::value_make_string(kDefaultWeightProperty, memory);
    auto default_coloring = mgp::value_make_bool(false, memory);
    auto default_min_graph_shrink = mgp::value_make_int(100000, memory);
    auto default_community_alg_threshold = mgp::value_make_double(0.000001, memory);
    auto default_coloring_alg_threshold = mgp::value_make_double(0.01, memory);

    // Query module arguments
    mgp::proc_add_opt_arg(community_proc, kDefaultWeightProperty, mgp::type_string(), default_weight_property);
    mgp::proc_add_opt_arg(community_proc, kArgumentColoring, mgp::type_bool(), default_coloring);
    mgp::proc_add_opt_arg(community_proc, kArgumentMinGraphShrink, mgp::type_int(), default_min_graph_shrink);
    mgp::proc_add_opt_arg(community_proc, kArgumentCommunityAlgThreshold, mgp::type_float(),
                          default_community_alg_threshold);
    mgp::proc_add_opt_arg(community_proc, kArgumentColoringAlgThreshold, mgp::type_float(),
                          default_coloring_alg_threshold);

    // Query module output record
    mgp::proc_add_result(community_proc, kFieldNode, mgp::type_node());
    mgp::proc_add_result(community_proc, kFieldCommunity, mgp::type_int());

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

// This is an optional function if you need to release any resources before the
// module is unloaded. You will probably need this if you acquired some
// resources in mgp_init_module.
extern "C" int mgp_shutdown_module() {
  // Return 0 to indicate success.
  return 0;
}
