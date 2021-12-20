#include <mg_exceptions.hpp>
#include <mg_utils.hpp>

#include "algorithm/louvain.hpp"

namespace {

const char *kFieldNode = "node";
const char *kFieldCommunity = "community_id";

const char *kArgumentColoring = "coloring";
const char *kArgumentMinGraphShrink = "min_graph_shrink";
const char *kArgumentCommunityAlgThreshold = "community_alg_threshold";
const char *kArgumentColoringAlgThreshold = "coloring_alg_threshold";

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
    auto coloring = mgp::value_get_bool(mgp::list_at(args, 0));
    auto min_graph_shrink = mgp::value_get_int(mgp::list_at(args, 1));
    auto community_alg_threshold = mgp::value_get_double(mgp::list_at(args, 2));
    auto coloring_alg_threshold = mgp::value_get_double(mgp::list_at(args, 3));

    auto graph = mg_utility::GetGraphView(memgraph_graph, result, memory, mg_graph::GraphType::kUndirectedGraph);
    auto communities = louvain_alg::GetCommunities(*graph, coloring, min_graph_shrink, community_alg_threshold,
                                                   coloring_alg_threshold);

    for (std::uint64_t node_id = 0; node_id < graph->Nodes().size(); ++node_id) {
      InsertLouvainRecord(memgraph_graph, result, memory, graph->GetMemgraphNodeId(node_id), communities[node_id]);
    }
  } catch (const std::exception &e) {
    // We must not let any exceptions out of our module.
    mgp_result_set_error_msg(result, e.what());
    return;
  }
}
}  // namespace

// Each module needs to define mgp_init_module function.
// Here you can register multiple procedures your module supports.
extern "C" int mgp_init_module(mgp_module *module, mgp_memory *memory) {
  try {
    auto community_proc = mgp::module_add_read_procedure(module, "get", LouvainCommunityDetection);

    // Query module arguments
    mgp::proc_add_opt_arg(community_proc, kArgumentColoring, mgp::type_bool(), mgp::value_make_bool(false, memory));
    mgp::proc_add_opt_arg(community_proc, kArgumentMinGraphShrink, mgp::type_int(),
                          mgp::value_make_int(100000, memory));
    mgp::proc_add_opt_arg(community_proc, kArgumentCommunityAlgThreshold, mgp::type_float(),
                          mgp::value_make_double(0.000001, memory));
    mgp::proc_add_opt_arg(community_proc, kArgumentColoringAlgThreshold, mgp::type_float(),
                          mgp::value_make_double(0.01, memory));

    // Query module output record
    mgp::proc_add_result(community_proc, kFieldNode, mgp::type_node());
    mgp::proc_add_result(community_proc, kFieldCommunity, mgp::type_int());
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
