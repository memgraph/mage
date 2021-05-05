#include <mg_exceptions.hpp>
#include <mg_utils.hpp>

#include "algorithm/louvain.hpp"

namespace {

const char *kFieldNode = "node";
const char *kFieldCommunity = "community";

void InsertLouvainRecord(const mgp_graph *graph, mgp_result *result, mgp_memory *memory, const std::uint64_t node_id,
                         const std::uint64_t community) {
  mgp_result_record *record = mgp_result_new_record(result);
  if (record == nullptr) {
    throw mg_exception::NotEnoughMemoryException();
  }

  mg_utility::InsertNodeValueResult(graph, record, kFieldNode, node_id, memory);
  mg_utility::InsertIntValueResult(record, kFieldCommunity, community, memory);
}

void LouvainCommunityDetection(const mgp_list *args, const mgp_graph *memgraph_graph, mgp_result *result,
                               mgp_memory *memory) {
  try {
    auto graph = mg_utility::GetGraphView(memgraph_graph, result, memory);
    auto communities = louvain_alg::GetCommunities(*graph);

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
  struct mgp_proc *community_proc = mgp_module_add_read_procedure(module, "get", LouvainCommunityDetection);
  if (!community_proc) return 1;
  if (!mgp_proc_add_opt_arg(community_proc, "coloring", mgp_type_bool(), mgp_value_make_bool(0, memory))) return 1;
  if (!mgp_proc_add_opt_arg(community_proc, "min_graph_size", mgp_type_int(), mgp_value_make_int(100000, memory)))
    return 1;
  if (!mgp_proc_add_opt_arg(community_proc, "threshold", mgp_type_float(), mgp_value_make_double(0.000001, memory)))
    return 1;
  if (!mgp_proc_add_opt_arg(community_proc, "coloring_threshold", mgp_type_float(),
                            mgp_value_make_double(0.01, memory)))
    return 1;
  if (!mgp_proc_add_result(community_proc, "node", mgp_type_node())) return 1;
  if (!mgp_proc_add_result(community_proc, "community", mgp_type_int())) return 1;

  return 0;
}

// This is an optional function if you need to release any resources before the
// module is unloaded. You will probably need this if you acquired some
// resources in mgp_init_module.
extern "C" int mgp_shutdown_module() {
  // Return 0 to indicate success.
  return 0;
}