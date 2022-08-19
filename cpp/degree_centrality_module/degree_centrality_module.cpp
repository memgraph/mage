#include <mg_exceptions.hpp>
#include <mg_utils.hpp>

#include "algorithm/degree_centrality.hpp"

namespace {

const char *kProcedureGet = "get";
const char *kProcedureGetSubgraph = "get_subgraph";

const char *kFieldNode = "node";
const char *kFieldDegree = "degree";

void InsertDegreeCentralityRecord(mgp_graph *graph, mgp_result *result, mgp_memory *memory, std::uint64_t node,
                                  double degree) {
  auto *record = mgp::result_new_record(result);

  mg_utility::InsertNodeValueResult(graph, record, kFieldNode, node, memory);
  mg_utility::InsertDoubleValueResult(record, kFieldDegree, degree, memory);
}

void GetDegreeCentrality(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  try {
    auto graph = mg_utility::GetGraphView(memgraph_graph, result, memory, mg_graph::GraphType::kUndirectedGraph);
    auto degree_centralities = degree_cenntrality_alg::GetDegreeCentrality(*graph);

    for (const auto [node_id] : graph->Nodes()) {
      auto centrality = degree_centralities[node_id];
      InsertDegreeCentralityRecord(memgraph_graph, result, memory, graph->GetMemgraphNodeId(node_id), centrality);
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
    auto *proc = mgp::module_add_read_procedure(module, kProcedureGet, GetDegreeCentrality);
    mgp::proc_add_result(proc, kFieldNode, mgp::type_node());
    mgp::proc_add_result(proc, kFieldDegree, mgp::type_float());
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