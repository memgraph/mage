#include <mg_exceptions.hpp>
#include <mg_utils.hpp>

#include "algorithm/degree_centrality.hpp"

namespace {

const char *kProcedureGet = "get";
const char *kProcedureGetSubgraph = "get_subgraph";

const char *kArgumentSubgraphNodes = "subgraph_nodes";
const char *kArgumentSubgraphRelationships = "subgraph_relationships";
const char *kArgumentType = "type";

const char *kFieldNode = "node";
const char *kFieldDegree = "degree";

const char *kAlgorithmUndirected = "undirected";
const char *kAlgorithmOut = "out";
const char *kAlgorithmIn = "in";

void InsertDegreeCentralityRecord(mgp_graph *graph, mgp_result *result, mgp_memory *memory, std::uint64_t node,
                                  double degree) {
  auto *record = mgp::result_new_record(result);

  mg_utility::InsertNodeValueResult(graph, record, kFieldNode, node, memory);
  mg_utility::InsertDoubleValueResult(record, kFieldDegree, degree, memory);
}

degree_cenntrality_alg::AlgorithmType ParseType(mgp_value *algorithm_type) {
  if (mgp::value_is_null(algorithm_type)) return degree_cenntrality_alg::AlgorithmType::kUndirected;

  auto algorithm_type_str = std::string(mgp::value_get_string(algorithm_type));
  std::transform(algorithm_type_str.begin(), algorithm_type_str.end(), algorithm_type_str.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  if (algorithm_type_str == kAlgorithmUndirected) return degree_cenntrality_alg::AlgorithmType::kUndirected;
  if (algorithm_type_str == kAlgorithmOut) return degree_cenntrality_alg::AlgorithmType::kOut;
  if (algorithm_type_str == kAlgorithmIn) return degree_cenntrality_alg::AlgorithmType::kIn;

  throw std::runtime_error("Unsupported algorithm type. Pick between out/in or undirected");
}

void GetDegreeCentralityWrapper(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory,
                                bool subgraph) {
  try {
    int i = 0;
    mgp_list *subgraph_nodes, *subgraph_relationships;
    if (subgraph) {
      subgraph_nodes = mgp::value_get_list(mgp::list_at(args, i++));
      subgraph_relationships = mgp::value_get_list(mgp::list_at(args, i++));
    }
    auto algorithm_type_value = mgp::list_at(args, i++);

    auto graph = subgraph
                     ? mg_utility::GetSubgraphView(memgraph_graph, result, memory, subgraph_nodes,
                                                   subgraph_relationships, mg_graph::GraphType::kDirectedGraph)
                     : mg_utility::GetGraphView(memgraph_graph, result, memory, mg_graph::GraphType::kDirectedGraph);

    auto algorithm_type = ParseType(algorithm_type_value);
    auto degree_centralities = degree_cenntrality_alg::GetDegreeCentrality(*graph, algorithm_type);

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

void GetDegreeCentrality(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  GetDegreeCentralityWrapper(args, memgraph_graph, result, memory, false);
}

void GetSubgraphDegreeCentrality(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  GetDegreeCentralityWrapper(args, memgraph_graph, result, memory, true);
}

}  // namespace

// Each module needs to define mgp_init_module function.
// Here you can register multiple procedures your module supports.
extern "C" int mgp_init_module(mgp_module *module, mgp_memory *memory) {
  try {
    auto default_type = mgp::value_make_null(memory);
    {
      auto *proc = mgp::module_add_read_procedure(module, kProcedureGet, GetDegreeCentrality);

      mgp::proc_add_opt_arg(proc, kArgumentType, mgp::type_nullable(mgp::type_string()), default_type);
      mgp::proc_add_result(proc, kFieldNode, mgp::type_node());
      mgp::proc_add_result(proc, kFieldDegree, mgp::type_float());
    }

    {
      auto *proc = mgp::module_add_read_procedure(module, kProcedureGetSubgraph, GetSubgraphDegreeCentrality);

      mgp::proc_add_arg(proc, kArgumentSubgraphNodes, mgp::type_list(mgp::type_node()));
      mgp::proc_add_arg(proc, kArgumentSubgraphRelationships, mgp::type_list(mgp::type_relationship()));
      mgp::proc_add_opt_arg(proc, kArgumentType, mgp::type_nullable(mgp::type_string()), default_type);
      mgp::proc_add_result(proc, kFieldNode, mgp::type_node());
      mgp::proc_add_result(proc, kFieldDegree, mgp::type_float());
    }

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
