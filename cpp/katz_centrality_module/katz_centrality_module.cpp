#include <mg_utils.hpp>

#include "algorithm/katz.hpp"

namespace {

constexpr char const *kProcedureGet = "get";

constexpr char const *kArgumentAlpha = "alpha";
constexpr char const *kArgumentEpsilon = "epsilon";

constexpr char const *kFieldNode = "node";
constexpr char const *kFieldRank = "rank";

void InsertKatzRecord(mgp_graph *graph, mgp_result *result, mgp_memory *memory, const double katz_centrality,
                      const int node_id) {
  auto *record = mgp::result_new_record(result);

  mg_utility::InsertNodeValueResult(graph, record, kFieldNode, node_id, memory);
  mg_utility::InsertDoubleValueResult(record, kFieldRank, katz_centrality, memory);
}

void GetKatzCentrality(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  try {
    auto alpha = mgp::value_get_double(mgp::list_at(args, 0));
    auto epsilon = mgp::value_get_double(mgp::list_at(args, 1));

    auto graph = mg_utility::GetGraphView(memgraph_graph, result, memory, mg_graph::GraphType::kDirectedGraph);
    auto katz_centralities = katz_alg::SetKatz(*graph);

    for (auto &[vertex_id, centrality] : katz_centralities) {
      // Insert the Katz centrality record
      InsertKatzRecord(memgraph_graph, result, memory, centrality, vertex_id);
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
    // Static Katz centrality
    {
      auto default_alpha = mgp::value_make_double(0.2, memory);
      auto default_epsilon = mgp::value_make_double(1e-2, memory);
      
      auto *proc = mgp::module_add_read_procedure(module, kProcedureGet, GetKatzCentrality);

      mgp::proc_add_opt_arg(proc, kArgumentAlpha, mgp::type_float(), default_alpha);
      mgp::proc_add_opt_arg(proc, kArgumentEpsilon, mgp::type_float(), default_epsilon);

      mgp::proc_add_result(proc, kFieldNode, mgp::type_node());
      mgp::proc_add_result(proc, kFieldRank, mgp::type_float());

      mgp::value_destroy(default_alpha);
      mgp::value_destroy(default_alpha);
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
