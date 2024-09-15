// Copyright (c) 2016-2022 Memgraph Ltd. [https://memgraph.com]
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mg_cugraph_utility.hpp"

namespace {
using vertex_t = int64_t;
using edge_t = int64_t;
using weight_t = double;

constexpr char const *kProcedureBetweenness = "get";

constexpr char const *kArgumentNormalized = "normalized";
constexpr char const *kArgumentDirected = "directed";
constexpr char const *kArgumentWeightProperty = "weight_property";

constexpr char const *kResultFieldNode = "node";
constexpr char const *kResultFieldBetweennessCentrality = "betweenness_centrality";

const double kDefaultWeight = 1.0;
constexpr char const *kDefaultWeightProperty = "weight";

void InsertBetweennessRecord(mgp_graph *graph, mgp_result *result, mgp_memory *memory, const std::uint64_t node_id,
                             double rank) {
  auto *node = mgp::graph_get_vertex_by_id(graph, mgp_vertex_id{.as_int = static_cast<int64_t>(node_id)}, memory);
  if (!node) {
    if (mgp::graph_is_transactional(graph)) {
      throw mg_exception::InvalidIDException();
    }
    return;
  }

  auto *record = mgp::result_new_record(result);
  if (record == nullptr) throw mg_exception::NotEnoughMemoryException();

  mg_utility::InsertNodeValueResult(record, kResultFieldNode, node, memory);
  mg_utility::InsertDoubleValueResult(record, kResultFieldBetweennessCentrality, rank, memory);
}

void BetweennessProc(mgp_list *args, mgp_graph *graph, mgp_result *result, mgp_memory *memory) {
  try {
    auto normalized = mgp::value_get_bool(mgp::list_at(args, 0));
    auto directed = mgp::value_get_bool(mgp::list_at(args, 1));
    auto weight_property = mgp::value_get_string(mgp::list_at(args, 2));

    raft::handle_t handle{};
    auto stream = handle.get_stream();
    auto mg_graph_type = directed ? mg_graph::GraphType::kDirectedGraph : mg_graph::GraphType::kUndirectedGraph;
    auto mg_graph =
        mg_utility::GetWeightedGraphView(graph, result, memory, mg_graph_type, weight_property, kDefaultWeight);
    if (mg_graph->Empty()) return;
    auto cu_graph = mg_cugraph::CreateCugraphFromMemgraph<vertex_t, edge_t, weight_t, false, false>(
        *mg_graph.get(), mg_graph_type, handle);
    auto cu_graph_view = cu_graph.view();

    // Calling https://github.com/rapidsai/cugraph/blob/branch-24.04/cpp/include/cugraph/algorithms.hpp#L341
    // NOTE: The current implementation does not support a weighted graph.
    auto betweenness_result = cugraph::betweenness_centrality<vertex_t, edge_t, weight_t, false>(
        handle, cu_graph_view, std::nullopt, std::nullopt);
    for (vertex_t node_id = 0; node_id < betweenness_result.size(); ++node_id) {
      auto rank = betweenness_result.element(node_id, stream);
      InsertBetweennessRecord(graph, result, memory, mg_graph->GetMemgraphNodeId(node_id), rank);
    }
  } catch (const std::exception &e) {
    // We must not let any exceptions out of our module.
    mgp::result_set_error_msg(result, e.what());
    return;
  }
}
}  // namespace

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  mgp_value *default_normalized;
  mgp_value *default_directed;
  mgp_value *default_weight_property;

  try {
    auto *betweenness_proc = mgp::module_add_read_procedure(module, kProcedureBetweenness, BetweennessProc);

    default_normalized = mgp::value_make_bool(true, memory);
    default_directed = mgp::value_make_bool(false, memory);
    default_weight_property = mgp::value_make_string(kDefaultWeightProperty, memory);

    mgp::proc_add_opt_arg(betweenness_proc, kArgumentNormalized, mgp::type_bool(), default_normalized);
    mgp::proc_add_opt_arg(betweenness_proc, kArgumentDirected, mgp::type_bool(), default_directed);
    mgp::proc_add_opt_arg(betweenness_proc, kArgumentWeightProperty, mgp::type_string(), default_weight_property);

    mgp::proc_add_result(betweenness_proc, kResultFieldNode, mgp::type_node());
    mgp::proc_add_result(betweenness_proc, kResultFieldBetweennessCentrality, mgp::type_float());
  } catch (const std::exception &e) {
    mgp_value_destroy(default_normalized);
    mgp_value_destroy(default_directed);
    mgp_value_destroy(default_weight_property);
    return 1;
  }

  mgp_value_destroy(default_normalized);
  mgp_value_destroy(default_directed);
  mgp_value_destroy(default_weight_property);
  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
