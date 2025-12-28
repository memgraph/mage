// Copyright (c) 2016-2022 Memgraph Ltd. [https://memgraph.com]
// Modified for cuGraph 25.x API compatibility
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
using result_t = double;

constexpr char const *kProcedureBetweennessCentrality = "get";

constexpr char const *kArgumentNormalized = "normalized";
constexpr char const *kArgumentDirected = "directed";

constexpr char const *kResultFieldNode = "node";
constexpr char const *kResultFieldBetweenness = "betweenness";

void InsertBetweennessRecord(mgp_graph *graph, mgp_result *result, mgp_memory *memory, const std::uint64_t node_id,
                             double betweenness) {
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
  mg_utility::InsertDoubleValueResult(record, kResultFieldBetweenness, betweenness, memory);
}

void BetweennessCentralityProc(mgp_list *args, mgp_graph *graph, mgp_result *result, mgp_memory *memory) {
  try {
    auto normalized = mgp::value_get_bool(mgp::list_at(args, 0));
    auto directed = mgp::value_get_bool(mgp::list_at(args, 1));

    auto graph_type = directed ? mg_graph::GraphType::kDirectedGraph : mg_graph::GraphType::kUndirectedGraph;
    auto mg_graph = mg_utility::GetGraphView(graph, result, memory, graph_type);
    if (mg_graph->Empty()) return;

    // Define handle and operation stream
    raft::handle_t handle{};
    auto stream = handle.get_stream();

    // Betweenness centrality uses store_transposed = false
    auto [cu_graph, edge_props] = mg_cugraph::CreateCugraphFromMemgraph<vertex_t, edge_t, weight_t, false, false>(
        *mg_graph.get(), graph_type, handle);

    auto cu_graph_view = cu_graph.view();
    auto n_vertices = cu_graph_view.number_of_vertices();

    // Get edge weight view from edge properties
    auto edge_weight_view = mg_cugraph::GetEdgeWeightView<edge_t>(edge_props);

    // Modern cuGraph 25.x Betweenness Centrality API - returns device_uvector
    auto betweenness = cugraph::betweenness_centrality<vertex_t, edge_t, weight_t, false>(
        handle,
        cu_graph_view,
        edge_weight_view,
        std::nullopt,  // vertices (use all)
        normalized,
        false,  // include_endpoints
        false); // do_expensive_check

    // Copy results to host and output
    std::vector<result_t> h_betweenness(n_vertices);
    raft::update_host(h_betweenness.data(), betweenness.data(), n_vertices, stream);
    handle.sync_stream();

    for (vertex_t node_id = 0; node_id < static_cast<vertex_t>(n_vertices); ++node_id) {
      InsertBetweennessRecord(graph, result, memory, mg_graph->GetMemgraphNodeId(node_id), h_betweenness[node_id]);
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
  try {
    auto *betweenness_proc =
        mgp::module_add_read_procedure(module, kProcedureBetweennessCentrality, BetweennessCentralityProc);

    default_normalized = mgp::value_make_bool(true, memory);
    default_directed = mgp::value_make_bool(true, memory);

    mgp::proc_add_opt_arg(betweenness_proc, kArgumentNormalized, mgp::type_bool(), default_normalized);
    mgp::proc_add_opt_arg(betweenness_proc, kArgumentDirected, mgp::type_bool(), default_directed);

    mgp::proc_add_result(betweenness_proc, kResultFieldNode, mgp::type_node());
    mgp::proc_add_result(betweenness_proc, kResultFieldBetweenness, mgp::type_float());
  } catch (const std::exception &e) {
    mgp_value_destroy(default_normalized);
    mgp_value_destroy(default_directed);
    return 1;
  }

  mgp_value_destroy(default_normalized);
  mgp_value_destroy(default_directed);
  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
