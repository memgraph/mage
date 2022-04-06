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

#include <cugraph/algorithms.hpp>
#include <cugraph/graph_functions.hpp>
#include <mg_exceptions.hpp>
#include <mg_utils.hpp>
#include <raft/distance/distance.hpp>
#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>

// Vertex and Edge types have to be signed, otherwise module fails to load
// inside Memgraph because CuGraph Pagerank only supports signed graph types
// https://github.com/rapidsai/cugraph/blob/branch-21.12/cpp/src/link_analysis/pagerank_sg.cu
using vertex_t = int64_t;
using edge_t = int64_t;
using weight_t = float;
using result_t = float;

constexpr char const *kProcedurePagerank = "pagerank";
constexpr char const *kArgumentMaxIterations = "max_iterations";
constexpr char const *kArgumentDampingFactor = "damping_factor";
constexpr char const *kArgumentStopEpsilon = "stop_epsilon";
constexpr char const *kResultFieldNode = "node";
constexpr char const *kResultFieldRank = "rank";

void InsertPagerankRecord(mgp_graph *graph, mgp_result *result, mgp_memory *memory, const std::uint64_t node_id,
                          double rank) {
  auto *record = mgp::result_new_record(result);
  mg_utility::InsertNodeValueResult(graph, record, kResultFieldNode, node_id, memory);
  mg_utility::InsertDoubleValueResult(record, kResultFieldRank, rank, memory);
}

template <bool TStoreTransposed = true, bool TMultiGPU = false>
auto CreateCugraphFromMemgraph(raft::handle_t const &handle, mgp_graph *mg_graph, mgp_result *result,
                               mgp_memory *memory) {
  auto h_graph = mg_utility::GetGraphView(mg_graph, result, memory, mg_graph::GraphType::kDirectedGraph);
  const auto &h_vertices = h_graph->Nodes();
  const auto &h_edges = h_graph->Edges();
  std::vector<vertex_t> h_rows;
  h_rows.reserve(h_edges.size());
  std::vector<vertex_t> h_cols;
  h_cols.reserve(h_edges.size());
  std::transform(h_edges.begin(), h_edges.end(), std::back_inserter(h_rows),
                 [](const auto &edge) -> vertex_t { return edge.from; });
  std::transform(h_edges.begin(), h_edges.end(), std::back_inserter(h_cols),
                 [](const auto &edge) -> vertex_t { return edge.to; });
  auto stream = handle.get_stream();
  rmm::device_uvector<vertex_t> d_rows(h_rows.size(), stream);
  raft::update_device(d_rows.data(), h_rows.data(), h_rows.size(), stream);
  rmm::device_uvector<vertex_t> d_cols(h_cols.size(), stream);
  raft::update_device(d_cols.data(), h_cols.data(), h_cols.size(), stream);
  // TODO(gitbuda): Deal_with/pass edge weights to CuGraph graph.
  cugraph::graph_t<vertex_t, edge_t, weight_t, TStoreTransposed, TMultiGPU> d_graph(handle);
  // NOTE: Renumbering is not required because graph coming from Memgraph is already correctly numbered.
  std::tie(d_graph, std::ignore) =
      cugraph::create_graph_from_edgelist<vertex_t, edge_t, weight_t, TStoreTransposed, TMultiGPU>(
          handle, std::nullopt, std::move(d_rows), std::move(d_cols), std::nullopt,
          cugraph::graph_properties_t{false, false}, false, false);
  stream.synchronize_no_throw();
  return std::make_pair(std::move(h_graph), std::move(d_graph));
}

void PagerankProc(mgp_list *args, mgp_graph *mg_graph, mgp_result *result, mgp_memory *memory) {
  try {
    auto max_iterations = mgp::value_get_int(mgp::list_at(args, 0));
    auto damping_factor = mgp::value_get_double(mgp::list_at(args, 1));
    auto stop_epsilon = mgp::value_get_double(mgp::list_at(args, 2));

    raft::handle_t handle{};
    auto stream = handle.get_stream();

    auto [mg_graph_view, mg_cugraph] = CreateCugraphFromMemgraph(handle, mg_graph, result, memory);
    auto mg_cugraph_view = mg_cugraph.view();
    rmm::device_uvector<result_t> d_pageranks(mg_cugraph_view.get_number_of_vertices(), stream);

    // IMPORTANT: store_transposed has to be true because cugraph::pagerank
    // only accepts true. It's hard to detect/debug problem because nvcc error
    // messages contain only the top call details + graph_view has many
    // template paremeters.
    cugraph::pagerank<vertex_t, edge_t, weight_t, result_t, false>(handle, mg_cugraph_view, std::nullopt, std::nullopt,
                                                                   std::nullopt, std::nullopt, d_pageranks.data(),
                                                                   damping_factor, stop_epsilon, max_iterations);

    for (int node_id = 0; node_id < d_pageranks.size(); ++node_id) {
      auto rank = d_pageranks.element(node_id, stream);
      InsertPagerankRecord(mg_graph, result, memory, mg_graph_view->GetMemgraphNodeId(node_id), rank);
    }
  } catch (const std::exception &e) {
    // We must not let any exceptions out of our module.
    mgp::result_set_error_msg(result, e.what());
    return;
  }
}

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  mgp_value *default_max_iterations;
  mgp_value *default_damping_factor;
  mgp_value *default_stop_epsilon;
  try {
    auto *pagerank_proc = mgp::module_add_read_procedure(module, kProcedurePagerank, PagerankProc);

    default_max_iterations = mgp::value_make_int(100, memory);
    default_damping_factor = mgp::value_make_double(0.85, memory);
    default_stop_epsilon = mgp::value_make_double(1e-5, memory);

    mgp::proc_add_opt_arg(pagerank_proc, kArgumentMaxIterations, mgp::type_int(), default_max_iterations);
    mgp::proc_add_opt_arg(pagerank_proc, kArgumentDampingFactor, mgp::type_float(), default_damping_factor);
    mgp::proc_add_opt_arg(pagerank_proc, kArgumentStopEpsilon, mgp::type_float(), default_stop_epsilon);

    mgp::proc_add_result(pagerank_proc, kResultFieldNode, mgp::type_node());
    mgp::proc_add_result(pagerank_proc, kResultFieldRank, mgp::type_float());

  } catch (const std::exception &e) {
    mgp_value_destroy(default_max_iterations);
    mgp_value_destroy(default_damping_factor);
    mgp_value_destroy(default_stop_epsilon);
    return 1;
  }

  mgp_value_destroy(default_max_iterations);
  mgp_value_destroy(default_damping_factor);
  mgp_value_destroy(default_stop_epsilon);

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
