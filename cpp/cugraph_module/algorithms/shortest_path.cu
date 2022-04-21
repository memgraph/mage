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

#include <omp.h>

#include <chrono>
#include <numeric>
#include <queue>
#include <unordered_map>
#include <unordered_set>

#include <mg_exceptions.hpp>
#include <mg_utils.hpp>

#include "mg_cugraph_utility.hpp"

namespace {
using vertex_t = int64_t;
using edge_t = int64_t;
using weight_t = double;
using result_t = double;

constexpr char const *kProcedureGet = "get";

constexpr char const *kArgumentSources = "sources";
constexpr char const *kArgumentTargets = "targets";

constexpr char const *kFieldSource = "source";
constexpr char const *kFieldTarget = "target";
constexpr char const *kFieldPath = "path";

// void InsertPathResult(mgp_graph *graph, mgp_result *result, mgp_memory *memory, std::uint64_t source_id,
//                       std::uint64_t target_id, std::vector<std::uint64_t> &edge_ids, mg_utility::EdgeStore &store) {
//   auto *record = mgp::result_new_record(result);

//   // Construct the graph out of reversed edge list
//   auto edges_size = edge_ids.size();
//   auto path = mgp::path_make_with_start(mgp::edge_get_from(store.Get(edge_ids[edges_size - 1])), memory);
//   for (std::int32_t i = edges_size - 1; i >= 0; --i) {
//     auto edge = store.Get(edge_ids[i]);
//     mgp::path_expand(path, edge);
//   }

//   // Insert records in Memgraph
//   mg_utility::InsertNodeValueResult(graph, record, kFieldSource, source_id, memory);
//   mg_utility::InsertNodeValueResult(graph, record, kFieldTarget, target_id, memory);
//   mg_utility::InsertPathValueResult(record, kFieldPath, path, memory);
// }

std::vector<std::uint64_t> TransformNodeIDs(const mg_graph::GraphView<> &mg_graph,
                                            std::vector<std::uint64_t> &mg_nodes) {
  std::vector<std::uint64_t> nodes;
  nodes.reserve(mg_nodes.size());
  std::transform(
      mg_nodes.begin(), mg_nodes.end(), std::back_inserter(nodes),
      [&mg_graph](const std::uint64_t node_id) -> std::uint64_t { return mg_graph.GetInnerNodeId(node_id); });
  return nodes;
}

std::vector<std::uint64_t> FetchAllNodesIDs(const mg_graph::GraphView<> &mg_graph) {
  std::vector<uint64_t> nodes(mg_graph.Nodes().size());
  std::iota(nodes.begin(), nodes.end(), 0);
  return nodes;
}

std::vector<std::uint64_t> FetchNodeIDs(const mg_graph::GraphView<> &mg_graph, mgp_list *mg_nodes) {
  std::vector<uint64_t> nodes;
  if (mg_nodes != nullptr) {
    auto sources_arg = mg_utility::GetNodeIDs(mg_nodes);
    nodes = TransformNodeIDs(mg_graph, sources_arg);
  } else {
    nodes = FetchAllNodesIDs(mg_graph);
  }
  return nodes;
}

void ShortestPath(mgp_list *args, mgp_graph *graph, mgp_result *result, mgp_memory *memory) {
  try {
    // Fetch the target & source IDs
    auto sources_arg =
        !mgp::value_is_null(mgp::list_at(args, 0)) ? mgp::value_get_list(mgp::list_at(args, 0)) : nullptr;

    auto targets_arg =
        !mgp::value_is_null(mgp::list_at(args, 1)) ? mgp::value_get_list(mgp::list_at(args, 1)) : nullptr;

    auto mg_graph = mg_utility::GetGraphView(graph, result, memory, mg_graph::GraphType::kDirectedGraph);
    // const auto &mg_graph = res.first;
    // const auto &edge_store = res.second;
    auto n_vertices = mg_graph.get()->Nodes().size();

    raft::handle_t handle{};
    auto stream = handle.get_stream();

    auto cu_graph = mg_cugraph::CreateCugraphFromMemgraph<vertex_t, edge_t, weight_t, false>(*mg_graph.get(), handle);
    auto cu_graph_view = cu_graph.view();

    // Fetch target inner IDs. If not provided, fetch all.
    auto targets = FetchNodeIDs(*mg_graph.get(), targets_arg);
    auto targets_size = targets.size();

    // Fetch sources inner IDs. If not provided, fetch all.
    auto sources = FetchNodeIDs(*mg_graph.get(), sources_arg);
    auto sources_size = sources.size();

    for (auto src_id : sources) {
      rmm::device_uvector<weight_t> distances_result(n_vertices, stream);
      thrust::uninitialized_fill(thrust::cuda::par.on(stream), distances_result.begin(), distances_result.end(), -1);
      rmm::device_uvector<vertex_t> predecessors_result(n_vertices, stream);
      thrust::uninitialized_fill(thrust::cuda::par.on(stream), predecessors_result.begin(), predecessors_result.end(),
                                 -1);

      cugraph::sssp<vertex_t, edge_t, weight_t, false>(handle, cu_graph_view, distances_result.data(),
                                                       predecessors_result.data(), static_cast<vertex_t>(src_id),
                                                       std::numeric_limits<weight_t>::max(), false);

      for (auto dst_id : targets) {
        auto curr_id = predecessors_result.element(dst_id, stream);
        if (curr_id == -1) continue;

        std::vector<std::uint64_t> path{dst_id};
        while (curr_id != src_id) {
          path.emplace_back(curr_id);
          curr_id = predecessors_result.element(curr_id, stream);
        };

        path.emplace_back(src_id);
        std::reverse(path.begin(), path.end());

        std::cout << "Path src-dst :";
        for (auto id : path) {
          std::cout << std::to_string(id) << ", ";
        }
        std::cout << std::endl;
      }
    }

  } catch (const std::exception &e) {
    mgp::result_set_error_msg(result, e.what());
    return;
  }
}
}  // namespace

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    auto *wcc_proc = mgp::module_add_read_procedure(module, kProcedureGet, ShortestPath);

    auto default_null = mgp::value_make_null(memory);
    mgp::proc_add_opt_arg(wcc_proc, kArgumentSources, mgp::type_nullable(mgp::type_list(mgp::type_node())),
                          default_null);
    mgp::proc_add_opt_arg(wcc_proc, kArgumentTargets, mgp::type_nullable(mgp::type_list(mgp::type_node())),
                          default_null);

    mgp::proc_add_result(wcc_proc, kFieldSource, mgp::type_node());
    mgp::proc_add_result(wcc_proc, kFieldTarget, mgp::type_node());
    mgp::proc_add_result(wcc_proc, kFieldPath, mgp::type_path());

    mgp::value_destroy(default_null);
  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
