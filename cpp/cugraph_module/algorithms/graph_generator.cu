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

constexpr char const *kProcedureGenerate = "rmat";

constexpr char const *kArgumentScale = "scale";
constexpr char const *kArgumentNumEdges = "num_edges";

constexpr char const *kFieldMessage = "message";

void InsertMessageRecord(mgp_result *result, mgp_memory *memory, const char *message) {
  auto *record = mgp::result_new_record(result);

  mg_utility::InsertStringValueResult(record, kFieldMessage, message, memory);
}

struct VertexDelete {
  void operator()(mgp_vertex *v) {
    if (v) mgp::vertex_destroy(v);
  }
};

void GenerateRMAT(mgp_list *args, mgp_graph *graph, mgp_result *result, mgp_memory *memory) {
  try {
    auto scale = mgp::value_get_int(mgp::list_at(args, 0));
    auto num_edges = mgp::value_get_int(mgp::list_at(args, 1));

    raft::handle_t handle{};

    auto num_vertices = 1 << scale;  // RMAT generator defines this
    auto edges = mg_cugraph::GenerateCugraphRMAT(scale, num_edges, handle);

    std::vector<std::unique_ptr<mgp_vertex, VertexDelete>> vertices(num_vertices);
    for (std::size_t i = 0; i < num_vertices; ++i) {
      auto new_vertex = mgp::graph_create_vertex(graph, memory);

      vertices[i] = std::unique_ptr<mgp_vertex, VertexDelete>(mgp::vertex_copy(new_vertex, memory));

      mgp_vertex_destroy(new_vertex);
    }

    for (auto [src, dst] : edges) {
      auto &src_vertex_ptr = vertices[src];
      auto &dst_vertex_ptr = vertices[dst];

      mgp_vertex *src_vertex = src_vertex_ptr.get();
      mgp_vertex *dst_vertex = dst_vertex_ptr.get();

      auto new_edge = mgp::graph_create_edge(graph, src_vertex, dst_vertex, mgp_edge_type{.name = "LINK"}, memory);

      mgp_edge_destroy(new_edge);
    }

  } catch (const std::exception &e) {
    // We must not let any exceptions out of our module.
    mgp::result_set_error_msg(result, e.what());
    return;
  }
}
}  // namespace

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  mgp_value *default_scale;
  mgp_value *default_num_edges;
  try {
    auto *rmat_proc = mgp::module_add_write_procedure(module, kProcedureGenerate, GenerateRMAT);

    default_scale = mgp::value_make_int(4, memory);
    default_num_edges = mgp::value_make_int(100, memory);

    mgp::proc_add_opt_arg(rmat_proc, kArgumentScale, mgp::type_int(), default_scale);
    mgp::proc_add_opt_arg(rmat_proc, kArgumentNumEdges, mgp::type_int(), default_num_edges);
  } catch (const std::exception &e) {
    mgp_value_destroy(default_scale);
    mgp_value_destroy(default_num_edges);
    return 1;
  }

  mgp_value_destroy(default_scale);
  mgp_value_destroy(default_num_edges);
  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
