#include <cugraph/algorithms.hpp>
#include <cugraph/graph_functions.hpp>
#include <raft/distance/distance.hpp>
#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>

#include <mg_exceptions.hpp>
#include <mg_utils.hpp>

namespace mg_cugraph {

using vertex_t = int64_t;
using edge_t = int64_t;
using weight_t = float;
using result_t = float;

template <bool TStoreTransposed = true, bool TMultiGPU = false>
auto CreateCugraphFromMemgraph(const mg_graph::GraphView<> &mg_graph, raft::handle_t const &handle) {
  const auto &mg_vertices = mg_graph.Nodes();
  const auto &mg_edges = mg_graph.Edges();

  // Flatten the data vector
  std::vector<vertex_t> mg_rows;
  mg_rows.reserve(mg_edges.size());
  std::vector<vertex_t> mg_cols;
  mg_cols.reserve(mg_edges.size());
  std::transform(mg_edges.begin(), mg_edges.end(), std::back_inserter(mg_rows),
                 [](const auto &edge) -> vertex_t { return edge.from; });
  std::transform(mg_edges.begin(), mg_edges.end(), std::back_inserter(mg_cols),
                 [](const auto &edge) -> vertex_t { return edge.to; });

  // Synchronize the data structures to the GPU
  auto stream = handle.get_stream();
  rmm::device_uvector<vertex_t> cu_rows(mg_rows.size(), stream);
  raft::update_device(cu_rows.data(), mg_rows.data(), mg_rows.size(), stream);
  rmm::device_uvector<vertex_t> cu_cols(mg_cols.size(), stream);
  raft::update_device(cu_cols.data(), mg_cols.data(), mg_cols.size(), stream);

  // TODO: Deal_with/pass edge weights to CuGraph graph.
  // TODO: Allow for multigraphs
  cugraph::graph_t<vertex_t, edge_t, weight_t, TStoreTransposed, TMultiGPU> cu_graph(handle);
  // NOTE: Renumbering is not required because graph coming from Memgraph is already correctly numbered.
  std::tie(cu_graph, std::ignore) =
      cugraph::create_graph_from_edgelist<vertex_t, edge_t, weight_t, TStoreTransposed, TMultiGPU>(
          handle, std::nullopt, std::move(cu_rows), std::move(cu_cols), std::nullopt,
          cugraph::graph_properties_t{false, false}, false, false);
  stream.synchronize_no_throw();

  return std::move(cu_graph);
}
}  // namespace mg_cugraph