#include <cugraph/algorithms.hpp>
#include <cugraph/functions.hpp>  // legacy coo_to_csr
#include <cugraph/graph_functions.hpp>

#include <raft/distance/distance.hpp>
#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>

#include <mg_exceptions.hpp>
#include <mg_utils.hpp>

namespace mg_cugraph {

template <typename TVertexT = int64_t, typename TEdgeT = int64_t, typename TWeightT = double,
          bool TStoreTransposed = true, bool TMultiGPU = false>
auto CreateCugraphFromMemgraph(const mg_graph::GraphView<> &mg_graph, raft::handle_t const &handle) {
  const auto &mg_edges = mg_graph.Edges();

  // Flatten the data vector
  std::vector<TVertexT> mg_src;
  mg_src.reserve(mg_edges.size());
  std::vector<TVertexT> mg_dst;
  mg_dst.reserve(mg_edges.size());
  std::vector<TWeightT> mg_weight;
  mg_weight.reserve(mg_edges.size());

  std::transform(mg_edges.begin(), mg_edges.end(), std::back_inserter(mg_src),
                 [](const auto &edge) -> TVertexT { return edge.from; });
  std::transform(mg_edges.begin(), mg_edges.end(), std::back_inserter(mg_dst),
                 [](const auto &edge) -> TVertexT { return edge.to; });
  std::transform(mg_edges.begin(), mg_edges.end(), std::back_inserter(mg_weight),
                 [mg_graph](const auto &edge) -> TWeightT { return mg_graph.GetWeight(edge.id); });

  // Synchronize the data structures to the GPU
  auto stream = handle.get_stream();
  rmm::device_uvector<TVertexT> cu_src(mg_src.size(), stream);
  raft::update_device(cu_src.data(), mg_src.data(), mg_src.size(), stream);
  rmm::device_uvector<TVertexT> cu_dst(mg_dst.size(), stream);
  raft::update_device(cu_dst.data(), mg_dst.data(), mg_dst.size(), stream);
  rmm::device_uvector<TWeightT> cu_weight(mg_weight.size(), stream);
  raft::update_device(cu_weight.data(), mg_weight.data(), mg_weight.size(), stream);

  // TODO: Deal_with/pass edge weights to CuGraph graph.
  // TODO: Allow for multigraphs
  cugraph::graph_t<TVertexT, TEdgeT, TWeightT, TStoreTransposed, TMultiGPU> cu_graph(handle);
  // NOTE: Renumbering is not required because graph coming from Memgraph is already correctly numbered.
  std::tie(cu_graph, std::ignore) =
      cugraph::create_graph_from_edgelist<TVertexT, TEdgeT, TWeightT, TStoreTransposed, TMultiGPU>(
          handle, std::nullopt, std::move(cu_src), std::move(cu_dst), std::move(cu_weight),
          cugraph::graph_properties_t{false, false}, false, false);
  stream.synchronize_no_throw();

  return std::move(cu_graph);
}

template <typename TVertexT = int64_t, typename TEdgeT = int64_t, typename TWeightT = double>
auto CreateCugraphLegacyFromMemgraph(const mg_graph::GraphView<> &mg_graph, raft::handle_t const &handle) {
  const auto &mg_edges = mg_graph.Edges();
  const auto n_edges = mg_edges.size();
  const auto n_vertices = mg_graph.Nodes().size();

  const auto &mg_edges = mg_graph.Edges();

  // Flatten the data vector
  std::vector<TVertexT> mg_src;
  mg_src.reserve(mg_edges.size());
  std::vector<TVertexT> mg_dst;
  mg_dst.reserve(mg_edges.size());
  std::vector<TWeightT> mg_weight;
  mg_weight.reserve(mg_edges.size());

  std::transform(mg_edges.begin(), mg_edges.end(), std::back_inserter(mg_src),
                 [](const auto &edge) -> TVertexT { return edge.from; });
  std::transform(mg_edges.begin(), mg_edges.end(), std::back_inserter(mg_dst),
                 [](const auto &edge) -> TVertexT { return edge.to; });
  std::transform(mg_edges.begin(), mg_edges.end(), std::back_inserter(mg_weight),
                 [mg_graph](const auto &edge) -> TWeightT { return mg_graph.GetWeight(edge.id); });

  // Synchronize the data structures to the GPU
  auto stream = handle.get_stream();
  rmm::device_uvector<TVertexT> cu_src(mg_src.size(), stream);
  raft::update_device(cu_src.data(), mg_src.data(), mg_src.size(), stream);
  rmm::device_uvector<TVertexT> cu_dst(mg_dst.size(), stream);
  raft::update_device(cu_dst.data(), mg_dst.data(), mg_dst.size(), stream);
  rmm::device_uvector<TWeightT> cu_weight(mg_weight.size(), stream);
  raft::update_device(cu_weight.data(), mg_weight.data(), mg_weight.size(), stream);

  cugraph::legacy::GraphCOOView<TVertexT, TEdgeT, TWeightT> cooview(
      cu_src.data(), cu_dst.data(), mg_weight.data(), static_cast<TVertexT>(n_vertices), static_cast<TEdgeT>(n_edges));

  return cugraph::coo_to_csr<TVertexT, TEdgeT, TWeightT>(cooview);
}
}  // namespace mg_cugraph