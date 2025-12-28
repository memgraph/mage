// Copyright 2022 Memgraph Ltd.
// Modified for cuGraph 25.x API compatibility
//
// Licensed under the Apache License, Version 2.0

#pragma once

#include <cugraph/algorithms.hpp>
#include <cugraph/arithmetic_variant_types.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_generators.hpp>
#include <cugraph/legacy/graph.hpp>

#include <raft/core/handle.hpp>
#include <raft/core/device_span.hpp>
#include <raft/random/rng_state.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <mg_exceptions.hpp>
#include <mg_utils.hpp>

namespace mg_cugraph {

///
///@brief Create a cuGraph graph object from a given Memgraph graph.
/// Modern cuGraph 25.x API - NO weight_t template parameter.
/// Edge properties returned as std::vector<edge_arithmetic_property_t<edge_t>>.
///
/// NOTE: Renumbering is NOT required because GraphView already provides
/// vertices as contiguous 0-based position indices. cuGraph indices will
/// match GraphView position indices, so GetMemgraphNodeId(cuGraph_index) works.
///
///@tparam TVertexT Vertex identifier type
///@tparam TEdgeT Edge identifier type
///@tparam TWeightT Weight type (used for edge property construction, not graph template)
///@tparam TStoreTransposed Store transposed in memory
///@tparam TMultiGPU Multi-GPU Graph
///@param mg_graph Memgraph graph object
///@param graph_type Type of the graph - directed/undirected
///@param handle Handle for GPU communication
///@return Tuple of cuGraph graph object and vector of edge properties
///
template <typename TVertexT = int64_t, typename TEdgeT = int64_t, typename TWeightT = double,
          bool TStoreTransposed = true, bool TMultiGPU = false>
auto CreateCugraphFromMemgraph(const mg_graph::GraphView<> &mg_graph, const mg_graph::GraphType graph_type,
                               raft::handle_t const &handle) {
  auto mg_edges = mg_graph.Edges();
  auto mg_nodes = mg_graph.Nodes();

  if (graph_type == mg_graph::GraphType::kUndirectedGraph) {
    std::vector<mg_graph::Edge<>> undirected_edges;
    std::transform(mg_edges.begin(), mg_edges.end(), std::back_inserter(undirected_edges),
                   [](const auto &edge) -> mg_graph::Edge<> {
                     return {edge.id, edge.to, edge.from};
                   });
    mg_edges.reserve(2 * mg_edges.size());
    mg_edges.insert(mg_edges.end(), undirected_edges.begin(), undirected_edges.end());
  }

  // Flatten the data vector
  std::vector<TVertexT> mg_src;
  mg_src.reserve(mg_edges.size());
  std::vector<TVertexT> mg_dst;
  mg_dst.reserve(mg_edges.size());
  std::vector<TWeightT> mg_weight;
  mg_weight.reserve(mg_edges.size());
  std::vector<TVertexT> mg_vertices;
  mg_vertices.reserve(mg_nodes.size());

  std::transform(mg_edges.begin(), mg_edges.end(), std::back_inserter(mg_src),
                 [](const auto &edge) -> TVertexT { return edge.from; });
  std::transform(mg_edges.begin(), mg_edges.end(), std::back_inserter(mg_dst),
                 [](const auto &edge) -> TVertexT { return edge.to; });
  std::transform(
      mg_edges.begin(), mg_edges.end(), std::back_inserter(mg_weight),
      [&mg_graph](const auto &edge) -> TWeightT { return mg_graph.IsWeighted() ? mg_graph.GetWeight(edge.id) : 1.0; });
  std::transform(mg_nodes.begin(), mg_nodes.end(), std::back_inserter(mg_vertices),
                 [](const auto &node) -> TVertexT { return node.id; });

  // Synchronize the data structures to the GPU
  auto stream = handle.get_stream();
  rmm::device_uvector<TVertexT> cu_src(mg_src.size(), stream);
  raft::update_device(cu_src.data(), mg_src.data(), mg_src.size(), stream);
  rmm::device_uvector<TVertexT> cu_dst(mg_dst.size(), stream);
  raft::update_device(cu_dst.data(), mg_dst.data(), mg_dst.size(), stream);
  rmm::device_uvector<TWeightT> cu_weight(mg_weight.size(), stream);
  raft::update_device(cu_weight.data(), mg_weight.data(), mg_weight.size(), stream);
  rmm::device_uvector<TVertexT> cu_vertices(mg_vertices.size(), stream);
  raft::update_device(cu_vertices.data(), mg_vertices.data(), mg_vertices.size(), stream);

  // Create edge properties vector using variant type
  std::vector<cugraph::arithmetic_device_uvector_t> edge_properties;
  edge_properties.push_back(std::move(cu_weight));

  // Modern cuGraph 25.x API - create_graph_from_edgelist
  // renumber=false because GraphView already provides 0-based contiguous indices
  auto [cu_graph, edge_props, renumber_map] =
      cugraph::create_graph_from_edgelist<TVertexT, TEdgeT, TStoreTransposed, TMultiGPU>(
          handle,
          std::make_optional(std::move(cu_vertices)),
          std::move(cu_src),
          std::move(cu_dst),
          std::move(edge_properties),
          cugraph::graph_properties_t{graph_type == mg_graph::GraphType::kDirectedGraph, false},
          false,       // renumber - NOT needed, GraphView already provides 0..n-1 indices
          std::nullopt,
          std::nullopt,
          false);

  handle.sync_stream();

  return std::make_tuple(std::move(cu_graph), std::move(edge_props));
}

///
///@brief Get edge weight view from edge properties vector (returns double weights).
/// Helper to extract weight view from the variant-based edge properties.
///
///@tparam TEdgeT Edge identifier type
///@param edge_props Vector of edge properties from create_graph_from_edgelist
///@return Optional edge property view for weights
///
template <typename TEdgeT = int64_t>
std::optional<cugraph::edge_property_view_t<TEdgeT, double const*>> GetEdgeWeightView(
    std::vector<cugraph::edge_arithmetic_property_t<TEdgeT>>& edge_props) {
  if (edge_props.empty()) {
    return std::nullopt;
  }
  // Edge properties are stored as variants - get the double version
  auto& prop = edge_props[0];
  if (std::holds_alternative<cugraph::edge_property_t<TEdgeT, double>>(prop)) {
    return std::get<cugraph::edge_property_t<TEdgeT, double>>(prop).view();
  }
  return std::nullopt;
}

///
///@brief Create a cuGraph legacy graph object from a given Memgraph graph.
/// This method generates the graph in the Compressed Sparse Row format.
///
/// NOTE: This legacy API is required for algorithms that only support CSR format:
/// - balancedCutClustering (cugraph::ext_raft::)
/// - spectralModularityMaximization (cugraph::ext_raft::)
///
///@tparam TVertexT Vertex identifier type
///@tparam TEdgeT Edge identifier type
///@tparam TWeightT Weight type
///@param mg_graph Memgraph graph object
///@param handle Handle for GPU communication
///@return cuGraph legacy graph object
///
template <typename TVertexT = int64_t, typename TEdgeT = int64_t, typename TWeightT = double>
auto CreateCugraphLegacyFromMemgraph(const mg_graph::GraphView<> &mg_graph, raft::handle_t const &handle) {
  auto mg_nodes = mg_graph.Nodes();
  auto mg_edges = mg_graph.Edges();
  const auto n_edges = mg_edges.size();
  const auto n_vertices = mg_nodes.size();

  // Flatten the data vector into CSR format
  std::vector<TEdgeT> mg_deg_sum;
  std::vector<TVertexT> mg_dst;
  std::vector<TWeightT> mg_weight;

  mg_deg_sum.push_back(0);
  for (std::size_t v_id = 0; v_id < n_vertices; v_id++) {
    mg_deg_sum.push_back(mg_deg_sum[v_id] + mg_graph.Neighbours(v_id).size());

    auto neighbors = mg_graph.Neighbours(v_id);
    sort(neighbors.begin(), neighbors.end(), [](mg_graph::Neighbour<> &l_neighbor, mg_graph::Neighbour<> &r_neighbor) {
      return l_neighbor.node_id < r_neighbor.node_id;
    });

    for (const auto &dst : neighbors) {
      mg_dst.push_back(static_cast<TVertexT>(dst.node_id));
      mg_weight.push_back(mg_graph.IsWeighted() ? mg_graph.GetWeight(dst.edge_id) : 1.0);
    }
  }

  // Synchronize the data structures to the GPU
  auto stream = handle.get_stream();
  rmm::mr::device_memory_resource *mr = rmm::mr::get_current_device_resource();

  rmm::device_buffer offsets_buffer(mg_deg_sum.data(), sizeof(TEdgeT) * (mg_deg_sum.size()), stream, mr);
  rmm::device_buffer dsts_buffer(mg_dst.data(), sizeof(TVertexT) * (mg_dst.size()), stream, mr);
  rmm::device_buffer weights_buffer(mg_weight.data(), sizeof(TWeightT) * (mg_weight.size()), stream, mr);

  cugraph::legacy::GraphSparseContents<TVertexT, TEdgeT, TWeightT> csr_contents{
      static_cast<TVertexT>(n_vertices), static_cast<TEdgeT>(n_edges),
      std::make_unique<rmm::device_buffer>(std::move(offsets_buffer)),
      std::make_unique<rmm::device_buffer>(std::move(dsts_buffer)),
      std::make_unique<rmm::device_buffer>(std::move(weights_buffer))};

  return std::make_unique<cugraph::legacy::GraphCSR<TVertexT, TEdgeT, TWeightT>>(std::move(csr_contents));
}

///
///@brief RMAT (Recursive MATrix) Generator of a graph.
///
///@tparam TVertexT Vertex identifier type
///@param rng_state RNG state for reproducibility
///@param scale Scale factor for number of vertices. |V| = 2 ** scale
///@param num_edges Number of edges generated
///@param a Probability of the first partition
///@param b Probability of the second partition
///@param c Probability of the third partition
///@param clip_and_flip Clip and flip
///@param handle Handle for GPU communication
///@return Edges in edge list format
///
template <typename TVertexT = int64_t>
auto GenerateCugraphRMAT(raft::random::RngState& rng_state, std::size_t scale, std::size_t num_edges,
                         double a, double b, double c, bool clip_and_flip, raft::handle_t const &handle) {
  auto stream = handle.get_stream();

  // cuGraph 25.x RMAT API takes RngState reference
  auto [cu_src, cu_dst] =
      cugraph::generate_rmat_edgelist<TVertexT>(handle, rng_state, scale, num_edges, a, b, c, clip_and_flip);

  std::vector<std::pair<std::uint64_t, std::uint64_t>> mg_edges;
  mg_edges.reserve(num_edges);

  std::vector<TVertexT> h_src(num_edges);
  std::vector<TVertexT> h_dst(num_edges);
  raft::update_host(h_src.data(), cu_src.data(), num_edges, stream);
  raft::update_host(h_dst.data(), cu_dst.data(), num_edges, stream);
  handle.sync_stream();

  for (std::size_t i = 0; i < num_edges; ++i) {
    mg_edges.emplace_back(static_cast<std::uint64_t>(h_src[i]),
                          static_cast<std::uint64_t>(h_dst[i]));
  }
  return mg_edges;
}

}  // namespace mg_cugraph
