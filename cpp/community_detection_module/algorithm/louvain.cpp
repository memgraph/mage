#include "louvain.hpp"
#include <omp.h>
#include <cstdint>
#include <unordered_set>
#include "mg_procedure.h"
#include "mg_utils.hpp"

namespace louvain_alg {

constexpr int kReplaceMap = 0;
constexpr int kThreadsOpt = 1;
constexpr int kNumColors = 16;

std::vector<std::int64_t> GrappoloCommunityDetection(GrappoloGraph &grappolo_graph, mgp_graph *graph, bool coloring,
                                                     std::uint64_t min_graph_size, double threshold,
                                                     double coloring_threshold, int num_threads) {

  auto number_of_vertices = grappolo_graph.numVertices;

  auto *cluster_array = (long *)malloc(number_of_vertices * sizeof(long));
#pragma omp parallel for
  for (long i = 0; i < number_of_vertices; i++) {
    cluster_array[i] = -1;
  }

  // Dynamically set currently.
  if (coloring) {
    runMultiPhaseColoring(&grappolo_graph, graph, cluster_array, coloring, kNumColors, kReplaceMap, min_graph_size, threshold,
                          coloring_threshold, num_threads, kThreadsOpt);
  } else {
    runMultiPhaseBasic(&grappolo_graph, graph, cluster_array, kReplaceMap, min_graph_size, threshold, coloring_threshold,
                       num_threads, kThreadsOpt);
  }

  // Store clustering information in vector
  std::vector<std::int64_t> result;
  for (long i = 0; i < number_of_vertices; ++i) {
    result.emplace_back(cluster_array[i]);
  }
  // Detach memory, no need to free graph instance, it will be removed inside algorithm
  free(cluster_array);

  // Return empty vector if algorithm has failed
  return result;
}

EdgesGraph GetGraphEdgeList(mgp_graph *memgraph_graph, mgp_memory *memory, const char *weight_property, double default_weight) {
  EdgesGraph edges; // source, destination, weight
  auto number_of_edges = 0;
  auto first_vertex_id = 0;
  bool first_vertex = true;

  auto *vertices_it = mgp::graph_iter_vertices(memgraph_graph, memory);  // Safe vertex iterator creation
  mg_utility::OnScopeExit delete_vertices_it([&vertices_it] { mgp::vertices_iterator_destroy(vertices_it); }); 
  for (auto *source = mgp::vertices_iterator_get(vertices_it); source;
        source = mgp::vertices_iterator_next(vertices_it)) {
    if (first_vertex) {
      first_vertex_id = mgp::vertex_get_id(source).as_int;
      first_vertex = false;
    }
    auto *edges_it = mgp::vertex_iter_out_edges(source, memory);  // Safe edge iterator creation
    mg_utility::OnScopeExit delete_edges_it([&edges_it] { mgp::edges_iterator_destroy(edges_it); });
    auto source_id = mgp::vertex_get_id(source).as_int - first_vertex_id;

    for (auto *out_edge = mgp::edges_iterator_get(edges_it); out_edge;
          out_edge = mgp::edges_iterator_next(edges_it)) {
      auto *destination = mgp::edge_get_to(out_edge);
      double weight = mg_utility::GetNumericProperty(out_edge, weight_property, memory, default_weight);
      auto destination_id = mgp::vertex_get_id(destination).as_int - first_vertex_id;
      number_of_edges++;
      edges.emplace_back(source_id, destination_id, weight);
    }
  }
  return edges;
}

EdgesGraph GetSubgraphEdgeList(mgp_graph *memgraph_graph, mgp_memory *memory, mgp_list *subgraph_nodes, mgp_list *subgraph_edges, const char *weight_property, double default_weight) {
  EdgesGraph edges; // source, destination, weight
  edges.reserve(mgp::list_size(subgraph_edges));
  std::unordered_map<int64_t, int64_t> subgraph_node_to_id;
  subgraph_node_to_id.reserve(mgp::list_size(subgraph_nodes));
  auto number_of_edges = 0;

  for (std::size_t i = 0; i < mgp::list_size(subgraph_nodes); i++) {
    auto *vertex = mgp::value_get_vertex(mgp::list_at(subgraph_nodes, i));
    subgraph_node_to_id.emplace(mgp::vertex_get_id(vertex).as_int, i);
  }

  for (std::size_t i = 0; i < mgp::list_size(subgraph_edges); i++) {
    auto *edge = mgp::value_get_edge(mgp::list_at(subgraph_edges, i));
    auto *source = mgp::edge_get_from(edge);
    auto *destination = mgp::edge_get_to(edge);
    auto source_id = mgp::vertex_get_id(source).as_int;
    auto destination_id = mgp::vertex_get_id(destination).as_int;

    if (subgraph_node_to_id.find(source_id) != subgraph_node_to_id.end() &&
        subgraph_node_to_id.find(destination_id) != subgraph_node_to_id.end()) {
      double weight = mg_utility::GetNumericProperty(edge, weight_property, memory, default_weight);
      number_of_edges++;
      edges.emplace_back(subgraph_node_to_id[source_id], subgraph_node_to_id[destination_id], weight);
    }
  }
  return edges;
}


void GetGrappoloSuitableGraph(GrappoloGraph &grappolo_graph, int num_threads, const EdgesGraph &edges) {
  std::unordered_set<int64_t> vertices;
  const auto number_of_edges = edges.size();
  
  auto edge_index = 0;
  auto tmp_edge_list = std::unique_ptr<edge[]>(new edge[number_of_edges]);  // Every edge stored ONCE
  for (auto [source, destination, weight] : edges) {
    tmp_edge_list[edge_index].head = source;  // The S indexmgp_list *subgraph_nodes, mgp_list *subgraph_edges, const char *weight_property, double default_weight, bool subgraph) {
    tmp_edge_list[edge_index].tail = destination;    // The T index: Zero-based indexing
    tmp_edge_list[edge_index].weight = weight; // The weight
    edge_index++;
    vertices.insert(source);
    vertices.insert(destination);
  }
  const auto number_of_vertices = vertices.size();

  omp_set_num_threads(num_threads);
  auto *edge_list_ptrs = static_cast<long *>(malloc((number_of_vertices + 1) * sizeof(long)));
  if (edge_list_ptrs == nullptr) {
    throw mg_exception::NotEnoughMemoryException();
  }

  auto *edge_list = static_cast<edge *>(malloc(number_of_edges * 2 * sizeof(edge)));  // Every edge stored twice
  if (edge_list == nullptr) {
    throw mg_exception::NotEnoughMemoryException();
  }

#pragma omp parallel for
  for (std::size_t i = 0; i <= number_of_vertices; i++) edge_list_ptrs[i] = 0;  // For first touch purposes

    // Build the EdgeListPtr Array: Cumulative addition
#pragma omp parallel for
  for (std::size_t i = 0; i < number_of_edges; i++) {
    __sync_fetch_and_add(&edge_list_ptrs[tmp_edge_list[i].head + 1], 1);  // Leave 0th position intact
    __sync_fetch_and_add(&edge_list_ptrs[tmp_edge_list[i].tail + 1], 1);  // Leave 0th position intact
  }
  for (std::size_t i = 0; i < number_of_vertices; i++) {
    edge_list_ptrs[i + 1] += edge_list_ptrs[i];  // Prefix Sum
  }
  // The last element of Cumulative will hold the total number of characters
  if (2 * number_of_edges != edge_list_ptrs[number_of_vertices]) {
    throw std::runtime_error("Community detection error: Error while graph fetching in the edge prefix sum creation");
  }

  // Keep track of how many edges have been added for a vertex:
  auto added = std::unique_ptr<long[]>(new long[number_of_vertices]);
  if (added == nullptr) {
    throw mg_exception::NotEnoughMemoryException();
  }
#pragma omp parallel for
  for (std::size_t i = 0; i < number_of_vertices; i++) added[i] = 0;

    // Build the edgeList from edgeListTmp:

#pragma omp parallel for 
  for (std::size_t i = 0; i < number_of_edges; i++) {
    auto head = tmp_edge_list[i].head;
    auto tail = tmp_edge_list[i].tail;
    auto weight = tmp_edge_list[i].weight;

    auto index = edge_list_ptrs[head] + __sync_fetch_and_add(&added[head], 1);
    edge_list[index].head = head;
    edge_list[index].tail = tail;
    edge_list[index].weight = weight;
    // Add the other way:
    index = edge_list_ptrs[tail] + __sync_fetch_and_add(&added[tail], 1);
    edge_list[index].head = tail;
    edge_list[index].tail = head;
    edge_list[index].weight = weight;
  }
  grappolo_graph.numVertices = number_of_vertices;
  grappolo_graph.numEdges = number_of_edges;
  grappolo_graph.edgeListPtrs = edge_list_ptrs;
  grappolo_graph.edgeList = edge_list;
  grappolo_graph.sVertices = number_of_vertices;
}
}  // namespace louvain_alg
