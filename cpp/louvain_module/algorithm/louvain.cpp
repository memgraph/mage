#include "louvain.hpp"

namespace louvain_util {

const int kReplaceMap = 0;
const int kThreadsOpt = 1;
const int kNumColors = 16;

std::vector<std::uint64_t> GrappoloCommunityDetection(graph *grappolo_graph, bool coloring,
                                                      std::uint64_t min_graph_size, double threshold,
                                                      double coloring_threshold) {
  int num_threads = 1;  // Default is one thread
#pragma omp parallel
  { num_threads = omp_get_num_threads(); }

  // Datastructures to store clustering information
  auto number_of_vertices = grappolo_graph->numVertices;
  auto *clustering = (long *)malloc(number_of_vertices * sizeof(long));

  // The original version of the graph
  graph *grappolo_graph_original = (graph *)malloc(sizeof(graph));
  duplicateGivenGraph(grappolo_graph, grappolo_graph_original);

  // Call the clustering algorithm:
#pragma omp parallel for
  for (long i = 0; i < number_of_vertices; i++) {
    clustering[i] = -1;
  }

  if (coloring) {
    runMultiPhaseColoring(grappolo_graph, clustering, coloring, kNumColors, kReplaceMap, min_graph_size, threshold,
                          coloring_threshold, num_threads, kThreadsOpt);
  } else {
    runMultiPhaseBasic(grappolo_graph, clustering, kReplaceMap, min_graph_size, threshold, coloring_threshold,
                       num_threads, kThreadsOpt);
  }

  // Store clustering information in vector
  std::vector<std::uint64_t> result;
  for (int i = 0; i < number_of_vertices; ++i) {
    result.emplace_back(clustering[i]);
  }
  // Detach memory, no need to free graph instance, it will be removed inside algorithm
  free(clustering);

  // Return empty vector if algorithm has failed
  return result;
}

void LoadUndirectedEdges(const mg_graph::GraphView<> &memgraph_graph, graph *grappolo_graph) {
  int num_threads = 0;
#pragma omp parallel
  { num_threads = omp_get_num_threads(); }

  auto number_of_vertices = memgraph_graph.Nodes().size();
  auto number_of_edges = memgraph_graph.Edges().size();

  if (number_of_edges == 0) {
    grappolo_graph->numEdges = 0;
    return;
  }

  std::vector<std::pair<std::uint64_t, std::uint64_t>> edges;
  for (const auto [id, from, to] : memgraph_graph.Edges()) {
    edges.emplace_back(from, to);
  }

  auto *tmp_edge_list = (edge *)malloc(number_of_edges * sizeof(edge));  // Every edge stored ONCE

  // TODO: (jmatak) Add different weights on edges
  double default_weight = 1.0;
  for (long i = 0; i < number_of_edges; i++) {
    const auto [s_index, t_index] = edges[i];
    tmp_edge_list[i].head = s_index;           // The S index
    tmp_edge_list[i].tail = t_index;           // The T index: Zero-based indexing
    tmp_edge_list[i].weight = default_weight;  // Make it positive and cast to Double, fixed to 1.0
  }

  auto *edge_list_ptrs = (long *)malloc((number_of_vertices + 1) * sizeof(long));
  if (edge_list_ptrs == NULL) {
    throw mg_exception::NotEnoughMemoryException();
  }

  auto *edge_list = (edge *)malloc(2 * number_of_edges * sizeof(edge));  // Every edge stored twice
  if (edge_list == NULL) {
    throw mg_exception::NotEnoughMemoryException();
  }

#pragma omp parallel for
  for (long i = 0; i <= number_of_vertices; i++) edge_list_ptrs[i] = 0;  // For first touch purposes

    // Build the EdgeListPtr Array: Cumulative addition
#pragma omp parallel for
  for (long i = 0; i < number_of_edges; i++) {
    __sync_fetch_and_add(&edge_list_ptrs[tmp_edge_list[i].head + 1], 1);  // Leave 0th position intact
    __sync_fetch_and_add(&edge_list_ptrs[tmp_edge_list[i].tail + 1], 1);  // Leave 0th position intact
  }
  for (long i = 0; i < number_of_vertices; i++) {
    edge_list_ptrs[i + 1] += edge_list_ptrs[i];  // Prefix Sum
  }
  // The last element of Cumulative will hold the total number of characters
  if (2 * number_of_edges != edge_list_ptrs[number_of_vertices]) {
    throw std::runtime_error("Community detection error: Number of edges should be 2*NE");
  }

  // Keep track of how many edges have been added for a vertex:
  auto *added = (long *)malloc(number_of_vertices * sizeof(long));
  if (added == NULL) {
    throw mg_exception::NotEnoughMemoryException();
  }
#pragma omp parallel for
  for (long i = 0; i < number_of_vertices; i++) added[i] = 0;

    // Build the edgeList from edgeListTmp:
#pragma omp parallel for
  for (long i = 0; i < number_of_edges; i++) {
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

  // Define Grappolo graph structure
  grappolo_graph->sVertices = number_of_vertices;
  grappolo_graph->numVertices = number_of_vertices;
  grappolo_graph->numEdges = number_of_edges;
  grappolo_graph->edgeListPtrs = edge_list_ptrs;
  grappolo_graph->edgeList = edge_list;

  // Clean up:
  free(tmp_edge_list);
  free(added);
}
}  // namespace louvain_util

namespace louvain_alg {
std::vector<std::uint64_t> GetCommunities(const mg_graph::GraphView<> &memgraph_graph, bool coloring,
                                          std::uint64_t min_graph_shrink, double threshold, double coloring_threshold) {
  if (!memgraph_graph.Nodes().size()) {
    return std::vector<std::uint64_t>();
  }

  auto *grappolo_graph = (graph *)malloc(sizeof(graph));
  louvain_util::LoadUndirectedEdges(memgraph_graph, grappolo_graph);

  return louvain_util::GrappoloCommunityDetection(grappolo_graph, coloring, min_graph_shrink, threshold,
                                                  coloring_threshold);
}
}  // namespace louvain_alg