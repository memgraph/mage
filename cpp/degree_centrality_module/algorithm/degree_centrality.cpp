#include "degree_centrality.hpp"

namespace degree_cenntrality_alg {

std::vector<double> GetDegreeCentrality(const mg_graph::GraphView<> &graph, const AlgorithmType algorithm_type) {
  auto nodes = graph.Nodes();
  auto number_of_nodes = nodes.size();

  // Initialize centrality values
  std::vector<double> degree_centralities(number_of_nodes, 0.0);

  // Degree centrality is the proportion of neighbors and maximum degree (n-1)
  for (const auto [node_id] : graph.Nodes()) {
    std::size_t degree;

    switch (algorithm_type) {
      case AlgorithmType::kUndirected:
        degree = graph.OutNeighbours(node_id).size() + graph.InNeighbours(node_id).size();
        break;

      case AlgorithmType::kOut:
        degree = graph.OutNeighbours(node_id).size();
        break;

      case AlgorithmType::kIn:
        degree = graph.InNeighbours(node_id).size();
        break;
    }

    // Degree centrality can be > 1 in multi-relational graphs
    degree_centralities[node_id] = degree / static_cast<double>((number_of_nodes - 1));
  }
  return degree_centralities;
}

}  // namespace degree_cenntrality_alg
