#include "degree_centrality.hpp"

namespace degree_cenntrality_alg {

std::vector<double> GetDegreeCentrality(const mg_graph::GraphView<> &graph) {
  auto nodes = graph.Nodes();
  auto number_of_nodes = nodes.size();

  // Initialize centrality values
  std::vector<double> degree_centralities(number_of_nodes, 0.0);

  // Degree centrality is the proportion of neighbors and maximum degree (n-1)
  for (const auto [node_id] : graph.Nodes()) {
    auto degree = graph.Neighbours(node_id).size();

    // Degree centrality can be > 1 in multi-relational graphs
    degree_centralities[node_id] = degree / static_cast<double>((number_of_nodes - 1));
  }
  return degree_centralities;
}

}  // namespace degree_cenntrality_alg