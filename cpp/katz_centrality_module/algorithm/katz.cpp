#include "katz.hpp"

namespace katz_alg {
namespace {

bool PairSortDesc(std::pair<std::uint64_t, double> a, std::pair<std::uint64_t, double> b) {
  return a.second < b.second;
}

bool Converged(std::vector<std::uint64_t> &active_nodes, std::uint64_t k, double epsilon,
               const std::unordered_map<std::uint64_t, double> &centrality,
               const std::unordered_map<std::uint64_t, double> &lr,
               const std::unordered_map<std::uint64_t, double> &ur) {
  // Create a vector of active centralities and their values
  std::vector<std::pair<std::uint64_t, double>> active_centrality;
  for (auto m : active_nodes) {
    active_centrality.emplace_back(m, centrality.at(m));
  }

  // Partially sort the centralities. Keep only the first `k` sorted
  std::partial_sort(active_centrality.begin(), active_centrality.begin() + k, active_centrality.end(), PairSortDesc);

  for (std::size_t i = k + 1; i < centrality.size(); i++) {
    if (ur.at(active_centrality[i].first) - epsilon < lr.at(active_centrality[k].first)) {
      active_centrality.erase(active_centrality.begin() + i);
    }
  }

  // If the size of active nodes is higher than `k`, algorithm hasn't converged
  if (active_centrality.size() > k) {
    return false;
  }

  // Next step is checking whether the top of the partial list has converged too
  auto size = std::min(active_centrality.size(), k);
  for (std::size_t i = 1; i < size; i++) {
    double u = ur.at(active_centrality[i].first);
    double l = lr.at(active_centrality[i - 1].first);

    if (u - epsilon < l) {
      return false;
    }
  }
  return true;
}

std::uint64_t MaxDegree(const mg_graph::GraphView<> &graph) {
  std::vector<std::size_t> deg_vector;
  std::transform(graph.Nodes().begin(), graph.Nodes().end(), std::back_inserter(deg_vector),
                 [&graph](mg_graph::Node<> vertex) -> std::size_t { return graph.Neighbours(vertex.id).size(); });
  auto deg_max = *std::max_element(deg_vector.begin(), deg_vector.end());
  return deg_max;
}

void InitVertexMap(std::unordered_map<std::uint64_t, double> &map, double default_value,
                   const mg_graph::GraphView<> &graph) {
  for (auto &[_v] : graph.Nodes()) {
    // Transform inner ID to Memgraph
    auto v = graph.GetMemgraphNodeId(_v);

    map[v] = default_value;
  }
}
}  // namespace

std::vector<std::pair<std::uint64_t, double>> GetKatzCentrality(const mg_graph::GraphView<> &graph, double alpha,
                                                                std::uint64_t k, double epsilon) {
  auto deg_max = MaxDegree(graph);
  auto n = graph.Nodes().size();
  double gamma = deg_max / (1. - (alpha * deg_max));

  // Initialize the centrality vector
  std::unordered_map<std::uint64_t, double> centrality;
  InitVertexMap(centrality, 0.0, graph);

  // Initialize the lower bound vector
  std::unordered_map<std::uint64_t, double> lr;
  InitVertexMap(lr, 0.0, graph);

  // Initialize the upper bound vector
  std::unordered_map<std::uint64_t, double> ur;
  InitVertexMap(ur, 0.0, graph);

  // Initialize the omega
  std::unordered_map<std::uint64_t, double> omega;
  InitVertexMap(omega, 1.0, graph);

  std::uint64_t iteration = 0;

  // Initialize the active vector
  std::vector<std::uint64_t> active_nodes;
  std::transform(graph.Nodes().begin(), graph.Nodes().end(), std::back_inserter(active_nodes),
                 [&graph](mg_graph::Node<> vertex) -> std::uint64_t { return graph.GetMemgraphNodeId(vertex.id); });

  while (!Converged(active_nodes, k, epsilon, centrality, lr, ur)) {
    iteration++;
    std::unordered_map<std::uint64_t, double> omega_new;
    InitVertexMap(omega_new, 0.0, graph);

    for (auto &[_v] : graph.Nodes()) {
      // Transform inner ID to Memgraph
      auto v = graph.GetMemgraphNodeId(_v);
      for (auto &[_u, _] : graph.Neighbours(v)) {
        // Transform inner ID to Memgraph
        auto u = graph.GetMemgraphNodeId(_u);
        omega_new[v] += omega[u];
      }
      centrality[v] += pow(alpha, iteration) * omega_new[v];
      lr[v] = centrality[v];
      ur[v] = centrality[v] + pow(alpha, iteration + 1) * omega_new[v];
    }

    omega = omega_new;
  }

  // Transform the resulting values
  return std::vector<std::pair<std::uint64_t, double>>(centrality.begin(), centrality.end());
}
}  // namespace katz_alg
