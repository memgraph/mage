#include "katz.hpp"

namespace katz_alg {
namespace {

class KatzCentralityData {
 public:
  void Init(const mg_graph::GraphView<> &graph) {
    centralities.clear();
    omegas.clear();

    // Initialize the centrality vector
    std::unordered_map<std::uint64_t, double> centrality;
    InitVertexMap(centrality, 0.0, graph);
    centralities.emplace_back(std::move(centrality));

    // Initialize the omega
    std::unordered_map<std::uint64_t, double> omega;
    InitVertexMap(omega, 1.0, graph);
    omegas.emplace_back(std::move(omega));

    // Initialize the lr vector
    InitVertexMap(lr, 0.0, graph);
    // Initialize the ur vector
    InitVertexMap(ur, 0.0, graph);
  }

  bool IsEmpty() const { return centralities.empty(); }

  void AddIteration(const mg_graph::GraphView<> &graph) {
    iteration++;

    // Initialize the centrality vector
    std::unordered_map<std::uint64_t, double> centrality;
    InitVertexMap(centrality, 0.0, graph);
    centralities.emplace_back(std::move(centrality));

    // Initialize the omega
    std::unordered_map<std::uint64_t, double> omega;
    InitVertexMap(omega, 0.0, graph);
    omegas.emplace_back(std::move(omega));
  }

  std::vector<std::unordered_map<std::uint64_t, double>> centralities;
  std::vector<std::unordered_map<std::uint64_t, double>> omegas;
  // Initialize the lower bound vector
  std::unordered_map<std::uint64_t, double> lr;
  // Initialize the upper bound vector
  std::unordered_map<std::uint64_t, double> ur;

  std::vector<std::uint64_t> active_nodes;

  std::uint64_t iteration = 0;
};

KatzCentralityData context;
double alpha;
std::uint64_t k;
double epsilon;

std::uint64_t MaxDegree(const mg_graph::GraphView<> &graph) {
  std::vector<std::size_t> deg_vector;
  std::transform(graph.Nodes().begin(), graph.Nodes().end(), std::back_inserter(deg_vector),
                 [&graph](mg_graph::Node<> vertex) -> std::size_t { return graph.Neighbours(vertex.id).size(); });
  auto deg_max = *std::max_element(deg_vector.begin(), deg_vector.end());
  return deg_max;
}

bool PairSortDesc(std::pair<std::uint64_t, double> a, std::pair<std::uint64_t, double> b) {
  return a.second < b.second;
}

bool Converged(std::vector<std::uint64_t> &active_nodes, std::uint64_t k, double epsilon) {
  auto centrality = katz_alg::context.centralities[context.iteration];
  auto lr = katz_alg::context.lr;
  auto ur = katz_alg::context.ur;

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

void InitVertexMap(std::unordered_map<std::uint64_t, double> &map, double default_value,
                   const mg_graph::GraphView<> &graph) {
  for (auto &[_v] : graph.Nodes()) {
    // Transform inner ID to Memgraph
    auto v = graph.GetMemgraphNodeId(_v);

    map[v] = default_value;
  }
}

std::vector<std::pair<std::uint64_t, double>> KatzCentralityLoop(std::vector<std::uint64_t> &active_nodes,
                                                                 const mg_graph::GraphView<> &graph, double alpha,
                                                                 std::uint64_t k, double epsilon, double gamma) {
  while (!Converged(active_nodes, k, epsilon)) {
    katz_alg::context.AddIteration(graph);
    auto iteration = katz_alg::context.iteration;

    for (auto &[_v] : graph.Nodes()) {
      // Transform inner ID to Memgraph
      auto v = graph.GetMemgraphNodeId(_v);
      for (auto &[_u, _] : graph.Neighbours(v)) {
        // Transform inner ID to Memgraph
        auto u = graph.GetMemgraphNodeId(_u);
        katz_alg::context.omegas[iteration][v] += katz_alg::context.omegas[iteration - 1][u];
      }
      katz_alg::context.centralities[iteration][v] =
          katz_alg::context.centralities[iteration][v] + pow(alpha, iteration) * katz_alg::context.omegas[iteration][v];

      // Update the lower and upper bound
      katz_alg::context.lr[v] = katz_alg::context.centralities[iteration][v];
      katz_alg::context.ur[v] = katz_alg::context.centralities[iteration][v] +
                                pow(alpha, iteration + 1) * katz_alg::context.omegas[iteration][v] * gamma;
    }
  }

  // Transform the resulting values
  return std::vector<std::pair<std::uint64_t, double>>(
      katz_alg::context.centralities[katz_alg::context.iteration].begin(),
      katz_alg::context.centralities[katz_alg::context.iteration].end());
}

void UpdateLevel(KatzCentralityData &context_new, std::set<std::uint64_t> &from_nodes,
                 const std::vector<std::pair<std::uint64_t, uint64_t>> &new_edges,
                 const std::vector<std::pair<std::uint64_t, uint64_t>> &deleted_edges,
                 const mg_graph::GraphView<> &graph) {
  auto i = context_new.iteration;
  for (auto v : from_nodes) {
    for (auto [w, _] : graph.OutNeighbours(v)) {
      from_nodes.emplace(w);
      context_new.omegas[i][w] += context_new.omegas[i - 1][v] - katz_alg::context.omegas[i - 1][w];
    }
  }

  for (auto [w, v] : new_edges) {
    context_new.omegas[i][w] += context_new.omegas[i - 1][v];
  }

  for (auto [w, v] : deleted_edges) {
    context_new.omegas[i][w] -= katz_alg::context.omegas[i - 1][v];
  }

  for (auto w : from_nodes) {
    katz_alg::context.centralities[i][w] - pow(katz_alg::alpha, i) * katz_alg::context.omegas[i][w] +
        pow(katz_alg::alpha, i) * context_new.omegas[i][w];
  }

  katz_alg::context.omegas[i] = std::move(context_new.omegas[i]);
}
}  // namespace
std::vector<std::pair<std::uint64_t, double>> GetKatzCentrality(const mg_graph::GraphView<> &graph, double alpha,
                                                                std::uint64_t k, double epsilon) {
  katz_alg::alpha = alpha;
  katz_alg::k = k;
  katz_alg::epsilon = epsilon;
  katz_alg::context.Init(graph);

  auto deg_max = MaxDegree(graph);
  double gamma = deg_max / (1. - (alpha * deg_max));

  // Initialize the active vector
  std::transform(graph.Nodes().begin(), graph.Nodes().end(), std::back_inserter(katz_alg::context.active_nodes),
                 [&graph](mg_graph::Node<> vertex) -> std::uint64_t { return graph.GetMemgraphNodeId(vertex.id); });

  return KatzCentralityLoop(katz_alg::context.active_nodes, graph, katz_alg::alpha, katz_alg::k, katz_alg::epsilon,
                            gamma);
}

std::vector<std::pair<std::uint64_t, double>> UpdateKatz(
    const mg_graph::GraphView<> &graph, const std::vector<std::uint64_t> &new_vertices,
    const std::vector<std::pair<std::uint64_t, uint64_t>> &new_edges,
    const std::vector<std::uint64_t> &deleted_vertices,
    const std::vector<std::pair<std::uint64_t, uint64_t>> &deleted_edges) {
  auto deg_max = MaxDegree(graph);
  double gamma = deg_max / (1. - (alpha * deg_max));

  std::set<std::uint64_t> from_nodes;
  std::set<std::uint64_t> to_nodes;

  for (auto [from, to] : new_edges) {
    from_nodes.emplace(from);
    to_nodes.emplace(to);
  }
  for (auto [w, v] : deleted_edges) {
    from_nodes.emplace(w);
    to_nodes.emplace(v);
  }

  KatzCentralityData context_new;
  context_new.Init(graph);
  for (std::uint64_t i = 0; i < katz_alg::context.iteration; i++) {
    context_new.AddIteration(graph);
    UpdateLevel(context_new, from_nodes, new_edges, deleted_edges, graph);
  }

  for (auto w : from_nodes) {
    // Update the lower and upper bound
    auto iteration = katz_alg::context.iteration;

    katz_alg::context.lr[w] = katz_alg::context.centralities[iteration][w];
    katz_alg::context.ur[w] = katz_alg::context.centralities[iteration][w] +
                              pow(katz_alg::alpha, iteration + 1) * katz_alg::context.omegas[iteration][w];
  }

  for (auto [w] : graph.Nodes()) {
    auto min_lower_element = *std::min_element(katz_alg::context.lr.begin(), katz_alg::context.lr.end(),
                                               [](const auto &l, const auto &r) { return l.second < r.second; });
    auto min_lr = min_lower_element.second;
    if (katz_alg::context.ur[w] >= (min_lr - katz_alg::epsilon)) {
      katz_alg::context.active_nodes.emplace_back(w);
    }
  }

  return KatzCentralityLoop(katz_alg::context.active_nodes, graph, katz_alg::alpha, katz_alg::k, katz_alg::epsilon,
                            gamma);
}
}  // namespace katz_alg
