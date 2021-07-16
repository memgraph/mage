#include <memory>
#include <random>
#include <unordered_map>

#include <mg_graph.hpp>
#include "pagerank.hpp"

namespace pagerank_approx_alg {
namespace {
PageRankData context;
std::uint64_t global_R;
double global_epsilon;

void NormalizeRank(std::vector<double> &rank) {
  const double sum = std::accumulate(rank.begin(), rank.end(), 0.0);
  for (double &value : rank) {
    value /= sum;
  }
}

std::vector<double> CalculatePageRank(const mg_graph::GraphView<> &graph) {
  auto nodes = graph.Nodes();
  auto n = nodes.size();

  std::vector<double> pageranks(n);
  for (auto [node_id] : nodes) {
    auto total = context.walks_counter[node_id];
    pageranks[node_id] = total / ((n * global_R) / global_epsilon);
  }

  NormalizeRank(pageranks);
  return pageranks;
}

void CreateRoute(const mg_graph::GraphView<> &graph, const std::uint64_t start_id, std::vector<std::uint64_t> &walk,
                 const std::uint64_t walk_index, const double epsilon, std::uniform_real_distribution<float> distr,
                 std::mt19937 gen) {
  std::uint64_t current_id = start_id;
  while (true) {
    auto neighbors = graph.Neighbours(current_id);
    if (neighbors.empty()) break;

    // Pick and add the random outer edge
    auto number_of_neighbors = neighbors.size();
    auto next_id = neighbors[std::rand() % number_of_neighbors].node_id;

    walk.emplace_back(next_id);
    context.walks_table[next_id].insert(walk_index);
    context.walks_counter[next_id]++;

    // Finish walk when random number is smaller than epsilon
    // Average length of walk is 1/epsilon
    if (distr(gen) < epsilon) {
      break;
    }

    current_id = next_id;
  }
}

}  // namespace

std::vector<double> PageRankApprox(const mg_graph::GraphView<> &graph, const std::uint64_t R, const double epsilon) {
  global_epsilon = epsilon;
  global_R = R;
  context.Init();

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> distr(0.0, 1.0);

  auto n = graph.Nodes().size();

  auto walk_index = 0;
  for (auto [node_id] : graph.Nodes()) {
    // We have R random walks for each node in the graph
    for (std::uint64_t i = 0; i < R; i++) {
      std::vector<std::uint64_t> walk;

      auto current_id = node_id;
      walk.emplace_back(current_id);
      context.walks_table[current_id].insert(walk_index);
      context.walks_counter[current_id]++;

      CreateRoute(graph, current_id, walk, walk_index, epsilon, distr, gen);

      context.walks.emplace_back(std::move(walk));
      walk_index++;
    }
  }
  return CalculatePageRank(graph);
}

std::vector<double> Update(const mg_graph::GraphView<> &graph, const std::pair<std::uint64_t, std::uint64_t> new_edge) {
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> distr(0.0, 1.0);

  auto n = graph.Nodes().size();
  auto [from, to] = new_edge;

  std::unordered_set<std::uint64_t> walk_table_copy(context.walks_table[from]);
  for (auto walk_index : walk_table_copy) {
    auto &walk = context.walks[walk_index];

    auto position = std::find(walk.begin(), walk.end(), from) + 1;
    while (position != walk.end()) {
      auto node_id = *position;
      context.walks_table[node_id].erase(walk_index);
      context.walks_counter[node_id]--;
      position++;
    }
    walk.erase(std::find(walk.begin(), walk.end(), from) + 1, walk.end());

    auto current_id = from;
    CreateRoute(graph, current_id, walk, walk_index, global_epsilon / 2.0, distr, gen);
  }

  return CalculatePageRank(graph);
}

std::vector<double> Update(const mg_graph::GraphView<> &graph, const std::uint64_t new_vertex) {
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> distr(0.0, 1.0);

  auto n = graph.Nodes().size();

  auto walk_index = context.walks.size();
  for (std::uint64_t i = 0; i < global_R; i++) {
    std::vector<std::uint64_t> walk;

    walk.emplace_back(new_vertex);
    context.walks_table[new_vertex].insert(walk_index);
    context.walks_counter[new_vertex]++;

    CreateRoute(graph, new_vertex, walk, walk_index, global_epsilon, distr, gen);

    context.walks.emplace_back(std::move(walk));
    walk_index++;
  }

  return CalculatePageRank(graph);
}

}  // namespace pagerank_approx_alg
