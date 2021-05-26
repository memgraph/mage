#include <memory>
#include <random>
#include <unordered_map>

#include <mg_graph.hpp>
#include "pagerank.hpp"

namespace pagerank_approx_util {

void NormalizeRank(std::vector<double> &rank) {
  const double sum = std::accumulate(rank.begin(), rank.end(), 0.0);
  for (double &value : rank) {
    value /= sum;
  }
}

}  // namespace pagerank_approx_util

namespace pagerank_approx_alg {

PageRankData context;
std::uint64_t global_R;
double global_epsilon;

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
      context.walks_tracker[current_id]++;

      while (true) {
        auto neighbors = graph.Neighbours(current_id);
        if (neighbors.empty()) break;

        // Pick and add the random outer edge
        auto number_of_neighbors = neighbors.size();
        auto next_id = neighbors[std::rand() % number_of_neighbors].node_id;

        walk.emplace_back(next_id);
        context.walks_table[next_id].insert(walk_index);
        context.walks_tracker[next_id]++;

        // Finish walk when random number is smaller than epsilon
        // Average length of walk is 1/epsilon
        if (distr(gen) < epsilon) {
          break;
        }

        current_id = next_id;
      }

      context.walks.emplace_back(std::move(walk));
      walk_index++;
    }
  }

  std::vector<double> pageranks(n);
  for (auto [node_id] : graph.Nodes()) {
    auto total = context.walks_tracker[node_id];
    pageranks[node_id] = total / ((n * R) / epsilon);
  }

  pagerank_approx_util::NormalizeRank(pageranks);
  return pageranks;
}

std::vector<double> Update(const mg_graph::GraphView<> &graph, const std::pair<std::uint64_t, uint64_t> new_edge) {
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> distr(0.0, 1.0);

  auto n = graph.Nodes().size();
  auto [from, to] = new_edge;

  std::unordered_set<std::uint64_t> walk_table_copy(context.walks_table[from]);
  for (auto walk_id : walk_table_copy) {
    auto &walk = context.walks[walk_id];

    auto position = std::find(walk.begin(), walk.end(), from) + 1;
    while (position != walk.end()) {
      auto node_id = *position;
      context.walks_table[node_id].erase(walk_id);
      context.walks_tracker[node_id]--;
      position++;
    }
    walk.erase(std::find(walk.begin(), walk.end(), from) + 1, walk.end());

    auto current_id = from;
    while (true) {
      auto neighbors = graph.Neighbours(current_id);
      if (neighbors.empty()) break;

      // Pick and add the random outer edge
      auto number_of_neighbors = neighbors.size();
      auto next_id = neighbors[std::rand() % number_of_neighbors].node_id;

      walk.emplace_back(next_id);
      context.walks_table[next_id].emplace(walk_id);
      context.walks_tracker[next_id]++;

      current_id = next_id;

      // Finish walk when random number is smaller than epsilon
      // Average length of walk is 1/epsilon
      if (distr(gen) < global_epsilon) {
        break;
      }

      current_id = next_id;
    }
  }

  std::vector<double> pageranks(n);
  for (auto [node_id] : graph.Nodes()) {
    auto total = context.walks_tracker[node_id];
    pageranks[node_id] = total / ((n * global_R) / global_epsilon);
  }

  pagerank_approx_util::NormalizeRank(pageranks);
  return pageranks;
}

}  // namespace pagerank_approx_alg
