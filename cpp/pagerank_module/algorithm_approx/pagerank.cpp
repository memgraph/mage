#include <memory>
#include <unordered_map>

#include <mg_graph.hpp>
#include "pagerank.hpp"

namespace pagerank_approx_alg {
namespace {
PageRankData context;
std::uint64_t global_R;
double global_epsilon;

void NormalizeRank(std::vector<std::pair<std::uint64_t, double>> &rank) {
  const double sum = std::accumulate(rank.begin(), rank.end(), 0.0, [](auto &sum, auto &p) { return sum + p.second; });
  for (auto &[node_id, value] : rank) {
    value /= sum;
  }
}

std::vector<std::pair<std::uint64_t, double>> CalculatePageRank() {
  std::vector<std::pair<std::uint64_t, double>> pageranks;

  auto n = context.walks_counter.size();
  for (auto const [node_id, total] : context.walks_counter) {
    auto rank = total / ((n * global_R) / global_epsilon);
    pageranks.emplace_back(node_id, rank);
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
    next_id = graph.GetMemgraphNodeId(next_id);

    walk.emplace_back(next_id);
    context.walks_table[next_id].insert(walk_index);
    context.walks_counter[next_id]++;

    // Finish walk when random number is smaller than epsilon
    // Average length of walk is 1/epsilon
    if (distr(gen) < epsilon) {
      break;
    }

    current_id = graph.GetInnerNodeId(next_id);
  }
}

void UpdateCreate(const mg_graph::GraphView<> &graph, const std::pair<std::uint64_t, std::uint64_t> new_edge) {
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
    CreateRoute(graph, graph.GetInnerNodeId(current_id), walk, walk_index, global_epsilon / 2.0, *context.distr,
                *context.gen);
  }
}

void UpdateCreate(const mg_graph::GraphView<> &graph, const std::uint64_t new_vertex) {
  auto walk_index = context.walks.size();
  for (std::uint64_t i = 0; i < global_R; i++) {
    std::vector<std::uint64_t> walk;

    walk.emplace_back(new_vertex);
    context.walks_table[new_vertex].insert(walk_index);
    context.walks_counter[new_vertex]++;

    CreateRoute(graph, graph.GetInnerNodeId(new_vertex), walk, walk_index, global_epsilon, *context.distr,
                *context.gen);

    context.walks.emplace_back(std::move(walk));
    walk_index++;
  }
}

void UpdateDelete(const mg_graph::GraphView<> &graph, const std::pair<std::uint64_t, std::uint64_t> removed_edge) {
  auto [from, to] = removed_edge;

  std::unordered_set<std::uint64_t> walk_table_copy(context.walks_table[from]);
  for (auto walk_index : walk_table_copy) {
    auto &walk = context.walks[walk_index];

    auto position = std::find(walk.begin(), walk.end(), from) + 1;

    if (position == walk.end()) {
      continue;
    }

    // auto next = *(position + 1);
    // if (next != walk.end()) {
    //   continue;
    // }

    while (position != walk.end()) {
      auto node_id = *position;
      context.walks_table[node_id].erase(walk_index);
      context.walks_counter[node_id]--;
      position++;
    }
    walk.erase(std::find(walk.begin(), walk.end(), from) + 1, walk.end());

    auto current_id = from;

    // Skip creating routes if node does not exist anymore
    if (!graph.NodeExists(current_id)) {
      continue;
    }

    CreateRoute(graph, graph.GetInnerNodeId(current_id), walk, walk_index, global_epsilon / 2.0, *context.distr,
                *context.gen);
  }
}

void UpdateDelete(const mg_graph::GraphView<> &graph, const std::uint64_t removed_vertex) {
  context.walks_table.erase(removed_vertex);
  context.walks_counter.erase(removed_vertex);
}
}  // namespace

std::vector<std::pair<std::uint64_t, double>> SetPagerank(const mg_graph::GraphView<> &graph, const std::uint64_t R,
                                                          const double epsilon) {
  global_epsilon = epsilon;
  global_R = R;
  context.Init();

  auto walk_index = 0;
  for (auto [node_id] : graph.Nodes()) {
    // We have R random walks for each node in the graph
    for (std::uint64_t i = 0; i < R; i++) {
      std::vector<std::uint64_t> walk;

      auto current_id = graph.GetMemgraphNodeId(node_id);
      walk.emplace_back(current_id);
      context.walks_table[current_id].insert(walk_index);
      context.walks_counter[current_id]++;

      CreateRoute(graph, graph.GetInnerNodeId(current_id), walk, walk_index, epsilon, *context.distr, *context.gen);

      context.walks.emplace_back(std::move(walk));
      walk_index++;
    }
  }

  return CalculatePageRank();
}

std::vector<std::pair<std::uint64_t, double>> GetPagerank(const mg_graph::GraphView<> &graph) {
  if (context.IsEmpty()) {
    return SetPagerank(graph);
  }
  return CalculatePageRank();
}

std::vector<std::pair<std::uint64_t, double>> UpdatePagerank(
    const mg_graph::GraphView<> &graph, const std::vector<std::uint64_t> new_vertices,
    const std::vector<std::pair<std::uint64_t, uint64_t>> new_edges, const std::vector<std::uint64_t> deleted_vertices,
    const std::vector<std::pair<std::uint64_t, uint64_t>> deleted_edges) {
  if (context.IsEmpty()) {
    return SetPagerank(graph);
  }

  for (const auto edge : deleted_edges) {
    UpdateDelete(graph, edge);
  }
  for (const auto vertex : deleted_vertices) {
    UpdateDelete(graph, vertex);
  }
  for (const auto vertex : new_vertices) {
    UpdateCreate(graph, vertex);
  }
  for (const auto edge : new_edges) {
    UpdateCreate(graph, edge);
  }

  return CalculatePageRank();
}

void Reset() {
  auto global_context = context;

  global_context.Init();
}
}  // namespace pagerank_approx_alg
