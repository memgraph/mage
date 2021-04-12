#pragma once

#include <chrono>
#include <memory>
#include <queue>
#include <random>
#include <set>

#include "mg_graph.hpp"

namespace mg_test_utility {
/// This class is threadsafe
class Timer {
 public:
  Timer() : start_time_(std::chrono::steady_clock::now()) {}

  template <typename TDuration = std::chrono::duration<double>>
  TDuration Elapsed() const {
    return std::chrono::duration_cast<TDuration>(std::chrono::steady_clock::now() - start_time_);
  }

 private:
  std::chrono::steady_clock::time_point start_time_;
};

/// Builds the graph from a given number of nodes and a list of edges.
/// Nodes should be 0-indexed and each edge should be provided in both
/// directions.
std::unique_ptr<mg_graph::Graph<>> BuildGraph(std::uint64_t nodes,
                                              std::vector<std::pair<std::uint64_t, std::uint64_t>> edges) {
  auto G = std::make_unique<mg_graph::Graph<>>();
  for (std::uint64_t i = 0; i < nodes; ++i) G->CreateNode(i);
  for (const auto [from, to] : edges) G->CreateEdge(from, to);

  return G;
}

/// Generates random undirected graph with a given numer of nodes and edges.
/// The generated graph is not picked out of a uniform distribution.
std::unique_ptr<mg_graph::Graph<>> GenRandomGraph(std::uint64_t nodes, std::uint64_t edges) {
  using IntPair = std::pair<std::uint64_t, std::uint64_t>;

  auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  std::mt19937 rng(seed);
  std::uniform_int_distribution<std::uint64_t> dist(0, nodes - 1);

  std::set<IntPair> E;
  for (std::uint64_t i = 0; i < edges; ++i) {
    std::optional<IntPair> edge;
    do {
      edge = std::minmax(dist(rng), dist(rng));
    } while (edge->first == edge->second || E.find(*edge) != E.end());
    E.insert(*edge);
  }
  return BuildGraph(nodes, {E.begin(), E.end()});
}

/// Generates a random undirected tree with a given number of nodes.
/// The generated tree is not picked out of a uniform distribution.
std::unique_ptr<mg_graph::Graph<>> GenRandomTree(std::uint64_t nodes) {
  auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  std::mt19937 rng(seed);
  std::vector<std::pair<std::uint64_t, std::uint64_t>> edges;
  for (std::uint64_t i = 1; i < nodes; ++i) {
    std::uniform_int_distribution<std::uint64_t> dist(0, i - 1);
    auto dad = dist(rng);
    edges.emplace_back(dad, i);
  }
  return BuildGraph(nodes, edges);
}
}  // namespace mg_test_utility