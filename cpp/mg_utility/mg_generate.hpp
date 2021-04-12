#pragma once

#include <chrono>
#include <memory>
#include <queue>
#include <random>
#include <set>

#include "mg_graph.hpp"

namespace mg_generate {

/// Builds the graph from a given number of nodes and a list of edges.
/// Nodes should be 0-indexed and each edge should be provided in both
/// directions.
std::unique_ptr<mg_graph::Graph<>> BuildGraph(std::uint64_t num_nodes,
                                              std::vector<std::pair<std::uint64_t, std::uint64_t>> edges) {
  auto G = std::make_unique<mg_graph::Graph<>>();
  for (std::uint64_t i = 0; i < num_nodes; ++i) G->CreateNode(i);
  for (const auto [from, to] : edges) G->CreateEdge(from, to);

  return G;
}

/// Generates random undirected graph with a given numer of nodes and edges.
/// The generated graph is not picked out of a uniform distribution.
std::unique_ptr<mg_graph::Graph<>> GenRandomGraph(std::uint64_t num_nodes, std::uint64_t num_edges) {
  using IntPair = std::pair<std::uint64_t, std::uint64_t>;

  auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  std::mt19937 rng(seed);
  std::uniform_int_distribution<std::uint64_t> dist(0, num_nodes - 1);

  std::set<IntPair> rand_edges;
  for (std::uint64_t i = 0; i < num_edges; ++i) {
    std::optional<IntPair> edge;
    do {
      edge = std::minmax(dist(rng), dist(rng));
    } while (edge->first == edge->second || rand_edges.find(*edge) != rand_edges.end());
    rand_edges.insert(*edge);
  }
  return BuildGraph(num_nodes, {rand_edges.begin(), rand_edges.end()});
}

/// Generates a random undirected tree with a given number of nodes.
/// The generated tree is not picked out of a uniform distribution.
std::unique_ptr<mg_graph::Graph<>> GenRandomTree(std::uint64_t num_nodes) {
  auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  std::mt19937 rng(seed);
  std::vector<std::pair<std::uint64_t, std::uint64_t>> edges;
  for (std::uint64_t i = 1; i < num_nodes; ++i) {
    std::uniform_int_distribution<std::uint64_t> dist(0, i - 1);
    edges.emplace_back(dist(rng), i);
  }
  return BuildGraph(num_nodes, edges);
}

}  // namespace mg_generate