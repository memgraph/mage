#pragma once

#include <chrono>
#include <random>
#include <set>

#include "data_structures/graph.hpp"

using namespace graphdata;

/// This class is threadsafe
class Timer {
 public:
  Timer() : start_time_(std::chrono::steady_clock::now()) {}

  template <typename TDuration = std::chrono::duration<double>>
  TDuration Elapsed() const {
    return std::chrono::duration_cast<TDuration>(
        std::chrono::steady_clock::now() - start_time_);
  }

 private:
  std::chrono::steady_clock::time_point start_time_;
};

/// Builds the graph from a given number of nodes and a list of edges.
/// Nodes should be 0-indexed and each edge should be provided in both
/// directions.
inline graphdata::Graph BuildGraph(
    uint32_t nodes, std::vector<std::pair<uint32_t, uint32_t>> edges) {
  graphdata::Graph G;
  for (uint32_t i = 0; i < nodes; ++i) G.CreateNode();

  for (auto &p : edges) G.CreateEdge(p.first, p.second);

  return G;
}

/// Generates random undirected graph with a given numer of nodes and edges.
/// The generated graph is not picked out of a uniform distribution.
inline graphdata::Graph GenRandomGraph(uint32_t nodes, uint32_t edges) {
  auto seed =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();
  std::mt19937 rng(seed);
  std::uniform_int_distribution<uint32_t> dist(0, nodes - 1);
  std::set<std::pair<uint32_t, uint32_t>> E;
  for (int i = 0; i < edges; ++i) {
    int u, v;
    do {
      u = dist(rng);
      v = dist(rng);
      if (u > v) std::swap(u, v);
    } while (u == v || E.find({u, v}) != E.end());
    E.insert({u, v});
  }
  return BuildGraph(
      nodes, std::vector<std::pair<uint32_t, uint32_t>>(E.begin(), E.end()));
}

/// Generates a random undirected tree with a given number of nodes.
/// The generated tree is not picked out of a uniform distribution.
inline graphdata::Graph GenRandomTree(uint32_t nodes) {
  auto seed =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();
  std::mt19937 rng(seed);
  std::vector<std::pair<uint32_t, uint32_t>> edges;
  for (int i = 1; i < nodes; ++i) {
    std::uniform_int_distribution<uint32_t> dist(0, i - 1);
    uint32_t dad = dist(rng);
    edges.emplace_back(dad, i);
  }
  return BuildGraph(nodes, edges);
}

/// Generates a complete graph with a given number of nodes.
inline graphdata::Graph GenCompleteGraph(uint32_t nodes) {
  std::vector<std::pair<uint32_t, uint32_t>> edges;
  for (int i = 0; i < nodes; ++i) {
    for (int j = i + 1; j < nodes; ++j) {
      edges.emplace_back(i, j);
    }
  }
  return BuildGraph(nodes, edges);
}

/// Generates a paper example graph.
inline graphdata::Graph PaperGraph() {
  Graph g;
  uint32_t spl = g.CreateNode();
  uint32_t sep = g.CreateNode();
  uint32_t hx2 = g.CreateNode();
  uint32_t hx3 = g.CreateNode();
  uint32_t mx = g.CreateNode();
  uint32_t rx = g.CreateNode();
  uint32_t e = g.CreateNode();
  uint32_t hx1 = g.CreateNode();

  g.CreateEdge(e, mx);
  g.CreateEdge(mx, rx);
  g.CreateEdge(rx, hx3);
  g.CreateEdge(hx3, spl);
  g.CreateEdge(spl, e);
  g.CreateEdge(spl, sep);
  g.CreateEdge(sep, hx2);
  g.CreateEdge(hx2, mx);
  g.CreateEdge(sep, e);
  g.CreateEdge(e, hx1);
  g.CreateEdge(hx1, e);

  g.CreateEdge(e, rx);
  g.CreateEdge(rx, e);
  g.CreateEdge(e, rx);
  g.CreateEdge(rx, e);
  g.CreateEdge(hx3, hx2);
  g.CreateEdge(sep, hx1);
  g.CreateEdge(e, rx);
  g.CreateEdge(e, rx);

  return g;
}
