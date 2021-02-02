#include <bits/stdc++.h>

#include <queue>

#include "algorithms/algorithms.hpp"

namespace {
size_t dfs(uint32_t node, const std::vector<std::vector<uint32_t>> &adj_list,
           std::vector<bool> *visited, std::vector<uint32_t> *matched) {
  if ((*visited)[node]) return 0;
  (*visited)[node] = true;
  for (uint32_t nxt : adj_list[node]) {
    if ((*matched)[nxt] == 0 ||
        dfs((*matched)[nxt], adj_list, visited, matched)) {
      (*matched)[nxt] = node;
      return 1;
    }
  }
  return 0;
}

/// The method runs a BFS algorithm and colors the graph in 2 colors.
/// The graph is bipartite if for any vertex all neighbors have
/// the opposite color. In that case, method will return true.
/// Otherwise, false.
bool IsSubgraphBipartite(const graphdata::GraphView &G,
                         std::vector<uint32_t> &colors, uint32_t node_index) {
  std::queue<uint32_t> unvisited;
  unvisited.push(node_index);
  colors[node_index] = 1;

  while (!unvisited.empty()) {
    uint32_t current_index = unvisited.front();
    unvisited.pop();
    for (const auto &neighbour : G.Neighbours(current_index)) {
      if (neighbour.node_id == current_index) {
        // Self loops are not allowed
        return false;
      }
      if (colors[neighbour.node_id] == colors[current_index]) {
        return false;
      }
      if (colors[neighbour.node_id] == -1) {
        colors[neighbour.node_id] = 1 - colors[current_index];
        unvisited.push(neighbour.node_id);
      }
    }
  }

  return true;
}
}  // namespace

namespace algorithms {
size_t BipartiteMatching(
    const std::vector<std::pair<uint32_t, uint32_t>> &edges) {

  if (!IsGraphBipartite(edges)) {
    return 0;
  }
    
  std::set<uint32_t> group_a;
  std::set<uint32_t> group_b;

  for (const auto &p : edges) {
    group_a.insert(p.first);
    group_b.insert(p.second);
  }

  size_t ret = 0;
  size_t size_a = group_a.size();
  size_t size_b = group_b.size();

  // tells us which nodes from A have been visited
  std::vector<bool> visited(size_a + 1);

  // matched[i] = j <==> i-th node from B is matched with j-th node from A
  std::vector<uint32_t> matched(size_b + 1);

  std::vector<std::vector<uint32_t>> adj_list(size_a + 1);
  for (const auto &p : edges) adj_list[p.first].push_back(p.second);

  for (uint32_t i = 1; i <= size_a; ++i) {
    std::fill(visited.begin(), visited.end(), false);
    ret += dfs(i, adj_list, &visited, &matched);
  }

  return ret;
}

size_t BipartiteMatching(const graphdata::GraphView &G) {
  std::vector<std::pair<uint32_t, uint32_t>> edges;
  for (const auto &edge : G.Edges()) {
    edges.emplace_back(std::make_pair(edge.from, edge.to));
  }

  size_t maximum_matching = BipartiteMatching(edges);
  return maximum_matching;
}

/// Returns true if graph represented by edges is bipartite.
/// Otherwise false.
bool IsGraphBipartite(const std::vector<std::pair<uint32_t, uint32_t>> &edges) {
  std::set<uint32_t> group_a;
  std::set<uint32_t> group_b;
  std::vector<uint32_t> intersection;

  for (const auto &p : edges) {
    group_a.insert(p.first);
    group_b.insert(p.second);
  }

  std::set_intersection(group_a.begin(), group_a.end(), group_b.begin(),
                        group_b.end(), std::back_inserter(intersection));

  if (intersection.size() > 0) {
    return false;
  }

  return true;
}

bool IsGraphBipartite(const graphdata::GraphView &G) {
  uint32_t node_size = G.Nodes().size();
  std::vector<uint32_t> colors(node_size);
  fill(colors.begin(), colors.end(), -1);

  for (uint32_t i = 0; i < node_size; i++) {
    if (colors[i] == -1) {
      if (!IsSubgraphBipartite(G, colors, i)) {
        return false;
      }
    }
  }

  return true;
}
}  // namespace algorithms
