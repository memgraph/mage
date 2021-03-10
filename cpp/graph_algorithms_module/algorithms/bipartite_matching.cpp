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
                         std::vector<int32_t> *colors, uint32_t node_index) {
  std::queue<uint32_t> unvisited;
  unvisited.push(node_index);

  std::vector<int32_t> &colorRef = *colors;
  colorRef[node_index] = 1;

  while (!unvisited.empty()) {
    uint32_t current_index = unvisited.front();
    unvisited.pop();
    for (const auto &neighbour : G.Neighbours(current_index)) {
      if (neighbour.node_id == current_index) {
        // Self loops are not allowed
        return false;
      }
      
      if (colorRef[neighbour.node_id] == colorRef[current_index]) {
        return false;
      }

      if (colorRef[neighbour.node_id] == -1) {
        colorRef[neighbour.node_id] = 1 - colorRef[current_index];
        unvisited.push(neighbour.node_id);
      }
    }
  }

  return true;
}
}  // namespace

namespace algorithms {
/// Returns the size of the maximum matching in a given bipartite graph. The
/// graph is defined by the sizes of two disjoint sets of nodes and by a set of
/// edges between those two sets of nodes. The nodes in both sets should be
/// indexed from 1 to `set_size`.
///
/// The algorithm runs in O(|V|*|E|) time where V represents a set of nodes and
/// E represents a set of edges.
size_t BipartiteMatching(
    const std::vector<std::pair<uint32_t, uint32_t>> &edges) {
  std::set<uint32_t> group_a;
  std::set<uint32_t> group_b;

  for (const auto &p : edges) {
    group_a.insert(p.first);
    group_b.insert(p.second);
  }

  size_t size_a = group_a.size();
  size_t size_b = group_b.size();

  // tells us which nodes from A have been visited
  std::vector<bool> visited(size_a + 1);

  // matched[i] = j <==> i-th node from B is matched with j-th node from A
  std::vector<uint32_t> matched(size_b + 1);

  std::vector<std::vector<uint32_t>> adj_list(size_a + 1);
  for (const auto &p : edges) adj_list[p.first].push_back(p.second);

  size_t ret = 0;
  for (uint32_t i = 1; i <= size_a; ++i) {
    std::fill(visited.begin(), visited.end(), false);
    ret += dfs(i, adj_list, &visited, &matched);
  }

  return ret;
}

/// Returns the size of the maximum matching in a given bipartite graph. The
/// graph is defined by the GraphView class. If the graph is bipartite, edge mapping
/// into 2 disjoint sets is performed and matching method is executed.
size_t BipartiteMatching(const graphdata::GraphView &G) {
  if (!IsGraphBipartite(G)) {
    return 0;
  }

  std::vector<std::pair<uint32_t, uint32_t>> edges;
  std::map<uint32_t, uint32_t> first_set;
  std::map<uint32_t, uint32_t> second_set;
  for (const auto &edge : G.Edges()) {
    if (first_set.count(edge.from) == 0) {
      first_set[edge.from] = first_set.size() + 1;
    }
    if (second_set.count(edge.to) == 0) {
      second_set[edge.to] = second_set.size() + 1;
    }

    edges.emplace_back(first_set[edge.from], second_set[edge.to]);
  }

  return BipartiteMatching(edges);
}

/// Returns true if graph is bipartite.
/// Otherwise returns false.
bool IsGraphBipartite(const graphdata::GraphView &G) {
  uint32_t node_size = G.Nodes().size();
  std::vector<int32_t> colors(node_size, -1);

  for (unsigned int i = 0; i < node_size; i++) {
    if (colors[i] == -1) {
      if (!IsSubgraphBipartite(G, &colors, i)) {
        return false;
      }
    }
  }

  return true;
}
}  // namespace algorithms
