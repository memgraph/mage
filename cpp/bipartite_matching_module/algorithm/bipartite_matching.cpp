#include <set>

#include "bipartite_matching.hpp"

namespace bipartite_matching_util {
bool BipartiteMatchinDFS(uint64_t node, const std::vector<std::vector<uint64_t>> &adj_list, std::vector<bool> &visited,
                         std::vector<uint64_t> &matched) {
  if (visited[node]) return false;

  visited[node] = true;
  for (auto nxt : adj_list[node]) {
    if (matched[nxt] == 0 || BipartiteMatchinDFS(matched[nxt], adj_list, visited, matched)) {
      matched[nxt] = node;
      return true;
    }
  }
  return false;
}
}  // namespace bipartite_matching_util

namespace bipartite_matching_alg {

/// The method runs a BFS algorithm and colors the graph in 2 colors.
/// The graph is bipartite if for any vertex all neighbors have
/// the opposite color. In that case, method will return true.
/// Otherwise, false.
bool IsSubgraphBipartite(const mg_graph::GraphView<> *G, std::vector<int64_t> colors, uint64_t node_index) {
  std::queue<uint64_t> unvisited;
  unvisited.push(node_index);

  auto &colorRef = colors;
  colorRef[node_index] = 1;

  while (!unvisited.empty()) {
    auto current_index = unvisited.front();
    unvisited.pop();
    for (const auto &neighbour : G->Neighbours(current_index)) {
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

/// Returns the size of the maximum matching in a given bipartite graph. The
/// graph is defined by the sizes of two disjoint sets of nodes and by a set of
/// edges between those two sets of nodes. The nodes in both sets should be
/// indexed from 1 to `set_size`.
///
/// The algorithm runs in O(|V|*|E|) time where V represents a set of nodes and
/// E represents a set of edges.
uint64_t BipartiteMatching(const std::vector<std::pair<uint64_t, uint64_t>> &edges) {
  std::set<uint64_t> group_a;
  std::set<uint64_t> group_b;

  for (const auto &p : edges) {
    group_a.insert(p.first);
    group_b.insert(p.second);
  }

  auto size_a = group_a.size();
  auto size_b = group_b.size();

  // tells us which nodes from A have been visited
  std::vector<bool> visited(size_a + 1);

  // matched[i] = j <==> i-th node from B is matched with j-th node from A
  std::vector<uint64_t> matched(size_b + 1);

  std::vector<std::vector<uint64_t>> adj_list(size_a + 1);
  for (const auto &p : edges) adj_list[p.first].push_back(p.second);

  uint64_t ret = 0;
  for (uint64_t i = 1; i <= size_a; ++i) {
    std::fill(visited.begin(), visited.end(), false);
    ret += bipartite_matching_util::BipartiteMatchinDFS(i, adj_list, visited, matched);
  }

  return ret;
}

/// Returns the size of the maximum matching in a given bipartite graph. The
/// graph is defined by the GraphView class. If the graph is bipartite, edge
/// mapping into 2 disjoint sets is performed and matching method is executed.
size_t BipartiteMatching(const mg_graph::GraphView<> *G) {
  if (!bipartite_matching_alg::IsGraphBipartite(G)) {
    return 0;
  }

  std::vector<std::pair<uint64_t, uint64_t>> edges;
  std::map<uint64_t, uint64_t> first_set;
  std::map<uint64_t, uint64_t> second_set;
  for (const auto &edge : G->Edges()) {
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
bool IsGraphBipartite(const mg_graph::GraphView<> *G) {
  uint64_t node_size = G->Nodes().size();
  std::vector<int64_t> colors(node_size, -1);

  for (uint64_t i = 0; i < node_size; i++) {
    if (colors[i] == -1) {
      if (!bipartite_matching_alg::IsSubgraphBipartite(G, colors, i)) {
        return false;
      }
    }
  }

  return true;
}
}  // namespace bipartite_matching_alg
