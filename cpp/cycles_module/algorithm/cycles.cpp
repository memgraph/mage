#include <algorithm>
#include <cassert>
#include <set>
#include <unordered_set>

#include <mg_graph.hpp>

#include "cycles.hpp"

namespace cycles_util {

mg_graph::Node<> NodeFromId(uint64_t node_id, const mg_graph::GraphView<> *G) {
  for (const auto &node : G->Nodes())
    if (node.id == node_id) return node;
  assert(false && "Couldn't find node with a given id");
  return mg_graph::Node();
}

std::vector<uint64_t> FindCycle(uint64_t a, uint64_t b, const cycles_util::NodeState &state) {
  std::vector<uint64_t> cycle;
  if (state.depth[a] < state.depth[b]) {
    std::swap(a, b);
  }

  // climb until a and b reach the same depth:
  //
  //       ()               ()
  //      /  \             /  \.
  //    ()   (b)   -->   (a)  (b)
  //   /  \             /   \.
  // (a)  ()           ()   ()
  cycle.push_back(a);
  while (state.depth[a] > state.depth[b]) {
    a = state.parent[a];
    cycle.push_back(a);
  }

  assert(a == b && "There should be no cross edges in DFS tree of an undirected graph.");

  cycle.push_back(cycle[0]);
  return cycle;
}

void FindNonSTEdges(uint64_t node_id, const mg_graph::GraphView<> *G, cycles_util::NodeState *state,
                    std::set<std::pair<uint64_t, uint64_t>> *non_st_edges) {
  std::unordered_set<uint64_t> unique_neighbour;
  state->visited[node_id] = true;
  for (const auto &neigh : G->Neighbours(node_id)) {
    auto next_id = neigh.node_id;
    if (next_id == state->parent[node_id]) continue;
    if (unique_neighbour.find(next_id) != unique_neighbour.end()) continue;
    unique_neighbour.insert(next_id);

    if (state->visited[next_id]) {
      auto e = std::minmax(node_id, next_id);
      non_st_edges->insert(e);
      continue;
    }
    state->parent[next_id] = node_id;
    state->depth[next_id] = state->depth[node_id] + 1;
    cycles_util::FindNonSTEdges(next_id, G, state, non_st_edges);
  }
}

void FindFundamentalCycles(const std::set<std::pair<uint64_t, uint64_t>> &non_st_edges,
                           const cycles_util::NodeState &state,
                           std::vector<std::vector<uint64_t>> *fundamental_cycles) {
  for (const auto &edge : non_st_edges) {
    fundamental_cycles->emplace_back(cycles_util::FindCycle(edge.first, edge.second, state));
  }
}

void SolveMask(int mask, const std::vector<std::vector<uint64_t>> &fundamental_cycles, const mg_graph::GraphView<> *G,
               std::vector<std::vector<mg_graph::Node<>>> *cycles) {
  std::map<std::pair<uint64_t, uint64_t>, uint64_t> edge_cnt;
  for (int i = 0; i < static_cast<int>(fundamental_cycles.size()); ++i) {
    if ((mask & (1 << i)) == 0) continue;
    for (int j = 1; j < static_cast<int>(fundamental_cycles[i].size()); ++j) {
      auto edge = std::minmax(fundamental_cycles[i][j], fundamental_cycles[i][j - 1]);
      edge_cnt[edge]++;
    }
  }

  std::map<uint64_t, std::vector<uint64_t>> adj_list;
  std::map<uint64_t, bool> visited;
  std::set<uint64_t> nodes;

  for (const auto &[key, value] : edge_cnt) {
    if (value % 2 == 0) continue;

    auto const [from, to] = key;
    adj_list[from].push_back(to);
    adj_list[to].push_back(from);
    nodes.insert(from);
    nodes.insert(to);
  }

  // deg(v) = 2 for all vertices in a cycle.
  for (auto node : nodes)
    if (adj_list[node].size() != 2) return;

  auto curr_node = *nodes.begin();
  std::vector<mg_graph::Node<>> cycle;
  while (!visited[curr_node]) {
    cycle.push_back(NodeFromId(curr_node, G));
    visited[curr_node] = true;
    for (auto next_node : adj_list[curr_node]) {
      if (!visited[next_node]) curr_node = next_node;
    }
  }

  for (auto node : nodes) {
    if (!visited[node]) return;
  }

  if (cycle.size() > 2) cycles->emplace_back(cycle);
}

void GetCyclesFromFundamentals(const std::vector<std::vector<uint64_t>> &fundamental_cycles,
                               const mg_graph::GraphView<> *G, std::vector<std::vector<mg_graph::Node<>>> *cycles) {
  // find cycles obtained from xoring each subset of fundamental cycles.
  for (int mask = 1; mask < (1 << fundamental_cycles.size()); ++mask) {
    cycles_util::SolveMask(mask, fundamental_cycles, G, cycles);
  }
}

}  // namespace cycles_util

namespace cycles_alg {

std::vector<std::vector<mg_graph::Node<>>> GetCycles(const mg_graph::GraphView<> *G) {
  cycles_util::NodeState state;
  size_t node_size = G->Nodes().size();
  state.visited.resize(node_size, false);
  state.parent.resize(node_size, 0);
  state.depth.resize(node_size, 0);

  std::vector<std::vector<mg_graph::Node<>>> cycles;

  // Solve for each connected component
  // TODO(josipmrden) solve for each biconnected component
  for (const auto &node : G->Nodes()) {
    if (state.visited[node.id]) continue;

    // First we find edges that do not lie on a DFS tree (basically,
    // backedges and crossedges). From those edges we expand into the
    // spanning tree to obtain all fundamental cycles.
    std::set<std::pair<uint64_t, uint64_t>> non_st_edges;
    state.parent[node.id] = node.id;
    cycles_util::FindNonSTEdges(node.id, G, &state, &non_st_edges);

    std::vector<std::vector<uint64_t>> fundamental_cycles;
    cycles_util::FindFundamentalCycles(non_st_edges, state, &fundamental_cycles);

    cycles_util::GetCyclesFromFundamentals(fundamental_cycles, G, &cycles);
  }

  return cycles;
}

std::vector<std::pair<mg_graph::Node<>, mg_graph::Node<>>> GetNeighbourCycles(const mg_graph::GraphView<> *G) {
  std::vector<std::pair<mg_graph::Node<>, mg_graph::Node<>>> cycles;
  std::set<std::pair<uint64_t, uint64_t>> multi_edges;
  for (const auto &edge : G->Edges()) {
    const auto [from, to] = std::minmax(edge.from, edge.to);
    std::pair<uint64_t, uint64_t> e = {from, to};
    if (multi_edges.find(e) != multi_edges.end()) {
      continue;
    }
    multi_edges.insert(e);
    const auto &edges = G->GetEdgesBetweenNodes(from, to);
    if (edges.size() > 1) cycles.push_back({G->GetNode(from), G->GetNode(to)});
  }
  return cycles;
}

}  // namespace cycles_alg
