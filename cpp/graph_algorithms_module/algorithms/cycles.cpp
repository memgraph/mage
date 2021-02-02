#include <algorithm>
#include <cassert>
#include <set>

#include "algorithms/algorithms.hpp"
#include "algorithms/utils.hpp"

namespace {

graphdata::Node NodeFromId(uint32_t node_id, const graphdata::GraphView &G) {
  for (const auto &node : G.Nodes())
    if (node.id == node_id) return node;
  assert(false && "Couldn't find node with a given id");
  return graphdata::Node();
}

std::vector<uint32_t> FindCycle(uint32_t a, uint32_t b,
                                const algorithms::NodeState &state) {
  std::vector<uint32_t> cycle;
  if (state.depth[a] < state.depth[b]) std::swap(a, b);

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

  assert(a == b &&
         "There should be no cross edges in DFS tree of an undirected graph.");

  cycle.push_back(cycle[0]);
  return cycle;
}

void FindNonSTEdges(uint32_t node_id, const graphdata::GraphView &G,
                    algorithms::NodeState *state,
                    std::set<std::pair<uint32_t, uint32_t>> *non_st_edges) {
  std::set<uint32_t> unique_neighbour;
  state->visited[node_id] = true;
  for (const auto &neigh : G.Neighbours(node_id)) {
    uint32_t next_id = neigh.node_id;
    if (next_id == state->parent[node_id]) continue;
    if (unique_neighbour.find(next_id) != unique_neighbour.end()) continue;
    unique_neighbour.insert(next_id);
    if (state->visited[next_id]) {
      std::pair<uint32_t, uint32_t> e = {std::min(node_id, next_id),
                                         std::max(node_id, next_id)};
      non_st_edges->insert(e);
      continue;
    }
    state->parent[next_id] = node_id;
    state->depth[next_id] = state->depth[node_id] + 1;
    FindNonSTEdges(next_id, G, state, non_st_edges);
  }
}

void FindFundamentalCycles(
    const std::set<std::pair<uint32_t, uint32_t>> &non_st_edges,
    const algorithms::NodeState &state,
    std::vector<std::vector<uint32_t>> *fundamental_cycles) {
  for (const auto &edge : non_st_edges)
    fundamental_cycles->emplace_back(FindCycle(edge.first, edge.second, state));
}

void SolveMask(int mask,
               const std::vector<std::vector<uint32_t>> &fundamental_cycles,
               const graphdata::GraphView &G,
               std::vector<std::vector<graphdata::Node>> *cycles) {
  std::map<std::pair<uint32_t, uint32_t>, int32_t> edge_cnt;
  for (int i = 0; i < static_cast<int>(fundamental_cycles.size()); ++i) {
    if ((mask & (1 << i)) == 0) continue;
    for (int j = 1; j < static_cast<int>(fundamental_cycles[i].size()); ++j) {
      std::pair<uint32_t, uint32_t> edge = {
          std::min(fundamental_cycles[i][j], fundamental_cycles[i][j - 1]),
          std::max(fundamental_cycles[i][j], fundamental_cycles[i][j - 1])};
      edge_cnt[edge]++;
    }
  }

  std::map<uint32_t, std::vector<uint32_t>> adj_list;
  std::map<uint32_t, bool> visited;
  std::set<uint32_t> nodes;

  for (const auto &kv : edge_cnt) {
    if (kv.second % 2 == 0) continue;
    auto edge = kv.first;
    adj_list[edge.first].push_back(edge.second);
    adj_list[edge.second].push_back(edge.first);
    nodes.insert(edge.first);
    nodes.insert(edge.second);
  }

  // deg(v) = 2 for all vertices in a cycle.
  for (uint32_t node : nodes)
    if (adj_list[node].size() != 2) return;

  uint32_t curr_node = *nodes.begin();
  std::vector<graphdata::Node> cycle;
  while (!visited[curr_node]) {
    cycle.push_back(NodeFromId(curr_node, G));
    visited[curr_node] = true;
    for (uint32_t next_node : adj_list[curr_node]) {
      if (!visited[next_node]) curr_node = next_node;
    }
  }

  bool ok = true;
  for (uint32_t node : nodes)
    if (!visited[node]) ok = false;

  if (ok && cycle.size() > 2) cycles->emplace_back(cycle);
}

void GetCyclesFromFundamentals(
    const std::vector<std::vector<uint32_t>> &fundamental_cycles,
    const graphdata::GraphView &G,
    std::vector<std::vector<graphdata::Node>> *cycles) {
  if (fundamental_cycles.size() > 25) {
  }
  // obsio::LOG << "Number of fundamental cycles (" << fundamental_cycles.size()
  // << ") is large.";

  // find cycles obtained from xoring each subset of fundamental cycles.
  for (int mask = 1; mask < (1 << fundamental_cycles.size()); ++mask)
    SolveMask(mask, fundamental_cycles, G, cycles);
}

}  // namespace

namespace algorithms {

std::vector<std::vector<graphdata::Node>> GetCycles(const graphdata::GraphView &G) {
  NodeState state;
  size_t node_size = G.Nodes().size();
  state.visited.resize(node_size, false);
  state.parent.resize(node_size, 0);
  state.depth.resize(node_size, 0);

  std::vector<std::vector<graphdata::Node>> cycles;

  // Solve for each connected component
  // TODO(ipaljak) solve for each biconnected component
  for (const graphdata::Node &node : G.Nodes()) {
    if (state.visited[node.id]) continue;

    // First we find edges that do not lie on a DFS tree (basically,
    // backedges and crossedges). From those edges we expand into the
    // spanning tree to obtain all fundamental cycles.
    std::set<std::pair<uint32_t, uint32_t>> non_st_edges;
    state.parent[node.id] = node.id;
    FindNonSTEdges(node.id, G, &state, &non_st_edges);

    std::vector<std::vector<uint32_t>> fundamental_cycles;
    FindFundamentalCycles(non_st_edges, state, &fundamental_cycles);

    GetCyclesFromFundamentals(fundamental_cycles, G, &cycles);
  }

  return cycles;
}

std::vector<std::pair<graphdata::Node, graphdata::Node>> GetNeighbourCycles(
    const graphdata::GraphView &G) {
  std::vector<std::pair<graphdata::Node, graphdata::Node>> cycles;
  std::set<std::pair<uint32_t, uint32_t>> multi_edges;
  for (const graphdata::Edge &edge : G.Edges()) {
    uint32_t from = std::min(edge.from, edge.to);
    uint32_t to = std::max(edge.from, edge.to);
    std::pair<uint32_t, uint32_t> e = {from, to};
    if (multi_edges.find(e) != multi_edges.end()) continue;
    multi_edges.insert(e);
    const auto &edges = G.GetEdgesBetweenNodes(from, to);
    if (edges.size() > 1) cycles.push_back({G.GetNode(from), G.GetNode(to)});
  }
  return cycles;
}

}  // namespace algorithms
