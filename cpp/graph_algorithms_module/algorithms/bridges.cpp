#include <algorithm>

#include "algorithms/algorithms.hpp"
#include "algorithms/utils.hpp"

namespace {

void BridgeDfs(uint32_t node_id, uint32_t parent_id,
               algorithms::NodeState *state,
               std::vector<graphdata::Edge> *bridges,
               const graphdata::GraphView &G) {
  static int tick = 0;
  state->visited[node_id] = true;
  tick++;
  state->discovery[node_id] = tick;
  state->low_link[node_id] = tick;

  for (const auto &neigh : G.Neighbours(node_id)) {
    uint32_t next_id = neigh.node_id;
    const auto &edge = G.GetEdge(neigh.edge_id);
    if (state->visited[next_id]) {
      if (next_id != parent_id) {
        state->low_link[node_id] =
            std::min(state->low_link[node_id], state->discovery[next_id]);
      }
      continue;
    }

    BridgeDfs(next_id, node_id, state, bridges, G);
    state->low_link[node_id] =
        std::min(state->low_link[node_id], state->low_link[next_id]);

    if (state->low_link[next_id] > state->discovery[node_id]) {
      if (G.GetEdgesBetweenNodes(edge.from, edge.to).size() == 1)
        bridges->push_back(edge);
    }
  }
}

}  // namespace

namespace algorithms {

std::vector<graphdata::Edge> GetBridges(const graphdata::GraphView &G) {
  NodeState state;
  size_t node_size = G.Nodes().size();
  state.visited.resize(node_size, false);
  state.discovery.resize(node_size, 0);
  state.low_link.resize(node_size, 0);

  std::vector<graphdata::Edge> bridges;
  for (const graphdata::Node &node : G.Nodes()) {
    if (!state.visited[node.id]) {
      BridgeDfs(node.id, node.id, &state, &bridges, G);
    }
  }
  return bridges;
}

}  // namespace algorithms
