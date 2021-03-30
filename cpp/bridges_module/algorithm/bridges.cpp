#include <algorithm>

#include "bridges.hpp"

void bridges_util::BridgeDfs(uint64_t node_id, uint64_t parent_id, bridges_util::NodeState *state,
                             std::vector<mg_graph::Edge<>> *bridges, const mg_graph::GraphView<> *G) {
  state->counter++;
  state->visited[node_id] = true;
  state->discovery[node_id] = state->counter;
  state->low_link[node_id] = state->counter;

  for (const auto &neigh : G->Neighbours(node_id)) {
    uint32_t next_id = neigh.node_id;
    const auto &edge = G->GetEdge(neigh.edge_id);
    if (state->visited[next_id]) {
      if (next_id != parent_id) {
        state->low_link[node_id] = std::min(state->low_link[node_id], state->discovery[next_id]);
      }
      continue;
    }

    bridges_util::BridgeDfs(next_id, node_id, state, bridges, G);
    state->low_link[node_id] = std::min(state->low_link[node_id], state->low_link[next_id]);

    if (state->low_link[next_id] > state->discovery[node_id]) {
      if (G->GetEdgesBetweenNodes(edge.from, edge.to).size() == 1) bridges->push_back(edge);
    }
  }
}

std::vector<mg_graph::Edge<>> bridges_alg::GetBridges(const mg_graph::GraphView<> *G) {
  bridges_util::NodeState state;
  size_t node_size = G->Nodes().size();
  state.visited.resize(node_size, false);
  state.discovery.resize(node_size, 0);
  state.low_link.resize(node_size, 0);

  std::vector<mg_graph::Edge<>> bridges;
  for (const auto &node : G->Nodes()) {
    if (!state.visited[node.id]) {
      BridgeDfs(node.id, node.id, &state, &bridges, G);
    }
  }
  return bridges;
}
