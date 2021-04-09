#include <algorithm>

#include "bridges.hpp"

void bridges_util::BridgeDfs(std::uint64_t node_id, std::uint64_t parent_id, bridges_util::NodeState *state,
                             std::vector<mg_graph::Edge<>> *bridges, const mg_graph::GraphView<> &graph) {
  state->Update(node_id);

  for (const auto &neigh : graph.Neighbours(node_id)) {
    auto next_id = neigh.node_id;
    const auto &edge = graph.GetEdge(neigh.edge_id);
    if (state->visited[next_id]) {
      if (next_id != parent_id) {
        state->low_link[node_id] = std::min(state->low_link[node_id], state->discovery[next_id]);
      }
      continue;
    }

    bridges_util::BridgeDfs(next_id, node_id, state, bridges, graph);
    state->low_link[node_id] = std::min(state->low_link[node_id], state->low_link[next_id]);

    if (state->low_link[next_id] > state->discovery[node_id]) {
      if (graph.GetEdgesBetweenNodes(edge.from, edge.to).size() == 1) bridges->push_back(edge);
    }
  }
}

std::vector<mg_graph::Edge<>> bridges_alg::GetBridges(const mg_graph::GraphView<> &graph) {
  auto number_of_nodes = graph.Nodes().size();
  bridges_util::NodeState state(number_of_nodes);

  std::vector<mg_graph::Edge<>> bridges;
  for (const auto &node : graph.Nodes()) {
    if (!state.visited[node.id]) {
      BridgeDfs(node.id, node.id, &state, &bridges, graph);
    }
  }
  return bridges;
}
