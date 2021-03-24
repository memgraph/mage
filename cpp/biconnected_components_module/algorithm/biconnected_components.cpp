#include "biconnected_components.hpp"

void bcc_utility::BccDfs(uint32_t node_id, uint32_t parent_id,
                         bcc_utility::NodeState *state,
                         std::stack<mg_graph::Edge> *edge_stack,
                         std::vector<std::vector<mg_graph::Edge>> *bcc,
                         const mg_graph::GraphView *graph) {

  static int tick = 0;
  bool root = node_id == parent_id;
  int ch_cnt = 0; // needed to handle the special case for root node.

  state->visited[node_id] = true;
  tick++;
  state->discovery[node_id] = tick;
  state->low_link[node_id] = tick;

  for (const auto &neigh : graph->Neighbours(node_id)) {
    uint32_t next_id = neigh.node_id;
    const auto &edge = graph->GetEdge(neigh.edge_id);
    if (state->visited[next_id]) {
      if (next_id != parent_id) {
        state->low_link[node_id] =
            std::min(state->low_link[node_id], state->discovery[next_id]);
        if (state->discovery[next_id] < state->discovery[node_id])
          edge_stack->push(edge);
      }
      continue;
    }

    ++ch_cnt;
    edge_stack->push(edge);
    BccDfs(next_id, node_id, state, edge_stack, bcc, graph);
    state->low_link[node_id] =
        std::min(state->low_link[node_id], state->low_link[next_id]);

    // Articulation point check
    if ((root && ch_cnt > 1) || // special case for root
        (!root && state->low_link[next_id] >= state->discovery[node_id])) {
      // obsdata::Edges until (node, next) on stack form a BCC
      bcc->emplace_back();
      while (edge_stack->top().id != edge.id) {
        bcc->back().push_back(edge_stack->top());
        edge_stack->pop();
      }
      bcc->back().push_back(edge_stack->top());
      edge_stack->pop();
    }
  }
}

std::vector<std::vector<mg_graph::Edge>>
bcc_algorithm::GetBiconnectedComponents(const mg_graph::GraphView *graph) {
  bcc_utility::NodeState state;
  size_t node_size = graph->Nodes().size();
  state.visited.resize(node_size, false);
  state.discovery.resize(node_size, 0);
  state.low_link.resize(node_size, 0);

  std::vector<std::vector<mg_graph::Edge>> bcc;
  std::stack<mg_graph::Edge> edge_stack;

  for (const mg_graph::Node &node : graph->Nodes()) {
    if (state.visited[node.id]) {
      continue;
    }

    BccDfs(node.id, node.id, &state, &edge_stack, &bcc, graph);

    // Any edges left on stack form a BCC
    if (!edge_stack.empty()) {
      bcc.emplace_back();
      while (!edge_stack.empty()) {
        bcc.back().push_back(edge_stack.top());
        edge_stack.pop();
      }
    }
  }

  return bcc;
}
