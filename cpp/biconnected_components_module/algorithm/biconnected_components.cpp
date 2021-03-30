#include "biconnected_components.hpp"

void bcc_utility::BccDfs(uint64_t node_id, uint64_t parent_id, bcc_utility::NodeState *state,
                         std::stack<mg_graph::Edge<>> *edge_stack, std::vector<std::vector<mg_graph::Edge<>>> *bcc,
                         const mg_graph::GraphView<> *graph) {
  bool root = node_id == parent_id;
  uint32_t root_count = 0;  // needed to handle the special case for root node.

  state->counter++;
  state->visited[node_id] = true;
  state->discovery[node_id] = state->counter;
  state->low_link[node_id] = state->counter;

  for (const auto &neigh : graph->Neighbours(node_id)) {
    auto next_id = neigh.node_id;

    const auto &edge = graph->GetEdge(neigh.edge_id);
    if (state->visited[next_id]) {
      if (next_id != parent_id) {
        state->low_link[node_id] = std::min(state->low_link[node_id], state->discovery[next_id]);
        if (state->discovery[next_id] < state->discovery[node_id]) edge_stack->push(edge);
      }
      continue;
    }

    ++root_count;
    edge_stack->push(edge);
    BccDfs(next_id, node_id, state, edge_stack, bcc, graph);
    state->low_link[node_id] = std::min(state->low_link[node_id], state->low_link[next_id]);

    // Articulation point check
    if ((root && root_count > 1) ||  // special case for root
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

std::vector<std::vector<mg_graph::Edge<>>> bcc_algorithm::GetBiconnectedComponents(const mg_graph::GraphView<> *graph) {
  bcc_utility::NodeState state;
  size_t node_size = graph->Nodes().size();
  state.visited.resize(node_size, false);
  state.discovery.resize(node_size, 0);
  state.low_link.resize(node_size, 0);
  state.counter = 0;

  std::vector<std::vector<mg_graph::Edge<>>> bcc;
  std::stack<mg_graph::Edge<>> edge_stack;

  for (const auto &node : graph->Nodes()) {
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
