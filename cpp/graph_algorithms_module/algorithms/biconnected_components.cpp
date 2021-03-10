#include <algorithm>
#include <stack>

#include "algorithms/algorithms.hpp"
#include "algorithms/utils.hpp"

namespace {

void BCCDfs(uint32_t node_id, uint32_t parent_id, algorithms::NodeState *state,
            std::stack<graphdata::Edge> *edge_stack,
            std::vector<std::vector<graphdata::Edge>> *BCC,
            const graphdata::GraphView &G) {
  static int tick = 0;
  bool root = node_id == parent_id;
  int ch_cnt = 0;  // needed to handle the special case for root node.
  
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
        if (state->discovery[next_id] < state->discovery[node_id])
          edge_stack->push(edge);
      }
      continue;
    }

    ++ch_cnt;
    edge_stack->push(edge);
    BCCDfs(next_id, node_id, state, edge_stack, BCC, G);
    state->low_link[node_id] =
        std::min(state->low_link[node_id], state->low_link[next_id]);

    // Articulation point check
    if ((root && ch_cnt > 1) ||  // special case for root
        (!root && state->low_link[next_id] >= state->discovery[node_id])) {
      // obsdata::Edges until (node, next) on stack form a BCC
      BCC->emplace_back();
      while (edge_stack->top().id != edge.id) {
        BCC->back().push_back(edge_stack->top());
        edge_stack->pop();
      }
      BCC->back().push_back(edge_stack->top());
      edge_stack->pop();
    }
  }
}

}  // namespace

namespace algorithms {

std::vector<std::vector<graphdata::Edge>> GetBiconnectedComponents(
    const graphdata::GraphView &G) {
  NodeState state;
  size_t node_size = G.Nodes().size();
  state.visited.resize(node_size, false);
  state.discovery.resize(node_size, 0);
  state.low_link.resize(node_size, 0);

  std::vector<std::vector<graphdata::Edge>> BCC;
  std::stack<graphdata::Edge> edge_stack;

  for (const graphdata::Node &node : G.Nodes()) {
    if (state.visited[node.id]) continue;
    BCCDfs(node.id, node.id, &state, &edge_stack, &BCC, G);
    // Any edges left on stack form a BCC
    if (!edge_stack.empty()) {
      BCC.emplace_back();
      while (!edge_stack.empty()) {
        BCC.back().push_back(edge_stack.top());
        edge_stack.pop();
      }
    }
  }

  return BCC;
}

}  // namespace algorithms
