#include "bcc_utility.hpp"

bcc_utility::NodeState::NodeState(std::uint64_t number_of_nodes) {
  visited.reserve(number_of_nodes);
  discovery.reserve(number_of_nodes);
  low_link.reserve(number_of_nodes);
  counter = 0;
}

void bcc_utility::NodeState::Update(std::uint64_t node_id) {
  counter++;
  visited[node_id] = true;
  discovery[node_id] = counter;
  low_link[node_id] = counter;
}

void bcc_utility::BccDFS(std::uint64_t node_id, std::uint64_t parent_id, bcc_utility::NodeState *state,
                         std::stack<mg_graph::Edge<>> *edge_stack,
                         std::vector<std::vector<mg_graph::Edge<>>> *bcc_edges,
                         std::vector<std::unordered_set<std::uint64_t>> *bcc_nodes, const mg_graph::GraphView<> &graph,
                         std::unordered_set<uint64_t> &articulationPoints) {
  auto root = node_id == parent_id;

  state->Update(node_id);

  std::uint64_t root_count = 0;  // needed to handle the special case for root node.

  for (const auto &neigh : graph.Neighbours(node_id)) {
    auto next_id = neigh.node_id;

    const auto &edge = graph.GetEdge(neigh.edge_id);
    if (state->visited[next_id]) {
      if (next_id != parent_id) {
        state->low_link[node_id] = std::min(state->low_link[node_id], state->discovery[next_id]);
        if (state->discovery[next_id] < state->discovery[node_id]) edge_stack->push(edge);
      }
      continue;
    }

    ++root_count;
    edge_stack->push(edge);
    BccDFS(next_id, node_id, state, edge_stack, bcc_edges, bcc_nodes, graph, articulationPoints);
    state->low_link[node_id] = std::min(state->low_link[node_id], state->low_link[next_id]);

    // Articulation point check
    if (((root && root_count > 1) ||  // special case for root
         (!root && state->low_link[next_id] >= state->discovery[node_id])) &&
        !edge_stack->empty()) {
      // get articulation point
      articulationPoints.insert(node_id);

      bcc_edges->emplace_back();
      bcc_nodes->emplace_back();
      while (edge_stack->top().id != edge.id) {
        bcc_edges->back().push_back(edge_stack->top());
        bcc_nodes->back().insert(edge_stack->top().from);
        bcc_nodes->back().insert(edge_stack->top().to);
        edge_stack->pop();
      }
      bcc_edges->back().push_back(edge_stack->top());
      bcc_nodes->back().insert(edge_stack->top().from);
      bcc_nodes->back().insert(edge_stack->top().to);
      edge_stack->pop();
    }
  }
}

std::vector<std::unordered_set<std::uint64_t>> bcc_algorithm::GetBiconnectedComponents(
    const mg_graph::GraphView<> &graph, std::unordered_set<uint64_t> &articulationPoints,
    std::vector<std::vector<mg_graph::Edge<>>> &bcc_edges) {
  auto number_of_nodes = graph.Nodes().size();
  bcc_utility::NodeState state(number_of_nodes);

  std::vector<std::unordered_set<std::uint64_t>> bcc_nodes;
  std::stack<mg_graph::Edge<>> edge_stack;

  for (const auto &node : graph.Nodes()) {
    if (state.visited[node.id]) {
      continue;
    }

    BccDFS(node.id, node.id, &state, &edge_stack, &bcc_edges, &bcc_nodes, graph, articulationPoints);

    // Any edges left on stack form a BCC
    if (!edge_stack.empty()) {
      bcc_edges.emplace_back();
      bcc_nodes.emplace_back();
      while (!edge_stack.empty()) {
        bcc_edges.back().push_back(edge_stack.top());
        bcc_nodes.back().insert(edge_stack.top().from);
        bcc_nodes.back().insert(edge_stack.top().to);
        edge_stack.pop();
      }
    }
  }

  return bcc_nodes;
}
