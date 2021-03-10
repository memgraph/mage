#include <algorithm>
#include <cassert>

#include "algorithms/algorithms.hpp"
#include "algorithms/utils.hpp"

namespace {

void CompCntDfs(uint32_t node_id, const graphdata::GraphView &G,
                const std::vector<graphdata::Edge> &deleted_edges,
                algorithms::NodeState *state) {
  state->visited[node_id] = true;
  for (const auto &neigh : G.Neighbours(node_id)) {
    uint32_t next_id = neigh.node_id;
    if (state->visited[next_id] ||
        std::find_if(deleted_edges.begin(), deleted_edges.end(),
                     [&neigh](const graphdata::Edge &e) {
                       return neigh.edge_id == e.id;
                     }) != deleted_edges.end()) {
      continue;
    }
    CompCntDfs(next_id, G, deleted_edges, state);
  }
}

int ComponentCount(const graphdata::GraphView &G,
                   const std::vector<uint32_t> &component,
                   const std::vector<graphdata::Edge> &deleted_edges) {
  algorithms::NodeState state;
  size_t node_size = G.Nodes().size();
  state.visited.resize(node_size, false);

  int component_count = 0;
  for (uint32_t node_id : component) {
    if (state.visited[node_id]) continue;
    ++component_count;
    CompCntDfs(node_id, G, deleted_edges, &state);
  }

  return component_count;
}

void FindComponent(uint32_t node_id, const graphdata::GraphView &G,
                   algorithms::NodeState *state,
                   std::vector<uint32_t> *component) {
  state->visited[node_id] = true;
  component->emplace_back(node_id);
  for (const auto &neigh : G.Neighbours(node_id)) {
    uint32_t next_id = neigh.node_id;
    if (state->visited[next_id]) continue;
    FindComponent(next_id, G, state, component);
  }
}

void SolveComponent(const graphdata::GraphView &G,
                    const std::vector<uint32_t> &component,
                    std::vector<std::vector<graphdata::Edge>> *cutsets) {
  if (component.size() > 25) {
  }
  // obsio::LOG << "Number of nodes in a connected component (" <<
  // component.size() << ") is large.";

  // special case for a single node in connected component.
  if (component.size() == 1) return;

  for (int mask = 1; mask < (1 << component.size()); ++mask) {
    int complement = (1 << component.size()) - 1 - mask;
    if (mask > complement) continue;  // this prevents doubling of cutsets

    // Find cutset for a given mask.
    std::vector<graphdata::Edge> cutset;

    for (size_t i = 0; i < component.size(); ++i) {
      uint32_t node_id = component[i];
      for (const auto &neigh : G.Neighbours(node_id)) {
        uint32_t next_id = neigh.node_id;
        auto pos = std::find(component.begin(), component.end(), next_id);
        if (next_id < node_id || pos == component.end()) continue;
        int j = pos - component.begin();
        bool color_node = (mask & (1 << i)) > 0;
        bool color_next = (mask & (1 << j)) > 0;
        if (color_node != color_next)
          cutset.emplace_back(G.GetEdge(neigh.edge_id));
      }
    }

    assert(!cutset.empty() && "Cutset cannot be empty!");
    if (ComponentCount(G, component, cutset) == 2)
      cutsets->emplace_back(cutset);
  }
}

}  // namespace

namespace algorithms_bf {

std::vector<std::vector<graphdata::Edge>> GetCutsets(
    const graphdata::GraphView &G) {
  algorithms::NodeState state;
  size_t node_size = G.Nodes().size();
  state.visited.resize(node_size, false);

  std::vector<std::vector<graphdata::Edge>> cutsets;

  for (const graphdata::Node &node : G.Nodes()) {
    if (state.visited[node.id]) continue;
    std::vector<uint32_t> component;
    FindComponent(node.id, G, &state, &component);
    SolveComponent(G, component, &cutsets);
  }

  return cutsets;
}

}  // namespace algorithms_bf
