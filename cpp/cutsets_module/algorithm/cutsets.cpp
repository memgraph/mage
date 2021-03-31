#include <algorithm>
#include <set>
#include <unordered_set>

#include "cutsets.hpp"

namespace cutsets_util {

void ComponentCountDFS(const uint64_t node_id, const mg_graph::GraphView<> *G, const std::vector<uint64_t> &component,
                       cutsets_util::NodeState *state) {
  state->visited[node_id] = true;
  for (const auto &neigh : G->Neighbours(node_id)) {
    auto next_id = neigh.node_id;
    if (state->visited[next_id] || std::find(component.begin(), component.end(), next_id) == component.end()) {
      continue;
    }
    ComponentCountDFS(next_id, G, component, state);
  }
}

uint64_t ComponentCount(const mg_graph::GraphView<> *G, const std::vector<uint64_t> &component) {
  cutsets_util::NodeState state;
  state.visited.resize(G->Nodes().size(), false);

  int component_count = 0;
  for (auto node_id : component) {
    if (state.visited[node_id]) continue;
    ++component_count;
    ComponentCountDFS(node_id, G, component, &state);
  }

  return component_count;
}

std::set<uint64_t> OmegaSet(const std::vector<uint64_t> &U, const std::vector<uint64_t> &V,
                            const mg_graph::GraphView<> *G) {
  std::set<uint64_t> ret;
  for (auto node_id : U) {
    for (const auto &neigh : G->Neighbours(node_id)) {
      auto next_id = neigh.node_id;
      if (std::find(V.begin(), V.end(), next_id) == V.end()) continue;
      ret.insert(neigh.edge_id);
    }
  }
  return ret;
}

void FindComponentOfSubgraph(const uint64_t node_id, const mg_graph::GraphView<> *G,
                             const std::vector<uint64_t> &subgraph, cutsets_util::NodeState *state,
                             std::vector<uint64_t> *component) {
  state->visited[node_id] = true;
  component->emplace_back(node_id);
  for (const auto &neigh : G->Neighbours(node_id)) {
    auto next_id = neigh.node_id;
    if (state->visited[next_id] || std::find(subgraph.begin(), subgraph.end(), next_id) == subgraph.end()) continue;
    FindComponentOfSubgraph(next_id, G, subgraph, state, component);
  }
}

void FindComponent(const uint64_t node_id, const mg_graph::GraphView<> *G, cutsets_util::NodeState *state,
                   std::vector<uint64_t> *component) {
  state->visited[node_id] = true;
  component->emplace_back(node_id);
  for (const auto &neigh : G->Neighbours(node_id)) {
    auto next_id = neigh.node_id;
    if (state->visited[next_id]) continue;
    FindComponent(next_id, G, state, component);
  }
}

void Backtrack(const mg_graph::GraphView<> *G, const std::vector<uint64_t> &component, const std::vector<uint64_t> &S,
               const std::vector<uint64_t> &T, uint64_t s, uint64_t t, std::set<std::set<uint64_t>> *cutsets) {
  std::set<uint64_t> S_neigh;  // Neighbourhood of S excluding T
  for (auto node_id : S) {
    for (const auto &neigh : G->Neighbours(node_id)) {
      auto next_id = neigh.node_id;
      if (std::find(S.begin(), S.end(), next_id) != S.end() || std::find(T.begin(), T.end(), next_id) != T.end()) {
        continue;
      }
      S_neigh.insert(next_id);
    }
  }

  std::vector<uint64_t> not_S;
  for (auto node_id : component) {
    if (std::find(S.begin(), S.end(), node_id) == S.end()) not_S.push_back(node_id);
  }

  if (S_neigh.empty()) {
    cutsets->insert(OmegaSet(S, not_S, G));
    return;
  }

  auto node_id = *S_neigh.begin();
  std::vector<uint64_t> comp;  // G[not_S - {node_id}]
  for (auto v : not_S)
    if (v != node_id) comp.push_back(v);

  if (ComponentCount(G, comp) == 1) {
    auto new_S = S;
    new_S.push_back(node_id);
    Backtrack(G, component, new_S, T, s, t, cutsets);
  } else {
    std::vector<uint64_t> subgraph;  // not_S - node_id
    for (auto v : not_S) {
      if (v != node_id) subgraph.push_back(v);
    }

    cutsets_util::NodeState state;
    state.visited.resize(G->Nodes().size(), false);
    std::vector<uint64_t> W, not_W;
    FindComponentOfSubgraph(t, G, subgraph, &state, &W);

    if (IsSubset(T, W)) {
      for (auto node_id : component) {
        if (std::find(W.begin(), W.end(), node_id) == W.end()) not_W.push_back(node_id);
      }
      Backtrack(G, component, not_W, T, s, t, cutsets);
    }
  }
  auto new_T = T;
  new_T.push_back(node_id);
  Backtrack(G, component, S, new_T, s, t, cutsets);
}

void SolveComponent(const mg_graph::GraphView<> *G, const std::vector<uint64_t> &component,
                    std::set<std::set<uint64_t>> *cutsets) {
  for (uint64_t i = 0; i < component.size(); ++i) {
    for (uint64_t j = i + 1; j < component.size(); ++j) {
      cutsets_alg::CutsetsAlgorithm(G, component, component[i], component[j], cutsets);
    }
  }
}

}  // namespace cutsets_util

namespace cutsets_alg {

void CutsetsAlgorithm(const mg_graph::GraphView<> *G, const std::vector<uint64_t> &component, uint64_t s, uint64_t t,
                      std::set<std::set<uint64_t>> *cutsets) {
  std::vector<uint64_t> subgraph;  // V - {s}
  for (auto node_id : component) {
    if (node_id != s) subgraph.push_back(node_id);
  }

  cutsets_util::NodeState state;
  state.visited.resize(G->Nodes().size(), false);

  std::vector<uint64_t> W, not_W;
  FindComponentOfSubgraph(t, G, subgraph, &state, &W);

  for (auto node_id : component) {
    if (std::find(W.begin(), W.end(), node_id) == W.end()) not_W.push_back(node_id);
  }

  cutsets_util::Backtrack(G, component, not_W, {t}, s, t, cutsets);
}

std::vector<std::vector<mg_graph::Edge<>>> GetCutsets(const mg_graph::GraphView<> *G) {
  cutsets_util::NodeState state;
  size_t node_size = G->Nodes().size();
  state.visited.resize(node_size, false);

  std::set<std::set<uint64_t>> cutsets;  // edge_id
  for (const auto &node : G->Nodes()) {
    if (state.visited[node.id]) continue;
    std::vector<uint64_t> component;
    cutsets_util::FindComponent(node.id, G, &state, &component);
    cutsets_util::SolveComponent(G, component, &cutsets);
  }

  std::vector<std::vector<mg_graph::Edge<>>> cutset_edges;
  for (const auto &cutset : cutsets) {
    std::vector<mg_graph::Edge<>> ret_cutset;
    for (const auto &edge_id : cutset) {
      ret_cutset.push_back(G->Edges()[edge_id]);
    }
    cutset_edges.push_back(ret_cutset);
  }

  return cutset_edges;
}

}  // namespace cutsets_alg
