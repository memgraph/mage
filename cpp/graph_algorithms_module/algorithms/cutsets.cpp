#include <algorithm>
#include <set>
#include <unordered_set>

#include "algorithms/algorithms.hpp"
#include "algorithms/utils.hpp"

namespace {

void ComponentCountDFS(uint32_t node_id, const graphdata::GraphView &G,
                       const std::vector<uint32_t> &component,
                       algorithms::NodeState *state) {
  state->visited[node_id] = true;
  for (const auto &neigh : G.Neighbours(node_id)) {
    uint32_t next_id = neigh.node_id;
    if (state->visited[next_id] || std::find(component.begin(), component.end(),
                                             next_id) == component.end()) {
      continue;
    }
    ComponentCountDFS(next_id, G, component, state);
  }
}

int ComponentCount(const graphdata::GraphView &G,
                   const std::vector<uint32_t> &component) {
  algorithms::NodeState state;
  state.visited.resize(G.Nodes().size(), false);

  int component_count = 0;
  for (uint32_t node_id : component) {
    if (state.visited[node_id]) continue;
    ++component_count;
    ComponentCountDFS(node_id, G, component, &state);
  }

  return component_count;
}

std::set<uint32_t> omega(const std::vector<uint32_t> &U,
                         const std::vector<uint32_t> &V,
                         const graphdata::GraphView &G) {
  std::set<uint32_t> ret;
  for (uint32_t node_id : U) {
    for (const auto &neigh : G.Neighbours(node_id)) {
      uint32_t next_id = neigh.node_id;
      if (std::find(V.begin(), V.end(), next_id) == V.end()) continue;
      ret.insert(neigh.edge_id);
    }
  }
  return ret;
}

bool subset(const std::vector<uint32_t> &A, const std::vector<uint32_t> &B) {
  std::unordered_set<uint32_t> AUB;
  AUB.insert(A.cbegin(), A.cend());
  AUB.insert(B.cbegin(), B.cend());
  return AUB.size() == B.size();
}

void FindComponentOfSubgraph(uint32_t node_id, const graphdata::GraphView &G,
                             const std::vector<uint32_t> &subgraph,
                             algorithms::NodeState *state,
                             std::vector<uint32_t> *component) {
  state->visited[node_id] = true;
  component->emplace_back(node_id);
  for (const auto &neigh : G.Neighbours(node_id)) {
    uint32_t next_id = neigh.node_id;
    if (state->visited[next_id] ||
        std::find(subgraph.begin(), subgraph.end(), next_id) == subgraph.end())
      continue;
    FindComponentOfSubgraph(next_id, G, subgraph, state, component);
  }
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

void Backtrack(const graphdata::GraphView &G,
               const std::vector<uint32_t> &component,
               const std::vector<uint32_t> &S, const std::vector<uint32_t> &T,
               uint32_t s, uint32_t t, std::set<std::set<uint32_t>> *cutsets) {
  std::set<uint32_t> S_neigh;  // Neighbourhood of S excluding T
  for (uint32_t node_id : S) {
    for (const auto &neigh : G.Neighbours(node_id)) {
      uint32_t next_id = neigh.node_id;
      if (std::find(S.begin(), S.end(), next_id) != S.end() ||
          std::find(T.begin(), T.end(), next_id) != T.end()) {
        continue;
      }
      S_neigh.insert(next_id);
    }
  }

  std::vector<uint32_t> not_S;
  for (uint32_t node_id : component) {
    if (std::find(S.begin(), S.end(), node_id) == S.end())
      not_S.push_back(node_id);
  }

  if (S_neigh.empty()) {
    cutsets->insert(omega(S, not_S, G));
    return;
  }

  uint32_t node_id = *S_neigh.begin();
  std::vector<uint32_t> comp;  // G[not_S - {node_id}]
  for (uint32_t v : not_S)
    if (v != node_id) comp.push_back(v);

  if (ComponentCount(G, comp) == 1) {
    std::vector<uint32_t> new_S = S;
    new_S.push_back(node_id);
    Backtrack(G, component, new_S, T, s, t, cutsets);
  } else {
    std::vector<uint32_t> subgraph;  // not_S - node_id
    for (uint32_t v : not_S) {
      if (v != node_id) subgraph.push_back(v);
    }

    algorithms::NodeState state;
    state.visited.resize(G.Nodes().size(), false);
    std::vector<uint32_t> W, not_W;
    FindComponentOfSubgraph(t, G, subgraph, &state, &W);

    if (subset(T, W)) {
      for (uint32_t node_id : component) {
        if (std::find(W.begin(), W.end(), node_id) == W.end())
          not_W.push_back(node_id);
      }
      Backtrack(G, component, not_W, T, s, t, cutsets);
    }
  }
  std::vector<uint32_t> new_T = T;
  new_T.push_back(node_id);
  Backtrack(G, component, S, new_T, s, t, cutsets);
}

void CutsetAlgo(const graphdata::GraphView &G,
                const std::vector<uint32_t> &component, uint32_t s, uint32_t t,
                std::set<std::set<uint32_t>> *cutsets) {
  std::vector<uint32_t> subgraph;  // V - {s}
  for (uint32_t node_id : component) {
    if (node_id != s) subgraph.push_back(node_id);
  }

  algorithms::NodeState state;
  state.visited.resize(G.Nodes().size(), false);

  std::vector<uint32_t> W, not_W;
  FindComponentOfSubgraph(t, G, subgraph, &state, &W);

  for (uint32_t node_id : component) {
    if (std::find(W.begin(), W.end(), node_id) == W.end())
      not_W.push_back(node_id);
  }

  Backtrack(G, component, not_W, {t}, s, t, cutsets);
}

void SolveComponent(const graphdata::GraphView &G,
                    const std::vector<uint32_t> &component,
                    std::set<std::set<uint32_t>> *cutsets) {
  for (size_t i = 0; i < component.size(); ++i)
    for (size_t j = i + 1; j < component.size(); ++j)
      CutsetAlgo(G, component, component[i], component[j], cutsets);
}

}  // namespace

namespace algorithms {

std::vector<std::vector<graphdata::Edge>> GetCutsets(
    const graphdata::GraphView &G) {
  NodeState state;
  size_t node_size = G.Nodes().size();
  state.visited.resize(node_size, false);

  std::set<std::set<uint32_t>> cutsets;  // edge_id

  for (const graphdata::Node &node : G.Nodes()) {
    if (state.visited[node.id]) continue;
    std::vector<uint32_t> component;
    FindComponent(node.id, G, &state, &component);
    SolveComponent(G, component, &cutsets);
  }

  std::vector<std::vector<graphdata::Edge>> ret;
  for (const auto &cutset : cutsets) {
    std::vector<graphdata::Edge> ret_cutset;
    for (const auto &edge_id : cutset) {
      ret_cutset.push_back(G.Edges()[edge_id]);
    }
    ret.push_back(ret_cutset);
  }

  return ret;
}

}  // namespace algorithms
