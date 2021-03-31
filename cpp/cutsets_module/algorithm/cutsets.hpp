#pragma once

#include <algorithm>
#include <set>
#include <unordered_set>

#include <mg_graph.hpp>

namespace cutsets_util {

/// Simple struct that keeps the state of nodes in algorithms that rely on
/// DFS traversal.
struct NodeState {
  std::vector<bool> visited;
  std::vector<int32_t> discovery, low_link, depth;
  std::vector<uint32_t> parent;
  uint64_t counter;
};

void ComponentCountDFS(const uint64_t node_id, const mg_graph::GraphView<> *G, const std::vector<uint64_t> &component,
                       cutsets_util::NodeState *state);

uint64_t ComponentCount(const mg_graph::GraphView<> *G, const std::vector<uint64_t> &component);

std::set<uint64_t> OmegaSet(const std::vector<uint64_t> &U, const std::vector<uint64_t> &V,
                            const mg_graph::GraphView<> *G);

void FindComponentOfSubgraph(const uint64_t node_id, const mg_graph::GraphView<> *G,
                             const std::vector<uint64_t> &subgraph, cutsets_util::NodeState *state,
                             std::vector<uint64_t> *component);

void FindComponent(const uint64_t node_id, const mg_graph::GraphView<> *G, cutsets_util::NodeState *state,
                   std::vector<uint64_t> *component);

void Backtrack(const mg_graph::GraphView<> *G, const std::vector<uint64_t> &component, const std::vector<uint64_t> &S,
               const std::vector<uint64_t> &T, uint64_t s, uint64_t t, std::set<std::set<uint64_t>> *cutsets);

void SolveComponent(const mg_graph::GraphView<> *G, const std::vector<uint64_t> &component,
                    std::set<std::set<uint64_t>> *cutsets);

template <typename T>
inline bool IsSubset(const std::vector<T> &A, const std::vector<T> &B) {
  std::unordered_set<T> AUB;
  AUB.insert(A.cbegin(), A.cend());
  AUB.insert(B.cbegin(), B.cend());
  return AUB.size() == B.size();
}

}  // namespace cutsets_util

namespace cutsets_alg {
void CutsetsAlgorithm(const mg_graph::GraphView<> *G, const std::vector<uint64_t> &component, uint64_t s, uint64_t t,
                      std::set<std::set<uint64_t>> *cutsets);

std::vector<std::vector<mg_graph::Edge<>>> GetCutsets(const mg_graph::GraphView<> *G);
}  // namespace cutsets_alg