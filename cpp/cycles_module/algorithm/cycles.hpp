#include <set>

#include <mg_graph.hpp>

namespace cycles_util {

/// Simple struct that keeps the state of nodes in algorithms that rely on
/// DFS traversal.
struct NodeState {
  std::vector<bool> visited;
  std::vector<std::uint64_t> discovery, depth;
  std::vector<std::uint64_t> parent;

  explicit NodeState(size_t number_of_nodes);

  void SetVisited(std::uint64_t node_id);

  bool IsVisited(std::uint64_t node_id);

  void SetParent(std::uint64_t parent_id, std::uint64_t node_id);

  std::uint64_t GetParent(std::uint64_t node_id);

  void SetDepth(std::uint64_t node_id, std::uint64_t node_depth);

  std::uint64_t GetDepth(std::uint64_t node_id);
};

std::vector<std::uint64_t> FindCycle(std::uint64_t node_a, std::uint64_t node_b,
                                     const NodeState &state);

void FindNonSTEdges(uint64_t node_id, const mg_graph::GraphView<> &graph,
                    NodeState *state,
                    std::set<std::pair<uint64_t, uint64_t>> *non_st_edges);

void FindFundamentalCycles(
    const std::set<std::pair<std::uint64_t, std::uint64_t>> &non_st_edges,
    const NodeState &state,
    std::vector<std::vector<std::uint64_t>> *fundamental_cycles);

void SolveMask(
    int mask, const std::vector<std::vector<std::uint64_t>> &fundamental_cycles,
    const mg_graph::GraphView<> &graph,
    std::vector<std::vector<mg_graph::Node<>>> *cycles);

void GetCyclesFromFundamentals(
    const std::vector<std::vector<std::uint64_t>> &fundamental_cycles,
    const mg_graph::GraphView<> &graph,
    std::vector<std::vector<mg_graph::Node<>>> *cycles);

} // namespace cycles_util

namespace cycles_alg {

std::vector<std::vector<mg_graph::Node<>>>
GetCycles(const mg_graph::GraphView<> &graph);

std::vector<std::pair<mg_graph::Node<>, mg_graph::Node<>>>
GetNeighbourCycles(const mg_graph::GraphView<> &graph);

} // namespace cycles_alg
