#include <set>

#include <mg_graph.hpp>

namespace cycles_util {

/// Simple struct that keeps the state of nodes in algorithms that rely on
/// DFS traversal.
struct NodeState {
  std::vector<bool> visited;
  std::vector<int32_t> discovery, low_link, depth;
  std::vector<uint32_t> parent;
};

mg_graph::Node<> NodeFromId(uint64_t node_id, const mg_graph::GraphView<> *G);

std::vector<uint64_t> FindCycle(uint64_t a, uint64_t b, const cycles_util::NodeState &state);

void FindNonSTEdges(uint64_t node_id, const mg_graph::GraphView<> *G, cycles_util::NodeState *state,
                    std::set<std::pair<uint64_t, uint64_t>> *non_st_edges);

void FindFundamentalCycles(const std::set<std::pair<uint64_t, uint64_t>> &non_st_edges,
                           const cycles_util::NodeState &state, std::vector<std::vector<uint64_t>> *fundamental_cycles);

void SolveMask(int mask, const std::vector<std::vector<uint64_t>> &fundamental_cycles, const mg_graph::GraphView<> *G,
               std::vector<std::vector<mg_graph::Node<>>> *cycles);

void GetCyclesFromFundamentals(const std::vector<std::vector<uint64_t>> &fundamental_cycles,
                               const mg_graph::GraphView<> *G, std::vector<std::vector<mg_graph::Node<>>> *cycles);
}  // namespace cycles_util

namespace cycles_alg {

std::vector<std::vector<mg_graph::Node<>>> GetCycles(const mg_graph::GraphView<> *G);

std::vector<std::pair<mg_graph::Node<>, mg_graph::Node<>>> GetNeighbourCycles(const mg_graph::GraphView<> *G);

}  // namespace cycles_alg
