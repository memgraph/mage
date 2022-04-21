#include <cstdint>
#include <set>
#include <optional>

#include <mg_graph.hpp>

namespace cycles_util {

///
///@brief Structure for the cycles detection algorithm. Keeps in track which nodes are visited, and is suitable for
/// keeping the information about spanning tree
///
struct NodeState {
  std::vector<bool> visited;
  std::vector<std::uint64_t> discovery, depth;
  std::vector<std::optional<std::uint64_t>> parent;

  explicit NodeState(std::size_t number_of_nodes);

  void SetVisited(std::uint64_t node_id);

  bool IsVisited(std::uint64_t node_id) const;

  void SetParent(std::uint64_t parent_id, std::uint64_t node_id);

  std::optional<std::uint64_t> GetParent(std::uint64_t node_id) const;

  void SetDepth(std::uint64_t node_id, std::uint64_t node_depth);

  std::uint64_t GetDepth(std::uint64_t node_id) const;
};

///
///@brief Function for finding edges that do not lie on the graph spanning tree starting from the node_id. Non
/// spanning-tree edges are stored in-place within the container provided to this function. Function is called
/// recursively until spanning tree is not generated.
///
///@param node_id Starting spanning tree node
///@param graph Graph for exploration
///@param state Current cycles detection algorithm state
///@param non_st_edges Non spanning tree edges container
///
void FindNonSpanningTreeEdges(uint64_t node_id, const mg_graph::GraphView<> &graph, NodeState *state,
                              std::set<std::pair<uint64_t, uint64_t>> *non_st_edges);

///
///@brief Function for creating a fundamental cycle from the edges not included on the spanning tree. Finding
/// fundamental graph is done by traversing parent nodes starting from the node with lower index number.
///
///@param node_a First edge node
///@param node_b Second edge node
///@param state Current cylce detection algorithm state
///@return std::vector<std::uint64_t> Container with fundamental cycle
///
std::vector<std::uint64_t> FindFundametalCycle(std::uint64_t node_a, std::uint64_t node_b, const NodeState &state);

///
///@brief Function for finding all fundamental cycles from the edges not included on the spanning tree. Result is stored
/// in \a fundamental_cycles  container
///
///@param non_st_edges Non spanning tree edges container
///@param state Current cylce detection algorithm state
///@param fundamental_cycles Container containing all fundamental cycles
///
void FindFundamentalCycles(const std::set<std::pair<std::uint64_t, std::uint64_t>> &non_st_edges,
                           const NodeState &state, std::vector<std::vector<std::uint64_t>> *fundamental_cycles);

///
///@brief Solving combination of fundamental cycles means finding an elementary cycle within the fundamental one. For
/// this purpose a concept of masking is used, for example having a mask 3 (dec) = 011 (bin) means combining first and
/// second fundamental cycle.
///
///@param mask Mask for combining cycles
///@param fundamental_cycles Fundamental cycles found by exploring spanning tree
///@param graph Graph for exploration
///@param cycles Elementary final graph cycles
///
void CombineCycles(std::uint32_t mask, const std::vector<std::vector<std::uint64_t>> &fundamental_cycles,
                   const mg_graph::GraphView<> &graph, std::vector<std::vector<mg_graph::Node<>>> *cycles);

///
///@brief Method for getting cycles from the found fundamental ones. This method explores 2^funtamental_cycles_size
/// different options of combining cycles (because combination of cycle can also be a cycle)
///
///@param fundamental_cycles Fundamental cycles found by exploring spanning tree
///@param graph Graph for exploration
///@param cycles Elementary final graph cycles
///
void GetCyclesFromFundamentals(const std::vector<std::vector<std::uint64_t>> &fundamental_cycles,
                               const mg_graph::GraphView<> &graph, std::vector<std::vector<mg_graph::Node<>>> *cycles);

}  // namespace cycles_util

namespace cycles_alg {

/// @brief Returns a list of all simple cycles of an undirected graph G. Each simple
/// cycle is represented by a list of nodes in cycle order. Cycles can be double connected neighboring nodes or self
/// loops.
///
/// There is no other guarantee on the order of the nodes in a cycle.
///
/// The implemented algorithm (Gibb) is described in the 1982 MIT report called
/// "Algorithmic approaches to circuit enumeration problems and applications"
/// by Boon Chai Lee.
///
/// The problem itself is not solvable in polynomial time. The bulk of the
/// algorithm's runtime is spent on finding all subsets of fundamental cycles
/// which takes about O(2^(|E|-|V|+1)) time, where E represents a set of
/// edges and V represents a set of vertices.
///
///@param graph Graph for exploration
///@return std::vector<std::vector<mg_graph::Node<>>> Cycles represented with Node
///
std::vector<std::vector<mg_graph::Node<>>> GetCycles(const mg_graph::GraphView<> &graph);

}  // namespace cycles_alg
