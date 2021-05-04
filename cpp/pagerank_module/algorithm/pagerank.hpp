#pragma once

#include <future>

namespace pagerank_util {

/// @tparam T Node ID data type.
template <typename T>
class AdjacencyList {
 public:
  AdjacencyList() = default;
  explicit AdjacencyList(int64_t node_count) : list_(node_count) {}

  /// If adjacency list size is unknown at the construction time this method
  /// can be used to reserve the required space. The method will also clear
  /// underlying storage if it contains something.
  void Init(int64_t node_count) {
    list_.clear();
    list_.resize(node_count);
  }

  auto GetNodeCount() const { return list_.size(); }

  /// AdjacentPair is a pair of T. Values of T have to be >= 0 and < node_count
  /// because they represent position in the underlying std::vector.
  void AddAdjacentPair(T left_node_id, T right_node_id, bool undirected = false) {
    list_[left_node_id].push_back(right_node_id);
    if (undirected) list_[right_node_id].push_back(left_node_id);
  }

  /// Be careful and don't call AddAdjacentPair while you have reference to this
  /// vector because the referenced vector could be resized (moved) which means
  /// that the reference is going to become invalid.
  ///
  /// @return A reference to std::vector of adjecent node ids.
  const auto &GetAdjacentNodes(T node_id) const { return list_[node_id]; }

 private:
  std::vector<std::vector<T>> list_;
};

}  // namespace pagerank_util

namespace pagerank_alg {

/// A directed, unweighted graph.
/// Self loops and multiple edges are allowed and they will affect the result.
/// Node ids are integers from interval [0, number of nodes in graph - 1].
/// Graph is allowed to be disconnected.
class PageRankGraph {
 public:
  /// Creates graph with given number of nodes with node ids from interval
  /// [0, number_of_nodes - 1] and with given edges between them.
  /// Node ids describing edges have to be integers from
  /// interval [0, number of nodes in graph - 1].
  /// @param number_of_nodes -- number of nodes in graph
  /// @param number_of_edges -- number of edges in graph
  /// @param edges -- pairs (source, target) representing directed edges
  PageRankGraph(std::uint64_t number_of_nodes, std::uint64_t number_of_edges,
                const std::vector<std::pair<std::uint64_t, std::uint64_t>> &edges);

  /// @return -- number of nodes in graph
  std::uint64_t GetNodeCount() const;

  /// @return -- nubmer of edges in graph
  std::uint64_t GetEdgeCount() const;

  /// @return -- a reference to ordered ordered vector of edges
  const std::vector<std::pair<std::uint64_t, std::uint64_t>> &GetOrderedEdges() const;

  /// Returns out degree of node node_id
  /// @param node_id -- node name
  /// @return -- out degree of node node_id
  std::uint64_t GetOutDegree(std::uint64_t node_id) const;

 private:
  /// node_count equals number of nodes in graph
  std::uint64_t node_count_;
  /// edge_count equals number of edges in graph
  std::uint64_t edge_count_;
  /// directed edges (source, target) (source -> target) ordered by target
  std::vector<std::pair<std::uint64_t, std::uint64_t>> ordered_edges_;
  /// out degree for each node in graph because it is required in calculating
  /// PageRank
  std::vector<std::uint64_t> out_degree_;
};

/// Calculates optimal borders for dividing edges in number_of_threads
/// consecutive partitions (blocks) such that the maximal block size is minimal
/// For example: if number_of_edges = 10 and number_of_threads = 3:
/// optimal borders = {0, 3, 6, 10} so obtained blocks of edges
/// are [0, 3>, [3, 6> and [6, 10>.
///
/// @param graph -- graph
/// @param number_of_threads -- number of threads
/// @param borders -- vector in which calculated optimal borders will be written
void CalcualteOptimalBorders(const PageRankGraph &graph, int number_of_threads, std::vector<std::uint64_t> *borders);

/// Calculating PageRank block related to [lo, hi> interval of edges
/// Graph edges are ordered by target node ids so target nodes of an interval
/// of edges also form an interval of nodes.
/// Required PageRank values of nodes from this interval will be sent
/// as vector of values using new_rank_promise.
///
/// @param graph -- graph
/// @param old_rank -- rank of nodes after previous iterations
/// @param damping_factor -- damping factor
/// @param lo -- left bound of interval of edges
/// @param hi -- right bound of interval of edges
/// @param new_rank_promise -- used for sending information about calculated new
/// PageRank block
void ThreadPageRankIteration(const PageRankGraph &graph, const std::vector<double> &old_rank, int lo, int hi,
                             std::promise<std::vector<double>> new_rank_promise);

/// Merging PageRank blocks from ThreadPageRankIteration.
///
/// @param graph -- graph
/// @param damping_factor -- damping factor
/// @param block -- PageRank block calculated in ThreadPageRankIteration
/// @param rank_next -- PageRanks which will be updated
void AddCurrentBlockToRankNext(const PageRankGraph &graph, double damping_factor, const std::vector<double> &block,
                               std::vector<double> *rank_next);

/// Adds remaining PageRank values
/// Adds PageRank values of nodes that haven't been added by
/// AddCurrentBlockToRankNext. That are nodes whose id is greater
/// than id of target node of the last edge in ordered edge list.
///
/// @param graph -- graph
/// @param damping_factor -- damping factor
/// @param rank_next -- PageRank which will be updated
void CompleteRankNext(const PageRankGraph &graph, double damping_factor, std::vector<double> *rank_next);

/// Checks whether PageRank algorithm should continue iterating
/// Checks if maximal number of iterations was reached or
/// if difference between every component of previous and current PageRank
/// was less than stop_epsilon.
///
/// @param rank -- PageRank before the last iteration
/// @param rank_next -- PageRank after the last iteration
/// @param max_iterations -- maximal number of iterations
/// @param stop_epsilon -- stop epsilon
/// @param number_of_iterations -- current number of operations
bool CheckContinueIterate(const std::vector<double> &rank, const std::vector<double> &rank_next, size_t max_iterations,
                          double stop_epsilon, size_t number_of_iterations);

/// Normalizing PageRank
/// Divides all values with sum of the values to get their sum equal 1
///
/// @param rank -- PageRank
void NormalizeRank(std::vector<double> *rank);

/// If we present nodes as pages and directed edges between them as links the
/// PageRank algorithm outputs a probability distribution used to represent the
/// likelihood that a person randomly clicking on links will arrive at any
/// particular page.
///
/// PageRank theory holds that an imaginary surfer who is randomly clicking on
/// links will eventually stop clicking. The probability, at any step, that the
/// person will continue randomly clicking on links is called a damping factor,
/// otherwise next page is chosen randomly among all pages.
///
/// PageRank is computed iteratively using following formula:
/// Rank(n, t + 1) = (1 - d) / number_of_nodes
///                + d * sum { Rank(in_neighbour_of_n, t) /
///                out_degree(in_neighbour_of_n)}
/// Where Rank(n, t) is PageRank of node n at iteration t
/// At the end Rank values are normalized to sum 1 to form probability
/// distribution.
///
/// Default arguments are equal to default arguments in NetworkX PageRank
/// implementation:
/// https://networkx.github.io/documentation/networkx-1.10/reference/generated/
/// networkx.algorithms.link_analysis.pagerank_alg.pagerank_module.html
///
/// @param graph -- a directed, unweighted, not necessarily connected graph
/// which can contain multiple edges and self-loops.
/// @param max_iterations -- maximum number of iterations performed by PageRank.
/// @param damping_factor -- a real number from interval [0, 1], as described
/// above
/// @return -- probability distribution, as described above
std::vector<double> ParallelIterativePageRank(const PageRankGraph &graph, size_t max_iterations = 100,
                                              double damping_factor = 0.85, double stop_epsilon = 10e-6);

}  // namespace pagerank_alg
