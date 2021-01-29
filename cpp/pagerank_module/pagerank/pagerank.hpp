/// @file

#pragma once

#include <filesystem>
#include <future>
#include <vector>

#include <utils/adjacency_list.hpp>

namespace pagerank {

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
  PageRankGraph(uint32_t number_of_nodes, uint32_t number_of_edges,
                const std::vector<std::pair<uint32_t, uint32_t>> &edges);

  /// @return -- number of nodes in graph
  uint32_t GetNodeCount() const;

  /// @return -- nubmer of edges in graph
  uint32_t GetEdgeCount() const;

  /// @return -- a reference to ordered ordered vector of edges
  const std::vector<std::pair<uint32_t, uint32_t>> &GetOrderedEdges() const;

  /// Returns out degree of node node_id
  /// @param node_id -- node name
  /// @return -- out degree of node node_id
  uint32_t GetOutDegree(uint32_t node_id) const;

 private:
  /// node_count equals number of nodes in graph
  uint32_t node_count_;
  /// edge_count equals number of edges in graph
  uint32_t edge_count_;
  /// directed edges (source, target) (source -> target) ordered by target
  std::vector<std::pair<uint32_t, uint32_t>> ordered_edges_;
  /// out degree for each node in graph because it is required in calculating
  /// PageRank
  std::vector<uint32_t> out_degree_;
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
void CalcualteOptimalBorders(const PageRankGraph &graph,
                             int number_of_threads,
                             std::vector<uint32_t> *borders);

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
void ThreadPageRankIteration(
    const PageRankGraph &graph, const std::vector<double> &old_rank, int lo,
    int hi, std::promise<std::vector<double>> new_rank_promise);

/// Merging PageRank blocks from ThreadPageRankIteration
/// In every iteration function is called in order of cluster_ids.
/// @param graph -- graph
/// @param damping_factor -- damping factor
/// @param block -- PageRank block calculated in ThreadPageRankIteration
/// @param cluster_id -- id of block we are currently adding to rank_next
/// @param borders -- vector of borders calculated by CalculateOptimalBorders
/// @param rank_next -- PageRank which will be updated
void AddCurrentBlockToRankNext(const PageRankGraph &graph,
                               double damping_factor,
                               const std::vector<double> &block,
                               size_t cluster_id,
                               const std::vector<uint32_t> &borders,
                               std::vector<double> *rank_next);

/// Adds remaining PageRank values
/// Adds PageRank values of nodes that haven't been added by
/// AddCurrentBlockToRankNext. That are nodes whose id is greater
/// than id of target node of the last edge in ordered edge list.
///
/// @param graph -- graph
/// @param damping_factor -- damping factor
/// @param rank_next -- PageRank which will be updated
void CompleteRankNext(const PageRankGraph &graph, double damping_factor,
                      std::vector<double> *rank_next);

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
bool CheckContinueIterate(const std::vector<double> &rank,
                          const std::vector<double> &rank_next,
                          size_t max_iterations, double stop_epsilon,
                          size_t number_of_iterations);

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
std::vector<double> ParallelIterativePageRank(const PageRankGraph &graph,
                                              size_t max_iterations = 100,
                                              double damping_factor = 0.85,
                                              double stop_epsilon = 10e-6);

}  // namespace pagerank
