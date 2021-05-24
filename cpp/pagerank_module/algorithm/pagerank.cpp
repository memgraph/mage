#include <algorithm>
#include <numeric>
#include <queue>

#include "pagerank.hpp"

namespace pagerank_alg {

PageRankGraph::PageRankGraph(const std::uint64_t number_of_nodes, const std::uint64_t number_of_edges,
                             const std::vector<EdgePair> &edges)
    : node_count_(number_of_nodes),
      edge_count_(number_of_edges),
      out_degree_(std::vector<std::uint64_t>(number_of_nodes)) {
  pagerank_util::AdjacencyList<std::uint64_t> in_neighbours;
  in_neighbours.Init(number_of_nodes);
  for (const auto &[from, to] : edges) {
    // Because PageRank needs a set of nodes that point to a given node.
    in_neighbours.AddAdjacentPair(from, to);
    out_degree_[from] += 1;
  }
  for (std::size_t node_id = 0; node_id < number_of_nodes; node_id++) {
    const auto &adjacent_nodes = in_neighbours.GetAdjacentNodes(node_id);
    for (const auto &adjacent_node : adjacent_nodes) {
      ordered_edges_.emplace_back(adjacent_node, node_id);
    }
  }
}

std::uint64_t PageRankGraph::GetNodeCount() const { return node_count_; }

std::uint64_t PageRankGraph::GetEdgeCount() const { return edge_count_; }

const std::vector<EdgePair> &PageRankGraph::GetOrderedEdges() const { return ordered_edges_; }

std::uint64_t PageRankGraph::GetOutDegree(const std::uint64_t node_id) const { return out_degree_[node_id]; }

void CalculateOptimalBorders(const PageRankGraph &graph, const int number_of_threads,
                             std::vector<std::uint64_t> *borders) {
  for (std::uint64_t border_index = 0; border_index <= number_of_threads; border_index++) {
    borders->push_back(border_index * graph.GetEdgeCount() / number_of_threads);
  }
}

void ThreadPageRankIteration(const PageRankGraph &graph, const std::vector<double> &old_rank, const std::uint64_t lo,
                             const std::uint64_t hi, std::promise<std::vector<double>> new_rank_promise) {
  std::vector<double> new_rank(graph.GetNodeCount(), 0);
  // Calculate sums of PR(page)/C(page) scores for the entire block (from lo to
  // hi edges).
  for (std::size_t edge_id = lo; edge_id < hi; edge_id++) {
    const auto [source, target] = graph.GetOrderedEdges()[edge_id];
    // Add the score of target node to the sum.
    new_rank[source] += old_rank[target] / graph.GetOutDegree(target);
  }
  new_rank_promise.set_value(new_rank);
}

void AddCurrentBlockToRankNext(const PageRankGraph &graph, const double damping_factor,
                               const std::vector<double> &block, std::vector<double> &rank_next) {
  // The block vector contains partially precalculated sums of PR(page)/C(page)
  // for each node. Node index in the block vector corresponds to the node index
  // in the rank_next vector.
  for (std::size_t node_index = 0; node_index < block.size(); node_index++) {
    rank_next[node_index] += damping_factor * block[node_index];
  }
}

void CompleteRankNext(const PageRankGraph &graph, const double damping_factor, std::vector<double> &rank_next) {
  while (rank_next.size() < graph.GetNodeCount()) {
    rank_next.push_back((1.0 - damping_factor) / graph.GetNodeCount());
  }
}

bool CheckContinueIterate(const std::vector<double> &rank, const std::vector<double> &rank_next,
                          const std::size_t max_iterations, const double stop_epsilon,
                          const std::size_t number_of_iterations) {
  if (number_of_iterations == max_iterations) {
    return false;
  }
  for (std::size_t node_id = 0; node_id < rank.size(); node_id++) {
    if (std::abs(rank[node_id] - rank_next[node_id]) > stop_epsilon) {
      return true;
    }
  }
  return false;
}

void NormalizeRank(std::vector<double> *rank) {
  const double sum = std::accumulate(rank->begin(), rank->end(), 0.0);
  for (double &value : *rank) {
    value /= sum;
  }
}

std::vector<double> ParallelIterativePageRank(const PageRankGraph &graph, std::size_t max_iterations,
                                              double damping_factor, double stop_epsilon) {
  const std::uint64_t number_of_threads = std::thread::hardware_concurrency();

  std::vector<std::uint64_t> borders;
  CalculateOptimalBorders(graph, number_of_threads, &borders);

  std::vector<double> rank(graph.GetNodeCount(), 1.0 / graph.GetNodeCount());
  // Because we increment number_of_iterations at the end of while loop.
  bool continue_iterate = max_iterations != 0;

  std::size_t number_of_iterations = 0;
  while (continue_iterate) {
    std::vector<std::promise<std::vector<double>>> page_rank_promise(number_of_threads);
    std::vector<std::future<std::vector<double>>> page_rank_future(number_of_threads);
    std::transform(page_rank_promise.begin(), page_rank_promise.end(), page_rank_future.begin(),
                   [](auto &pr_promise) { return pr_promise.get_future(); });

    std::vector<std::thread> my_threads;
    my_threads.reserve(number_of_threads);
    for (std::size_t cluster_id = 0; cluster_id < number_of_threads; cluster_id++) {
      my_threads.emplace_back([&, lo = borders[cluster_id], hi = borders[cluster_id + 1]](
                                  auto promise) { ThreadPageRankIteration(graph, rank, lo, hi, std::move(promise)); },
                              std::move(page_rank_promise[cluster_id]));
    }

    std::vector<double> rank_next(graph.GetNodeCount(), (1.0 - damping_factor) / graph.GetNodeCount());
    for (std::size_t cluster_id = 0; cluster_id < number_of_threads; cluster_id++) {
      std::vector<double> block = page_rank_future[cluster_id].get();
      AddCurrentBlockToRankNext(graph, damping_factor, block, rank_next);
    }
    CompleteRankNext(graph, damping_factor, rank_next);

    for (std::uint64_t i = 0; i < number_of_threads; i++) {
      if (my_threads[i].joinable()) {
        my_threads[i].join();
      }
    }
    rank.swap(rank_next);
    number_of_iterations++;
    continue_iterate = CheckContinueIterate(rank, rank_next, max_iterations, stop_epsilon, number_of_iterations);
  }
  NormalizeRank(&rank);
  return rank;
}

}  // namespace pagerank_alg
