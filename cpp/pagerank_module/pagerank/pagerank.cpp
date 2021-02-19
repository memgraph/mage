#include "pagerank.hpp"

#include <stdlib.h>

#include <algorithm>
#include <numeric>
#include <queue>

namespace pagerank {

PageRankGraph::PageRankGraph(
    const uint32_t number_of_nodes, const uint32_t number_of_edges,
    const std::vector<std::pair<uint32_t, uint32_t>> &edges)
    : node_count_(number_of_nodes),
      edge_count_(number_of_edges),
      out_degree_(std::vector<uint32_t>(number_of_nodes)) {
  utils::AdjacencyList<uint32_t> in_neighbours;
  in_neighbours.Init(number_of_nodes);
  for (const auto &[from, to] : edges) {
    // Because PageRank needs a set of nodes that point to a given node.
    in_neighbours.AddAdjacentPair(from, to);
    out_degree_[from] += 1;
  }
  for (size_t node_id = 0; node_id < number_of_nodes; node_id++) {
    const auto &adjacent_nodes = in_neighbours.GetAdjacentNodes(node_id);
    for (const auto &adjacent_node : adjacent_nodes) {
      ordered_edges_.emplace_back(adjacent_node, node_id);
    }
  }
}

uint32_t PageRankGraph::GetNodeCount() const { return node_count_; }

uint32_t PageRankGraph::GetEdgeCount() const { return edge_count_; }

const std::vector<std::pair<uint32_t, uint32_t>>
    &PageRankGraph::GetOrderedEdges() const {
  return ordered_edges_;
}

uint32_t PageRankGraph::GetOutDegree(const uint32_t node_id) const {
  return out_degree_[node_id];
}

void CalculateOptimalBorders(const PageRankGraph &graph,
                             const int number_of_threads,
                             std::vector<uint32_t> *borders) {
  for (int i = 0; i <= number_of_threads; i++) {
    borders->push_back(static_cast<uint64_t>(i) * graph.GetEdgeCount() /
                       number_of_threads);
  }
}

void ThreadPageRankIteration(
    const PageRankGraph &graph, const std::vector<double> &old_rank,
    const uint32_t lo, const uint32_t hi,
    std::promise<std::vector<double>> new_rank_promise) {
  std::vector<double> new_rank;

  for (size_t edge_id = lo; edge_id < hi; edge_id++) {
    const auto [source, target] = graph.GetOrderedEdges()[edge_id];
    while (new_rank.size() < target - graph.GetOrderedEdges()[lo].second + 1) {
      new_rank.push_back(0);
    }
    new_rank.back() += old_rank[source] / graph.GetOutDegree(source);
  }
  new_rank_promise.set_value(new_rank);
}

void AddCurrentBlockToRankNext(const PageRankGraph &graph,
                               const double damping_factor,
                               const std::vector<double> &block,
                               const size_t cluster_id,
                               const std::vector<uint32_t> &borders,
                               std::vector<double> *rank_next) {
  const uint32_t lo = borders[cluster_id];
  if (lo < graph.GetEdgeCount()) {
    while (rank_next->size() <= graph.GetOrderedEdges()[lo].second) {
      rank_next->push_back((1.0 - damping_factor) / graph.GetNodeCount());
    }
  }
  for (size_t i = 0; i < block.size(); i++) {
    if (i > 0) {
      rank_next->push_back((1.0 - damping_factor) / graph.GetNodeCount());
    }
    rank_next->back() += damping_factor * block[i];
  }
}

void CompleteRankNext(const PageRankGraph &graph, const double damping_factor,
                      std::vector<double> *rank_next) {
  while (rank_next->size() < graph.GetNodeCount()) {
    rank_next->push_back((1.0 - damping_factor) / graph.GetNodeCount());
  }
}

bool CheckContinueIterate(const std::vector<double> &rank,
                          const std::vector<double> &rank_next,
                          const size_t max_iterations,
                          const double stop_epsilon,
                          const size_t number_of_iterations) {
  if (number_of_iterations == max_iterations) {
    return false;
  }
  for (size_t node_id = 0; node_id < rank.size(); node_id++) {
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

std::vector<double> ParallelIterativePageRank(const PageRankGraph &graph,
                                              size_t max_iterations,
                                              double damping_factor,
                                              double stop_epsilon) {
  const uint32_t number_of_threads = std::thread::hardware_concurrency();

  std::vector<uint32_t> borders;
  CalculateOptimalBorders(graph, number_of_threads, &borders);

  std::vector<double> rank(graph.GetNodeCount(), 1.0 / graph.GetNodeCount());
  bool continue_iterate = true;
  // Because we increment number_of_iterations at the end of while loop.
  if (max_iterations == 0) {
    continue_iterate = false;
  }
  size_t number_of_iterations = 0;
  while (continue_iterate) {
    std::vector<std::promise<std::vector<double>>> page_rank_promise(
        number_of_threads);
    std::vector<std::future<std::vector<double>>> page_rank_future(
        number_of_threads);
    std::transform(page_rank_promise.begin(), page_rank_promise.end(),
                   page_rank_future.begin(),
                   [](auto &pr_promise) { return pr_promise.get_future(); });

    std::vector<std::thread> my_threads;
    my_threads.reserve(number_of_threads);
    for (size_t cluster_id = 0; cluster_id < number_of_threads; cluster_id++) {
      my_threads.emplace_back(
          [&, lo = borders[cluster_id],
           hi = borders[cluster_id + 1]](auto promise) {
            ThreadPageRankIteration(graph, rank, lo, hi, std::move(promise));
          },
          std::move(page_rank_promise[cluster_id]));
    }

    std::vector<double> rank_next;
    for (size_t cluster_id = 0; cluster_id < number_of_threads; cluster_id++) {
      std::vector<double> block = page_rank_future[cluster_id].get();
      AddCurrentBlockToRankNext(graph, damping_factor, block, cluster_id,
                                borders, &rank_next);
    }
    CompleteRankNext(graph, damping_factor, &rank_next);

    for (uint32_t i = 0; i < number_of_threads; i++) {
      if (my_threads[i].joinable()) {
        my_threads[i].join();
      }
    }
    rank.swap(rank_next);
    number_of_iterations++;
    continue_iterate = CheckContinueIterate(rank, rank_next, max_iterations,
                                            stop_epsilon, number_of_iterations);
  }
  NormalizeRank(&rank);
  return rank;
}

}  // namespace pagerank
