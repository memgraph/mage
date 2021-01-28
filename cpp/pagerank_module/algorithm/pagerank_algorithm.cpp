#include "pagerank_algorithm.hpp"

#include <stdlib.h>
#include <numeric>
#include <queue>

namespace page_rank::parallel {
    /**
     * Make a PageRank graph data structure from the list of graph edges
     *
     * @param number_of_nodes Number of nodes
     * @param number_of_edges Number of edges
     * @param edges List of graph edges
     */
    PageRankGraph::PageRankGraph(uint32_t number_of_nodes, uint32_t number_of_edges,
                                 const std::vector<std::pair<uint32_t, uint32_t>> &edges) :
            node_count_(number_of_nodes), edge_count_(number_of_edges),
            out_degree_(std::vector<uint32_t>(number_of_nodes)) {

        utils::AdjacencyList<uint32_t> in_neighbours;
        in_neighbours.Init(number_of_nodes);
        for (const auto &edge : edges) {
            // Because PageRank needs a set of nodes that point to a given node.
            in_neighbours.AddAdjacentPair(edge.second, edge.first);
            out_degree_[edge.first] += 1;
        }
        for (size_t node_id = 0; node_id < number_of_nodes; node_id++) {
            const std::vector<uint32_t> &adjacent_nodes =
                    in_neighbours.GetAdjacentNodes(node_id);
            for (const auto &adjacent_node : adjacent_nodes) {
                ordered_edges_.emplace_back(adjacent_node, node_id);
            }
        }
    }

    /**
     * Returns number of graph nodes
     *
     * @return Node count
     */
    uint32_t PageRankGraph::GetNodeCount() const { return node_count_; }

    /**
     * Returns number of graph edges
     *
     * @return Edges count
     */
    uint32_t PageRankGraph::GetEdgeCount() const { return edge_count_; }

    /**
     * Returns ordered edges
     *
     * @return Ordered edges
     */
    const std::vector<std::pair<uint32_t, uint32_t>> &PageRankGraph::GetOrderedEdges() const {
        return ordered_edges_;
    }

    /**
     * Returns node's out degree
     *
     * @return Out degree of a node with ID = <node_id>
     */
    uint32_t PageRankGraph::GetOutDegree(uint32_t node_id) const {
        return out_degree_[node_id];
    }

    /**
     * Calculate optimal borders for PageRank algorithm
     *
     * @param graph PageRank graph
     * @param number_of_threads Number of threads
     * @param borders Bords
     */
    void CalculateOptimalBorders(const PageRankGraph &graph, int number_of_threads, std::vector<uint32_t> *borders) {
        for (size_t i = 0; i <= number_of_threads; i++)
            borders->push_back(static_cast<uint64_t>(i) * graph.GetEdgeCount() / number_of_threads);
    }

    /**
     * One iteration for threaded PageRank algorihm
     *
     * @param graph PageRank algorithm
     * @param old_rank Old rank
     * @param lo
     * @param hi
     * @param new_rank_promise
     */
    void ThreadPageRankIteration(
            const PageRankGraph &graph, const std::vector<double> &old_rank,
            uint32_t lo, uint32_t hi,
            std::promise<std::vector<double>> new_rank_promise) {
        std::vector<double> new_rank;

        for (size_t edge_id = lo; edge_id < hi; edge_id++) {
            uint32_t source = graph.GetOrderedEdges()[edge_id].first;
            uint32_t target = graph.GetOrderedEdges()[edge_id].second;
            while (new_rank.size() < target - graph.GetOrderedEdges()[lo].second + 1)
                new_rank.push_back(0);
            new_rank.back() += old_rank[source] / graph.GetOutDegree(source);
        }
        new_rank_promise.set_value(new_rank);
    }

    /**
     * Adds current block to the next rank
     *
     * @param graph Pagerank graph
     * @param damping_factor Damping factor
     * @param block
     * @param cluster_id  Cluster ID
     * @param borders
     * @param rank_next
     */
    void AddCurrentBlockToRankNext(const PageRankGraph &graph,
                                   double damping_factor,
                                   const std::vector<double> &block,
                                   size_t cluster_id,
                                   const std::vector<uint32_t> &borders,
                                   std::vector<double> *rank_next) {
        const uint32_t &lo = borders[cluster_id];
        if (lo < graph.GetEdgeCount())
            while (rank_next->size() <= graph.GetOrderedEdges()[lo].second)
                rank_next->push_back((1.0 - damping_factor) / graph.GetNodeCount());
        for (size_t i = 0; i < block.size(); i++) {
            if (i > 0)
                rank_next->push_back((1.0 - damping_factor) / graph.GetNodeCount());
            rank_next->back() += damping_factor * block[i];
        }
    }

    /**
     * Method for completing next rank iteration.
     *
     * @param graph Pagerank graph
     * @param damping_factor Damping factor
     * @param rank_next
     */
    void CompleteRankNext(const PageRankGraph &graph, double damping_factor,
                          std::vector<double> *rank_next) {
        while (rank_next->size() < graph.GetNodeCount())
            rank_next->push_back((1.0 - damping_factor) / graph.GetNodeCount());
    }

    /**
     *
     *
     * @param rank
     * @param rank_next
     * @param max_iterations
     * @param stop_epsilon
     * @param number_of_iterations
     * @return
     */
    bool CheckContinueIterate(const std::vector<double> &rank,
                              const std::vector<double> &rank_next,
                              size_t max_iterations, double stop_epsilon,
                              size_t number_of_iterations) {
        if (number_of_iterations == max_iterations) return false;
        for (size_t node_id = 0; node_id < rank.size(); node_id++)
            if (std::abs(rank[node_id] - rank_next[node_id]) > stop_epsilon)
                return true;
        return false;
    }

    /**
     * Function for normalizing the rank of PageRank algorihtm
     *
     * @param rank Vector of ranks
     */
    void NormalizeRank(std::vector<double> *rank) {
        double sum = std::accumulate(rank->begin(), rank->end(), 0.0);
        for (double &value : *rank) value /= sum;
    }

    /**
     * Implementation of parallel iterative PageRank algorithm
     *
     * @param graph PageRank graph
     * @param max_iterations Maximum algorithm iterations
     * @param damping_factor Damping factor
     * @param stop_epsilon Stopping factor
     * @return Vector of normalized ranks
     */
    std::vector<double> ParallelIterativePageRank(const PageRankGraph &graph,
                                                  size_t max_iterations,
                                                  double damping_factor,
                                                  double stop_epsilon) {
        const uint32_t number_of_threads = std::thread::hardware_concurrency();

        std::vector<uint32_t> borders;
        CalculateOptimalBorders(graph, number_of_threads, &borders);

        std::vector<double> rank(graph.GetNodeCount(), 1.0 / graph.GetNodeCount());
        bool continue_iterate = true;
        // Because we increment number_of_iterations at the end of while loop
        if (max_iterations == 0) continue_iterate = false;
        size_t number_of_iterations = 0;
        while (continue_iterate) {
            std::vector<std::promise<std::vector<double>>> page_rank_promise(
                    number_of_threads);
            std::vector<std::future<std::vector<double>>> page_rank_future(
                    number_of_threads);
            for (size_t cluster_id = 0; cluster_id < number_of_threads; cluster_id++)
                page_rank_future[cluster_id] = page_rank_promise[cluster_id].get_future();

            std::vector<std::thread> my_threads;
            my_threads.reserve(number_of_threads);
            for (size_t cluster_id = 0; cluster_id < number_of_threads; cluster_id++) {
                my_threads.emplace_back(
                        [](const auto &graph, const auto &old_rank, auto lo, auto hi,
                           auto promise) {
                            ThreadPageRankIteration(graph, old_rank, lo, hi,
                                                    std::move(promise));
                        },
                        std::cref(graph), std::cref(rank), borders[cluster_id],
                        borders[cluster_id + 1], std::move(page_rank_promise[cluster_id]));
            }

            std::vector<double> rank_next;
            for (size_t cluster_id = 0; cluster_id < number_of_threads; cluster_id++) {
                std::vector<double> block = page_rank_future[cluster_id].get();
                AddCurrentBlockToRankNext(graph, damping_factor, block, cluster_id,
                                          borders, &rank_next);
            }
            CompleteRankNext(graph, damping_factor, &rank_next);

            for (int i = 0; i < number_of_threads; i++) my_threads[i].join();
            rank.swap(rank_next);
            number_of_iterations++;
            continue_iterate = CheckContinueIterate(rank, rank_next, max_iterations,
                                                    stop_epsilon, number_of_iterations);
        }
        NormalizeRank(&rank);
        return rank;
    }

}  // namespace page_rank::parallel