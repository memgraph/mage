#include <algorithm>
#include <numeric>
#include <queue>
#include <iostream>

#include "hits.hpp"


namespace hits_alg {

/// @tparam T Node ID data type.
namespace {
    template<typename T>
    class AdjacencyList {
    public:
        AdjacencyList() = default;

        explicit AdjacencyList(std::uint64_t node_count) : list_(node_count) {}

        auto GetNodeCount() const { return list_.size(); }
        /// AdjacentPair is a pair of T. Values of T have to be >= 0 and < node_count
        /// because they represent position in the underlying std::vector.
        void AddAdjacentPair(T left_node_id, T right_node_id, bool undirected = false) {
            list_[left_node_id].push_back(right_node_id);
            if (undirected) {
                list_[right_node_id].push_back(left_node_id);
            }
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

/// Calculates optimal borders for dividing edges in number_of_threads
/// consecutive partitions (blocks) such that the maximal block size is minimal
/// For example: if number_of_edges = 10 and number_of_threads = 3:
/// optimal borders = {0, 3, 6, 10} so obtained blocks of edges
/// are [0, 3>, [3, 6> and [6, 10>.
/// @param graph -- graph
/// @param number_of_threads -- number of threads
    std::vector<std::uint64_t> CalculateOptimalBorders(const HitsGraph &graph, const std::uint64_t number_of_threads) {
        std::vector<std::uint64_t> borders;

        if (number_of_threads == 0) {
            throw std::runtime_error("Number of threads can't be zero (0)!");
        }

        for (std::uint64_t border_index = 0; border_index <= number_of_threads; border_index++) {
            borders.push_back(border_index * graph.GetEdgeCount() / number_of_threads);
        }
        return borders;
    }

/// Calculating hub and auth scores block related to [lo, hi> interval of edges
/// Graph edges are ordered by target node ids so target nodes of an interval
/// of edges also form an interval of nodes.
/// Required hub and auth  values of nodes from this interval will be sent
/// as vector of values using new_rank_promise.
///
/// @param graph -- graph
/// @param old_hub -- hub scores of nodes after previous iterations
/// @param old_auth -- auth scores of nodes after previous iterations
/// @param lo -- left bound of interval of edges
/// @param hi -- right bound of interval of edges
/// @param new_hits_promise -- used for sending information about calculated new hub and auth block block
    void ThreadHitsIteration(const HitsGraph &graph, const std::vector<double> old_hub, const std::vector<double> old_auth,
                        const std::uint64_t lo,
                        const std::uint64_t hi,
                        std::promise<std::pair<std::vector<double>, std::vector<double>>> new_hits_promise) {

        std::vector<double> new_hub(graph.GetNodeCount(), 0);
        std::vector<double> new_auth(graph.GetNodeCount(), 0);
        // Calculate hub and auth scores for the entire block
        for (std::size_t edge_id = lo; edge_id < hi; edge_id++) {
            const auto [source, target] = graph.GetOrderedEdges()[edge_id];
            // Add the score of target node to the sum.
            new_hub[target] += old_auth[source];
            new_auth[source] += old_hub[target];
        }
        std::pair<std::vector<double>, std::vector<double>> hub_auth = {new_hub, new_auth};
        new_hits_promise.set_value(hub_auth);
    };

/// Merging hub and auth scores blocks from ThreadPageRankIteration.
///
/// @param graph -- graph
/// @param block -- hub and auth  block calculated in ThreadPageRankIteration
/// @param hub_next -- hub scores which will be updated
/// @param hub_next -- auth scores which will be updated
    void AddCurrentBlockToNext(const HitsGraph &graph, const std::pair<std::vector<double>, std::vector<double>> block,
                               std::vector<double> &hub_next, std::vector<double> &auth_next) {
        for (std::size_t node_index = 0; node_index < block.first.size(); node_index++) {
            hub_next[node_index] += block.first[node_index];
            auth_next[node_index] += block.second[node_index];
        }

    };

/// Adds remaining hub and auth values
/// Adds hub and auth values of nodes that haven't been added by
/// AddCurrentBlockToRankNext. That are nodes whose id is greater
/// than id of target node of the last edge in ordered edge list.
///
/// @param graph -- graph
/// @param hub_next -- hub scores which will be updated
/// @param hub_next -- auth scores which will be updated
    void CompleteRankNext(const HitsGraph &graph, std::vector<double> &hub_next, std::vector<double> &auth_next) {
        while (hub_next.size() < graph.GetNodeCount() && auth_next.size() < graph.GetNodeCount()) {
            hub_next.push_back(0);
            auth_next.push_back(0);
        }
    }
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
    bool CheckContinueIterate(const std::vector<double> &hub, const std::vector<double> &hub_next,
                              const std::vector<double> &auth, const std::vector<double> &auth_next,
                              const std::size_t max_iterations, const double stop_epsilon,
                              const std::size_t number_of_iterations) {
        if (number_of_iterations == max_iterations) {
            return false;
        }
        for (std::size_t node_id = 0; node_id < hub.size(); node_id++) {
            if (std::abs(hub[node_id] - hub_next[node_id]) > stop_epsilon &&
                std::abs(auth[node_id] - auth_next[node_id]) > stop_epsilon) {
                return true;
            }
        }
        return false;
    }
/// Normalizing PageRank
/// Divides all values with sum of the values to get their sum equal 1
///
/// @param score -- hub scores or auth scores
    void Normalise(std::vector<double> &scores) {
        const double sum = std::accumulate(scores.begin(), scores.end(), 0.0);
        if (sum!=0){for (double &value: scores) { value /= sum; }}
    }
} // namespace


HitsGraph::HitsGraph(std::uint64_t number_of_nodes, std::uint64_t number_of_edges,
                     const std::vector<EdgePair> &edges) :
                node_count_(number_of_nodes), edge_count_(number_of_edges), out_degree_(number_of_nodes){
            AdjacencyList<std::uint64_t> in_neighbours(number_of_nodes);
            for (const auto [from, to]: edges) {
                // Because PageRank needs a set of nodes that point to a given node.
                in_neighbours.AddAdjacentPair(from, to);
                out_degree_[from] += 1;
            }

            for (std::size_t node_id = 0; node_id < number_of_nodes; node_id++) {
                const auto &adjacent_nodes = in_neighbours.GetAdjacentNodes(node_id);
                for (const auto adjacent_node: adjacent_nodes) {
                    ordered_edges_.emplace_back(adjacent_node, node_id);
                };
            }
        };

std::uint64_t HitsGraph::GetNodeCount() const { return node_count_; };

std::uint64_t HitsGraph::GetEdgeCount() const { return edge_count_; };

const std::vector<EdgePair> &HitsGraph::GetOrderedEdges() const { return ordered_edges_; };


std::tuple<std::vector<double>, std::vector<double>>ParallelIterativeHits(const HitsGraph &graph, std::size_t max_iterations,
                                                                          double stop_epsilon){
        const std::uint64_t number_of_threads =  std::thread::hardware_concurrency()/2;
        auto borders = CalculateOptimalBorders(graph, number_of_threads);
        std::vector<double> hub(graph.GetNodeCount(), 1);
        std::vector<double> auth(graph.GetNodeCount(), 1);
        bool continue_iterate = max_iterations != 0;
        std::size_t number_of_iterations = 0;
        while (continue_iterate) {
            std::vector<std::promise<std::pair<std::vector<double>, std::vector<double>>>> hits_promise(number_of_threads);
            std::vector<std::future<std::pair<std::vector<double>, std::vector<double>>>> hits_future;
            hits_future.reserve(number_of_threads);

            std::transform(hits_promise.begin(), hits_promise.end(), std::back_inserter(hits_future),
                           [](auto &pr_promise) { return pr_promise.get_future(); });
            std::vector<std::thread> my_thread;
            my_thread.reserve(number_of_threads);
            for (std::size_t cluster_id = 0; cluster_id < number_of_threads; cluster_id++) {
                my_thread.emplace_back(
                        [&, lo = borders[cluster_id], hi = borders[cluster_id + 1]](
                                auto promise) { ThreadHitsIteration(graph, hub, auth, lo, hi, std::move(promise)); },
                        std::move(hits_promise[cluster_id])
                );
            }
            std::vector<double> hub_next(graph.GetNodeCount(), 0);
            std::vector<double> auth_next(graph.GetNodeCount(), 0);
            for (std::size_t cluster_id = 0; cluster_id < number_of_threads; cluster_id++) {
                std::pair<std::vector<double>, std::vector<double>> block = hits_future[cluster_id].get();
                AddCurrentBlockToNext(graph, block, hub_next, auth_next);

            }
            CompleteRankNext(graph, hub_next, auth_next);
            for (std::uint64_t i = 0; i < number_of_threads; i++) {
                if (my_thread[i].joinable()) {
                    my_thread[i].join();
                }
            }
            hub.swap(hub_next);
            auth.swap(auth_next);
            number_of_iterations++;
            continue_iterate = CheckContinueIterate(hub, hub_next, auth, auth_next, max_iterations, stop_epsilon,
                                                    number_of_iterations);
            Normalise(hub);
            Normalise(auth);
        }
        return {hub, auth};
    }
};
