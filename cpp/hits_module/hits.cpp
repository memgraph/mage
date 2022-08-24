#include <iostream>
#include <future>
#include <algorithm>
#include <numeric>
#include <queue>

# pragma once


namespace hits_alg {

    template<typename T>
    class AdjacencyList {
    public:
        AdjacencyList() = default;

        explicit AdjacencyList(std::uint64_t node_count) : list_(node_count) {}

        auto GetNodeCount() const { return list_.size(); }

        void AddAdjacentPair(T left_node_id, T right_node_id, bool undirected = false) {
            list_[left_node_id].push_back(right_node_id);
            if (undirected) {
                list_[right_node_id].push_back(left_node_id);
            }
        }

        const auto &GetAdjacentNodes(T node_id) const { return list_[node_id]; }

    private:
        std::vector <std::vector<T>> list_;
    };


    using EdgePair = std::pair<std::uint64_t, std::uint64_t>;
    using momo = std::pair <std::vector<double>, std::vector<double>>;

    class HitsGraph {
    public:
        HitsGraph(std::uint64_t number_of_nodes, std::uint64_t number_of_edges, const std::vector <EdgePair> &edges) :
                node_count_(number_of_nodes), edg_count_(number_of_edges), out_degree_(number_of_nodes),
                in_degree_(number_of_nodes) {
            AdjacencyList<std::uint64_t> in_neighbours(number_of_nodes);
//            {{0, 3}, {0, 2}, {0, 1},
            for (const auto [from, to]: edges) {
                in_neighbours.AddAdjacentPair(from, to);
                out_degree_[from] += 1;
                in_degree_[to] += 1;
            }

            for (std::size_t node_id = 0; node_id < number_of_nodes; node_id++) {
                const auto &adjacent_nodes = in_neighbours.GetAdjacentNodes(node_id);
                for (const auto adjacent_node: adjacent_nodes) {
                    ordered_edges_.emplace_back(adjacent_node, node_id);
                };
            }
        };

        std::uint64_t GetNodeCount() const { return node_count_; };

        std::uint64_t GetEdgeCount() const { return edg_count_; };

        const std::vector <EdgePair> &GetOrderedEdges() const { return ordered_edges_; };

        std::uint64_t GetOutDegree(std::uint64_t node_id) const { return out_degree_[node_id]; };

        std::uint64_t GetInDegree(std::uint64_t node_id) const { return in_degree_[node_id]; };
    private:
        std::uint64_t node_count_;
        std::uint64_t edg_count_;
        std::vector <EdgePair> ordered_edges_;
        std::vector <std::uint64_t> out_degree_;
        std::vector <std::uint64_t> in_degree_;
    };

    std::vector <std::uint64_t> CalculateOptimalBorders(const HitsGraph &graph, const std::uint64_t number_of_threads) {
        std::vector <std::uint64_t> borders;

        if (number_of_threads == 0) {
            throw std::runtime_error("Number of threads can't be zero (0)!");
        }

        for (std::uint64_t border_index = 0; border_index <= number_of_threads; border_index++) {
            borders.push_back(border_index * graph.GetEdgeCount() / number_of_threads);
        }
        return borders;
    }


    void
    ThreadHitsIteration(const HitsGraph &graph, const std::vector<double> old_hub, const std::vector<double> old_auth,
                        const std::uint64_t lo,
                        const std::uint64_t hi,
                        std::promise <std::pair<std::vector < double>, std::vector<double>>

    > new_hits_promise) {

    std::vector<double> new_hub(graph.GetNodeCount(), 0);
    std::vector<double> new_auth(graph.GetNodeCount(), 0);
    for (
    std::size_t edge_id = lo;
    edge_id<hi;
    edge_id++) {
    const auto [source, target] = graph.GetOrderedEdges()[edge_id];
    new_hub[target] += old_auth[source];
    new_auth[source] += old_hub[target];
}
std::pair <std::vector<double>, std::vector<double>> hub_auth = {new_hub, new_auth};
new_hits_promise.
set_value(hub_auth);
};

void AddCurrentBlockToNext(const HitsGraph &graph, const std::pair <std::vector<double>, std::vector<double>> block,
                           std::vector<double> &hub_next, std::vector<double> &auth_next) {
    for (std::size_t node_index = 0; node_index < block.first.size(); node_index++) {
        hub_next[node_index] += block.first[node_index];
        auth_next[node_index] += block.second[node_index];
    }

};

void CompleteRankNext(const HitsGraph &graph, std::vector<double> &hub_next, std::vector<double> &auth_next) {
    while (hub_next.size() < graph.GetNodeCount() && auth_next.size() < graph.GetNodeCount()) {
        hub_next.push_back(0);
        auth_next.push_back(0);
    }
}

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

void Normalise(std::vector<double> &rank) {
    const double sum = std::accumulate(rank.begin(), rank.end(), 0.0);
    for (double &value: rank) { value /= sum; }
}

std::tuple <std::vector<double>, std::vector<double>> ParallelIterativeHits(const HitsGraph &graph, std::size_t max_iterations, double stop_epsilon) {
    const std::uint64_t number_of_threads = 3;
    auto borders = CalculateOptimalBorders(graph, number_of_threads);
    std::vector<double> hub(graph.GetNodeCount(), 1);
    std::vector<double> auth(graph.GetNodeCount(), 1);
    bool continue_iterate = max_iterations != 0;
    std::size_t number_of_iterations = 0;
    while (continue_iterate) {
        std::vector < std::promise < std::pair < std::vector < double > , std::vector <
                                                                          double >> >> hits_promise(number_of_threads);
        std::vector < std::future < std::pair < std::vector < double > , std::vector < double >> >> hits_future;
        hits_future.reserve(number_of_threads);

        std::transform(hits_promise.begin(), hits_promise.end(), std::back_inserter(hits_future),
                       [](auto &pr_promise) { return pr_promise.get_future(); });
        std::vector <std::thread> my_thread;
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
            std::pair <std::vector<double>, std::vector<double>> block = hits_future[cluster_id].get();
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


int main() {
    std::cout << "Hits Rank is !" << std::endl;
    hits_alg::HitsGraph reza(4, 3, {{1, 0},
                                    {2, 0},
                                    {3, 0}});
    hits_alg::HitsGraph reza1(6, 10, {{0, 1},
                                      {1, 0},
                                      {1, 2},
                                      {0, 2},
                                      {2, 3},
                                      {3, 2},
                                      {4, 3},
                                      {5, 3},
                                      {4, 5},
                                      {5, 4}});
    auto [hub, auth] = hits_alg::ParallelIterativeHits(reza1, 10, 10 ^ (-6));
    std::cout << "hub :";
    for (auto i: hub) { std::cout << i << "  "; };
    std::cout << std::endl;
    std::cout << "auth :";
    for (auto i: auth) { std::cout << i << "  "; };


}