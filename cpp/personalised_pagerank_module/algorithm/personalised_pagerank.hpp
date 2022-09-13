#pragma once

#include <future>

namespace personalised_pagerank_alg {

// Defines edge's from and to node index. Just an alias for user convenience.
    using EdgePair = std::pair<std::uint64_t, std::uint64_t>;

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
        PageRankGraph(std::uint64_t number_of_nodes, std::uint64_t number_of_edges, const std::vector<EdgePair> &edges);

        /// @return -- number of nodes in graph
        std::uint64_t GetNodeCount() const;

        /// @return -- nubmer of edges in graph
        std::uint64_t GetEdgeCount() const;

        /// @return -- a reference to ordered ordered vector of edges
        const std::vector<EdgePair> &GetOrderedEdges() const;

        /// Returns out degree of node node_id
        /// @param node_id -- node name
        /// @return -- out degree of node node_id
        std::uint64_t GetOutDegree(std::uint64_t node_id) const;

        /// @return -- vector of dangle node_id
        std::vector<std::uint64_t> GetDangleNodes() const;


    private:
        /// node_count equals number of nodes in graph
        std::uint64_t node_count_;
        /// edge_count equals number of edges in graph
        std::uint64_t edge_count_;
        /// directed edges (source, target) (source -> target) ordered by target
        std::vector<EdgePair> ordered_edges_;
        /// out degree for each node in graph because it is required in calculating
        /// PageRank
        std::vector<std::uint64_t> out_degree_;
    };

/// If we present nodes as pages and directed edges between them as links the
/// PageRank algorithm outputs a probability distribution used to represent the
/// likelihood that a person randomly clicking on links will arrive at any
/// particular page.
///
/// PageRank theory holds that an imaginary surfer who is randomly clicking on
/// links will eventually stop clicking. The probability, at any step, that the
/// person will continue randomly clicking on links is called a damping factor,
/// otherwise next page is chosen randomly among all pages.

////The calculation start with initial rank of 1/N fo each node,
/// However, if a personalization vector is exist,  a weight to each assigns node that
/// influences the random walk restart.It biases the walk towards specific nodes.

/// PageRank is computed iteratively using following formula:
/// Rank(n, t + 1) = (1 - d) / number_of_nodes
///                + d * sum { Rank(in_neighbour_of_n, t) /
///                out_degree(in_neighbour_of_n)}
/// Where Rank(n, t) is PageRank of node n at iteration t
/// At the end Rank values are normalized to sum 1 to form probability
/// distribution.
///

/// @param graph -- a directed, unweighted, not necessarily connected graph
/// which can contain multiple edges and self-loops.
/// @param personalisation -- The "personalization vector" consisting of a vector with a
//     subset of  pairs of graph nodes and personalization value each of those for ex: {{3,1},{29,1}}.
/// @param max_iterations -- maximum number of iterations performed by PageRank.
/// @param damping_factor -- a real number from interval [0, 1], as described above
/// @param stop_epsilon -- stop epsilon
/// @return -- probability distribution, as described above
    std::vector<double> ParallelIterativePPageRank(const PageRankGraph &graph, const std::vector<std::pair<int, double>>& personalisation = {}, size_t max_iterations = 100,
                                                  double damping_factor = 0.85, double stop_epsilon = 10e-6);

}  // namespace personalised_pagerank_alg
