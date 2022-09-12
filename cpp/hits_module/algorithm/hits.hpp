#pragma once

#include <future>

namespace hits_alg {

// Defines edge's from and to node index. Just an alias for user convenience.
    using EdgePair = std::pair<std::uint64_t, std::uint64_t>;

/// A directed, unweighted graph.
/// Self loops and multiple edges are allowed and they will affect the result.
/// Node ids are integers from interval [0, number of nodes in graph - 1].
/// Graph is allowed to be disconnected.
    class HitsGraph {
    public:
    /// Creates graph with given number of nodes with node ids from interval
    /// [0, number_of_nodes - 1] and with given edges between them.
    /// Node ids describing edges have to be integers from
    /// interval [0, number of nodes in graph - 1].
    /// @param number_of_nodes -- number of nodes in graph
    /// @param number_of_edges -- number of edges in graph
    /// @param edges -- pairs (source, target) representing directed edges
    HitsGraph(std::uint64_t number_of_nodes, std::uint64_t number_of_edges, const std::vector<EdgePair> &edges);

    /// @return -- number of nodes in graph
    std::uint64_t GetNodeCount() const;

    /// @return -- nubmer of edges in graph
    std::uint64_t GetEdgeCount() const;

    /// @return -- a reference to ordered ordered vector of edges
    const std::vector<EdgePair> &GetOrderedEdges() const;

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
/// Hits algorithm outputs is two numbers for a node. Authorities estimates the node
/// value based on the incoming links. Hubs estimates the node value based
/// on outgoing links

/// The set of highly relevant nodes are called Roots. They are potential Authorities.
/// Nodes that are not very relevant but point to pages in the Root are called Hubs.
/// So, an Authority is a page that many hubs link to whereas a Hub is a page that links to many authorities.

///HITS Algorithm is a Link Analysis Algorithm that rates webpages.
/// This algorithm is used to the web link-structures to discover and rank the webpages relevant
/// for a particular search. Its uses hubs and authorities ranks to define a recursive relationship
/// between webpages.

// Let number of iterations be k.
// Each node is assigned a Hub score = 1 and an Authority score = 1.
// Hits is computed iteratively using following formula:

/// Hub update : Each node’s Hub score = \Sigma  (Authority score of each node it points to).
/// Authority update : Each node’s Authority score = \Sigma  (Hub score of each node pointing to it).
/// At the end hub and authority Ranks values are normalized to sum 1 to form probability distribution.


/// @param graph -- a directed, unweighted, not necessarily connected graph
/// which can contain multiple edges and self-loops.
/// @param max_iterations -- maximum number of iterations performed by Hits.
/// @param stop_epsilon -- stop epsilon
/// @return --  two numbers for a node hub an auth
    std::tuple<std::vector<double>, std::vector<double>> ParallelIterativeHits(const HitsGraph &graph, size_t max_iterations = 100,
                                                   double stop_epsilon = 10e-6);

}  // namespace hits_alg
