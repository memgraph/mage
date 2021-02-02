/// @file
///
/// The file contains function declarations of general graph algorithms. In our
/// case, these are the algorithms that don't rely on any domain knowledge
/// for solving the observability problem.

#pragma once

#include "data_structures/graph.hpp"

namespace algorithms {

/// Returns true if graph represented by edges is bipartite.
/// Otherwise false.
bool IsGraphBipartite(const std::vector<std::pair<uint32_t, uint32_t>> &edges);

/// Returns true if graph is bipartite.
/// Otherwise returns false.
bool IsGraphBipartite(const graphdata::GraphView &G);

/// Returns the size of the maximum matching in a given bipartite graph. The
/// graph is defined by the sizes of two disjoint sets of nodes and by a set of
/// edges between those two sets of nodes. The nodes in both sets should be
/// indexed from 1 to `set_size`.
///
/// The algorithm runs in O(|V|*|E|) time where V represents a set of nodes and
/// E represents a set of edges.
size_t BipartiteMatching(const graphdata::GraphView &G);

/// Returns the size of the maximum matching in a given bipartite graph. The
/// graph is defined by the sizes of two disjoint sets of nodes and by a set of
/// edges between those two sets of nodes. The nodes in both sets should be
/// indexed from 1 to `set_size`.
///
/// The algorithm runs in O(|V|*|E|) time where V represents a set of nodes and
/// E represents a set of edges.
size_t BipartiteMatching(
    const std::vector<std::pair<uint32_t, uint32_t>> &edges);

/// Returns a list of edges of an undirected graph whose removal would
/// add another connected component to G.
///
/// The algorithm runs in O(|E| + |V|) time where E represents a set
/// of edges and V represents a set of vertices.
std::vector<graphdata::Edge> GetBridges(const graphdata::GraphView &G);

/// Returns a list of Biconnected Components (BCCs) of an undirected graph G
/// Each BCC is represented by a list of edges that are inside that component.
///
/// The implemented algorithm is authored by Hopcroft and Tarjan and runs in
/// O(|E| + |V|) time, where E represents a set of edges and V represents a
/// set of vertices.
std::vector<std::vector<graphdata::Edge>> GetBiconnectedComponents(
    const graphdata::GraphView &G);

/// Returns a list of all simple cycles of an undirected graph G. Each simple
/// cycle is represented by a list of nodes in cycle order. For example, the
/// cycle below would be represented by a list {1, 2, 3, 4}.
///
/// (1)--(2)
///  |    |
/// (4)--(3)
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
/// TODO(ipaljak): Optimize the algorithm by decomposition into biconnected
///                components.
std::vector<std::vector<graphdata::Node>> GetCycles(
    const graphdata::GraphView &G);

/// Returns all cutsets of a given undirected graph G. Here, we will define
/// a cutset of G as a set of edges whose removal would add exactly one more
/// connected component to G and that condition doesn't hold for any of its
/// subsets.
///
/// The algorithm runs in O(k|V|^2(|E| + |V|)(|V| - log(k))) where V and E
/// represent sets of nodes and edges of G. The total number of cutsets is
/// denoted by k and has an upper bound of sum{2^(|Vi|-1)}.
///
/// The pseudocode of this algorithm is outlined in figure 1 of 1980s paper
/// "An Algorithm to Enumerate All Cutsets of a Graph in Linear Time
/// per Cutset" by Tsukiyama, Shirakawa, Ozaki and Ariyoshi.
std::vector<std::vector<graphdata::Edge>> GetCutsets(
    const graphdata::GraphView &G);

/// Gets the neighbour cycles.
///
/// Neighbour cycles are special type of cycles that are not included in
/// GetCycles function.
/// Neighbour cycle is a cycle between two adjacent nodes. If two nodes share
/// more than one edge then they from a neighbour cycle.
std::vector<std::pair<graphdata::Node, graphdata::Node>> GetNeighbourCycles(
    const graphdata::GraphView &G);

}  // namespace algorithms

namespace algorithms_bf {

/// Returns all cutsets of a given undirected graph G. Here, we will define
/// a cutset of G as a set of edges whose removal would add exactly one more
/// connected component to G and that condition doesn't hold for any of its
/// subsets.
///
/// The algorithm runs in O(sum{|Ei|*2^|Vi|}) where Vi and Ei represent the
/// set of nodes and edges in i-th connected component of G. The total number
/// of cutsets has an upper bound of sum{2^(|Vi|-1)}.
std::vector<std::vector<graphdata::Edge>> GetCutsets(
    const graphdata::GraphView &G);

}  // namespace algorithms_bf
