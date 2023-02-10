#pragma once

#include <mgp.hpp>
#include <random>
#include <unordered_set>

namespace pagerank_online_alg {
///
///@brief Computes PageRank based on the method by Bahmani et al.
/// [http://snap.stanford.edu/class/cs224w-readings/bahmani10pagerank.pdf] and saves the algorithm context. Creates R
/// random walks from each graph node and approximates PageRank scores (determined by the N of walks a node appears in)
///
///@param graph Current graph
///@param R Number of random walks per node
///@param epsilon Walk stopping probability (i.e. average walk length is 1/ɛ)
///@return (node id, PageRank) for each node
///
std::vector<std::pair<std::uint64_t, double>> SetPageRank(const mg_graph::GraphView<> &graph, std::uint64_t R = 10,
                                                          double epsilon = 0.2);

///
///@brief Retrieves previously computed PageRank scores from the context. If the graph has since been modified, throws
/// an exception.
///
///@param graph Current graph
///@return (node id, PageRank) for each node
///
std::vector<std::pair<std::uint64_t, double>> GetPageRank(const mg_graph::GraphView<> &graph);

///
///@brief Dynamically updates PageRank scores. The method works on the updated graph and respectively accounts for
/// deleted relationships, deleted nodes, created nodes, and created relationships.
/// and relationships among them
///
///@param graph Current graph
///@param new_nodes Nodes created since the last computation
///@param new_relationships Relationships created since the last computation
///@param deleted_nodes Nodes deleted since the last computation
///@param deleted_relationships Relationships deleted since the last computation
///@return (node id, PageRank) for each node
///
std::vector<std::pair<std::uint64_t, double>> UpdatePageRank(
    const mgp::Graph &graph, const std::vector<std::uint64_t> &new_nodes,
    const std::vector<std::pair<std::uint64_t, uint64_t>> &new_relationships,
    const std::vector<std::uint64_t> &deleted_nodes,
    const std::vector<std::pair<std::uint64_t, uint64_t>> &deleted_relationships);

///
///@brief Returns whether the algorithm context is empty. If the user calls .get() or .update() without there being
/// previous results, the algorithm defaults default to SetPageRank().
///
///@return whether the context is empty
///
bool ContextEmpty();

///
///@brief Resets the context.
///
void Reset();

}  // namespace pagerank_online_alg
