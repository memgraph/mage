#include "leiden_utils.hpp"
#include <algorithm>
#include <omp.h>
#include <set>
#include <iterator>

namespace leiden_alg {

bool edgeBetweenCommunities(const std::vector<std::uint64_t> &community1, const std::vector<std::uint64_t> &community2, const Graph &graph) {
    for (const auto &node1 : community1) {
        for (const auto &node2 : community2) {
            if (std::find(graph.neighbors(node1).begin(), graph.neighbors(node1).end(), node2) != graph.neighbors(node1).end()) {
                return true;
            }
        }
    }
    return false;
}

bool isSubset(std::vector<std::uint64_t> &set1, std::vector<std::uint64_t> &set2) {
    std::sort(set1.begin(), set1.end());
    std::sort(set2.begin(), set2.end());
    return std::includes(set2.begin(), set2.end(), set1.begin(), set1.end());
}

// |E(node, C \ node)|
std::uint64_t countEdgesBetweenNodeAndCommunity(const Graph &graph, std::uint64_t node_id, std::uint64_t community_id, Partitions &partitions) {
    if (partitions.node_and_community_cache.find({node_id, community_id}) != partitions.node_and_community_cache.end()) {
        return partitions.node_and_community_cache[{node_id, community_id}];
    }
    int count = 0;
    #pragma omp parallel for reduction(+:count) 
    for (const auto &neighbor : graph.neighbors(node_id)) {
         if (partitions.getCommunityForNode(neighbor) == community_id) {
            count++;
        }
    }
    partitions.node_and_community_cache[{node_id, community_id}] = count;
    return count;
}

// |E(C, S \ C)|
std::uint64_t countEdgesBetweenCommunities(std::uint64_t community_id, std::uint64_t subset, Partitions &refined_partitions, Partitions &partitions, const Graph &graph) {
    std::vector<std::uint64_t> set_intersection;
    const auto &refined_community = refined_partitions.communities[community_id];
    const auto &original_community = partitions.communities[subset];
    std::set_difference(original_community.begin(), original_community.end(), refined_community.begin(), refined_community.end(), std::inserter(set_intersection, set_intersection.begin()));
    std::uint64_t count = 0;
    for (const auto &node : set_intersection) {
        count += countEdgesBetweenNodeAndCommunity(graph, node, community_id, refined_partitions);
    }
    return count;
}

std::uint64_t getNumOfPossibleEdges(std::uint64_t n) {
    return n * (n - 1) / 2;
}

// CPM - Constant Potts Model
// it computes the change in modularity when a node is moved from its current community to a new community
std::pair<double, std::uint64_t> computeDeltaCPM(Partitions &partitions, const std::uint64_t node_id, const std::uint64_t new_community_id, const Graph &graph, const double gamma) {
    double result = 0.0;
    
    const auto current_community_id = partitions.getCommunityForNode(node_id);
    const auto current_community_size = partitions.communities[current_community_id].size();
    const auto new_community_size = partitions.communities[new_community_id].size();
    const auto current_community_weight = partitions.community_weights[current_community_id];
    const auto new_community_weight = partitions.community_weights[new_community_id];

    const auto source_weight = current_community_weight - gamma * getNumOfPossibleEdges(current_community_size);
    const auto target_weight = new_community_weight - gamma * getNumOfPossibleEdges(new_community_size);
    
    const auto num_edges_between_node_and_current_community = countEdgesBetweenNodeAndCommunity(graph, node_id, current_community_id, partitions);
    const auto num_edges_between_node_and_new_community = countEdgesBetweenNodeAndCommunity(graph, node_id, new_community_id, partitions);

    const auto new_source_weight = current_community_weight - num_edges_between_node_and_current_community - gamma * getNumOfPossibleEdges(current_community_size + 1);
    const auto new_target_weight = new_community_weight + num_edges_between_node_and_new_community - gamma * getNumOfPossibleEdges(new_community_size - 1);

    result = new_source_weight + new_target_weight - source_weight - target_weight;
    return {result, num_edges_between_node_and_new_community};
}

}  // namespace leiden_alg
