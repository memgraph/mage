#include "leiden_utils.hpp"
#include <algorithm>
#include <cstdint>
namespace leiden_alg {

std::vector<std::uint64_t> calculateEdgeWeightsPerCommunity(const Partitions &partitions, const Graph &graph) {
    std::vector<std::uint64_t> edge_weights_per_community(partitions.communities.size(), 0);
    for (const auto & community : partitions.communities) {
        for (const auto &node : community) {
            edge_weights_per_community[partitions.getCommunityForNode(node)] += graph.neighbors(node).size();
        }
    }
    return edge_weights_per_community;
}

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


std::uint64_t getNumOfPossibleEdges(std::uint64_t n) {
    return n * (n - 1) / 2;
}

}  // namespace leiden_alg
